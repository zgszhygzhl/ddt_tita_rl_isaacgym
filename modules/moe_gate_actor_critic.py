import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from modules.actor_critic import ActorCriticBarlowTwins, ActorCriticRMA
from modules.moe_terrain_estimator import MoeTerrainEstimator
from modules.residual_expert_actor_critic import ResidualExpertActorCritic

POLICY_CLASSES = {
    "ActorCriticBarlowTwins": ActorCriticBarlowTwins,
    "ActorCriticRMA": ActorCriticRMA,
}


def _extract_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a state_dict")
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if "actor_critic_state_dict" in checkpoint:
        return checkpoint["actor_critic_state_dict"]
    return checkpoint


def load_state_with_prefix_fallback(module, ckpt_path, tag):
    if not ckpt_path:
        raise ValueError(f"A checkpoint path is required for the frozen {tag} policy")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    prefixes = ("", "module.", "actor_critic.", "residual_actor_critic.", "base_actor_critic.")
    candidates = []
    for prefix in prefixes:
        candidate = {
            (key[len(prefix):] if prefix else key): value
            for key, value in state_dict.items()
            if not prefix or key.startswith(prefix)
        }
        if candidate:
            candidates.append(candidate)

    target = module.state_dict()
    best = max(candidates, key=lambda item: sum(
        key in target and target[key].shape == value.shape for key, value in item.items()
    ))
    compatible = {
        key: value for key, value in best.items()
        if key in target and target[key].shape == value.shape
    }
    matched_keys = len(compatible)
    if matched_keys == 0:
        raise RuntimeError(f"[moe_gate] loaded {tag} matched_keys=0 path={ckpt_path}")
    incompatible = module.load_state_dict(compatible, strict=False)
    print(
        f"[moe_gate] loaded {tag} path={ckpt_path} matched_keys={matched_keys} "
        f"missing={len(incompatible.missing_keys)} unexpected={len(best) - matched_keys}"
    )
    return checkpoint


def load_full_expert_checkpoint(module, ckpt_path, tag):
    """Load both base and residual halves of a residual expert wrapper."""
    if not ckpt_path:
        raise ValueError(f"A checkpoint path is required for the frozen {tag} expert")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict(checkpoint)
    prefixes = ("", "module.", "actor_critic.", "module.actor_critic.")
    candidates = []
    for prefix in prefixes:
        candidate = {
            (key[len(prefix):] if prefix else key): value
            for key, value in state_dict.items()
            if not prefix or key.startswith(prefix)
        }
        if candidate:
            candidates.append(candidate)

    target = module.state_dict()
    best = max(
        candidates,
        key=lambda item: sum(
            key in target and target[key].shape == value.shape
            for key, value in item.items()
        ),
    )
    compatible = {
        key: value
        for key, value in best.items()
        if key in target and target[key].shape == value.shape
    }
    base_keys = sum(key.startswith("base_actor_critic.") for key in compatible)
    residual_keys = sum(key.startswith("residual_actor_critic.") for key in compatible)
    if base_keys == 0 or residual_keys == 0:
        raise RuntimeError(
            f"[moe_gate] {tag} checkpoint is not a complete residual wrapper: "
            f"base_keys={base_keys} residual_keys={residual_keys} path={ckpt_path}"
        )

    incompatible = module.load_state_dict(compatible, strict=False)
    print(
        f"[moe_gate] loaded {tag} full_expert path={ckpt_path} "
        f"matched_keys={len(compatible)} base_keys={base_keys} "
        f"residual_keys={residual_keys} missing={len(incompatible.missing_keys)} "
        f"unexpected={len(best) - len(compatible)}"
    )
    return checkpoint

def topk_softmax(logits, top_k=2, temperature=1.0):
    logits = logits / max(float(temperature), 1.0e-6)
    if 0 < top_k < logits.shape[-1]:
        values, indices = torch.topk(logits, k=top_k, dim=-1)
        masked = torch.full_like(logits, -1.0e9)
        masked.scatter_(dim=-1, index=indices, src=values)
        logits = masked
    return torch.softmax(logits, dim=-1)


def _mlp(input_dim, hidden_dims, output_dim, activation, final_activation=None):
    activation_class = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}.get(
        str(activation).lower()
    )
    if activation_class is None:
        raise ValueError(f"Unsupported activation: {activation}")
    layers = []
    last_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(last_dim, hidden_dim), activation_class()))
        last_dim = hidden_dim
    layers.append(nn.Linear(last_dim, output_dim))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


class ActorCriticMoEGate(nn.Module):
    is_recurrent = False

    def __init__(
        self, num_prop, num_scan, num_critic_obs, num_priv_latent, num_hist, num_actions,
        init_noise_std=0.35, activation="elu", num_costs=6,
        gate_hidden_dims=(128, 64), critic_hidden_dims=(256, 128, 64),
        gate_top_k=2, gate_temperature=1.0, gate_init_weight=0.05,
        gate_aux_target_mode="estimator_full", residual_alpha=1.0,
        residual_delta_clip=0.0,
        stair_residual_alpha=0.60, slip_residual_alpha=0.45,
        recovery_residual_alpha=0.60,
        stair_residual_delta_clip=0.65, slip_residual_delta_clip=0.65,
        recovery_residual_delta_clip=0.85,
        base_ckpt="", stair_ckpt="", slip_ckpt="",
        recovery_ckpt="", estimator_ckpt="",
        base_policy_class_name="ActorCriticBarlowTwins",
        stair_policy_class_name="ActorCriticBarlowTwins",
        slip_policy_class_name="ActorCriticBarlowTwins",
        recovery_policy_class_name="ActorCriticBarlowTwins",
        base_policy_cfg=None, stair_policy_cfg=None, slip_policy_cfg=None,
        recovery_policy_cfg=None, **kwargs,
    ):
        super().__init__()
        self.num_prop = num_prop
        self.num_scan = num_scan
        self.num_critic_obs = num_critic_obs
        self.num_priv_latent = num_priv_latent
        self.num_hist = num_hist
        self.num_actions = num_actions
        self.imi_flag = False
        self.distribution = None
        self.current_delta = None
        self.last_gate_weights = None
        self.last_gate_logits = None
        self.last_v_hat = None
        self.last_gray_hat = None
        self.last_delta_norm = None
        self.last_saturation_ratio = None
        self.gate_top_k = int(gate_top_k)
        self.gate_temperature = float(gate_temperature)
        self.residual_alpha = float(residual_alpha)
        self.residual_delta_clip = float(residual_delta_clip)
        self.gate_aux_target_mode = str(gate_aux_target_mode).lower()
        valid_target_modes = ("estimator_full", "stair_full")
        if self.gate_aux_target_mode not in valid_target_modes:
            raise ValueError(
                f"Unsupported gate_aux_target_mode: {gate_aux_target_mode!r}"
            )

        self.base_actor = self._build_frozen_actor(
            base_policy_class_name, base_policy_cfg
        )
        load_state_with_prefix_fallback(self.base_actor, base_ckpt, "base")

        expert_specs = (
            (
                "stair",
                stair_policy_class_name,
                stair_policy_cfg,
                stair_ckpt,
                stair_residual_alpha,
                stair_residual_delta_clip,
            ),
            (
                "slip",
                slip_policy_class_name,
                slip_policy_cfg,
                slip_ckpt,
                slip_residual_alpha,
                slip_residual_delta_clip,
            ),
            (
                "recovery",
                recovery_policy_class_name,
                recovery_policy_cfg,
                recovery_ckpt,
                recovery_residual_alpha,
                recovery_residual_delta_clip,
            ),
        )
        for tag, class_name, policy_cfg, ckpt_path, alpha, delta_clip in expert_specs:
            expert_base = self._build_actor(base_policy_class_name, base_policy_cfg)
            residual_actor = self._build_actor(class_name, policy_cfg)
            expert = ResidualExpertActorCritic(
                base_actor_critic=expert_base,
                residual_actor_critic=residual_actor,
                alpha=float(alpha),
                freeze_base=True,
                residual_delta_clip=float(delta_clip),
                alpha_warmup_steps=0,
                zero_init_residual=False,
            )
            load_full_expert_checkpoint(expert, ckpt_path, tag)
            self._freeze(expert)
            setattr(self, f"{tag}_actor", expert)
        if not estimator_ckpt:
            raise ValueError("A checkpoint path is required for the frozen terrain estimator")
        estimator_state = torch.load(estimator_ckpt, map_location="cpu")
        self.estimator = MoeTerrainEstimator(
            n_proprio=estimator_state["n_proprio"],
            history_len=estimator_state["history_len"],
            hidden_dims=(256, 128, 64),
        )
        self.estimator.load_state_dict(estimator_state["model_state_dict"])
        self._freeze(self.estimator)
        print(f"[moe_gate] loaded estimator epoch={estimator_state.get('epoch', 'unknown')} path={estimator_ckpt}")

        self.gate_actor = _mlp(num_prop + 8, gate_hidden_dims, 3, activation)
        gate_init_weight = float(gate_init_weight)
        if not 0.0 < gate_init_weight < 1.0:
            raise ValueError("gate_init_weight must be strictly between 0 and 1")
        gate_output = next(
            module for module in reversed(self.gate_actor) if isinstance(module, nn.Linear)
        )
        nn.init.zeros_(gate_output.weight)
        nn.init.constant_(
            gate_output.bias,
            math.log(gate_init_weight / (1.0 - gate_init_weight)),
        )
        self.critic = _mlp(num_critic_obs, critic_hidden_dims, 1, activation)
        self.cost = _mlp(num_critic_obs, critic_hidden_dims, num_costs, activation, nn.Softplus())
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        Normal.set_default_validate_args = False
        print("ActorCriticMoEGate initialized")

    def _build_actor(self, class_name, policy_cfg):
        if class_name not in POLICY_CLASSES:
            raise ValueError(f"Unsupported frozen policy class: {class_name}")
        return POLICY_CLASSES[class_name](
            self.num_prop, self.num_scan, self.num_critic_obs, self.num_priv_latent,
            self.num_hist, self.num_actions, **deepcopy(policy_cfg or {}),
        )

    def _build_frozen_actor(self, class_name, policy_cfg):
        actor = self._build_actor(class_name, policy_cfg)
        self._freeze(actor)
        return actor

    @staticmethod
    def _freeze(module):
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False

    @staticmethod
    def _policy_mean(actor, obs):
        if hasattr(actor, "act_inference"):
            return actor.act_inference(obs)
        if getattr(actor, "teacher_act", False) and hasattr(actor, "act_teacher"):
            return actor.act_teacher(obs)
        if hasattr(actor, "act_student"):
            return actor.act_student(obs)
        if hasattr(actor, "act_teacher"):
            return actor.act_teacher(obs)
        raise AttributeError(f"{type(actor).__name__} has no deterministic actor interface")

    def compute_gate_weights(self, obs):
        obs_prop = obs[:, :self.num_prop]
        obs_hist = obs[:, -self.num_hist * self.num_prop:].view(
            -1, self.num_hist, self.num_prop
        )
        with torch.no_grad():
            v_hat, gray_hat = self.estimator(obs_hist)

        self.last_v_hat = v_hat.detach()
        self.last_gray_hat = gray_hat.detach()
        self.last_gate_logits = self.gate_actor(
            torch.cat((obs_prop, v_hat, gray_hat), dim=-1)
        )
        self.last_gate_weights = torch.sigmoid(self.last_gate_logits)
        return self.last_gate_weights

    def act_mean(self, obs):
        with torch.no_grad():
            base_action = self._policy_mean(self.base_actor, obs)
            stair_action = self._policy_mean(self.stair_actor, obs)
            slip_action = self._policy_mean(self.slip_actor, obs)
            recovery_action = self._policy_mean(self.recovery_actor, obs)

        weights = self.compute_gate_weights(obs)
        w_stair = weights[:, 0:1]
        w_slip = weights[:, 1:2]
        w_recovery = weights[:, 2:3]
        residual_delta = (
            w_stair * (stair_action - base_action)
            + w_slip * (slip_action - base_action)
            + w_recovery * (recovery_action - base_action)
        )
        if self.residual_delta_clip > 0.0:
            residual_delta = torch.clamp(
                residual_delta, -self.residual_delta_clip, self.residual_delta_clip
            )

        self.current_delta = self.residual_alpha * residual_delta
        mean_action = torch.clamp(base_action + self.current_delta, -1.0, 1.0)
        with torch.no_grad():
            self.last_delta_norm = torch.norm(
                self.current_delta.detach(), dim=-1
            ).mean()
            self.last_saturation_ratio = (
                mean_action.detach().abs() > 0.95
            ).float().mean()
        return mean_action

    def gate_auxiliary_loss(self):
        if self.last_gate_logits is None or self.last_gray_hat is None:
            return torch.zeros((), device=next(self.parameters()).device)

        gray_hat = self.last_gray_hat.detach()
        if self.gate_aux_target_mode == "stair_full":
            target_stair = torch.ones_like(gray_hat[:, 0])
            target_slip = torch.zeros_like(gray_hat[:, 0])
            target_recovery = torch.zeros_like(gray_hat[:, 0])
        else:
            target_stair = torch.clamp(2.5 * gray_hat[:, 0], 0.0, 1.0)
            target_slip = torch.clamp(gray_hat[:, 2], 0.0, 1.0)
            target_recovery = torch.clamp(gray_hat[:, 3], 0.0, 1.0)

        target = torch.stack(
            (target_stair, target_slip, target_recovery),
            dim=-1,
        )
        bce = F.binary_cross_entropy_with_logits(
            self.last_gate_logits,
            target,
            reduction="none",
        )
        branch_weights = torch.tensor(
            [4.0, 1.0, 1.0],
            device=bce.device,
            dtype=bce.dtype,
        )
        return (bce * branch_weights).mean()

    def update_distribution(self, obs):
        mean = self.act_mean(obs)
        self.distribution = Normal(mean, mean * 0.0 + self.std)

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        return self.act_mean(obs)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_std(self):
        return self.std

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def evaluate(self, critic_obs, **kwargs):
        return self.critic(critic_obs)

    def evaluate_cost(self, critic_obs, **kwargs):
        return self.cost(critic_obs)

    def reset(self, dones=None):
        pass

    def train(self, mode=True):
        super().train(mode)
        for module in (
            self.base_actor, self.stair_actor, self.slip_actor,
            self.recovery_actor, self.estimator,
        ):
            self._freeze(module)
        return self
