from copy import deepcopy

import torch
import torch.nn as nn
from torch.distributions import Normal

from modules.actor_critic import ActorCriticBarlowTwins, ActorCriticRMA
from modules.moe_terrain_estimator import MoeTerrainEstimator

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
        gate_top_k=2, gate_temperature=1.0, residual_alpha=0.60,
        residual_delta_clip=0.0, base_ckpt="", stair_ckpt="", slip_ckpt="",
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
        self.gate_top_k = int(gate_top_k)
        self.gate_temperature = float(gate_temperature)
        self.residual_alpha = float(residual_alpha)
        self.residual_delta_clip = float(residual_delta_clip)

        specs = (
            ("base", base_policy_class_name, base_policy_cfg, base_ckpt),
            ("stair", stair_policy_class_name, stair_policy_cfg, stair_ckpt),
            ("slip", slip_policy_class_name, slip_policy_cfg, slip_ckpt),
            ("recovery", recovery_policy_class_name, recovery_policy_cfg, recovery_ckpt),
        )
        for tag, class_name, policy_cfg, ckpt_path in specs:
            actor = self._build_frozen_actor(class_name, policy_cfg)
            load_state_with_prefix_fallback(actor, ckpt_path, tag)
            setattr(self, f"{tag}_actor", actor)

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

        self.gate_actor = _mlp(num_prop + 8, gate_hidden_dims, 4, activation)
        self.critic = _mlp(num_critic_obs, critic_hidden_dims, 1, activation)
        self.cost = _mlp(num_critic_obs, critic_hidden_dims, num_costs, activation, nn.Softplus())
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        Normal.set_default_validate_args = False
        print("ActorCriticMoEGate initialized")

    def _build_frozen_actor(self, class_name, policy_cfg):
        if class_name not in POLICY_CLASSES:
            raise ValueError(f"Unsupported frozen policy class: {class_name}")
        actor = POLICY_CLASSES[class_name](
            self.num_prop, self.num_scan, self.num_critic_obs, self.num_priv_latent,
            self.num_hist, self.num_actions, **deepcopy(policy_cfg or {}),
        )
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
        obs_hist = obs[:, -self.num_hist * self.num_prop:].view(-1, self.num_hist, self.num_prop)
        with torch.no_grad():
            v_hat, gray_hat = self.estimator(obs_hist)
        self.last_gate_logits = self.gate_actor(torch.cat((obs_prop, v_hat, gray_hat), dim=-1))
        self.last_gate_weights = topk_softmax(
            self.last_gate_logits, self.gate_top_k, self.gate_temperature
        )
        return self.last_gate_weights

    def act_mean(self, obs):
        with torch.no_grad():
            base_action = self._policy_mean(self.base_actor, obs)
            deltas = (
                self._policy_mean(self.stair_actor, obs),
                self._policy_mean(self.slip_actor, obs),
                self._policy_mean(self.recovery_actor, obs),
            )
        weights = self.compute_gate_weights(obs)
        residual_delta = sum(
            weights[:, index:index + 1] * delta
            for index, delta in enumerate(deltas, start=1)
        )
        if self.residual_delta_clip > 0.0:
            residual_delta = torch.clamp(
                residual_delta, -self.residual_delta_clip, self.residual_delta_clip
            )
        self.current_delta = self.residual_alpha * residual_delta
        return torch.clamp(base_action + self.current_delta, -1.0, 1.0)

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
