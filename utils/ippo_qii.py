import math
from typing import Any, Dict, Tuple

import torch


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def safe_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.reshape(-1).float()
    y = y.reshape(-1).float()
    valid = torch.isfinite(x) & torch.isfinite(y)
    if valid.sum() < 2:
        return x.new_tensor(0.0)
    x = x[valid] - x[valid].mean()
    y = y[valid] - y[valid].mean()
    denom = x.norm() * y.norm() + eps
    return (x * y).sum() / denom


class IPPOQIIComputer:
    def __init__(
        self,
        z_dim_limit: int = 64,
        ridge: float = 1e-3,
        score_mode: str = "sm_inverse",
        z_norm: bool = True,
        device: str = "cuda",
        eps: float = 1e-6,
    ):
        self.z_dim_limit = int(z_dim_limit)
        self.ridge = float(ridge)
        self.score_mode = score_mode
        self.z_norm = bool(z_norm)
        self.device = device
        self.eps = eps

    def reset(self):
        pass

    def _build_z(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        obs = obs.detach().float()
        actions = actions.detach().float()
        action_dim = actions.shape[-1]
        if self.z_dim_limit > action_dim:
            obs_dim = min(obs.shape[-1], self.z_dim_limit - action_dim)
            z = torch.cat((obs[..., :obs_dim], actions), dim=-1)
        else:
            z = torch.cat((obs, actions), dim=-1)[..., : self.z_dim_limit]
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        if self.z_norm:
            if z.dim() == 3:
                mean = z.mean(dim=1, keepdim=True)
                std = z.std(dim=1, keepdim=True, unbiased=False).clamp_min(self.eps)
            else:
                mean = z.mean(dim=0)
                std = z.std(dim=0, unbiased=False).clamp_min(self.eps)
            z = (z - mean) / std
        return torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

    def _score_from_leverage(self, leverage: torch.Tensor) -> torch.Tensor:
        leverage = torch.nan_to_num(leverage.clamp_min(0.0), nan=0.0, posinf=0.0, neginf=0.0)
        if self.score_mode == "leverage":
            raw = leverage
        elif self.score_mode == "log_leverage":
            raw = torch.log1p(leverage)
        else:
            raw = leverage / (1.0 + leverage)
        raw = torch.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
        raw_min = raw.min()
        raw_max = raw.max()
        return (raw - raw_min) / (raw_max - raw_min + self.eps)

    def compute_scores(self, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute rollout-local information scores from z_t = [obs_t_subset, action_t].

        The update is batched by rollout time step. This keeps the first version
        usable for thousands of parallel Isaac Gym environments while preserving
        the current-rollout, no-future-data constraint across time.
        """
        original_shape = obs.shape[:-1]
        if obs.dim() == 2:
            obs_seq = obs.unsqueeze(0)
            actions_seq = actions.unsqueeze(0)
        else:
            obs_seq = obs
            actions_seq = actions

        z = self._build_z(obs_seq, actions_seq)
        time_steps, _, z_dim = z.shape
        eye = torch.eye(z_dim, device=z.device, dtype=z.dtype)
        precision = self.ridge * eye
        scores = []
        leverages = []

        for step in range(time_steps):
            try:
                t_inv = torch.linalg.inv(precision + self.eps * eye)
            except RuntimeError:
                t_inv = torch.linalg.pinv(precision + self.eps * eye)
            z_step = z[step]
            leverage = (z_step @ t_inv * z_step).sum(dim=-1)
            leverages.append(leverage)
            precision = precision + z_step.transpose(0, 1) @ z_step

        leverage_tensor = torch.stack(leverages, dim=0)
        score_tensor = self._score_from_leverage(leverage_tensor)
        if len(original_shape) == 1:
            score_tensor = score_tensor.squeeze(0)

        info = {
            "z_dim": float(z_dim),
            "leverage_mean": float(leverage_tensor.mean().detach().cpu()),
            "leverage_max": float(leverage_tensor.max().detach().cpu()),
            "score_nan_count": float(torch.isnan(score_tensor).sum().detach().cpu()),
        }
        return score_tensor.reshape(*original_shape, 1), info


def make_ippo_mask_and_weight(
    scores: torch.Tensor,
    cfg: Any,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    flat_scores = torch.nan_to_num(scores.reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    num_samples = flat_scores.numel()
    mode = _cfg_get(cfg, "ippo_select_mode", "none")
    retain_ratio = float(_cfg_get(cfg, "ippo_retain_ratio", 0.5))
    min_ratio = float(_cfg_get(cfg, "ippo_min_retain_ratio", 0.1))
    max_ratio = float(_cfg_get(cfg, "ippo_max_retain_ratio", 1.0))
    retain_ratio = min(max(retain_ratio, min_ratio), max_ratio)
    threshold = float(_cfg_get(cfg, "ippo_threshold", 0.5))

    mask = torch.ones(num_samples, device=scores.device, dtype=torch.bool)
    selected_threshold = threshold

    if mode == "none" or num_samples == 0:
        pass
    elif mode == "weight_only":
        pass
    elif mode == "threshold":
        mask = flat_scores >= threshold
    elif mode == "random_same_ratio":
        mask = torch.rand(num_samples, device=scores.device) < retain_ratio
    elif mode == "topk":
        mask, selected_threshold = _topk_mask(flat_scores, retain_ratio)
    else:
        raise ValueError(f"Unknown ippo_select_mode: {mode}")

    retained_ratio = mask.float().mean() if num_samples > 0 else flat_scores.new_tensor(1.0)
    if num_samples > 0 and retained_ratio < min_ratio:
        mask, selected_threshold = _topk_mask(flat_scores, min_ratio)
        retained_ratio = mask.float().mean()

    weight_mode = _cfg_get(cfg, "ippo_weight_mode", "normalized_score")
    weight_clip = float(_cfg_get(cfg, "ippo_weight_clip", 3.0))
    eps = float(_cfg_get(cfg, "ippo_weight_eps", 1e-6))
    weights = torch.ones(num_samples, device=scores.device, dtype=scores.dtype)
    if mode == "none":
        weights = torch.ones(num_samples, device=scores.device, dtype=scores.dtype)
    elif mode == "random_same_ratio":
        weights = mask.to(scores.dtype)
    elif weight_mode == "normalized_score":
        weights = torch.zeros(num_samples, device=scores.device, dtype=scores.dtype)
        if mask.any():
            selected_scores = flat_scores[mask].to(scores.dtype).clamp_min(0.0)
            if selected_scores.mean() <= eps:
                weights[mask] = 1.0
            else:
                weights[mask] = (selected_scores / (selected_scores.mean() + eps)).clamp(0.0, weight_clip)
    elif weight_mode == "ones":
        weights = mask.to(scores.dtype)
    else:
        raise ValueError(f"Unknown ippo_weight_mode: {weight_mode}")

    info = {
        "retained_ratio": float(retained_ratio.detach().cpu()),
        "random_same_ratio_retained_ratio": float(retained_ratio.detach().cpu()) if mode == "random_same_ratio" else 0.0,
        "weight_mean": float(weights[mask].mean().detach().cpu()) if mask.any() else 0.0,
        "weight_std": float(weights[mask].std(unbiased=False).detach().cpu()) if mask.any() else 0.0,
        "weight_max": float(weights.max().detach().cpu()) if num_samples > 0 else 0.0,
        "threshold": float(selected_threshold),
    }
    return mask.reshape_as(scores), weights.reshape_as(scores), info


def _topk_mask(scores: torch.Tensor, retain_ratio: float) -> Tuple[torch.Tensor, float]:
    num_samples = scores.numel()
    k = int(math.ceil(float(retain_ratio) * num_samples))
    k = min(max(k, 1), num_samples)
    values, indices = torch.topk(scores, k=k, largest=True, sorted=False)
    mask = torch.zeros(num_samples, device=scores.device, dtype=torch.bool)
    mask[indices] = True
    threshold = float(values.min().detach().cpu())
    return mask, threshold


def build_ippo_statistics(
    scores: torch.Tensor,
    masks: torch.Tensor,
    weights: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
) -> Dict[str, float]:
    score_flat = torch.nan_to_num(scores.reshape(-1).float(), nan=0.0, posinf=0.0, neginf=0.0)
    mask_flat = masks.reshape(-1).bool()
    weight_flat = torch.nan_to_num(weights.reshape(-1).float(), nan=0.0, posinf=0.0, neginf=0.0)
    adv_flat = torch.nan_to_num(advantages.reshape(-1).float(), nan=0.0, posinf=0.0, neginf=0.0)
    td_flat = torch.nan_to_num((returns - values).reshape(-1).float(), nan=0.0, posinf=0.0, neginf=0.0)

    return {
        "score_mean": float(score_flat.mean().detach().cpu()),
        "score_std": float(score_flat.std(unbiased=False).detach().cpu()),
        "score_min": float(score_flat.min().detach().cpu()),
        "score_max": float(score_flat.max().detach().cpu()),
        "retained_ratio": float(mask_flat.float().mean().detach().cpu()),
        "weight_mean": float(weight_flat[mask_flat].mean().detach().cpu()) if mask_flat.any() else 0.0,
        "weight_std": float(weight_flat[mask_flat].std(unbiased=False).detach().cpu()) if mask_flat.any() else 0.0,
        "weight_max": float(weight_flat.max().detach().cpu()),
        "corr_score_abs_adv": float(safe_corr(score_flat, adv_flat.abs()).detach().cpu()),
        "corr_score_abs_td": float(safe_corr(score_flat, td_flat.abs()).detach().cpu()),
        "corr_score_loss": float(safe_corr(score_flat, adv_flat.abs()).detach().cpu()),
    }
