import torch
import torch.nn as nn


TERRAIN_LABEL_NAMES = [
    "step_up_score",
    "slope_score",
    "traction_loss_score",
    "instability_score",
    "stall_score",
]


class MoeTerrainEstimator(nn.Module):
    """
    Proprioceptive terrain/state estimator.

    Input:
        obs_hist: [B, T, n_proprio] or [B, T * n_proprio]

    Output:
        v_hat:    [B, 3]
        gray_hat: [B, 5]
            0 step_up_score          in [0, 1]
            1 slope_score            in [-1, 1]
            2 traction_loss_score    in [0, 1]
            3 instability_score      in [0, 1]
            4 stall_score            in [0, 1]
    """

    def __init__(
        self,
        n_proprio=33,
        history_len=10,
        hidden_dims=(256, 128, 64),
    ):
        super().__init__()

        self.n_proprio = int(n_proprio)
        self.history_len = int(history_len)
        input_dim = self.n_proprio * self.history_len

        layers = []
        last_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ELU())
            last_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        self.vel_head = nn.Linear(last_dim, 3)

        # sigmoid scores:
        # 0 step_up_score
        # 1 traction_loss_score
        # 2 instability_score
        # 3 stall_score
        self.score_head = nn.Linear(last_dim, 4)

        # signed terrain slope score in [-1, 1]
        self.slope_head = nn.Linear(last_dim, 1)

    def forward(self, obs_hist):
        if obs_hist.dim() == 3:
            x = obs_hist.reshape(obs_hist.shape[0], -1)
        elif obs_hist.dim() == 2:
            x = obs_hist
        else:
            raise RuntimeError(f"Unexpected obs_hist shape: {tuple(obs_hist.shape)}")

        h = self.encoder(x)

        v_hat = self.vel_head(h)

        scores = torch.sigmoid(self.score_head(h))
        slope = torch.tanh(self.slope_head(h))

        gray_hat = torch.cat(
            [
                scores[:, 0:1],  # step_up_score
                slope,           # slope_score
                scores[:, 1:2],  # traction_loss_score
                scores[:, 2:3],  # instability_score
                scores[:, 3:4],  # stall_score
            ],
            dim=-1,
        )

        return v_hat, gray_hat
