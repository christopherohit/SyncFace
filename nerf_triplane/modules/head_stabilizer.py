"""Head stabilizer.

Diagram block under the stability branch. The functional version below
penalizes second-order differences of the head-pose sequence, which is
the loss term we apply at training time when poses are being optimized
(`--train_camera`). The class version holds a small running buffer for
poses across consecutive optimizer steps so the same penalty can be
applied even when poses are loaded as data and not optimized — in that
case the gradient flows through whatever motion module produced the
deformations driving those poses.
"""

import torch
import torch.nn as nn


def head_pose_smoothness_loss(poses: torch.Tensor) -> torch.Tensor:
    """Second-difference penalty on a sequence of poses.

    Args:
        poses: [T, ...] tensor of pose parameters; T must be >= 3.

    Returns:
        scalar tensor; zero if T < 3.
    """
    if poses.size(0) < 3:
        return torch.zeros((), device=poses.device, dtype=poses.dtype)
    second_diff = poses[2:] - 2.0 * poses[1:-1] + poses[:-2]
    return (second_diff ** 2).mean()


class HeadStabilizer(nn.Module):
    """Running-buffer wrapper around `head_pose_smoothness_loss`.

    Keeps the last two poses across train_step calls and emits a
    second-difference penalty from t-2, t-1, t. No-op for the first
    two steps and whenever `lambda_stab == 0`.
    """

    def __init__(self, lambda_stab: float = 0.0):
        super().__init__()
        self.lambda_stab = float(lambda_stab)
        self.register_buffer("p_prev", torch.zeros(1), persistent=False)
        self.register_buffer("p_prev2", torch.zeros(1), persistent=False)
        self._n_seen = 0

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        device = pose.device
        if self.lambda_stab == 0.0:
            self.p_prev2 = self.p_prev.detach().clone() if self._n_seen >= 1 else pose.detach().clone()
            self.p_prev = pose.detach().clone()
            self._n_seen += 1
            return torch.zeros((), device=device, dtype=pose.dtype)

        if self._n_seen < 2 or self.p_prev.shape != pose.shape or self.p_prev2.shape != pose.shape:
            self.p_prev2 = self.p_prev.detach().clone() if self._n_seen >= 1 else pose.detach().clone()
            self.p_prev = pose.detach().clone()
            self._n_seen += 1
            return torch.zeros((), device=device, dtype=pose.dtype)

        second_diff = pose - 2.0 * self.p_prev + self.p_prev2
        loss = (second_diff ** 2).mean()
        self.p_prev2 = self.p_prev.detach().clone()
        self.p_prev = pose.detach().clone()
        self._n_seen += 1
        return self.lambda_stab * loss
