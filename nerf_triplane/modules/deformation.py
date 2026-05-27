"""Deformation MLP.

Predicts a per-point Δxyz to apply *before* the canonical hash-grid
encoding, conditioned on the aligned motion latent. This separates the
motion modeling (audio/prosody/AU → Δxyz) from the canonical geometry
(σ from xyz) so that during Stage A the aligned-motion-aware
deformation can be shared across identities while each identity keeps
its own canonical hash grid.

Not yet called from NeRFNetwork.forward; the existing audio→sigma path
remains in place. Hooking this in is Phase 2c (see plan).
"""

import torch
import torch.nn as nn


class DeformationMLP(nn.Module):
    def __init__(
        self,
        xyz_dim: int = 3,
        motion_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 4,
        max_offset: float = 0.05,
    ):
        super().__init__()
        layers = []
        in_dim = xyz_dim + motion_dim
        for i in range(num_layers):
            out = 3 if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, out))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
        self.max_offset = max_offset

    def forward(self, xyz: torch.Tensor, motion: torch.Tensor) -> torch.Tensor:
        m = motion.expand(xyz.shape[0], -1) if motion.dim() == 2 and motion.size(0) == 1 else motion
        h = torch.cat([xyz, m], dim=-1)
        delta = self.net(h)
        return torch.tanh(delta) * self.max_offset
