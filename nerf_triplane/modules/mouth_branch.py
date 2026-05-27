"""Mouth branch.

A second triplane + small MLP gated by the mouth mask. The full network
blends face-branch and mouth-branch density/color along the lower-face
mask so the mouth interior (teeth, tongue, lip-line transitions) gets
its own representation budget and is more responsive to audio.

Wiring this into `NeRFNetwork.density` is Phase 2a; for now this module
defines the contract so Stage A pretraining can already include it in
the shared-prior parameter group.
"""

import torch
import torch.nn as nn


class MouthBranch(nn.Module):
    def __init__(
        self,
        triplane_feature_dim: int,
        motion_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 3,
        out_geo_feat_dim: int = 64,
    ):
        super().__init__()
        layers = []
        in_dim = triplane_feature_dim + motion_dim
        for i in range(num_layers):
            out = (1 + out_geo_feat_dim) if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, out))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
        self.out_geo_feat_dim = out_geo_feat_dim

    def forward(self, enc_x: torch.Tensor, motion: torch.Tensor):
        m = motion.expand(enc_x.shape[0], -1) if motion.size(0) == 1 else motion
        h = self.net(torch.cat([enc_x, m], dim=-1))
        sigma = torch.exp(h[..., 0])
        geo_feat = h[..., 1:]
        return sigma, geo_feat
