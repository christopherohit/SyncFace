"""Motion aligner.

Receives the audio motion code and (optionally) prosody / AU / pose
features and emits an aligned motion latent that downstream modules
(`DeformationMLP`, `MouthBranch`) consume. Splitting alignment out of
the audio encoder lets Stage A pretrain the aligner across identities
while Stage B fine-tunes only the small adapter layers.

Not yet wired into NeRFNetwork; Phase 2 uses the audio path that already
exists. This stub exists so the shape contract is fixed for Stage A.
"""

import torch
import torch.nn as nn


class MotionAligner(nn.Module):
    def __init__(
        self,
        audio_dim: int = 32,
        prosody_dim: int = 0,
        au_dim: int = 0,
        pose_dim: int = 0,
        out_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        in_dim = audio_dim + prosody_dim + au_dim + pose_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.02, inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(
        self,
        audio: torch.Tensor,
        prosody: torch.Tensor = None,
        au: torch.Tensor = None,
        pose: torch.Tensor = None,
    ) -> torch.Tensor:
        feats = [audio]
        if prosody is not None:
            feats.append(prosody)
        if au is not None:
            feats.append(au)
        if pose is not None:
            feats.append(pose)
        x = torch.cat(feats, dim=-1)
        return self.net(x)
