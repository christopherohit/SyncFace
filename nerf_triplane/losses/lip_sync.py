"""Lip-sync loss skeleton.

The architecture diagram lists "lip-sync" as one of the Stage-B losses.
The standard implementation uses a two-tower SyncNet (audio tower +
lip-image tower) and a cosine/contrastive distance between the two
embeddings on a short temporal window of the rendered mouth crop.

This module provides the contract:
    * a small image-side encoder that maps a [B, 3, Hm, Wm] mouth crop
      to a fixed-dim embedding;
    * a hook to load a pretrained SyncNet checkpoint (`load_syncnet`);
    * a `forward` that returns 0.0 when no checkpoint has been loaded
      *or* `lambda_lipsync == 0`, so it is a no-op by default.

A real SyncNet checkpoint can be plugged in later without touching the
training loop.
"""

import torch
import torch.nn as nn


class _LipImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x).flatten(1)
        return self.fc(h)


class LipSyncLoss(nn.Module):
    """Cosine-distance lip-sync loss between rendered mouth and audio.

    No-op when `lambda_lipsync == 0` or no checkpoint has been loaded.
    """

    def __init__(self, lambda_lipsync: float = 0.0, audio_dim: int = 512, image_dim: int = 512):
        super().__init__()
        self.lambda_lipsync = float(lambda_lipsync)
        self.image_encoder = _LipImageEncoder(out_dim=image_dim)
        self.audio_proj = nn.Linear(audio_dim, image_dim)
        self._ckpt_loaded = False

    def load_syncnet(self, path: str):
        """Load pretrained SyncNet weights.

        The checkpoint is expected to expose `image_encoder` and
        `audio_proj` keys; layouts from the public Wav2Lip / SyncNet
        repos differ, so adapter logic should live here rather than
        spreading through the trainer.
        """
        state = torch.load(path, map_location="cpu")
        if "image_encoder" in state and "audio_proj" in state:
            self.image_encoder.load_state_dict(state["image_encoder"])
            self.audio_proj.load_state_dict(state["audio_proj"])
            self._ckpt_loaded = True
        else:
            raise KeyError(
                "Expected keys 'image_encoder' and 'audio_proj' in the lip-sync "
                "checkpoint. Adapt LipSyncLoss.load_syncnet for your layout."
            )

    def forward(self, mouth_crop: torch.Tensor, audio_emb: torch.Tensor) -> torch.Tensor:
        """Args:
            mouth_crop: [B, 3, Hm, Wm] rendered mouth region in [0, 1].
            audio_emb:  [B, audio_dim] audio embedding (e.g. AVE output).
        """
        if self.lambda_lipsync == 0.0 or not self._ckpt_loaded:
            return torch.zeros((), device=mouth_crop.device, dtype=mouth_crop.dtype)

        v_emb = self.image_encoder(mouth_crop)
        a_emb = self.audio_proj(audio_emb)
        v_emb = nn.functional.normalize(v_emb, dim=-1)
        a_emb = nn.functional.normalize(a_emb, dim=-1)
        cos = (v_emb * a_emb).sum(dim=-1)
        return self.lambda_lipsync * (1.0 - cos).mean()
