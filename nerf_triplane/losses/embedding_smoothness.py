"""Embedding smoothness loss for the audio motion code.

The architecture diagram lists "temporal / embedding smoothness" as one
of the Stage-B regularizers. This penalizes large jumps in the encoded
audio feature `enc_a` between consecutive optimizer steps, which
corresponds to consecutive frames during training.
"""

import torch
import torch.nn as nn


class EmbeddingSmoothnessLoss(nn.Module):
    """L2 distance between current and previous detached audio embedding."""

    def __init__(self, lambda_embsmooth: float = 0.0):
        super().__init__()
        self.lambda_embsmooth = float(lambda_embsmooth)
        self.register_buffer("prev_emb", torch.zeros(1), persistent=False)
        self._has_prev = False

    def forward(self, enc_a: torch.Tensor) -> torch.Tensor:
        device = enc_a.device
        if self.lambda_embsmooth == 0.0:
            self.prev_emb = enc_a.detach().clone()
            self._has_prev = True
            return torch.zeros((), device=device, dtype=enc_a.dtype)

        if not self._has_prev or self.prev_emb.shape != enc_a.shape:
            self.prev_emb = enc_a.detach().clone()
            self._has_prev = True
            return torch.zeros((), device=device, dtype=enc_a.dtype)

        loss = ((enc_a - self.prev_emb) ** 2).mean()
        self.prev_emb = enc_a.detach().clone()
        return self.lambda_embsmooth * loss
