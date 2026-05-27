"""Negative-contrast / disentangle loss for Stage A pretraining.

Stage A in the architecture diagram trains a *universal motion prior*
across many identities while keeping per-identity appearance fields
disentangled. The standard formulation is a triplet/contrastive loss
on the motion code:

    * positive pair: same audio, different identity -> motion code
      should be close (motion is identity-invariant);
    * negative pair: different audio, same identity -> motion code
      should diverge.

This module is invoked only by the Stage A training entrypoint; for
Stage B it is unused. The default lambda is 0 so importing the module
never affects single-identity Stage B training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NegativeContrastDisentangleLoss(nn.Module):
    def __init__(self, lambda_disentangle: float = 0.0, margin: float = 0.2):
        super().__init__()
        self.lambda_disentangle = float(lambda_disentangle)
        self.margin = margin

    def forward(
        self,
        motion_anchor: torch.Tensor,
        motion_positive: torch.Tensor,
        motion_negative: torch.Tensor,
    ) -> torch.Tensor:
        """Cosine-margin triplet loss on motion codes.

        Args:
            motion_anchor:   [B, D] anchor motion code.
            motion_positive: [B, D] same-audio / different-identity code.
            motion_negative: [B, D] different-audio / same-identity code.
        """
        if self.lambda_disentangle == 0.0:
            return torch.zeros((), device=motion_anchor.device, dtype=motion_anchor.dtype)

        a = F.normalize(motion_anchor, dim=-1)
        p = F.normalize(motion_positive, dim=-1)
        n = F.normalize(motion_negative, dim=-1)
        d_pos = 1.0 - (a * p).sum(dim=-1)
        d_neg = 1.0 - (a * n).sum(dim=-1)
        loss = F.relu(d_pos - d_neg + self.margin).mean()
        return self.lambda_disentangle * loss
