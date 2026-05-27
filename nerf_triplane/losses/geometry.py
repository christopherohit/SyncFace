"""Geometry regularizer.

The architecture diagram lists a "geometry regularizer" inside the
stability branch. We provide a finite-difference Eikonal-style
regularizer on the density field: the magnitude of the discrete
gradient of `sigma` w.r.t. point position should not blow up. This is
cheaper than an autograd-based Eikonal and reuses the same finite
samples that the existing ambient regularizer in the trainer already
draws.
"""

import torch
import torch.nn as nn


class EikonalDensityLoss(nn.Module):
    """Penalize ‖σ(x+δ) − σ(x)‖² / ‖δ‖² so density gradients stay bounded.

    Args:
        lambda_geom: weight on the loss; 0.0 disables.
        target_norm: reference gradient norm; deviation from this is penalized.
    """

    def __init__(self, lambda_geom: float = 0.0, target_norm: float = 0.0):
        super().__init__()
        self.lambda_geom = float(lambda_geom)
        self.target_norm = float(target_norm)

    def forward(
        self,
        sigma_a: torch.Tensor,
        sigma_b: torch.Tensor,
        delta: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            sigma_a: σ at xyz, shape [N].
            sigma_b: σ at xyz+δ, shape [N].
            delta:   xyz perturbation, shape [N, 3].
        """
        if self.lambda_geom == 0.0:
            return torch.zeros((), device=sigma_a.device, dtype=sigma_a.dtype)

        delta_norm = delta.norm(dim=-1).clamp(min=1e-6)
        grad_norm = (sigma_b - sigma_a).abs() / delta_norm
        residual = (grad_norm - self.target_norm) ** 2
        return self.lambda_geom * residual.mean()
