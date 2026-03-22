import torch
import torch.nn as nn


class TemporalConsistencyLoss(nn.Module):
    """Temporal consistency loss for NeRF-based talking head synthesis.

    Encourages smooth predictions across consecutive frames by penalizing
    large differences between the current frame's prediction and the previous
    frame's prediction (detached from the computation graph).

    Args:
        warmup_steps: Number of global training steps before the loss activates.
        lambda_temporal: Weighting factor for the temporal loss term.
        loss_type: Type of loss to use ('l1' or 'l2').
    """

    def __init__(self, warmup_steps=5000, lambda_temporal=0.1, loss_type='l1'):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.lambda_temporal = lambda_temporal
        self.loss_type = loss_type
        self.register_buffer('prev_pred', torch.zeros(1), persistent=False)
        self._has_prev = False

    def forward(self, pred_rgb, face_mask, global_step):
        """Compute temporal consistency loss.

        Args:
            pred_rgb: [B, N, 3] or [B*N, 3] current frame prediction.
            face_mask: [B, N] or [B*N] boolean mask for face region.
            global_step: Current training step.

        Returns:
            Scalar loss tensor. Zero during warmup or first frame.
        """
        device = pred_rgb.device

        if global_step < self.warmup_steps or not self._has_prev:
            self.prev_pred = pred_rgb.detach().clone()
            self._has_prev = True
            return torch.tensor(0.0, device=device, requires_grad=False)

        prev = self.prev_pred

        if prev.shape != pred_rgb.shape:
            self.prev_pred = pred_rgb.detach().clone()
            return torch.tensor(0.0, device=device, requires_grad=False)

        if self.loss_type == 'l1':
            diff = torch.abs(pred_rgb - prev)
        else:
            diff = (pred_rgb - prev) ** 2

        diff = diff.mean(dim=-1)
        face_mask_flat = face_mask.view(-1).float()
        diff_flat = diff.view(-1)
        masked_diff = diff_flat * face_mask_flat
        num_face_pixels = face_mask_flat.sum().clamp(min=1.0)
        loss = masked_diff.sum() / num_face_pixels
        loss = self.lambda_temporal * loss

        self.prev_pred = pred_rgb.detach().clone()
        return loss
