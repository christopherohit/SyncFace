"""
Loss functions for InsTaG-inspired training:
- Negative Contrast Loss (NCLoss): Encourages orthogonality between personalized fields
- Geometry Prior Regularizer: Enforces geometric consistency using monocular depth/normal estimates
- Scale-Invariant Depth Loss: Robust depth supervision
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class NegativeContrastLoss(nn.Module):
    """
    Negative Contrast Loss (NCLoss) for pre-training.
    
    Encourages personalized motion fields of different identities to be orthogonal,
    which helps the UMF capture shared motion patterns while PersonalizedFields
    capture identity-specific variations.
    
    For video i and a randomly sampled video j (j != i) in the batch:
        L_C(delta_x_personal^i, delta_x_personal^j) = max(0, <delta_x_personal^i, delta_x_personal^j>)
    
    Where <·,·> denotes inner product (dot product) of flattened displacement vectors.
    
    Total loss: L_pre = L_photometric + lambda_C * sum_{j != i} L_C
    """
    
    def __init__(self, lambda_c: float = 0.01):
        super().__init__()
        self.lambda_c = lambda_c
    
    def forward(self, delta_personal_dict: Dict[int, torch.Tensor]) -> torch.Tensor:
        """
        Compute negative contrast loss across all identity pairs in batch.
        
        Args:
            delta_personal_dict: Dictionary mapping identity_id -> delta_x_personal tensor [N, 3]
                                Each tensor contains displacement vectors for one identity.
        
        Returns:
            loss: Scalar negative contrast loss
        """
        identity_ids = list(delta_personal_dict.keys())
        n_identities = len(identity_ids)
        
        if n_identities < 2:
            # Need at least 2 identities for contrast
            return torch.tensor(0.0, device=next(iter(delta_personal_dict.values())).device)
        
        total_loss = 0.0
        num_pairs = 0
        
        # Iterate over all pairs (i, j) where i != j
        for i in range(n_identities):
            for j in range(i + 1, n_identities):
                id_i = identity_ids[i]
                id_j = identity_ids[j]
                
                delta_i = delta_personal_dict[id_i]  # [N_i, 3]
                delta_j = delta_personal_dict[id_j]  # [N_j, 3]
                
                # Flatten and normalize
                delta_i_flat = delta_i.reshape(-1)  # [N_i * 3]
                delta_j_flat = delta_j.reshape(-1)  # [N_j * 3]
                
                # L2 normalize to make it scale-invariant
                delta_i_norm = F.normalize(delta_i_flat, p=2, dim=0)
                delta_j_norm = F.normalize(delta_j_flat, p=2, dim=0)
                
                # If sizes differ, truncate or pad (truncate is simpler)
                min_len = min(len(delta_i_norm), len(delta_j_norm))
                delta_i_norm = delta_i_norm[:min_len]
                delta_j_norm = delta_j_norm[:min_len]
                
                # Compute inner product
                inner_prod = torch.dot(delta_i_norm, delta_j_norm)
                
                # Hinge loss: penalize positive correlation
                loss_ij = torch.clamp(inner_prod, min=0.0)
                
                total_loss += loss_ij
                num_pairs += 1
        
        # Average over all pairs
        if num_pairs > 0:
            total_loss = total_loss / num_pairs
        
        return self.lambda_c * total_loss


class ScaleInvariantDepthLoss(nn.Module):
    """
    Scale-Invariant Depth Loss for geometry prior.
    
    Computes scale and shift invariant error between predicted and estimated depth:
        L_D = sqrt(E[(log d_pred - log d_est)^2] - alpha * E[log d_pred - log d_est]^2)
    
    where alpha in [0, 1] controls the scale-invariance (alpha=1 is fully scale-invariant).
    
    Reference: Eigen et al. "Depth Map Prediction from a Single Image using a Multi-Scale Deep Network"
    """
    
    def __init__(self, alpha: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
    
    def forward(self, d_pred: torch.Tensor, d_est: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute scale-invariant depth loss.
        
        Args:
            d_pred: [N,] predicted depth values
            d_est: [N,] estimated depth values (from monocular estimator)
            mask: [N,] optional validity mask (e.g., to exclude background)
        
        Returns:
            loss: Scalar depth loss
        """
        # Apply mask if provided
        if mask is not None:
            d_pred = d_pred[mask]
            d_est = d_est[mask]
        
        if d_pred.numel() == 0:
            return torch.tensor(0.0, device=d_pred.device)
        
        # Clamp to avoid log(0)
        d_pred = torch.clamp(d_pred, min=self.eps)
        d_est = torch.clamp(d_est, min=self.eps)
        
        # Compute log difference
        log_diff = torch.log(d_pred) - torch.log(d_est)  # [N,]
        
        # Scale-invariant loss
        # L_D = E[log_diff^2] - alpha * E[log_diff]^2
        mean_log_diff = log_diff.mean()
        mean_log_diff_sq = (log_diff ** 2).mean()
        
        loss = torch.sqrt(torch.clamp(
            mean_log_diff_sq - self.alpha * (mean_log_diff ** 2),
            min=0.0
        ))
        
        return loss


class NormalConsistencyLoss(nn.Module):
    """
    Normal Consistency Loss for geometry prior.
    
    Encourages predicted surface normals to align with estimated normals:
        L_N = sum_i (1 - dot(N_pred_i, N_est_i))
    
    where dot(·,·) is the dot product of normalized normal vectors.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, n_pred: torch.Tensor, n_est: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute normal consistency loss.
        
        Args:
            n_pred: [N, 3] predicted normals (should be normalized)
            n_est: [N, 3] estimated normals from monocular estimator
            mask: [N,] optional validity mask
        
        Returns:
            loss: Scalar normal consistency loss
        """
        # Apply mask if provided
        if mask is not None:
            n_pred = n_pred[mask]
            n_est = n_est[mask]
        
        if n_pred.shape[0] == 0:
            return torch.tensor(0.0, device=n_pred.device)
        
        # Normalize both
        n_pred = F.normalize(n_pred, p=2, dim=-1)
        n_est = F.normalize(n_est, p=2, dim=-1)
        
        # Compute cosine similarity (dot product)
        cos_sim = (n_pred * n_est).sum(dim=-1)  # [N,]
        
        # Loss: 1 - cos_sim (0 when aligned, 2 when opposite)
        loss = (1.0 - cos_sim).mean()
        
        return loss


class GeometryPriorRegularizer(nn.Module):
    """
    Geometry Prior Regularizer combining depth and normal consistency.
    
    Total geometry loss:
        L_Geo = lambda_D * L_D(D_pred, D_est) + lambda_N * L_N(N_pred, N_est)
    
    Where:
        - D_pred, N_pred: predicted depth and normals from NeRF rendering
        - D_est, N_est: estimated from pre-trained monocular geometry estimator
        - lambda_D, lambda_N: loss weights
    """
    
    def __init__(self, lambda_d: float = 0.1, lambda_n: float = 0.05, 
                 alpha_depth: float = 0.5):
        super().__init__()
        self.lambda_d = lambda_d
        self.lambda_n = lambda_n
        
        self.depth_loss = ScaleInvariantDepthLoss(alpha=alpha_depth)
        self.normal_loss = NormalConsistencyLoss()
    
    def forward(self, 
                d_pred: torch.Tensor, 
                d_est: torch.Tensor,
                n_pred: Optional[torch.Tensor] = None,
                n_est: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute geometry prior loss.
        
        Args:
            d_pred: [N,] predicted depth
            d_est: [N,] estimated depth
            n_pred: [N, 3] predicted normals (optional)
            n_est: [N, 3] estimated normals (optional)
            mask: [N,] validity mask
        
        Returns:
            loss_dict: Dictionary with 'total', 'depth', 'normal' losses
        """
        loss_dict = {}
        
        # Depth loss
        loss_d = self.depth_loss(d_pred, d_est, mask)
        loss_dict['depth'] = loss_d
        
        # Normal loss (if provided)
        if n_pred is not None and n_est is not None:
            loss_n = self.normal_loss(n_pred, n_est, mask)
            loss_dict['normal'] = loss_n
        else:
            loss_n = torch.tensor(0.0, device=d_pred.device)
            loss_dict['normal'] = loss_n
        
        # Total geometry loss
        loss_total = self.lambda_d * loss_d + self.lambda_n * loss_n
        loss_dict['total'] = loss_total
        
        return loss_dict


def compute_normals_from_depth(depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Compute surface normals from depth map using finite differences.
    
    Args:
        depth: [H, W] depth map
        K: [3, 3] camera intrinsic matrix
    
    Returns:
        normals: [H, W, 3] surface normals
    """
    H, W = depth.shape
    
    # Compute gradients
    depth_dx = depth[:, 1:] - depth[:, :-1]  # [H, W-1]
    depth_dy = depth[1:, :] - depth[:-1, :]  # [H-1, W]
    
    # Pad to original size
    depth_dx = F.pad(depth_dx, (0, 1), mode='replicate')  # [H, W]
    depth_dy = F.pad(depth_dy, (0, 0, 0, 1), mode='replicate')  # [H, W]
    
    # Camera parameters
    fx = K[0, 0]
    fy = K[1, 1]
    
    # Normal computation (simplified)
    # In camera space: dz/dx and dz/dy give tangent vectors
    normal_x = -depth_dx / fx
    normal_y = -depth_dy / fy
    normal_z = torch.ones_like(depth)
    
    normals = torch.stack([normal_x, normal_y, normal_z], dim=-1)  # [H, W, 3]
    normals = F.normalize(normals, p=2, dim=-1)
    
    return normals


# Hyperparameter ranges and recommendations
"""
Recommended hyperparameter ranges:

Pre-training phase:
    - lambda_C (NCLoss weight): 0.001 - 0.05
      Start with 0.01, increase if personalized fields are too similar
      Decrease if training becomes unstable
    
Adaptation phase:
    - lambda_D (depth loss weight): 0.05 - 0.2
      Higher for datasets with good depth estimates
      Lower if depth estimates are noisy
    
    - lambda_N (normal loss weight): 0.01 - 0.1
      Typically 2-5x smaller than lambda_D
      Can be 0 if normals are not available
    
    - alpha_depth (scale-invariance): 0.3 - 0.7
      Higher values (0.5-0.7) for more scale-invariance
      Lower values (0.3-0.4) when scale is reliable
    
    - Learning rate for adaptation:
      - Static Field: 1e-2 to 5e-2
      - Motion Aligner: 1e-3 to 1e-2  
      - Personalized Field: 1e-3 to 5e-3
      - UMF: 0 (frozen)

General guidelines:
    - Start with recommended default values
    - Monitor loss curves and rendered quality
    - Adjust weights if one loss term dominates
    - Use gradient clipping if training is unstable
"""


