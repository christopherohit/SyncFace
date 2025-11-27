"""
NeRF Renderer for Mouth Region
===============================
Volumetric ray marching renderer for mouth interior.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class NeRFRenderer:
    """
    Volumetric renderer for mouth NeRF.
    
    Features:
    - Pure PyTorch implementation (no CUDA raymarching required)
    - Lips rect filtering for efficiency
    - Smooth boundary compositing
    """
    
    def __init__(
        self,
        network,
        near: float = 0.01,
        far: float = 1.0,
        num_samples: int = 64,
        perturb: bool = True,
    ):
        self.network = network
        self.near = near
        self.far = far
        self.num_samples = num_samples
        self.perturb = perturb
    
    def render_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        blendshape: Optional[torch.Tensor] = None,
        ind_code: Optional[int] = None,
        bg_color: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Render rays using volumetric integration.
        
        Args:
            rays_o: [N, 3] ray origins
            rays_d: [N, 3] ray directions (normalized)
            audio: Audio features
            blendshape: Blendshape coefficients
            ind_code: Individual code index
            bg_color: [3] background color
        
        Returns:
            Dict with rgb, depth, alpha, weights
        """
        N = rays_o.shape[0]
        device = rays_o.device
        
        # Sample points along rays
        t_vals = torch.linspace(self.near, self.far, self.num_samples, device=device)
        
        if self.perturb and self.network.training:
            mids = 0.5 * (t_vals[:-1] + t_vals[1:])
            upper = torch.cat([mids, t_vals[-1:]])
            lower = torch.cat([t_vals[:1], mids])
            t_rand = torch.rand(self.num_samples, device=device)
            t_vals = lower + (upper - lower) * t_rand
        
        t_vals = t_vals.expand(N, -1)  # [N, S]
        
        # Compute 3D sample positions
        pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_vals.unsqueeze(-1)  # [N, S, 3]
        pts_flat = pts.reshape(-1, 3)
        
        # Expand directions
        dirs = rays_d.unsqueeze(1).expand(-1, self.num_samples, -1).reshape(-1, 3)
        
        # Query network
        sigma, color, _ = self.network(pts_flat, dirs, audio, blendshape, ind_code)
        
        sigma = sigma.reshape(N, self.num_samples)
        color = color.reshape(N, self.num_samples, 3)
        
        # Volumetric rendering
        dists = t_vals[:, 1:] - t_vals[:, :-1]
        dists = torch.cat([dists, torch.full((N, 1), 1e10, device=device)], dim=-1)
        
        # Alpha from density
        alpha = 1.0 - torch.exp(-sigma * dists)
        
        # Transmittance
        T = torch.cumprod(
            torch.cat([torch.ones(N, 1, device=device), 1.0 - alpha + 1e-10], dim=-1),
            dim=-1
        )[:, :-1]
        
        # Weights
        weights = alpha * T
        
        # Weighted sum
        rgb = (weights.unsqueeze(-1) * color).sum(dim=1)
        depth = (weights * t_vals).sum(dim=-1)
        acc_alpha = weights.sum(dim=-1)
        
        # Add background
        if bg_color is not None:
            if len(bg_color.shape) == 1:
                bg_color = bg_color.unsqueeze(0).expand(N, -1)
            rgb = rgb + (1.0 - acc_alpha.unsqueeze(-1)) * bg_color
        
        return {
            'rgb': rgb,
            'depth': depth,
            'alpha': acc_alpha,
            'weights': weights,
        }
    
    def render_mouth_region(
        self,
        pose: torch.Tensor,
        lips_rect: Tuple[int, int, int, int],
        H: int,
        W: int,
        focal: float,
        audio: Optional[torch.Tensor] = None,
        blendshape: Optional[torch.Tensor] = None,
        ind_code: Optional[int] = None,
        bg_color: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Render entire mouth region.
        
        Args:
            pose: [4, 4] camera-to-world matrix
            lips_rect: [x_min, y_min, x_max, y_max]
            H, W: Image dimensions
            focal: Focal length
            audio: Audio features
            blendshape: Blendshape coefficients
            ind_code: Individual code
            bg_color: Background color
        
        Returns:
            Dict with rendered mouth image and metadata
        """
        device = pose.device
        x_min, y_min, x_max, y_max = lips_rect
        
        cx, cy = W / 2, H / 2
        
        # Generate pixel grid
        xx, yy = torch.meshgrid(
            torch.arange(x_min, x_max, device=device, dtype=torch.float32),
            torch.arange(y_min, y_max, device=device, dtype=torch.float32),
            indexing='xy'
        )
        
        # Ray directions in camera space
        directions = torch.stack([
            (xx - cx) / focal,
            -(yy - cy) / focal,
            -torch.ones_like(xx)
        ], dim=-1)
        directions = directions / directions.norm(dim=-1, keepdim=True)
        
        # Transform to world space
        rays_d = directions.reshape(-1, 3) @ pose[:3, :3].T
        rays_o = pose[:3, 3].expand(rays_d.shape[0], -1)
        
        # Render
        result = self.render_rays(rays_o, rays_d, audio, blendshape, ind_code, bg_color)
        
        # Reshape to image
        h = y_max - y_min
        w = x_max - x_min
        
        result['rgb'] = result['rgb'].reshape(h, w, 3)
        result['depth'] = result['depth'].reshape(h, w)
        result['alpha'] = result['alpha'].reshape(h, w)
        result['rect'] = lips_rect
        
        return result
    
    def composite_with_face(
        self,
        face_image: torch.Tensor,
        mouth_result: Dict[str, torch.Tensor],
        mouth_mask: torch.Tensor,
        blend_margin: int = 8,
    ) -> torch.Tensor:
        """
        Composite mouth render onto face image.
        
        Args:
            face_image: [H, W, 3] face render
            mouth_result: Output from render_mouth_region
            mouth_mask: [H, W] binary mouth mask
            blend_margin: Pixels for boundary blending
        
        Returns:
            [H, W, 3] composited image
        """
        H, W = face_image.shape[:2]
        device = face_image.device
        
        lips_rect = mouth_result['rect']
        x_min, y_min, x_max, y_max = lips_rect
        
        mouth_rgb = mouth_result['rgb']
        
        # Create smooth blend mask
        kernel_size = blend_margin * 2 + 1
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)
        kernel = kernel / kernel.sum()
        
        blend_mask = F.conv2d(
            mouth_mask.float().unsqueeze(0).unsqueeze(0),
            kernel,
            padding=kernel_size // 2
        ).squeeze().clamp(0, 1)
        
        # Insert mouth into full image
        result = face_image.clone()
        
        h, w = mouth_rgb.shape[:2]
        if y_min + h <= H and x_min + w <= W:
            mouth_region = blend_mask[y_min:y_min+h, x_min:x_min+w].unsqueeze(-1)
            face_region = face_image[y_min:y_min+h, x_min:x_min+w]
            
            result[y_min:y_min+h, x_min:x_min+w] = (
                face_region * (1 - mouth_region) + mouth_rgb * mouth_region
            )
        
        return result



