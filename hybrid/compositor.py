"""
Mouth Compositor
=================
Advanced compositing methods for mouth/face blending.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class MouthCompositor:
    """
    Compositor for blending NeRF mouth with Gaussian face.
    
    Supports multiple blending strategies:
    - Alpha blending with smooth falloff
    - Poisson blending for seamless integration
    - Color harmonization at boundaries
    """
    
    def __init__(
        self,
        blend_margin: int = 8,
        blend_type: str = 'smooth',  # 'smooth', 'poisson', 'feather'
        color_harmonize: bool = True,
    ):
        self.blend_margin = blend_margin
        self.blend_type = blend_type
        self.color_harmonize = color_harmonize
    
    def create_blend_mask(
        self,
        mouth_mask: torch.Tensor,
        dilate: bool = True,
    ) -> torch.Tensor:
        """
        Create smooth blending mask.
        
        Args:
            mouth_mask: [H, W] binary mask
            dilate: Whether to dilate before blurring
        
        Returns:
            [H, W] smooth blend weights
        """
        device = mouth_mask.device
        mask = mouth_mask.float()
        
        if dilate:
            # Dilate mask
            kernel = torch.ones(1, 1, 5, 5, device=device)
            mask = F.conv2d(
                mask.unsqueeze(0).unsqueeze(0),
                kernel, padding=2
            ).squeeze() > 0
            mask = mask.float()
        
        # Gaussian blur for smooth transition
        kernel_size = self.blend_margin * 2 + 1
        sigma = self.blend_margin / 2
        
        # Create Gaussian kernel
        x = torch.arange(kernel_size, device=device).float() - kernel_size // 2
        gauss_1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
        gauss_2d = gauss_1d.unsqueeze(0) * gauss_1d.unsqueeze(1)
        gauss_2d = gauss_2d / gauss_2d.sum()
        gauss_2d = gauss_2d.unsqueeze(0).unsqueeze(0)
        
        smooth = F.conv2d(
            mask.unsqueeze(0).unsqueeze(0),
            gauss_2d, padding=kernel_size // 2
        ).squeeze()
        
        return smooth.clamp(0, 1)
    
    def feather_blend(
        self,
        face: torch.Tensor,
        mouth: torch.Tensor,
        mask: torch.Tensor,
        rect: Tuple[int, int, int, int],
    ) -> torch.Tensor:
        """
        Feather blending with distance-based falloff.
        """
        x_min, y_min, x_max, y_max = rect
        h, w = mouth.shape[:2]
        
        # Create distance-based weights
        y_coords, x_coords = torch.meshgrid(
            torch.arange(h, device=face.device),
            torch.arange(w, device=face.device),
            indexing='ij'
        )
        
        center_y, center_x = h // 2, w // 2
        dist = torch.sqrt((y_coords - center_y).float()**2 + (x_coords - center_x).float()**2)
        max_dist = np.sqrt(center_y**2 + center_x**2)
        
        # Falloff from center
        weights = 1.0 - (dist / max_dist).clamp(0, 1)
        weights = weights.unsqueeze(-1)
        
        # Combine with mask
        mask_crop = mask[y_min:y_min+h, x_min:x_min+w].unsqueeze(-1)
        combined_weights = weights * mask_crop
        
        # Blend
        result = face.clone()
        face_region = face[y_min:y_min+h, x_min:x_min+w]
        result[y_min:y_min+h, x_min:x_min+w] = (
            face_region * (1 - combined_weights) + mouth * combined_weights
        )
        
        return result
    
    def harmonize_colors(
        self,
        face: torch.Tensor,
        mouth: torch.Tensor,
        mask: torch.Tensor,
        rect: Tuple[int, int, int, int],
    ) -> torch.Tensor:
        """
        Adjust mouth colors to match face at boundary.
        """
        x_min, y_min, x_max, y_max = rect
        h, w = mouth.shape[:2]
        
        # Get boundary region
        kernel = torch.ones(1, 1, 5, 5, device=face.device)
        dilated = F.conv2d(
            mask.float().unsqueeze(0).unsqueeze(0),
            kernel, padding=2
        ).squeeze() > 0
        
        boundary = dilated.float() - mask.float()
        
        # Get boundary colors
        boundary_crop = boundary[y_min:y_min+h, x_min:x_min+w]
        
        if boundary_crop.sum() > 0:
            face_region = face[y_min:y_min+h, x_min:x_min+w]
            
            # Mean color at boundary
            face_boundary_mean = (face_region * boundary_crop.unsqueeze(-1)).sum(dim=(0,1)) / (boundary_crop.sum() + 1e-6)
            mouth_boundary_mean = (mouth * boundary_crop.unsqueeze(-1)).sum(dim=(0,1)) / (boundary_crop.sum() + 1e-6)
            
            # Color shift
            color_shift = face_boundary_mean - mouth_boundary_mean
            
            # Apply shift to mouth
            mouth = mouth + color_shift.unsqueeze(0).unsqueeze(0) * 0.5
            mouth = mouth.clamp(0, 1)
        
        return mouth
    
    def composite(
        self,
        face: torch.Tensor,
        mouth: torch.Tensor,
        mouth_mask: torch.Tensor,
        lips_rect: Tuple[int, int, int, int],
    ) -> torch.Tensor:
        """
        Full compositing pipeline.
        
        Args:
            face: [H, W, 3] face render
            mouth: [h, w, 3] mouth render
            mouth_mask: [H, W] binary mask
            lips_rect: [x_min, y_min, x_max, y_max]
        
        Returns:
            [H, W, 3] composited image
        """
        x_min, y_min, x_max, y_max = lips_rect
        h, w = mouth.shape[:2]
        
        # Color harmonization
        if self.color_harmonize:
            mouth = self.harmonize_colors(face, mouth, mouth_mask, lips_rect)
        
        # Blending
        if self.blend_type == 'smooth':
            blend_mask = self.create_blend_mask(mouth_mask)
            
            result = face.clone()
            mask_crop = blend_mask[y_min:y_min+h, x_min:x_min+w].unsqueeze(-1)
            face_region = face[y_min:y_min+h, x_min:x_min+w]
            
            result[y_min:y_min+h, x_min:x_min+w] = (
                face_region * (1 - mask_crop) + mouth * mask_crop
            )
            return result
        
        elif self.blend_type == 'feather':
            return self.feather_blend(face, mouth, mouth_mask, lips_rect)
        
        else:  # Simple alpha blend
            result = face.clone()
            mask_crop = mouth_mask[y_min:y_min+h, x_min:x_min+w].unsqueeze(-1).float()
            face_region = face[y_min:y_min+h, x_min:x_min+w]
            
            result[y_min:y_min+h, x_min:x_min+w] = (
                face_region * (1 - mask_crop) + mouth * mask_crop
            )
            return result



