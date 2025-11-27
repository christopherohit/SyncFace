"""
Hybrid Face Renderer
=====================
Combined Gaussian + NeRF rendering pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class HybridFaceRenderer:
    """
    Renders face using combined Gaussian splatting and NeRF.
    
    Pipeline:
    1. Render face with 3DGS (with mouth region masked)
    2. Render mouth with NeRF
    3. Composite with smooth blending
    """
    
    def __init__(
        self,
        model,  # SyncFaceModel
        pipe = None,
        blend_margin: int = 8,
    ):
        self.model = model
        self.pipe = pipe
        self.blend_margin = blend_margin
    
    def _create_blend_mask(
        self,
        mouth_mask: torch.Tensor,
        margin: int,
    ) -> torch.Tensor:
        """Create smooth blending mask for mouth region."""
        device = mouth_mask.device
        
        # Dilate mouth mask
        kernel_size = margin * 2 + 1
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)
        kernel = kernel / kernel.sum()
        
        smooth = F.conv2d(
            mouth_mask.float().unsqueeze(0).unsqueeze(0),
            kernel,
            padding=kernel_size // 2
        ).squeeze()
        
        return smooth.clamp(0, 1)
    
    def render(
        self,
        camera,
        audio: Optional[torch.Tensor] = None,
        blendshape: Optional[torch.Tensor] = None,
        mouth_mask: Optional[torch.Tensor] = None,
        lips_rect: Optional[Tuple[int, int, int, int]] = None,
        bg_color: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full hybrid rendering.
        
        Args:
            camera: Camera object with pose, intrinsics
            audio: Audio features
            blendshape: Blendshape coefficients
            mouth_mask: [H, W] binary mouth mask
            lips_rect: [x_min, y_min, x_max, y_max]
            bg_color: [3] background color
        
        Returns:
            Dict with final render and intermediate results
        """
        device = next(self.model.parameters()).device
        H, W = camera.height, camera.width
        
        if bg_color is None:
            bg_color = torch.zeros(3, device=device)
        
        # 1. Render face with Gaussians
        face_result = self._render_face(camera, audio, blendshape, bg_color)
        face_image = face_result['render']  # [3, H, W]
        face_alpha = face_result['alpha']   # [1, H, W]
        
        # 2. Render mouth with NeRF (if rect provided)
        if lips_rect is not None:
            mouth_result = self._render_mouth(
                camera, lips_rect, H, W,
                audio, blendshape, bg_color
            )
            
            # 3. Composite
            if mouth_mask is not None:
                final = self._composite(
                    face_image, mouth_result['rgb'],
                    mouth_mask, lips_rect
                )
            else:
                final = face_image
            
            return {
                'render': final,
                'face_render': face_image,
                'mouth_render': mouth_result['rgb'],
                'alpha': face_alpha,
                'mouth_alpha': mouth_result['alpha'],
                'motion': face_result.get('motion'),
            }
        
        return {
            'render': face_image,
            'alpha': face_alpha,
            'motion': face_result.get('motion'),
        }
    
    def _render_face(
        self,
        camera,
        audio: Optional[torch.Tensor],
        blendshape: Optional[torch.Tensor],
        bg_color: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Render face with Gaussian splatting."""
        # Get deformed Gaussians
        xyz = self.model.gaussians.get_xyz
        motion = self.model.motion_net(xyz, audio, blendshape)
        
        deformed_xyz = xyz + motion['d_xyz']
        
        # Use diff-gaussian-rasterization
        try:
            from diff_gauss import GaussianRasterizer, GaussianRasterizationSettings
            
            # Setup rasterization settings
            settings = GaussianRasterizationSettings(
                image_height=camera.height,
                image_width=camera.width,
                tanfovx=camera.tanfovx,
                tanfovy=camera.tanfovy,
                bg=bg_color,
                scale_modifier=1.0,
                viewmatrix=camera.world_view_transform,
                projmatrix=camera.full_proj_transform,
                sh_degree=self.model.gaussians.active_sh_degree,
                campos=camera.camera_center,
                prefiltered=False,
                debug=False,
            )
            
            rasterizer = GaussianRasterizer(raster_settings=settings)
            
            # Rasterize
            rendered_image, radii = rasterizer(
                means3D=deformed_xyz,
                means2D=torch.zeros_like(deformed_xyz[:, :2]),
                shs=self.model.gaussians.get_features,
                colors_precomp=None,
                opacities=self.model.gaussians.get_opacity,
                scales=self.model.gaussians.get_scaling,
                rotations=self.model.gaussians.get_rotation,
                cov3D_precomp=None,
            )
            
            return {
                'render': rendered_image,
                'alpha': torch.ones(1, camera.height, camera.width, device=xyz.device),
                'motion': motion,
            }
            
        except ImportError:
            # Fallback - return placeholder
            device = xyz.device
            return {
                'render': torch.zeros(3, camera.height, camera.width, device=device),
                'alpha': torch.zeros(1, camera.height, camera.width, device=device),
                'motion': motion,
            }
    
    def _render_mouth(
        self,
        camera,
        lips_rect: Tuple[int, int, int, int],
        H: int,
        W: int,
        audio: Optional[torch.Tensor],
        blendshape: Optional[torch.Tensor],
        bg_color: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Render mouth with NeRF."""
        # Get camera pose
        pose = camera.get_world_to_camera_transform().inverse()
        focal = camera.focal_x if hasattr(camera, 'focal_x') else W / 2
        
        return self.model.mouth_renderer.render_mouth_region(
            pose, lips_rect, H, W, focal,
            audio, blendshape, bg_color=bg_color
        )
    
    def _composite(
        self,
        face_image: torch.Tensor,
        mouth_rgb: torch.Tensor,
        mouth_mask: torch.Tensor,
        lips_rect: Tuple[int, int, int, int],
    ) -> torch.Tensor:
        """Composite mouth onto face."""
        x_min, y_min, x_max, y_max = lips_rect
        
        # Create blend mask
        blend_mask = self._create_blend_mask(mouth_mask, self.blend_margin)
        
        # Composite
        face_hwc = face_image.permute(1, 2, 0)  # [H, W, 3]
        result = face_hwc.clone()
        
        h, w = mouth_rgb.shape[:2]
        if y_min + h <= face_hwc.shape[0] and x_min + w <= face_hwc.shape[1]:
            mouth_region = blend_mask[y_min:y_min+h, x_min:x_min+w].unsqueeze(-1)
            face_region = face_hwc[y_min:y_min+h, x_min:x_min+w]
            
            result[y_min:y_min+h, x_min:x_min+w] = (
                face_region * (1 - mouth_region) + mouth_rgb * mouth_region
            )
        
        return result.permute(2, 0, 1)  # [3, H, W]



