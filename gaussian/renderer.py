"""
Gaussian Renderer
==================
Wrapper for diff-gaussian-rasterization.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

# Try to import diff-gaussian-rasterization
try:
    from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
    DIFF_GAUSS_AVAILABLE = True
except ImportError:
    try:
        from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
        DIFF_GAUSS_AVAILABLE = True
    except ImportError:
        DIFF_GAUSS_AVAILABLE = False
        print("[WARNING] diff-gaussian-rasterization not available")


class GaussianRenderer:
    """
    Wrapper class for Gaussian rasterization.
    """
    
    def __init__(self, sh_degree: int = 3):
        self.sh_degree = sh_degree
        self.active_sh_degree = sh_degree
    
    def render(
        self,
        gaussians,
        viewpoint_camera,
        bg_color: torch.Tensor,
        scaling_modifier: float = 1.0,
        override_color: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Render Gaussians from a viewpoint.
        
        Args:
            gaussians: GaussianModel instance
            viewpoint_camera: Camera with projection matrices
            bg_color: [3] background color
            scaling_modifier: Scale factor for Gaussians
            override_color: Optional precomputed colors
        
        Returns:
            Dict with 'render', 'alpha', 'viewspace_points', etc.
        """
        if not DIFF_GAUSS_AVAILABLE:
            # Return placeholder
            H = getattr(viewpoint_camera, 'image_height', 512)
            W = getattr(viewpoint_camera, 'image_width', 512)
            device = gaussians.get_xyz.device if hasattr(gaussians, 'get_xyz') else 'cuda'
            return {
                'render': torch.zeros(3, H, W, device=device),
                'alpha': torch.zeros(1, H, W, device=device),
                'viewspace_points': torch.zeros(0, 3, device=device),
                'visibility_filter': torch.zeros(0, dtype=torch.bool, device=device),
                'radii': torch.zeros(0, device=device),
            }
        
        # Setup rasterization settings
        tanfovx = viewpoint_camera.tanfovx if hasattr(viewpoint_camera, 'tanfovx') else 0.5
        tanfovy = viewpoint_camera.tanfovy if hasattr(viewpoint_camera, 'tanfovy') else 0.5
        
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )
        
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        # Get Gaussian properties
        means3D = gaussians.get_xyz
        means2D = torch.zeros_like(means3D[:, :2], requires_grad=True, device=means3D.device)
        opacity = gaussians.get_opacity
        
        # Get scales and rotations
        scales = gaussians.get_scaling
        rotations = gaussians.get_rotation
        
        # Get SH features or precomputed colors
        if override_color is not None:
            colors_precomp = override_color
            shs = None
        else:
            shs = gaussians.get_features
            colors_precomp = None
        
        # Rasterize
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        
        # Compute alpha (visibility)
        # This is approximate - actual alpha would need depth sorting
        alpha = (rendered_image.sum(dim=0, keepdim=True) > 0).float()
        
        return {
            'render': rendered_image,
            'alpha': alpha,
            'viewspace_points': means2D,
            'visibility_filter': radii > 0,
            'radii': radii,
        }


def render_gaussians(
    viewpoint_camera,
    gaussians,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    override_color: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """
    Functional interface for Gaussian rendering.
    
    Args:
        viewpoint_camera: Camera object
        gaussians: GaussianModel instance
        pipe: Pipeline configuration
        bg_color: [3] background color
        scaling_modifier: Scale modifier
        override_color: Optional precomputed colors
    
    Returns:
        Rendering results dict
    """
    if not DIFF_GAUSS_AVAILABLE:
        H = getattr(viewpoint_camera, 'image_height', 512)
        W = getattr(viewpoint_camera, 'image_width', 512)
        device = gaussians.get_xyz.device if hasattr(gaussians, 'get_xyz') else 'cuda'
        return {
            'render': torch.zeros(3, H, W, device=device),
            'alpha': torch.zeros(1, H, W, device=device),
            'viewspace_points': torch.zeros(0, 3, device=device),
            'visibility_filter': torch.zeros(0, dtype=torch.bool, device=device),
            'radii': torch.zeros(0, device=device),
            'depth': torch.zeros(1, H, W, device=device),
        }
    
    # Get camera properties
    tanfovx = getattr(viewpoint_camera, 'tanfovx', 0.5)
    tanfovy = getattr(viewpoint_camera, 'tanfovy', 0.5)
    
    # Setup rasterization
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=gaussians.active_sh_degree if hasattr(gaussians, 'active_sh_degree') else 3,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=getattr(pipe, 'debug', False),
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    means3D = gaussians.get_xyz
    means2D = torch.zeros_like(means3D[:, :2], requires_grad=True, device=means3D.device)
    opacity = gaussians.get_opacity
    scales = gaussians.get_scaling
    rotations = gaussians.get_rotation
    
    if override_color is not None:
        colors_precomp = override_color
        shs = None
    else:
        shs = gaussians.get_features
        colors_precomp = None
    
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )
    
    alpha = (rendered_image.sum(dim=0, keepdim=True) > 0).float()
    
    return {
        'render': rendered_image,
        'alpha': alpha,
        'viewspace_points': means2D,
        'visibility_filter': radii > 0,
        'radii': radii,
        'depth': torch.zeros(1, viewpoint_camera.image_height, viewpoint_camera.image_width, device=means3D.device),
    }


