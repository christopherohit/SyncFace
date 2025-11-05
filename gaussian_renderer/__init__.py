"""
3D Gaussian Splatting Renderer
Adapted from InsTaG for SyncTalk integration
"""

import torch
import math
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'submodules', 'diff-gaussian-rasterization'))

try:
    from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
except ImportError:
    print('[WARNING] diff_gauss not found, trying to import from build location...')
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'submodules', 'diff-gaussian-rasterization', 'diff_gauss'))
    from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer

from scene_gs.gaussian_model import GaussianModel


def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor = None, override_color = None, override_params = None):
    """
    Render the scene using 3D Gaussian Splatting
    
    Args:
        viewpoint_camera : Scene camera
        pc : GaussianModel or dict of gaussian parameters
        pipe : Pipeline/rasterization settings
        bg_color : Background color
        override_color : Optional color override
        override_params : Dictionary with dynamic gaussian parameters
        
    Returns:
        Dictionary with rendered image and other outputs
    """
    
    # Set background color
    if bg_color is None:
        bg_color = torch.zeros(3, device="cuda")
    
    # If override_params is provided, use those instead of pc's parameters
    if override_params is not None:
        means3D = override_params['xyz']
        opacity = override_params['opacity']
        scales = override_params['scaling']
        rotations = override_params['rotation']
        shs = override_params.get('features_dc', torch.zeros((means3D.shape[0], 1, 3), device=means3D.device))
        if override_params.get('features_rest', None) is not None:
            shs = torch.cat([shs, override_params['features_rest']], dim=1)
    else:
        # Get parameters from GaussianModel
        means3D = pc.get_xyz
        opacity = pc.get_opacity
        scales = pc.get_scaling
        rotations = pc.get_rotation
        shs = pc.get_features
    
    # Create rasterization settings
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5) if hasattr(viewpoint_camera, 'FoVx') else pipe.tanfovx
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5) if hasattr(viewpoint_camera, 'FoVy') else pipe.tanfovy
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height) if hasattr(viewpoint_camera, 'image_height') else pipe.image_height,
        image_width=int(viewpoint_camera.image_width) if hasattr(viewpoint_camera, 'image_width') else pipe.image_width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=1.0,
        viewmatrix=viewpoint_camera.world_view_transform if hasattr(viewpoint_camera, 'world_view_transform') else pipe.viewmatrix,
        projmatrix=viewpoint_camera.projection_matrix if hasattr(viewpoint_camera, 'projection_matrix') else pipe.projmatrix,
        sh_degree=pc.max_sh_degree if hasattr(pc, 'max_sh_degree') else 0,
        campos=viewpoint_camera.camera_center if hasattr(viewpoint_camera, 'camera_center') else pipe.campos,
        prefiltered=False,
        debug=False
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Handle screen space points (2D projection)
    means2D = torch.zeros_like(means3D, requires_grad=True, device="cuda")
    
    # Compute colors - use precomputed colors for simplicity
    shs = None
    if override_color is not None:
        colors_precomp = override_color
    else:
        # Use DC component as colors (simplified approach)
        colors_precomp = pc.get_features[:, 0, :]  # DC component

    # Ensure rotations are float type
    rotations = rotations.float()

    # Rasterize
    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3Ds_precomp=None
    )
    
    # Return render package
    return {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
    }


__all__ = ['render', 'GaussianRasterizationSettings', 'GaussianRasterizer', 'GaussianModel']