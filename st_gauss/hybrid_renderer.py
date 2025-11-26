"""
Hybrid Renderer for ST-Gauss Pipeline.

Combines:
- Face Branch: Deformable 3D Gaussian Splatting for skin, eyes, head shape
- Mouth Branch: Tri-Plane Hash NeRF for teeth, tongue, oral cavity

Fusion Strategy:
    final_image = face_rgb * face_alpha + mouth_rgb * (1 - face_alpha)

The mouth is rendered BEHIND the face - we see the mouth only where
the face is transparent (when lips are open).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer


class HybridRenderer(nn.Module):
    """
    Hybrid renderer combining 3D Gaussian Splatting (face) and NeRF (mouth).
    
    The rendering pipeline:
    1. Render face using deformable 3D Gaussian Splatting
    2. Render mouth using Tri-Plane Hash NeRF
    3. Composite mouth behind face using alpha blending
    """
    
    def __init__(
        self,
        background_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
        mouth_near: float = 0.01,
        mouth_far: float = 0.5,
        mouth_samples: int = 64,
        compute_cov3D_python: bool = False,
        convert_SHs_python: bool = False,
        debug: bool = False,
    ):
        super().__init__()
        
        self.background_color = background_color
        self.mouth_near = mouth_near
        self.mouth_far = mouth_far
        self.mouth_samples = mouth_samples
        self.compute_cov3D_python = compute_cov3D_python
        self.convert_SHs_python = convert_SHs_python
        self.debug = debug
        
        # Register background as buffer
        self.register_buffer(
            'background',
            torch.tensor(background_color, dtype=torch.float32)
        )
    
    def render_gaussian_face(
        self,
        viewpoint_camera,
        gaussian_model,
        motion_network,
        pipe,
        scaling_modifier: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Render face using Deformable 3D Gaussian Splatting.
        
        Args:
            viewpoint_camera: Camera viewpoint with transformation matrices
            gaussian_model: GaussianModel containing Gaussian primitives
            motion_network: BlendshapeMotionNetwork for deformation
            pipe: Pipeline configuration
            scaling_modifier: Scale modifier for Gaussians
        
        Returns:
            Dictionary containing:
            - render: [3, H, W] RGB image
            - alpha: [1, H, W] accumulated opacity
            - depth: [1, H, W] depth map
            - viewspace_points: [N, 3] 2D screen-space points
            - visibility_filter: [N] boolean visibility mask
            - radii: [N] screen-space radii
            - motion: Dictionary of motion predictions
        """
        # Create gradient-enabled screen-space points
        screenspace_points = torch.zeros_like(
            gaussian_model.get_xyz,
            dtype=gaussian_model.get_xyz.dtype,
            requires_grad=True,
            device="cuda"
        ) + 0
        
        try:
            screenspace_points.retain_grad()
        except:
            pass
        
        # Setup rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        
        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.background,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=gaussian_model.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug if hasattr(pipe, 'debug') else self.debug,
        )
        
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        
        # Get audio and blendshape features from camera
        audio_feat = viewpoint_camera.talking_dict["auds"].cuda()
        blendshapes = viewpoint_camera.talking_dict.get("blendshapes", None)
        
        # Handle legacy au_exp format
        if blendshapes is None:
            blendshapes = viewpoint_camera.talking_dict.get("au_exp", torch.zeros(52)).cuda()
        else:
            blendshapes = blendshapes.cuda()
        
        # Predict motion
        motion_preds = motion_network(
            gaussian_model.get_xyz,
            audio_feat,
            blendshapes,
        )
        
        # Apply deformations
        means3D = gaussian_model.get_xyz + motion_preds['d_xyz']
        means2D = screenspace_points
        opacity = gaussian_model.get_opacity
        
        # Apply scale and rotation deformations
        scales = gaussian_model.scaling_activation(
            gaussian_model._scaling + motion_preds['d_scale']
        )
        rotations = gaussian_model.rotation_activation(
            gaussian_model._rotation + motion_preds['d_rot']
        )
        
        # Get SH features
        shs = gaussian_model.get_features
        
        # Rasterize
        rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        
        return {
            "render": rendered_image,
            "alpha": rendered_alpha,
            "depth": rendered_depth,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "motion": motion_preds,
        }
    
    def render_nerf_mouth(
        self,
        viewpoint_camera,
        mouth_nerf,
        mouth_mask: torch.Tensor,
        audio_features: torch.Tensor,
        individual_code: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Render mouth using Tri-Plane Hash NeRF.
        
        For efficiency, we only render rays within the mouth region.
        
        Args:
            viewpoint_camera: Camera viewpoint
            mouth_nerf: MouthNeRFNetwork
            mouth_mask: [H, W] boolean mask indicating mouth region
            audio_features: Audio features for conditioning
            individual_code: Optional individual embedding
        
        Returns:
            Dictionary containing:
            - render: [3, H, W] RGB image (full resolution, zeros outside mask)
            - alpha: [1, H, W] opacity map
            - depth: [1, H, W] depth map
        """
        device = mouth_mask.device
        H, W = mouth_mask.shape
        
        # Initialize outputs
        mouth_rgb = torch.zeros(3, H, W, device=device)
        mouth_alpha = torch.zeros(1, H, W, device=device)
        mouth_depth = torch.zeros(1, H, W, device=device)
        
        # Get pixel coordinates within mouth mask
        mask_coords = torch.nonzero(mouth_mask, as_tuple=False)  # [M, 2] (y, x)
        
        if mask_coords.shape[0] == 0:
            return {
                "render": mouth_rgb,
                "alpha": mouth_alpha,
                "depth": mouth_depth,
            }
        
        # Convert to normalized device coordinates
        y_coords = mask_coords[:, 0].float()
        x_coords = mask_coords[:, 1].float()
        
        # Camera intrinsics
        fx = viewpoint_camera.fx if hasattr(viewpoint_camera, 'fx') else W / (2 * math.tan(viewpoint_camera.FoVx * 0.5))
        fy = viewpoint_camera.fy if hasattr(viewpoint_camera, 'fy') else H / (2 * math.tan(viewpoint_camera.FoVy * 0.5))
        cx = viewpoint_camera.cx if hasattr(viewpoint_camera, 'cx') else W / 2
        cy = viewpoint_camera.cy if hasattr(viewpoint_camera, 'cy') else H / 2
        
        # Generate ray directions in camera space
        dirs_x = (x_coords - cx) / fx
        dirs_y = (y_coords - cy) / fy
        dirs_z = torch.ones_like(dirs_x)
        
        rays_d_cam = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)  # [M, 3]
        rays_d_cam = F.normalize(rays_d_cam, dim=-1)
        
        # Transform to world space
        c2w = viewpoint_camera.world_view_transform.inverse()[:3, :3]  # [3, 3]
        rays_d = (rays_d_cam @ c2w.T)  # [M, 3]
        
        # Ray origins (camera position)
        rays_o = viewpoint_camera.camera_center.unsqueeze(0).expand(rays_d.shape[0], -1)  # [M, 3]
        
        # Render rays through NeRF
        render_results = mouth_nerf.render_rays(
            rays_o=rays_o,
            rays_d=rays_d,
            audio_features=audio_features,
            near=self.mouth_near,
            far=self.mouth_far,
            num_samples=self.mouth_samples,
            perturb=mouth_nerf.training,
            individual_code=individual_code,
        )
        
        # Fill in the output images
        mouth_rgb[:, mask_coords[:, 0], mask_coords[:, 1]] = render_results['rgb'].T
        mouth_alpha[0, mask_coords[:, 0], mask_coords[:, 1]] = render_results['alpha']
        mouth_depth[0, mask_coords[:, 0], mask_coords[:, 1]] = render_results['depth']
        
        return {
            "render": mouth_rgb,
            "alpha": mouth_alpha,
            "depth": mouth_depth,
            "ambient_aud": render_results.get('ambient_aud', None),
            "uncertainty": render_results.get('uncertainty', None),
        }
    
    def composite(
        self,
        face_rgb: torch.Tensor,
        face_alpha: torch.Tensor,
        mouth_rgb: torch.Tensor,
        background: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Composite mouth behind face using alpha blending.
        
        Formula: final = face_rgb * face_alpha + mouth_rgb * (1 - face_alpha)
        
        The mouth is visible only where the face (lips) are transparent.
        
        Args:
            face_rgb: [3, H, W] face rendering
            face_alpha: [1, H, W] face opacity
            mouth_rgb: [3, H, W] mouth rendering
            background: Optional [3, H, W] background image
        
        Returns:
            [3, H, W] composited image
        """
        if background is None:
            background = self.background[:, None, None].expand_as(face_rgb)
        
        # Composite mouth behind face
        # Where face_alpha is high (solid face), we see face
        # Where face_alpha is low (open mouth), we see mouth behind
        combined = face_rgb * face_alpha + mouth_rgb * (1.0 - face_alpha)
        
        return combined
    
    def forward(
        self,
        viewpoint_camera,
        gaussian_model,
        face_motion_network,
        mouth_nerf,
        pipe,
        mouth_mask: Optional[torch.Tensor] = None,
        individual_code: Optional[torch.Tensor] = None,
        render_mouth: bool = True,
        scaling_modifier: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Full hybrid rendering pipeline.
        
        Args:
            viewpoint_camera: Camera viewpoint
            gaussian_model: GaussianModel for face
            face_motion_network: BlendshapeMotionNetwork for face deformation
            mouth_nerf: MouthNeRFNetwork for mouth interior
            pipe: Pipeline configuration
            mouth_mask: [H, W] mask for mouth region (from preprocessing)
            individual_code: Optional individual embedding
            render_mouth: Whether to render mouth (False for face-only)
            scaling_modifier: Scale modifier for Gaussians
        
        Returns:
            Dictionary containing:
            - render: [3, H, W] final composited image
            - face_render: [3, H, W] face-only rendering
            - face_alpha: [1, H, W] face opacity
            - mouth_render: [3, H, W] mouth-only rendering (if render_mouth)
            - mouth_alpha: [1, H, W] mouth opacity (if render_mouth)
            - depth: [1, H, W] depth map
            - motion: face motion predictions
        """
        # Render face (3DGS)
        face_pkg = self.render_gaussian_face(
            viewpoint_camera,
            gaussian_model,
            face_motion_network,
            pipe,
            scaling_modifier,
        )
        
        face_rgb = face_pkg['render']
        face_alpha = face_pkg['alpha']
        
        results = {
            'face_render': face_rgb,
            'face_alpha': face_alpha,
            'face_depth': face_pkg['depth'],
            'viewspace_points': face_pkg['viewspace_points'],
            'visibility_filter': face_pkg['visibility_filter'],
            'radii': face_pkg['radii'],
            'motion': face_pkg['motion'],
        }
        
        # Render mouth (NeRF) if requested
        if render_mouth and mouth_nerf is not None:
            # Get mouth mask from camera data if not provided
            if mouth_mask is None:
                mouth_mask = torch.as_tensor(
                    viewpoint_camera.talking_dict.get("mouth_mask", 
                        torch.zeros(face_rgb.shape[1:], dtype=torch.bool))
                ).cuda()
            
            # Dilate mask slightly for better coverage
            if mouth_mask.sum() > 0:
                kernel = torch.ones(5, 5, device=mouth_mask.device)
                mouth_mask_dilated = F.conv2d(
                    mouth_mask.float().unsqueeze(0).unsqueeze(0),
                    kernel.unsqueeze(0).unsqueeze(0),
                    padding=2
                ).squeeze() > 0
            else:
                mouth_mask_dilated = mouth_mask
            
            audio_features = viewpoint_camera.talking_dict["auds"].cuda()
            
            mouth_pkg = self.render_nerf_mouth(
                viewpoint_camera,
                mouth_nerf,
                mouth_mask_dilated,
                audio_features,
                individual_code,
            )
            
            mouth_rgb = mouth_pkg['render']
            mouth_alpha = mouth_pkg['alpha']
            
            results['mouth_render'] = mouth_rgb
            results['mouth_alpha'] = mouth_alpha
            results['mouth_depth'] = mouth_pkg['depth']
            
            # Get background from camera or use default
            if hasattr(viewpoint_camera, 'background') and viewpoint_camera.background is not None:
                background = viewpoint_camera.background.cuda() / 255.0
            else:
                background = None
            
            # Composite
            final_image = self.composite(face_rgb, face_alpha, mouth_rgb, background)
        else:
            # Face only - composite with background
            if hasattr(viewpoint_camera, 'background') and viewpoint_camera.background is not None:
                background = viewpoint_camera.background.cuda() / 255.0
                final_image = face_rgb * face_alpha + background * (1.0 - face_alpha)
            else:
                final_image = face_rgb
        
        results['render'] = final_image
        
        return results


def render_hybrid(
    viewpoint_camera,
    gaussian_model,
    face_motion_network,
    mouth_nerf,
    pipe,
    background: torch.Tensor,
    mouth_mask: Optional[torch.Tensor] = None,
    scaling_modifier: float = 1.0,
    render_mouth: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Convenience function for hybrid rendering.
    
    This function provides a simple interface matching the original
    TalkingGaussian render functions.
    """
    renderer = HybridRenderer(
        background_color=tuple(background.tolist()),
        debug=pipe.debug if hasattr(pipe, 'debug') else False,
    )
    renderer.background = background
    
    return renderer(
        viewpoint_camera=viewpoint_camera,
        gaussian_model=gaussian_model,
        face_motion_network=face_motion_network,
        mouth_nerf=mouth_nerf,
        pipe=pipe,
        mouth_mask=mouth_mask,
        render_mouth=render_mouth,
        scaling_modifier=scaling_modifier,
    )


# ============= Lip Opacity Constraint Utility =============

def compute_lip_opacity_loss(
    gaussian_model,
    motion_predictions: dict,
    lip_landmarks: torch.Tensor,
    min_opacity: float = 0.8,
    sigma: float = 0.02,
) -> torch.Tensor:
    """
    Compute loss to enforce minimum opacity on lip Gaussians.
    
    This prevents the NeRF mouth from bleeding through incorrectly
    by ensuring Gaussians near lip landmarks maintain solid opacity.
    
    Args:
        gaussian_model: GaussianModel
        motion_predictions: Dictionary with 'd_xyz' deformations
        lip_landmarks: [L, 3] 3D positions of lip landmarks
        min_opacity: Minimum desired opacity for lip region
        sigma: Gaussian falloff for soft assignment
    
    Returns:
        Scalar loss value
    """
    # Get deformed Gaussian positions
    xyz = gaussian_model.get_xyz + motion_predictions['d_xyz']
    opacity = gaussian_model.get_opacity
    
    # Compute distances to lip landmarks
    # xyz: [N, 3], lip_landmarks: [L, 3]
    dists = torch.cdist(xyz, lip_landmarks)  # [N, L]
    min_dists = dists.min(dim=1).values  # [N]
    
    # Soft assignment: Gaussians close to lips should have high opacity
    lip_weights = torch.exp(-min_dists ** 2 / (2 * sigma ** 2))  # [N]
    
    # Loss: encourage opacity >= min_opacity for Gaussians near lips
    opacity_deficit = torch.relu(min_opacity - opacity.squeeze())
    loss = (lip_weights * opacity_deficit).mean()
    
    return loss


# ============= Lip Sync Loss Utility =============

def compute_lip_sync_loss(
    rendered_image: torch.Tensor,
    lip_landmarks_2d: torch.Tensor,
    gt_lip_landmarks_2d: torch.Tensor,
    image_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Compute lip synchronization loss based on 2D landmark positions.
    
    Args:
        rendered_image: [3, H, W] rendered image
        lip_landmarks_2d: [L, 2] predicted lip landmark positions (normalized)
        gt_lip_landmarks_2d: [L, 2] ground truth lip landmark positions
        image_size: (H, W) image dimensions
    
    Returns:
        Scalar loss value
    """
    # Simple L2 loss on landmark positions
    loss = F.mse_loss(lip_landmarks_2d, gt_lip_landmarks_2d)
    
    return loss

