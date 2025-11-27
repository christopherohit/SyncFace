"""
SyncFace Hybrid Model
======================
Combined 3DGS face + NeRF mouth model.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from gaussian.gaussian_model import GaussianModel
from gaussian.motion_network import BlendshapeMotionNetwork
from nerf.network import NeRFNetwork
from nerf.renderer import NeRFRenderer


class SyncFaceModel(nn.Module):
    """
    Unified SyncFace model combining:
    - Deformable 3D Gaussian Splatting for face
    - Tri-plane Hash NeRF for mouth interior
    
    Features:
    - Audio and blendshape conditioning
    - Automatic mouth region handling
    - Seamless compositing
    """
    
    def __init__(
        self,
        # Gaussian params
        sh_degree: int = 3,
        # Motion network params
        audio_dim: int = 64,
        audio_in_dim: int = 1024,
        blendshape_dim: int = 52,
        # NeRF params
        nerf_bound: float = 0.3,
        nerf_samples: int = 64,
    ):
        super().__init__()
        
        # Gaussian face model
        self.gaussians = GaussianModel(sh_degree)
        
        # Motion network for Gaussian deformation
        self.motion_net = BlendshapeMotionNetwork(
            audio_dim=audio_dim,
            audio_in_dim=audio_in_dim,
            blendshape_dim=blendshape_dim,
        )
        
        # NeRF mouth model
        self.mouth_nerf = NeRFNetwork(
            bound=nerf_bound,
            audio_dim=audio_dim // 2,
            audio_in_dim=audio_in_dim,
            blendshape_dim=blendshape_dim,
        )
        
        # NeRF renderer
        self.mouth_renderer = NeRFRenderer(
            self.mouth_nerf,
            near=0.01,
            far=1.0,
            num_samples=nerf_samples,
        )
        
        # Compositing parameters
        self.blend_margin = 8
        self.mouth_weight = nn.Parameter(torch.ones(1))
    
    def get_deformed_gaussians(
        self,
        audio: Optional[torch.Tensor] = None,
        blendshape: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Get deformed Gaussian parameters.
        
        Returns positions, rotations, scales, opacities with applied deformations.
        """
        xyz = self.gaussians.get_xyz
        
        # Apply motion network
        motion = self.motion_net(xyz, audio, blendshape)
        
        deformed_xyz = xyz + motion['d_xyz']
        
        return {
            'xyz': deformed_xyz,
            'rotation': self.gaussians.get_rotation,
            'scaling': self.gaussians.get_scaling,
            'opacity': self.gaussians.get_opacity,
            'features': self.gaussians.get_features,
            'motion': motion,
        }
    
    def render_face(
        self,
        camera,
        pipe,
        bg_color: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        blendshape: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Render face using Gaussian splatting.
        
        This should be called with the appropriate Gaussian renderer.
        """
        # Get deformed Gaussians
        deformed = self.get_deformed_gaussians(audio, blendshape)
        
        # Temporarily set deformed positions
        original_xyz = self.gaussians._xyz.clone()
        self.gaussians._xyz = nn.Parameter(deformed['xyz'])
        
        # Import renderer
        try:
            from diff_gauss import rasterize_gaussians
            # Actual rendering would go here
            # For now, return placeholder
        except ImportError:
            pass
        
        # Restore original
        self.gaussians._xyz = nn.Parameter(original_xyz)
        
        return {
            'render': None,  # Would be actual render
            'alpha': None,
            'deformed': deformed,
        }
    
    def render_mouth(
        self,
        pose: torch.Tensor,
        lips_rect: Tuple[int, int, int, int],
        H: int,
        W: int,
        focal: float,
        audio: Optional[torch.Tensor] = None,
        blendshape: Optional[torch.Tensor] = None,
        bg_color: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Render mouth interior using NeRF.
        """
        return self.mouth_renderer.render_mouth_region(
            pose, lips_rect, H, W, focal,
            audio, blendshape, bg_color=bg_color
        )
    
    def composite(
        self,
        face_render: torch.Tensor,
        mouth_render: torch.Tensor,
        mouth_mask: torch.Tensor,
        lips_rect: Tuple[int, int, int, int],
    ) -> torch.Tensor:
        """
        Composite face and mouth renders.
        """
        return self.mouth_renderer.composite_with_face(
            face_render, 
            {'rgb': mouth_render, 'rect': lips_rect},
            mouth_mask,
            self.blend_margin
        )
    
    def forward(
        self,
        camera,
        pipe,
        bg_color: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        blendshape: Optional[torch.Tensor] = None,
        mouth_mask: Optional[torch.Tensor] = None,
        lips_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass with hybrid rendering.
        """
        # Render face
        face_result = self.render_face(camera, pipe, bg_color, audio, blendshape)
        
        # Render mouth if mask/rect provided
        if lips_rect is not None:
            pose = camera.get_pose()  # Camera-to-world
            mouth_result = self.render_mouth(
                pose, lips_rect, camera.height, camera.width, camera.focal,
                audio, blendshape, bg_color
            )
            
            # Composite
            if face_result['render'] is not None and mouth_mask is not None:
                final = self.composite(
                    face_result['render'].permute(1, 2, 0),
                    mouth_result['rgb'],
                    mouth_mask,
                    lips_rect
                )
                face_result['render'] = final.permute(2, 0, 1)
            
            face_result['mouth'] = mouth_result
        
        return face_result
    
    def get_params(self, stage: str = 'all'):
        """
        Get optimizer parameters for different training stages.
        
        Args:
            stage: 'face', 'mouth', 'fusion', or 'all'
        """
        params = []
        
        if stage in ['face', 'all']:
            params.extend(self.motion_net.get_params(5e-3, 5e-4))
        
        if stage in ['mouth', 'all']:
            params.extend(self.mouth_nerf.get_params(1e-3, 1e-4))
        
        if stage == 'fusion':
            # Only color parameters
            params.append({
                'params': [self.gaussians._features_dc, self.gaussians._features_rest],
                'lr': 1e-5
            })
            params.append({
                'params': self.mouth_nerf.color_net.parameters(),
                'lr': 1e-4
            })
            params.append({
                'params': [self.mouth_weight],
                'lr': 1e-4
            })
        
        return params
    
    def save(self, path: str):
        """Save complete model."""
        torch.save({
            'gaussians': self.gaussians.capture(),
            'motion_net': self.motion_net.state_dict(),
            'mouth_nerf': self.mouth_nerf.state_dict(),
            'blend_margin': self.blend_margin,
            'mouth_weight': self.mouth_weight.data,
        }, path)
    
    def load(self, path: str):
        """Load complete model."""
        ckpt = torch.load(path)
        self.gaussians.restore(ckpt['gaussians'])
        self.motion_net.load_state_dict(ckpt['motion_net'])
        self.mouth_nerf.load_state_dict(ckpt['mouth_nerf'])
        if 'blend_margin' in ckpt:
            self.blend_margin = ckpt['blend_margin']
        if 'mouth_weight' in ckpt:
            self.mouth_weight.data = ckpt['mouth_weight']



