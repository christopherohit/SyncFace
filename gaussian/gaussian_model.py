"""
Gaussian Model for Face Rendering
==================================
3D Gaussian representation with deformable properties.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
from plyfile import PlyData, PlyElement

# Import from TalkingGaussian
import sys
sys.path.insert(0, '../TalkingGaussian')

try:
    from TalkingGaussian.scene.gaussian_model import GaussianModel as _GaussianModel
    from TalkingGaussian.utils.sh_utils import RGB2SH, SH2RGB, eval_sh
except ImportError:
    _GaussianModel = None


class GaussianModel(nn.Module):
    """
    3D Gaussian Model for face representation.
    
    Each Gaussian has:
    - Position (xyz)
    - Covariance (scale + rotation)
    - Opacity (alpha)
    - Color (SH coefficients)
    
    Wraps TalkingGaussian's GaussianModel with additional features.
    """
    
    def __init__(self, sh_degree: int = 3):
        super().__init__()
        self.sh_degree = sh_degree
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0
        
        # Initialize empty parameters
        self._xyz = nn.Parameter(torch.empty(0, 3))
        self._features_dc = nn.Parameter(torch.empty(0, 1, 3))
        self._features_rest = nn.Parameter(torch.empty(0, (sh_degree + 1) ** 2 - 1, 3))
        self._scaling = nn.Parameter(torch.empty(0, 3))
        self._rotation = nn.Parameter(torch.empty(0, 4))
        self._opacity = nn.Parameter(torch.empty(0, 1))
        
        # Densification stats
        self.xyz_gradient_accum = None
        self.denom = None
        self.max_radii2D = None
        
        # Use TalkingGaussian implementation if available
        if _GaussianModel is not None:
            self._base_model = _GaussianModel(sh_degree)
        else:
            self._base_model = None
    
    @property
    def get_xyz(self) -> torch.Tensor:
        return self._xyz
    
    @property
    def get_scaling(self) -> torch.Tensor:
        return torch.exp(self._scaling)
    
    @property
    def get_rotation(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self._rotation, dim=-1)
    
    @property
    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self._opacity)
    
    @property
    def get_features(self) -> torch.Tensor:
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        """Compute 3D covariance matrix from scale and rotation."""
        from utils.graphics import build_covariance_from_scaling_rotation
        return build_covariance_from_scaling_rotation(
            self.get_scaling * scaling_modifier,
            self.get_rotation
        )
    
    def init_from_pcd(self, pcd, spatial_lr_scale: float = 1.0):
        """Initialize Gaussians from point cloud."""
        points = torch.tensor(pcd.points, dtype=torch.float32)
        colors = torch.tensor(pcd.colors, dtype=torch.float32)
        
        N = points.shape[0]
        
        # Convert colors to SH
        fused_color = RGB2SH(colors)
        features_dc = fused_color.unsqueeze(1)
        features_rest = torch.zeros((N, (self.sh_degree + 1) ** 2 - 1, 3))
        
        # Initialize scales based on point density
        dist2 = torch.clamp_min(
            torch.cdist(points, points).sort(dim=1).values[:, 1:4].mean(dim=1),
            1e-7
        )
        scales = torch.log(torch.sqrt(dist2)).unsqueeze(-1).repeat(1, 3)
        
        # Initialize rotations (identity)
        rots = torch.zeros((N, 4))
        rots[:, 0] = 1
        
        # Initialize opacities
        opacities = torch.logit(torch.ones((N, 1)) * 0.1)
        
        self._xyz = nn.Parameter(points)
        self._features_dc = nn.Parameter(features_dc)
        self._features_rest = nn.Parameter(features_rest)
        self._scaling = nn.Parameter(scales)
        self._rotation = nn.Parameter(rots)
        self._opacity = nn.Parameter(opacities)
        
        self.xyz_gradient_accum = torch.zeros((N, 1), device='cuda')
        self.denom = torch.zeros((N, 1), device='cuda')
        self.max_radii2D = torch.zeros((N,), device='cuda')
    
    def training_setup(self, opt):
        """Setup training optimizers."""
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device='cuda')
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0],), device='cuda')
        
        l = [
            {'params': [self._xyz], 'lr': opt.position_lr_init * opt.spatial_lr_scale, 'name': 'xyz'},
            {'params': [self._features_dc], 'lr': opt.feature_lr, 'name': 'f_dc'},
            {'params': [self._features_rest], 'lr': opt.feature_lr / 20.0, 'name': 'f_rest'},
            {'params': [self._opacity], 'lr': opt.opacity_lr, 'name': 'opacity'},
            {'params': [self._scaling], 'lr': opt.scaling_lr, 'name': 'scaling'},
            {'params': [self._rotation], 'lr': opt.rotation_lr, 'name': 'rotation'},
        ]
        
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
    
    def oneupSHdegree(self):
        """Increase active SH degree."""
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1
    
    def capture(self) -> Dict:
        """Capture model state for checkpointing."""
        return {
            'xyz': self._xyz.data,
            'features_dc': self._features_dc.data,
            'features_rest': self._features_rest.data,
            'scaling': self._scaling.data,
            'rotation': self._rotation.data,
            'opacity': self._opacity.data,
            'active_sh_degree': self.active_sh_degree,
        }
    
    def restore(self, model_params: Dict, opt=None):
        """Restore model state from checkpoint."""
        self._xyz = nn.Parameter(model_params['xyz'])
        self._features_dc = nn.Parameter(model_params['features_dc'])
        self._features_rest = nn.Parameter(model_params['features_rest'])
        self._scaling = nn.Parameter(model_params['scaling'])
        self._rotation = nn.Parameter(model_params['rotation'])
        self._opacity = nn.Parameter(model_params['opacity'])
        self.active_sh_degree = model_params.get('active_sh_degree', 0)
        
        if opt is not None:
            self.training_setup(opt)
    
    def prune_points(self, mask: torch.Tensor):
        """Remove Gaussians based on mask."""
        valid = ~mask
        
        self._xyz = nn.Parameter(self._xyz[valid])
        self._features_dc = nn.Parameter(self._features_dc[valid])
        self._features_rest = nn.Parameter(self._features_rest[valid])
        self._opacity = nn.Parameter(self._opacity[valid])
        self._scaling = nn.Parameter(self._scaling[valid])
        self._rotation = nn.Parameter(self._rotation[valid])
        
        if self.xyz_gradient_accum is not None:
            self.xyz_gradient_accum = self.xyz_gradient_accum[valid]
        if self.denom is not None:
            self.denom = self.denom[valid]
        if self.max_radii2D is not None:
            self.max_radii2D = self.max_radii2D[valid]
    
    def save_ply(self, path: str):
        """Save Gaussians to PLY file."""
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        
        dtype_full = [(attribute, 'f4') for attribute in 
                      ['x', 'y', 'z', 'nx', 'ny', 'nz']]
        
        for i in range(f_dc.shape[1]):
            dtype_full.append((f'f_dc_{i}', 'f4'))
        for i in range(f_rest.shape[1]):
            dtype_full.append((f'f_rest_{i}', 'f4'))
        
        dtype_full.append(('opacity', 'f4'))
        for i in range(scale.shape[1]):
            dtype_full.append((f'scale_{i}', 'f4'))
        for i in range(rotation.shape[1]):
            dtype_full.append((f'rot_{i}', 'f4'))
        
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        
        attributes = np.concatenate([xyz, normals, f_dc, f_rest, opacities, scale, rotation], axis=1)
        elements[:] = list(map(tuple, attributes))
        
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)



