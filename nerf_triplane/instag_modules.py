"""
InsTaG-inspired modules for SyncTalk:
- Universal Motion Field (UMF): Shared deformation field learned during pre-training
- Personalized Motion Field: Identity-specific motion refinement
- Static Field: Identity-specific appearance and geometry (replaces integrated NeRF)
- Motion Aligner: Adapter network to align UMF to new identities
- Face-Mouth Hook: Mechanism to couple lip motion between face and mouth branches
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from .encoding import get_encoder


class MLP(nn.Module):
    """Multi-layer perceptron with ReLU activations."""
    
    def __init__(self, dim_in: int, dim_out: int, dim_hidden: int, num_layers: int, 
                 use_bias: bool = False):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            in_dim = self.dim_in if l == 0 else self.dim_hidden
            out_dim = self.dim_out if l == num_layers - 1 else self.dim_hidden
            net.append(nn.Linear(in_dim, out_dim, bias=use_bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class UniversalMotionField(nn.Module):
    """
    Universal Motion Field (UMF): Identity-agnostic deformation field.
    Learned during multi-person pre-training and frozen during adaptation.
    
    Input: 
        - x: spatial coordinates [N, 3]
        - f_l: audio/lip sync features [N, audio_dim] 
        - f_e: optional eye/expression features [N, eye_dim]
    Output:
        - delta_x: displacement vector [N, 3]
    """
    
    def __init__(self, 
                 audio_dim: int = 32,
                 eye_dim: int = 4,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 use_hash_encoding: bool = True):
        super().__init__()
        
        self.audio_dim = audio_dim
        self.eye_dim = eye_dim
        
        # Spatial encoding (use tri-plane hash encoding like SyncTalk)
        if use_hash_encoding:
            num_levels = 12
            level_dim = 1
            self.encoder_xy, in_dim_xy = get_encoder(
                'hashgrid', input_dim=2, num_levels=num_levels, level_dim=level_dim,
                base_resolution=32, log2_hashmap_size=14, desired_resolution=256
            )
            self.encoder_yz, in_dim_yz = get_encoder(
                'hashgrid', input_dim=2, num_levels=num_levels, level_dim=level_dim,
                base_resolution=32, log2_hashmap_size=14, desired_resolution=256
            )
            self.encoder_xz, in_dim_xz = get_encoder(
                'hashgrid', input_dim=2, num_levels=num_levels, level_dim=level_dim,
                base_resolution=32, log2_hashmap_size=14, desired_resolution=256
            )
            self.spatial_dim = in_dim_xy + in_dim_yz + in_dim_xz
        else:
            self.encoder_xy = self.encoder_yz = self.encoder_xz = None
            self.spatial_dim = 3
            
        # Motion prediction network
        input_dim = self.spatial_dim + audio_dim + eye_dim
        self.motion_net = MLP(input_dim, 3, hidden_dim, num_layers)
        
    @staticmethod
    def split_xyz(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split 3D coordinates into three 2D planes: xy, yz, xz."""
        xy = x[:, :-1]  # [N, 2] (x, y)
        yz = x[:, 1:]   # [N, 2] (y, z)
        xz = torch.cat([x[:, :1], x[:, -1:]], dim=-1)  # [N, 2] (x, z)
        return xy, yz, xz
    
    def encode_spatial(self, x: torch.Tensor, bound: float = 1.0) -> torch.Tensor:
        """Encode spatial coordinates using tri-plane hash grids."""
        if self.encoder_xy is None:
            return x
        
        xy, yz, xz = self.split_xyz(x)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)
    
    def forward(self, x: torch.Tensor, f_l: torch.Tensor, 
                f_e: Optional[torch.Tensor] = None, bound: float = 1.0) -> torch.Tensor:
        """
        Forward pass of UMF.
        
        Args:
            x: [N, 3] spatial coordinates in [-bound, bound]
            f_l: [N, audio_dim] audio/lip features (replicated to match N)
            f_e: [N, eye_dim] optional eye features
            bound: spatial bound for encoding
            
        Returns:
            delta_x: [N, 3] displacement vectors
        """
        # Encode spatial location
        enc_x = self.encode_spatial(x, bound=bound)  # [N, spatial_dim]
        
        # Concatenate all features
        if f_e is not None:
            h = torch.cat([enc_x, f_l, f_e], dim=-1)
        else:
            h = torch.cat([enc_x, f_l], dim=-1)
        
        # Predict displacement
        delta_x = self.motion_net(h)  # [N, 3]
        
        return delta_x


class PersonalizedMotionField(nn.Module):
    """
    Personalized Motion Field: Small identity-specific motion refinement.
    Much smaller capacity than UMF (fewer layers, smaller hidden dim).
    
    During pre-training: Each identity has its own PersonalizedField.
    During adaptation: A new PersonalizedField is trained for the target identity.
    """
    
    def __init__(self,
                 audio_dim: int = 32,
                 eye_dim: int = 4,
                 hidden_dim: int = 32,  # Smaller than UMF
                 num_layers: int = 2,   # Fewer layers than UMF
                 use_hash_encoding: bool = True):
        super().__init__()
        
        self.audio_dim = audio_dim
        self.eye_dim = eye_dim
        
        # Lighter spatial encoding
        if use_hash_encoding:
            num_levels = 8  # Fewer levels than UMF
            level_dim = 1
            self.encoder_xy, in_dim_xy = get_encoder(
                'hashgrid', input_dim=2, num_levels=num_levels, level_dim=level_dim,
                base_resolution=32, log2_hashmap_size=12, desired_resolution=128
            )
            self.encoder_yz, in_dim_yz = get_encoder(
                'hashgrid', input_dim=2, num_levels=num_levels, level_dim=level_dim,
                base_resolution=32, log2_hashmap_size=12, desired_resolution=128
            )
            self.encoder_xz, in_dim_xz = get_encoder(
                'hashgrid', input_dim=2, num_levels=num_levels, level_dim=level_dim,
                base_resolution=32, log2_hashmap_size=12, desired_resolution=128
            )
            self.spatial_dim = in_dim_xy + in_dim_yz + in_dim_xz
        else:
            self.encoder_xy = self.encoder_yz = self.encoder_xz = None
            self.spatial_dim = 3
        
        # Smaller motion prediction network
        input_dim = self.spatial_dim + audio_dim + eye_dim
        self.motion_net = MLP(input_dim, 3, hidden_dim, num_layers)
        
    @staticmethod
    def split_xyz(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xy = x[:, :-1]
        yz = x[:, 1:]
        xz = torch.cat([x[:, :1], x[:, -1:]], dim=-1)
        return xy, yz, xz
    
    def encode_spatial(self, x: torch.Tensor, bound: float = 1.0) -> torch.Tensor:
        if self.encoder_xy is None:
            return x
        
        xy, yz, xz = self.split_xyz(x)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)
    
    def forward(self, x: torch.Tensor, f_l: torch.Tensor,
                f_e: Optional[torch.Tensor] = None, bound: float = 1.0) -> torch.Tensor:
        """
        Forward pass of PersonalizedField.
        
        Args:
            x: [N, 3] spatial coordinates
            f_l: [N, audio_dim] audio features
            f_e: [N, eye_dim] optional eye features
            bound: spatial bound
            
        Returns:
            delta_x_personal: [N, 3] personalized displacement
        """
        enc_x = self.encode_spatial(x, bound=bound)
        
        if f_e is not None:
            h = torch.cat([enc_x, f_l, f_e], dim=-1)
        else:
            h = torch.cat([enc_x, f_l], dim=-1)
        
        delta_x_personal = self.motion_net(h)
        
        return delta_x_personal


class StaticField(nn.Module):
    """
    Static Field: Identity-specific appearance and geometry network.
    Replaces the integrated NeRF in original SyncTalk.
    Queries at deformed coordinates: x' = x + delta_x
    
    Output: color and density (sigma)
    """
    
    def __init__(self,
                 individual_dim: int = 4,
                 geo_feat_dim: int = 64,
                 hidden_dim: int = 64,
                 hidden_dim_color: int = 64,
                 num_layers: int = 3,
                 num_layers_color: int = 2,
                 bound: float = 1.0):
        super().__init__()
        
        self.individual_dim = individual_dim
        self.geo_feat_dim = geo_feat_dim
        self.bound = bound
        
        # Tri-plane hash encoding for deformed coordinates
        num_levels = 12
        level_dim = 1
        self.encoder_xy, self.in_dim_xy = get_encoder(
            'hashgrid', input_dim=2, num_levels=num_levels, level_dim=level_dim,
            base_resolution=64, log2_hashmap_size=14, desired_resolution=512 * bound
        )
        self.encoder_yz, self.in_dim_yz = get_encoder(
            'hashgrid', input_dim=2, num_levels=num_levels, level_dim=level_dim,
            base_resolution=64, log2_hashmap_size=14, desired_resolution=512 * bound
        )
        self.encoder_xz, self.in_dim_xz = get_encoder(
            'hashgrid', input_dim=2, num_levels=num_levels, level_dim=level_dim,
            base_resolution=64, log2_hashmap_size=14, desired_resolution=512 * bound
        )
        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz
        
        # Density network: outputs sigma + geometric features
        self.sigma_net = MLP(self.in_dim, 1 + geo_feat_dim, hidden_dim, num_layers, use_bias=False)
        
        # Direction encoder (spherical harmonics)
        self.encoder_dir, self.in_dim_dir = get_encoder('spherical_harmonics')
        
        # Color network: takes geometric features + view direction + optional individual code
        color_input_dim = self.in_dim_dir + geo_feat_dim + individual_dim
        self.color_net = MLP(color_input_dim, 3, hidden_dim_color, num_layers_color, use_bias=False)
    
    @staticmethod
    def split_xyz(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xy = x[:, :-1]
        yz = x[:, 1:]
        xz = torch.cat([x[:, :1], x[:, -1:]], dim=-1)
        return xy, yz, xz
    
    def encode_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """Encode deformed spatial coordinates."""
        xy, yz, xz = self.split_xyz(x)
        feat_xy = self.encoder_xy(xy, bound=self.bound)
        feat_yz = self.encoder_yz(yz, bound=self.bound)
        feat_xz = self.encoder_xz(xz, bound=self.bound)
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)
    
    def forward(self, x_deformed: torch.Tensor, d: torch.Tensor, 
                c: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of StaticField.
        
        Args:
            x_deformed: [N, 3] deformed coordinates (x + delta_x)
            d: [N, 3] view direction (normalized)
            c: [1, individual_dim] optional individual code
            
        Returns:
            sigma: [N,] density values
            color: [N, 3] RGB colors
        """
        # Encode deformed position
        enc_x = self.encode_spatial(x_deformed)  # [N, in_dim]
        
        # Predict density and geometric features
        h = self.sigma_net(enc_x)  # [N, 1 + geo_feat_dim]
        sigma = torch.exp(h[..., 0])  # [N,]
        geo_feat = h[..., 1:]  # [N, geo_feat_dim]
        
        # Encode view direction
        enc_d = self.encoder_dir(d)  # [N, in_dim_dir]
        
        # Predict color
        if self.individual_dim > 0:
            if c is not None:
                h_color = torch.cat([enc_d, geo_feat, c.repeat(x_deformed.shape[0], 1)], dim=-1)
            else:
                # If individual_dim > 0 but c is None, use zeros
                c_zero = torch.zeros(x_deformed.shape[0], self.individual_dim, device=x_deformed.device)
                h_color = torch.cat([enc_d, geo_feat, c_zero], dim=-1)
        else:
            h_color = torch.cat([enc_d, geo_feat], dim=-1)
        
        h_color = self.color_net(h_color)
        color = torch.sigmoid(h_color) * (1 + 2*0.001) - 0.001  # Slight extension beyond [0,1]
        
        return sigma, color


class MotionAligner(nn.Module):
    """
    Motion Aligner: Adapts frozen UMF to new identities.
    Outputs spatial offset (Delta_x_A) and temporal/magnitude scale (tau_A).
    
    During adaptation:
        1. Compute alignment: Delta_x_A, tau_A = MotionAligner(x)
        2. Query UMF at aligned position: delta_x_univ = UMF(x + Delta_x_A, f_l, f_e)
        3. Scale and combine: delta_x = tau_A * delta_x_univ + delta_x_personal
    """
    
    def __init__(self,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 scale_init: float = 1.0,
                 use_hash_encoding: bool = True):
        super().__init__()
        
        self.scale_init = scale_init
        
        # Spatial encoding
        if use_hash_encoding:
            num_levels = 8
            level_dim = 1
            self.encoder_xy, in_dim_xy = get_encoder(
                'hashgrid', input_dim=2, num_levels=num_levels, level_dim=level_dim,
                base_resolution=32, log2_hashmap_size=12, desired_resolution=128
            )
            self.encoder_yz, in_dim_yz = get_encoder(
                'hashgrid', input_dim=2, num_levels=num_levels, level_dim=level_dim,
                base_resolution=32, log2_hashmap_size=12, desired_resolution=128
            )
            self.encoder_xz, in_dim_xz = get_encoder(
                'hashgrid', input_dim=2, num_levels=num_levels, level_dim=level_dim,
                base_resolution=32, log2_hashmap_size=12, desired_resolution=128
            )
            self.spatial_dim = in_dim_xy + in_dim_yz + in_dim_xz
        else:
            self.encoder_xy = self.encoder_yz = self.encoder_xz = None
            self.spatial_dim = 3
        
        # Alignment offset network
        self.offset_net = MLP(self.spatial_dim, 3, hidden_dim, num_layers)
        
        # Scale network (can output per-dimension scales or single scalar)
        self.scale_net = MLP(self.spatial_dim, 3, hidden_dim, num_layers)
        
        # Initialize scale network to output ~1.0 initially
        with torch.no_grad():
            self.scale_net.net[-1].weight.data.fill_(0.0)
            if self.scale_net.net[-1].bias is not None:
                self.scale_net.net[-1].bias.data.fill_(0.0)
    
    @staticmethod
    def split_xyz(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        xy = x[:, :-1]
        yz = x[:, 1:]
        xz = torch.cat([x[:, :1], x[:, -1:]], dim=-1)
        return xy, yz, xz
    
    def encode_spatial(self, x: torch.Tensor, bound: float = 1.0) -> torch.Tensor:
        if self.encoder_xy is None:
            return x
        
        xy, yz, xz = self.split_xyz(x)
        feat_xy = self.encoder_xy(xy, bound=bound)
        feat_yz = self.encoder_yz(yz, bound=bound)
        feat_xz = self.encoder_xz(xz, bound=bound)
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)
    
    def forward(self, x: torch.Tensor, bound: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of MotionAligner.
        
        Args:
            x: [N, 3] original spatial coordinates
            bound: spatial bound
            
        Returns:
            Delta_x_A: [N, 3] spatial alignment offset
            tau_A: [N, 3] per-dimension scale factors (or [N, 1] for uniform scale)
        """
        enc_x = self.encode_spatial(x, bound=bound)
        
        # Spatial offset (small perturbation)
        Delta_x_A = torch.tanh(self.offset_net(enc_x)) * 0.1  # Limit to small offset
        
        # Scale (centered around scale_init)
        tau_A_raw = self.scale_net(enc_x)
        tau_A = torch.sigmoid(tau_A_raw) * 2.0  # Range [0, 2], centered at 1
        
        return Delta_x_A, tau_A


class FaceMouthHook(nn.Module):
    """
    Face-Mouth Hook (FM Hook): Couples lip region motion between face and mouth branches.
    
    Computes lip motion statistics from face branch:
        - Delta_mu_max: maximum displacement in lip region
        - Delta_mu_min: minimum displacement in lip region  
        - mu_dist: Delta_mu_max - Delta_mu_min (motion range)
    
    These features are concatenated and fed into the inside-mouth branch
    to ensure consistent animation.
    """
    
    def __init__(self, hook_dim: int = 9):
        super().__init__()
        self.hook_dim = hook_dim  # 3 + 3 + 3 = 9 for max, min, dist vectors
    
    def compute_hook_feature(self, delta_x_face: torch.Tensor, 
                            lip_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute hook feature from face motion field.
        
        Args:
            delta_x_face: [N, 3] displacement vectors from face branch
            lip_mask: [N,] boolean mask indicating lip region points
            
        Returns:
            phi: [1, hook_dim] hook feature vector
        """
        if lip_mask.sum() == 0:
            # No lip points, return zero feature
            return torch.zeros(1, self.hook_dim, device=delta_x_face.device)
        
        # Extract lip region displacements
        delta_lip = delta_x_face[lip_mask]  # [M, 3] where M = num lip points
        
        # Compute statistics
        Delta_mu_max = delta_lip.max(dim=0)[0]  # [3,]
        Delta_mu_min = delta_lip.min(dim=0)[0]  # [3,]
        mu_dist = Delta_mu_max - Delta_mu_min   # [3,]
        
        # Concatenate into hook feature
        phi = torch.cat([Delta_mu_max, Delta_mu_min, mu_dist], dim=0).unsqueeze(0)  # [1, 9]
        
        return phi
    
    def forward(self, delta_x_face: torch.Tensor, lip_mask: torch.Tensor) -> torch.Tensor:
        return self.compute_hook_feature(delta_x_face, lip_mask)


def create_lip_mask(points: torch.Tensor, lip_region_bounds: Optional[Dict] = None) -> torch.Tensor:
    """
    Create a boolean mask for lip region points.
    
    Args:
        points: [N, 3] 3D coordinates
        lip_region_bounds: Dictionary with 'y_min', 'y_max', 'z_min', 'z_max' defining lip region.
                          If None, uses heuristic based on face landmarks or central region.
    
    Returns:
        mask: [N,] boolean tensor
    """
    if lip_region_bounds is None:
        # Heuristic: lip region is roughly in lower-center of face
        # Assuming face is centered at origin, lips are roughly at:
        # y in [-0.15, -0.05] (below nose), z in [-0.1, 0.1] (centered)
        lip_region_bounds = {
            'y_min': -0.2,
            'y_max': -0.05,
            'z_min': -0.1,
            'z_max': 0.1
        }
    
    mask = (
        (points[:, 1] >= lip_region_bounds['y_min']) &
        (points[:, 1] <= lip_region_bounds['y_max']) &
        (points[:, 2] >= lip_region_bounds['z_min']) &
        (points[:, 2] <= lip_region_bounds['z_max'])
    )
    
    return mask

