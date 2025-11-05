# SyncTalk + InsTaG Integration: DynamicGaussianNetwork
# This replaces the NeRFNetwork with a 3D Gaussian Splatting based network

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from typing import Dict, Optional, Tuple

# Import InsTaG's Gaussian components 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scene_gs.gaussian_model import GaussianModel
from scene_gs.motion_net import MotionNetwork

# Import SyncTalk's audio encoders from the original network
from .network import AudioNet_ave, AudioAttNet, AudioNet, Conv2d


class MLP(nn.Module):
    """Simple MLP network for feature transformation"""
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(
                self.dim_in if l == 0 else self.dim_hidden, 
                self.dim_out if l == num_layers - 1 else self.dim_hidden, 
                bias=True
            ))

        self.net = nn.ModuleList(net)

    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class DynamicGaussianNetwork(nn.Module):
    """
    Dynamic Gaussian Network that combines:
    - SyncTalk's audio and expression encoders for control signals
    - InsTaG's 3D Gaussian Splatting representation and deformation network
    """
    
    def __init__(self, opt):
        super(DynamicGaussianNetwork, self).__init__()
        self.opt = opt
        self.device = torch.device('cuda')
        
        # Store important dimensions
        self.audio_dim = opt.audio_dim if hasattr(opt, 'audio_dim') else 32
        self.eye_dim = opt.eye_dim if hasattr(opt, 'eye_dim') else 6
        
        # 1. Initialize the Canonical 3D Gaussians
        # Create a simple args object for GaussianModel
        class GaussianArgs:
            def __init__(self):
                self.sh_degree = opt.sh_degree if hasattr(opt, 'sh_degree') else 0
        
        gaussian_args = GaussianArgs()
        self.gaussians = GaussianModel(gaussian_args)
        
        # 2. Initialize Audio Encoders (from SyncTalk)
        # These provide our f_l (lip/audio) features
        if hasattr(opt, 'asr_model') and opt.asr_model == 'ave':
            # AVE mode uses different audio network
            self.audio_in_dim = 512  # AVE features from audio encoder
            self.audio_net = AudioNet_ave(self.audio_in_dim, self.audio_dim)
        else:
            # Standard mode with regular audio features
            self.audio_in_dim = 29  # Standard audio features
            self.audio_net = AudioNet_ave(self.audio_in_dim, self.audio_dim)
        
        # Audio attention network for temporal modeling
        # For AVE, we don't use attention since features are already processed
        if hasattr(opt, 'att') and opt.att > 0 and (not hasattr(opt, 'asr_model') or opt.asr_model != 'ave'):
            self.audio_att_net = AudioAttNet(self.audio_dim, seq_len=opt.att)
        else:
            self.audio_att_net = None
            
        # 3. Initialize Expression/Eye Encoder
        # This provides our f_e (expression) features
        if hasattr(opt, 'exp_eye') and opt.exp_eye:
            # Use the eye/expression features
            self.use_exp = True
            # Transform raw expression to encoded features
            self.exp_encode_net = MLP(self.eye_dim, self.eye_dim, 32, 2)
        else:
            self.use_exp = False
            self.exp_encode_net = None
            
        # 4. Initialize the Deformation Network
        # This predicts per-Gaussian deformations based on audio and expression
        self.setup_deformation_network()
        
        # 5. Load canonical Gaussians if provided
        if hasattr(opt, 'canonical_ply') and opt.canonical_ply and os.path.exists(opt.canonical_ply):
            print(f"[INFO] Loading canonical Gaussians from {opt.canonical_ply}")
            self.load_canonical_gaussians(opt.canonical_ply)
        else:
            print("[WARNING] No canonical Gaussians provided. Will need to initialize them.")
            
    def setup_deformation_network(self):
        """
        Set up the deformation network that predicts per-Gaussian deformations
        based on audio and expression conditions
        """
        # Calculate total conditioning dimension
        condition_dim = self.audio_dim
        if self.use_exp:
            condition_dim += self.eye_dim
            
        # Position encoder for spatial features
        from .encoding import get_encoder
        self.bound = 0.2  # Normalized coordinate bound
        
        # Multi-plane position encoding (xy, yz, xz planes)
        self.num_levels = 8
        self.level_dim = 2
        self.encoder_xy, self.in_dim_xy = get_encoder(
            'hashgrid', input_dim=2, num_levels=self.num_levels, 
            level_dim=self.level_dim, base_resolution=16, 
            log2_hashmap_size=15, desired_resolution=256 * self.bound
        )
        self.encoder_yz, self.in_dim_yz = get_encoder(
            'hashgrid', input_dim=2, num_levels=self.num_levels,
            level_dim=self.level_dim, base_resolution=16,
            log2_hashmap_size=15, desired_resolution=256 * self.bound
        )
        self.encoder_xz, self.in_dim_xz = get_encoder(
            'hashgrid', input_dim=2, num_levels=self.num_levels,
            level_dim=self.level_dim, base_resolution=16,
            log2_hashmap_size=15, desired_resolution=256 * self.bound
        )
        
        self.spatial_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz
        
        # Deformation prediction network
        # Outputs: rotation (4) + position (3) + scaling (3) + opacity (1) = 11
        self.deform_dim = 11
        self.deform_net = MLP(
            self.spatial_dim + condition_dim,
            self.deform_dim,
            dim_hidden=128,
            num_layers=4
        )
        
        # Attention networks for modulating features
        self.spatial_audio_att = MLP(self.spatial_dim, self.audio_dim, 32, 2)
        if self.use_exp:
            self.spatial_exp_att = MLP(self.spatial_dim, self.eye_dim, 32, 2)
    
    @staticmethod
    @torch.jit.script
    def split_xyz(x):
        """Split 3D coordinates into 2D plane projections"""
        xy = x[:, :2]
        yz = x[:, 1:3]
        xz = torch.cat([x[:, :1], x[:, 2:3]], dim=-1)
        return xy, yz, xz
    
    def encode_position(self, xyz):
        """
        Encode 3D positions using multi-plane hash grids
        Args:
            xyz: [N, 3] positions in [-bound, bound]
        Returns:
            [N, spatial_dim] encoded features
        """
        # Normalize to [-1, 1] then to encoding bound
        xyz = xyz / (xyz.max(dim=0, keepdim=True)[0] + 1e-6) * self.bound
        
        xy, yz, xz = self.split_xyz(xyz)
        feat_xy = self.encoder_xy(xy, bound=self.bound)
        feat_yz = self.encoder_yz(yz, bound=self.bound)
        feat_xz = self.encoder_xz(xz, bound=self.bound)
        
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)
    
    def encode_audio(self, auds):
        """
        Process audio features through SyncTalk's audio encoder
        Args:
            auds: Audio features from data provider
        Returns:
            [B, audio_dim] encoded audio features
        """
        if auds is None:
            return torch.zeros(1, self.audio_dim, device=self.device)
            
        # Process through audio network
        enc_a = self.audio_net(auds)  # [B, audio_dim] or [B, T, audio_dim]
        
        # Apply attention if available
        if self.audio_att_net is not None:
            if enc_a.dim() == 2:
                enc_a = enc_a.unsqueeze(0)  # Add batch dim if needed
            enc_a = self.audio_att_net(enc_a)  # [B, audio_dim]
        
        # Ensure batch dimension
        if enc_a.dim() == 1:
            enc_a = enc_a.unsqueeze(0)
            
        return enc_a
    
    def encode_expression(self, eye):
        """
        Process expression/eye features
        Args:
            eye: Expression features from data provider
        Returns:
            [B, eye_dim] encoded expression features
        """
        if not self.use_exp or eye is None:
            return torch.zeros(1, self.eye_dim, device=self.device)
            
        # Ensure batch dimension
        if eye.dim() == 1:
            eye = eye.unsqueeze(0)
            
        # Encode expression features
        enc_e = self.exp_encode_net(eye) if self.exp_encode_net else eye
        
        return enc_e
    
    def predict_deformation(self, xyz, audio_feat, exp_feat):
        """
        Predict per-Gaussian deformations based on position and conditions
        Args:
            xyz: [N, 3] Gaussian positions
            audio_feat: [B, audio_dim] audio features
            exp_feat: [B, eye_dim] expression features
        Returns:
            Dictionary with deformation parameters
        """
        N = xyz.shape[0]
        B = audio_feat.shape[0]
        
        # Encode spatial positions
        spatial_feat = self.encode_position(xyz)  # [N, spatial_dim]
        
        # Apply spatial attention to condition features
        audio_att = self.spatial_audio_att(spatial_feat)  # [N, audio_dim]
        audio_feat_expanded = audio_feat[0:1].repeat(N, 1)  # [N, audio_dim]
        audio_modulated = audio_feat_expanded * torch.sigmoid(audio_att)
        
        # Combine features
        if self.use_exp:
            exp_att = self.spatial_exp_att(spatial_feat)  # [N, eye_dim]
            exp_feat_expanded = exp_feat[0:1].repeat(N, 1)  # [N, eye_dim]
            exp_modulated = exp_feat_expanded * torch.sigmoid(exp_att)
            combined_feat = torch.cat([spatial_feat, audio_modulated, exp_modulated], dim=-1)
        else:
            combined_feat = torch.cat([spatial_feat, audio_modulated], dim=-1)
        
        # Predict deformations
        deform = self.deform_net(combined_feat)  # [N, 11]
        
        # Split deformation outputs
        d_rotation = deform[:, :4]  # Quaternion rotation delta
        d_xyz = deform[:, 4:7]  # Position delta
        d_scaling = deform[:, 7:10]  # Scaling delta
        d_opacity = deform[:, 10:11]  # Opacity delta
        
        return {
            'd_rotation': d_rotation,
            'd_xyz': d_xyz,
            'd_scaling': d_scaling,
            'd_opacity': d_opacity
        }
    
    def apply_deformation(self, base_params, deformations):
        """
        Apply predicted deformations to base Gaussian parameters
        Args:
            base_params: Dictionary with base Gaussian parameters
            deformations: Dictionary with predicted deformations
        Returns:
            Dictionary with deformed Gaussian parameters
        """
        # Apply deformations with appropriate activations
        deformed = {}
        
        # Position: simple addition
        deformed['xyz'] = base_params['xyz'] + deformations['d_xyz'] * 0.01  # Scale down for stability
        
        # Rotation: quaternion multiplication
        base_rot = base_params['rotation']
        d_rot = F.normalize(deformations['d_rotation'], dim=-1)
        # Simple approximation: add and renormalize
        deformed['rotation'] = F.normalize(base_rot + d_rot * 0.1, dim=-1)
        
        # Scaling: multiplicative
        deformed['scaling'] = base_params['scaling'] * (1 + deformations['d_scaling'] * 0.1)
        
        # Opacity: additive in logit space
        base_opacity_logit = torch.logit(base_params['opacity'].clamp(0.01, 0.99))
        deformed['opacity'] = torch.sigmoid(base_opacity_logit + deformations['d_opacity'] * 0.1)
        
        # Copy unchanged parameters
        deformed['features_dc'] = base_params['features_dc']
        deformed['features_rest'] = base_params['features_rest']
        
        return deformed
    
    def forward(self, auds, eye, poses=None):
        """
        Forward pass: deform Gaussians based on audio and expression
        Args:
            auds: Audio features
            eye: Expression features
            poses: Camera poses (not used for deformation, only for rendering)
        Returns:
            Dictionary with deformed Gaussian parameters ready for rendering
        """
        # Check if Gaussians are initialized
        if self.gaussians._xyz is None or self.gaussians._xyz.shape[0] == 0:
            # Return dummy parameters for uninitialized model
            device = self.device
            return {
                'xyz': torch.zeros(1, 3, device=device),
                'rotation': torch.tensor([[1, 0, 0, 0]], device=device),  # Identity quaternion
                'scaling': torch.ones(1, 3, device=device) * 0.01,
                'opacity': torch.ones(1, 1, device=device) * 0.5,
                'features_dc': torch.zeros(1, 1, 3, device=device),
                'features_rest': None
            }
            
        # Get base Gaussian parameters
        base_params = {
            'xyz': self.gaussians.get_xyz,
            'rotation': self.gaussians.get_rotation,
            'scaling': self.gaussians.get_scaling,
            'opacity': self.gaussians.get_opacity,
            'features_dc': self.gaussians.get_features[:, :1, :],  # DC component
            'features_rest': self.gaussians.get_features[:, 1:, :] if self.gaussians.get_features.shape[1] > 1 else None
        }
        
        # Encode control signals
        audio_feat = self.encode_audio(auds)  # [B, audio_dim]
        exp_feat = self.encode_expression(eye)  # [B, eye_dim]
        
        # Predict deformations
        deformations = self.predict_deformation(
            base_params['xyz'],
            audio_feat,
            exp_feat
        )
        
        # Apply deformations
        deformed_params = self.apply_deformation(base_params, deformations)
        
        return deformed_params
    
    def load_canonical_gaussians(self, ply_path):
        """Load pre-trained canonical Gaussians from PLY file"""
        self.gaussians.load_ply(ply_path)
        print(f"[INFO] Loaded {self.gaussians.get_xyz.shape[0]} Gaussians")
    
    def initialize_gaussians(self, num_gaussians=50000, scale=1.0):
        """Initialize Gaussians with random positions and default parameters"""
        import torch
        from scene_gs.gaussian_model import GaussianModel

        print(f"[INFO] Initializing {num_gaussians} Gaussians randomly")

        # Create random positions in a sphere around origin
        positions = torch.randn(num_gaussians, 3, device=self.device) * scale

        # Initialize other parameters with defaults
        self.gaussians._xyz = positions
        self.gaussians._features_dc = torch.zeros((num_gaussians, 1, 3), device=self.device)
        self.gaussians._features_rest = torch.zeros((num_gaussians, 15, 3), device=self.device)
        self.gaussians._scaling = torch.ones((num_gaussians, 3), device=self.device) * 0.01
        self.gaussians._rotation = torch.zeros((num_gaussians, 4), device=self.device)
        self.gaussians._rotation[:, 0] = 1.0  # Identity quaternion
        self.gaussians._opacity = torch.ones((num_gaussians, 1), device=self.device) * 0.1

        print(f"[INFO] Initialized {num_gaussians} Gaussians")

    def save_canonical_gaussians(self, ply_path):
        """Save current Gaussians as canonical model"""
        # Check if Gaussians are initialized
        if self.gaussians._xyz is None or self.gaussians._xyz.shape[0] == 0:
            print("[WARNING] Gaussians not initialized, cannot save")
            return

        self.gaussians.save_ply(ply_path)
        print(f"[INFO] Saved canonical Gaussians to {ply_path}")
    
    def get_params(self, lr_dict):
        """
        Get parameters for optimizer
        Args:
            lr_dict: Dictionary with learning rates for different components
        Returns:
            List of parameter groups for optimizer
        """
        params = []
        
        # Audio encoder parameters
        if 'audio_net' in lr_dict:
            params.append({
                'params': list(self.audio_net.parameters()) + 
                         (list(self.audio_att_net.parameters()) if self.audio_att_net else []),
                'lr': lr_dict['audio_net']
            })
        
        # Expression encoder parameters
        if self.use_exp and 'exp_net' in lr_dict:
            params.append({
                'params': self.exp_encode_net.parameters(),
                'lr': lr_dict['exp_net']
            })
        
        # Deformation network parameters
        if 'deform_net' in lr_dict:
            deform_params = list(self.deform_net.parameters())
            deform_params += list(self.spatial_audio_att.parameters())
            if self.use_exp:
                deform_params += list(self.spatial_exp_att.parameters())
            # Add hash grid parameters
            deform_params += list(self.encoder_xy.parameters())
            deform_params += list(self.encoder_yz.parameters())
            deform_params += list(self.encoder_xz.parameters())
            
            params.append({
                'params': deform_params,
                'lr': lr_dict['deform_net']
            })
        
        # Gaussian parameters (optional, for fine-tuning)
        if 'gaussians' in lr_dict and lr_dict['gaussians'] > 0:
            gaussian_params = []
            if self.gaussians._xyz.requires_grad:
                gaussian_params.append(self.gaussians._xyz)
            if self.gaussians._features_dc.requires_grad:
                gaussian_params.append(self.gaussians._features_dc)
            if self.gaussians._features_rest.requires_grad:
                gaussian_params.append(self.gaussians._features_rest)
            if self.gaussians._scaling.requires_grad:
                gaussian_params.append(self.gaussians._scaling)
            if self.gaussians._rotation.requires_grad:
                gaussian_params.append(self.gaussians._rotation)
            if self.gaussians._opacity.requires_grad:
                gaussian_params.append(self.gaussians._opacity)
                
            if gaussian_params:
                params.append({
                    'params': gaussian_params,
                    'lr': lr_dict['gaussians']
                })
        
        return params
    
    @property
    def num_gaussians(self):
        """Get number of Gaussians in the model"""
        return self.gaussians.get_xyz.shape[0] if self.gaussians.get_xyz is not None else 0
