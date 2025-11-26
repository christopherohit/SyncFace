"""
Mouth NeRF Network (Tri-Plane Hash Representation)
Ported from SyncTalk's nerf_triplane/network.py for the mouth interior rendering.

This module handles teeth, tongue, and oral cavity with view-consistent NeRF rendering.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nerf_triplane.encoding import get_encoder


class MLP(nn.Module):
    """Multi-Layer Perceptron with ReLU activations."""
    
    def __init__(self, dim_in: int, dim_out: int, dim_hidden: int, num_layers: int):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for layer_idx in range(num_layers):
            in_dim = dim_in if layer_idx == 0 else dim_hidden
            out_dim = dim_out if layer_idx == num_layers - 1 else dim_hidden
            net.append(nn.Linear(in_dim, out_dim, bias=False))
        
        self.net = nn.ModuleList(net)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer_idx in range(self.num_layers):
            x = self.net[layer_idx](x)
            if layer_idx != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x


class AudioNet(nn.Module):
    """Audio feature encoder network."""
    
    def __init__(self, dim_in: int = 29, dim_aud: int = 64, win_size: int = 16):
        super().__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(dim_in, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
        )
        
        self.encoder_fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_w = int(self.win_size / 2)
        x = x[:, :, 8 - half_w:8 + half_w]
        x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc(x)
        return x


class AudioNetAVE(nn.Module):
    """Audio feature encoder for AVE (Audio-Visual Encoder) features."""
    
    def __init__(self, dim_in: int = 512, dim_aud: int = 64):
        super().__init__()
        self.dim_aud = dim_aud
        
        self.encoder_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.02, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.02, True),
            nn.Linear(128, dim_aud),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder_fc(x).permute(1, 0, 2).squeeze(0)
        return x


class AudioAttNet(nn.Module):
    """Attention network for audio features."""
    
    def __init__(self, dim_aud: int = 64, seq_len: int = 8):
        super().__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        
        self.attention_conv = nn.Sequential(
            nn.Conv1d(dim_aud, 16, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        
        self.attention_fc = nn.Sequential(
            nn.Linear(self.seq_len, self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [1, seq_len, dim_aud]
        y = x.permute(0, 2, 1)  # [1, dim_aud, seq_len]
        y = self.attention_conv(y)
        y = self.attention_fc(y.view(1, self.seq_len)).view(1, self.seq_len, 1)
        return torch.sum(y * x, dim=1)  # [1, dim_aud]


class MouthNeRFNetwork(nn.Module):
    """
    Tri-Plane Hash NeRF Network for Mouth Interior Rendering.
    
    This network renders the interior mouth region (teeth, tongue, oral cavity)
    using a tri-plane hash encoding combined with audio-conditioned MLPs.
    
    Key Features:
    - Tri-plane hash grid encoding (XY, YZ, XZ planes)
    - Audio-conditioned density and color prediction
    - View-dependent color rendering
    - Supports multiple audio feature types (AVE, HuBERT, DeepSpeech)
    """
    
    def __init__(
        self,
        audio_type: str = "ave",
        audio_dim: int = 32,
        bound: float = 0.15,
        num_hash_levels: int = 12,
        hash_level_dim: int = 1,
        hash_base_resolution: int = 64,
        hash_log2_size: int = 14,
        hidden_dim: int = 64,
        num_layers: int = 3,
        geo_feat_dim: int = 64,
        individual_dim: int = 0,
        individual_num: int = 10000,
        use_attention: bool = True,
        attention_seq_len: int = 8,
    ):
        super().__init__()
        
        self.bound = bound
        self.audio_type = audio_type
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.geo_feat_dim = geo_feat_dim
        self.individual_dim = individual_dim
        self.use_attention = use_attention
        
        # Determine audio input dimension based on type
        if audio_type == "ave":
            self.audio_in_dim = 512
        elif audio_type == "hubert":
            self.audio_in_dim = 1024
        elif audio_type == "deepspeech":
            self.audio_in_dim = 29
        else:
            self.audio_in_dim = 32
        
        # Audio encoding network
        if audio_type == "ave":
            self.audio_net = AudioNetAVE(self.audio_in_dim, audio_dim)
        else:
            self.audio_net = AudioNet(self.audio_in_dim, audio_dim)
        
        # Audio attention (optional)
        if use_attention:
            self.audio_att_net = AudioAttNet(audio_dim, attention_seq_len)
        
        # Individual codes for per-subject learning
        if individual_dim > 0:
            self.individual_codes = nn.Parameter(
                torch.randn(individual_num, individual_dim) * 0.1
            )
        
        # Tri-plane hash encoders
        desired_resolution = int(512 * bound)
        
        self.encoder_xy, self.in_dim_xy = get_encoder(
            'hashgrid',
            input_dim=2,
            num_levels=num_hash_levels,
            level_dim=hash_level_dim,
            base_resolution=hash_base_resolution,
            log2_hashmap_size=hash_log2_size,
            desired_resolution=desired_resolution
        )
        
        self.encoder_yz, self.in_dim_yz = get_encoder(
            'hashgrid',
            input_dim=2,
            num_levels=num_hash_levels,
            level_dim=hash_level_dim,
            base_resolution=hash_base_resolution,
            log2_hashmap_size=hash_log2_size,
            desired_resolution=desired_resolution
        )
        
        self.encoder_xz, self.in_dim_xz = get_encoder(
            'hashgrid',
            input_dim=2,
            num_levels=num_hash_levels,
            level_dim=hash_level_dim,
            base_resolution=hash_base_resolution,
            log2_hashmap_size=hash_log2_size,
            desired_resolution=desired_resolution
        )
        
        self.in_dim = self.in_dim_xy + self.in_dim_yz + self.in_dim_xz
        
        # Sigma (density) network
        sigma_in_dim = self.in_dim + audio_dim
        self.sigma_net = MLP(sigma_in_dim, 1 + geo_feat_dim, hidden_dim, num_layers)
        
        # Direction encoder for view-dependent color
        self.encoder_dir, self.in_dim_dir = get_encoder('spherical_harmonics')
        
        # Color network
        color_in_dim = self.in_dim_dir + geo_feat_dim + individual_dim
        self.color_net = MLP(color_in_dim, 3, hidden_dim, 2)
        
        # Audio-channel attention for spatial audio influence
        self.aud_ch_att_net = MLP(self.in_dim, audio_dim, 64, 2)
        
        # Uncertainty prediction (for loss weighting)
        self.unc_net = MLP(self.in_dim, 1, 32, 2)
        
        self.testing = False
    
    @staticmethod
    @torch.jit.script
    def split_xyz(x: torch.Tensor):
        """Split 3D coordinates into 2D plane projections."""
        xy = x[:, :-1]
        yz = x[:, 1:]
        xz = torch.cat([x[:, :1], x[:, -1:]], dim=-1)
        return xy, yz, xz
    
    def encode_position(self, xyz: torch.Tensor) -> torch.Tensor:
        """Encode 3D position using tri-plane hash encoding."""
        xy, yz, xz = self.split_xyz(xyz)
        
        feat_xy = self.encoder_xy(xy, bound=self.bound)
        feat_yz = self.encoder_yz(yz, bound=self.bound)
        feat_xz = self.encoder_xz(xz, bound=self.bound)
        
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)
    
    def encode_audio(self, audio_features: torch.Tensor) -> torch.Tensor:
        """Encode audio features."""
        if audio_features is None:
            return None
        
        enc_a = self.audio_net(audio_features)
        
        if self.use_attention:
            enc_a = self.audio_att_net(enc_a.unsqueeze(0))
        
        return enc_a
    
    def density(
        self,
        x: torch.Tensor,
        enc_a: torch.Tensor,
        enc_x: torch.Tensor = None
    ) -> dict:
        """
        Compute density at given positions.
        
        Args:
            x: [N, 3] positions in [-bound, bound]
            enc_a: [1, audio_dim] encoded audio features
            enc_x: Optional precomputed position encoding
        
        Returns:
            Dictionary with 'sigma', 'geo_feat', 'ambient_aud'
        """
        if enc_x is None:
            enc_x = self.encode_position(x)
        
        # Audio-channel attention for spatial modulation
        enc_a_expanded = enc_a.repeat(enc_x.shape[0], 1)
        aud_ch_att = self.aud_ch_att_net(enc_x)
        enc_w = enc_a_expanded * aud_ch_att
        
        # Concatenate and predict density
        h = torch.cat([enc_x, enc_w], dim=-1)
        h = self.sigma_net(h)
        
        sigma = torch.exp(h[..., 0])
        geo_feat = h[..., 1:]
        
        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
            'ambient_aud': aud_ch_att.norm(dim=-1, keepdim=True),
        }
    
    def forward(
        self,
        x: torch.Tensor,
        d: torch.Tensor,
        enc_a: torch.Tensor,
        c: torch.Tensor = None
    ) -> tuple:
        """
        Forward pass: compute density and color.
        
        Args:
            x: [N, 3] positions in [-bound, bound]
            d: [N, 3] normalized view directions
            enc_a: [1, audio_dim] encoded audio features
            c: Optional [1, individual_dim] individual code
        
        Returns:
            Tuple of (sigma, color, ambient_aud, uncertainty)
        """
        enc_x = self.encode_position(x)
        
        # Get density and geometry features
        density_result = self.density(x, enc_a, enc_x)
        sigma = density_result['sigma']
        geo_feat = density_result['geo_feat']
        ambient_aud = density_result['ambient_aud']
        
        # Encode view direction
        enc_d = self.encoder_dir(d)
        
        # Compute view-dependent color
        if c is not None:
            h = torch.cat([enc_d, geo_feat, c.repeat(x.shape[0], 1)], dim=-1)
        else:
            h = torch.cat([enc_d, geo_feat], dim=-1)
        
        color = torch.sigmoid(self.color_net(h)) * 1.002 - 0.001
        
        # Uncertainty estimation
        if self.testing:
            uncertainty = torch.zeros_like(enc_x[:, :1])
        else:
            uncertainty = self.unc_net(enc_x.detach())
            uncertainty = torch.log(1 + torch.exp(uncertainty))
        
        return sigma, color, ambient_aud, uncertainty
    
    def render_rays(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        audio_features: torch.Tensor,
        near: float = 0.01,
        far: float = 1.0,
        num_samples: int = 64,
        perturb: bool = True,
        individual_code: torch.Tensor = None
    ) -> dict:
        """
        Volume rendering along rays.
        
        Args:
            rays_o: [N, 3] ray origins
            rays_d: [N, 3] ray directions
            audio_features: Audio features for conditioning
            near: Near plane distance
            far: Far plane distance
            num_samples: Number of samples per ray
            perturb: Whether to add noise to sample positions
            individual_code: Optional individual code
        
        Returns:
            Dictionary with 'rgb', 'depth', 'alpha', 'weights'
        """
        N = rays_o.shape[0]
        device = rays_o.device
        
        # Sample points along rays
        t_vals = torch.linspace(near, far, num_samples, device=device)
        
        if perturb and self.training:
            # Add noise for stratified sampling
            mids = 0.5 * (t_vals[:-1] + t_vals[1:])
            upper = torch.cat([mids, t_vals[-1:]])
            lower = torch.cat([t_vals[:1], mids])
            t_rand = torch.rand(N, num_samples, device=device)
            t_vals = lower + (upper - lower) * t_rand
        else:
            t_vals = t_vals.expand(N, num_samples)
        
        # Compute sample positions
        pts = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * t_vals.unsqueeze(-1)  # [N, S, 3]
        pts_flat = pts.reshape(-1, 3)
        dirs_flat = rays_d.unsqueeze(1).expand_as(pts).reshape(-1, 3)
        
        # Encode audio
        enc_a = self.encode_audio(audio_features)
        
        # Query network
        sigma, color, ambient_aud, uncertainty = self.forward(
            pts_flat, dirs_flat, enc_a, individual_code
        )
        
        # Reshape
        sigma = sigma.reshape(N, num_samples)
        color = color.reshape(N, num_samples, 3)
        
        # Volume rendering
        dists = t_vals[:, 1:] - t_vals[:, :-1]
        dists = torch.cat([dists, torch.full((N, 1), 1e10, device=device)], dim=-1)
        
        alpha = 1.0 - torch.exp(-sigma * dists)
        
        # Compute transmittance
        T = torch.cumprod(
            torch.cat([torch.ones((N, 1), device=device), 1.0 - alpha + 1e-10], dim=-1),
            dim=-1
        )[:, :-1]
        
        weights = alpha * T
        
        # Composite
        rgb = torch.sum(weights.unsqueeze(-1) * color, dim=1)
        depth = torch.sum(weights * t_vals, dim=1)
        acc = torch.sum(weights, dim=1)
        
        return {
            'rgb': rgb,
            'depth': depth,
            'alpha': acc,
            'weights': weights,
            'ambient_aud': ambient_aud.reshape(N, num_samples).mean(dim=1),
            'uncertainty': uncertainty.reshape(N, num_samples).mean(dim=1),
        }
    
    def get_params(self, lr: float, lr_net: float, weight_decay: float = 0):
        """Get optimizer parameter groups."""
        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': weight_decay},
            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': weight_decay},
            {'params': self.color_net.parameters(), 'lr': lr_net, 'weight_decay': weight_decay},
            {'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': weight_decay},
            {'params': self.unc_net.parameters(), 'lr': lr_net, 'weight_decay': weight_decay},
        ]
        
        if self.use_attention:
            params.append({
                'params': self.audio_att_net.parameters(),
                'lr': lr_net * 5,
                'weight_decay': 0.0001
            })
        
        if self.individual_dim > 0:
            params.append({
                'params': self.individual_codes,
                'lr': lr_net,
                'weight_decay': weight_decay
            })
        
        return params

