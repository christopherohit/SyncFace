"""
NeRF Network for Mouth Rendering
=================================
Tri-plane hash-grid NeRF for mouth interior.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

# Try to import encoders
try:
    from gridencoder import GridEncoder
    GRID_ENCODER_AVAILABLE = True
except ImportError:
    GRID_ENCODER_AVAILABLE = False

try:
    from shencoder import SHEncoder
    SH_ENCODER_AVAILABLE = True
except ImportError:
    SH_ENCODER_AVAILABLE = False


class MLP(nn.Module):
    """Simple MLP block."""
    
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int, bias: bool = False):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, out_dim, bias=bias))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AudioEncoder(nn.Module):
    """Audio encoder for NeRF conditioning."""
    
    def __init__(self, input_dim: int = 1024, output_dim: int = 64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.02, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.02, True),
            nn.Linear(128, output_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:
            x = x.mean(dim=1)
        return self.encoder(x)


class NeRFNetwork(nn.Module):
    """
    Tri-plane Hash-grid NeRF for Mouth Rendering.
    
    Features:
    - Hash-grid encoding on XY, YZ, XZ planes
    - Audio and blendshape conditioning
    - Separate density and color networks
    """
    
    def __init__(
        self,
        bound: float = 0.3,
        audio_dim: int = 32,
        audio_in_dim: int = 1024,
        blendshape_dim: int = 52,
        geo_feat_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 3,
        individual_dim: int = 0,
        individual_num: int = 10000,
    ):
        super().__init__()
        
        self.bound = bound
        self.audio_dim = audio_dim
        self.geo_feat_dim = geo_feat_dim
        
        # Audio encoder
        self.audio_encoder = AudioEncoder(audio_in_dim, audio_dim)
        
        # Blendshape encoder
        self.blendshape_encoder = nn.Sequential(
            nn.Linear(blendshape_dim, 32),
            nn.LeakyReLU(0.02, True),
            nn.Linear(32, 16),
        )
        self.bs_dim = 16
        
        # Individual codes
        self.individual_dim = individual_dim
        if individual_dim > 0:
            self.individual_codes = nn.Parameter(
                torch.randn(individual_num, individual_dim) * 0.1
            )
        
        # Tri-plane encoders
        self.num_levels = 12
        self.level_dim = 1
        self.base_resolution = 64
        self.log2_hashmap_size = 14
        self.desired_resolution = int(512 * bound)
        
        if GRID_ENCODER_AVAILABLE:
            self.encoder_xy = GridEncoder(
                input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim,
                base_resolution=self.base_resolution,
                log2_hashmap_size=self.log2_hashmap_size,
                desired_resolution=self.desired_resolution
            )
            self.encoder_yz = GridEncoder(
                input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim,
                base_resolution=self.base_resolution,
                log2_hashmap_size=self.log2_hashmap_size,
                desired_resolution=self.desired_resolution
            )
            self.encoder_xz = GridEncoder(
                input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim,
                base_resolution=self.base_resolution,
                log2_hashmap_size=self.log2_hashmap_size,
                desired_resolution=self.desired_resolution
            )
            self.in_dim = self.num_levels * self.level_dim * 3
        else:
            # Fallback to frequency encoding
            self.encoder_xy = self.encoder_yz = self.encoder_xz = None
            self.in_dim = 3 * 20
        
        # Audio-position attention
        self.audio_pos_att = MLP(self.in_dim, audio_dim, 64, 2)
        
        # Density network
        sigma_in_dim = self.in_dim + audio_dim + self.bs_dim
        self.sigma_net = MLP(sigma_in_dim, 1 + geo_feat_dim, hidden_dim, num_layers)
        
        # Direction encoder
        if SH_ENCODER_AVAILABLE:
            self.dir_encoder = SHEncoder(input_dim=3, degree=4)
            self.dir_dim = 16
        else:
            self.dir_encoder = None
            self.dir_dim = 3
        
        # Color network
        color_in_dim = self.dir_dim + geo_feat_dim + individual_dim
        self.color_net = MLP(color_in_dim, 3, hidden_dim, 2)
        
        # Uncertainty network
        self.unc_net = MLP(self.in_dim, 1, 32, 2)
        
        self.training_mode = True
    
    def encode_position(self, x: torch.Tensor) -> torch.Tensor:
        """Encode 3D position using tri-plane hash grid."""
        if self.encoder_xy is None:
            # Frequency encoding fallback
            freqs = 2 ** torch.linspace(0, 9, 10, device=x.device)
            encoded = []
            for freq in freqs:
                encoded.append(torch.sin(x * freq * 3.14159))
                encoded.append(torch.cos(x * freq * 3.14159))
            return torch.cat(encoded, dim=-1)
        
        xy = x[..., :2]
        yz = x[..., 1:]
        xz = torch.stack([x[..., 0], x[..., 2]], dim=-1)
        
        feat_xy = self.encoder_xy(xy, bound=self.bound)
        feat_yz = self.encoder_yz(yz, bound=self.bound)
        feat_xz = self.encoder_xz(xz, bound=self.bound)
        
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)
    
    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio features."""
        if audio is None:
            return None
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)
        return self.audio_encoder(audio)
    
    def encode_blendshape(self, bs: torch.Tensor) -> torch.Tensor:
        """Encode blendshape coefficients."""
        if bs is None:
            return None
        return self.blendshape_encoder(bs)
    
    def density(
        self, 
        x: torch.Tensor, 
        enc_audio: torch.Tensor, 
        enc_bs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute density and geometry features."""
        enc_x = self.encode_position(x)
        N = x.shape[0]
        
        # Audio-position attention
        audio_att = self.audio_pos_att(enc_x)
        
        if enc_audio is not None:
            enc_audio = enc_audio.expand(N, -1) * audio_att
        else:
            enc_audio = torch.zeros(N, self.audio_dim, device=x.device)
        
        if enc_bs is not None:
            enc_bs = enc_bs.expand(N, -1)
        else:
            enc_bs = torch.zeros(N, self.bs_dim, device=x.device)
        
        # Concatenate and compute density
        h = torch.cat([enc_x, enc_audio, enc_bs], dim=-1)
        h = self.sigma_net(h)
        
        sigma = torch.exp(h[..., 0])
        geo_feat = h[..., 1:]
        
        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
            'enc_x': enc_x,
        }
    
    def color(
        self, 
        d: torch.Tensor, 
        geo_feat: torch.Tensor, 
        ind_code: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute RGB color."""
        # Encode direction
        if self.dir_encoder is not None:
            enc_d = self.dir_encoder(d)
        else:
            enc_d = d
        
        # Concatenate
        if ind_code is not None:
            N = d.shape[0]
            ind_code = ind_code.expand(N, -1)
            h = torch.cat([enc_d, geo_feat, ind_code], dim=-1)
        else:
            h = torch.cat([enc_d, geo_feat], dim=-1)
        
        return torch.sigmoid(self.color_net(h))
    
    def forward(
        self,
        x: torch.Tensor,
        d: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        blendshape: Optional[torch.Tensor] = None,
        ind_code: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass.
        
        Args:
            x: [N, 3] positions
            d: [N, 3] view directions
            audio: Audio features
            blendshape: [52] blendshape coefficients
            ind_code: Individual code index
        
        Returns:
            sigma: [N] densities
            color: [N, 3] RGB
            uncertainty: [N, 1]
        """
        enc_audio = self.encode_audio(audio)
        enc_bs = self.encode_blendshape(blendshape)
        
        # Get individual code
        if ind_code is not None and self.individual_dim > 0:
            if isinstance(ind_code, int):
                ind = self.individual_codes[ind_code:ind_code+1]
            else:
                ind = self.individual_codes[ind_code.item():ind_code.item()+1]
        else:
            ind = None
        
        # Density
        density_out = self.density(x, enc_audio, enc_bs)
        sigma = density_out['sigma']
        geo_feat = density_out['geo_feat']
        
        # Color
        color = self.color(d, geo_feat, ind)
        
        # Uncertainty
        if self.training_mode:
            unc = torch.log(1 + torch.exp(self.unc_net(density_out['enc_x'])))
        else:
            unc = torch.zeros_like(sigma.unsqueeze(-1))
        
        return sigma, color, unc
    
    def get_params(self, lr: float = 1e-3, lr_net: float = 1e-4, wd: float = 0):
        """Get optimizer parameter groups."""
        params = [
            {'params': self.audio_encoder.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.blendshape_encoder.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.audio_pos_att.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.sigma_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.color_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.unc_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        
        if self.encoder_xy is not None:
            params.extend([
                {'params': self.encoder_xy.parameters(), 'lr': lr},
                {'params': self.encoder_yz.parameters(), 'lr': lr},
                {'params': self.encoder_xz.parameters(), 'lr': lr},
            ])
        
        if self.individual_dim > 0:
            params.append({'params': self.individual_codes, 'lr': lr_net, 'weight_decay': wd})
        
        return params



