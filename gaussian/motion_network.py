"""
Motion Network for Gaussian Deformation
========================================
Audio and blendshape conditioned deformation network.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

# Try to import grid encoder
try:
    from gridencoder import GridEncoder
    GRID_ENCODER_AVAILABLE = True
except ImportError:
    GRID_ENCODER_AVAILABLE = False
    print("[WARNING] GridEncoder not available, using fallback")


class AudioAttentionNet(nn.Module):
    """Attention-based audio feature aggregation."""
    
    def __init__(self, dim_aud: int = 64, seq_len: int = 8):
        super().__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        
        self.conv_net = nn.Sequential(
            nn.Conv1d(dim_aud, 16, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.02, True),
        )
        
        self.attention = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, seq_len, dim_aud]
        y = x.permute(0, 2, 1)  # [B, dim_aud, seq_len]
        y = self.conv_net(y)
        y = self.attention(y.view(-1, self.seq_len)).view(-1, self.seq_len, 1)
        return torch.sum(y * x, dim=1)  # [B, dim_aud]


class AudioEncoder(nn.Module):
    """Audio feature encoder supporting multiple input types."""
    
    def __init__(self, input_dim: int = 1024, output_dim: int = 64):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if input_dim >= 512:  # HuBERT or AVE
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.LeakyReLU(0.02, True),
                nn.Linear(256, 128),
                nn.LeakyReLU(0.02, True),
                nn.Linear(128, output_dim),
            )
        else:  # DeepSpeech
            self.encoder = nn.Sequential(
                nn.Conv1d(input_dim, 32, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.02, True),
                nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.02, True),
                nn.Flatten(),
                nn.Linear(64 * 4, output_dim),
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) == 3:  # [B, T, F]
            x = x.mean(dim=1)  # Simple averaging
        return self.encoder(x)


class BlendshapePrior(nn.Module):
    """
    Blendshape prior network for deformation guidance.
    
    ARKit blendshape indices:
    - Jaw: 0-4
    - Mouth: 17-37
    - Eyes: 5-16
    - Brows: 38-51
    """
    
    JAW_INDICES = list(range(0, 5))
    MOUTH_INDICES = list(range(17, 38))
    EYE_INDICES = list(range(5, 17))
    BROW_INDICES = list(range(38, 52))
    
    def __init__(
        self, 
        blendshape_dim: int = 52, 
        output_dim: int = 64,
        use_groups: str = 'all',
        prior_scale: float = 0.1
    ):
        super().__init__()
        
        self.prior_scale = prior_scale
        
        # Select indices based on group
        if use_groups == 'all':
            self.indices = list(range(52))
        elif use_groups == 'jaw':
            self.indices = self.JAW_INDICES
        elif use_groups == 'mouth':
            self.indices = self.MOUTH_INDICES
        elif use_groups == 'jaw_mouth':
            self.indices = self.JAW_INDICES + self.MOUTH_INDICES
        else:
            self.indices = list(range(52))
        
        self.mlp = nn.Sequential(
            nn.Linear(len(self.indices), 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, output_dim),
        )
    
    def forward(self, blendshape: torch.Tensor) -> torch.Tensor:
        if len(blendshape.shape) == 1:
            bs = blendshape[self.indices]
        else:
            bs = blendshape[:, self.indices]
        return self.mlp(bs) * self.prior_scale


class MLP(nn.Module):
    """Simple MLP block."""
    
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int, num_layers: int):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim, bias=False))
            elif i == num_layers - 1:
                layers.append(nn.Linear(hidden_dim, out_dim, bias=False))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MotionNetwork(nn.Module):
    """
    Basic motion network for Gaussian deformation.
    
    Takes audio features and outputs per-Gaussian deformations.
    """
    
    def __init__(
        self,
        audio_dim: int = 64,
        audio_in_dim: int = 29,
        hidden_dim: int = 64,
        num_layers: int = 3,
        bound: float = 0.15,
    ):
        super().__init__()
        
        self.bound = bound
        self.audio_dim = audio_dim
        
        # Audio encoder
        self.audio_encoder = AudioEncoder(audio_in_dim, audio_dim)
        self.audio_attention = AudioAttentionNet(audio_dim)
        
        # Tri-plane encoders
        self.num_levels = 12
        self.level_dim = 1
        
        if GRID_ENCODER_AVAILABLE:
            self.encoder_xy = GridEncoder(
                input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim,
                base_resolution=16, log2_hashmap_size=17, 
                desired_resolution=int(256 * bound)
            )
            self.encoder_yz = GridEncoder(
                input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim,
                base_resolution=16, log2_hashmap_size=17,
                desired_resolution=int(256 * bound)
            )
            self.encoder_xz = GridEncoder(
                input_dim=2, num_levels=self.num_levels, level_dim=self.level_dim,
                base_resolution=16, log2_hashmap_size=17,
                desired_resolution=int(256 * bound)
            )
            self.pos_dim = self.num_levels * self.level_dim * 3
        else:
            # Fallback to simple positional encoding
            self.pos_dim = 3 * 20
            self.encoder_xy = self.encoder_yz = self.encoder_xz = None
        
        # Attention networks
        self.audio_pos_attention = MLP(self.pos_dim, audio_dim, 32, 2)
        
        # Output network: xyz(3) + rot(4) + opacity(1) + scale(3) = 11
        self.out_dim = 11
        self.deform_net = MLP(
            self.pos_dim + audio_dim, self.out_dim, hidden_dim, num_layers
        )
    
    def encode_position(self, xyz: torch.Tensor) -> torch.Tensor:
        """Encode 3D position using tri-plane hash grid."""
        if self.encoder_xy is None:
            # Fallback: frequency encoding
            freqs = 2 ** torch.linspace(0, 9, 10, device=xyz.device)
            x_encoded = []
            for freq in freqs:
                x_encoded.append(torch.sin(xyz * freq * 3.14159))
                x_encoded.append(torch.cos(xyz * freq * 3.14159))
            return torch.cat(x_encoded, dim=-1)
        
        xy = xyz[:, :2]
        yz = xyz[:, 1:]
        xz = torch.stack([xyz[:, 0], xyz[:, 2]], dim=-1)
        
        feat_xy = self.encoder_xy(xy, bound=self.bound)
        feat_yz = self.encoder_yz(yz, bound=self.bound)
        feat_xz = self.encoder_xz(xz, bound=self.bound)
        
        return torch.cat([feat_xy, feat_yz, feat_xz], dim=-1)
    
    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio features."""
        if audio is None:
            return None
        enc = self.audio_encoder(audio)
        if len(enc.shape) == 2 and enc.shape[0] > 1:
            enc = self.audio_attention(enc.unsqueeze(0))
        return enc
    
    def forward(
        self, 
        xyz: torch.Tensor, 
        audio: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute deformations for Gaussians.
        
        Args:
            xyz: [N, 3] Gaussian positions
            audio: Audio features
        
        Returns:
            Dict with d_xyz, d_rot, d_opa, d_scale
        """
        N = xyz.shape[0]
        
        # Encode position
        enc_pos = self.encode_position(xyz)
        
        # Encode audio
        enc_audio = self.encode_audio(audio)
        if enc_audio is None:
            enc_audio = torch.zeros(1, self.audio_dim, device=xyz.device)
        
        # Audio-position attention
        audio_att = self.audio_pos_attention(enc_pos)
        enc_audio = enc_audio.expand(N, -1) * audio_att
        
        # Concatenate and decode
        h = torch.cat([enc_pos, enc_audio], dim=-1)
        out = self.deform_net(h)
        
        return {
            'd_xyz': out[:, :3] * 1e-2,
            'd_rot': out[:, 3:7],
            'd_opa': out[:, 7:8],
            'd_scale': out[:, 8:11],
        }
    
    def get_params(self, lr: float, lr_net: float, wd: float = 0):
        """Get optimizer parameter groups."""
        params = [
            {'params': self.audio_encoder.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.audio_attention.parameters(), 'lr': lr_net * 5, 'weight_decay': 1e-4},
            {'params': self.audio_pos_attention.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.deform_net.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ]
        
        if self.encoder_xy is not None:
            params.extend([
                {'params': self.encoder_xy.parameters(), 'lr': lr},
                {'params': self.encoder_yz.parameters(), 'lr': lr},
                {'params': self.encoder_xz.parameters(), 'lr': lr},
            ])
        
        return params


class BlendshapeMotionNetwork(MotionNetwork):
    """
    Enhanced motion network with 52-dim blendshape conditioning.
    
    Extends MotionNetwork with:
    - ARKit blendshape input
    - Blendshape prior injection
    - Separate attention for blendshape features
    """
    
    def __init__(
        self,
        audio_dim: int = 64,
        audio_in_dim: int = 1024,  # HuBERT default
        blendshape_dim: int = 52,
        hidden_dim: int = 64,
        num_layers: int = 3,
        bound: float = 0.15,
        blendshape_prior_scale: float = 0.1,
        blendshape_groups: str = 'all',
    ):
        super().__init__(audio_dim, audio_in_dim, hidden_dim, num_layers, bound)
        
        # Blendshape prior
        self.blendshape_prior = BlendshapePrior(
            blendshape_dim, audio_dim, blendshape_groups, blendshape_prior_scale
        )
        
        # Blendshape encoder
        self.blendshape_encoder = nn.Sequential(
            nn.Linear(blendshape_dim, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, 32),
        )
        self.bs_dim = 32
        
        # Blendshape attention
        self.bs_attention = MLP(self.pos_dim, self.bs_dim, 32, 2)
        
        # Updated output network
        self.deform_net = MLP(
            self.pos_dim + audio_dim + self.bs_dim,
            self.out_dim, hidden_dim, num_layers
        )
    
    def forward(
        self,
        xyz: torch.Tensor,
        audio: Optional[torch.Tensor] = None,
        blendshape: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute deformations with blendshape conditioning.
        
        Args:
            xyz: [N, 3] Gaussian positions
            audio: Audio features
            blendshape: [52] blendshape coefficients
        
        Returns:
            Dict with d_xyz, d_rot, d_opa, d_scale, attention magnitudes
        """
        N = xyz.shape[0]
        
        # Encode position
        enc_pos = self.encode_position(xyz)
        
        # Encode audio with prior
        enc_audio = self.encode_audio(audio)
        if enc_audio is None:
            enc_audio = torch.zeros(1, self.audio_dim, device=xyz.device)
        
        # Add blendshape prior
        if blendshape is not None:
            bs_prior = self.blendshape_prior(blendshape)
            if len(bs_prior.shape) == 1:
                bs_prior = bs_prior.unsqueeze(0)
            enc_audio = enc_audio + bs_prior
        
        # Audio-position attention
        audio_att = self.audio_pos_attention(enc_pos)
        enc_audio = enc_audio.expand(N, -1) * audio_att
        
        # Encode blendshape
        if blendshape is not None:
            enc_bs = self.blendshape_encoder(blendshape)
            if len(enc_bs.shape) == 1:
                enc_bs = enc_bs.unsqueeze(0)
            bs_att = self.bs_attention(enc_pos)
            enc_bs = enc_bs.expand(N, -1) * bs_att
        else:
            enc_bs = torch.zeros(N, self.bs_dim, device=xyz.device)
            bs_att = torch.zeros(N, self.bs_dim, device=xyz.device)
        
        # Concatenate and decode
        h = torch.cat([enc_pos, enc_audio, enc_bs], dim=-1)
        out = self.deform_net(h)
        
        return {
            'd_xyz': out[:, :3] * 1e-2,
            'd_rot': out[:, 3:7],
            'd_opa': out[:, 7:8],
            'd_scale': out[:, 8:11],
            'audio_attention': audio_att.norm(dim=-1, keepdim=True),
            'bs_attention': bs_att.norm(dim=-1, keepdim=True),
        }
    
    def get_params(self, lr: float, lr_net: float, wd: float = 0):
        """Get optimizer parameter groups."""
        params = super().get_params(lr, lr_net, wd)
        params.extend([
            {'params': self.blendshape_prior.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.blendshape_encoder.parameters(), 'lr': lr_net, 'weight_decay': wd},
            {'params': self.bs_attention.parameters(), 'lr': lr_net, 'weight_decay': wd},
        ])
        return params



