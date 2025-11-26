"""
Face Motion Network with Blendshape Conditioning and Facial-Aware Masked Attention.

This module implements the Face Branch for ST-Gauss:
- Deformable 3D Gaussian motion field
- 52 ARKit blendshape conditioning (replacing simple Action Units)
- SyncTalk's Facial-Aware Masked-Attention for audio/expression disentanglement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from TalkingGaussian.encoding import get_encoder


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
    """Audio encoder for AVE (Audio-Visual Encoder) features."""
    
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
    """Audio attention network."""
    
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
        y = x.permute(0, 2, 1)
        y = self.attention_conv(y)
        y = self.attention_fc(y.view(1, self.seq_len)).view(1, self.seq_len, 1)
        return torch.sum(y * x, dim=1)


class FacialAwareMaskedAttention(nn.Module):
    """
    Facial-Aware Masked-Attention Module (from SyncTalk).
    
    This module explicitly disentangles:
    - Audio features → primarily affects lips and jaw
    - Expression features → affects eyes, eyebrows, forehead
    
    This prevents "crosstalk" where loud audio accidentally causes 
    eyebrow twitches or unnatural forehead movements.
    """
    
    def __init__(
        self,
        position_dim: int,
        audio_dim: int,
        expression_dim: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        
        self.position_dim = position_dim
        self.audio_dim = audio_dim
        self.expression_dim = expression_dim
        
        # Audio attention mask predictor
        # Learns which spatial regions should be affected by audio
        self.audio_mask_net = MLP(position_dim, audio_dim, hidden_dim, 2)
        
        # Expression attention mask predictor
        # Learns which spatial regions should be affected by expressions
        self.expression_mask_net = MLP(position_dim, expression_dim, hidden_dim, 2)
        
        # Region classification (optional, for supervision)
        # Predicts if a point belongs to: 0=neutral, 1=lip, 2=eye, 3=brow
        self.region_classifier = MLP(position_dim, 4, 32, 2)
        
    def forward(
        self,
        position_encoding: torch.Tensor,
        audio_features: torch.Tensor,
        expression_features: torch.Tensor,
    ) -> dict:
        """
        Apply facial-aware masked attention.
        
        Args:
            position_encoding: [N, pos_dim] encoded 3D positions
            audio_features: [1, audio_dim] encoded audio
            expression_features: [1, expr_dim] encoded expression/blendshape
        
        Returns:
            Dictionary containing:
            - 'audio_weighted': Audio features weighted by spatial attention
            - 'expression_weighted': Expression features weighted by attention
            - 'audio_mask': Attention weights for audio
            - 'expression_mask': Attention weights for expression
            - 'region_logits': Region classification logits
        """
        N = position_encoding.shape[0]
        
        # Compute attention masks based on spatial position
        audio_mask = torch.sigmoid(self.audio_mask_net(position_encoding))  # [N, audio_dim]
        expression_mask = torch.sigmoid(self.expression_mask_net(position_encoding))  # [N, expr_dim]
        
        # Apply masks to features
        audio_expanded = audio_features.repeat(N, 1)  # [N, audio_dim]
        expression_expanded = expression_features.repeat(N, 1)  # [N, expr_dim]
        
        audio_weighted = audio_expanded * audio_mask
        expression_weighted = expression_expanded * expression_mask
        
        # Region classification for supervision (optional)
        region_logits = self.region_classifier(position_encoding)
        
        return {
            'audio_weighted': audio_weighted,
            'expression_weighted': expression_weighted,
            'audio_mask': audio_mask,
            'expression_mask': expression_mask,
            'region_logits': region_logits,
        }


class BlendshapeEncoder(nn.Module):
    """
    Encoder for 52 ARKit blendshape parameters.
    
    Blendshapes provide granular control over facial expressions including:
    - Eye: eyeBlink_L/R, eyeWide_L/R, eyeSquint_L/R, etc.
    - Brow: browDown_L/R, browInner_U, browOuterUp_L/R
    - Jaw: jawOpen, jawForward, jawLeft, jawRight
    - Mouth: mouthSmile_L/R, mouthFunnel, mouthPucker, etc.
    - Nose: noseSneer_L/R
    - Cheek: cheekPuff, cheekSquint_L/R
    """
    
    # ARKit blendshape indices for different facial regions
    EYE_INDICES = list(range(0, 14))  # Eye-related blendshapes
    BROW_INDICES = list(range(14, 19))  # Brow blendshapes
    JAW_INDICES = list(range(19, 23))  # Jaw blendshapes
    MOUTH_INDICES = list(range(23, 44))  # Mouth blendshapes
    OTHER_INDICES = list(range(44, 52))  # Nose, cheek, tongue
    
    def __init__(
        self,
        blendshape_dim: int = 52,
        output_dim: int = 32,
        hidden_dim: int = 64,
    ):
        super().__init__()
        
        self.blendshape_dim = blendshape_dim
        self.output_dim = output_dim
        
        # Separate encoders for different facial regions
        self.eye_encoder = nn.Sequential(
            nn.Linear(14, 32),
            nn.LeakyReLU(0.02, True),
            nn.Linear(32, output_dim // 4),
        )
        
        self.brow_encoder = nn.Sequential(
            nn.Linear(5, 16),
            nn.LeakyReLU(0.02, True),
            nn.Linear(16, output_dim // 8),
        )
        
        self.jaw_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.LeakyReLU(0.02, True),
            nn.Linear(16, output_dim // 8),
        )
        
        self.mouth_encoder = nn.Sequential(
            nn.Linear(21, 48),
            nn.LeakyReLU(0.02, True),
            nn.Linear(48, output_dim // 2),
        )
        
        self.other_encoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.LeakyReLU(0.02, True),
            nn.Linear(16, output_dim // 8),
        )
        
        # Combine all region encodings
        combined_dim = (output_dim // 4) + (output_dim // 8) * 3 + (output_dim // 2)
        self.combine = nn.Linear(combined_dim, output_dim)
    
    def forward(self, blendshapes: torch.Tensor) -> dict:
        """
        Encode blendshape parameters.
        
        Args:
            blendshapes: [B, 52] blendshape coefficients
        
        Returns:
            Dictionary with encoded features and regional encodings
        """
        # Split by region
        eye_bs = blendshapes[:, :14]
        brow_bs = blendshapes[:, 14:19]
        jaw_bs = blendshapes[:, 19:23]
        mouth_bs = blendshapes[:, 23:44]
        other_bs = blendshapes[:, 44:52]
        
        # Encode each region
        eye_feat = self.eye_encoder(eye_bs)
        brow_feat = self.brow_encoder(brow_bs)
        jaw_feat = self.jaw_encoder(jaw_bs)
        mouth_feat = self.mouth_encoder(mouth_bs)
        other_feat = self.other_encoder(other_bs)
        
        # Combine
        combined = torch.cat([eye_feat, brow_feat, jaw_feat, mouth_feat, other_feat], dim=-1)
        encoded = self.combine(combined)
        
        return {
            'encoded': encoded,
            'eye': eye_feat,
            'brow': brow_feat,
            'jaw': jaw_feat,
            'mouth': mouth_feat,
            'other': other_feat,
        }


class BlendshapeMotionNetwork(nn.Module):
    """
    Face Motion Network with Blendshape Conditioning.
    
    This is the Face Branch of ST-Gauss, which predicts per-Gaussian deformations:
    - Position offset (d_xyz)
    - Rotation offset (d_rot)
    - Scale offset (d_scale)
    - Opacity offset (d_opa)
    
    Key improvements over TalkingGaussian:
    1. Uses 52 ARKit blendshapes instead of simple Action Units
    2. Uses SyncTalk's AVE audio features instead of DeepSpeech
    3. Implements Facial-Aware Masked-Attention for disentanglement
    """
    
    def __init__(
        self,
        audio_type: str = "ave",
        audio_dim: int = 32,
        blendshape_dim: int = 52,
        blendshape_encoded_dim: int = 32,
        bound: float = 0.15,
        num_hash_levels: int = 12,
        hash_level_dim: int = 1,
        hash_base_resolution: int = 16,
        hash_log2_size: int = 17,
        hidden_dim: int = 64,
        num_layers: int = 3,
        individual_dim: int = 0,
        use_facial_attention: bool = True,
        args=None,  # For compatibility with original TalkingGaussian
    ):
        super().__init__()
        
        self.bound = bound
        self.audio_dim = audio_dim
        self.blendshape_dim = blendshape_dim
        self.hidden_dim = hidden_dim
        self.individual_dim = individual_dim
        self.use_facial_attention = use_facial_attention
        
        # Handle args for backward compatibility
        if args is not None:
            audio_type = getattr(args, 'audio_extractor', 'ave')
            if 'esperanto' in audio_type:
                self.audio_in_dim = 44
            elif 'deepspeech' in audio_type:
                self.audio_in_dim = 29
            elif 'hubert' in audio_type:
                self.audio_in_dim = 1024
            elif 'ave' in audio_type:
                self.audio_in_dim = 512
            else:
                self.audio_in_dim = 32
        else:
            if audio_type == "ave":
                self.audio_in_dim = 512
            elif audio_type == "hubert":
                self.audio_in_dim = 1024
            elif audio_type == "deepspeech":
                self.audio_in_dim = 29
            else:
                self.audio_in_dim = 32
        
        # Audio network
        if audio_type == "ave" or (args and 'ave' in getattr(args, 'audio_extractor', '')):
            self.audio_net = AudioNetAVE(512, audio_dim)
        else:
            self.audio_net = AudioNet(self.audio_in_dim, audio_dim)
        
        self.audio_att_net = AudioAttNet(audio_dim)
        
        # Blendshape encoder
        self.blendshape_encoder = BlendshapeEncoder(
            blendshape_dim=blendshape_dim,
            output_dim=blendshape_encoded_dim,
            hidden_dim=hidden_dim,
        )
        
        # Individual codes
        if individual_dim > 0:
            self.individual_codes = nn.Parameter(
                torch.randn(10000, individual_dim) * 0.1
            )
        
        # Tri-plane hash encoders for position
        desired_resolution = int(256 * bound)
        
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
        
        # Facial-Aware Masked Attention
        if use_facial_attention:
            self.facial_attention = FacialAwareMaskedAttention(
                position_dim=self.in_dim,
                audio_dim=audio_dim,
                expression_dim=blendshape_encoded_dim,
                hidden_dim=hidden_dim,
            )
        
        # Audio-channel attention (spatial audio influence)
        self.aud_ch_att_net = MLP(self.in_dim, audio_dim, 32, 2)
        
        # Expression attention (spatial expression influence)
        self.exp_att_net = MLP(self.in_dim, blendshape_encoded_dim, 32, 2)
        
        # Motion prediction network
        # Output: d_xyz (3) + d_rot (4) + d_opa (1) + d_scale (3) = 11
        self.out_dim = 11
        motion_in_dim = self.in_dim + audio_dim + blendshape_encoded_dim + individual_dim
        self.motion_net = MLP(motion_in_dim, self.out_dim, hidden_dim, num_layers)
    
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
        """Encode audio features with attention."""
        if audio_features is None:
            return None
        
        enc_a = self.audio_net(audio_features)
        enc_a = self.audio_att_net(enc_a.unsqueeze(0))
        
        return enc_a
    
    def forward(
        self,
        xyz: torch.Tensor,
        audio_features: torch.Tensor,
        blendshapes: torch.Tensor,
        individual_code: torch.Tensor = None,
    ) -> dict:
        """
        Predict per-Gaussian deformations.
        
        Args:
            xyz: [N, 3] Gaussian positions
            audio_features: Audio features (format depends on audio_type)
            blendshapes: [52] or [B, 52] blendshape coefficients
            individual_code: Optional individual embedding
        
        Returns:
            Dictionary with deformation predictions:
            - d_xyz: [N, 3] position offsets
            - d_rot: [N, 4] rotation offsets (quaternion)
            - d_opa: [N, 1] opacity offsets
            - d_scale: [N, 3] scale offsets
            - ambient_aud: [N, 1] audio attention magnitude
            - ambient_exp: [N, 1] expression attention magnitude
        """
        N = xyz.shape[0]
        
        # Encode position
        enc_x = self.encode_position(xyz)
        
        # Encode audio
        enc_a = self.encode_audio(audio_features)  # [1, audio_dim]
        
        # Encode blendshapes
        if blendshapes.dim() == 1:
            blendshapes = blendshapes.unsqueeze(0)
        blendshape_out = self.blendshape_encoder(blendshapes)
        enc_bs = blendshape_out['encoded']  # [1, bs_encoded_dim]
        
        # Apply facial-aware masked attention
        if self.use_facial_attention:
            attention_out = self.facial_attention(enc_x, enc_a, enc_bs)
            enc_a_weighted = attention_out['audio_weighted']
            enc_bs_weighted = attention_out['expression_weighted']
            audio_mask = attention_out['audio_mask']
            expression_mask = attention_out['expression_mask']
        else:
            # Simple attention without disentanglement
            aud_ch_att = self.aud_ch_att_net(enc_x)
            exp_att = self.exp_att_net(enc_x)
            
            enc_a_weighted = enc_a.repeat(N, 1) * aud_ch_att
            enc_bs_weighted = enc_bs.repeat(N, 1) * torch.relu(exp_att)
            audio_mask = aud_ch_att
            expression_mask = exp_att
        
        # Combine features
        if individual_code is not None:
            individual_code_expanded = individual_code.repeat(N, 1)
            h = torch.cat([enc_x, enc_a_weighted, enc_bs_weighted, individual_code_expanded], dim=-1)
        else:
            h = torch.cat([enc_x, enc_a_weighted, enc_bs_weighted], dim=-1)
        
        # Predict motion
        motion = self.motion_net(h)
        
        # Parse outputs
        d_xyz = motion[..., :3] * 1e-2  # Small position offsets
        d_rot = motion[..., 3:7]  # Quaternion offsets
        d_opa = motion[..., 7:8]  # Opacity offsets
        d_scale = motion[..., 8:11]  # Scale offsets
        
        return {
            'd_xyz': d_xyz,
            'd_rot': d_rot,
            'd_opa': d_opa,
            'd_scale': d_scale,
            'ambient_aud': audio_mask.norm(dim=-1, keepdim=True),
            'ambient_exp': expression_mask.norm(dim=-1, keepdim=True),
            'blendshape_features': blendshape_out,
        }
    
    def get_params(self, lr: float, lr_net: float, weight_decay: float = 0):
        """Get optimizer parameter groups."""
        params = [
            {'params': self.audio_net.parameters(), 'lr': lr_net, 'weight_decay': weight_decay},
            {'params': self.audio_att_net.parameters(), 'lr': lr_net * 5, 'weight_decay': 0.0001},
            {'params': self.blendshape_encoder.parameters(), 'lr': lr_net, 'weight_decay': weight_decay},
            {'params': self.encoder_xy.parameters(), 'lr': lr},
            {'params': self.encoder_yz.parameters(), 'lr': lr},
            {'params': self.encoder_xz.parameters(), 'lr': lr},
            {'params': self.aud_ch_att_net.parameters(), 'lr': lr_net, 'weight_decay': weight_decay},
            {'params': self.exp_att_net.parameters(), 'lr': lr_net, 'weight_decay': weight_decay},
            {'params': self.motion_net.parameters(), 'lr': lr_net, 'weight_decay': weight_decay},
        ]
        
        if self.use_facial_attention:
            params.append({
                'params': self.facial_attention.parameters(),
                'lr': lr_net,
                'weight_decay': weight_decay
            })
        
        if self.individual_dim > 0:
            params.append({
                'params': self.individual_codes,
                'lr': lr_net,
                'weight_decay': weight_decay
            })
        
        return params


# Backward compatibility alias
MotionNetwork = BlendshapeMotionNetwork

