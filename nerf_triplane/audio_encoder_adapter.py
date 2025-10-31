"""
Audio Encoder Adapter Module

This module provides adapter classes to integrate enhanced foundation model encoders
(Whisper, SpeechT5, EnCodec) with the existing SyncFace architecture while maintaining
backward compatibility with the original AudioEncoder (LRS2-based).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict
import librosa
import numpy as np


class MelSpectrogramExtractor(nn.Module):
    """
    Extract mel-spectrogram from raw audio waveform.
    Compatible with foundation model audio encoders.
    """
    def __init__(self,
                 sample_rate: int = 16000,
                 n_fft: int = 400,
                 hop_length: int = 160,
                 n_mels: int = 80,
                 window_size: int = 400):
        super().__init__()
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.window_size = window_size
        
        # Create mel filterbank
        mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels
        )
        self.register_buffer('mel_basis', torch.from_numpy(mel_basis).float())
        
        # Create Hann window
        window = torch.hann_window(window_size)
        self.register_buffer('window', window)
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [B, T] raw audio waveform
        Returns:
            mel: [B, n_mels, T'] mel spectrogram
        """
        # Compute STFT
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.window_size,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode='reflect'
        )
        
        # Compute magnitude spectrogram
        magnitude = torch.abs(stft)  # [B, freq, time]
        
        # Apply mel filterbank
        mel = torch.matmul(self.mel_basis, magnitude)  # [B, n_mels, time]
        
        # Convert to log scale
        mel = torch.log(torch.clamp(mel, min=1e-5))
        
        return mel


class AudioEncoderAdapter(nn.Module):
    """
    Adapter to make enhanced audio encoders compatible with SyncFace's
    AudioNet interface. This allows seamless integration with the existing
    NeRFNetwork architecture.
    
    The adapter:
    1. Processes mel spectrograms from audio
    2. Extracts features using foundation models
    3. Projects to the expected dimension for AudioNet
    4. Maintains temporal consistency for frame-by-frame processing
    """
    def __init__(self,
                 encoder_type: str = "whisper",
                 foundation_model_dim: int = 512,
                 output_dim: int = 512,
                 use_temporal_conv: bool = True):
        super().__init__()
        
        self.encoder_type = encoder_type
        self.foundation_model_dim = foundation_model_dim
        
        # Import enhanced encoder
        from .enhanced_audio_encoder import EnhancedAudioEncoder
        
        self.enhanced_encoder = EnhancedAudioEncoder(
            encoder_type=encoder_type,
            output_dim=foundation_model_dim,
            use_prosody=True,
            use_contrastive=False,  # Contrastive learning done separately
            freeze_backbone=True
        )
        
        # Mel spectrogram extractor
        self.mel_extractor = MelSpectrogramExtractor()
        
        # Temporal convolution for frame-level features
        if use_temporal_conv:
            self.temporal_conv = nn.Sequential(
                nn.Conv1d(foundation_model_dim, foundation_model_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(foundation_model_dim, foundation_model_dim, kernel_size=3, padding=1),
                nn.ReLU()
            )
        else:
            self.temporal_conv = None
        
        # Final projection to match AudioEncoder output dimension
        self.output_projection = nn.Sequential(
            nn.Linear(foundation_model_dim, output_dim),
            nn.ReLU()
        )
    
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, 1, H, W] mel spectrogram (matching original AudioEncoder input)
                 where H is mel bins, W is time frames
        Returns:
            features: [B, output_dim] audio features
        """
        batch_size = mel.shape[0]
        
        # Reshape mel if needed: [B, 1, H, W] -> [B, H, W]
        if mel.dim() == 4:
            mel = mel.squeeze(1)
        
        # Process through enhanced encoder
        # The enhanced encoder expects raw audio or mel features
        result = self.enhanced_encoder(mel)
        features = result['audio_features_with_prosody'] if 'audio_features_with_prosody' in result else result['audio_features']
        
        # Apply temporal convolution if enabled
        if self.temporal_conv is not None and features.dim() > 2:
            features = features.permute(0, 2, 1)  # [B, C, T]
            features = self.temporal_conv(features)
            features = features.mean(dim=-1)  # Pool over time [B, C]
        elif features.dim() > 2:
            features = features.mean(dim=-2)  # Pool over time
        
        # Project to output dimension
        features = self.output_projection(features)  # [B, output_dim]
        
        return features


class HybridAudioEncoder(nn.Module):
    """
    Hybrid encoder that combines the original LRS2-trained AudioEncoder
    with enhanced foundation model encoders for best of both worlds:
    - LRS2 encoder: Strong audio-lip synchronization
    - Foundation encoder: Prosodic features, emotional context, speaking style
    """
    def __init__(self,
                 use_lrs2: bool = True,
                 use_foundation: bool = True,
                 foundation_type: str = "whisper",
                 lrs2_weight: float = 0.5,
                 foundation_weight: float = 0.5,
                 output_dim: int = 512):
        super().__init__()
        
        self.use_lrs2 = use_lrs2
        self.use_foundation = use_foundation
        self.lrs2_weight = lrs2_weight
        self.foundation_weight = foundation_weight
        
        # Original LRS2-trained encoder
        if use_lrs2:
            from .network import AudioEncoder
            self.lrs2_encoder = AudioEncoder()
            # LRS2 encoder output is 512-dim
            self.lrs2_projection = nn.Linear(512, output_dim)
        
        # Enhanced foundation model encoder
        if use_foundation:
            self.foundation_encoder = AudioEncoderAdapter(
                encoder_type=foundation_type,
                foundation_model_dim=512,
                output_dim=output_dim
            )
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(
            torch.tensor([lrs2_weight, foundation_weight])
        )
        
        # Fusion network
        fusion_input_dim = output_dim * (int(use_lrs2) + int(use_foundation))
        self.fusion_net = nn.Sequential(
            nn.Linear(fusion_input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: [B, 1, H, W] mel spectrogram
        Returns:
            features: [B, output_dim] fused audio features
        """
        features_list = []
        
        # Get LRS2 features (audio-lip sync)
        if self.use_lrs2:
            lrs2_features = self.lrs2_encoder(mel)  # [B, 512]
            lrs2_features = self.lrs2_projection(lrs2_features)  # [B, output_dim]
            features_list.append(lrs2_features)
        
        # Get foundation model features (prosody, emotion, style)
        if self.use_foundation:
            foundation_features = self.foundation_encoder(mel)  # [B, output_dim]
            features_list.append(foundation_features)
        
        # Fuse features
        if len(features_list) == 1:
            return features_list[0]
        
        # Weighted fusion
        weights = F.softmax(self.fusion_weights, dim=0)
        weighted_features = sum(w * f for w, f in zip(weights, features_list))
        
        # Concatenate for fusion network
        concat_features = torch.cat(features_list, dim=-1)
        fused_features = self.fusion_net(concat_features)
        
        # Residual connection with weighted features
        final_features = fused_features + weighted_features
        
        return final_features


class FoundationModelAudioNet(nn.Module):
    """
    Replacement for AudioNet that uses foundation models.
    Drop-in compatible with the original AudioNet interface.
    
    This processes the audio features in the same temporal window manner
    as the original AudioNet, but uses enhanced foundation model features.
    """
    def __init__(self, 
                 dim_in: int = 512,  # Input from enhanced encoder
                 dim_aud: int = 64,  # Output audio dimension
                 win_size: int = 16,
                 use_attention: bool = True):
        super().__init__()
        
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.use_attention = use_attention
        
        # Temporal processing
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(dim_in, 256, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Conv1d(256, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1, bias=True),
            nn.LayerNorm(128),
            nn.GELU(),
        )
        
        # Multi-head attention for temporal context
        if use_attention:
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=128,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )
        
        # Final projection
        self.encoder_fc = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, dim_aud),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, dim_in, win_size] audio features in temporal window
        Returns:
            features: [B, dim_aud] processed audio features
        """
        # Temporal convolution
        x = self.encoder_conv(x)  # [B, 128, win_size/8]
        
        # Apply attention over temporal dimension
        if self.use_attention:
            x = x.permute(0, 2, 1)  # [B, T, 128]
            x, _ = self.temporal_attention(x, x, x)
            x = x.mean(dim=1)  # [B, 128]
        else:
            x = x.mean(dim=-1)  # [B, 128]
        
        # Final projection
        x = self.encoder_fc(x)  # [B, dim_aud]
        
        return x


class FoundationModelAudioNetAVE(nn.Module):
    """
    Enhanced version of AudioNet_ave that processes features from
    the Audio-Visual Encoder (AVE) with foundation model enhancements.
    """
    def __init__(self,
                 dim_in: int = 512,  # Input from enhanced AVE encoder
                 dim_aud: int = 64,  # Output audio dimension
                 win_size: int = 16):
        super().__init__()
        
        self.win_size = win_size
        self.dim_aud = dim_aud
        
        # Enhanced feature processing with residual connections
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(dim_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, dim_aud),
        )
        
        # Prosodic enhancement branch
        self.prosody_branch = nn.Sequential(
            nn.Linear(dim_in, 128),
            nn.ReLU(),
            nn.Linear(128, dim_aud)
        )
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(dim_aud * 2, dim_aud),
            nn.LayerNorm(dim_aud)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, dim_in] or [B, dim_in] features from enhanced AVE encoder
        Returns:
            features: [B, dim_aud] processed features
        """
        # Handle different input shapes
        if x.dim() == 3:
            x = x.squeeze(1)  # [B, dim_in]
        
        # Main feature extraction
        main_features = self.encoder_fc1(x)  # [B, dim_aud]
        
        # Prosodic enhancement
        prosody_features = self.prosody_branch(x)  # [B, dim_aud]
        
        # Fuse main and prosodic features
        combined = torch.cat([main_features, prosody_features], dim=-1)
        fused = self.fusion(combined)  # [B, dim_aud]
        
        # Add residual connection
        output = fused + main_features
        
        return output


def load_enhanced_encoder_checkpoint(encoder: nn.Module, 
                                     checkpoint_path: str,
                                     strict: bool = False) -> nn.Module:
    """
    Load checkpoint for enhanced encoder, handling key mismatches gracefully.
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Load with strict=False to allow partial loading
        encoder.load_state_dict(state_dict, strict=strict)
        print(f"[INFO] Loaded enhanced encoder checkpoint from {checkpoint_path}")
        
    except Exception as e:
        print(f"[WARN] Could not load checkpoint: {e}")
        print(f"[INFO] Using randomly initialized enhanced encoder")
    
    return encoder

