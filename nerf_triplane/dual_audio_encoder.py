"""
Dual Audio Encoder Architecture

This module implements a dual-branch audio encoder that separates:
1. Content/Phoneme Branch: Captures semantic content and phoneme-level features
2. Sync Branch: Captures fine-grained temporal dynamics for precise lip synchronization

The dual design allows the model to:
- Learn rich phoneme representations from foundation models (Wav2Vec2/AV-HuBERT)
- Maintain temporal precision for lip-sync from raw waveform features
- Fuse both streams for superior talking head generation

Reference:
- "Speech Driven Video Editing via an Audio-Conditioned Diffusion Model" (Dual encoders)
- Wav2Lip's approach to content vs sync separation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class ContentBranch(nn.Module):
    """
    Content/Phoneme Branch: Extracts semantic and phoneme-level features.
    
    This branch processes features from foundation models (Wav2Vec2, AV-HuBERT, HuBERT)
    which provide rich phoneme and semantic representations.
    """
    
    def __init__(self, 
                 input_dim: int = 1024,  # Wav2Vec2/HuBERT feature dim
                 hidden_dim: int = 512,
                 output_dim: int = 64,
                 num_layers: int = 3,
                 use_attention: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_attention = use_attention
        
        # Multi-layer feature extraction
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else output_dim
            layers.extend([
                nn.Linear(current_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
            current_dim = out_dim
        
        self.encoder = nn.Sequential(*layers[:-1])  # Remove last dropout
        
        # Self-attention for capturing long-range phoneme dependencies
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(output_dim)
        
        # Temporal smoothing for stable phoneme features
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1, groups=output_dim),
            nn.GELU(),
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1)
        )
        self.temporal_norm = nn.LayerNorm(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, input_dim] phoneme features from foundation models
        
        Returns:
            content_features: [B, T, output_dim] semantic/phoneme features
        """
        # Extract features
        features = self.encoder(x)  # [B, T, output_dim]
        
        # Self-attention for long-range dependencies
        if self.use_attention:
            attn_out, _ = self.attention(features, features, features)
            features = self.attention_norm(features + attn_out)  # Residual
        
        # Temporal smoothing
        features_conv = features.permute(0, 2, 1)  # [B, C, T]
        features_conv = self.temporal_conv(features_conv)
        features_conv = features_conv.permute(0, 2, 1)  # [B, T, C]
        content_features = self.temporal_norm(features + features_conv)  # Residual
        
        return content_features


class SyncBranch(nn.Module):
    """
    Sync Branch: Extracts fine-grained temporal features for lip synchronization.
    
    This branch processes raw waveform or mel-spectrogram features to capture
    rapid temporal dynamics essential for accurate lip movements.
    """
    
    def __init__(self,
                 input_type: str = 'mel',  # 'mel' or 'waveform'
                 mel_bins: int = 80,
                 output_dim: int = 64,
                 use_residual: bool = True):
        super().__init__()
        
        self.input_type = input_type
        self.output_dim = output_dim
        
        if input_type == 'mel':
            # Process mel-spectrogram (similar to Wav2Lip)
            self.conv_encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=not use_residual),
                nn.BatchNorm2d(32) if use_residual else nn.Identity(),
                nn.ReLU(),
                
                nn.Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=not use_residual),
                nn.BatchNorm2d(64) if use_residual else nn.Identity(),
                nn.ReLU(),
                
                nn.Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=not use_residual),
                nn.BatchNorm2d(128) if use_residual else nn.Identity(),
                nn.ReLU(),
                
                nn.Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                
                nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            )
            
            # Project to output dimension
            self.projection = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, output_dim)
            )
        
        elif input_type == 'waveform':
            # Process raw waveform with 1D convolutions
            self.conv_encoder = nn.Sequential(
                nn.Conv1d(1, 64, kernel_size=80, stride=16, padding=40),  # ~20ms stride
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(512),
                nn.ReLU(),
            )
            
            self.projection = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, output_dim)
            )
        else:
            raise ValueError(f"Unknown input_type: {input_type}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, H, W] for mel or [B, 1, T] for waveform
        
        Returns:
            sync_features: [B, output_dim] fine-grained sync features
        """
        features = self.conv_encoder(x)
        
        if self.input_type == 'mel':
            # Global pooling for mel features
            features = features.mean(dim=[2, 3])  # [B, 512]
        
        sync_features = self.projection(features)  # [B, output_dim]
        
        return sync_features


class FusionModule(nn.Module):
    """
    Fusion Module: Combines content and sync features intelligently.
    
    Uses cross-attention and gated fusion to balance semantic content
    with temporal synchronization cues.
    """
    
    def __init__(self,
                 content_dim: int = 64,
                 sync_dim: int = 64,
                 output_dim: int = 64,
                 fusion_mode: str = 'cross_attention'):  # 'concat', 'add', 'cross_attention', 'gated'
        super().__init__()
        
        self.fusion_mode = fusion_mode
        
        if fusion_mode == 'concat':
            # Simple concatenation + projection
            self.fusion = nn.Sequential(
                nn.Linear(content_dim + sync_dim, output_dim * 2),
                nn.LayerNorm(output_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim * 2, output_dim)
            )
        
        elif fusion_mode == 'add':
            # Weighted addition
            self.content_proj = nn.Linear(content_dim, output_dim) if content_dim != output_dim else nn.Identity()
            self.sync_proj = nn.Linear(sync_dim, output_dim) if sync_dim != output_dim else nn.Identity()
            self.weights = nn.Parameter(torch.tensor([0.5, 0.5]))
        
        elif fusion_mode == 'cross_attention':
            # Cross-attention between content and sync
            self.content_proj = nn.Linear(content_dim, output_dim)
            self.sync_proj = nn.Linear(sync_dim, output_dim)
            
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )
            
            self.fusion = nn.Sequential(
                nn.LayerNorm(output_dim),
                nn.Linear(output_dim, output_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(output_dim * 2, output_dim)
            )
        
        elif fusion_mode == 'gated':
            # Gated fusion (inspired by GRU)
            self.content_proj = nn.Linear(content_dim, output_dim)
            self.sync_proj = nn.Linear(sync_dim, output_dim)
            
            self.gate = nn.Sequential(
                nn.Linear(output_dim * 2, output_dim),
                nn.Sigmoid()
            )
            
            self.fusion = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim)
            )
        
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
    
    def forward(self, content_features: torch.Tensor, sync_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            content_features: [B, T, content_dim] or [B, content_dim]
            sync_features: [B, sync_dim]
        
        Returns:
            fused_features: [B, output_dim] or [B, T, output_dim]
        """
        if self.fusion_mode == 'concat':
            # Expand sync features to match content temporal dimension if needed
            if content_features.dim() == 3:
                sync_expanded = sync_features.unsqueeze(1).expand(-1, content_features.size(1), -1)
                combined = torch.cat([content_features, sync_expanded], dim=-1)
            else:
                combined = torch.cat([content_features, sync_features], dim=-1)
            
            fused = self.fusion(combined)
        
        elif self.fusion_mode == 'add':
            content_proj = self.content_proj(content_features)
            sync_proj = self.sync_proj(sync_features)
            
            if content_proj.dim() == 3:
                sync_proj = sync_proj.unsqueeze(1).expand_as(content_proj)
            
            weights = F.softmax(self.weights, dim=0)
            fused = weights[0] * content_proj + weights[1] * sync_proj
        
        elif self.fusion_mode == 'cross_attention':
            content_proj = self.content_proj(content_features)  # [B, T, D]
            sync_proj = self.sync_proj(sync_features)  # [B, D]
            
            # Ensure content_proj is 3D
            if content_proj.dim() == 2:
                content_proj = content_proj.unsqueeze(1)
            
            # Expand sync for attention
            sync_proj = sync_proj.unsqueeze(1)  # [B, 1, D]
            
            # Cross-attention: sync attends to content
            attn_out, _ = self.cross_attention(sync_proj, content_proj, content_proj)
            
            # Fuse attended features
            fused = self.fusion(attn_out.squeeze(1) if attn_out.dim() == 3 and attn_out.size(1) == 1 else attn_out)
        
        elif self.fusion_mode == 'gated':
            content_proj = self.content_proj(content_features)
            sync_proj = self.sync_proj(sync_features)
            
            if content_proj.dim() == 3:
                sync_proj = sync_proj.unsqueeze(1).expand_as(content_proj)
            
            # Gated fusion
            gate_input = torch.cat([content_proj, sync_proj], dim=-1)
            gate = self.gate(gate_input)
            
            gated = gate * content_proj + (1 - gate) * sync_proj
            fused = self.fusion(gated)
        
        return fused


class DualAudioEncoder(nn.Module):
    """
    Dual Audio Encoder: Combines content and sync branches for superior lip-sync.
    
    Architecture:
    - Content Branch: Processes Wav2Vec2/AV-HuBERT/HuBERT features (phoneme-rich)
    - Sync Branch: Processes mel-spectrogram or raw waveform (temporal precision)
    - Fusion Module: Intelligently combines both streams
    
    Benefits:
    - Richer phoneme representations from foundation models
    - Precise temporal alignment from raw audio
    - Balanced gradient flow during training
    """
    
    def __init__(self,
                 # Content branch config
                 content_input_dim: int = 1024,
                 content_output_dim: int = 64,
                 use_content_attention: bool = True,
                 
                 # Sync branch config
                 sync_input_type: str = 'mel',
                 sync_output_dim: int = 64,
                 
                 # Fusion config
                 fusion_mode: str = 'cross_attention',
                 final_output_dim: int = 64,
                 
                 # Gradient balancing
                 content_weight: float = 1.0,
                 sync_weight: float = 1.0):
        super().__init__()
        
        self.content_weight = content_weight
        self.sync_weight = sync_weight
        
        # Content branch (phoneme features)
        self.content_branch = ContentBranch(
            input_dim=content_input_dim,
            output_dim=content_output_dim,
            use_attention=use_content_attention
        )
        
        # Sync branch (temporal features)
        self.sync_branch = SyncBranch(
            input_type=sync_input_type,
            output_dim=sync_output_dim
        )
        
        # Fusion module
        self.fusion = FusionModule(
            content_dim=content_output_dim,
            sync_dim=sync_output_dim,
            output_dim=final_output_dim,
            fusion_mode=fusion_mode
        )
    
    def forward(self, 
                content_input: torch.Tensor,
                sync_input: torch.Tensor,
                return_separate: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            content_input: [B, T, content_dim] features from Wav2Vec2/HuBERT
            sync_input: [B, 1, H, W] mel-spec or [B, 1, T] waveform
            return_separate: Whether to return individual branch outputs
        
        Returns:
            Dictionary containing:
            - fused_features: [B, final_output_dim] combined features
            - content_features: [B, T, content_dim] (if return_separate)
            - sync_features: [B, sync_dim] (if return_separate)
        """
        # Extract features from both branches
        content_features = self.content_branch(content_input)  # [B, T, content_dim]
        sync_features = self.sync_branch(sync_input)  # [B, sync_dim]
        
        # Apply gradient balancing if weights are set
        if self.training:
            content_features = content_features * self.content_weight
            sync_features = sync_features * self.sync_weight
        
        # Fuse features
        fused_features = self.fusion(content_features, sync_features)  # [B, final_output_dim]
        
        result = {'fused_features': fused_features}
        
        if return_separate:
            result['content_features'] = content_features
            result['sync_features'] = sync_features
        
        return result
    
    def set_gradient_weights(self, content_weight: float, sync_weight: float):
        """
        Dynamically adjust gradient weights for balancing training.
        
        Args:
            content_weight: Weight for content branch gradients
            sync_weight: Weight for sync branch gradients
        """
        self.content_weight = content_weight
        self.sync_weight = sync_weight


# Factory function for easy instantiation
def create_dual_audio_encoder(config: Dict) -> DualAudioEncoder:
    """
    Factory function to create dual audio encoder from config.
    
    Example config:
    {
        'content_input_dim': 1024,  # Wav2Vec2/HuBERT dimension
        'content_output_dim': 64,
        'sync_input_type': 'mel',  # or 'waveform'
        'sync_output_dim': 64,
        'fusion_mode': 'cross_attention',  # or 'concat', 'add', 'gated'
        'final_output_dim': 64,
        'content_weight': 1.0,
        'sync_weight': 1.0
    }
    """
    return DualAudioEncoder(**config)

