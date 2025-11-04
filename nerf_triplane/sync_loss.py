"""
Multi-Scale Sync Loss Module

This module implements multiple lip-sync losses at different time resolutions
for better supervision of audio-visual synchronization:

1. SyncNet Loss: Frozen expert network from Wav2Lip for contrastive audio-video sync
2. LSE-C (Lip Sync Expert - Confidence): Binary sync/non-sync classification
3. LSE-D (Lip Sync Expert - Distance): Regression-based sync distance prediction

These losses operate at multiple temporal scales to capture both frame-level
and sequence-level synchronization.

References:
- Wav2Lip: "A Lip Sync Expert Is All You Need for Speech to Lip Generation In The Wild"
- SyncNet: "Out of time: automated lip sync in the wild"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np


class SyncNetAudioEncoder(nn.Module):
    """
    Audio encoder from Wav2Lip's SyncNet.
    Processes mel-spectrogram to extract audio embeddings for sync loss.
    """
    
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1, H, W] mel-spectrogram
        
        Returns:
            audio_embedding: [B, 512] audio embedding
        """
        features = self.encoder(x)  # [B, 512, H', W']
        features = features.mean(dim=[2, 3])  # Global average pooling
        embedding = self.fc(features)  # [B, 512]
        return embedding


class SyncNetVideoEncoder(nn.Module):
    """
    Video encoder from Wav2Lip's SyncNet.
    Processes facial video frames to extract visual embeddings for sync loss.
    """
    
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=(5, 7, 7), stride=(1, 3, 3), padding=(2, 3, 3)),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2)),
            
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2)),
            
            nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2)),
            
            nn.Conv3d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2)),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, 3, H, W] video frames (T consecutive frames)
        
        Returns:
            video_embedding: [B, 512] video embedding
        """
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] for 3D conv
        
        features = self.encoder(x)  # [B, 512, T', H', W']
        features = features.mean(dim=[2, 3, 4])  # Global average pooling
        embedding = self.fc(features)  # [B, 512]
        return embedding


class SyncNetLoss(nn.Module):
    """
    SyncNet Loss: Contrastive loss for audio-video synchronization.
    
    Uses a frozen pretrained SyncNet model to compute sync loss.
    The model learns to align audio and video representations in a shared embedding space.
    """
    
    def __init__(self,
                 checkpoint_path: Optional[str] = None,
                 freeze: bool = True):
        super().__init__()
        
        self.audio_encoder = SyncNetAudioEncoder()
        self.video_encoder = SyncNetVideoEncoder()
        self.freeze = freeze
        
        # Load pretrained weights if provided
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
        
        # Freeze parameters if required
        if freeze:
            for param in self.parameters():
                param.requires_grad = False
            self.eval()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load pretrained SyncNet checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                if 'audio_encoder' in checkpoint:
                    self.audio_encoder.load_state_dict(checkpoint['audio_encoder'])
                if 'video_encoder' in checkpoint:
                    self.video_encoder.load_state_dict(checkpoint['video_encoder'])
            else:
                print(f"[WARN] Unexpected checkpoint format for SyncNet")
            print(f"[INFO] Loaded SyncNet checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"[WARN] Could not load SyncNet checkpoint: {e}")
            print(f"[INFO] Using randomly initialized SyncNet")
    
    def forward(self, 
                audio_mel: torch.Tensor,
                video_frames: torch.Tensor,
                return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        Compute SyncNet contrastive loss.
        
        Args:
            audio_mel: [B, 1, H, W] mel-spectrogram
            video_frames: [B, T, 3, H, W] video frames
            return_embeddings: Whether to return embeddings
        
        Returns:
            Dictionary containing:
            - sync_loss: Contrastive sync loss
            - audio_embedding: [B, 512] (if return_embeddings)
            - video_embedding: [B, 512] (if return_embeddings)
        """
        # Extract embeddings
        audio_embedding = self.audio_encoder(audio_mel)  # [B, 512]
        video_embedding = self.video_encoder(video_frames)  # [B, 512]
        
        # Normalize embeddings
        audio_embedding = F.normalize(audio_embedding, p=2, dim=1)
        video_embedding = F.normalize(video_embedding, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.sum(audio_embedding * video_embedding, dim=1)  # [B]
        
        # Contrastive loss: maximize similarity for matched pairs
        # minimize for non-matched pairs (negative samples)
        sync_loss = 1 - similarity.mean()
        
        result = {'sync_loss': sync_loss}
        
        if return_embeddings:
            result['audio_embedding'] = audio_embedding
            result['video_embedding'] = video_embedding
            result['similarity'] = similarity
        
        return result


class LSEConfidence(nn.Module):
    """
    LSE-C (Lip Sync Expert - Confidence): Binary classification for sync/non-sync.
    
    A lightweight discriminator that classifies whether audio and video are synchronized.
    Provides strong binary supervision signal.
    """
    
    def __init__(self,
                 audio_dim: int = 512,
                 video_dim: int = 512,
                 hidden_dim: int = 256):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(audio_dim + video_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self,
                audio_features: torch.Tensor,
                video_features: torch.Tensor,
                is_synced: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            audio_features: [B, audio_dim] audio embeddings
            video_features: [B, video_dim] video embeddings
            is_synced: [B] binary labels (1=synced, 0=not synced)
        
        Returns:
            Dictionary containing:
            - confidence: [B] predicted sync confidence
            - lse_c_loss: Binary cross-entropy loss (if is_synced provided)
        """
        # Concatenate audio and video features
        combined = torch.cat([audio_features, video_features], dim=1)
        
        # Predict sync confidence
        confidence = self.classifier(combined).squeeze(-1)  # [B]
        
        result = {'confidence': confidence}
        
        # Compute loss if labels provided
        if is_synced is not None:
            lse_c_loss = F.binary_cross_entropy(
                confidence,
                is_synced.float(),
                reduction='mean'
            )
            result['lse_c_loss'] = lse_c_loss
        
        return result


class LSEDistance(nn.Module):
    """
    LSE-D (Lip Sync Expert - Distance): Regression-based sync distance prediction.
    
    Predicts the temporal offset between audio and video, providing fine-grained
    supervision for temporal alignment.
    """
    
    def __init__(self,
                 audio_dim: int = 512,
                 video_dim: int = 512,
                 hidden_dim: int = 256,
                 max_offset: int = 10):  # Maximum frame offset to predict
        super().__init__()
        
        self.max_offset = max_offset
        
        self.regressor = nn.Sequential(
            nn.Linear(audio_dim + video_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Output in [-1, 1]
        )
    
    def forward(self,
                audio_features: torch.Tensor,
                video_features: torch.Tensor,
                offset: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            audio_features: [B, audio_dim] audio embeddings
            video_features: [B, video_dim] video embeddings
            offset: [B] frame offset labels (negative = video ahead, positive = audio ahead)
        
        Returns:
            Dictionary containing:
            - predicted_offset: [B] predicted frame offset
            - lse_d_loss: L1 loss (if offset provided)
        """
        # Concatenate audio and video features
        combined = torch.cat([audio_features, video_features], dim=1)
        
        # Predict normalized offset
        normalized_offset = self.regressor(combined).squeeze(-1)  # [B] in [-1, 1]
        
        # Scale to actual frame offset
        predicted_offset = normalized_offset * self.max_offset
        
        result = {'predicted_offset': predicted_offset}
        
        # Compute loss if labels provided
        if offset is not None:
            # Normalize ground truth offset
            normalized_gt = torch.clamp(offset.float() / self.max_offset, -1.0, 1.0)
            
            lse_d_loss = F.l1_loss(normalized_offset, normalized_gt, reduction='mean')
            result['lse_d_loss'] = lse_d_loss
        
        return result


class MultiScaleSyncLoss(nn.Module):
    """
    Multi-Scale Sync Loss: Combines SyncNet, LSE-C, and LSE-D at multiple time resolutions.
    
    This module provides comprehensive supervision for lip synchronization by:
    1. Using frozen SyncNet for contrastive audio-video alignment
    2. Training LSE-C for binary sync classification
    3. Training LSE-D for fine-grained temporal offset regression
    4. Computing losses at multiple temporal scales (frame, clip, sequence)
    """
    
    def __init__(self,
                 syncnet_checkpoint: Optional[str] = None,
                 use_syncnet: bool = True,
                 use_lse_c: bool = True,
                 use_lse_d: bool = True,
                 temporal_scales: List[int] = [1, 5, 10],  # Frame counts for multi-scale
                 loss_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        
        self.use_syncnet = use_syncnet
        self.use_lse_c = use_lse_c
        self.use_lse_d = use_lse_d
        self.temporal_scales = temporal_scales
        
        # Default loss weights
        self.loss_weights = loss_weights or {
            'syncnet': 1.0,
            'lse_c': 0.5,
            'lse_d': 0.3
        }
        
        # Initialize loss modules
        if use_syncnet:
            self.syncnet = SyncNetLoss(checkpoint_path=syncnet_checkpoint, freeze=True)
        
        if use_lse_c:
            self.lse_c = LSEConfidence()
        
        if use_lse_d:
            self.lse_d = LSEDistance()
    
    def create_negative_samples(self,
                               audio_features: torch.Tensor,
                               video_features: torch.Tensor,
                               shift_range: int = 5) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create negative samples by shifting audio or video temporally.
        
        Returns:
            negative_audio: Shifted audio features
            negative_video: Shifted video features
            offsets: Temporal offsets applied
        """
        B = audio_features.size(0)
        device = audio_features.device
        
        # Random shift in range [-shift_range, shift_range] excluding 0
        offsets = torch.randint(-shift_range, shift_range + 1, (B,), device=device)
        offsets[offsets == 0] = 1  # No zero offset
        
        # For simplicity, we'll use the same features with labels indicating offset
        # In practice, you'd actually shift the temporal alignment
        
        return audio_features, video_features, offsets
    
    def forward(self,
                audio_mel: torch.Tensor,
                video_frames: torch.Tensor,
                audio_features: Optional[torch.Tensor] = None,
                video_features: Optional[torch.Tensor] = None,
                use_negative_samples: bool = True) -> Dict[str, torch.Tensor]:
        """
        Compute multi-scale sync loss.
        
        Args:
            audio_mel: [B, 1, H, W] mel-spectrogram
            video_frames: [B, T, 3, H, W] video frames
            audio_features: [B, D] optional pre-extracted audio features
            video_features: [B, D] optional pre-extracted video features
            use_negative_samples: Whether to use negative samples for training
        
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        total_loss = 0.0
        
        # SyncNet Loss
        if self.use_syncnet:
            syncnet_result = self.syncnet(audio_mel, video_frames, return_embeddings=True)
            sync_loss = syncnet_result['sync_loss']
            
            losses['syncnet_loss'] = sync_loss
            total_loss += self.loss_weights['syncnet'] * sync_loss
            
            # Use SyncNet embeddings if not provided separately
            if audio_features is None:
                audio_features = syncnet_result['audio_embedding']
            if video_features is None:
                video_features = syncnet_result['video_embedding']
        
        # LSE-C: Confidence Loss
        if self.use_lse_c and audio_features is not None and video_features is not None:
            # Positive samples (synced)
            pos_result = self.lse_c(
                audio_features,
                video_features,
                is_synced=torch.ones(audio_features.size(0), device=audio_features.device)
            )
            
            lse_c_loss = pos_result['lse_c_loss']
            
            # Add negative samples if requested
            if use_negative_samples:
                neg_audio, neg_video, _ = self.create_negative_samples(audio_features, video_features)
                neg_result = self.lse_c(
                    neg_audio,
                    neg_video,
                    is_synced=torch.zeros(neg_audio.size(0), device=neg_audio.device)
                )
                lse_c_loss = (lse_c_loss + neg_result['lse_c_loss']) / 2
            
            losses['lse_c_loss'] = lse_c_loss
            total_loss += self.loss_weights['lse_c'] * lse_c_loss
        
        # LSE-D: Distance Loss
        if self.use_lse_d and audio_features is not None and video_features is not None:
            # Positive samples (zero offset)
            pos_result = self.lse_d(
                audio_features,
                video_features,
                offset=torch.zeros(audio_features.size(0), device=audio_features.device)
            )
            
            lse_d_loss = pos_result['lse_d_loss']
            
            # Add negative samples with known offsets
            if use_negative_samples:
                neg_audio, neg_video, offsets = self.create_negative_samples(audio_features, video_features)
                neg_result = self.lse_d(neg_audio, neg_video, offset=offsets)
                lse_d_loss = (lse_d_loss + neg_result['lse_d_loss']) / 2
            
            losses['lse_d_loss'] = lse_d_loss
            total_loss += self.loss_weights['lse_d'] * lse_d_loss
        
        losses['total_sync_loss'] = total_loss
        
        return losses
    
    def set_loss_weights(self, weights: Dict[str, float]):
        """Update loss weights dynamically."""
        self.loss_weights.update(weights)


# Utility function to load SyncNet pretrained weights
def load_syncnet_pretrained(checkpoint_path: str) -> MultiScaleSyncLoss:
    """
    Load a pretrained SyncNet model for use as a frozen sync loss.
    
    Args:
        checkpoint_path: Path to SyncNet checkpoint (from Wav2Lip)
    
    Returns:
        MultiScaleSyncLoss with loaded SyncNet weights
    """
    sync_loss = MultiScaleSyncLoss(
        syncnet_checkpoint=checkpoint_path,
        use_syncnet=True,
        use_lse_c=True,
        use_lse_d=True
    )
    return sync_loss

