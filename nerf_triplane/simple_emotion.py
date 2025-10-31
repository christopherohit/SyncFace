"""
Simple Emotion Recognition for SyncTalk
Lightweight implementation without heavy dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleEmotionRecognizer(nn.Module):
    """
    Lightweight emotion recognition from audio features.
    Works directly with existing audio features (deepspeech, hubert, etc.)
    """
    
    def __init__(self, audio_dim=29, emotion_dim=64, num_emotions=7):
        super().__init__()
        
        self.audio_dim = audio_dim
        self.emotion_dim = emotion_dim
        self.num_emotions = num_emotions
        
        # Simple CNN for emotion classification
        self.emotion_encoder = nn.Sequential(
            nn.Conv1d(audio_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Emotion classifier
        self.emotion_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_emotions)
        )
        
        # Emotion embedding
        self.emotion_proj = nn.Linear(num_emotions, emotion_dim)
    
    def forward(self, audio_features):
        """
        Args:
            audio_features: [B, audio_dim, T] audio features
        Returns:
            emotion_embedding: [B, emotion_dim]
        """
        # Extract features
        features = self.emotion_encoder(audio_features)  # [B, 128, 1]
        features = features.squeeze(-1)  # [B, 128]
        
        # Classify emotion
        emotion_logits = self.emotion_head(features)  # [B, num_emotions]
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        
        # Project to embedding
        emotion_embedding = self.emotion_proj(emotion_probs)  # [B, emotion_dim]
        
        return emotion_embedding, emotion_probs


class SimpleEmotionIntegration(nn.Module):
    """
    Simple integration of emotion into audio features.
    No external dependencies required.
    """
    
    def __init__(self, audio_dim=29, output_dim=64, emotion_strength=0.7):
        super().__init__()
        
        self.emotion_strength = emotion_strength
        
        # Simple emotion recognizer
        self.emotion_recognizer = SimpleEmotionRecognizer(
            audio_dim=audio_dim,
            emotion_dim=output_dim
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.ReLU()
        )
    
    def forward(self, audio_features, base_features):
        """
        Args:
            audio_features: [B, audio_dim, T] raw audio features
            base_features: [B, output_dim] base audio encoding
        Returns:
            enhanced_features: [B, output_dim] emotion-enhanced features
        """
        # Get emotion embedding
        emotion_emb, emotion_probs = self.emotion_recognizer(audio_features)
        
        # Fuse with base features
        combined = torch.cat([base_features, emotion_emb * self.emotion_strength], dim=-1)
        enhanced_features = self.fusion(combined)
        
        return enhanced_features

