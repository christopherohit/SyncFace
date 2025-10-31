"""
Emotion Integration Module

Integrates emotion recognition and blendshape mapping into the SyncTalk pipeline.
This enables emotion-sensitive facial expressions synchronized with voice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple

from .emotion_module import (
    EmotionRecognitionModule,
    EmotionBlendshapeMapper,
    EmotionConditionedAudioNet
)


class EmotionAwareNeRFModule(nn.Module):
    """
    Emotion-aware module that integrates into NeRFNetwork.
    
    This module:
    1. Extracts emotion from audio
    2. Maps emotion to blendshape adjustments
    3. Conditions facial features on emotion
    """
    
    def __init__(self,
                 opt,
                 emotion_model: str = 'wav2vec2',
                 emotion_dim: int = 64,
                 num_blendshapes: int = 52,
                 use_emotion_conditioning: bool = True):
        super().__init__()
        
        self.opt = opt
        self.emotion_dim = emotion_dim
        self.use_emotion_conditioning = use_emotion_conditioning
        
        # Emotion recognition module
        self.emotion_recognizer = EmotionRecognitionModule(
            emotion_model=emotion_model,
            emotion_dim=emotion_dim
        )
        
        # Emotion to blendshape mapper
        self.emotion_blendshape_mapper = EmotionBlendshapeMapper(
            emotion_dim=emotion_dim,
            num_blendshapes=num_blendshapes
        )
        
        # Emotion embedding for conditioning NeRF
        self.emotion_mlp = nn.Sequential(
            nn.Linear(emotion_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64)  # Same dim as audio features
        )
        
        # Temporal smoothing for emotions (avoid jitter)
        self.emotion_smoothing = EmotionTemporalSmoothing(
            emotion_dim=emotion_dim,
            window_size=5
        )
    
    def forward(self,
                audio: torch.Tensor,
                base_blendshapes: Optional[torch.Tensor] = None,
                return_all: bool = False) -> Dict[str, torch.Tensor]:
        """
        Process audio to extract emotion and generate emotion-aware features.
        
        Args:
            audio: [B, T] or [B, 1, H, W] audio input
            base_blendshapes: [B, num_blendshapes] base blendshapes (optional)
            return_all: Whether to return all intermediate results
        
        Returns:
            Dictionary containing emotion features and blendshapes
        """
        # Extract emotion
        emotion_outputs = self.emotion_recognizer(audio)
        emotion_embedding = emotion_outputs['emotion_embedding']  # [B, emotion_dim]
        
        # Smooth emotion over time
        emotion_embedding = self.emotion_smoothing(emotion_embedding)
        
        # Map emotion to blendshapes
        emotion_strength = getattr(self.opt, 'emotion_strength', 0.7)
        emotion_blendshapes = self.emotion_blendshape_mapper(
            emotion_embedding,
            base_blendshapes=base_blendshapes,
            emotion_strength=emotion_strength
        )
        
        # Create emotion conditioning for NeRF
        emotion_cond = self.emotion_mlp(emotion_embedding)  # [B, 64]
        
        result = {
            'emotion_embedding': emotion_embedding,
            'emotion_blendshapes': emotion_blendshapes,
            'emotion_cond': emotion_cond,
        }
        
        if return_all:
            result.update(emotion_outputs)
        
        return result


class EmotionTemporalSmoothing(nn.Module):
    """
    Temporal smoothing for emotion predictions to avoid jittery expressions.
    Uses exponential moving average or learned temporal convolution.
    """
    
    def __init__(self,
                 emotion_dim: int = 64,
                 window_size: int = 5,
                 method: str = 'ema'):  # 'ema' or 'conv'
        super().__init__()
        
        self.emotion_dim = emotion_dim
        self.window_size = window_size
        self.method = method
        
        if method == 'ema':
            # Exponential moving average
            self.alpha = nn.Parameter(torch.tensor(0.7))  # Learnable smoothing factor
            self.register_buffer('prev_emotion', torch.zeros(1, emotion_dim))
        
        elif method == 'conv':
            # Learned temporal convolution
            self.temporal_conv = nn.Conv1d(
                emotion_dim, emotion_dim,
                kernel_size=window_size,
                padding=window_size // 2,
                groups=emotion_dim  # Depthwise convolution
            )
    
    def forward(self, emotion: torch.Tensor) -> torch.Tensor:
        """
        Smooth emotion predictions over time.
        
        Args:
            emotion: [B, emotion_dim] current emotion
        
        Returns:
            smoothed_emotion: [B, emotion_dim] temporally smoothed emotion
        """
        if self.method == 'ema':
            # Exponential moving average
            alpha = torch.sigmoid(self.alpha)  # Clamp to [0, 1]
            
            if self.training:
                # During training, just apply smoothing
                smoothed = alpha * emotion + (1 - alpha) * self.prev_emotion.expand_as(emotion)
                self.prev_emotion = smoothed.detach().mean(dim=0, keepdim=True)
            else:
                # During inference, maintain state
                smoothed = alpha * emotion + (1 - alpha) * self.prev_emotion
                self.prev_emotion = smoothed.detach()
            
            return smoothed
        
        elif self.method == 'conv':
            # Learned temporal convolution
            if emotion.dim() == 2:
                emotion = emotion.unsqueeze(0)  # [1, B, emotion_dim]
            
            emotion = emotion.permute(0, 2, 1)  # [1, emotion_dim, B]
            smoothed = self.temporal_conv(emotion)
            smoothed = smoothed.permute(0, 2, 1).squeeze(0)  # [B, emotion_dim]
            
            return smoothed


class BlendshapeEmotionController(nn.Module):
    """
    Controls blendshape coefficients based on detected emotions.
    
    Maps discrete emotions (happy, sad, angry, etc.) to specific blendshape
    activation patterns based on FACS (Facial Action Coding System).
    """
    
    def __init__(self, num_blendshapes: int = 52):
        super().__init__()
        
        self.num_blendshapes = num_blendshapes
        
        # Define emotion-to-blendshape mappings based on FACS
        self.emotion_mappings = self._init_emotion_mappings()
    
    def _init_emotion_mappings(self) -> Dict[str, torch.Tensor]:
        """Initialize emotion to blendshape activation mappings."""
        mappings = {}
        
        # Happy: AU6 (cheek raiser) + AU12 (lip corner puller)
        happy = torch.zeros(self.num_blendshapes)
        happy[[12, 13, 48, 49]] = 0.8  # mouthSmile, cheekPuff
        mappings['happy'] = happy
        
        # Sad: AU1 (inner brow raiser) + AU4 (brow lowerer) + AU15 (lip corner depressor)
        sad = torch.zeros(self.num_blendshapes)
        sad[[0, 1, 2, 3, 14, 15]] = 0.7  # browDown, mouthFrown
        mappings['sad'] = sad
        
        # Angry: AU4 (brow lowerer) + AU5 (upper lid raiser) + AU7 (lid tightener) + AU23 (lip tightener)
        angry = torch.zeros(self.num_blendshapes)
        angry[[2, 3, 23, 14, 15]] = 0.9  # browDown, jawForward, mouthPress
        mappings['angry'] = angry
        
        # Fearful: AU1 + AU2 + AU4 + AU5 + AU20 + AU26
        fearful = torch.zeros(self.num_blendshapes)
        fearful[[4, 5, 10, 11, 25, 26]] = 0.8  # browOuterUp, eyeWide, mouthOpen
        mappings['fearful'] = fearful
        
        # Disgusted: AU9 (nose wrinkler) + AU15 + AU16
        disgusted = torch.zeros(self.num_blendshapes)
        disgusted[[8, 16, 17]] = 0.8  # noseSneer, mouthUpperUp
        mappings['disgusted'] = disgusted
        
        # Surprised: AU1 + AU2 + AU5 + AU26
        surprised = torch.zeros(self.num_blendshapes)
        surprised[[4, 5, 10, 11, 25, 26, 27]] = 0.9  # browOuterUp, eyeWide, jawOpen
        mappings['surprised'] = surprised
        
        # Neutral: minimal activation
        neutral = torch.zeros(self.num_blendshapes)
        mappings['neutral'] = neutral
        
        return mappings
    
    def get_blendshapes_for_emotion(self, 
                                   emotion_name: str, 
                                   intensity: float = 1.0) -> torch.Tensor:
        """Get blendshape coefficients for a specific emotion."""
        if emotion_name not in self.emotion_mappings:
            emotion_name = 'neutral'
        
        blendshapes = self.emotion_mappings[emotion_name] * intensity
        return blendshapes
    
    def blend_emotions(self,
                      emotion_probs: torch.Tensor,
                      emotion_labels: list) -> torch.Tensor:
        """
        Blend multiple emotions based on probabilities.
        
        Args:
            emotion_probs: [B, num_emotions] emotion probabilities
            emotion_labels: List of emotion names
        
        Returns:
            blendshapes: [B, num_blendshapes] blended blendshape coefficients
        """
        batch_size = emotion_probs.shape[0]
        device = emotion_probs.device
        
        blendshapes = torch.zeros(batch_size, self.num_blendshapes, device=device)
        
        for i, emotion_name in enumerate(emotion_labels):
            if emotion_name in self.emotion_mappings:
                emotion_bs = self.emotion_mappings[emotion_name].to(device)
                blendshapes += emotion_probs[:, i:i+1] * emotion_bs.unsqueeze(0)
        
        return blendshapes


class MelodyEmotionExtractor(nn.Module):
    """
    Extracts melody/prosody features and maps them to emotional content.
    
    This captures the "melody" of speech - pitch contours, rhythm patterns,
    and energy dynamics that convey emotion.
    """
    
    def __init__(self, output_dim: int = 64):
        super().__init__()
        
        # Pitch contour encoder
        self.pitch_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Rhythm encoder (from energy envelope)
        self.rhythm_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Melody fusion
        self.melody_fusion = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
    
    def extract_pitch(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract pitch contour from audio."""
        # Simplified - in practice use WORLD, CREPE, or PyYIN
        # This is a placeholder that extracts pitch-like features
        return audio.mean(dim=-1, keepdim=True)
    
    def extract_energy(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract energy envelope from audio."""
        energy = torch.sqrt(torch.mean(audio ** 2, dim=-1, keepdim=True))
        return energy
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract melody/prosody features from audio.
        
        Args:
            audio: [B, T] or [B, 1, T] raw audio waveform
        
        Returns:
            melody_features: [B, output_dim] melody/prosody features
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # [B, 1, T]
        
        # Extract pitch and energy
        pitch = self.extract_pitch(audio)  # [B, 1, T]
        energy = self.extract_energy(audio)  # [B, 1, T]
        
        # Encode pitch and rhythm
        pitch_feat = self.pitch_encoder(pitch).squeeze(-1)  # [B, 64]
        rhythm_feat = self.rhythm_encoder(energy).squeeze(-1)  # [B, 64]
        
        # Fuse melody features
        melody_features = torch.cat([pitch_feat, rhythm_feat], dim=-1)  # [B, 128]
        melody_features = self.melody_fusion(melody_features)  # [B, output_dim]
        
        return melody_features


# Utility functions for emotion-driven animation

def interpolate_emotions(emotion_a: str, emotion_b: str, 
                        alpha: float, controller: BlendshapeEmotionController) -> torch.Tensor:
    """
    Interpolate between two emotions.
    
    Args:
        emotion_a: First emotion name
        emotion_b: Second emotion name
        alpha: Interpolation factor (0 = emotion_a, 1 = emotion_b)
        controller: BlendshapeEmotionController instance
    
    Returns:
        interpolated_bs: Interpolated blendshape coefficients
    """
    bs_a = controller.get_blendshapes_for_emotion(emotion_a)
    bs_b = controller.get_blendshapes_for_emotion(emotion_b)
    
    return (1 - alpha) * bs_a + alpha * bs_b


def apply_emotion_to_blendshapes(blendshapes: torch.Tensor,
                                 emotion_adjustment: torch.Tensor,
                                 blend_mode: str = 'add') -> torch.Tensor:
    """
    Apply emotion adjustments to base blendshapes.
    
    Args:
        blendshapes: [B, num_bs] base blendshape coefficients
        emotion_adjustment: [B, num_bs] emotion-driven adjustments
        blend_mode: 'add', 'multiply', or 'replace'
    
    Returns:
        adjusted_bs: [B, num_bs] emotion-adjusted blendshapes
    """
    if blend_mode == 'add':
        return torch.clamp(blendshapes + emotion_adjustment, 0, 1)
    elif blend_mode == 'multiply':
        return blendshapes * (1 + emotion_adjustment)
    elif blend_mode == 'replace':
        return emotion_adjustment
    else:
        raise ValueError(f"Unknown blend_mode: {blend_mode}")


def create_emotion_dataset_from_audio(audio_dir: str, 
                                     emotion_labels_file: str) -> list:
    """
    Create emotion recognition training dataset from audio files.
    
    Args:
        audio_dir: Directory containing audio files
        emotion_labels_file: File containing emotion labels for each audio
    
    Returns:
        dataset: List of (audio_path, emotion_label) tuples
    """
    import os
    import json
    
    with open(emotion_labels_file, 'r') as f:
        labels = json.load(f)
    
    dataset = []
    for audio_file, emotion in labels.items():
        audio_path = os.path.join(audio_dir, audio_file)
        if os.path.exists(audio_path):
            dataset.append((audio_path, emotion))
    
    return dataset

