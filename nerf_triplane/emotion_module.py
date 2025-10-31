"""
Emotion/Melody Embedding Module for SyncTalk

This module implements Speech Emotion Recognition (SER) and integrates emotion
embeddings into blendshapes for emotion-sensitive facial expressions.

References:
- EmoTalk: Speech-Driven Emotional 3D Face Animation
- AffectNet: Facial Expression Recognition Dataset
- Wav2Vec2 + Emotion Fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List


class EmotionRecognitionModule(nn.Module):
    """
    Speech Emotion Recognition (SER) module that extracts emotion embeddings from audio.
    
    Supports multiple emotion recognition approaches:
    1. Wav2Vec2-based emotion recognition
    2. Custom CNN-based emotion extractor
    3. Prosody-based emotion inference
    
    Emotions: Neutral, Happy, Sad, Angry, Fearful, Disgusted, Surprised
    """
    
    def __init__(self,
                 emotion_model: str = 'wav2vec2',  # 'wav2vec2', 'cnn', 'prosody'
                 num_emotions: int = 7,
                 emotion_dim: int = 64,
                 use_pretrained: bool = True):
        super().__init__()
        
        self.emotion_model = emotion_model
        self.num_emotions = num_emotions
        self.emotion_dim = emotion_dim
        
        # Emotion labels
        self.emotion_labels = ['neutral', 'happy', 'sad', 'angry', 'fearful', 'disgusted', 'surprised']
        
        if emotion_model == 'wav2vec2':
            self._init_wav2vec2_emotion(use_pretrained)
        elif emotion_model == 'cnn':
            self._init_cnn_emotion()
        elif emotion_model == 'prosody':
            self._init_prosody_emotion()
        else:
            raise ValueError(f"Unknown emotion model: {emotion_model}")
        
        # Emotion embedding layer
        self.emotion_embedding = nn.Embedding(num_emotions, emotion_dim)
        
        # Continuous emotion encoder (for soft predictions)
        self.emotion_encoder = nn.Sequential(
            nn.Linear(num_emotions, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, emotion_dim)
        )
    
    def _init_wav2vec2_emotion(self, use_pretrained: bool):
        """Initialize Wav2Vec2-based emotion recognition."""
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Processor
            
            if use_pretrained:
                # Use pre-trained emotion recognition model
                model_name = "ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"
                try:
                    self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
                    self.processor = Wav2Vec2Processor.from_pretrained(model_name)
                    print(f"[INFO] Loaded pretrained emotion model: {model_name}")
                except:
                    # Fallback to base model
                    print("[WARN] Could not load emotion model, using base Wav2Vec2")
                    self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
                    self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            else:
                self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
                self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            
            # Emotion classification head
            wav2vec_dim = self.wav2vec2.config.hidden_size
            self.emotion_head = nn.Sequential(
                nn.Linear(wav2vec_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, self.num_emotions)
            )
            
            self.available = True
            
        except ImportError:
            print("[WARN] Transformers not available for Wav2Vec2 emotion recognition")
            self.available = False
    
    def _init_cnn_emotion(self):
        """Initialize CNN-based emotion recognition from mel spectrogram."""
        self.emotion_cnn = nn.Sequential(
            # Input: [B, 1, mel_bins, time]
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.emotion_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_emotions)
        )
        
        self.available = True
    
    def _init_prosody_emotion(self):
        """Initialize prosody-based emotion inference."""
        # Extract prosody features: pitch, energy, speaking rate, etc.
        self.prosody_extractor = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Prosody to emotion mapping
        # High pitch + high energy = happy/excited
        # Low pitch + low energy = sad
        # High energy + sharp changes = angry
        self.emotion_head = nn.Sequential(
            nn.Linear(64 * 3, 128),  # 3 prosody features: pitch, energy, rate
            nn.ReLU(),
            nn.Linear(128, self.num_emotions)
        )
        
        self.available = True
    
    def extract_emotion_from_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract emotion predictions from audio.
        
        Args:
            audio: [B, T] or [B, 1, T] raw audio waveform or [B, 1, H, W] mel spectrogram
        
        Returns:
            emotion_probs: [B, num_emotions] emotion probabilities
            emotion_logits: [B, num_emotions] raw emotion logits
        """
        if self.emotion_model == 'wav2vec2':
            return self._extract_wav2vec2_emotion(audio)
        elif self.emotion_model == 'cnn':
            return self._extract_cnn_emotion(audio)
        elif self.emotion_model == 'prosody':
            return self._extract_prosody_emotion(audio)
    
    def _extract_wav2vec2_emotion(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract emotion using Wav2Vec2."""
        if audio.dim() == 3:
            audio = audio.squeeze(1)  # [B, T]
        
        # Process through Wav2Vec2
        outputs = self.wav2vec2(audio)
        hidden_states = outputs.last_hidden_state  # [B, T, D]
        
        # Pool over time
        pooled = hidden_states.mean(dim=1)  # [B, D]
        
        # Emotion classification
        emotion_logits = self.emotion_head(pooled)  # [B, num_emotions]
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        
        return emotion_probs, emotion_logits
    
    def _extract_cnn_emotion(self, mel: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract emotion using CNN from mel spectrogram."""
        if mel.dim() == 2:
            mel = mel.unsqueeze(0).unsqueeze(0)  # [B, 1, H, W]
        elif mel.dim() == 3:
            mel = mel.unsqueeze(1)  # [B, 1, H, W]
        
        # CNN feature extraction
        features = self.emotion_cnn(mel)  # [B, 512, 1, 1]
        features = features.squeeze(-1).squeeze(-1)  # [B, 512]
        
        # Emotion classification
        emotion_logits = self.emotion_head(features)  # [B, num_emotions]
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        
        return emotion_probs, emotion_logits
    
    def _extract_prosody_emotion(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract emotion from prosody features."""
        # Extract pitch, energy, speaking rate
        # Simplified version - in practice use librosa or parselmouth
        
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # [B, 1, T]
        
        # Extract prosody features (simplified)
        pitch_feat = self.prosody_extractor(audio)  # [B, 64, 1]
        energy_feat = torch.sqrt(torch.mean(audio ** 2, dim=-1, keepdim=True))
        energy_feat = self.prosody_extractor(energy_feat)
        
        # Speaking rate (zero crossing rate approximation)
        rate = torch.abs(audio[:, :, 1:] - audio[:, :, :-1]).mean(dim=-1, keepdim=True)
        rate_feat = self.prosody_extractor(rate)
        
        # Combine prosody features
        prosody_features = torch.cat([
            pitch_feat.squeeze(-1),
            energy_feat.squeeze(-1),
            rate_feat.squeeze(-1)
        ], dim=-1)
        
        # Map to emotions
        emotion_logits = self.emotion_head(prosody_features)
        emotion_probs = F.softmax(emotion_logits, dim=-1)
        
        return emotion_probs, emotion_logits
    
    def forward(self, audio: torch.Tensor, return_probs: bool = True) -> Dict[str, torch.Tensor]:
        """
        Forward pass: extract emotion embeddings from audio.
        
        Args:
            audio: Input audio tensor
            return_probs: Whether to return emotion probabilities
        
        Returns:
            Dictionary containing:
            - emotion_embedding: [B, emotion_dim] continuous emotion representation
            - emotion_probs: [B, num_emotions] emotion probabilities
            - emotion_class: [B] predicted emotion class
        """
        # Extract emotion predictions
        emotion_probs, emotion_logits = self.extract_emotion_from_audio(audio)
        
        # Get emotion class
        emotion_class = torch.argmax(emotion_probs, dim=-1)  # [B]
        
        # Get continuous emotion embedding
        emotion_embedding = self.emotion_encoder(emotion_probs)  # [B, emotion_dim]
        
        result = {
            'emotion_embedding': emotion_embedding,
            'emotion_probs': emotion_probs,
            'emotion_logits': emotion_logits,
            'emotion_class': emotion_class,
        }
        
        return result


class EmotionBlendshapeMapper(nn.Module):
    """
    Maps emotion embeddings to blendshape coefficients for emotion-sensitive
    facial expressions.
    
    This integrates emotion information into the blendshape system to create
    natural emotional expressions that synchronize with voice.
    """
    
    def __init__(self,
                 emotion_dim: int = 64,
                 num_blendshapes: int = 52,  # Standard ARKit blendshapes
                 hidden_dim: int = 128):
        super().__init__()
        
        self.emotion_dim = emotion_dim
        self.num_blendshapes = num_blendshapes
        
        # Emotion to blendshape mapping network
        self.mapper = nn.Sequential(
            nn.Linear(emotion_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_blendshapes),
            nn.Sigmoid()  # Blendshape weights in [0, 1]
        )
        
        # Emotion-specific blendshape templates
        # These define which blendshapes are most relevant for each emotion
        self.register_emotion_templates()
    
    def register_emotion_templates(self):
        """Register emotion-specific blendshape templates."""
        # Emotion templates: which blendshapes are activated for each emotion
        # Based on FACS (Facial Action Coding System) and AffectNet
        
        templates = torch.zeros(7, 52)  # [num_emotions, num_blendshapes]
        
        # Happy: mouth smile, cheek raise, eye squint
        templates[1, [12, 13, 6, 7, 48, 49]] = 1.0  # mouthSmile, cheekPuff, eyeSquint
        
        # Sad: mouth frown, inner brow raise, eye close
        templates[2, [14, 15, 0, 1, 8, 9]] = 1.0  # mouthFrown, browDown, eyeBlink
        
        # Angry: brow down, jaw forward, mouth press
        templates[3, [2, 3, 23, 14, 15]] = 1.0  # browDownLeft, browDownRight, jawForward
        
        # Fearful: brow raise, eye wide, mouth open
        templates[4, [4, 5, 10, 11, 25, 26]] = 1.0  # browOuterUp, eyeWide, mouthOpen
        
        # Disgusted: nose wrinkle, mouth upper up
        templates[5, [8, 16, 17]] = 1.0  # noseSneer, mouthUpperUp
        
        # Surprised: brow raise, eye wide, jaw drop
        templates[6, [4, 5, 10, 11, 25, 26, 27]] = 1.0  # browOuterUp, eyeWide, jawOpen
        
        self.register_buffer('emotion_templates', templates)
    
    def forward(self, emotion_embedding: torch.Tensor, 
                base_blendshapes: Optional[torch.Tensor] = None,
                emotion_strength: float = 1.0) -> torch.Tensor:
        """
        Map emotion embedding to blendshape coefficients.
        
        Args:
            emotion_embedding: [B, emotion_dim] emotion features
            base_blendshapes: [B, num_blendshapes] base blendshapes (optional)
            emotion_strength: Strength of emotion influence (0-1)
        
        Returns:
            blendshapes: [B, num_blendshapes] emotion-modulated blendshapes
        """
        # Map emotion to blendshape adjustments
        emotion_blendshapes = self.mapper(emotion_embedding)  # [B, num_blendshapes]
        
        # Apply emotion strength
        emotion_blendshapes = emotion_blendshapes * emotion_strength
        
        # Combine with base blendshapes if provided
        if base_blendshapes is not None:
            # Additive combination with clamping
            blendshapes = torch.clamp(base_blendshapes + emotion_blendshapes, 0, 1)
        else:
            blendshapes = emotion_blendshapes
        
        return blendshapes


class EmotionConditionedAudioNet(nn.Module):
    """
    Audio network conditioned on emotion embeddings.
    
    This replaces the standard AudioNet to integrate emotion information
    directly into the audio feature processing pipeline.
    """
    
    def __init__(self,
                 dim_in: int = 29,
                 dim_aud: int = 64,
                 emotion_dim: int = 64,
                 win_size: int = 16):
        super().__init__()
        
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.emotion_dim = emotion_dim
        
        # Audio encoder
        self.audio_conv = nn.Sequential(
            nn.Conv1d(dim_in, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.02, True),
        )
        
        # Emotion-conditioned fusion
        self.emotion_fusion = nn.Sequential(
            nn.Linear(64 + emotion_dim, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.02, True),
            nn.Linear(128, dim_aud)
        )
    
    def forward(self, audio_features: torch.Tensor, 
                emotion_embedding: torch.Tensor) -> torch.Tensor:
        """
        Process audio features conditioned on emotion.
        
        Args:
            audio_features: [B, dim_in, win_size] audio features
            emotion_embedding: [B, emotion_dim] emotion embedding
        
        Returns:
            features: [B, dim_aud] emotion-conditioned audio features
        """
        # Audio encoding
        half_w = int(self.win_size / 2)
        x = audio_features[:, :, 8-half_w:8+half_w]
        x = self.audio_conv(x).squeeze(-1)  # [B, 64]
        
        # Fuse with emotion
        combined = torch.cat([x, emotion_embedding], dim=-1)  # [B, 64 + emotion_dim]
        features = self.emotion_fusion(combined)  # [B, dim_aud]
        
        return features


def load_emotion_dataset(dataset_name: str = 'ravdess'):
    """
    Load emotion dataset for training.
    
    Supported datasets:
    - RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song
    - CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset
    - IEMOCAP: Interactive Emotional Dyadic Motion Capture
    """
    raise NotImplementedError("Dataset loading to be implemented based on available data")


def train_emotion_recognition(model: EmotionRecognitionModule,
                              train_loader,
                              val_loader,
                              num_epochs: int = 50,
                              lr: float = 1e-4):
    """
    Train the emotion recognition module.
    
    Args:
        model: EmotionRecognitionModule to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        lr: Learning rate
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        # Training loop
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            audio, emotion_labels = batch
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(audio)
            loss = criterion(outputs['emotion_logits'], emotion_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_accuracy = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                audio, emotion_labels = batch
                outputs = model(audio)
                predictions = outputs['emotion_class']
                val_accuracy += (predictions == emotion_labels).float().mean()
        
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {train_loss:.4f}, Val Acc: {val_accuracy:.4f}")

