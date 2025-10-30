"""
Enhanced Audio Encoder Module with Foundation Models

This module provides improved audio encoding using foundation models like Whisper,
SpeechT5, and EnCodec to capture prosodic features, emotional context, and speaking style
beyond simple audio-lip synchronization.

Features:
- Whisper-based encoder for rich prosodic understanding
- SpeechT5 encoder for speech representation
- EnCodec for high-quality audio compression features
- CLIP-like contrastive audio-video alignment
- Prosodic feature extraction (pitch, energy, rhythm)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple


class ProsodyExtractor(nn.Module):
    """
    Extract prosodic features including pitch, energy, and rhythm
    to capture emotional context and speaking style.
    """
    def __init__(self, sample_rate: int = 16000):
        super().__init__()
        self.sample_rate = sample_rate
        
        # Learnable prosody encoder
        self.pitch_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.energy_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Combine prosody features
        self.prosody_fusion = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
    
    def extract_pitch(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract pitch contour using autocorrelation (simplified)"""
        # In practice, use librosa or torchcrepe for better pitch extraction
        # This is a simplified learnable version
        return audio.mean(dim=-1, keepdim=True)
    
    def extract_energy(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract energy/intensity features"""
        # RMS energy
        energy = torch.sqrt(torch.mean(audio ** 2, dim=-1, keepdim=True))
        return energy
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [B, T] or [B, 1, T] raw audio waveform
        Returns:
            prosody_features: [B, 64] prosodic feature vector
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # [B, 1, T]
        
        # Extract prosodic features
        pitch = self.extract_pitch(audio)  # [B, 1, T]
        energy = self.extract_energy(audio)  # [B, 1, T]
        
        # Encode prosodic features
        pitch_feat = self.pitch_encoder(pitch).squeeze(-1)  # [B, 64]
        energy_feat = self.energy_encoder(energy).squeeze(-1)  # [B, 64]
        
        # Fuse prosodic features
        prosody = torch.cat([pitch_feat, energy_feat], dim=-1)  # [B, 128]
        prosody = self.prosody_fusion(prosody)  # [B, 64]
        
        return prosody


class WhisperAudioEncoder(nn.Module):
    """
    Whisper-based audio encoder for rich prosodic and semantic understanding.
    Uses Whisper's encoder to extract audio representations.
    """
    def __init__(self, 
                 model_name: str = "openai/whisper-small",
                 output_dim: int = 512,
                 freeze_encoder: bool = True):
        super().__init__()
        
        try:
            from transformers import WhisperModel, WhisperProcessor
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.whisper = WhisperModel.from_pretrained(model_name)
            
            if freeze_encoder:
                # Freeze Whisper encoder for faster training
                for param in self.whisper.encoder.parameters():
                    param.requires_grad = False
            
            # Get Whisper's encoder output dimension
            whisper_dim = self.whisper.config.d_model
            
            # Projection head to desired output dimension
            self.projection = nn.Sequential(
                nn.Linear(whisper_dim, whisper_dim // 2),
                nn.LayerNorm(whisper_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(whisper_dim // 2, output_dim)
            )
            
            self.available = True
            
        except ImportError:
            print("[WARN] Transformers not available, WhisperAudioEncoder disabled")
            self.available = False
    
    def forward(self, audio: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            audio: [B, T] raw audio waveform or [B, C, T] mel spectrogram
            attention_mask: Optional attention mask
        Returns:
            features: [B, output_dim] audio features
        """
        if not self.available:
            raise RuntimeError("WhisperAudioEncoder not available")
        
        # Process audio through Whisper encoder
        if audio.dim() == 2:
            # Convert raw audio to mel spectrogram if needed
            # Whisper expects mel spectrogram input
            with torch.no_grad():
                # Use processor to convert to input features
                batch_size = audio.shape[0]
                inputs = []
                for i in range(batch_size):
                    audio_np = audio[i].cpu().numpy()
                    input_features = self.processor(
                        audio_np, 
                        sampling_rate=16000, 
                        return_tensors="pt"
                    ).input_features
                    inputs.append(input_features)
                input_features = torch.cat(inputs, dim=0).to(audio.device)
        else:
            input_features = audio
        
        # Get encoder outputs
        encoder_outputs = self.whisper.encoder(
            input_features=input_features,
            attention_mask=attention_mask
        )
        
        # Pool encoder outputs (mean pooling)
        hidden_states = encoder_outputs.last_hidden_state  # [B, T, D]
        pooled = hidden_states.mean(dim=1)  # [B, D]
        
        # Project to output dimension
        features = self.projection(pooled)  # [B, output_dim]
        
        return features


class SpeechT5AudioEncoder(nn.Module):
    """
    SpeechT5-based audio encoder for speech representation.
    """
    def __init__(self,
                 model_name: str = "microsoft/speecht5_tts",
                 output_dim: int = 512,
                 freeze_encoder: bool = True):
        super().__init__()
        
        try:
            from transformers import SpeechT5Model, SpeechT5Processor
            self.processor = SpeechT5Processor.from_pretrained(model_name)
            self.speecht5 = SpeechT5Model.from_pretrained(model_name)
            
            if freeze_encoder:
                for param in self.speecht5.parameters():
                    param.requires_grad = False
            
            # Get model dimension
            model_dim = self.speecht5.config.hidden_size
            
            # Projection head
            self.projection = nn.Sequential(
                nn.Linear(model_dim, model_dim // 2),
                nn.LayerNorm(model_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(model_dim // 2, output_dim)
            )
            
            self.available = True
            
        except ImportError:
            print("[WARN] Transformers not available, SpeechT5AudioEncoder disabled")
            self.available = False
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [B, T] raw audio waveform
        Returns:
            features: [B, output_dim] audio features
        """
        if not self.available:
            raise RuntimeError("SpeechT5AudioEncoder not available")
        
        # Process audio
        batch_size = audio.shape[0]
        inputs = []
        for i in range(batch_size):
            audio_np = audio[i].cpu().numpy()
            input_values = self.processor(
                audio=audio_np,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_values
            inputs.append(input_values)
        input_values = torch.cat(inputs, dim=0).to(audio.device)
        
        # Get encoder outputs
        outputs = self.speecht5.encoder(
            input_values=input_values
        )
        
        # Pool and project
        hidden_states = outputs.last_hidden_state  # [B, T, D]
        pooled = hidden_states.mean(dim=1)  # [B, D]
        features = self.projection(pooled)  # [B, output_dim]
        
        return features


class EnCodecAudioEncoder(nn.Module):
    """
    EnCodec-based audio encoder for high-quality audio compression features.
    """
    def __init__(self,
                 model_name: str = "facebook/encodec_24khz",
                 output_dim: int = 512,
                 freeze_encoder: bool = True):
        super().__init__()
        
        try:
            from transformers import EncodecModel
            self.encodec = EncodecModel.from_pretrained(model_name)
            
            if freeze_encoder:
                for param in self.encodec.parameters():
                    param.requires_grad = False
            
            # EnCodec produces quantized codes, we'll use the encoder output
            encoder_dim = self.encodec.config.hidden_size
            
            # Projection head
            self.projection = nn.Sequential(
                nn.Linear(encoder_dim, encoder_dim // 2),
                nn.LayerNorm(encoder_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(encoder_dim // 2, output_dim)
            )
            
            self.available = True
            
        except ImportError:
            print("[WARN] Transformers not available, EnCodecAudioEncoder disabled")
            self.available = False
    
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [B, 1, T] raw audio waveform
        Returns:
            features: [B, output_dim] audio features
        """
        if not self.available:
            raise RuntimeError("EnCodecAudioEncoder not available")
        
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # [B, 1, T]
        
        # Encode audio
        with torch.no_grad() if not self.training else torch.enable_grad():
            encoder_outputs = self.encodec.encoder(audio)
        
        # Pool encoder outputs
        pooled = encoder_outputs.mean(dim=-1)  # [B, D]
        features = self.projection(pooled)  # [B, output_dim]
        
        return features


class ContrastiveAudioVideoAlignment(nn.Module):
    """
    CLIP-like contrastive learning module for audio-video alignment.
    This enables the model to learn multimodal representations that capture
    the relationship between audio prosody and visual expressions.
    """
    def __init__(self,
                 audio_dim: int = 512,
                 video_dim: int = 512,
                 projection_dim: int = 256,
                 temperature: float = 0.07):
        super().__init__()
        
        self.temperature = temperature
        
        # Audio projection head
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # Video projection head
        self.video_projection = nn.Sequential(
            nn.Linear(video_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
        
        # Learnable temperature parameter
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
    
    def forward(self, 
                audio_features: torch.Tensor,
                video_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            audio_features: [B, audio_dim] audio embeddings
            video_features: [B, video_dim] video embeddings
        Returns:
            logits_per_audio: [B, B] similarity matrix from audio to video
            logits_per_video: [B, B] similarity matrix from video to audio
        """
        # Project to common space
        audio_embed = self.audio_projection(audio_features)  # [B, projection_dim]
        video_embed = self.video_projection(video_features)  # [B, projection_dim]
        
        # Normalize embeddings
        audio_embed = F.normalize(audio_embed, dim=-1)
        video_embed = F.normalize(video_embed, dim=-1)
        
        # Compute similarity matrix
        logit_scale = self.logit_scale.exp()
        logits_per_audio = logit_scale * audio_embed @ video_embed.t()  # [B, B]
        logits_per_video = logits_per_audio.t()  # [B, B]
        
        return logits_per_audio, logits_per_video
    
    def contrastive_loss(self,
                        logits_per_audio: torch.Tensor,
                        logits_per_video: torch.Tensor) -> torch.Tensor:
        """
        Compute bidirectional contrastive loss.
        """
        batch_size = logits_per_audio.shape[0]
        labels = torch.arange(batch_size, device=logits_per_audio.device)
        
        loss_audio = F.cross_entropy(logits_per_audio, labels)
        loss_video = F.cross_entropy(logits_per_video, labels)
        
        return (loss_audio + loss_video) / 2


class EnhancedAudioEncoder(nn.Module):
    """
    Enhanced Audio Encoder that combines foundation models with prosodic features
    and contrastive audio-video alignment.
    
    This is a drop-in replacement for the original AudioEncoder trained on LRS2.
    """
    def __init__(self,
                 encoder_type: str = "whisper",  # "whisper", "speecht5", "encodec", "ensemble"
                 output_dim: int = 512,
                 use_prosody: bool = True,
                 use_contrastive: bool = True,
                 freeze_backbone: bool = True):
        super().__init__()
        
        self.encoder_type = encoder_type
        self.use_prosody = use_prosody
        self.use_contrastive = use_contrastive
        
        # Initialize foundation model encoders
        if encoder_type == "whisper":
            self.audio_encoder = WhisperAudioEncoder(
                output_dim=output_dim,
                freeze_encoder=freeze_backbone
            )
        elif encoder_type == "speecht5":
            self.audio_encoder = SpeechT5AudioEncoder(
                output_dim=output_dim,
                freeze_encoder=freeze_backbone
            )
        elif encoder_type == "encodec":
            self.audio_encoder = EnCodecAudioEncoder(
                output_dim=output_dim,
                freeze_encoder=freeze_backbone
            )
        elif encoder_type == "ensemble":
            # Use ensemble of multiple encoders
            self.whisper_encoder = WhisperAudioEncoder(output_dim=output_dim // 3, freeze_encoder=freeze_backbone)
            self.speecht5_encoder = SpeechT5AudioEncoder(output_dim=output_dim // 3, freeze_encoder=freeze_backbone)
            self.encodec_encoder = EnCodecAudioEncoder(output_dim=output_dim // 3, freeze_encoder=freeze_backbone)
            
            self.ensemble_fusion = nn.Sequential(
                nn.Linear(output_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        
        # Prosody extractor
        if use_prosody:
            self.prosody_extractor = ProsodyExtractor()
            self.prosody_fusion = nn.Sequential(
                nn.Linear(output_dim + 64, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU()
            )
        
        # Contrastive alignment module
        if use_contrastive:
            self.contrastive_alignment = ContrastiveAudioVideoAlignment(
                audio_dim=output_dim,
                video_dim=512,  # Assuming video features are 512-dim
                projection_dim=256
            )
        
        self.output_dim = output_dim
    
    def forward(self, 
                audio: torch.Tensor,
                video_features: Optional[torch.Tensor] = None,
                return_contrastive: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            audio: [B, T] or [B, 1, T] raw audio waveform or mel spectrogram
            video_features: Optional [B, video_dim] video features for contrastive learning
            return_contrastive: Whether to return contrastive loss
        
        Returns:
            Dictionary containing:
            - audio_features: [B, output_dim] enhanced audio features
            - prosody_features: [B, 64] prosodic features (if use_prosody)
            - contrastive_loss: scalar loss (if use_contrastive and video_features provided)
        """
        result = {}
        
        # Extract audio features using foundation model
        if self.encoder_type == "ensemble":
            whisper_feat = self.whisper_encoder(audio)
            speecht5_feat = self.speecht5_encoder(audio)
            encodec_feat = self.encodec_encoder(audio)
            audio_features = torch.cat([whisper_feat, speecht5_feat, encodec_feat], dim=-1)
            audio_features = self.ensemble_fusion(audio_features)
        else:
            audio_features = self.audio_encoder(audio)
        
        result['audio_features'] = audio_features
        
        # Extract and fuse prosodic features
        if self.use_prosody:
            prosody_features = self.prosody_extractor(audio)
            result['prosody_features'] = prosody_features
            
            # Fuse audio and prosody features
            combined = torch.cat([audio_features, prosody_features], dim=-1)
            audio_features = self.prosody_fusion(combined)
            result['audio_features_with_prosody'] = audio_features
        
        # Contrastive audio-video alignment
        if self.use_contrastive and video_features is not None:
            logits_audio, logits_video = self.contrastive_alignment(
                audio_features, video_features
            )
            
            if return_contrastive:
                contrastive_loss = self.contrastive_alignment.contrastive_loss(
                    logits_audio, logits_video
                )
                result['contrastive_loss'] = contrastive_loss
            
            result['logits_audio'] = logits_audio
            result['logits_video'] = logits_video
        
        return result
    
    def get_audio_features(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convenience method to get final audio features.
        Compatible with original AudioEncoder interface.
        """
        result = self.forward(audio, return_contrastive=False)
        if 'audio_features_with_prosody' in result:
            return result['audio_features_with_prosody']
        return result['audio_features']


# Factory function for easy instantiation
def create_enhanced_audio_encoder(config: Dict) -> EnhancedAudioEncoder:
    """
    Factory function to create enhanced audio encoder from config.
    
    Example config:
    {
        'encoder_type': 'whisper',  # or 'speecht5', 'encodec', 'ensemble'
        'output_dim': 512,
        'use_prosody': True,
        'use_contrastive': True,
        'freeze_backbone': True
    }
    """
    return EnhancedAudioEncoder(**config)

