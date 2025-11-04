"""
AV-HuBERT and Wav2Vec2 Feature Extraction for Lip-Sync
This script extracts rich phoneme-level features from audio for better lip synchronization.

AV-HuBERT: Audio-Visual HuBERT trained on audio-visual speech (LRS3)
Wav2Vec2: Self-supervised speech representation learning

Usage:
    python data_utils/avhubert_wav2vec2.py --wav path/to/audio.wav --model avhubert
    python data_utils/avhubert_wav2vec2.py --wav path/to/audio.wav --model wav2vec2
"""

import torch
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa
from argparse import ArgumentParser
from typing import Optional, Tuple

# Try importing transformers for Wav2Vec2
try:
    from transformers import Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2ForCTC
    WAV2VEC2_AVAILABLE = True
except ImportError:
    print("[WARN] transformers not available, Wav2Vec2 disabled")
    WAV2VEC2_AVAILABLE = False

# Try importing fairseq for AV-HuBERT
try:
    import fairseq
    FAIRSEQ_AVAILABLE = True
except ImportError:
    print("[WARN] fairseq not available, AV-HuBERT disabled")
    FAIRSEQ_AVAILABLE = False


class Wav2Vec2FeatureExtractor:
    """
    Extract phoneme-rich features using Wav2Vec2 for improved lip-sync.
    
    Wav2Vec2 provides better phoneme representations than DeepSpeech,
    leading to more accurate lip synchronization.
    """
    
    def __init__(self, model_name: str = "facebook/wav2vec2-large-960h-lv60-self", device: str = "cuda:0"):
        """
        Args:
            model_name: HuggingFace model name
            device: Device to run model on
        """
        if not WAV2VEC2_AVAILABLE:
            raise ImportError("transformers not available. Install with: pip install transformers")
        
        print(f"[INFO] Loading Wav2Vec2 model: {model_name}")
        self.device = device
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        self.model.eval()
        print("[INFO] Wav2Vec2 model loaded successfully")
    
    @torch.no_grad()
    def extract_features(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract Wav2Vec2 features from audio.
        
        Args:
            audio: Audio waveform [T] or [T, C]
            sr: Sample rate (should be 16000 for Wav2Vec2)
        
        Returns:
            features: [N, 1024] where N is the number of frames
        """
        # Handle stereo audio
        if audio.ndim == 2:
            audio = audio[:, 0]  # Take first channel
        
        # Process audio through Wav2Vec2
        # Wav2Vec2 has a stride of 20ms (320 samples at 16kHz)
        input_values = self.processor(audio, return_tensors="pt", sampling_rate=sr).input_values
        input_values = input_values.to(self.device)
        
        # Process in chunks for long audio
        kernel = 400  # Wav2Vec2 kernel size
        stride = 320  # Wav2Vec2 stride
        clip_length = stride * 1000  # Process ~6.4 seconds at a time
        
        num_iter = input_values.shape[1] // clip_length
        res_lst = []
        
        for i in range(num_iter):
            if i == 0:
                start_idx = 0
                end_idx = clip_length - stride + kernel
            else:
                start_idx = clip_length * i
                end_idx = start_idx + (clip_length - stride + kernel)
            
            chunk = input_values[:, start_idx:end_idx]
            outputs = self.model(chunk)
            features = outputs.last_hidden_state  # [1, T, 1024]
            res_lst.append(features[0])
        
        # Process remaining audio
        if num_iter > 0:
            chunk = input_values[:, clip_length * num_iter:]
        else:
            chunk = input_values
        
        if chunk.shape[1] >= kernel:
            outputs = self.model(chunk)
            features = outputs.last_hidden_state  # [1, T, 1024]
            res_lst.append(features[0])
        
        # Concatenate all chunks
        features = torch.cat(res_lst, dim=0).cpu().numpy()  # [T, 1024]
        
        return features


class AVHuBERTFeatureExtractor:
    """
    Extract audio-visual features using AV-HuBERT for superior lip-sync.
    
    AV-HuBERT is trained on audio-visual data (LRS3) and provides better
    alignment between speech and visual features compared to audio-only models.
    
    Note: This requires fairseq and the AV-HuBERT checkpoint.
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "cuda:0"):
        """
        Args:
            checkpoint_path: Path to AV-HuBERT checkpoint (.pt file)
            device: Device to run model on
        """
        if not FAIRSEQ_AVAILABLE:
            raise ImportError("fairseq not available. Install with: pip install fairseq")
        
        self.device = device
        
        # If no checkpoint provided, try to load from default location
        if checkpoint_path is None:
            checkpoint_path = "checkpoints/avhubert_large_lrs3.pt"
        
        print(f"[INFO] Loading AV-HuBERT from: {checkpoint_path}")
        
        try:
            # Load AV-HuBERT model using fairseq
            models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
            self.model = models[0].to(device)
            self.model.eval()
            self.task = task
            print("[INFO] AV-HuBERT model loaded successfully")
        except Exception as e:
            print(f"[ERROR] Failed to load AV-HuBERT: {e}")
            print("[INFO] You can download AV-HuBERT from: https://github.com/facebookresearch/av_hubert")
            raise
    
    @torch.no_grad()
    def extract_features(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Extract AV-HuBERT features from audio.
        
        Args:
            audio: Audio waveform [T] or [T, C]
            sr: Sample rate (should be 16000 for AV-HuBERT)
        
        Returns:
            features: [N, 1024] where N is the number of frames
        """
        # Handle stereo audio
        if audio.ndim == 2:
            audio = audio[:, 0]
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)  # [1, T]
        
        # Process through AV-HuBERT (audio-only mode)
        # AV-HuBERT can work with audio-only input
        padding_mask = torch.zeros(1, audio_tensor.shape[1], dtype=torch.bool, device=self.device)
        
        # Process in chunks for long audio (similar to HuBERT)
        kernel = 400
        stride = 320
        clip_length = stride * 1000
        
        num_iter = audio_tensor.shape[1] // clip_length
        res_lst = []
        
        for i in range(num_iter):
            if i == 0:
                start_idx = 0
                end_idx = clip_length - stride + kernel
            else:
                start_idx = clip_length * i
                end_idx = start_idx + (clip_length - stride + kernel)
            
            chunk = audio_tensor[:, start_idx:end_idx]
            chunk_padding = padding_mask[:, start_idx:end_idx]
            
            # Extract features
            features, _ = self.model.extract_features(
                source=chunk,
                padding_mask=chunk_padding,
                mask=False
            )
            res_lst.append(features[0])
        
        # Process remaining audio
        if num_iter > 0:
            chunk = audio_tensor[:, clip_length * num_iter:]
            chunk_padding = padding_mask[:, clip_length * num_iter:]
        else:
            chunk = audio_tensor
            chunk_padding = padding_mask
        
        if chunk.shape[1] >= kernel:
            features, _ = self.model.extract_features(
                source=chunk,
                padding_mask=chunk_padding,
                mask=False
            )
            res_lst.append(features[0])
        
        # Concatenate all chunks
        features = torch.cat(res_lst, dim=0).cpu().numpy()  # [T, 1024]
        
        return features


def make_even_first_dim(tensor: np.ndarray) -> np.ndarray:
    """Make first dimension even by truncating if necessary."""
    if tensor.shape[0] % 2 == 1:
        return tensor[:-1]
    return tensor


def align_features_to_video_fps(features: np.ndarray, 
                                 audio_sr: int = 16000,
                                 feature_fps: float = 50.0,
                                 video_fps: float = 25.0) -> np.ndarray:
    """
    Align audio features to video frame rate.
    
    Args:
        features: Audio features [T, D]
        audio_sr: Audio sample rate
        feature_fps: Feature extraction rate (frames per second)
        video_fps: Target video frame rate
    
    Returns:
        aligned_features: Features aligned to video FPS [N, D]
    """
    # Wav2Vec2 and AV-HuBERT have a stride of 320 samples at 16kHz
    # This gives 50 fps (16000 / 320 = 50)
    
    # If feature fps matches video fps, no resampling needed
    if feature_fps == video_fps:
        return features
    
    # Resample features to match video fps
    num_video_frames = int(features.shape[0] * video_fps / feature_fps)
    
    # Use linear interpolation for resampling
    indices = np.linspace(0, features.shape[0] - 1, num_video_frames)
    resampled_features = []
    
    for i in indices:
        i_low = int(np.floor(i))
        i_high = min(int(np.ceil(i)), features.shape[0] - 1)
        weight = i - i_low
        
        if i_low == i_high:
            resampled_features.append(features[i_low])
        else:
            interpolated = (1 - weight) * features[i_low] + weight * features[i_high]
            resampled_features.append(interpolated)
    
    return np.array(resampled_features)


def extract_and_save_features(wav_path: str, 
                              model_type: str = "wav2vec2",
                              output_path: Optional[str] = None,
                              video_fps: float = 25.0,
                              avhubert_checkpoint: Optional[str] = None) -> str:
    """
    Extract features from audio and save to .npy file.
    
    Args:
        wav_path: Path to audio file (.wav)
        model_type: "wav2vec2" or "avhubert"
        output_path: Output path for .npy file (default: same as wav_path)
        video_fps: Target video frame rate for alignment
        avhubert_checkpoint: Path to AV-HuBERT checkpoint (if using avhubert)
    
    Returns:
        output_path: Path to saved .npy file
    """
    # Load audio
    print(f"[INFO] Loading audio from: {wav_path}")
    speech, sr = sf.read(wav_path)
    
    # Resample to 16kHz if needed
    if sr != 16000:
        print(f"[INFO] Resampling from {sr}Hz to 16000Hz")
        speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Extract features based on model type
    if model_type == "wav2vec2":
        extractor = Wav2Vec2FeatureExtractor()
        features = extractor.extract_features(speech, sr)
        suffix = "_w2v2"
    elif model_type == "avhubert":
        extractor = AVHuBERTFeatureExtractor(checkpoint_path=avhubert_checkpoint)
        features = extractor.extract_features(speech, sr)
        suffix = "_avhub"
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'wav2vec2' or 'avhubert'")
    
    print(f"[INFO] Extracted features shape: {features.shape}")
    
    # Align features to video FPS
    # Both Wav2Vec2 and AV-HuBERT have ~50 fps (stride of 320 at 16kHz)
    feature_fps = 16000 / 320  # 50 fps
    features_aligned = align_features_to_video_fps(features, sr, feature_fps, video_fps)
    print(f"[INFO] Aligned features shape: {features_aligned.shape} (target fps: {video_fps})")
    
    # Reshape to match expected format: [N/2, 2, D] for compatibility
    # This matches the HuBERT format used in the codebase
    features_aligned = make_even_first_dim(features_aligned)
    features_aligned = features_aligned.reshape(-1, 2, features_aligned.shape[-1])
    
    # Save features
    if output_path is None:
        output_path = wav_path.replace('.wav', f'{suffix}.npy')
    
    np.save(output_path, features_aligned)
    print(f"[INFO] Saved features to: {output_path}")
    print(f"[INFO] Final features shape: {features_aligned.shape}")
    
    return output_path


if __name__ == '__main__':
    parser = ArgumentParser(description="Extract AV-HuBERT or Wav2Vec2 features for lip-sync")
    parser.add_argument('--wav', type=str, required=True, help='Path to audio file')
    parser.add_argument('--model', type=str, default='wav2vec2', 
                       choices=['wav2vec2', 'avhubert'],
                       help='Model type to use for feature extraction')
    parser.add_argument('--output', type=str, default=None, 
                       help='Output path for .npy file (default: same as wav with suffix)')
    parser.add_argument('--fps', type=float, default=25.0,
                       help='Target video frame rate for alignment')
    parser.add_argument('--avhubert_checkpoint', type=str, default=None,
                       help='Path to AV-HuBERT checkpoint (required for avhubert model)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run model on')
    
    args = parser.parse_args()
    
    # Set device
    if 'cuda' in args.device and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Extract and save features
    output_path = extract_and_save_features(
        wav_path=args.wav,
        model_type=args.model,
        output_path=args.output,
        video_fps=args.fps,
        avhubert_checkpoint=args.avhubert_checkpoint
    )
    
    print(f"[INFO] Feature extraction complete: {output_path}")

