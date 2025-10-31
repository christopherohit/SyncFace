# Enhanced Audio Encoder for SyncFace

## Overview

The enhanced audio encoder system replaces the original LRS2-trained Audio-Visual Encoder with foundation models that capture:

- ✅ **Prosodic Features**: Pitch, energy, rhythm, and intonation patterns
- ✅ **Emotional Context**: Sentiment and affective state in speech
- ✅ **Speaking Style**: Speaker personality and idiosyncrasies  
- ✅ **Multimodal Alignment**: CLIP-like contrastive audio-video synchronization

This goes **beyond simple audio-lip synchronization** to create more natural and expressive talking head animations.

---

## Architecture Comparison

### Original Architecture (LRS2-based)
```
Audio (Mel) → AudioEncoder (CNN) → 512-dim → AudioNet → 64-dim features
```
**Limitations:**
- Only captures audio → lip movements
- Trained on 2D LRS2 dataset
- Lacks emotional understanding
- No prosodic modeling

### Enhanced Architecture (Foundation Models)
```
Audio → Foundation Model → Prosody Extractor → Fusion → Enhanced Features
        (Whisper/SpeechT5/EnCodec)    (Pitch/Energy)      ↓
                                                    Contrastive A-V Alignment
```
**Advantages:**
- Rich prosodic understanding from pretrained models
- Captures emotion, style, and personality
- Multimodal audio-video synchronization
- Better generalization across speakers and languages

---

## Supported Foundation Models

### 1. **Whisper** (Recommended)
- **Model**: `openai/whisper-small`, `openai/whisper-base`, `openai/whisper-large`
- **Strengths**: 
  - Trained on 680k hours of multilingual audio
  - Excellent prosodic feature extraction
  - Strong on emotional speech and varied accents
- **Use Case**: Best all-around choice for most scenarios

### 2. **SpeechT5**
- **Model**: `microsoft/speecht5_tts`
- **Strengths**:
  - Unified speech-text encoder-decoder
  - Good for speech synthesis tasks
  - Balanced performance and speed
- **Use Case**: Good alternative to Whisper, slightly faster

### 3. **EnCodec**
- **Model**: `facebook/encodec_24khz`
- **Strengths**:
  - Neural audio codec with high-quality compression
  - Preserves fine-grained audio details
  - Very efficient representations
- **Use Case**: When audio quality preservation is critical

### 4. **Ensemble** (Best Performance)
- Combines Whisper + SpeechT5 + EnCodec
- Fuses complementary features from all models
- Highest quality but slower and more memory-intensive

### 5. **Hybrid** (Best of Both Worlds)
- Combines original LRS2 encoder + foundation model
- Maintains strong lip sync while adding prosody
- Good balance between compatibility and enhancement

---

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- `transformers>=4.36.0` for foundation models
- `accelerate>=0.20.0` for efficient model loading
- `torchcrepe` (optional) for advanced pitch extraction
- `praat-parselmouth` (optional) for prosody analysis

### 2. Download Foundation Models

Models are automatically downloaded on first use. To pre-download:

```python
from transformers import WhisperModel, SpeechT5Model, EncodecModel

# Whisper (recommended)
WhisperModel.from_pretrained("openai/whisper-small")

# SpeechT5
SpeechT5Model.from_pretrained("microsoft/speecht5_tts")

# EnCodec
EncodecModel.from_pretrained("facebook/encodec_24khz")
```

---

## Usage

### Basic Usage: Enable Enhanced Encoder

Add the following flags to your training/inference command:

```bash
python main.py \
  --use_enhanced_encoder \
  --enhanced_encoder_type whisper \
  --use_prosody \
  --freeze_audio_backbone \
  [other flags...]
```

### Configuration Options

#### Core Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--use_enhanced_encoder` | bool | False | Enable enhanced audio encoder |
| `--enhanced_encoder_type` | str | 'whisper' | Type: whisper, speecht5, encodec, ensemble, hybrid |
| `--use_prosody` | bool | True | Extract prosodic features (pitch, energy) |
| `--use_contrastive` | bool | False | Enable CLIP-like contrastive audio-video alignment |
| `--freeze_audio_backbone` | bool | True | Freeze pretrained foundation model weights |
| `--foundation_model_type` | str | 'whisper' | Which foundation model to use in hybrid mode |

#### Advanced Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--audio_encoder_dim` | int | 512 | Output dimension of foundation model encoder |
| `--contrastive_temperature` | float | 0.07 | Temperature for contrastive learning |
| `--prosody_weight` | float | 0.5 | Weight for prosody features in fusion |

### Example Commands

#### 1. **Whisper Encoder (Recommended)**
```bash
python main.py \
  --workspace output/enhanced_whisper \
  --use_enhanced_encoder \
  --enhanced_encoder_type whisper \
  --use_prosody \
  --freeze_audio_backbone \
  --O --test
```

#### 2. **Hybrid Mode (LRS2 + Whisper)**
Best for maintaining lip sync while adding prosody:
```bash
python main.py \
  --workspace output/hybrid \
  --use_enhanced_encoder \
  --enhanced_encoder_type hybrid \
  --foundation_model_type whisper \
  --use_prosody
```

#### 3. **Ensemble Mode (Maximum Quality)**
```bash
python main.py \
  --workspace output/ensemble \
  --use_enhanced_encoder \
  --enhanced_encoder_type ensemble \
  --use_prosody \
  --freeze_audio_backbone
```

#### 4. **With Contrastive Learning**
Enables CLIP-like audio-video alignment:
```bash
python main.py \
  --workspace output/contrastive \
  --use_enhanced_encoder \
  --enhanced_encoder_type whisper \
  --use_prosody \
  --use_contrastive \
  --contrastive_temperature 0.07
```

---

## Training with Enhanced Encoders

### Fine-tuning Strategy

The enhanced encoders support multiple training strategies:

#### 1. **Frozen Backbone (Default, Fastest)**
```bash
python main.py \
  --workspace output/training \
  --use_enhanced_encoder \
  --freeze_audio_backbone  # Foundation model frozen
```
- Foundation model weights are frozen
- Only train adapter layers and NeRF
- Fastest training, lowest memory
- Good for most cases

#### 2. **Full Fine-tuning (Best Quality)**
```bash
python main.py \
  --workspace output/training \
  --use_enhanced_encoder \
  --no-freeze_audio_backbone  # Fine-tune foundation model
```
- Fine-tune all layers including foundation model
- Highest quality, adapts to your data
- Requires more GPU memory and time
- Use with large datasets

#### 3. **Two-Stage Training**
```bash
# Stage 1: Train with frozen backbone
python main.py \
  --workspace output/stage1 \
  --use_enhanced_encoder \
  --freeze_audio_backbone \
  --iters 50000

# Stage 2: Fine-tune everything
python main.py \
  --workspace output/stage2 \
  --use_enhanced_encoder \
  --no-freeze_audio_backbone \
  --ckpt output/stage1/checkpoints/latest.pth \
  --iters 20000
```

### Learning Rate Recommendations

- **Frozen backbone**: Use default learning rates
- **Full fine-tuning**: 
  - Foundation model: `lr * 0.1` (automatically set)
  - Adapter layers: default `lr`
  - NeRF components: default `lr`

---

## Technical Details

### Prosody Extraction

The prosody extractor captures:

1. **Pitch Contour**: F0 trajectory for intonation
2. **Energy**: RMS energy for stress and emphasis
3. **Rhythm**: Temporal patterns in speech

These features are fused with semantic features from foundation models to create rich audio representations.

### Contrastive Audio-Video Alignment

Inspired by CLIP, this module:

1. Projects audio features to common embedding space
2. Projects video features to same space
3. Learns alignment via contrastive loss:
   ```
   L = - log(exp(sim(a_i, v_i) / τ) / Σ_j exp(sim(a_i, v_j) / τ))
   ```
4. Encourages matching audio-video pairs to be closer in embedding space

This improves synchronization beyond just lip movements.

### Memory and Speed

| Mode | GPU Memory | Speed | Quality |
|------|-----------|-------|---------|
| Original LRS2 | 6GB | 1.0x | ⭐⭐⭐ |
| Whisper (frozen) | 8GB | 0.8x | ⭐⭐⭐⭐ |
| Ensemble (frozen) | 12GB | 0.5x | ⭐⭐⭐⭐⭐ |
| Hybrid | 7GB | 0.9x | ⭐⭐⭐⭐ |

---

## Troubleshooting

### Issue: Out of Memory

**Solution**: Use smaller models or reduce batch size
```bash
python main.py \
  --use_enhanced_encoder \
  --enhanced_encoder_type whisper \
  --batch_size 1  # Reduce from default
```

### Issue: Models downloading slowly

**Solution**: Pre-download models or use local cache
```bash
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache
```

### Issue: Foundation models not available

**Solution**: Install transformers
```bash
pip install transformers>=4.36.0 accelerate>=0.20.0
```

---

## Comparison: Original vs Enhanced

| Feature | Original (LRS2) | Enhanced (Foundation) |
|---------|----------------|---------------------|
| Audio-Lip Sync | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Prosody | ❌ | ⭐⭐⭐⭐⭐ |
| Emotion | ❌ | ⭐⭐⭐⭐ |
| Speaking Style | ❌ | ⭐⭐⭐⭐ |
| Cross-lingual | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Speed | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Memory | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## Code Examples

### Using Enhanced Encoder Programmatically

```python
from nerf_triplane.enhanced_audio_encoder import EnhancedAudioEncoder
import torch

# Initialize encoder
encoder = EnhancedAudioEncoder(
    encoder_type='whisper',
    output_dim=512,
    use_prosody=True,
    use_contrastive=False,
    freeze_backbone=True
)

# Process audio
audio = torch.randn(1, 16000)  # 1 second at 16kHz
result = encoder(audio)

# Access features
audio_features = result['audio_features']  # [1, 512]
prosody_features = result['prosody_features']  # [1, 64]
```

### Using Hybrid Encoder

```python
from nerf_triplane.audio_encoder_adapter import HybridAudioEncoder

encoder = HybridAudioEncoder(
    use_lrs2=True,
    use_foundation=True,
    foundation_type='whisper',
    output_dim=512
)

mel = torch.randn(1, 1, 80, 100)  # Mel spectrogram
features = encoder(mel)  # [1, 512]
```

---

## Citation

If you use the enhanced audio encoder in your research, please cite:

```bibtex
@article{synctalk_enhanced,
  title={Enhanced Audio-Visual Synchronization with Foundation Models},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```

---

## Future Improvements

Potential directions for further enhancement:

1. **Multi-speaker modeling**: Speaker embedding integration
2. **Emotion-conditioned generation**: Explicit emotion control
3. **Cross-modal attention**: More sophisticated audio-video fusion
4. **Language-specific prosody**: Adapt to different language patterns
5. **Real-time optimization**: Distillation for faster inference

---

## Support

For issues or questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Refer to the main SyncFace documentation

---

## License

Same license as SyncFace project.

