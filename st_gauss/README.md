# ST-Gauss: Hybrid SyncTalk-Gaussian Pipeline

A hybrid architecture combining the best of both worlds:
- **Face Branch**: Deformable 3D Gaussian Splatting (from TalkingGaussian) for skin, eyes, and head shape
- **Mouth Branch**: Tri-Plane Hash NeRF (from SyncTalk) for teeth, tongue, and oral cavity

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      ST-Gauss Pipeline                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input: Audio + Video/Transforms                             │
│                                                              │
│  ┌──────────────────┐    ┌──────────────────────────────┐   │
│  │ SyncTalk AVE     │    │ 52 ARKit Blendshapes         │   │
│  │ Audio Encoder    │    │ (from blendshape_capture)    │   │
│  └────────┬─────────┘    └──────────────┬───────────────┘   │
│           │                              │                   │
│           └──────────────┬───────────────┘                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │        Facial-Aware Masked Attention                  │   │
│  │   (Disentangles audio/expression for face regions)   │   │
│  └─────────────────────┬────────────────────────────────┘   │
│                        │                                     │
│           ┌────────────┴────────────┐                       │
│           ▼                          ▼                       │
│  ┌─────────────────────┐   ┌─────────────────────────┐      │
│  │   FACE BRANCH       │   │    MOUTH BRANCH         │      │
│  │   (3D Gaussian      │   │    (Tri-Plane Hash      │      │
│  │    Splatting)       │   │     NeRF)               │      │
│  │                     │   │                         │      │
│  │ - BlendshapeMotion  │   │ - MouthNeRFNetwork      │      │
│  │   Network           │   │ - Audio-conditioned     │      │
│  │ - Deformable GS     │   │   density/color         │      │
│  └──────────┬──────────┘   └───────────┬─────────────┘      │
│             │                           │                    │
│             ▼                           ▼                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              HYBRID RENDERER                         │    │
│  │                                                      │    │
│  │   final = face_rgb * alpha + mouth_rgb * (1-alpha)  │    │
│  │                                                      │    │
│  └──────────────────────────────────────────────────────┘   │
│                          │                                   │
│                          ▼                                   │
│                   Final Rendered Image                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Key Improvements over TalkingGaussian

| Feature | TalkingGaussian | ST-Gauss |
|---------|-----------------|----------|
| Audio Features | DeepSpeech | SyncTalk AVE (LRS2-trained) |
| Expression Control | 6 Action Units | 52 ARKit Blendshapes |
| Mouth Interior | Deformable Gaussian | Tri-Plane Hash NeRF |
| Pose Estimation | Standard 3DMM | Bundle Adjustment (Head-Sync) |
| Attention | Simple channel attention | Facial-Aware Masked Attention |

## Installation

The ST-Gauss module requires the same dependencies as the base SyncFace project:

```bash
# Install base dependencies
pip install -r requirements.txt

# Build CUDA extensions
cd gridencoder && pip install . && cd ..
cd raymarching && pip install . && cd ..
cd freqencoder && pip install . && cd ..
cd shencoder && pip install . && cd ..
```

## Usage

### 1. Data Preprocessing

Use SyncTalk's preprocessing pipeline:

```bash
# Full preprocessing
python st_gauss/prepare_data.py /path/to/video.mp4

# Or use the shell script
./TalkingGaussian/scripts/prepare.sh /path/to/video.mp4
```

### 2. Training

Train all three stages:

```bash
# Full pipeline
python st_gauss/train_all.py -s /path/to/data --config default

# Or train stages separately
python st_gauss/train_mouth_nerf.py -s /path/to/data      # Stage 1
python st_gauss/train_face_gaussian.py -s /path/to/data   # Stage 2
python st_gauss/train_joint.py -s /path/to/data           # Stage 3
```

### 3. Synthesis

Generate talking head video:

```bash
python st_gauss/synthesize.py -s /path/to/data -m /path/to/model
```

## Training Stages

### Stage 1: Mouth NeRF Training
- **Goal**: Train Tri-Plane Hash NeRF for mouth interior
- **Loss**: L1 + LPIPS on mouth_mask region only
- **Iterations**: 30,000 (default)

### Stage 2: Face Gaussian Training  
- **Goal**: Train Deformable 3DGS for face skin
- **Conditioning**: 52 ARKit blendshapes + AVE audio
- **Loss**: L1 + SSIM + LPIPS on face_mask (excluding mouth)
- **Iterations**: 60,000 (default)

### Stage 3: Joint Fine-tuning
- **Goal**: Harmonize face-mouth boundary
- **Frozen**: NeRF geometry (sigma network)
- **Trainable**: Gaussian colors/opacity, NeRF colors
- **Loss**: Full image L1 + LPIPS
- **Iterations**: 10,000 (default)

## Configuration

Available presets in `config.py`:

- `default`: Balanced quality/speed
- `high_quality`: More iterations, higher resolution encoders
- `fast`: Quick training for testing

Override individual settings:

```python
from st_gauss.config import get_config

config = get_config('default')
config.stage1_iterations = 50000
config.audio_type = 'hubert'
```

## Module Structure

```
st_gauss/
├── __init__.py           # Package exports
├── config.py             # Configuration dataclass
├── mouth_nerf.py         # Tri-Plane Hash NeRF for mouth
├── face_motion.py        # BlendshapeMotionNetwork + Attention
├── hybrid_renderer.py    # Face GS + Mouth NeRF fusion
├── train_mouth_nerf.py   # Stage 1 training
├── train_face_gaussian.py # Stage 2 training
├── train_joint.py        # Stage 3 training
├── train_all.py          # Full pipeline orchestrator
├── prepare_data.py       # Data preprocessing
├── synthesize.py         # Inference/synthesis
└── README.md             # This file
```

## Key Components

### MouthNeRFNetwork (`mouth_nerf.py`)
Tri-plane hash encoding NeRF for view-consistent mouth rendering:
- Prevents teeth flickering common in point-based methods
- Audio-conditioned for lip-sync
- Efficient ray marching within mouth mask only

### BlendshapeMotionNetwork (`face_motion.py`)
Deformable Gaussian motion field with:
- 52 ARKit blendshape conditioning
- Facial-Aware Masked Attention for disentanglement
- Predicts per-Gaussian: position, rotation, scale, opacity offsets

### HybridRenderer (`hybrid_renderer.py`)
Combines both branches:
```python
final_image = face_rgb * face_alpha + mouth_rgb * (1 - face_alpha)
```
Mouth is rendered BEHIND face - visible only where lips are open.

## References

- [TalkingGaussian](https://github.com/xxx) - Base 3DGS talking head
- [SyncTalk](https://github.com/xxx) - Tri-plane NeRF and AVE encoder
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [Instant-NGP](https://github.com/NVlabs/instant-ngp) - Hash encoding

## License

This project combines code from TalkingGaussian and SyncTalk under their respective licenses.

