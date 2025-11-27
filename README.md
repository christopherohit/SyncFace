# SyncFace: Hybrid Talking Head Synthesis

**Unified 3D Gaussian Splatting + Tri-plane NeRF for Audio-Driven Talking Heads**

SyncFace combines the best of both approaches:
- **Gaussian Splatting** (TalkingGaussian) for fast, high-quality face rendering
- **Tri-plane NeRF** (SyncTalk) for detailed mouth interior (teeth, tongue)

## 🏗️ Project Structure

```
SyncFace/
├── gaussian/           # 3D Gaussian Splatting module
│   ├── gaussian_model.py
│   ├── motion_network.py
│   └── renderer.py
├── nerf/              # Tri-plane NeRF module
│   ├── network.py
│   ├── renderer.py
│   └── encoding.py
├── hybrid/            # Combined rendering
│   ├── model.py       # SyncFaceModel
│   ├── renderer.py    # HybridFaceRenderer
│   └── compositor.py  # MouthCompositor
├── data/              # Data utilities
├── utils/             # Shared utilities
├── data_utils/        # Preprocessing
├── train.py           # Main training script
└── requirements.txt
```

## 🚀 Quick Start

### Installation

```bash
# Create environment
conda create -n syncface python=3.10
conda activate syncface

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt

# Compile CUDA extensions
cd diff-gaussian-rasterization && pip install -e . && cd ..
cd gridencoder && pip install -e . && cd ..
cd raymarching && pip install -e . && cd ..
```

### Preprocessing

```bash
python data_utils/process_sync.py /path/to/video.mp4 --task -1 --asr hubert
```

### Training

```bash
# Full pipeline
python train.py --data_dir ./data/person --output_dir ./output/person

# Stage-specific
python train.py --data_dir ./data/person --stage face
python train.py --data_dir ./data/person --stage mouth
python train.py --data_dir ./data/person --stage fusion
```

## 📊 Training Pipeline

### Stage 1: Face (Deformable 3DGS)
- Learns head shape, skin, hair, facial expressions
- Excludes mouth interior from loss
- Conditioned on audio + 52-dim blendshapes
- **50k iterations**

### Stage 2: Mouth (Tri-plane NeRF)
- Learns teeth, tongue, oral cavity
- Ray marches only in `lips_rect` region
- High LPIPS weight for detail
- **100k iterations**

### Stage 3: Fusion
- Blends face and mouth seamlessly
- Fine-tunes color consistency
- Freezes geometry
- **20k iterations**

## 🎯 Key Features

### BlendshapeMotionNetwork
52-dim ARKit blendshape conditioning with prior injection:
```python
from gaussian.motion_network import BlendshapeMotionNetwork

motion_net = BlendshapeMotionNetwork(
    audio_in_dim=1024,  # HuBERT
    blendshape_dim=52,
    blendshape_prior_scale=0.1,
)
```

### NeRFNetwork (Mouth)
Tri-plane hash-grid NeRF:
```python
from nerf.network import NeRFNetwork

mouth_nerf = NeRFNetwork(
    bound=0.3,
    audio_dim=32,
    audio_in_dim=1024,
)
```

### SyncFaceModel (Unified)
```python
from hybrid.model import SyncFaceModel

model = SyncFaceModel(
    sh_degree=3,
    audio_dim=64,
    blendshape_dim=52,
)

# Render hybrid
result = model(camera, pipe, bg_color, audio, blendshape, mouth_mask, lips_rect)
```

## 📁 Data Format

After preprocessing:
```
data/person/
├── transforms_train.json    # Camera poses
├── transforms_val.json
├── audio_feats.npy         # [T, 1024] HuBERT
├── blendshapes.npy         # [T, 52] ARKit
├── lips_rect.json          # Per-frame mouth boxes
├── ori_imgs/               # Original frames
├── gt_imgs/                # Inpainted GT
├── torso_imgs/             # Torso images
├── parsing/                # Semantic masks
└── mouth_mask/             # Binary mouth masks
```

## ⚙️ Hyperparameters

| Stage | Parameter | Default | Description |
|-------|-----------|---------|-------------|
| Face | iterations | 50000 | Training iterations |
| Face | lr_motion | 1e-4 | Motion network LR |
| Mouth | iterations | 100000 | Training iterations |
| Mouth | lr_hash | 1e-3 | Hash grid LR |
| Mouth | rays_per_batch | 8192 | Rays per batch |
| Fusion | iterations | 20000 | Fine-tuning iterations |
| Fusion | lr_sh | 1e-5 | SH coefficient LR |

## 📚 References

- [TalkingGaussian](https://github.com/Fictionarry/TalkingGaussian) - 3D Gaussian Splatting for Talking Heads
- [SyncTalk](https://github.com/ZiqiaoPeng/SyncTalk) - Synchronized Talking Head Synthesis
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) - Original 3DGS

## 📄 License

Research use only. See LICENSE for details.
