"""
SyncFace: Unified Talking Head Synthesis
=========================================
Combines 3D Gaussian Splatting (TalkingGaussian) with Tri-plane NeRF (SyncTalk)
for high-quality audio-driven talking head synthesis.

Components:
- gaussian/  : Deformable 3DGS for face rendering
- nerf/      : Tri-plane Hash NeRF for mouth interior
- hybrid/    : Combined rendering pipeline
- data/      : Data loading and preprocessing
- utils/     : Shared utilities

Usage:
    from SyncFace import GaussianFace, NeRFMouth, HybridRenderer
"""

__version__ = "1.0.0"
__author__ = "SyncFace Team"

# Import main components
from .gaussian import GaussianModel, MotionNetwork
from .nerf import NeRFNetwork, NeRFRenderer
from .hybrid import HybridFaceRenderer, SyncFaceModel

__all__ = [
    'GaussianModel',
    'MotionNetwork', 
    'NeRFNetwork',
    'NeRFRenderer',
    'HybridFaceRenderer',
    'SyncFaceModel',
]

