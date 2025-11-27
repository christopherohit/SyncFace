"""
Gaussian Splatting Module
=========================
Imports from existing TalkingGaussian implementation.
"""

import sys
from pathlib import Path

# Add TalkingGaussian to path
_tg_path = Path(__file__).parent.parent / 'TalkingGaussian'
sys.path.insert(0, str(_tg_path))

# Import from TalkingGaussian
from scene.gaussian_model import GaussianModel
from scene.motion_net import MotionNetwork, MouthMotionNetwork
from scene.motion_net_sync import BlendshapeMotionNetwork, MouthMotionNetworkSync
from gaussian_renderer import render, render_motion, render_motion_mouth

# Aliases
GaussianRenderer = None  # Use render function directly
render_gaussians = render

__all__ = [
    'GaussianModel',
    'MotionNetwork',
    'MouthMotionNetwork',
    'BlendshapeMotionNetwork',
    'MouthMotionNetworkSync',
    'render',
    'render_motion',
    'render_motion_mouth',
    'render_gaussians',
]
