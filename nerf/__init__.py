"""
NeRF Module
===========
Imports from existing nerf_triplane implementation.
"""

import sys
from pathlib import Path

# Add nerf_triplane to path
_nerf_path = Path(__file__).parent.parent / 'nerf_triplane'
sys.path.insert(0, str(_nerf_path))

# Import from nerf_triplane
from network import NeRFNetwork, AudioNet, AudioAttNet, AudioEncoder, MLP
from renderer import NeRFRenderer
from encoding import get_encoder

__all__ = [
    'NeRFNetwork',
    'AudioNet',
    'AudioAttNet', 
    'AudioEncoder',
    'MLP',
    'NeRFRenderer',
    'get_encoder',
]
