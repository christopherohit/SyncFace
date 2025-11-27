"""
Encoding Utilities for NeRF
============================
Position and direction encoding functions.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


def get_encoder(
    encoding_type: str = 'hashgrid',
    **kwargs
) -> Tuple[nn.Module, int]:
    """
    Get encoder by type.
    
    Args:
        encoding_type: 'hashgrid', 'frequency', or 'spherical_harmonics'
        **kwargs: Encoder-specific parameters
    
    Returns:
        Tuple of (encoder, output_dim)
    """
    if encoding_type == 'hashgrid':
        try:
            from gridencoder import GridEncoder
            encoder = GridEncoder(
                input_dim=kwargs.get('input_dim', 3),
                num_levels=kwargs.get('num_levels', 16),
                level_dim=kwargs.get('level_dim', 2),
                base_resolution=kwargs.get('base_resolution', 16),
                log2_hashmap_size=kwargs.get('log2_hashmap_size', 19),
                desired_resolution=kwargs.get('desired_resolution', 2048),
            )
            out_dim = kwargs.get('num_levels', 16) * kwargs.get('level_dim', 2)
            return encoder, out_dim
        except ImportError:
            print("[WARNING] GridEncoder not available, using frequency encoding")
            return get_encoder('frequency', **kwargs)
    
    elif encoding_type == 'frequency':
        encoder = FrequencyEncoder(
            input_dim=kwargs.get('input_dim', 3),
            num_freqs=kwargs.get('num_freqs', 10),
            log_sampling=kwargs.get('log_sampling', True),
        )
        return encoder, encoder.output_dim
    
    elif encoding_type == 'spherical_harmonics':
        try:
            from shencoder import SHEncoder
            degree = kwargs.get('degree', 4)
            encoder = SHEncoder(input_dim=3, degree=degree)
            return encoder, (degree + 1) ** 2
        except ImportError:
            print("[WARNING] SHEncoder not available, using identity")
            return nn.Identity(), 3
    
    else:
        raise ValueError(f"Unknown encoding type: {encoding_type}")


class FrequencyEncoder(nn.Module):
    """
    Frequency-based positional encoding.
    
    Encodes input as [sin(2^0 * pi * x), cos(2^0 * pi * x), ..., sin(2^L * pi * x), cos(2^L * pi * x)]
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        num_freqs: int = 10,
        log_sampling: bool = True,
        include_input: bool = True,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        self.include_input = include_input
        
        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        else:
            freq_bands = torch.linspace(1, 2 ** (num_freqs - 1), num_freqs)
        
        self.register_buffer('freq_bands', freq_bands)
        
        self.output_dim = input_dim * num_freqs * 2
        if include_input:
            self.output_dim += input_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input positions.
        
        Args:
            x: [..., input_dim] input positions
        
        Returns:
            [..., output_dim] encoded positions
        """
        encoded = []
        
        if self.include_input:
            encoded.append(x)
        
        for freq in self.freq_bands:
            encoded.append(torch.sin(x * freq * np.pi))
            encoded.append(torch.cos(x * freq * np.pi))
        
        return torch.cat(encoded, dim=-1)


class SinusoidalEncoder(nn.Module):
    """
    Sinusoidal positional encoding (as used in Transformers).
    """
    
    def __init__(self, dim: int, max_freq: float = 10000.0):
        super().__init__()
        
        self.dim = dim
        self.max_freq = max_freq
        
        half_dim = dim // 2
        freqs = torch.exp(
            -np.log(max_freq) * torch.arange(half_dim) / half_dim
        )
        self.register_buffer('freqs', freqs)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode scalar positions.
        
        Args:
            x: [...] scalar positions
        
        Returns:
            [..., dim] encoded positions
        """
        x = x.unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)



