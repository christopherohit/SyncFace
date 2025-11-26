"""
ST-Gauss Configuration
Configuration dataclass for the hybrid SyncTalk-Gaussian pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple


@dataclass
class STGaussConfig:
    """Configuration for ST-Gauss hybrid pipeline."""
    
    # ============= Data Paths =============
    data_dir: str = ""
    output_dir: str = "./output"
    checkpoint_dir: str = "./checkpoints"
    
    # ============= Audio Features =============
    audio_type: str = "ave"  # 'ave', 'hubert', or 'deepspeech'
    audio_dim: int = 32  # Encoded audio feature dimension
    ave_encoder_path: str = "./checkpoints/ave_encoder.pth"
    
    # ============= Blendshape Parameters =============
    blendshape_dim: int = 52  # ARKit 52 blendshapes
    use_blendshape: bool = True
    blendshape_weight_path: str = ""  # Path to pretrained blendshape capture weights
    
    # ============= Face Branch (3DGS) =============
    face_sh_degree: int = 3
    face_bound: float = 0.15
    face_motion_hidden_dim: int = 64
    face_motion_num_layers: int = 3
    face_hash_levels: int = 12
    face_hash_level_dim: int = 1
    face_hash_base_resolution: int = 16
    face_hash_log2_hashmap_size: int = 17
    
    # ============= Mouth Branch (NeRF) =============
    mouth_bound: float = 0.15
    mouth_hash_levels: int = 12
    mouth_hash_level_dim: int = 1
    mouth_hash_base_resolution: int = 64
    mouth_hash_log2_hashmap_size: int = 14
    mouth_hidden_dim: int = 64
    mouth_num_layers: int = 3
    mouth_geo_feat_dim: int = 64
    
    # ============= Masked Attention (SyncTalk) =============
    use_facial_aware_attention: bool = True
    attention_audio_regions: List[str] = field(default_factory=lambda: ["lips", "jaw"])
    attention_expression_regions: List[str] = field(default_factory=lambda: ["eye", "brow", "forehead"])
    
    # ============= Training Stage 1: Mouth NeRF =============
    stage1_iterations: int = 30000
    stage1_lr: float = 5e-3
    stage1_lr_net: float = 5e-4
    stage1_warmup_iterations: int = 1000
    stage1_save_interval: int = 10000
    
    # ============= Training Stage 2: Face 3DGS =============
    stage2_iterations: int = 60000
    stage2_lr: float = 5e-3
    stage2_lr_net: float = 5e-4
    stage2_warmup_iterations: int = 3000
    stage2_densify_until_iter: int = 50000
    stage2_densify_from_iter: int = 500
    stage2_densification_interval: int = 100
    stage2_opacity_reset_interval: int = 3000
    stage2_save_interval: int = 10000
    
    # ============= Training Stage 3: Joint Fine-tuning =============
    stage3_iterations: int = 10000
    stage3_lr: float = 1e-4
    stage3_lr_net: float = 1e-5
    stage3_freeze_nerf_geometry: bool = True
    stage3_save_interval: int = 2000
    
    # ============= Loss Weights =============
    lambda_l1: float = 1.0
    lambda_ssim: float = 0.2
    lambda_lpips: float = 0.1
    lambda_lip_sync: float = 0.05  # Lip landmark supervision loss
    lambda_motion_reg: float = 1e-5
    lambda_alpha_reg: float = 1e-3
    lambda_lip_opacity: float = 0.01  # Minimum opacity constraint for lips
    
    # ============= Rendering =============
    background_color: Tuple[float, float, float] = (0.0, 1.0, 0.0)  # Green screen
    image_height: int = 512
    image_width: int = 512
    
    # ============= Device & Performance =============
    device: str = "cuda"
    mixed_precision: bool = True
    num_workers: int = 4
    batch_size: int = 1
    
    # ============= Portrait-Sync Generator (Optional Enhancement) =============
    use_portrait_sync: bool = False
    portrait_sync_path: str = ""
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.audio_type in ["ave", "hubert", "deepspeech"], \
            f"Invalid audio_type: {self.audio_type}"
        assert self.blendshape_dim > 0, "blendshape_dim must be positive"
        assert self.stage1_iterations > 0, "stage1_iterations must be positive"
        assert self.stage2_iterations > 0, "stage2_iterations must be positive"
        assert self.stage3_iterations > 0, "stage3_iterations must be positive"
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "STGaussConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }


# Preset configurations for common scenarios
PRESETS = {
    "default": STGaussConfig(),
    
    "high_quality": STGaussConfig(
        stage1_iterations=50000,
        stage2_iterations=100000,
        stage3_iterations=20000,
        face_hash_levels=16,
        mouth_hash_levels=16,
        use_portrait_sync=True,
    ),
    
    "fast": STGaussConfig(
        stage1_iterations=15000,
        stage2_iterations=30000,
        stage3_iterations=5000,
        face_hash_levels=8,
        mouth_hash_levels=8,
    ),
}


def get_config(preset: str = "default", **overrides) -> STGaussConfig:
    """Get configuration with optional overrides."""
    if preset not in PRESETS:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")
    
    config = PRESETS[preset]
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown config key: {key}")
    
    return config

