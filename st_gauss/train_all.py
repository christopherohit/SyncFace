"""
ST-Gauss Full Training Pipeline

Orchestrates all three training stages:
1. Stage 1: Mouth NeRF Training
2. Stage 2: Face Gaussian Training  
3. Stage 3: Joint Fine-tuning

Usage:
    python st_gauss/train_all.py -s /path/to/data --config default
"""

import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from st_gauss.config import get_config, STGaussConfig
from st_gauss.train_mouth_nerf import training_stage1
from st_gauss.train_face_gaussian import training_stage2
from st_gauss.train_joint import training_stage3

from TalkingGaussian.arguments import ModelParams, PipelineParams, OptimizationParams
from TalkingGaussian.utils.general_utils import safe_state


def train_st_gauss(
    dataset,
    config: STGaussConfig,
    pipe,
    start_stage: int = 1,
    end_stage: int = 3,
):
    """
    Train ST-Gauss hybrid pipeline through all stages.
    
    Args:
        dataset: Model parameters / dataset configuration
        config: STGaussConfig configuration
        pipe: Pipeline parameters
        start_stage: Stage to start from (1, 2, or 3)
        end_stage: Stage to end at (1, 2, or 3)
    """
    print("=" * 60)
    print("ST-Gauss: Hybrid SyncTalk-Gaussian Training Pipeline")
    print("=" * 60)
    print(f"\nConfiguration: {config.audio_type} audio, {config.blendshape_dim} blendshapes")
    print(f"Stages to run: {start_stage} -> {end_stage}")
    print(f"Data path: {dataset.source_path}")
    print(f"Output path: {dataset.model_path}")
    print("=" * 60 + "\n")
    
    mouth_nerf = None
    gaussians = None
    motion_net = None
    
    # ============= Stage 1: Mouth NeRF =============
    if start_stage <= 1 and end_stage >= 1:
        print("\n" + "=" * 60)
        print("STAGE 1: Mouth NeRF Training")
        print("=" * 60)
        print("Training Tri-Plane Hash NeRF for mouth interior (teeth, tongue)")
        print(f"Iterations: {config.stage1_iterations}")
        print("=" * 60 + "\n")
        
        mouth_nerf = training_stage1(
            dataset=dataset,
            config=config,
            pipe=pipe,
            checkpoint_path=None,
        )
        
        torch.cuda.empty_cache()
    
    # ============= Stage 2: Face Gaussian =============
    if start_stage <= 2 and end_stage >= 2:
        print("\n" + "=" * 60)
        print("STAGE 2: Face Gaussian Training")
        print("=" * 60)
        print("Training Deformable 3D Gaussians for face skin")
        print(f"Using {config.blendshape_dim} ARKit blendshapes")
        print(f"Iterations: {config.stage2_iterations}")
        print("=" * 60 + "\n")
        
        gaussians, motion_net = training_stage2(
            dataset=dataset,
            config=config,
            pipe=pipe,
            checkpoint_path=None,
        )
        
        torch.cuda.empty_cache()
    
    # ============= Stage 3: Joint Fine-tuning =============
    if start_stage <= 3 and end_stage >= 3:
        print("\n" + "=" * 60)
        print("STAGE 3: Joint Fine-tuning")
        print("=" * 60)
        print("Harmonizing Face Gaussians + Mouth NeRF boundary")
        print(f"Iterations: {config.stage3_iterations}")
        print("=" * 60 + "\n")
        
        gaussians, motion_net, mouth_nerf = training_stage3(
            dataset=dataset,
            config=config,
            pipe=pipe,
            face_checkpoint=None,  # Will auto-load from model_path
            mouth_checkpoint=None,  # Will auto-load from model_path
        )
        
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 60)
    print("ST-GAUSS TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nFinal checkpoints saved to: {dataset.model_path}")
    print("\nTo synthesize:")
    print(f"  python st_gauss/synthesize.py -s {dataset.source_path} -m {dataset.model_path}")
    print("=" * 60 + "\n")
    
    return gaussians, motion_net, mouth_nerf


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ST-Gauss Full Training Pipeline")
    
    # Model and pipeline parameters
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    # ST-Gauss specific
    parser.add_argument('--config', type=str, default='default',
                       help='Config preset: default, high_quality, fast')
    parser.add_argument('--start_stage', type=int, default=1,
                       help='Stage to start from (1, 2, or 3)')
    parser.add_argument('--end_stage', type=int, default=3,
                       help='Stage to end at (1, 2, or 3)')
    parser.add_argument('--quiet', action='store_true')
    
    # Override config values
    parser.add_argument('--audio_type', type=str, default=None,
                       help='Override audio type: ave, hubert, deepspeech')
    parser.add_argument('--stage1_iter', type=int, default=None,
                       help='Override Stage 1 iterations')
    parser.add_argument('--stage2_iter', type=int, default=None,
                       help='Override Stage 2 iterations')
    parser.add_argument('--stage3_iter', type=int, default=None,
                       help='Override Stage 3 iterations')
    
    args = parser.parse_args()
    
    # Load and override config
    config = get_config(args.config)
    
    if args.audio_type:
        config.audio_type = args.audio_type
    if args.stage1_iter:
        config.stage1_iterations = args.stage1_iter
    if args.stage2_iter:
        config.stage2_iterations = args.stage2_iter
    if args.stage3_iter:
        config.stage3_iterations = args.stage3_iter
    
    print(f"\nST-Gauss Training")
    print(f"Config preset: {args.config}")
    print(f"Audio type: {config.audio_type}")
    
    safe_state(args.quiet)
    
    train_st_gauss(
        dataset=lp.extract(args),
        config=config,
        pipe=pp.extract(args),
        start_stage=args.start_stage,
        end_stage=args.end_stage,
    )

