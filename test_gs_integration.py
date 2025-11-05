#!/usr/bin/env python
"""
Test script for 3DGS integration with SyncTalk
Tests the pipeline with May sample dataset
"""

import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("="*60)
    print("Testing imports...")
    print("="*60)
    
    try:
        # Test 3DGS components
        print("✓ Testing gaussian_renderer imports...")
        from gaussian_renderer import render, GaussianRasterizationSettings
        print("  Success!")
        
        print("✓ Testing scene_gs imports...")
        from scene_gs.gaussian_model import GaussianModel
        print("  Success!")
        
        print("✓ Testing utils_gs imports...")
        from utils_gs.loss_utils import l1_loss, ssim
        print("  Success!")
        
        # Test modified SyncTalk components
        print("✓ Testing network_gs...")
        from nerf_triplane.network_gs import DynamicGaussianNetwork
        print("  Success!")
        
        print("✓ Testing provider_gs...")
        from nerf_triplane.provider_gs import GaussianDataset, create_gaussian_data_loader
        print("  Success!")
        
        print("✓ Testing trainer_gs...")
        from nerf_triplane.trainer_gs import GaussianTrainer
        print("  Success!")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False


def test_model_creation():
    """Test if the model can be created"""
    print("\n" + "="*60)
    print("Testing model creation...")
    print("="*60)
    
    try:
        from nerf_triplane.network_gs import DynamicGaussianNetwork
        
        # Create minimal config
        class Config:
            def __init__(self):
                self.sh_degree = 0
                self.canonical_ply = ''
                self.audio_dim = 32
                self.eye_dim = 6
                self.exp_eye = True
                self.att = 2
                self.asr = False
                
        config = Config()
        
        print("Creating DynamicGaussianNetwork...")
        model = DynamicGaussianNetwork(config)
        
        print(f"✓ Model created successfully!")
        print(f"  Number of Gaussians: {model.num_gaussians}")
        
        # Test forward pass with dummy data
        print("\nTesting forward pass...")
        B = 1
        auds = torch.randn(B, 32, 16)  # Dummy audio features
        eye = torch.randn(B, 6)  # Dummy eye features
        poses = torch.eye(4).unsqueeze(0)  # Identity pose
        
        with torch.no_grad():
            output = model(auds, eye, poses)
        
        print("✓ Forward pass successful!")
        print(f"  Output keys: {list(output.keys())}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading(data_path):
    """Test if data can be loaded"""
    print("\n" + "="*60)
    print("Testing data loading...")
    print("="*60)
    
    if not os.path.exists(data_path):
        print(f"⚠️  Data path not found: {data_path}")
        print("   Please ensure May sample is available")
        return False
        
    try:
        from nerf_triplane.provider_gs import create_gaussian_data_loader
        
        # Create minimal config
        class Config:
            def __init__(self, path):
                self.path = path
                self.data_range = [0, 10]  # Load only first 10 frames for testing
                self.preload = 0
                self.bound = 1
                self.scale = 4
                self.offset = [0, 0, 0]
                self.exp_eye = True
                self.eye_dim = 6
                self.att = 2
                self.aud = ''
                self.asr = False
                self.asr_model = 'ave'
                self.torso = False
                self.portrait = False
                self.smooth_eye = False
                self.au45 = False
                self.bs_area = 'upper'
                self.fix_eye = -1
                
        config = Config(data_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading dataset from: {data_path}")
        loader = create_gaussian_data_loader(config, device, type='train', downscale=1)
        
        print(f"✓ DataLoader created successfully!")
        print(f"  Dataset size: {len(loader)}")
        
        # Test loading one batch
        print("\nLoading first batch...")
        for i, data in enumerate(loader):
            if i >= 1:
                break
            
            print(f"✓ Batch loaded successfully!")
            print(f"  Batch keys: {list(data.keys())}")
            print(f"  Image shape: {data['images'].shape if 'images' in data else 'N/A'}")
            print(f"  Pose shape: {data['poses'].shape if 'poses' in data else 'N/A'}")
            
        return True
        
    except Exception as e:
        print(f"\n❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step(data_path):
    """Test if a training step can be executed"""
    print("\n" + "="*60)
    print("Testing training step...")
    print("="*60)
    
    if not os.path.exists(data_path):
        print(f"⚠️  Data path not found: {data_path}")
        return False
        
    try:
        from nerf_triplane.network_gs import DynamicGaussianNetwork
        from nerf_triplane.provider_gs import create_gaussian_data_loader
        from nerf_triplane.trainer_gs import GaussianTrainer
        
        # Create config
        class Config:
            def __init__(self, path):
                # Model config
                self.sh_degree = 0
                self.canonical_ply = ''
                self.audio_dim = 32
                self.eye_dim = 6
                self.exp_eye = True
                self.att = 2
                self.asr = False
                self.asr_model = 'ave'
                
                # Data config
                self.path = path
                self.data_range = [0, 5]
                self.preload = 0
                self.bound = 1
                self.scale = 4
                self.offset = [0, 0, 0]
                self.torso = False
                self.portrait = False
                self.smooth_eye = False
                self.au45 = False
                self.bs_area = 'upper'
                self.fix_eye = -1
                self.aud = ''
                
                # Training config
                self.lambda_ssim = 0.2
                self.finetune_lips = False
                self.lr = 1e-3
                self.lr_net = 5e-4
                self.lr_gaussian = 0.0
                self.lr_decay_steps = 10000
                self.lr_decay_gamma = 0.5
                
        config = Config(data_path)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create model
        print("Creating model...")
        model = DynamicGaussianNetwork(config)
        
        # Initialize with some random Gaussians for testing
        if model.num_gaussians == 0:
            print("Initializing random Gaussians...")
            N = 100  # Small number for testing
            model.gaussians._xyz = torch.randn(N, 3, device=device) * 0.1
            model.gaussians._features_dc = torch.randn(N, 1, 3, device=device)
            model.gaussians._features_rest = torch.zeros(N, 0, 3, device=device)
            model.gaussians._scaling = torch.ones(N, 3, device=device) * -3
            model.gaussians._rotation = torch.zeros(N, 4, device=device)
            model.gaussians._rotation[:, 0] = 1  # Identity quaternion
            model.gaussians._opacity = torch.ones(N, 1, device=device) * 0.1
            
        # Create data loader
        print("Creating data loader...")
        loader = create_gaussian_data_loader(config, device, type='train', downscale=2)  # Downscale for faster testing
        
        # Create trainer
        print("Creating trainer...")
        trainer = GaussianTrainer(
            name='test',
            opt=config,
            model=model,
            device=device,
            workspace='test_workspace',
            fp16=False
        )
        
        # Test one training step
        print("\nExecuting training step...")
        for i, data in enumerate(loader):
            if i >= 1:
                break
                
            loss, pred = trainer.train_step(data)
            print(f"✓ Training step successful!")
            print(f"  Loss: {loss.item():.6f}")
            print(f"  Prediction shape: {pred.shape}")
            
        return True
        
    except Exception as e:
        print(f"\n❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Test 3DGS Integration')
    parser.add_argument('--data_path', type=str, default='data/May', 
                       help='Path to May sample dataset')
    parser.add_argument('--skip_data_tests', action='store_true',
                       help='Skip tests that require data')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("3DGS Integration Test Suite")
    print("="*60)
    
    # Track test results
    results = []
    
    # Test 1: Imports
    results.append(("Import Test", test_imports()))
    
    # Test 2: Model Creation
    results.append(("Model Creation", test_model_creation()))
    
    if not args.skip_data_tests:
        # Test 3: Data Loading
        results.append(("Data Loading", test_data_loading(args.data_path)))
        
        # Test 4: Training Step
        results.append(("Training Step", test_training_step(args.data_path)))
    else:
        print("\n⚠️  Skipping data-dependent tests")
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n🎉 All tests passed! The 3DGS integration is ready.")
        print("\nNext steps:")
        print("1. Prepare canonical model:")
        print(f"   python main_gs.py {args.data_path} --stage canonical --iters 30000")
        print("\n2. Train deformation network:")
        print(f"   python main_gs.py {args.data_path} --stage deform --canonical_ply workspace_gs/canonical.ply")
        print("\n3. Fine-tune all parameters:")
        print(f"   python main_gs.py {args.data_path} --stage finetune --canonical_ply workspace_gs/canonical.ply")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
        
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
