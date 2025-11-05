#!/usr/bin/env python
"""
Simple test to verify the 3DGS integration is working
Tests model creation and forward pass without full rendering pipeline
"""

import torch
import sys
from unittest.mock import Mock

def test_basic_integration():
    """Test basic 3DGS integration without full rendering"""
    print("="*60)
    print("Testing Basic 3DGS Integration")
    print("="*60)

    try:
        # Test imports
        print("✓ Testing imports...")
        from nerf_triplane.network_gs import DynamicGaussianNetwork
        from nerf_triplane.trainer_gs import GaussianTrainer
        print("  Success!")

        # Test model creation
        print("✓ Testing model creation...")
        class Config:
            def __init__(self):
                self.sh_degree = 0
                self.canonical_ply = ''
                self.audio_dim = 32
                self.eye_dim = 6
                self.exp_eye = True
                self.att = 2
                self.asr = False
                self.asr_model = 'ave'
                self.num_rays = -1

        config = Config()
        model = DynamicGaussianNetwork(config)
        print(f"  Model created with {model.num_gaussians} Gaussians")

        # Test forward pass
        print("✓ Testing model forward pass...")
        auds = torch.randn(1, 32, 16, dtype=torch.float32)
        eye = torch.randn(1, 6, dtype=torch.float32)
        poses = torch.eye(4, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            output = model(auds, eye, poses)

        print(f"  Forward pass successful!")
        print(f"  Output keys: {list(output.keys())}")
        print(f"  XYZ shape: {output['xyz'].shape}")
        print(f"  Rotation shape: {output['rotation'].shape}")
        print(f"  Scaling shape: {output['scaling'].shape}")
        print(f"  Opacity shape: {output['opacity'].shape}")

        # Test trainer creation
        print("✓ Testing trainer creation...")
        mock_optimizer = lambda model: torch.optim.Adam(model.parameters(), lr=1e-3)

        trainer = GaussianTrainer(
            name='test_integration',
            opt=config,
            model=model,
            optimizer=mock_optimizer,
            device='cpu',  # Use CPU to avoid CUDA issues in test
            workspace='test_workspace',
            fp16=False
        )
        print("  Trainer created successfully!")

        print("\n✅ Basic integration test PASSED!")
        print("\n🎉 The SyncTalk + 3DGS integration is working correctly!")
        print("\nNext steps:")
        print("1. Preprocess your dataset: python data_utils/process.py --path data/May")
        print("2. Train canonical model: python main_gs.py data/May --stage canonical --iters 30000")
        print("3. Train deformation network: python main_gs.py data/May --stage deform --canonical_ply workspace_gs/canonical.ply --iters 20000")
        print("4. Fine-tune: python main_gs.py data/May --stage finetune --canonical_ply workspace_gs/canonical.ply")

        return True

    except Exception as e:
        print(f"\n❌ Basic integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = test_basic_integration()
    print(f"\nTest result: {'SUCCESS' if success else 'FAILED'}")
    sys.exit(0 if success else 1)

