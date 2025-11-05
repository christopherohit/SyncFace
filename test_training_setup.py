#!/usr/bin/env python
"""
Test script to verify that the 3DGS training setup works correctly
without requiring full dataset preprocessing
"""

import os
import sys
import torch
import numpy as np
from unittest.mock import Mock, MagicMock

# Mock the dataset loading to avoid needing preprocessed data
def mock_create_gaussian_data_loader(*args, **kwargs):
    """Mock data loader that returns dummy data"""
    mock_loader = Mock()
    mock_loader.has_gt = True

    # Create a mock dataset that returns dummy batches
    mock_dataset = Mock()

    def mock_collate(index):
        """Return a mock batch"""
        batch_size = len(index)
        return {
            'images': torch.randn(batch_size, 3, 256, 256),  # [B, C, H, W]
            'auds': torch.randn(batch_size, 32, 16),  # [B, audio_dim, seq_len]
            'eye': torch.randn(batch_size, 6),  # [B, eye_dim]
            'poses': torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1),  # [B, 4, 4]
            'H': 256,
            'W': 256,
            'K': torch.tensor([[[1000, 0, 128], [0, 1000, 128], [0, 0, 1], [0, 0, 0]]], dtype=torch.float32).repeat(batch_size, 1, 1),  # [B, 4, 4]
            'face_mask': torch.ones(batch_size, 256, 256, dtype=torch.bool),  # [B, H, W]
        }

    mock_dataset.collate = mock_collate
    mock_loader._data = mock_dataset

    # Create a simple iterator class
    class MockIterator:
        def __init__(self, collate_fn):
            self.collate_fn = collate_fn
            self.index = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.index < 1:  # Return one batch
                self.index += 1
                return self.collate_fn([0])
            else:
                raise StopIteration

    mock_loader.__iter__ = lambda: MockIterator(mock_collate)
    mock_loader.__len__ = lambda: 1
    return mock_loader

def test_training_setup():
    """Test that the training setup works with mock data"""
    print("="*60)
    print("Testing 3DGS Training Setup")
    print("="*60)

    # Temporarily replace the data loader with our mock
    original_loader = __import__('nerf_triplane.provider_gs').provider_gs.create_gaussian_data_loader
    __import__('nerf_triplane.provider_gs').provider_gs.create_gaussian_data_loader = mock_create_gaussian_data_loader

    try:
        from nerf_triplane.network_gs import DynamicGaussianNetwork
        from nerf_triplane.trainer_gs import GaussianTrainer

        # Create config
        class Config:
            def __init__(self):
                # Model config
                self.sh_degree = 0
                self.canonical_ply = ''
                self.audio_dim = 32
                self.eye_dim = 6
                self.exp_eye = True
                self.att = 2
                self.asr = False
                self.asr_model = 'ave'
                self.num_rays = -1

                # Training config
                self.lambda_ssim = 0.2
                self.finetune_lips = False
                self.lr = 1e-3
                self.lr_net = 5e-4
                self.lr_gaussian = 0.0
                self.lr_decay_steps = 10000
                self.lr_decay_gamma = 0.5
                self.iters = 100  # Small number for testing

                # Data config
                self.path = 'dummy_path'
                self.downscale = 1
                self.exp_eye = True
                self.preload = 0

        config = Config()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("✓ Creating DynamicGaussianNetwork...")
        model = DynamicGaussianNetwork(config)

        print("✓ Creating mock data loader...")
        train_loader = mock_create_gaussian_data_loader(config, device, type='train')

        print("✓ Creating GaussianTrainer...")
        trainer = GaussianTrainer(
            name='test_training',
            opt=config,
            model=model,
            device=device,
            workspace='test_workspace',
            fp16=False
        )

        print("✓ Testing training step...")
        # Create mock data directly
        mock_data = {
            'images': torch.randn(1, 3, 64, 64, dtype=torch.float32),  # [B, C, H, W] - smaller for testing
            'auds': torch.randn(1, 32, 16, dtype=torch.float32),  # [B, audio_dim, seq_len]
            'eye': torch.randn(1, 6, dtype=torch.float32),  # [B, eye_dim]
            'poses': torch.eye(4, dtype=torch.float32).unsqueeze(0),  # [B, 4, 4]
            'H': 64,
            'W': 64,
            'K': torch.tensor([[[1000, 0, 32], [0, 1000, 32], [0, 0, 1], [0, 0, 0]]], dtype=torch.float32),  # [B, 4, 4]
            'face_mask': torch.ones(1, 64, 64, dtype=torch.bool),  # [B, H, W]
        }

        loss, pred = trainer.train_step(mock_data)
        print(f"  Training step successful! Loss: {loss.item():.6f}")
        print(f"  Prediction shape: {pred.shape}")

        print("\n✅ Training setup test PASSED!")
        return True

    except Exception as e:
        print(f"\n❌ Training setup test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Restore original loader
        __import__('nerf_triplane.provider_gs').provider_gs.create_gaussian_data_loader = original_loader


if __name__ == '__main__':
    success = test_training_setup()
    sys.exit(0 if success else 1)
