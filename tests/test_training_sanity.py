"""
Sanity check for end-to-end training with synthetic data.

Tests that pre-training and adaptation can run without errors.
Uses random tensors instead of real data.

Run with: python tests/test_training_sanity.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from typing import Dict

from nerf_triplane.instag_network import InsTaGNetwork
from nerf_triplane.instag_losses import NegativeContrastLoss, GeometryPriorRegularizer


def create_mock_opt() -> argparse.Namespace:
    """Create mock options for testing."""
    opt = argparse.Namespace()
    
    # Model options
    opt.audio_dim = 32
    opt.asr_model = 'deepspeech'
    opt.att = 0  # Disable attention for testing (simpler)
    opt.emb = False
    opt.exp_eye = True
    opt.au45 = False
    opt.bs_area = 'upper'
    opt.bound = 1.0
    
    # Individual codes
    opt.ind_dim = 4
    opt.ind_num = 100
    opt.ind_dim_torso = 8
    
    # Training options
    opt.fp16 = False
    opt.cuda_ray = True
    opt.max_steps = 16
    opt.num_rays = 4096
    opt.update_extra_interval = 16
    
    # Loss weights
    opt.lambda_amb = 1e-4
    opt.amb_aud_loss = 1
    opt.amb_eye_loss = 1
    opt.unc_loss = 1
    
    # Other
    opt.torso = False
    opt.color_space = 'srgb'
    opt.test_train = False
    opt.smooth_lips = False
    opt.train_camera = False
    opt.density_thresh = 10
    opt.density_thresh_torso = 0.01
    opt.min_near = 0.05
    opt.dt_gamma = 1/256
    opt.scale = 4
    opt.offset = [0, 0, 0]
    opt.preload = 0
    opt.data_range = [0, -1]
    
    return opt


def create_synthetic_batch(batch_size: int = 1, num_rays: int = 1024, 
                          device='cpu') -> Dict[str, torch.Tensor]:
    """Create synthetic batch data for testing."""
    H, W = 256, 256
    
    batch = {
        'rays_o': torch.randn(batch_size, num_rays, 3, device=device),
        'rays_d': torch.randn(batch_size, num_rays, 3, device=device),
        'images': torch.rand(batch_size, num_rays, 3, device=device),  # RGB in [0, 1]
        'auds': torch.randn(batch_size, 29, 16, device=device),  # DeepSpeech features
        'eye': torch.randn(batch_size, 7, device=device),  # Eye features
        'bg_coords': torch.rand(batch_size, num_rays, 2, device=device) * 2 - 1,  # [-1, 1]
        'poses': torch.eye(4, device=device).unsqueeze(0),  # Identity pose
        'index': 0,
        'H': H,
        'W': W,
    }
    
    # Normalize ray directions
    batch['rays_d'] = batch['rays_d'] / batch['rays_d'].norm(dim=-1, keepdim=True)
    
    return batch


def test_pretrain_step():
    """Test one pre-training step with multiple identities."""
    print("\n" + "="*60)
    print("Test: Pre-training Step")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create model
    opt = create_mock_opt()
    model = InsTaGNetwork(opt, phase='pretrain').to(device)
    
    # Add 3 identities
    identities = ['person1', 'person2', 'person3']
    for identity_id in identities:
        model.add_identity(identity_id)
    print(f"Added {len(identities)} identities")
    
    # Loss and optimizer
    criterion = nn.L1Loss(reduction='mean')
    nc_loss = NegativeContrastLoss(lambda_c=0.01)
    
    optimizer = optim.AdamW(model.get_params(lr=1e-2, lr_net=1e-3), lr=1e-3)
    
    # Training steps for each identity
    print("\nRunning training steps...")
    total_loss = 0.0
    delta_personal_dict = {}
    
    for i, identity_id in enumerate(identities):
        # Create synthetic batch
        batch = create_synthetic_batch(batch_size=1, num_rays=512, device=device)
        
        # Forward pass (simplified, without full renderer)
        N = 128  # Sample points
        x = torch.randn(N, 3, device=device) * 0.5
        d = torch.randn(N, 3, device=device)
        d = d / d.norm(dim=-1, keepdim=True)
        
        enc_a = model.encode_audio(batch['auds'])
        eye = batch['eye']
        
        # Forward through network
        sigma, color, delta_x = model.forward(x, d, enc_a, None, eye, identity_id)
        
        # Simple photometric loss (mock)
        target_color = torch.rand_like(color)
        loss_rgb = criterion(color, target_color)
        
        # Store delta_personal for NCLoss
        with torch.no_grad():
            f_l = enc_a.repeat(x.shape[0], 1) if enc_a is not None else torch.zeros(x.shape[0], model.audio_dim, device=device)
            f_e = eye.repeat(x.shape[0], 1) if eye is not None else None
            delta_personal = model.personalized_fields[identity_id](x, f_l, f_e, bound=model.bound)
            delta_personal_dict[identity_id] = delta_personal
        
        total_loss += loss_rgb
        
        print(f"  Identity {identity_id}: loss_rgb={loss_rgb.item():.4f}, "
              f"sigma range=[{sigma.min().item():.2f}, {sigma.max().item():.2f}], "
              f"color range=[{color.min().item():.3f}, {color.max().item():.3f}]")
    
    # Compute NCLoss
    loss_nc = nc_loss(delta_personal_dict)
    total_loss = total_loss + loss_nc
    
    print(f"\n  Total loss: {total_loss.item():.4f} (NC loss: {loss_nc.item():.6f})")
    
    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    print("\n✓ Pre-training step completed successfully")
    return True


def test_adaptation_step():
    """Test one adaptation step with frozen UMF."""
    print("\n" + "="*60)
    print("Test: Adaptation Step")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Create model in adaptation mode
    opt = create_mock_opt()
    model = InsTaGNetwork(opt, phase='adapt').to(device)
    
    print("Model created in adaptation mode")
    
    # Freeze UMF (simulates loading pre-trained checkpoint)
    model.freeze_umf()
    print("UMF frozen")
    
    # Loss and optimizer
    criterion = nn.L1Loss(reduction='mean')
    geo_regularizer = GeometryPriorRegularizer(lambda_d=0.1, lambda_n=0.05).to(device)
    
    optimizer = optim.AdamW(model.get_params(lr=2e-2, lr_net=5e-3), lr=1e-3)
    
    # Create synthetic batch
    batch = create_synthetic_batch(batch_size=1, num_rays=512, device=device)
    
    # Forward pass
    N = 128
    x = torch.randn(N, 3, device=device) * 0.5
    d = torch.randn(N, 3, device=device)
    d = d / d.norm(dim=-1, keepdim=True)
    
    enc_a = model.encode_audio(batch['auds'])
    eye = batch['eye']
    
    print("\nForward pass...")
    sigma, color, delta_x = model.forward(x, d, enc_a, None, eye)
    
    # Photometric loss
    target_color = torch.rand_like(color)
    loss_rgb = criterion(color, target_color)
    
    # Geometry prior loss (with synthetic depth)
    d_pred = torch.rand(N, device=device) * 2 + 0.5
    d_est = d_pred * 1.2 + 0.1
    geo_losses = geo_regularizer(d_pred, d_est)
    loss_geo = geo_losses['total']
    
    total_loss = loss_rgb + loss_geo
    
    print(f"  loss_rgb: {loss_rgb.item():.4f}")
    print(f"  loss_geo: {loss_geo.item():.4f}")
    print(f"  total_loss: {total_loss.item():.4f}")
    print(f"  Sigma range: [{sigma.min().item():.2f}, {sigma.max().item():.2f}]")
    print(f"  Color range: [{color.min().item():.3f}, {color.max().item():.3f}]")
    print(f"  Deformation magnitude: {delta_x.norm(dim=-1).mean().item():.4f}")
    
    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    
    # Check UMF gradients are None (frozen)
    umf_has_grad = any(p.grad is not None for p in model.umf.parameters() if p.requires_grad)
    assert not umf_has_grad, "UMF should not have gradients (frozen)"
    
    optimizer.step()
    
    print("\n✓ Adaptation step completed successfully")
    print("✓ UMF gradients confirmed frozen")
    return True


def test_checkpoint_save_load():
    """Test saving and loading UMF checkpoint."""
    print("\n" + "="*60)
    print("Test: Checkpoint Save/Load")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model and add identity
    opt = create_mock_opt()
    model1 = InsTaGNetwork(opt, phase='pretrain').to(device)
    model1.add_identity('test_person')
    
    # Save UMF checkpoint
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.pth') as f:
        ckpt_path = f.name
    
    print(f"Saving UMF to {ckpt_path}...")
    model1.save_umf_checkpoint(ckpt_path)
    
    # Create new model in adaptation mode
    model2 = InsTaGNetwork(opt, phase='adapt').to(device)
    
    # Load UMF
    print(f"Loading UMF from {ckpt_path}...")
    model2.load_umf_checkpoint(ckpt_path)
    
    # Verify UMF parameters are frozen
    umf_frozen = all(not p.requires_grad for p in model2.umf.parameters())
    assert umf_frozen, "UMF parameters should be frozen after loading"
    
    # Compare parameters
    for (name1, p1), (name2, p2) in zip(model1.umf.named_parameters(), model2.umf.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-6), f"Parameter {name1} mismatch"
    
    # Clean up
    os.remove(ckpt_path)
    
    print("\n✓ Checkpoint save/load successful")
    print("✓ UMF parameters verified")
    return True


def test_motion_aligner_adapter():
    """Test Motion Aligner functionality."""
    print("\n" + "="*60)
    print("Test: Motion Aligner Adapter")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    opt = create_mock_opt()
    model = InsTaGNetwork(opt, phase='adapt').to(device)
    
    # Test with and without motion aligner
    N = 256
    x = torch.randn(N, 3, device=device) * 0.5
    f_l = torch.randn(N, 32, device=device)
    f_e = torch.randn(N, 7, device=device)
    
    print("\nComputing deformation with Motion Aligner...")
    
    # With motion aligner (should apply offset and scale)
    if model.motion_aligner is not None:
        Delta_x_A, tau_A = model.motion_aligner(x, bound=model.bound)
        print(f"  Delta_x_A range: [{Delta_x_A.min().item():.4f}, {Delta_x_A.max().item():.4f}]")
        print(f"  tau_A range: [{tau_A.min().item():.4f}, {tau_A.max().item():.4f}]")
        print(f"  tau_A mean: {tau_A.mean().item():.4f} (should be ~1.0)")
        
        x_aligned = x + Delta_x_A
        delta_universal = model.umf(x_aligned, f_l, f_e, bound=model.bound)
        delta_universal_scaled = delta_universal * tau_A
        
        print(f"  delta_universal magnitude: {delta_universal.norm(dim=-1).mean().item():.4f}")
        print(f"  delta_universal_scaled magnitude: {delta_universal_scaled.norm(dim=-1).mean().item():.4f}")
    
    # Full deformation
    delta_x = model.compute_deformation(x, f_l[0:1], f_e[0:1])
    print(f"  Total deformation magnitude: {delta_x.norm(dim=-1).mean().item():.4f}")
    
    print("\n✓ Motion Aligner test passed")
    return True


def run_all_sanity_checks():
    """Run all sanity checks."""
    print("\n" + "="*70)
    print(" " * 15 + "InsTaG Training Sanity Checks")
    print("="*70)
    
    tests = [
        ("Pre-training Step", test_pretrain_step),
        ("Adaptation Step", test_adaptation_step),
        ("Checkpoint Save/Load", test_checkpoint_save_load),
        ("Motion Aligner Adapter", test_motion_aligner_adapter),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} FAILED with exception:")
            print(f"  {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("Sanity Check Summary")
    print("="*70)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*70)
    
    return passed == total


if __name__ == '__main__':
    success = run_all_sanity_checks()
    
    if success:
        print("\n🎉 All sanity checks passed!")
        print("The InsTaG implementation is ready for training.")
    else:
        print("\n⚠️  Some sanity checks failed. Review errors above.")
    
    exit(0 if success else 1)

