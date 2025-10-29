"""
Unit tests for InsTaG-inspired modules.

Tests module shapes, forward passes, and integration.
Run with: python -m pytest tests/test_instag_modules.py -v
Or: python tests/test_instag_modules.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from typing import Dict

from nerf_triplane.instag_modules import (
    UniversalMotionField,
    PersonalizedMotionField,
    StaticField,
    MotionAligner,
    FaceMouthHook,
    create_lip_mask
)
from nerf_triplane.instag_losses import (
    NegativeContrastLoss,
    ScaleInvariantDepthLoss,
    NormalConsistencyLoss,
    GeometryPriorRegularizer
)


class TestUniversalMotionField:
    """Test UMF module."""
    
    def test_initialization(self):
        """Test UMF can be created."""
        umf = UniversalMotionField(
            audio_dim=32,
            eye_dim=4,
            hidden_dim=64,
            num_layers=3
        )
        assert umf is not None
        print("✓ UMF initialization passed")
    
    def test_forward_pass(self):
        """Test UMF forward pass with correct shapes."""
        if not torch.cuda.is_available():
            print("⊘ UMF forward pass skipped (requires CUDA)")
            return
            
        device = 'cuda'
        umf = UniversalMotionField(audio_dim=32, eye_dim=4).to(device)
        
        # Inputs
        N = 1024
        x = torch.randn(N, 3, device=device)  # Spatial coordinates
        f_l = torch.randn(1, 32, device=device)  # Audio features
        f_e = torch.randn(1, 4, device=device)  # Eye features
        
        # Forward
        delta_x = umf(x, f_l.repeat(N, 1), f_e.repeat(N, 1))
        
        # Check output shape
        assert delta_x.shape == (N, 3), f"Expected {(N, 3)}, got {delta_x.shape}"
        print(f"✓ UMF forward pass: {x.shape} -> {delta_x.shape}")
    
    def test_without_eye_features(self):
        """Test UMF works without eye features."""
        if not torch.cuda.is_available():
            print("⊘ UMF without eye features skipped (requires CUDA)")
            return
            
        device = 'cuda'
        umf = UniversalMotionField(audio_dim=32, eye_dim=0).to(device)
        
        N = 512
        x = torch.randn(N, 3, device=device)
        f_l = torch.randn(N, 32, device=device)
        
        delta_x = umf(x, f_l, f_e=None)
        assert delta_x.shape == (N, 3)
        print("✓ UMF without eye features passed")


class TestPersonalizedMotionField:
    """Test PersonalizedField module."""
    
    def test_initialization(self):
        """Test PersonalizedField can be created."""
        pf = PersonalizedMotionField(audio_dim=32, eye_dim=4)
        assert pf is not None
        print("✓ PersonalizedField initialization passed")
    
    def test_forward_pass(self):
        """Test PersonalizedField forward pass."""
        if not torch.cuda.is_available():
            print("⊘ PersonalizedField forward pass skipped (requires CUDA)")
            return
            
        device = 'cuda'
        pf = PersonalizedMotionField(audio_dim=32, eye_dim=4, hidden_dim=32, num_layers=2).to(device)
        
        N = 256
        x = torch.randn(N, 3, device=device)
        f_l = torch.randn(N, 32, device=device)
        f_e = torch.randn(N, 4, device=device)
        
        delta_personal = pf(x, f_l, f_e)
        
        assert delta_personal.shape == (N, 3)
        print(f"✓ PersonalizedField forward: {x.shape} -> {delta_personal.shape}")
    
    def test_smaller_capacity(self):
        """Verify PersonalizedField has fewer parameters than UMF."""
        umf = UniversalMotionField(audio_dim=32, eye_dim=4, hidden_dim=64, num_layers=3)
        pf = PersonalizedMotionField(audio_dim=32, eye_dim=4, hidden_dim=32, num_layers=2)
        
        umf_params = sum(p.numel() for p in umf.parameters())
        pf_params = sum(p.numel() for p in pf.parameters())
        
        assert pf_params < umf_params, f"PersonalizedField should be smaller: {pf_params} vs {umf_params}"
        print(f"✓ Parameter count: UMF={umf_params}, PersonalizedField={pf_params} (ratio={pf_params/umf_params:.2f})")


class TestStaticField:
    """Test StaticField module."""
    
    def test_initialization(self):
        """Test StaticField can be created."""
        sf = StaticField(individual_dim=4, geo_feat_dim=64)
        assert sf is not None
        print("✓ StaticField initialization passed")
    
    def test_forward_pass(self):
        """Test StaticField forward pass."""
        if not torch.cuda.is_available():
            print("⊘ StaticField forward pass skipped (requires CUDA)")
            return
            
        device = 'cuda'
        sf = StaticField(individual_dim=4, geo_feat_dim=64, bound=1.0).to(device)
        
        N = 512
        x_deformed = torch.randn(N, 3, device=device) * 0.5  # Deformed coordinates in [-bound, bound]
        d = torch.randn(N, 3, device=device)
        d = d / d.norm(dim=-1, keepdim=True)  # Normalize directions
        c = torch.randn(1, 4, device=device)  # Individual code
        
        sigma, color = sf(x_deformed, d, c)
        
        assert sigma.shape == (N,), f"Expected sigma shape {(N,)}, got {sigma.shape}"
        assert color.shape == (N, 3), f"Expected color shape {(N, 3)}, got {color.shape}"
        assert (sigma >= 0).all(), "Sigma should be non-negative"
        assert (color >= 0).all() and (color <= 1).all(), "Color should be in [0, 1]"
        
        print(f"✓ StaticField forward: deformed_x={x_deformed.shape}, d={d.shape} -> sigma={sigma.shape}, color={color.shape}")
    
    def test_without_individual_code(self):
        """Test StaticField works without individual code."""
        if not torch.cuda.is_available():
            print("⊘ StaticField without individual code skipped (requires CUDA)")
            return
            
        device = 'cuda'
        sf = StaticField(individual_dim=0, geo_feat_dim=64).to(device)
        
        N = 128
        x = torch.randn(N, 3, device=device) * 0.5
        d = torch.randn(N, 3, device=device)
        d = d / d.norm(dim=-1, keepdim=True)
        
        sigma, color = sf(x, d, c=None)
        assert sigma.shape == (N,)
        assert color.shape == (N, 3)
        print("✓ StaticField without individual code passed")


class TestMotionAligner:
    """Test MotionAligner module."""
    
    def test_initialization(self):
        """Test MotionAligner can be created."""
        ma = MotionAligner(hidden_dim=64, num_layers=2)
        assert ma is not None
        print("✓ MotionAligner initialization passed")
    
    def test_forward_pass(self):
        """Test MotionAligner forward pass."""
        if not torch.cuda.is_available():
            print("⊘ MotionAligner forward pass skipped (requires CUDA)")
            return
            
        device = 'cuda'
        ma = MotionAligner(hidden_dim=64, num_layers=2).to(device)
        
        N = 256
        x = torch.randn(N, 3, device=device)
        
        Delta_x_A, tau_A = ma(x)
        
        assert Delta_x_A.shape == (N, 3), f"Expected Delta_x_A {(N, 3)}, got {Delta_x_A.shape}"
        assert tau_A.shape == (N, 3), f"Expected tau_A {(N, 3)}, got {tau_A.shape}"
        
        # Check tau_A is in reasonable range (around 1.0)
        assert (tau_A > 0).all() and (tau_A < 3).all(), f"tau_A should be in (0, 3), got range [{tau_A.min()}, {tau_A.max()}]"
        
        print(f"✓ MotionAligner forward: {x.shape} -> Delta_x_A={Delta_x_A.shape}, tau_A={tau_A.shape}")
        print(f"  tau_A range: [{tau_A.min().item():.3f}, {tau_A.max().item():.3f}]")


class TestFaceMouthHook:
    """Test Face-Mouth Hook module."""
    
    def test_initialization(self):
        """Test FaceMouthHook can be created."""
        fm_hook = FaceMouthHook(hook_dim=9)
        assert fm_hook is not None
        print("✓ FaceMouthHook initialization passed")
    
    def test_hook_feature_computation(self):
        """Test hook feature computation."""
        fm_hook = FaceMouthHook(hook_dim=9)
        
        N = 1024
        delta_x_face = torch.randn(N, 3)
        lip_mask = torch.zeros(N, dtype=torch.bool)
        lip_mask[300:500] = True  # 200 lip points
        
        phi = fm_hook(delta_x_face, lip_mask)
        
        assert phi.shape == (1, 9), f"Expected {(1, 9)}, got {phi.shape}"
        print(f"✓ FM Hook feature: delta_x={delta_x_face.shape}, mask={lip_mask.sum()} points -> phi={phi.shape}")
    
    def test_lip_mask_creation(self):
        """Test lip mask creation utility."""
        N = 1024
        points = torch.randn(N, 3)
        
        # Place some points in lip region
        points[100:200, 1] = -0.1  # y coordinate (vertical)
        points[100:200, 2] = 0.0   # z coordinate (depth)
        
        mask = create_lip_mask(points)
        
        assert mask.shape == (N,)
        assert mask.sum() > 0, "Should detect some lip points"
        print(f"✓ Lip mask creation: {N} points -> {mask.sum()} lip points")


class TestLossFunctions:
    """Test loss functions."""
    
    def test_negative_contrast_loss(self):
        """Test NCLoss computation."""
        nc_loss = NegativeContrastLoss(lambda_c=0.01)
        
        # Create mock personalized field outputs for 3 identities
        delta_personal_dict = {
            'id1': torch.randn(512, 3),
            'id2': torch.randn(512, 3),
            'id3': torch.randn(512, 3),
        }
        
        loss = nc_loss(delta_personal_dict)
        
        assert loss.item() >= 0, "NCLoss should be non-negative"
        print(f"✓ NCLoss: 3 identities -> loss={loss.item():.6f}")
    
    def test_scale_invariant_depth_loss(self):
        """Test scale-invariant depth loss."""
        depth_loss = ScaleInvariantDepthLoss(alpha=0.5)
        
        N = 1024
        d_pred = torch.rand(N) + 0.1  # Predicted depth
        d_est = d_pred * 1.5 + 0.2    # Scaled and shifted
        
        loss = depth_loss(d_pred, d_est)
        
        assert loss.item() >= 0, "Depth loss should be non-negative"
        print(f"✓ Scale-invariant depth loss: {N} points -> loss={loss.item():.6f}")
    
    def test_normal_consistency_loss(self):
        """Test normal consistency loss."""
        normal_loss = NormalConsistencyLoss()
        
        N = 512
        n_pred = torch.randn(N, 3)
        n_pred = n_pred / n_pred.norm(dim=-1, keepdim=True)  # Normalize
        
        # Create similar normals (should have low loss)
        n_est = n_pred + torch.randn(N, 3) * 0.1
        n_est = n_est / n_est.norm(dim=-1, keepdim=True)
        
        loss = normal_loss(n_pred, n_est)
        
        assert 0 <= loss.item() <= 2, f"Normal loss should be in [0, 2], got {loss.item()}"
        print(f"✓ Normal consistency loss: {N} normals -> loss={loss.item():.6f}")
    
    def test_geometry_prior_regularizer(self):
        """Test combined geometry prior."""
        geo_reg = GeometryPriorRegularizer(lambda_d=0.1, lambda_n=0.05)
        
        N = 1024
        d_pred = torch.rand(N) + 0.1
        d_est = d_pred * 1.2 + 0.1
        n_pred = torch.randn(N, 3)
        n_pred = n_pred / n_pred.norm(dim=-1, keepdim=True)
        n_est = n_pred + torch.randn(N, 3) * 0.2
        n_est = n_est / n_est.norm(dim=-1, keepdim=True)
        
        loss_dict = geo_reg(d_pred, d_est, n_pred, n_est)
        
        assert 'total' in loss_dict
        assert 'depth' in loss_dict
        assert 'normal' in loss_dict
        assert loss_dict['total'].item() >= 0
        
        print(f"✓ Geometry prior: depth={loss_dict['depth'].item():.4f}, "
              f"normal={loss_dict['normal'].item():.4f}, total={loss_dict['total'].item():.4f}")


class TestIntegration:
    """Test integration of modules."""
    
    def test_deformation_pipeline(self):
        """Test complete deformation pipeline."""
        if not torch.cuda.is_available():
            print("⊘ Deformation pipeline skipped (requires CUDA)")
            return
            
        device = 'cuda'
        # Create modules
        umf = UniversalMotionField(audio_dim=32, eye_dim=4).to(device)
        pf = PersonalizedMotionField(audio_dim=32, eye_dim=4).to(device)
        ma = MotionAligner().to(device)
        
        # Inputs
        N = 512
        x = torch.randn(N, 3, device=device) * 0.5
        f_l = torch.randn(N, 32, device=device)
        f_e = torch.randn(N, 4, device=device)
        
        # Pipeline: x -> MotionAligner -> UMF + PersonalizedField -> delta_x
        Delta_x_A, tau_A = ma(x)
        x_aligned = x + Delta_x_A
        delta_univ = umf(x_aligned, f_l, f_e) * tau_A
        delta_personal = pf(x, f_l, f_e)
        delta_x_total = delta_univ + delta_personal
        
        assert delta_x_total.shape == (N, 3)
        print(f"✓ Deformation pipeline: x={x.shape} -> delta_x={delta_x_total.shape}")
        print(f"  Mean deformation magnitude: {delta_x_total.norm(dim=-1).mean().item():.4f}")
    
    def test_rendering_pipeline(self):
        """Test complete rendering pipeline."""
        if not torch.cuda.is_available():
            print("⊘ Rendering pipeline skipped (requires CUDA)")
            return
            
        device = 'cuda'
        # Create modules
        umf = UniversalMotionField(audio_dim=32, eye_dim=4).to(device)
        pf = PersonalizedMotionField(audio_dim=32, eye_dim=4).to(device)
        sf = StaticField(individual_dim=4).to(device)
        
        # Inputs
        N = 256
        x = torch.randn(N, 3, device=device) * 0.5
        d = torch.randn(N, 3, device=device)
        d = d / d.norm(dim=-1, keepdim=True)
        f_l = torch.randn(N, 32, device=device)
        f_e = torch.randn(N, 4, device=device)
        c = torch.randn(1, 4, device=device)
        
        # Pipeline: deformation -> static field
        delta_univ = umf(x, f_l, f_e)
        delta_personal = pf(x, f_l, f_e)
        delta_x = delta_univ + delta_personal
        
        x_deformed = x + delta_x
        sigma, color = sf(x_deformed, d, c)
        
        assert sigma.shape == (N,)
        assert color.shape == (N, 3)
        print(f"✓ Rendering pipeline: x={x.shape} -> sigma={sigma.shape}, color={color.shape}")
        print(f"  Sigma range: [{sigma.min().item():.2f}, {sigma.max().item():.2f}]")
        print(f"  Color range: [{color.min().item():.3f}, {color.max().item():.3f}]")


def run_all_tests():
    """Run all test suites."""
    print("\n" + "="*60)
    print("Running InsTaG Module Tests")
    print("="*60 + "\n")
    
    test_suites = [
        ("UniversalMotionField", TestUniversalMotionField),
        ("PersonalizedMotionField", TestPersonalizedMotionField),
        ("StaticField", TestStaticField),
        ("MotionAligner", TestMotionAligner),
        ("FaceMouthHook", TestFaceMouthHook),
        ("Loss Functions", TestLossFunctions),
        ("Integration", TestIntegration),
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for suite_name, test_class in test_suites:
        print(f"\n--- {suite_name} Tests ---")
        test_instance = test_class()
        
        # Get all test methods
        test_methods = [m for m in dir(test_instance) if m.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                method = getattr(test_instance, method_name)
                method()
                passed_tests += 1
            except Exception as e:
                print(f"✗ {method_name} FAILED: {e}")
    
    print("\n" + "="*60)
    print(f"Test Results: {passed_tests}/{total_tests} passed")
    print("="*60)
    
    return passed_tests == total_tests


if __name__ == '__main__':
    success = run_all_tests()
    exit(0 if success else 1)

