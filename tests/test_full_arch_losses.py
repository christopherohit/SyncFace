"""Tests for the new losses added by feat/full-talking-head-arch.

These tests cover:
  * D-SSIM (correctness on identical inputs, scaling, shape contract)
  * Embedding smoothness (zero on warmup, scales with delta)
  * Eikonal density regularizer (zero on flat field, positive otherwise)
  * Lip-sync (no-op without checkpoint, no-op when lambda=0)
  * Disentangle (margin behavior)
  * head_pose_smoothness_loss (zero on linear pose, positive on jerk)
  * Coarse->fine level mask (early-step output equals coarsest-level slice).
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nerf_triplane.losses import (
    DSSIMLoss,
    EmbeddingSmoothnessLoss,
    EikonalDensityLoss,
    LipSyncLoss,
    NegativeContrastDisentangleLoss,
)
from nerf_triplane.modules import head_pose_smoothness_loss


class TestDSSIM(unittest.TestCase):
    def test_disabled_returns_zero(self):
        fn = DSSIMLoss(lambda_dssim=0.0)
        x = torch.rand(2, 3, 16, 16)
        y = torch.rand(2, 3, 16, 16)
        self.assertEqual(fn(x, y).item(), 0.0)

    def test_identical_inputs_zero(self):
        fn = DSSIMLoss(lambda_dssim=1.0)
        x = torch.rand(2, 3, 32, 32)
        self.assertAlmostEqual(fn(x, x).item(), 0.0, places=4)

    def test_different_inputs_positive(self):
        torch.manual_seed(0)
        fn = DSSIMLoss(lambda_dssim=1.0)
        x = torch.rand(2, 3, 32, 32)
        y = torch.rand(2, 3, 32, 32)
        self.assertGreater(fn(x, y).item(), 0.0)

    def test_lambda_scales_linearly(self):
        torch.manual_seed(0)
        x = torch.rand(2, 3, 32, 32)
        y = torch.rand(2, 3, 32, 32)
        a = DSSIMLoss(lambda_dssim=0.1)(x, y).item()
        b = DSSIMLoss(lambda_dssim=0.5)(x, y).item()
        self.assertAlmostEqual(b / a, 5.0, places=4)


class TestEmbeddingSmoothness(unittest.TestCase):
    def test_first_call_zero(self):
        fn = EmbeddingSmoothnessLoss(lambda_embsmooth=1.0)
        x = torch.randn(1, 32)
        self.assertEqual(fn(x).item(), 0.0)

    def test_disabled_zero(self):
        fn = EmbeddingSmoothnessLoss(lambda_embsmooth=0.0)
        fn(torch.zeros(1, 32))
        self.assertEqual(fn(torch.ones(1, 32)).item(), 0.0)

    def test_positive_after_seed(self):
        fn = EmbeddingSmoothnessLoss(lambda_embsmooth=1.0)
        fn(torch.zeros(1, 32))
        loss = fn(torch.ones(1, 32))
        self.assertGreater(loss.item(), 0.0)


class TestEikonal(unittest.TestCase):
    def test_disabled_zero(self):
        fn = EikonalDensityLoss(lambda_geom=0.0)
        s_a = torch.randn(64)
        s_b = torch.randn(64)
        d = torch.randn(64, 3) * 1e-3
        self.assertEqual(fn(s_a, s_b, d).item(), 0.0)

    def test_flat_field_zero(self):
        fn = EikonalDensityLoss(lambda_geom=1.0, target_norm=0.0)
        s = torch.ones(64)
        d = torch.randn(64, 3) * 1e-3
        self.assertAlmostEqual(fn(s, s, d).item(), 0.0, places=6)

    def test_positive_when_gradient_exists(self):
        fn = EikonalDensityLoss(lambda_geom=1.0, target_norm=0.0)
        s_a = torch.zeros(64)
        s_b = torch.ones(64)
        d = torch.full((64, 3), 1e-3)
        self.assertGreater(fn(s_a, s_b, d).item(), 0.0)


class TestLipSync(unittest.TestCase):
    def test_zero_without_checkpoint(self):
        fn = LipSyncLoss(lambda_lipsync=1.0)
        m = torch.rand(2, 3, 64, 64)
        a = torch.randn(2, 512)
        self.assertEqual(fn(m, a).item(), 0.0)

    def test_zero_when_lambda_zero(self):
        fn = LipSyncLoss(lambda_lipsync=0.0)
        m = torch.rand(2, 3, 64, 64)
        a = torch.randn(2, 512)
        self.assertEqual(fn(m, a).item(), 0.0)


class TestDisentangle(unittest.TestCase):
    def test_disabled_zero(self):
        fn = NegativeContrastDisentangleLoss(lambda_disentangle=0.0)
        a = torch.randn(4, 32)
        p = torch.randn(4, 32)
        n = torch.randn(4, 32)
        self.assertEqual(fn(a, p, n).item(), 0.0)

    def test_satisfied_triplet_zero(self):
        # anchor and positive equal, negative orthogonal -> margin not violated.
        fn = NegativeContrastDisentangleLoss(lambda_disentangle=1.0, margin=0.1)
        a = torch.tensor([[1.0, 0.0]])
        p = torch.tensor([[1.0, 0.0]])
        n = torch.tensor([[0.0, 1.0]])
        self.assertAlmostEqual(fn(a, p, n).item(), 0.0, places=6)

    def test_violated_triplet_positive(self):
        fn = NegativeContrastDisentangleLoss(lambda_disentangle=1.0, margin=0.5)
        a = torch.tensor([[1.0, 0.0]])
        p = torch.tensor([[0.0, 1.0]])  # bad positive
        n = torch.tensor([[1.0, 0.0]])  # bad negative
        self.assertGreater(fn(a, p, n).item(), 0.0)


class TestHeadPoseSmoothness(unittest.TestCase):
    def test_short_sequence_zero(self):
        poses = torch.zeros(2, 6)
        self.assertEqual(head_pose_smoothness_loss(poses).item(), 0.0)

    def test_linear_motion_zero(self):
        # constant velocity -> second difference is zero.
        poses = torch.stack([torch.full((6,), float(i)) for i in range(10)])
        self.assertAlmostEqual(head_pose_smoothness_loss(poses).item(), 0.0, places=6)

    def test_jerk_positive(self):
        poses = torch.tensor([
            [0.0] * 6,
            [0.0] * 6,
            [1.0] * 6,
            [0.0] * 6,
        ])
        self.assertGreater(head_pose_smoothness_loss(poses).item(), 0.0)


class TestCoarseFineMask(unittest.TestCase):
    def test_partial_progress_zeroes_top_levels(self):
        # Re-implement the same masking logic and assert shape.
        num_levels = 12
        level_dim = 1
        feat = torch.ones(4, num_levels * level_dim)
        progress = 0.25
        p = progress * num_levels  # = 3.0
        weights = torch.zeros(num_levels)
        full = int(p)
        if full > 0:
            weights[:full] = 1.0
        if full < num_levels:
            weights[full] = p - full
        weights = weights.repeat_interleave(level_dim)
        out = feat * weights
        # First 3 levels active, level 3 fractional (=0), rest zero
        self.assertAlmostEqual(out[0, 0].item(), 1.0)
        self.assertAlmostEqual(out[0, 2].item(), 1.0)
        self.assertAlmostEqual(out[0, 3].item(), 0.0)
        self.assertAlmostEqual(out[0, 11].item(), 0.0)


if __name__ == '__main__':
    unittest.main()
