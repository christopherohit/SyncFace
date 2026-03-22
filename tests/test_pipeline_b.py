"""Tests for Option B: Temporal Consistency."""

import math
import sys
import os
import unittest

import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nerf_triplane.temporal_loss import TemporalConsistencyLoss


class TestTemporalConsistencyLoss(unittest.TestCase):

    def test_instantiation(self):
        loss_fn = TemporalConsistencyLoss()
        self.assertEqual(loss_fn.warmup_steps, 5000)
        self.assertFalse(loss_fn._has_prev)

    def test_first_frame_returns_zero(self):
        loss_fn = TemporalConsistencyLoss(warmup_steps=0)
        pred = torch.randn(1, 100, 3)
        mask = torch.ones(1, 100, dtype=torch.bool)
        loss = loss_fn(pred, mask, global_step=10000)
        self.assertEqual(loss.item(), 0.0)
        self.assertTrue(loss_fn._has_prev)

    def test_warmup_returns_zero(self):
        loss_fn = TemporalConsistencyLoss(warmup_steps=1000)
        pred1 = torch.randn(1, 100, 3)
        mask = torch.ones(1, 100, dtype=torch.bool)
        loss_fn(pred1, mask, global_step=0)
        loss = loss_fn(torch.randn(1, 100, 3), mask, global_step=500)
        self.assertEqual(loss.item(), 0.0)

    def test_after_warmup_nonzero(self):
        loss_fn = TemporalConsistencyLoss(warmup_steps=0, lambda_temporal=1.0)
        mask = torch.ones(1, 50, dtype=torch.bool)
        loss_fn(torch.zeros(1, 50, 3), mask, global_step=1)
        loss = loss_fn(torch.ones(1, 50, 3), mask, global_step=2)
        self.assertGreater(loss.item(), 0.0)

    def test_identical_frames_zero_loss(self):
        loss_fn = TemporalConsistencyLoss(warmup_steps=0, lambda_temporal=1.0)
        pred = torch.randn(1, 50, 3)
        mask = torch.ones(1, 50, dtype=torch.bool)
        loss_fn(pred.clone(), mask, global_step=1)
        loss = loss_fn(pred.clone(), mask, global_step=2)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_face_mask_zeros_background(self):
        loss_fn = TemporalConsistencyLoss(warmup_steps=0, lambda_temporal=1.0)
        mask = torch.zeros(1, 50, dtype=torch.bool)
        loss_fn(torch.zeros(1, 50, 3), mask, global_step=1)
        loss = loss_fn(torch.ones(1, 50, 3), mask, global_step=2)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_shape_mismatch_returns_zero(self):
        loss_fn = TemporalConsistencyLoss(warmup_steps=0)
        mask1 = torch.ones(1, 50, dtype=torch.bool)
        loss_fn(torch.randn(1, 50, 3), mask1, global_step=1)
        mask2 = torch.ones(1, 100, dtype=torch.bool)
        loss = loss_fn(torch.randn(1, 100, 3), mask2, global_step=2)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_lambda_scaling(self):
        pred1, pred2 = torch.zeros(1, 50, 3), torch.ones(1, 50, 3)
        mask = torch.ones(1, 50, dtype=torch.bool)

        fn_a = TemporalConsistencyLoss(warmup_steps=0, lambda_temporal=0.1)
        fn_a(pred1.clone(), mask, global_step=1)
        loss_a = fn_a(pred2.clone(), mask, global_step=2)

        fn_b = TemporalConsistencyLoss(warmup_steps=0, lambda_temporal=0.5)
        fn_b(pred1.clone(), mask, global_step=1)
        loss_b = fn_b(pred2.clone(), mask, global_step=2)

        self.assertAlmostEqual(loss_b.item() / loss_a.item(), 5.0, places=4)


class TestCosineLRSchedule(unittest.TestCase):

    def _make_scheduler(self, lr=1e-2, warmup_steps=100, total_iters=1000):
        model = nn.Linear(10, 10)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        def cosine_with_warmup(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_iters - warmup_steps)
            return max(1e-6 / lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, cosine_with_warmup)
        return optimizer, scheduler

    def test_warmup_starts_near_zero(self):
        _, sched = self._make_scheduler()
        self.assertAlmostEqual(sched.get_last_lr()[0], 0.0, places=6)

    def test_warmup_midpoint(self):
        _, sched = self._make_scheduler()
        for _ in range(50):
            sched.step()
        self.assertAlmostEqual(sched.get_last_lr()[0], 0.5e-2, places=5)

    def test_warmup_end(self):
        _, sched = self._make_scheduler()
        for _ in range(100):
            sched.step()
        self.assertAlmostEqual(sched.get_last_lr()[0], 1e-2, places=5)

    def test_cosine_decay_end(self):
        _, sched = self._make_scheduler()
        for _ in range(1000):
            sched.step()
        self.assertLess(sched.get_last_lr()[0], 1e-4)

    def test_monotonic_warmup(self):
        _, sched = self._make_scheduler()
        prev = 0.0
        for _ in range(100):
            sched.step()
            lr = sched.get_last_lr()[0]
            self.assertGreaterEqual(lr, prev)
            prev = lr


class TestEarlyStopping(unittest.TestCase):

    def test_triggers_after_patience(self):
        patience, best, counter = 3, float('inf'), 0
        losses = [1.0, 0.9, 0.8, 0.8, 0.8, 0.8]
        stopped_at = None
        for i, loss in enumerate(losses):
            if loss < best - 1e-4:
                best, counter = loss, 0
            else:
                counter += 1
            if counter >= patience:
                stopped_at = i
                break
        self.assertEqual(stopped_at, 5)

    def test_no_stop_when_improving(self):
        patience, best, counter = 3, float('inf'), 0
        stopped = False
        for loss in [1.0, 0.9, 0.8, 0.7, 0.6]:
            if loss < best - 1e-4:
                best, counter = loss, 0
            else:
                counter += 1
            if counter >= patience:
                stopped = True
                break
        self.assertFalse(stopped)


class TestIntegration(unittest.TestCase):

    def test_temporal_loss_in_train_loop(self):
        temporal_loss_fn = TemporalConsistencyLoss(warmup_steps=0, lambda_temporal=0.1)
        model = nn.Linear(3, 3)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.L1Loss()
        mask = torch.ones(1, 64, dtype=torch.bool)

        for step in range(5):
            optimizer.zero_grad()
            pred = model(torch.randn(1, 64, 3))
            loss = criterion(pred, torch.randn(1, 64, 3)) + temporal_loss_fn(pred, mask, step + 1)
            loss.backward()
            optimizer.step()
        self.assertTrue(True)  # no crash


if __name__ == '__main__':
    unittest.main()
