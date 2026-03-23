"""
Benchmark FPS of SyncTalk model during inference (generation).

Measures:
  - Per-frame inference time (model forward + rendering)
  - Total pipeline time (including data loading, postprocessing)
  - Pure rendering FPS (excluding I/O)
  - GPU memory usage

Usage:
  python scripts/benchmark_fps.py <data_path> --workspace <model_dir> [options]

Example:
  python scripts/benchmark_fps.py data/May --workspace model/trial_may -O --asr_model ave --portrait
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.cuda

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from nerf_triplane.provider import NeRFDataset
from nerf_triplane.utils import (
    PSNRMeter, LPIPSMeter, LMDMeter, Trainer, seed_everything,
    linear_to_srgb, blend_with_mask_cuda,
)
from nerf_triplane.network import NeRFNetwork


def benchmark(opt):
    seed_everything(opt.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = NeRFNetwork(opt)
    criterion = torch.nn.L1Loss(reduction='none')
    metrics = []

    trainer = Trainer(
        'ngp', opt, model, device=device, workspace=opt.workspace,
        criterion=criterion, fp16=opt.fp16, metrics=metrics,
        use_checkpoint=opt.ckpt,
    )

    # Load test data
    test_loader = NeRFDataset(opt, device=device, type='test').dataloader()
    model.aud_features = test_loader._data.auds
    model.eye_areas = test_loader._data.eye_area

    model.eval()
    model.testing = True

    num_frames = len(test_loader)
    warmup_frames = min(5, num_frames)
    print(f"\n{'='*60}")
    print(f"  SyncTalk FPS Benchmark")
    print(f"{'='*60}")
    print(f"  Device:       {device} ({torch.cuda.get_device_name(0)})")
    print(f"  Resolution:   {test_loader._data.H} x {test_loader._data.W}")
    print(f"  Total frames: {num_frames}")
    print(f"  Warmup:       {warmup_frames} frames")
    print(f"  FP16:         {opt.fp16}")
    print(f"{'='*60}\n")

    # --- Warmup ---
    print("[1/3] Warming up GPU...")
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            if i >= warmup_frames:
                break
            with torch.cuda.amp.autocast(enabled=opt.fp16):
                trainer.test_step(data)
    torch.cuda.synchronize()

    # --- Benchmark: pure rendering (no I/O, no postprocess) ---
    print("[2/3] Benchmarking pure rendering FPS...")
    render_times = []
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.cuda.amp.autocast(enabled=opt.fp16):
                preds, preds_depth = trainer.test_step(data)

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            render_times.append(t1 - t0)

    render_times = np.array(render_times)
    gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    # --- Benchmark: full pipeline (rendering + postprocessing) ---
    print("[3/3] Benchmarking full pipeline FPS...")
    pipeline_times = []

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.cuda.amp.autocast(enabled=opt.fp16):
                preds, preds_depth = trainer.test_step(data)

            # Postprocessing (same as test())
            if opt.color_space == 'linear':
                preds = linear_to_srgb(preds)
            if opt.portrait:
                pred = blend_with_mask_cuda(
                    preds[0], data["bg_gt_images"].squeeze(0),
                    data["bg_face_mask"].squeeze(0),
                )
                pred = (pred * 255).astype(np.uint8)
            else:
                pred = preds[0].detach().cpu().numpy()
                pred = (pred * 255).astype(np.uint8)

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            pipeline_times.append(t1 - t0)

    pipeline_times = np.array(pipeline_times)

    # --- Results ---
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Pure Rendering (model forward only):")
    print(f"    Mean:   {render_times.mean()*1000:.2f} ms/frame")
    print(f"    Median: {np.median(render_times)*1000:.2f} ms/frame")
    print(f"    Min:    {render_times.min()*1000:.2f} ms/frame")
    print(f"    Max:    {render_times.max()*1000:.2f} ms/frame")
    print(f"    FPS:    {1.0/render_times.mean():.2f}")
    print()
    print(f"  Full Pipeline (render + postprocess + CPU transfer):")
    print(f"    Mean:   {pipeline_times.mean()*1000:.2f} ms/frame")
    print(f"    Median: {np.median(pipeline_times)*1000:.2f} ms/frame")
    print(f"    FPS:    {1.0/pipeline_times.mean():.2f}")
    print()
    print(f"  GPU Peak Memory: {gpu_mem_mb:.1f} MB")
    print(f"  Real-time capable (>= 25 FPS): {'YES' if 1.0/render_times.mean() >= 25 else 'NO'}")
    print(f"{'='*60}\n")

    # Save results to file
    results_path = os.path.join(opt.workspace, 'benchmark_fps.txt')
    with open(results_path, 'w') as f:
        f.write(f"SyncTalk FPS Benchmark\n")
        f.write(f"Device: {torch.cuda.get_device_name(0)}\n")
        f.write(f"Resolution: {test_loader._data.H}x{test_loader._data.W}\n")
        f.write(f"Frames: {num_frames}\n")
        f.write(f"FP16: {opt.fp16}\n")
        f.write(f"Render FPS: {1.0/render_times.mean():.2f}\n")
        f.write(f"Render ms/frame: {render_times.mean()*1000:.2f}\n")
        f.write(f"Pipeline FPS: {1.0/pipeline_times.mean():.2f}\n")
        f.write(f"Pipeline ms/frame: {pipeline_times.mean()*1000:.2f}\n")
        f.write(f"GPU Peak Memory MB: {gpu_mem_mb:.1f}\n")
    print(f"Results saved to {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true')
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--cuda_ray', action='store_true')
    parser.add_argument('--portrait', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_train', action='store_true')

    # All model args needed for NeRFNetwork init
    parser.add_argument('--asr_model', type=str, default='deepspeech')
    parser.add_argument('--att', type=int, default=2)
    parser.add_argument('--aud', type=str, default='')
    parser.add_argument('--emb', action='store_true')
    parser.add_argument('--ind_dim', type=int, default=4)
    parser.add_argument('--ind_num', type=int, default=20000)
    parser.add_argument('--ind_dim_torso', type=int, default=8)
    parser.add_argument('--bound', type=float, default=1)
    parser.add_argument('--scale', type=float, default=4)
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0])
    parser.add_argument('--exp_eye', action='store_true')
    parser.add_argument('--au45', action='store_true')
    parser.add_argument('--bs_area', type=str, default='upper')
    parser.add_argument('--torso', action='store_true')
    parser.add_argument('--torso_shrink', type=float, default=0.8)
    parser.add_argument('--head_ckpt', type=str, default='')
    parser.add_argument('--smooth_lips', action='store_true')
    parser.add_argument('--smooth_eye', action='store_true')
    parser.add_argument('--smooth_path', action='store_true')
    parser.add_argument('--smooth_path_window', type=int, default=7)
    parser.add_argument('--min_near', type=float, default=0.05)
    parser.add_argument('--density_thresh', type=float, default=10)
    parser.add_argument('--density_thresh_torso', type=float, default=0.01)
    parser.add_argument('--dt_gamma', type=float, default=1/256)
    parser.add_argument('--max_steps', type=int, default=16)
    parser.add_argument('--num_steps', type=int, default=16)
    parser.add_argument('--upsample_steps', type=int, default=0)
    parser.add_argument('--update_extra_interval', type=int, default=16)
    parser.add_argument('--max_ray_batch', type=int, default=4096)
    parser.add_argument('--num_rays', type=int, default=4096 * 16)
    parser.add_argument('--patch_size', type=int, default=1)
    parser.add_argument('--color_space', type=str, default='srgb')
    parser.add_argument('--preload', type=int, default=0)
    parser.add_argument('--bg_img', type=str, default='')
    parser.add_argument('--fbg', action='store_true')
    parser.add_argument('--fix_eye', type=float, default=-1)
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1])
    parser.add_argument('--train_camera', action='store_true')
    parser.add_argument('--unc_loss', type=int, default=1)
    parser.add_argument('--amb_aud_loss', type=int, default=1)
    parser.add_argument('--amb_eye_loss', type=int, default=1)
    parser.add_argument('--finetune_lips', action='store_true')
    parser.add_argument('--init_lips', action='store_true')
    parser.add_argument('--amb_dim', type=int, default=2)
    parser.add_argument('--part', action='store_true')
    parser.add_argument('--part2', action='store_true')

    # GUI (unused but needed for compat)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--W', type=int, default=450)
    parser.add_argument('--H', type=int, default=450)
    parser.add_argument('--radius', type=float, default=3.35)
    parser.add_argument('--fovy', type=float, default=21.24)
    parser.add_argument('--max_spp', type=int, default=1)

    # ASR
    parser.add_argument('--asr', action='store_true')
    parser.add_argument('--asr_wav', type=str, default='')
    parser.add_argument('--asr_play', action='store_true')
    parser.add_argument('--asr_save_feats', action='store_true')
    parser.add_argument('--fps', type=int, default=50)
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=50)
    parser.add_argument('-r', type=int, default=10)

    # Option B specific (harmless defaults)
    parser.add_argument('--temporal_loss', type=int, default=0)
    parser.add_argument('--lambda_temporal', type=float, default=0.1)
    parser.add_argument('--lambda_sync', type=float, default=0.0)
    parser.add_argument('--lr_schedule', type=str, default='cosine')
    parser.add_argument('--lr_warmup_steps', type=int, default=2000)
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--early_stop_min_delta', type=float, default=1e-4)
    parser.add_argument('--warmup_step', type=int, default=10000)
    parser.add_argument('--lambda_amb', type=float, default=1e-4)
    parser.add_argument('--pyramid_loss', type=int, default=0)
    parser.add_argument('--iters', type=int, default=200000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--lr_net', type=float, default=1e-3)

    opt = parser.parse_args()

    if opt.O:
        opt.fp16 = True
        opt.exp_eye = True

    opt.cuda_ray = True
    opt.test = True

    benchmark(opt)
