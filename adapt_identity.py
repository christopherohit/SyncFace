"""
Phase 2: Adaptation script for new identity using frozen UMF

This script performs fast few-shot adaptation:
- Load frozen UMF from pre-training
- Train new StaticField, PersonalizedField, and MotionAligner
- Apply Geometry Prior regularizer using monocular depth/normal estimates
- Few-shot training (10-100 frames typical)

Usage:
    conda activate synctalk && python adapt_identity.py \
        --umf_checkpoint workspace_pretrain/umf_final.pth \
        --data /path/to/target_person \
        --workspace workspace_adapt \
        --iters 20000 \
        --lambda_D 0.1 \
        --lambda_N 0.05
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional
import cv2

from nerf_triplane.instag_network import InsTaGNetwork
from nerf_triplane.instag_losses import GeometryPriorRegularizer
from nerf_triplane.utils import *
from nerf_triplane.provider import NeRFDataset


def load_geometry_estimates(data_path: str, device) -> Dict[str, torch.Tensor]:
    """
    Load pre-computed monocular geometry estimates (depth and normals).
    
    Expected format:
        - depth_estimates/frame_XXXXX.npy: [H, W] depth maps
        - normal_estimates/frame_XXXXX.npy: [H, W, 3] normal maps
    
    Returns:
        Dictionary mapping frame_id -> {'depth': tensor, 'normal': tensor}
    """
    geometry_data = {}
    
    depth_dir = os.path.join(data_path, 'depth_estimates')
    normal_dir = os.path.join(data_path, 'normal_estimates')
    
    if not os.path.exists(depth_dir):
        print("[WARN] Depth estimates not found. Geometry prior will be disabled.")
        print(f"       Expected directory: {depth_dir}")
        return None
    
    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith('.npy')])
    
    print(f"[INFO] Loading {len(depth_files)} geometry estimates...")
    
    for depth_file in depth_files:
        frame_id = depth_file.replace('.npy', '')
        
        # Load depth
        depth_path = os.path.join(depth_dir, depth_file)
        depth = np.load(depth_path)
        depth_tensor = torch.from_numpy(depth).float().to(device)
        
        # Load normal (if available)
        normal_path = os.path.join(normal_dir, depth_file)
        normal_tensor = None
        if os.path.exists(normal_path):
            normal = np.load(normal_path)
            normal_tensor = torch.from_numpy(normal).float().to(device)
        
        geometry_data[frame_id] = {
            'depth': depth_tensor,
            'normal': normal_tensor
        }
    
    print(f"[INFO] Loaded geometry estimates for {len(geometry_data)} frames")
    return geometry_data


def extract_geometry_from_rendering(depth_map: torch.Tensor, 
                                   rays_o: torch.Tensor,
                                   rays_d: torch.Tensor,
                                   H: int, W: int) -> torch.Tensor:
    """
    Convert NeRF depth map to absolute depth values.
    
    Args:
        depth_map: [N,] normalized depth values from NeRF
        rays_o: [N, 3] ray origins
        rays_d: [N, 3] ray directions
        H, W: Image dimensions
    
    Returns:
        depth_abs: [H, W] absolute depth map
    """
    # Depth is along ray direction
    depth_abs = depth_map.reshape(H, W)
    return depth_abs


def train_step_adapt(model: InsTaGNetwork,
                    data: Dict,
                    criterion,
                    geo_regularizer: Optional[GeometryPriorRegularizer],
                    geometry_data: Optional[Dict],
                    opt,
                    scaler=None) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Single training step for adaptation phase.
    
    Includes photometric loss and geometry prior regularizer.
    """
    # Get data
    rays_o = data['rays_o']  # [B, N, 3]
    rays_d = data['rays_d']  # [B, N, 3]
    images = data['images']  # [B, N, 3]
    auds = data.get('auds', None)
    eye = data.get('eye', None)
    bg_coords = data.get('bg_coords', None)
    poses = data.get('poses', None)
    index = data.get('index', 0)
    H = data.get('H', 450)
    W = data.get('W', 450)
    
    # Frame ID for geometry lookup
    frame_id = data.get('frame_id', None)
    
    with torch.cuda.amp.autocast(enabled=opt.fp16):
        # Render
        results = model.render(
            rays_o, rays_d, auds, bg_coords, poses,
            index=index, eye=eye
        )
        
        pred_rgb = results['image']
        pred_depth = results.get('depth', None)
        
        # Photometric loss
        loss_rgb = criterion(pred_rgb, images).mean()
        loss_total = loss_rgb
        
        # Geometry prior loss
        if geo_regularizer is not None and geometry_data is not None and frame_id is not None:
            if frame_id in geometry_data and pred_depth is not None:
                geo_est = geometry_data[frame_id]
                
                # Extract predicted depth
                depth_pred = pred_depth.flatten()  # [N,]
                
                # Get estimated depth (resize if needed)
                depth_est = geo_est['depth']  # [H_est, W_est]
                if depth_est.shape != (H, W):
                    depth_est = torch.nn.functional.interpolate(
                        depth_est.unsqueeze(0).unsqueeze(0),
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze()
                
                depth_est = depth_est.flatten()  # [N,]
                
                # Create mask (valid depth values)
                mask = (depth_est > 0) & (depth_pred > 0)
                
                # Compute geometry loss
                geo_losses = geo_regularizer(
                    d_pred=depth_pred,
                    d_est=depth_est,
                    n_pred=None,  # Normal computation from NeRF is complex, skip for now
                    n_est=None,
                    mask=mask
                )
                
                loss_geo = geo_losses['total']
                loss_total = loss_total + loss_geo
        
        # Additional losses
        if 'ambient_aud' in results and opt.amb_aud_loss:
            loss_amb = results['ambient_aud'].mean()
            loss_total = loss_total + opt.lambda_amb * loss_amb
        
        if 'ambient_eye' in results and opt.amb_eye_loss:
            loss_amb_eye = results['ambient_eye'].mean()
            loss_total = loss_total + opt.lambda_amb * loss_amb_eye
        
        if 'uncertainty' in results and opt.unc_loss:
            loss_unc = results['uncertainty'].mean()
            loss_total = loss_total + 0.01 * loss_unc
    
    losses = {
        'loss': loss_total.item(),
        'loss_rgb': loss_rgb.item(),
    }
    
    if geo_regularizer is not None and 'loss_geo' in locals():
        losses['loss_geo'] = loss_geo.item()
        if 'geo_losses' in locals():
            losses['loss_depth'] = geo_losses['depth'].item()
            losses['loss_normal'] = geo_losses['normal'].item()
    
    return loss_total, losses


def train_epoch_adapt(model: InsTaGNetwork,
                     train_loader,
                     optimizer,
                     criterion,
                     geo_regularizer: Optional[GeometryPriorRegularizer],
                     geometry_data: Optional[Dict],
                     opt,
                     scaler=None,
                     scheduler=None) -> Dict[str, float]:
    """Train one epoch during adaptation."""
    model.train()
    
    total_losses = {}
    num_batches = 0
    
    for i, data in enumerate(train_loader):
        loss, losses = train_step_adapt(
            model, data, criterion, geo_regularizer, geometry_data, opt, scaler
        )
        
        # Accumulate losses
        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0
            total_losses[k] += v
        
        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        optimizer.zero_grad()
        
        if scheduler is not None:
            scheduler.step()
        
        num_batches += 1
        
        # Print progress
        if (i + 1) % 50 == 0:
            avg_loss = total_losses['loss'] / num_batches
            print(f"  Step {i+1}/{len(train_loader)}: loss={avg_loss:.4f}")
    
    # Average losses
    for k in total_losses:
        total_losses[k] /= num_batches
    
    return total_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--umf_checkpoint', type=str, required=True,
                       help="Path to pre-trained UMF checkpoint")
    parser.add_argument('--data', type=str, required=True,
                       help="Path to target identity data")
    parser.add_argument('--workspace', type=str, default='workspace_adapt')
    parser.add_argument('--iters', type=int, default=20000,
                       help="Adaptation iterations (fewer than pre-training)")
    parser.add_argument('--lr', type=float, default=2e-2,
                       help="Learning rate for hash grids")
    parser.add_argument('--lr_net', type=float, default=5e-3,
                       help="Learning rate for MLPs")
    
    # Geometry prior weights
    parser.add_argument('--lambda_D', type=float, default=0.1,
                       help="Depth loss weight")
    parser.add_argument('--lambda_N', type=float, default=0.05,
                       help="Normal loss weight")
    parser.add_argument('--alpha_depth', type=float, default=0.5,
                       help="Scale-invariance parameter for depth loss")
    
    # Network options (should match pre-training)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bound', type=float, default=1.0)
    parser.add_argument('--audio_dim', type=int, default=32)
    parser.add_argument('--asr_model', type=str, default='deepspeech')
    parser.add_argument('--att', type=int, default=2)
    parser.add_argument('--emb', action='store_true')
    parser.add_argument('--exp_eye', action='store_true')
    parser.add_argument('--au45', action='store_true')
    parser.add_argument('--bs_area', type=str, default='upper')
    
    # Training options
    parser.add_argument('--num_rays', type=int, default=4096 * 16)
    parser.add_argument('--cuda_ray', action='store_true', default=True)
    parser.add_argument('--max_steps', type=int, default=16)
    parser.add_argument('--update_extra_interval', type=int, default=16)
    
    # Loss options
    parser.add_argument('--lambda_amb', type=float, default=1e-4)
    parser.add_argument('--amb_aud_loss', type=int, default=1)
    parser.add_argument('--amb_eye_loss', type=int, default=1)
    parser.add_argument('--unc_loss', type=int, default=1)
    
    # Dataset options
    parser.add_argument('--preload', type=int, default=0)
    parser.add_argument('--scale', type=float, default=4)
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0])
    parser.add_argument('--dt_gamma', type=float, default=1/256)
    parser.add_argument('--min_near', type=float, default=0.05)
    parser.add_argument('--density_thresh', type=float, default=10)
    parser.add_argument('--density_thresh_torso', type=float, default=0.01)
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1])
    
    # Individual codes
    parser.add_argument('--ind_dim', type=int, default=4)
    parser.add_argument('--ind_num', type=int, default=20000)
    parser.add_argument('--ind_dim_torso', type=int, default=8)
    
    # Other
    parser.add_argument('--torso', action='store_true', default=False)
    parser.add_argument('--color_space', type=str, default='srgb')
    parser.add_argument('--test_train', action='store_true', default=False)
    parser.add_argument('--smooth_lips', action='store_true', default=False)
    parser.add_argument('--train_camera', action='store_true', default=False)
    
    # Geometry estimation options
    parser.add_argument('--use_geometry_prior', action='store_true', default=True,
                       help="Use geometry prior regularizer")
    
    opt = parser.parse_args()
    opt.path = opt.data  # For compatibility with NeRFDataset
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(opt.workspace, exist_ok=True)
    
    print("="*60)
    print("Phase 2: Few-Shot Adaptation with Frozen UMF")
    print("="*60)
    
    # Create model in adaptation mode
    print("[INFO] Creating InsTaGNetwork in adaptation mode...")
    model = InsTaGNetwork(opt, phase='adapt').to(device)
    
    # Load UMF checkpoint
    print(f"[INFO] Loading UMF from {opt.umf_checkpoint}...")
    model.load_umf_checkpoint(opt.umf_checkpoint)
    print("[INFO] UMF loaded and frozen")
    
    # Loss functions
    criterion = nn.L1Loss(reduction='none')
    
    # Geometry prior (optional)
    geo_regularizer = None
    geometry_data = None
    if opt.use_geometry_prior:
        print("[INFO] Setting up geometry prior regularizer...")
        geo_regularizer = GeometryPriorRegularizer(
            lambda_d=opt.lambda_D,
            lambda_n=opt.lambda_N,
            alpha_depth=opt.alpha_depth
        ).to(device)
        
        # Load pre-computed geometry estimates
        geometry_data = load_geometry_estimates(opt.data, device)
        
        if geometry_data is None:
            print("[WARN] Geometry prior disabled (no estimates found)")
            geo_regularizer = None
    
    # Dataset
    print(f"[INFO] Loading target identity data from {opt.data}...")
    train_dataset = NeRFDataset(opt, device, type='train')
    train_loader = train_dataset.dataloader()
    
    print(f"[INFO] Training samples: {len(train_dataset)}")
    print(f"[INFO] Few-shot adaptation with {len(train_dataset)} frames")
    
    # Set audio features for the model (needed for density grid updates)
    model.aud_features = train_loader._data.auds
    model.eye_area = train_loader._data.eye_area
    model.poses = train_loader._data.poses
    
    # Optimizer (only trainable parameters)
    print("[INFO] Setting up optimizer...")
    print("  Trainable modules:")
    print("    - StaticField (hash grids + MLPs)")
    print("    - PersonalizedField")
    print("    - MotionAligner")
    print("  Frozen modules:")
    print("    - UMF")
    print("    - AudioNet (optional: can fine-tune)")
    
    optimizer = optim.AdamW(
        model.get_params(opt.lr, opt.lr_net),
        betas=(0, 0.99),
        eps=1e-8
    )
    
    # Learning rate scheduler
    max_epochs = int(np.ceil(opt.iters / len(train_loader)))
    print(f"[INFO] Training for {max_epochs} epochs (~{opt.iters} iterations)")
    
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda iter: 0.5 ** (iter / opt.iters)
    )
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=opt.fp16)
    
    # Training loop
    print("\n[INFO] Starting adaptation training...")
    best_loss = float('inf')
    
    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")
        
        losses = train_epoch_adapt(
            model, train_loader, optimizer, criterion,
            geo_regularizer, geometry_data, opt, scaler, scheduler
        )
        
        print(f"  Average losses: {losses}")
        
        # Save best checkpoint
        if losses['loss'] < best_loss:
            best_loss = losses['loss']
            ckpt_path = os.path.join(opt.workspace, 'best_adapted.pth')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'losses': losses,
            }, ckpt_path)
            print(f"  Saved best checkpoint: loss={best_loss:.4f}")
        
        # Save periodic checkpoints
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(opt.workspace, f'adapted_epoch{epoch+1}.pth')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'losses': losses,
            }, ckpt_path)
    
    # Save final checkpoint
    final_path = os.path.join(opt.workspace, 'adapted_final.pth')
    torch.save({
        'model': model.state_dict(),
        'losses': losses,
    }, final_path)
    
    print("\n" + "="*60)
    print(f"Adaptation complete! Model saved to {final_path}")
    print("="*60)
    
    # Print summary
    print("\nAdaptation Summary:")
    print(f"  Target identity: {opt.data}")
    print(f"  Training frames: {len(train_dataset)}")
    print(f"  Final loss: {losses['loss']:.4f}")
    if 'loss_geo' in losses:
        print(f"  Geometry loss: {losses['loss_geo']:.4f}")
    print(f"  Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    # Close tf32 for numerical accuracy
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except AttributeError:
        pass
    
    main()


