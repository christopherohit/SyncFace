"""
Phase 1: Pre-training script for Universal Motion Field (UMF)

This script trains the UMF on multiple identities simultaneously:
- Each identity has its own PersonalizedField and StaticField
- UMF is shared across all identities
- NCLoss encourages orthogonality between personalized fields
- After training, only UMF is saved for adaptation

Usage:
    conda activate synctalk && python pretrain_umf.py \
        --data_root /path/to/multi_person_data \
        --workspace workspace_pretrain \
        --iters 200000 \
        --lambda_C 0.01
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from typing import Dict, List, Tuple

from nerf_triplane.instag_network import InsTaGNetwork
from nerf_triplane.instag_losses import NegativeContrastLoss, GeometryPriorRegularizer
from nerf_triplane.utils import *
from nerf_triplane.provider import NeRFDataset


class MultiPersonDataset(Dataset):
    """
    Dataset for multi-person pre-training.
    Wraps multiple single-person NeRFDatasets.
    """
    
    def __init__(self, opt, device, type='train'):
        super().__init__()
        self.opt = opt
        self.device = device
        self.type = type
        
        # Load all person directories
        data_root = opt.data_root
        person_dirs = [d for d in os.listdir(data_root) 
                      if os.path.isdir(os.path.join(data_root, d))]
        
        print(f"[INFO] Found {len(person_dirs)} identities for pre-training")
        
        # Create individual datasets
        self.datasets = {}
        self.identity_ids = []
        
        for person_dir in person_dirs:
            person_path = os.path.join(data_root, person_dir)
            
            # Check if valid dataset (has required files)
            if not os.path.exists(os.path.join(person_path, 'transforms_train.json')):
                print(f"[WARN] Skipping {person_dir}: missing transforms_train.json")
                continue
            
            # Create a copy of opt with this person's path
            person_opt = argparse.Namespace(**vars(opt))
            person_opt.path = person_path
            
            try:
                dataset = NeRFDataset(person_opt, device, type=type)
                self.datasets[person_dir] = dataset
                self.identity_ids.append(person_dir)
                print(f"[INFO] Loaded identity {person_dir}: {len(dataset)} frames")
            except Exception as e:
                print(f"[WARN] Failed to load {person_dir}: {e}")
        
        if len(self.identity_ids) == 0:
            raise ValueError("No valid identities found for pre-training!")
        
        # Compute total samples
        self.total_samples = sum(len(ds) for ds in self.datasets.values())
        
        # Create index mapping: global_idx -> (identity_id, local_idx)
        self.index_mapping = []
        for identity_id in self.identity_ids:
            dataset = self.datasets[identity_id]
            for local_idx in range(len(dataset)):
                self.index_mapping.append((identity_id, local_idx))
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        identity_id, local_idx = self.index_mapping[idx]
        data = self.datasets[identity_id].collate([local_idx])
        data['identity_id'] = identity_id
        return data
    
    def dataloader(self):
        # Use simple sequential sampling for multi-person training
        loader = DataLoader(
            self,
            batch_size=1,  # Each "batch" is one frame from one person
            shuffle=True,
            num_workers=0
        )
        loader._data = self  # For compatibility
        return loader


def train_step(model: InsTaGNetwork, 
               data_batch: Dict,
               criterion,
               nc_loss: NegativeContrastLoss,
               opt,
               scaler=None) -> Dict[str, float]:
    """
    Single training step for pre-training.
    
    Args:
        model: InsTaGNetwork in 'pretrain' mode
        data_batch: Batch data containing samples from multiple identities
        criterion: Photometric loss (L1 or MSE)
        nc_loss: Negative contrast loss module
        opt: Options
        scaler: GradScaler for mixed precision
    
    Returns:
        Dictionary of loss values
    """
    identity_id = data_batch['identity_id']
    
    # Ensure this identity exists in model
    if identity_id not in model.personalized_fields:
        model.add_identity(identity_id)
    
    # Get rays and target images
    rays_o = data_batch['rays_o']  # [B, N, 3]
    rays_d = data_batch['rays_d']  # [B, N, 3]
    images = data_batch['images']  # [B, N, 3]
    
    # Audio features
    auds = data_batch.get('auds', None)
    
    # Eye features (if available)
    eye = data_batch.get('eye', None)
    
    # Background coordinates for torso (if used)
    bg_coords = data_batch.get('bg_coords', None)
    poses = data_batch.get('poses', None)
    
    # Index for individual codes
    index = data_batch.get('index', 0)
    
    # Forward pass through renderer
    with torch.cuda.amp.autocast(enabled=opt.fp16):
        results = model.render(
            rays_o, rays_d, auds, bg_coords, poses,
            index=index, eye=eye,
            identity_id=identity_id
        )
        
        pred_rgb = results['image']
        
        # Photometric loss
        loss_rgb = criterion(pred_rgb, images).mean()
        
        # Additional losses (ambient, uncertainty, etc.)
        loss_total = loss_rgb
        
        if 'ambient_aud' in results and opt.amb_aud_loss:
            loss_amb_aud = results['ambient_aud'].mean()
            loss_total = loss_total + opt.lambda_amb * loss_amb_aud
        
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
    
    return loss_total, losses


def train_epoch(model: InsTaGNetwork,
                train_loader,
                optimizer,
                criterion,
                nc_loss: NegativeContrastLoss,
                opt,
                scaler=None,
                scheduler=None) -> Dict[str, float]:
    """
    Train for one epoch with multi-person data.
    """
    model.train()
    
    total_losses = {}
    num_batches = 0
    
    # Accumulate personalized field outputs for NCLoss
    # We compute NCLoss every N steps to reduce memory
    nc_loss_interval = 10
    delta_personal_accumulator = {}
    
    for i, data in enumerate(train_loader):
        identity_id = data['identity_id']
        
        # Forward and photometric loss
        loss, losses = train_step(model, data, criterion, nc_loss, opt, scaler)
        
        # Accumulate losses for logging
        for k, v in losses.items():
            if k not in total_losses:
                total_losses[k] = 0
            total_losses[k] += v
        
        # Store delta_personal for NCLoss (sample a subset of rays to save memory)
        if hasattr(model, 'personalized_fields') and identity_id in model.personalized_fields:
            # Get deformation from last forward pass
            # For simplicity, we'll compute it here on a subset
            with torch.no_grad():
                # Sample 128 random points in the volume
                x_sample = torch.rand(128, 3, device=model.device) * 2 - 1  # [-1, 1]
                enc_a = model.encode_audio(data.get('auds', None))
                eye = data.get('eye', None)
                
                # Compute only personalized field
                if enc_a is not None:
                    f_l = enc_a.repeat(x_sample.shape[0], 1)
                else:
                    f_l = torch.zeros(x_sample.shape[0], model.audio_dim, device=model.device)
                
                f_e = eye.repeat(x_sample.shape[0], 1) if eye is not None else None
                
                delta_personal = model.personalized_fields[identity_id](
                    x_sample, f_l, f_e, bound=model.bound
                )
                
                delta_personal_accumulator[identity_id] = delta_personal
        
        # Compute NCLoss periodically
        if (i + 1) % nc_loss_interval == 0 and len(delta_personal_accumulator) >= 2:
            loss_nc = nc_loss(delta_personal_accumulator)
            loss = loss + loss_nc
            total_losses['loss_nc'] = total_losses.get('loss_nc', 0) + loss_nc.item()
            delta_personal_accumulator.clear()
        
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
        if (i + 1) % 100 == 0:
            avg_loss = total_losses['loss'] / num_batches
            print(f"  Step {i+1}/{len(train_loader)}: loss={avg_loss:.4f}")
    
    # Average losses
    for k in total_losses:
        total_losses[k] /= num_batches
    
    return total_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, 
                       help="Root directory containing multiple person folders")
    parser.add_argument('--workspace', type=str, default='workspace_pretrain')
    parser.add_argument('--iters', type=int, default=200000)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--lr_net', type=float, default=1e-3)
    parser.add_argument('--lambda_C', type=float, default=0.01, 
                       help="NCLoss weight")
    
    # Network options
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
    
    opt = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(opt.workspace, exist_ok=True)
    
    print("="*60)
    print("Phase 1: Pre-training Universal Motion Field")
    print("="*60)
    
    # Create model
    print("[INFO] Creating InsTaGNetwork in pre-training mode...")
    model = InsTaGNetwork(opt, phase='pretrain').to(device)
    
    # Loss functions
    criterion = nn.L1Loss(reduction='none')
    nc_loss = NegativeContrastLoss(lambda_c=opt.lambda_C)
    
    # Dataset
    print("[INFO] Loading multi-person dataset...")
    train_dataset = MultiPersonDataset(opt, device, type='train')
    train_loader = train_dataset.dataloader()
    
    print(f"[INFO] Total training samples: {len(train_dataset)}")
    print(f"[INFO] Identities: {train_dataset.identity_ids}")
    
    # Initialize identity models
    print("[INFO] Initializing identity-specific modules...")
    for identity_id in train_dataset.identity_ids:
        model.add_identity(identity_id)
    print(f"[INFO] Initialized {len(model.personalized_fields)} PersonalizedFields")
    print(f"[INFO] Initialized {len(model.static_fields)} StaticFields")
    
    # Optimizer
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
    print("\n[INFO] Starting pre-training...")
    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")
        
        losses = train_epoch(
            model, train_loader, optimizer, criterion, nc_loss,
            opt, scaler, scheduler
        )
        
        print(f"  Average losses: {losses}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(opt.workspace, f'umf_epoch{epoch+1}.pth')
            model.save_umf_checkpoint(ckpt_path)
    
    # Save final UMF checkpoint
    final_path = os.path.join(opt.workspace, 'umf_final.pth')
    model.save_umf_checkpoint(final_path)
    
    print("\n" + "="*60)
    print(f"Pre-training complete! UMF saved to {final_path}")
    print("Use this checkpoint for adaptation phase.")
    print("="*60)


if __name__ == '__main__':
    # Close tf32 for numerical accuracy
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except AttributeError:
        pass
    
    main()


