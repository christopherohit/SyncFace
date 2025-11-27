#!/usr/bin/env python3
"""
SyncFace Training Script
=========================
Unified training for hybrid 3DGS-NeRF talking head synthesis.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# Setup paths - DO NOT add nerf_triplane yet (has utils.py that conflicts)
_root = Path(__file__).parent.resolve()
_tg_path = _root / 'TalkingGaussian'
_nerf_path = _root / 'nerf_triplane'

# Add ONLY TalkingGaussian first
sys.path.insert(0, str(_tg_path))

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Import TalkingGaussian modules BEFORE adding nerf_triplane
from scene.gaussian_model import GaussianModel
from scene.motion_net_sync import BlendshapeMotionNetwork

# NOW add nerf_triplane (after TalkingGaussian's utils is already cached)
sys.path.insert(0, str(_nerf_path))

# Import nerf_triplane modules
try:
    from network import NeRFNetwork
    from renderer import NeRFRenderer
    NERF_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] NeRF modules not available: {e}")
    NeRFNetwork = None
    NeRFRenderer = None
    NERF_AVAILABLE = False

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False


class SyncFaceTrainer:
    """Unified trainer for SyncFace model."""
    
    DEFAULT_CONFIG = {
        'face_iterations': 50000,
        'face_lr_gaussian': 5e-4,
        'face_lr_motion': 1e-4,
        'mouth_iterations': 100000,
        'mouth_lr_hash': 1e-3,
        'mouth_lr_net': 1e-4,
        'mouth_rays_per_batch': 8192,
        'fusion_iterations': 20000,
        'audio_extractor': 'hubert',
        'audio_dim': 64,
        'blendshape_dim': 52,
        'sh_degree': 3,
        'save_interval': 5000,
        'log_interval': 100,
    }
    
    def __init__(self, data_dir: str, output_dir: str, config: dict = None, device: str = 'cuda'):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        
        self.log_file = self.output_dir / 'training.log'
        self.writer = SummaryWriter(str(self.output_dir / 'tensorboard')) if TENSORBOARD_AVAILABLE else None
        self.lpips_fn = lpips.LPIPS(net='alex').eval().to(device) if LPIPS_AVAILABLE else None
        
        self.log("SyncFace Trainer initialized")
        self.log(f"Data: {self.data_dir}")
        self.log(f"Output: {self.output_dir}")
    
    def log(self, message: str):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + '\n')
    
    def load_data(self):
        self.log("Loading data...")
        
        train_path = self.data_dir / 'transforms_train.json'
        if train_path.exists():
            with open(train_path) as f:
                self.train_transforms = json.load(f)
        else:
            self.log(f"[ERROR] transforms_train.json not found")
            return False
        
        val_path = self.data_dir / 'transforms_val.json'
        self.val_transforms = None
        if val_path.exists():
            with open(val_path) as f:
                self.val_transforms = json.load(f)
        
        # Load audio
        self.audio_feats = None
        for audio_name in ['audio_feats.npy', 'aud_hu.npy', 'aud.npy']:
            audio_path = self.data_dir / audio_name
            if audio_path.exists():
                self.audio_feats = torch.from_numpy(np.load(audio_path)).float()
                self.log(f"Loaded audio: {audio_name}, shape={self.audio_feats.shape}")
                break
        
        if self.audio_feats is None:
            self.log("[WARNING] No audio features found")
        
        # Load blendshapes
        self.blendshapes = None
        bs_path = self.data_dir / 'blendshapes.npy'
        if bs_path.exists():
            self.blendshapes = torch.from_numpy(np.load(bs_path)).float()
            self.log(f"Loaded blendshapes: {self.blendshapes.shape}")
        
        self.log(f"Loaded {len(self.train_transforms['frames'])} training frames")
        return True
    
    def build_model(self):
        self.log("Building model...")
        
        self.gaussians = GaussianModel(self.config['sh_degree'])
        
        self.motion_net = BlendshapeMotionNetwork(
            audio_extractor=self.config['audio_extractor'],
            blendshape_prior_scale=0.1,
        ).to(self.device)
        
        # Initialize Gaussians
        xyz = torch.randn(5000, 3, device=self.device) * 0.1
        self.gaussians._xyz = torch.nn.Parameter(xyz)
        self.gaussians._features_dc = torch.nn.Parameter(torch.randn(5000, 1, 3, device=self.device) * 0.1)
        self.gaussians._features_rest = torch.nn.Parameter(torch.zeros(5000, 15, 3, device=self.device))
        self.gaussians._scaling = torch.nn.Parameter(torch.ones(5000, 3, device=self.device) * -3)
        self.gaussians._rotation = torch.nn.Parameter(torch.zeros(5000, 4, device=self.device))
        self.gaussians._rotation.data[:, 0] = 1
        self.gaussians._opacity = torch.nn.Parameter(torch.ones(5000, 1, device=self.device) * -2)
        
        self.log("Model built")
    
    def train_face(self, start_iter: int = 0):
        self.log("=" * 60)
        self.log("Stage 1: Face Training")
        self.log("=" * 60)
        
        iterations = self.config['face_iterations']
        
        motion_params = self.motion_net.get_params(
            self.config['face_lr_gaussian'],
            self.config['face_lr_motion']
        )
        optimizer = torch.optim.AdamW(motion_params, betas=(0.9, 0.99), eps=1e-8)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
        
        pbar = tqdm(range(start_iter + 1, iterations + 1), desc="Stage 1: Face")
        ema_loss = 0.0
        
        for iteration in pbar:
            frame_idx = np.random.randint(len(self.train_transforms['frames']))
            frame = self.train_transforms['frames'][frame_idx]
            img_id = frame['img_id']
            
            audio = self._get_audio(img_id)
            blendshape = self._get_blendshape(img_id)
            
            xyz = self.gaussians.get_xyz
            motion = self.motion_net(xyz, audio, blendshape)
            
            # Motion regularization losses
            loss_motion = motion['d_xyz'].abs().mean()
            if 'd_rot' in motion:
                loss_motion = loss_motion + 0.1 * motion['d_rot'].abs().mean()
            if 'd_scale' in motion:
                loss_motion = loss_motion + 0.1 * motion['d_scale'].abs().mean()
            
            # Smoothness loss - encourage similar deformations for nearby points
            if xyz.shape[0] > 1:
                perm = torch.randperm(xyz.shape[0])[:min(1000, xyz.shape[0])]
                d_xyz_sample = motion['d_xyz'][perm]
                loss_smooth = (d_xyz_sample[1:] - d_xyz_sample[:-1]).pow(2).mean()
            else:
                loss_smooth = torch.tensor(0.0, device=self.device)
            
            # Total loss
            loss = 0.1 * loss_motion + 0.01 * loss_smooth
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            ema_loss = 0.9 * ema_loss + 0.1 * loss.item()
            
            if iteration % self.config['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({'loss': f'{ema_loss:.6f}', 'lr': f'{lr:.2e}'})
        
        # Save only final checkpoint
        self._save_checkpoint('face', iterations)
        self.log("Stage 1 complete")
    
    def train_mouth(self, start_iter: int = 0):
        self.log("=" * 60)
        self.log("Stage 2: Mouth Training")
        self.log("=" * 60)
        
        if not NERF_AVAILABLE:
            self.log("[ERROR] NeRF modules not available")
            return
        
        self.log("Stage 2 placeholder - implement full NeRF training")
        self._save_checkpoint('mouth', self.config['mouth_iterations'])
    
    def train_fusion(self, start_iter: int = 0):
        self.log("=" * 60)
        self.log("Stage 3: Fusion")
        self.log("=" * 60)
        self.log("Stage 3 placeholder")
        self._save_checkpoint('fusion', self.config['fusion_iterations'])
    
    def _get_audio(self, img_id: int, win_size: int = 16, att_seq_len: int = 8):
        """Get audio features for attention network.
        
        Returns audio windows for att_seq_len consecutive frames,
        each window being [1024, win_size].
        Output: [att_seq_len, 1024, win_size]
        """
        if self.audio_feats is None:
            return None
        
        T = len(self.audio_feats)
        half_w = win_size // 2
        half_seq = att_seq_len // 2
        
        # Get att_seq_len audio windows centered around img_id
        audio_batch = []
        for seq_idx in range(img_id - half_seq, img_id + half_seq):
            center = max(0, min(seq_idx, T - 1))
            
            # Get window of audio frames for this sequence position
            audio_window = []
            for i in range(center - half_w, center + half_w):
                idx = max(0, min(i, T - 1))
                # audio_feats[idx] is [2, 1024], take first channel
                audio_window.append(self.audio_feats[idx, 0, :])  # [1024]
            
            # Stack to [win_size, 1024] then transpose to [1024, win_size]
            audio = torch.stack(audio_window, dim=0).T  # [1024, win_size]
            audio_batch.append(audio)
        
        # Stack: [att_seq_len, 1024, win_size]
        return torch.stack(audio_batch, dim=0).to(self.device)
    
    def _get_blendshape(self, img_id: int):
        if self.blendshapes is None:
            return torch.zeros(1, 52, device=self.device)
        if img_id < len(self.blendshapes):
            # Add batch dimension: [52] -> [1, 52]
            return self.blendshapes[img_id].unsqueeze(0).to(self.device)
        return torch.zeros(1, 52, device=self.device)
    
    def _save_checkpoint(self, stage: str, iteration: int):
        ckpt_dir = self.output_dir / 'checkpoints'
        ckpt_dir.mkdir(exist_ok=True)
        
        state = {
            'stage': stage,
            'iteration': iteration,
            'motion_net': self.motion_net.state_dict(),
            'gaussians': {
                'xyz': self.gaussians._xyz.data,
                'features_dc': self.gaussians._features_dc.data,
                'features_rest': self.gaussians._features_rest.data,
                'scaling': self.gaussians._scaling.data,
                'rotation': self.gaussians._rotation.data,
                'opacity': self.gaussians._opacity.data,
            },
            'config': self.config,
        }
        
        torch.save(state, ckpt_dir / f'{stage}_{iteration}.pth')
        torch.save(state, ckpt_dir / f'{stage}_latest.pth')
        self.log(f"Saved: {stage}_{iteration}.pth")


def main():
    parser = argparse.ArgumentParser(description="SyncFace Training")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./output')
    parser.add_argument('--stage', type=str, default='all', choices=['all', 'face', 'mouth', 'fusion'])
    parser.add_argument('--face_iterations', type=int, default=50000)
    parser.add_argument('--mouth_iterations', type=int, default=100000)
    parser.add_argument('--audio', type=str, default='hubert')
    
    args = parser.parse_args()
    
    config = {
        'face_iterations': args.face_iterations,
        'mouth_iterations': args.mouth_iterations,
        'audio_extractor': args.audio,
    }
    
    trainer = SyncFaceTrainer(args.data_dir, args.output_dir, config)
    
    if not trainer.load_data():
        return
    
    trainer.build_model()
    
    if args.stage in ['all', 'face']:
        trainer.train_face()
    if args.stage in ['all', 'mouth']:
        trainer.train_mouth()
    if args.stage in ['all', 'fusion']:
        trainer.train_fusion()


if __name__ == '__main__':
    main()
