"""
3D Gaussian Splatting Trainer for SyncTalk
Replaces NeRF ray-based training with 3DGS full image rasterization
"""

import os
import time
import glob
import numpy as np
import tqdm
import tensorboardX
import cv2
from typing import Optional, Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader

# Import 3DGS components
from gaussian_renderer import render, GaussianRasterizationSettings
from utils_gs.loss_utils import l1_loss, ssim
from utils_gs.image_utils import psnr
from scene_gs.cameras import Camera as GaussianCamera

# Import base trainer utilities
from .utils import Trainer as BaseTrainer
from .provider_gs import Camera, create_gaussian_data_loader


class GaussianTrainer:
    """
    Trainer for 3D Gaussian Splatting based SyncTalk model
    """
    
    def __init__(
        self,
        name,  # experiment name
        opt,   # configuration options
        model, # DynamicGaussianNetwork
        criterion=None,
        optimizer=None,
        ema_decay=None,
        ema_update_interval=1000,
        lr_scheduler=None,
        metrics=[],
        local_rank=0,
        world_size=1,
        device=None,
        mute=False,
        fp16=False,
        eval_interval=1,
        max_keep_ckpt=50,
        workspace='workspace',
        best_mode='min',
        use_loss_as_metric=True,
        report_metric_at_train=False,
        use_checkpoint="latest",
        use_tensorboardX=True,
        scheduler_update_every_step=False,
    ):
        self.name = name
        self.opt = opt
        self.mute = mute
        self.metrics = metrics
        self.local_rank = local_rank
        self.world_size = world_size
        self.workspace = workspace
        self.ema_decay = ema_decay
        self.ema_update_interval = ema_update_interval
        self.fp16 = fp16
        self.best_mode = best_mode
        self.use_loss_as_metric = use_loss_as_metric
        self.report_metric_at_train = report_metric_at_train
        self.max_keep_ckpt = max_keep_ckpt
        self.eval_interval = eval_interval
        self.use_checkpoint = use_checkpoint
        self.use_tensorboardX = use_tensorboardX
        self.time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        self.scheduler_update_every_step = scheduler_update_every_step
        self.device = device if device is not None else torch.device(f'cuda:{local_rank}')
        
        self.model = model.to(self.device)
        
        # Setup loss functions for 3DGS
        self.l1_loss = l1_loss
        self.ssim_loss = ssim
        self.lambda_ssim = opt.lambda_ssim if hasattr(opt, 'lambda_ssim') else 0.2

        # Additional loss functions
        try:
            from utils_gs.loss_utils import l2_loss, lpips_loss
            self.l2_loss = l2_loss
            self.lpips_loss = lpips_loss
        except ImportError:
            print("[WARNING] Additional loss functions not available")
            self.l2_loss = self.l1_loss  # Fallback
            self.lpips_loss = None

        # Loss weights (configurable)
        self.lambda_l1 = getattr(opt, 'gs_lambda_l1', 1.0)
        self.lambda_l2 = getattr(opt, 'gs_lambda_l2', 0.0)
        self.lambda_lpips = getattr(opt, 'gs_lambda_lpips', 0.0)
        self.lambda_temporal = getattr(opt, 'gs_lambda_temporal', 0.0)
        self.lambda_smoothness = getattr(opt, 'gs_lambda_smoothness', 0.0)
        
        # Initialize optimizer (follow base Trainer pattern)
        if optimizer is None:
            self.optimizer = self.get_optimizer()
        else:
            self.optimizer = optimizer(self.model)

        # Gradient clipping to prevent explosion
        self.max_grad_norm = getattr(opt, 'max_grad_norm', 1.0)
            
        # Initialize scheduler
        if lr_scheduler is None:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, 
                step_size=opt.lr_decay_steps if hasattr(opt, 'lr_decay_steps') else 10000,
                gamma=opt.lr_decay_gamma if hasattr(opt, 'lr_decay_gamma') else 0.5
            )
        else:
            self.lr_scheduler = lr_scheduler(self.optimizer)
            
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.fp16)
        
        # Logging
        self.log_path = os.path.join(workspace, 'logs', self.name)
        self.ckpt_path = os.path.join(workspace, 'checkpoints', self.name)
        os.makedirs(self.log_path, exist_ok=True)
        os.makedirs(self.ckpt_path, exist_ok=True)
        
        # Tensorboard writer
        if self.use_tensorboardX and self.local_rank == 0:
            self.writer = tensorboardX.SummaryWriter(self.log_path)
        else:
            self.writer = None
            
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.local_step = 0
        self.training = True  # Will be set in train() method
        self.stats = {
            "loss": [],
            "valid_loss": [],
            "psnr": [],
        }
        
        # Load checkpoint if specified
        if self.use_checkpoint:
            self.load_checkpoint(self.use_checkpoint)
            
        # Log model info
        if self.local_rank == 0:
            self.log(f"[INFO] Trainer: {self.name}")
            self.log(f"[INFO] Model: {model.__class__.__name__}")
            self.log(f"[INFO] Workspace: {self.workspace}")
            self.log(f"[INFO] Device: {self.device}")
            self.log(f"[INFO] FP16: {self.fp16}")
            
    def log(self, msg, *args, **kwargs):
        """Log message"""
        if self.local_rank == 0 and not self.mute:
            print(f'[{time.strftime("%Y-%m-%d_%H-%M-%S")}] {msg}', *args, **kwargs)
            
    def get_optimizer(self):
        """Setup optimizer with appropriate learning rates"""
        lr_dict = {
            'audio_net': self.opt.lr if hasattr(self.opt, 'lr') else 1e-3,
            'exp_net': self.opt.lr if hasattr(self.opt, 'lr') else 1e-3,
            'deform_net': self.opt.lr_net if hasattr(self.opt, 'lr_net') else 5e-4,
            'gaussians': self.opt.lr_gaussian if hasattr(self.opt, 'lr_gaussian') else 0.0,  # Usually frozen initially
        }
        
        params = self.model.get_params(lr_dict)
        optimizer = torch.optim.Adam(params, betas=(0.9, 0.999))
        return optimizer
    
    def create_camera(self, data):
        """
        Create 3DGS camera from data batch
        
        Args:
            data: Batch dictionary from dataloader
            
        Returns:
            Camera object for 3DGS rendering
        """
        pose = data['poses'][0]  # [4, 4]
        H = data['H']
        W = data['W']
        K = data['K'][0]  # [4, 4]
        FovX = data.get('FovX', None)
        FovY = data.get('FovY', None)
        
        camera = Camera(
            C2W=pose,
            W=W,
            H=H,
            K=K,
            FovX=FovX,
            FovY=FovY
        )
        
        return camera
    
    def train_step(self, data):
        """
        Single training step with 3DGS rendering
        
        Args:
            data: Batch from dataloader
            
        Returns:
            loss value and predictions
        """
        # Data validation - check for NaN/Inf in inputs
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                if torch.isnan(value).any() or torch.isinf(value).any():
                    print(f"[WARNING] NaN/Inf found in input data '{key}', skipping batch")
                    return torch.tensor(0.0, device=self.device, requires_grad=True)

        # Get data
        gt_image = data['images']  # [B, C, H, W]
        auds = data.get('auds', None)
        eye = data.get('eye', None)
        poses = data['poses']
        
        # Create camera for rendering
        camera = self.create_camera(data)
        
        # Forward pass: Deform Gaussians based on audio/expression
        with torch.cuda.amp.autocast(enabled=self.fp16):
            deformed_gaussians = self.model(auds, eye, poses)
            
            # Setup rasterization settings
            bg_color = torch.zeros(3, device=self.device)
            pipe = GaussianRasterizationSettings(
                image_height=data['H'],
                image_width=data['W'], 
                tanfovx=np.tan(camera.FoVx * 0.5),
                tanfovy=np.tan(camera.FoVy * 0.5),
                bg=bg_color,
                scale_modifier=1.0,
                viewmatrix=camera.world_view_transform,
                projmatrix=camera.projection_matrix,
                sh_degree=0,  # Start with SH degree 0
                campos=camera.camera_center,
                prefiltered=False,
                debug=False
            )

            # For canonical training, don't use override_params to allow base Gaussians to be trained
            # For deformation training, use override_params to apply deformations
            if hasattr(self.opt, 'stage') and self.opt.stage == 'canonical':
                # Canonical stage: train base Gaussians directly
                render_pkg = render(
                    camera,
                    self.model.gaussians,  # Base gaussian model (trainable)
                    pipe,
                    bg_color=bg_color,
                    override_params=None  # Don't override
                )
            else:
                # Deformation stages: apply deformations to base Gaussians
                render_pkg = render(
                    camera,
                    self.model.gaussians,  # Base gaussian model
                    pipe,
                    bg_color=bg_color,
                    override_params=deformed_gaussians  # Dynamic deformations
                )
            
            rendered_image = render_pkg["render"]  # [C, H, W]
            rendered_image = rendered_image.unsqueeze(0)  # [1, C, H, W]
            
            # Compute photometric losses
            loss_l1 = self.lambda_l1 * self.l1_loss(rendered_image, gt_image)
            loss_l2 = self.lambda_l2 * self.l2_loss(rendered_image, gt_image) if self.lambda_l2 > 0 else 0.0
            loss_ssim = self.lambda_ssim * (1.0 - self.ssim_loss(rendered_image, gt_image))
            loss_lpips = self.lambda_lpips * self.lpips_loss(rendered_image, gt_image) if self.lpips_loss and self.lambda_lpips > 0 else 0.0

            # Apply masks if training with face regions
            if self.training and 'face_mask' in data:
                face_mask = data['face_mask']  # [H, W]
                if face_mask is not None:
                    face_mask = face_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                    # Weight the loss more on face regions
                    face_weight = 1.0 + 2.0 * face_mask  # Face regions have 3x weight

                    # Apply mask to each loss component
                    loss_l1 = (loss_l1 * face_weight).mean()
                    if self.lambda_l2 > 0:
                        loss_l2 = (loss_l2 * face_weight).mean()
                    loss_ssim = (loss_ssim * face_weight).mean()
                    if self.lambda_lpips > 0:
                        loss_lpips = (loss_lpips * face_weight).mean()
                else:
                    loss_l1 = loss_l1.mean()
                    if self.lambda_l2 > 0:
                        loss_l2 = loss_l2.mean()
                    loss_ssim = loss_ssim.mean()
                    if self.lambda_lpips > 0:
                        loss_lpips = loss_lpips.mean()
            else:
                loss_l1 = loss_l1.mean()
                if self.lambda_l2 > 0:
                    loss_l2 = loss_l2.mean()
                loss_ssim = loss_ssim.mean()
                if self.lambda_lpips > 0:
                    loss_lpips = loss_lpips.mean()

            # Add temporal consistency loss (for deformation stage)
            loss_temporal = 0.0
            if self.lambda_temporal > 0 and hasattr(self, '_prev_rendered'):
                # Temporal consistency - penalize large changes between frames
                temporal_diff = torch.abs(rendered_image - self._prev_rendered)
                loss_temporal = self.lambda_temporal * self.l1_loss(temporal_diff, torch.zeros_like(temporal_diff))
                loss_temporal = loss_temporal.mean()
            self._prev_rendered = rendered_image.detach()

            # Add smoothness regularization (for deformation parameters)
            loss_smoothness = 0.0
            if self.lambda_smoothness > 0 and hasattr(deformed_gaussians, 'd_xyz'):
                # Penalize large deformations (encourage smooth motion)
                d_xyz = deformed_gaussians['d_xyz']
                smoothness_loss = torch.mean(torch.abs(d_xyz))
                loss_smoothness = self.lambda_smoothness * smoothness_loss

            # Total loss with NaN checking
            total_loss = loss_l1 + loss_l2 + loss_ssim + loss_lpips + loss_temporal + loss_smoothness

            # Check for NaN/Inf losses
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"[WARNING] NaN/Inf detected in loss components:")
                print(f"  L1: {loss_l1.item() if torch.is_tensor(loss_l1) else loss_l1}")
                print(f"  L2: {loss_l2.item() if torch.is_tensor(loss_l2) else loss_l2}")
                print(f"  SSIM: {loss_ssim.item() if torch.is_tensor(loss_ssim) else loss_ssim}")
                print(f"  LPIPS: {loss_lpips.item() if torch.is_tensor(loss_lpips) else loss_lpips}")
                print(f"  Temporal: {loss_temporal.item() if torch.is_tensor(loss_temporal) else loss_temporal}")
                print(f"  Smoothness: {loss_smoothness.item() if torch.is_tensor(loss_smoothness) else loss_smoothness}")

                # Skip this batch if loss is invalid
                print("[WARNING] Skipping invalid batch")
                # Return dummy values in expected format: (preds, truths, loss)
                dummy_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                dummy_pred = torch.zeros_like(gt_image)
                return dummy_pred, gt_image, dummy_loss

            loss = total_loss

            # Additional losses for specific regions if needed
            if self.opt.finetune_lips and 'lips_mask' in data:
                lips_mask = data['lips_mask']
                if lips_mask is not None:
                    lips_mask = lips_mask.unsqueeze(0).unsqueeze(0)
                    lips_loss = self.l1_loss(rendered_image * lips_mask, gt_image * lips_mask)
                    loss = loss + 2.0 * lips_loss  # Extra weight on lips
            
        # Return in format expected by base Trainer: preds, truths, loss
        return rendered_image.detach(), gt_image, loss

    def train_one_epoch(self, loader):
        """Override base method to add gradient clipping"""
        self.model.train()
        self.training = True

        total_loss = 0
        step = 0

        if self.rank == 0:
            pbar = tqdm.tqdm(total=len(loader) * loader.batch_size, bar_format='{desc}{percentage:3.0f}% {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

        for data in loader:
            self.local_step += 1
            self.global_step += 1
            step += 1

            data = self.prepare_data(data)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.fp16):
                preds, truths, loss = self.train_step(data)

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient clipping to prevent explosion
            if self.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler_update_every_step:
                self.lr_scheduler.step()

            total_loss += loss.detach()

            if self.ema is not None and self.global_step % self.ema_update_interval == 0:
                self.ema.update()

            if self.rank == 0:
                if self.use_tensorboardX:
                    self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                    self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]['lr'], self.global_step)

                pbar.set_description(f"loss={loss.item():.4f} ({total_loss.item()/step:.4f})")
                pbar.update(loader.batch_size)

        average_loss = total_loss.item() / step

        if self.rank == 0:
            pbar.close()

        if not self.scheduler_update_every_step:
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_scheduler.step(average_loss)
            else:
                self.lr_scheduler.step()

        return average_loss

    def eval_step(self, data):
        """
        Evaluation step (no gradients)
        
        Args:
            data: Batch from dataloader
            
        Returns:
            Rendered image and metrics
        """
        with torch.no_grad():
            pred, gt_image, loss = self.train_step(data)

            # Calculate metrics
            psnr_val = psnr(pred, gt_image).mean()
            
            return pred, {
                'loss': loss.item(),
                'psnr': psnr_val.item()
            }
    
    def train(self, train_loader, valid_loader=None, max_epochs=1000):
        """
        Main training loop

        Args:
            train_loader: Training data loader
            valid_loader: Validation data loader (optional)
            max_epochs: Maximum number of epochs
        """
        self.log(f"[INFO] Start training, max epochs: {max_epochs}")

        for epoch in range(self.epoch, max_epochs):
            self.training = True  # Set training mode
            self.epoch = epoch
            
            # Training
            self.model.train()
            total_loss = 0
            
            if self.local_rank == 0:
                pbar = tqdm.tqdm(total=len(train_loader), desc=f"Epoch {epoch}")
            else:
                pbar = None
                
            for i, data in enumerate(train_loader):
                self.local_step += 1
                self.global_step += 1
                
                # Forward and backward
                self.optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=self.fp16):
                    preds, truths, loss = self.train_step(data)
                    
                self.scaler.scale(loss).backward()

            # Debug: Check if gradients exist and model parameters
            if self.global_step % 100 == 0:
                total_grad_norm = 0
                param_count = 0
                total_params = 0
                trainable_params = 0

                for name, param in self.model.named_parameters():
                    total_params += 1
                    if param.requires_grad:
                        trainable_params += 1
                        if param.grad is not None:
                            grad_norm = param.grad.data.norm(2)
                            total_grad_norm += grad_norm.item() ** 2
                            param_count += 1
                    total_grad_norm = total_grad_norm ** 0.5
                    self.log(f"[DEBUG] Step {self.global_step}: Grad norm: {total_grad_norm:.6f}, Params with grad: {param_count}")

                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # Scheduler step
                if self.scheduler_update_every_step:
                    self.lr_scheduler.step()

                # Debug: Log learning rate occasionally
                if self.global_step % 500 == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.log(f"[DEBUG] Step {self.global_step}: Learning rate: {current_lr}")

                # Logging
                total_loss += loss.item()
                
                if self.local_rank == 0:
                    pbar.update(1)
                    pbar.set_postfix({"loss": loss.item()})
                    
                    if self.writer and self.global_step % 10 == 0:
                        self.writer.add_scalar("train/loss", loss.item(), self.global_step)
                        
                # Skip intermediate checkpoint saves - only save final weights
                    
            if pbar:
                pbar.close()
                
            # Average loss
            avg_loss = total_loss / len(train_loader)
            self.stats["loss"].append(avg_loss)
            self.log(f"[INFO] Epoch {epoch} train loss: {avg_loss:.6f}")
            
            # Scheduler step (per epoch)
            if not self.scheduler_update_every_step:
                self.lr_scheduler.step()
                
            # Validation
            if valid_loader and (epoch + 1) % self.eval_interval == 0:
                self.evaluate(valid_loader)

        # Save final weights only at the end of training
        self.log("[INFO] Training completed, saving final weights...")
        self.save_checkpoint("final")
                
    def evaluate(self, loader, save_path=None):
        """
        Evaluate model on validation/test set

        Args:
            loader: Data loader
            save_path: Path to save rendered images (optional)
        """
        self.log("[INFO] Start evaluation...")
        self.training = False  # Set evaluation mode
        self.model.eval()
        
        total_loss = 0
        total_psnr = 0
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader), desc="Evaluating")
        else:
            pbar = None
            
        for i, data in enumerate(loader):
            pred, metrics = self.eval_step(data)
            
            total_loss += metrics['loss']
            total_psnr += metrics['psnr']
            
            # Save images
            if save_path and i < 100:  # Save first 100 frames
                # Convert to numpy and save
                pred_np = pred[0].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
                pred_np = (pred_np * 255).astype(np.uint8)
                cv2.imwrite(os.path.join(save_path, f"{i:04d}.png"), cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR))
                
            if pbar:
                pbar.update(1)
                pbar.set_postfix(metrics)
                
        if pbar:
            pbar.close()
            
        # Average metrics
        avg_loss = total_loss / len(loader)
        avg_psnr = total_psnr / len(loader)
        
        self.stats["valid_loss"].append(avg_loss)
        self.stats["psnr"].append(avg_psnr)
        
        self.log(f"[INFO] Evaluation - Loss: {avg_loss:.6f}, PSNR: {avg_psnr:.2f}")
        
        if self.writer:
            self.writer.add_scalar("val/loss", avg_loss, self.epoch)
            self.writer.add_scalar("val/psnr", avg_psnr, self.epoch)
            
        return avg_loss, avg_psnr
    
    def test(self, loader, save_path, write_video=True):
        """
        Test model and save results
        
        Args:
            loader: Test data loader
            save_path: Path to save results
            write_video: Whether to create video from frames
        """
        self.log(f"[INFO] Start testing, save to {save_path}")
        self.model.eval()
        
        os.makedirs(save_path, exist_ok=True)
        
        if self.local_rank == 0:
            pbar = tqdm.tqdm(total=len(loader), desc="Testing")
        else:
            pbar = None
            
        all_preds = []
        
        for i, data in enumerate(loader):
            with torch.no_grad():
                pred, metrics = self.eval_step(data)
                
            # Convert to numpy
            pred_np = pred[0].permute(1, 2, 0).cpu().numpy()  # [H, W, C]
            pred_np = (pred_np * 255).astype(np.uint8)
            
            # Save frame
            cv2.imwrite(os.path.join(save_path, f"{i:06d}.png"), cv2.cvtColor(pred_np, cv2.COLOR_RGB2BGR))
            all_preds.append(pred_np)
            
            if pbar:
                pbar.update(1)
                pbar.set_postfix({"frame": i})
                
        if pbar:
            pbar.close()
            
        # Create video if requested
        if write_video and len(all_preds) > 0:
            video_path = os.path.join(save_path, "output.mp4")
            self.log(f"[INFO] Writing video to {video_path}")
            
            # Get video properties
            H, W = all_preds[0].shape[:2]
            fps = 25  # Default FPS
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(video_path, fourcc, fps, (W, H))
            
            for frame in all_preds:
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
            video_writer.release()
            self.log(f"[INFO] Video saved: {video_path}")
            
    def save_checkpoint(self, name=None):
        """Save model checkpoint"""
        if self.local_rank != 0:
            return
            
        state = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'stats': self.stats,
        }
        
        if name is None:
            name = f"checkpoint_{self.global_step}"
            
        file_path = os.path.join(self.ckpt_path, f"{name}.pth")
        torch.save(state, file_path)
        
        # Also save as latest
        latest_path = os.path.join(self.ckpt_path, "latest.pth")
        torch.save(state, latest_path)
        
        self.log(f"[INFO] Saved checkpoint: {file_path}")
        
    def load_checkpoint(self, checkpoint):
        """Load model checkpoint"""
        if checkpoint == "latest":
            checkpoint_path = os.path.join(self.ckpt_path, "latest.pth")
        else:
            checkpoint_path = checkpoint
            
        if not os.path.exists(checkpoint_path):
            self.log(f"[WARN] Checkpoint {checkpoint_path} not found, skip loading.")
            return
            
        checkpoint_dict = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if 'model' in checkpoint_dict:
            missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model'], strict=False)
            if missing_keys:
                self.log(f"[WARN] Missing keys: {missing_keys}")
            if unexpected_keys:
                self.log(f"[WARN] Unexpected keys: {unexpected_keys}")
                
        # Load training state
        if 'epoch' in checkpoint_dict:
            self.epoch = checkpoint_dict['epoch']
        if 'global_step' in checkpoint_dict:
            self.global_step = checkpoint_dict['global_step']
        if 'optimizer' in checkpoint_dict and self.optimizer:
            self.optimizer.load_state_dict(checkpoint_dict['optimizer'])
        if 'lr_scheduler' in checkpoint_dict and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint_dict['lr_scheduler'])
        if 'scaler' in checkpoint_dict and self.scaler:
            self.scaler.load_state_dict(checkpoint_dict['scaler'])
        if 'stats' in checkpoint_dict:
            self.stats = checkpoint_dict['stats']
            
        self.log(f"[INFO] Loaded checkpoint from {checkpoint_path} (epoch={self.epoch})")
