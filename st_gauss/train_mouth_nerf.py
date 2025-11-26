"""
ST-Gauss Stage 1: Mouth NeRF Training

Train the Tri-Plane Hash NeRF to reconstruct the interior mouth region.
This stage focuses on teeth, tongue, and oral cavity rendering.

Key Features:
- Uses mouth_mask to focus loss on mouth interior only
- NeRF provides view-consistent rendering for complex concave structures
- Prevents "teeth flickering" often seen in point-based methods
"""

import os
import sys
import copy
import random
import argparse
import torch
import torch.nn.functional as F
import lpips
from tqdm import tqdm
from argparse import Namespace

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from st_gauss.mouth_nerf import MouthNeRFNetwork
from st_gauss.config import STGaussConfig, get_config

# TalkingGaussian imports for data loading
from TalkingGaussian.scene import Scene, GaussianModel
from TalkingGaussian.utils.loss_utils import l1_loss, ssim
from TalkingGaussian.utils.general_utils import safe_state
from TalkingGaussian.utils.camera_utils import loadCamOnTheFly
from TalkingGaussian.arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training_stage1(
    dataset,
    config: STGaussConfig,
    pipe,
    checkpoint_path: str = None,
):
    """
    Stage 1: Train Mouth NeRF.
    
    Goal: Train the TriHash NeRF to reconstruct the interior mouth.
    Loss Mask: Calculate loss only on pixels inside the mouth_mask.
    
    Why NeRF: Excellent at view-consistency for complex concave structures
    like the oral cavity, preventing teeth flickering.
    """
    # Setup iterations and checkpoints
    iterations = config.stage1_iterations
    warmup_iter = config.stage1_warmup_iterations
    save_interval = config.stage1_save_interval
    testing_iterations = list(range(0, iterations + 1, 2000))
    saving_iterations = list(range(0, iterations + 1, save_interval)) + [iterations]
    
    # Initialize tensorboard
    tb_writer = prepare_output_and_logger(dataset)
    
    # Initialize scene (for camera/data loading)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    
    # Initialize Mouth NeRF
    mouth_nerf = MouthNeRFNetwork(
        audio_type=config.audio_type,
        audio_dim=config.audio_dim,
        bound=config.mouth_bound,
        num_hash_levels=config.mouth_hash_levels,
        hash_level_dim=config.mouth_hash_level_dim,
        hash_base_resolution=config.mouth_hash_base_resolution,
        hash_log2_size=config.mouth_hash_log2_hashmap_size,
        hidden_dim=config.mouth_hidden_dim,
        num_layers=config.mouth_num_layers,
        geo_feat_dim=config.mouth_geo_feat_dim,
    ).cuda()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        mouth_nerf.get_params(config.stage1_lr, config.stage1_lr_net),
        betas=(0.9, 0.99),
        eps=1e-8
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda iter: 0.5 ** (iter / (iterations // 2)) if iter < iterations // 2 else 0.1
    )
    
    # LPIPS loss
    lpips_criterion = lpips.LPIPS(net='alex').eval().cuda()
    
    # Load checkpoint if provided
    first_iter = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        mouth_nerf.load_state_dict(checkpoint['mouth_nerf'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        first_iter = checkpoint['iteration']
    
    # Background color
    bg_color = list(config.background_color)
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Training loop
    viewpoint_stack = None
    ema_loss = 0.0
    progress_bar = tqdm(range(first_iter, iterations), desc="Stage 1: Mouth NeRF")
    
    for iteration in range(first_iter + 1, iterations + 1):
        mouth_nerf.train()
        
        # Pick random camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack) - 1))
        
        # Load image on the fly if needed
        if viewpoint_cam.original_image is None:
            viewpoint_cam = loadCamOnTheFly(copy.deepcopy(viewpoint_cam))
        
        # Get masks
        mouth_mask = torch.as_tensor(viewpoint_cam.talking_dict["mouth_mask"]).cuda()
        
        # Skip frames with very small mouth regions
        if mouth_mask.sum() < 50:
            continue
        
        # Get ground truth
        gt_image = viewpoint_cam.original_image.cuda() / 255.0
        H, W = gt_image.shape[1], gt_image.shape[2]
        
        # Get audio features
        audio_features = viewpoint_cam.talking_dict["auds"].cuda()
        
        # Generate rays for mouth region only (for efficiency)
        mouth_coords = torch.nonzero(mouth_mask, as_tuple=False)  # [M, 2] (y, x)
        
        if mouth_coords.shape[0] == 0:
            continue
        
        # Sample subset of rays for training (for memory efficiency)
        num_rays = min(mouth_coords.shape[0], 4096)
        ray_indices = torch.randperm(mouth_coords.shape[0])[:num_rays]
        sampled_coords = mouth_coords[ray_indices]
        
        # Camera intrinsics
        fx = W / (2 * torch.tan(torch.tensor(viewpoint_cam.FoVx * 0.5)))
        fy = H / (2 * torch.tan(torch.tensor(viewpoint_cam.FoVy * 0.5)))
        cx, cy = W / 2, H / 2
        
        # Generate rays
        y_coords = sampled_coords[:, 0].float()
        x_coords = sampled_coords[:, 1].float()
        
        dirs_x = (x_coords - cx) / fx
        dirs_y = (y_coords - cy) / fy
        dirs_z = torch.ones_like(dirs_x)
        
        rays_d_cam = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)
        rays_d_cam = F.normalize(rays_d_cam, dim=-1)
        
        # Transform to world space
        c2w = viewpoint_cam.world_view_transform.inverse()[:3, :3]
        rays_d = rays_d_cam @ c2w.T.cuda()
        rays_o = viewpoint_cam.camera_center.unsqueeze(0).expand(num_rays, -1)
        
        # Render through NeRF
        render_results = mouth_nerf.render_rays(
            rays_o=rays_o,
            rays_d=rays_d,
            audio_features=audio_features,
            near=0.01,
            far=0.5,
            num_samples=64,
            perturb=True,
        )
        
        # Get ground truth pixels
        gt_pixels = gt_image[:, sampled_coords[:, 0], sampled_coords[:, 1]].T  # [M, 3]
        
        # Losses
        pred_rgb = render_results['rgb']
        
        # L1 loss
        loss_l1 = F.l1_loss(pred_rgb, gt_pixels)
        
        # Uncertainty-weighted loss (if available)
        if render_results.get('uncertainty') is not None:
            uncertainty = render_results['uncertainty'].clamp(min=0.01)
            loss_l1 = (loss_l1 / uncertainty).mean() + uncertainty.mean() * 0.01
        
        # Total loss
        loss = loss_l1 * config.lambda_l1
        
        # Regularization during warmup
        if iteration < warmup_iter:
            # Encourage learning basic structure first
            loss = loss * 0.5
        
        # LPIPS loss after warmup (on patches)
        if iteration > warmup_iter * 2 and mouth_mask.sum() > 256:
            # Render full mouth region for LPIPS
            with torch.no_grad():
                full_render = render_full_mouth(
                    mouth_nerf, viewpoint_cam, mouth_mask, audio_features
                )
            
            if full_render is not None:
                # Extract mouth patch
                y_min, y_max = mouth_coords[:, 0].min(), mouth_coords[:, 0].max()
                x_min, x_max = mouth_coords[:, 1].min(), mouth_coords[:, 1].max()
                
                pred_patch = full_render[:, y_min:y_max+1, x_min:x_max+1]
                gt_patch = gt_image[:, y_min:y_max+1, x_min:x_max+1]
                
                if pred_patch.shape[1] >= 16 and pred_patch.shape[2] >= 16:
                    lpips_loss = lpips_criterion(
                        pred_patch.unsqueeze(0) * 2 - 1,
                        gt_patch.unsqueeze(0) * 2 - 1
                    ).mean()
                    loss = loss + config.lambda_lpips * lpips_loss
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Logging
        ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
        
        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{ema_loss:.5f}"})
            progress_bar.update(10)
        
        # Tensorboard logging
        if tb_writer and iteration % 100 == 0:
            tb_writer.add_scalar('stage1/loss', loss.item(), iteration)
            tb_writer.add_scalar('stage1/lr', scheduler.get_last_lr()[0], iteration)
        
        # Save checkpoint
        if iteration in saving_iterations:
            print(f"\n[Stage 1 - Iter {iteration}] Saving Checkpoint")
            checkpoint = {
                'mouth_nerf': mouth_nerf.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': iteration,
                'config': config.to_dict(),
            }
            save_path = os.path.join(scene.model_path, f"chkpnt_mouth_nerf_{iteration}.pth")
            torch.save(checkpoint, save_path)
            torch.save(checkpoint, os.path.join(scene.model_path, "chkpnt_mouth_nerf_latest.pth"))
        
        # Validation
        if iteration in testing_iterations:
            validate_mouth_nerf(
                mouth_nerf, scene, tb_writer, iteration, config
            )
    
    progress_bar.close()
    print("\nStage 1: Mouth NeRF Training Complete.")
    
    return mouth_nerf


def render_full_mouth(mouth_nerf, viewpoint_cam, mouth_mask, audio_features):
    """Render full mouth region for validation/LPIPS."""
    mouth_nerf.eval()
    
    with torch.no_grad():
        mouth_coords = torch.nonzero(mouth_mask, as_tuple=False)
        
        if mouth_coords.shape[0] == 0:
            return None
        
        H, W = mouth_mask.shape
        
        fx = W / (2 * torch.tan(torch.tensor(viewpoint_cam.FoVx * 0.5)))
        fy = H / (2 * torch.tan(torch.tensor(viewpoint_cam.FoVy * 0.5)))
        cx, cy = W / 2, H / 2
        
        y_coords = mouth_coords[:, 0].float()
        x_coords = mouth_coords[:, 1].float()
        
        dirs_x = (x_coords - cx) / fx
        dirs_y = (y_coords - cy) / fy
        dirs_z = torch.ones_like(dirs_x)
        
        rays_d_cam = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)
        rays_d_cam = F.normalize(rays_d_cam, dim=-1)
        
        c2w = viewpoint_cam.world_view_transform.inverse()[:3, :3]
        rays_d = rays_d_cam @ c2w.T.cuda()
        rays_o = viewpoint_cam.camera_center.unsqueeze(0).expand(mouth_coords.shape[0], -1)
        
        # Render in batches
        batch_size = 4096
        all_rgb = []
        
        for i in range(0, rays_o.shape[0], batch_size):
            batch_o = rays_o[i:i+batch_size]
            batch_d = rays_d[i:i+batch_size]
            
            results = mouth_nerf.render_rays(
                rays_o=batch_o,
                rays_d=batch_d,
                audio_features=audio_features,
                near=0.01,
                far=0.5,
                num_samples=64,
                perturb=False,
            )
            all_rgb.append(results['rgb'])
        
        all_rgb = torch.cat(all_rgb, dim=0)
        
        # Create full image
        full_image = torch.zeros(3, H, W, device=mouth_mask.device)
        full_image[:, mouth_coords[:, 0], mouth_coords[:, 1]] = all_rgb.T
        
        return full_image


def validate_mouth_nerf(mouth_nerf, scene, tb_writer, iteration, config):
    """Validation for mouth NeRF."""
    mouth_nerf.eval()
    
    val_cameras = scene.getTestCameras()[:5]
    
    for idx, viewpoint in enumerate(val_cameras):
        if viewpoint.original_image is None:
            viewpoint = loadCamOnTheFly(copy.deepcopy(viewpoint))
        
        mouth_mask = torch.as_tensor(viewpoint.talking_dict["mouth_mask"]).cuda()
        audio_features = viewpoint.talking_dict["auds"].cuda()
        
        with torch.no_grad():
            rendered = render_full_mouth(mouth_nerf, viewpoint, mouth_mask, audio_features)
        
        if rendered is not None and tb_writer:
            gt_image = viewpoint.original_image.cuda() / 255.0
            
            # Apply mask for visualization
            masked_render = rendered * mouth_mask.float()
            masked_gt = gt_image * mouth_mask.float()
            
            tb_writer.add_images(
                f"stage1_val/render_{idx}",
                masked_render.unsqueeze(0).clamp(0, 1),
                global_step=iteration
            )
            tb_writer.add_images(
                f"stage1_val/gt_{idx}",
                masked_gt.unsqueeze(0),
                global_step=iteration
            )
    
    mouth_nerf.train()


def prepare_output_and_logger(args):
    """Setup output directory and tensorboard."""
    if not args.model_path:
        import uuid
        args.model_path = os.path.join("./output/", str(uuid.uuid4())[:10])
    
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as f:
        f.write(str(Namespace(**vars(args))))
    
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available")
    
    return tb_writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ST-Gauss Stage 1: Mouth NeRF Training")
    
    # Model parameters
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    # ST-Gauss specific
    parser.add_argument('--config', type=str, default='default', help='Config preset')
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--quiet', action='store_true')
    
    args = parser.parse_args()
    
    # Load config
    config = get_config(args.config)
    config.stage1_iterations = args.iterations if hasattr(args, 'iterations') else config.stage1_iterations
    
    print(f"Training ST-Gauss Stage 1: Mouth NeRF")
    print(f"Data path: {args.source_path}")
    
    safe_state(args.quiet)
    
    training_stage1(
        dataset=lp.extract(args),
        config=config,
        pipe=pp.extract(args),
        checkpoint_path=args.checkpoint,
    )
    
    print("\nStage 1 Training Complete.")

