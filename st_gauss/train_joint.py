"""
ST-Gauss Stage 3: Joint Fine-tuning

Harmonize the boundary between Gaussian Lips and NeRF Teeth.
Fuses both branches for final high-quality output.

Key Features:
- Freeze NeRF geometry, allow gradients to flow into:
  - Color/Alpha of Gaussian Lips (for blending edge refinement)
  - Color of NeRF (for lighting consistency)
- L1 + LPIPS loss on full image
- Optional Portrait-Sync Generator for hair/detail enhancement
"""

import os
import sys
import copy
import random
import argparse
import torch
import torch.nn.functional as F
import lpips
from random import randint
from tqdm import tqdm
from argparse import Namespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from st_gauss.mouth_nerf import MouthNeRFNetwork
from st_gauss.face_motion import BlendshapeMotionNetwork
from st_gauss.hybrid_renderer import HybridRenderer, render_hybrid
from st_gauss.config import STGaussConfig, get_config

from TalkingGaussian.scene import Scene, GaussianModel
from TalkingGaussian.utils.loss_utils import l1_loss, ssim, patchify
from TalkingGaussian.utils.general_utils import safe_state
from TalkingGaussian.utils.camera_utils import loadCamOnTheFly
from TalkingGaussian.arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training_stage3(
    dataset,
    config: STGaussConfig,
    pipe,
    face_checkpoint: str = None,
    mouth_checkpoint: str = None,
):
    """
    Stage 3: Joint Fine-tuning.
    
    Goal: Harmonize the boundary between Gaussian Lips and NeRF Teeth.
    
    Operation:
    - Freeze geometry of NeRF
    - Allow gradients to flow into:
        - Color/Alpha of Gaussian Lips (blending edge)
        - Color of NeRF (lighting consistency)
    
    Loss: L1 + LPIPS on full image
    """
    # Setup iterations
    iterations = config.stage3_iterations
    save_interval = config.stage3_save_interval
    lpips_start_iter = iterations // 4
    
    testing_iterations = list(range(0, iterations + 1, 1000))
    saving_iterations = list(range(0, iterations + 1, save_interval)) + [iterations]
    
    # Initialize tensorboard
    tb_writer = prepare_output_and_logger(dataset)
    
    # Initialize models
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    
    # Face motion network
    motion_net = BlendshapeMotionNetwork(
        audio_type=config.audio_type,
        audio_dim=config.audio_dim,
        blendshape_dim=config.blendshape_dim,
        bound=config.face_bound,
        use_facial_attention=config.use_facial_aware_attention,
        args=dataset,
    ).cuda()
    
    # Mouth NeRF
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
    
    # Load Stage 1 checkpoint (Mouth NeRF)
    if mouth_checkpoint is None:
        mouth_checkpoint = os.path.join(scene.model_path, "chkpnt_mouth_nerf_latest.pth")
    
    if os.path.exists(mouth_checkpoint):
        print(f"Loading Mouth NeRF from {mouth_checkpoint}")
        mouth_ckpt = torch.load(mouth_checkpoint)
        mouth_nerf.load_state_dict(mouth_ckpt['mouth_nerf'])
    else:
        raise FileNotFoundError(f"Mouth NeRF checkpoint not found: {mouth_checkpoint}")
    
    # Load Stage 2 checkpoint (Face Gaussians)
    if face_checkpoint is None:
        face_checkpoint = os.path.join(scene.model_path, "chkpnt_face_latest.pth")
    
    if os.path.exists(face_checkpoint):
        print(f"Loading Face Gaussians from {face_checkpoint}")
        face_ckpt = torch.load(face_checkpoint)
        gaussians.restore(face_ckpt['gaussians'], None)
        motion_net.load_state_dict(face_ckpt['motion_net'])
    else:
        raise FileNotFoundError(f"Face checkpoint not found: {face_checkpoint}")
    
    # Freeze NeRF geometry
    if config.stage3_freeze_nerf_geometry:
        print("Freezing NeRF geometry (sigma network, encoders)")
        for param in mouth_nerf.encoder_xy.parameters():
            param.requires_grad = False
        for param in mouth_nerf.encoder_yz.parameters():
            param.requires_grad = False
        for param in mouth_nerf.encoder_xz.parameters():
            param.requires_grad = False
        for param in mouth_nerf.sigma_net.parameters():
            param.requires_grad = False
        for param in mouth_nerf.audio_net.parameters():
            param.requires_grad = False
        if hasattr(mouth_nerf, 'audio_att_net'):
            for param in mouth_nerf.audio_att_net.parameters():
                param.requires_grad = False
    
    # Freeze motion network (keep face geometry fixed)
    for param in motion_net.parameters():
        param.requires_grad = False
    
    # Setup optimizers for fine-tuning
    # Only optimize: Gaussian colors/opacity + NeRF color network
    gaussian_params = [
        {'params': [gaussians._features_dc], 'lr': config.stage3_lr, 'name': 'f_dc'},
        {'params': [gaussians._features_rest], 'lr': config.stage3_lr / 20.0, 'name': 'f_rest'},
        {'params': [gaussians._opacity], 'lr': config.stage3_lr * 0.5, 'name': 'opacity'},
    ]
    gaussian_optimizer = torch.optim.Adam(gaussian_params, lr=0.0, eps=1e-15)
    
    nerf_color_params = [
        {'params': mouth_nerf.color_net.parameters(), 'lr': config.stage3_lr_net},
    ]
    nerf_optimizer = torch.optim.Adam(nerf_color_params, lr=config.stage3_lr_net, eps=1e-8)
    
    # LPIPS loss
    lpips_criterion = lpips.LPIPS(net='alex').eval().cuda()
    
    # Hybrid renderer
    renderer = HybridRenderer(
        background_color=config.background_color,
        debug=pipe.debug if hasattr(pipe, 'debug') else False,
    ).cuda()
    
    # Background
    background = torch.tensor(config.background_color, dtype=torch.float32, device="cuda")
    
    # Training loop
    viewpoint_stack = None
    ema_loss = 0.0
    progress_bar = tqdm(range(iterations), desc="Stage 3: Joint Fine-tuning")
    
    for iteration in range(1, iterations + 1):
        gaussians.update_learning_rate(iteration)
        
        # Pick random camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        
        if viewpoint_cam.original_image is None:
            viewpoint_cam = loadCamOnTheFly(copy.deepcopy(viewpoint_cam))
        
        # Get masks
        face_mask = torch.as_tensor(viewpoint_cam.talking_dict["face_mask"]).cuda()
        hair_mask = torch.as_tensor(viewpoint_cam.talking_dict.get("hair_mask", torch.zeros_like(face_mask))).cuda()
        mouth_mask = torch.as_tensor(viewpoint_cam.talking_dict["mouth_mask"]).cuda()
        head_mask = face_mask | hair_mask | mouth_mask
        
        # Hybrid render
        render_pkg = renderer(
            viewpoint_camera=viewpoint_cam,
            gaussian_model=gaussians,
            face_motion_network=motion_net,
            mouth_nerf=mouth_nerf,
            pipe=pipe,
            mouth_mask=mouth_mask,
            render_mouth=True,
        )
        
        image = render_pkg['render']
        face_alpha = render_pkg['face_alpha']
        
        # Ground truth
        gt_image = viewpoint_cam.original_image.cuda() / 255.0
        
        # L1 + SSIM loss on full image
        loss_l1 = l1_loss(image, gt_image)
        loss = config.lambda_l1 * loss_l1
        loss += config.lambda_ssim * (1.0 - ssim(image, gt_image))
        
        # Alpha regularization
        loss += config.lambda_alpha_reg * (
            ((1 - face_alpha) * head_mask.float()).mean() +
            (face_alpha * (~head_mask).float()).mean()
        )
        
        # LPIPS loss
        if iteration > lpips_start_iter:
            patch_size = random.randint(16, 24) * 2
            loss += config.lambda_lpips * lpips_criterion(
                patchify(image[None, ...] * 2 - 1, patch_size),
                patchify(gt_image[None, ...] * 2 - 1, patch_size)
            ).mean()
            
            # Mouth region LPIPS
            lips_rect = viewpoint_cam.talking_dict.get('lips_rect', None)
            if lips_rect is not None:
                xmin, xmax, ymin, ymax = lips_rect
                if (xmax - xmin) >= 16 and (ymax - ymin) >= 16:
                    loss += config.lambda_lpips * 0.5 * lpips_criterion(
                        image[:, xmin:xmax, ymin:ymax].unsqueeze(0) * 2 - 1,
                        gt_image[:, xmin:xmax, ymin:ymax].unsqueeze(0) * 2 - 1
                    ).mean()
        
        # Backward
        gaussian_optimizer.zero_grad()
        nerf_optimizer.zero_grad()
        loss.backward()
        gaussian_optimizer.step()
        nerf_optimizer.step()
        
        # Logging
        ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
        
        if iteration % 10 == 0:
            progress_bar.set_postfix({"Loss": f"{ema_loss:.5f}"})
            progress_bar.update(10)
        
        with torch.no_grad():
            # Tensorboard
            if tb_writer and iteration % 100 == 0:
                tb_writer.add_scalar('stage3/loss', loss.item(), iteration)
            
            # Validation
            if iteration in testing_iterations:
                validate_joint(
                    renderer, gaussians, motion_net, mouth_nerf,
                    scene, tb_writer, iteration, pipe
                )
            
            # Save
            if iteration in saving_iterations:
                print(f"\n[Stage 3 - Iter {iteration}] Saving Checkpoint")
                
                checkpoint = {
                    'gaussians': gaussians.capture(),
                    'motion_net': motion_net.state_dict(),
                    'mouth_nerf': mouth_nerf.state_dict(),
                    'iteration': iteration,
                }
                
                torch.save(checkpoint, os.path.join(scene.model_path, f"chkpnt_joint_{iteration}.pth"))
                torch.save(checkpoint, os.path.join(scene.model_path, "chkpnt_joint_latest.pth"))
    
    progress_bar.close()
    print("\nStage 3: Joint Fine-tuning Complete.")
    
    return gaussians, motion_net, mouth_nerf


def validate_joint(renderer, gaussians, motion_net, mouth_nerf, scene, tb_writer, iteration, pipe):
    """Validation for joint model."""
    torch.cuda.empty_cache()
    
    val_cameras = scene.getTestCameras()[:5]
    
    for idx, viewpoint in enumerate(val_cameras):
        if viewpoint.original_image is None:
            viewpoint = loadCamOnTheFly(copy.deepcopy(viewpoint))
        
        mouth_mask = torch.as_tensor(viewpoint.talking_dict["mouth_mask"]).cuda()
        
        with torch.no_grad():
            render_pkg = renderer(
                viewpoint_camera=viewpoint,
                gaussian_model=gaussians,
                face_motion_network=motion_net,
                mouth_nerf=mouth_nerf,
                pipe=pipe,
                mouth_mask=mouth_mask,
                render_mouth=True,
            )
        
        image = torch.clamp(render_pkg['render'], 0.0, 1.0)
        gt_image = viewpoint.original_image.cuda() / 255.0
        
        if tb_writer:
            tb_writer.add_images(f"stage3_val/render_{idx}", image.unsqueeze(0), global_step=iteration)
            tb_writer.add_images(f"stage3_val/gt_{idx}", gt_image.unsqueeze(0), global_step=iteration)
            
            # Show face and mouth separately
            if 'face_render' in render_pkg:
                face_img = torch.clamp(render_pkg['face_render'], 0.0, 1.0)
                tb_writer.add_images(f"stage3_val/face_{idx}", face_img.unsqueeze(0), global_step=iteration)
            
            if 'mouth_render' in render_pkg:
                mouth_img = torch.clamp(render_pkg['mouth_render'], 0.0, 1.0)
                tb_writer.add_images(f"stage3_val/mouth_{idx}", mouth_img.unsqueeze(0), global_step=iteration)
    
    torch.cuda.empty_cache()


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
    
    return tb_writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ST-Gauss Stage 3: Joint Fine-tuning")
    
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument('--config', type=str, default='default')
    parser.add_argument('--face_checkpoint', type=str, default=None)
    parser.add_argument('--mouth_checkpoint', type=str, default=None)
    parser.add_argument('--quiet', action='store_true')
    
    args = parser.parse_args()
    
    config = get_config(args.config)
    
    print(f"Training ST-Gauss Stage 3: Joint Fine-tuning")
    
    safe_state(args.quiet)
    
    training_stage3(
        dataset=lp.extract(args),
        config=config,
        pipe=pp.extract(args),
        face_checkpoint=args.face_checkpoint,
        mouth_checkpoint=args.mouth_checkpoint,
    )
    
    print("\nStage 3 Training Complete.")

