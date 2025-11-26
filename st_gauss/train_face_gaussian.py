"""
ST-Gauss Stage 2: Face Gaussian Training

Train Deformable 3D Gaussians for face skin with BlendshapeMotionNetwork.
This stage focuses on skin, eyes, lips exterior, and general head shape.

Key Features:
- Uses 52 ARKit blendshapes instead of simple Action Units
- Facial-Aware Masked-Attention for audio/expression disentanglement
- Loss mask excludes mouth interior (trained separately in Stage 1)
- Lip-Sync loss for proper lip closure supervision
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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from st_gauss.face_motion import BlendshapeMotionNetwork
from st_gauss.config import STGaussConfig, get_config
from st_gauss.hybrid_renderer import compute_lip_opacity_loss

# TalkingGaussian imports
from TalkingGaussian.scene import Scene, GaussianModel
from TalkingGaussian.gaussian_renderer import render
from TalkingGaussian.utils.loss_utils import l1_loss, ssim, patchify
from TalkingGaussian.utils.general_utils import safe_state
from TalkingGaussian.utils.camera_utils import loadCamOnTheFly
from TalkingGaussian.utils.sh_utils import eval_sh
from TalkingGaussian.arguments import ModelParams, PipelineParams, OptimizationParams

from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
import math

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def render_motion_blendshape(
    viewpoint_camera,
    pc: GaussianModel,
    motion_net: BlendshapeMotionNetwork,
    pipe,
    bg_color: torch.Tensor,
    scaling_modifier: float = 1.0,
    return_attn: bool = False,
):
    """
    Render face with BlendshapeMotionNetwork deformations.
    """
    # Create screen-space points for gradient tracking
    screenspace_points = torch.zeros_like(
        pc.get_xyz,
        dtype=pc.get_xyz.dtype,
        requires_grad=True,
        device="cuda"
    ) + 0
    
    try:
        screenspace_points.retain_grad()
    except:
        pass
    
    # Rasterization settings
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )
    
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # Get conditioning features
    audio_feat = viewpoint_camera.talking_dict["auds"].cuda()
    
    # Get blendshapes (52 ARKit params) or fall back to au_exp
    blendshapes = viewpoint_camera.talking_dict.get("blendshapes", None)
    if blendshapes is None:
        # Fallback: use au_exp and pad to 52 dimensions
        au_exp = viewpoint_camera.talking_dict.get("au_exp", torch.zeros(6))
        blendshapes = torch.zeros(52).cuda()
        blendshapes[:min(6, len(au_exp))] = au_exp[:min(6, len(au_exp))].cuda()
    else:
        blendshapes = blendshapes.cuda()
    
    # Predict deformations
    motion_preds = motion_net(pc.get_xyz, audio_feat, blendshapes)
    
    # Apply deformations
    means3D = pc.get_xyz + motion_preds['d_xyz']
    means2D = screenspace_points
    opacity = pc.get_opacity
    
    scales = pc.scaling_activation(pc._scaling + motion_preds['d_scale'])
    rotations = pc.rotation_activation(pc._rotation + motion_preds['d_rot'])
    
    shs = pc.get_features
    
    # Rasterize
    rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=None,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=None,
    )
    
    result = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "depth": rendered_depth,
        "alpha": rendered_alpha,
        "radii": radii,
        "motion": motion_preds,
    }
    
    # Render attention maps if requested
    if return_attn:
        attn_precomp = torch.cat([
            motion_preds['ambient_aud'],
            motion_preds['ambient_exp'],
            torch.zeros_like(motion_preds['ambient_exp'])
        ], dim=-1)
        
        rendered_attn, _, _, _ = rasterizer(
            means3D=means3D.detach(),
            means2D=means2D,
            shs=None,
            colors_precomp=attn_precomp,
            opacities=opacity.detach(),
            scales=scales.detach(),
            rotations=rotations.detach(),
            cov3D_precomp=None,
        )
        result['attn'] = rendered_attn
    
    return result


def training_stage2(
    dataset,
    config: STGaussConfig,
    pipe,
    checkpoint_path: str = None,
):
    """
    Stage 2: Train Face Gaussians with Blendshape conditioning.
    
    Goal: Train Deformable Gaussians for face skin.
    Loss Mask: Mask out mouth interior. Train on face_mask + hair_mask.
    Conditioning: BlendshapeMotionNetwork with 52 ARKit blendshapes.
    Constraint: Lip-Sync loss for proper lip closure.
    """
    # Setup iterations
    iterations = config.stage2_iterations
    warmup_iter = config.stage2_warmup_iterations
    densify_until_iter = config.stage2_densify_until_iter
    densify_from_iter = config.stage2_densify_from_iter
    densification_interval = config.stage2_densification_interval
    opacity_reset_interval = config.stage2_opacity_reset_interval
    save_interval = config.stage2_save_interval
    
    lpips_start_iter = densify_until_iter - 2000
    motion_stop_iter = iterations - 1000
    mouth_select_iter = iterations - 10000
    mouth_step = 1 / max(mouth_select_iter, 1)
    
    testing_iterations = list(range(0, iterations + 1, 2000))
    saving_iterations = list(range(0, iterations + 1, save_interval)) + [iterations]
    
    # Initialize tensorboard
    tb_writer = prepare_output_and_logger(dataset)
    
    # Initialize models
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    
    motion_net = BlendshapeMotionNetwork(
        audio_type=config.audio_type,
        audio_dim=config.audio_dim,
        blendshape_dim=config.blendshape_dim,
        bound=config.face_bound,
        num_hash_levels=config.face_hash_levels,
        hash_level_dim=config.face_hash_level_dim,
        hidden_dim=config.face_motion_hidden_dim,
        num_layers=config.face_motion_num_layers,
        use_facial_attention=config.use_facial_aware_attention,
        args=dataset,
    ).cuda()
    
    # Optimizers
    motion_optimizer = torch.optim.AdamW(
        motion_net.get_params(config.stage2_lr, config.stage2_lr_net),
        betas=(0.9, 0.99),
        eps=1e-8
    )
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        motion_optimizer,
        lambda iter: (0.5 ** (iter / mouth_select_iter)) if iter < mouth_select_iter else 0.1 ** (iter / iterations)
    )
    
    gaussians.training_setup(OptimizationParams(argparse.ArgumentParser()).extract(
        argparse.Namespace(iterations=iterations)
    ))
    
    # LPIPS loss
    lpips_criterion = lpips.LPIPS(net='alex').eval().cuda()
    
    # Load checkpoint if provided
    first_iter = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        gaussians.restore(checkpoint['gaussians'], None)
        motion_net.load_state_dict(checkpoint['motion_net'])
        motion_optimizer.load_state_dict(checkpoint['motion_optimizer'])
        first_iter = checkpoint['iteration']
    
    # Background color
    bg_color = list(config.background_color)
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    # Training loop
    viewpoint_stack = None
    ema_loss = 0.0
    progress_bar = tqdm(range(first_iter, iterations), desc="Stage 2: Face Gaussians")
    
    for iteration in range(first_iter + 1, iterations + 1):
        gaussians.update_learning_rate(iteration)
        
        # SH degree increase
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()
        
        # Pick random camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        
        # Progressive mouth opening curriculum
        mouth_global_lb = viewpoint_cam.talking_dict.get('mouth_bound', [0, 0, 0.5])[0]
        mouth_global_ub = viewpoint_cam.talking_dict.get('mouth_bound', [0, 1, 0.5])[1]
        mouth_global_lb += (mouth_global_ub - mouth_global_lb) * 0.2
        mouth_window = (mouth_global_ub - mouth_global_lb) * 0.2
        
        mouth_lb = mouth_global_lb + mouth_step * iteration * (mouth_global_ub - mouth_global_lb)
        mouth_ub = mouth_lb + mouth_window
        mouth_lb = mouth_lb - mouth_window
        
        # Sample selection during early training
        if iteration < warmup_iter:
            select_interval = 15
            if iteration % select_interval == 0:
                max_tries = 100
                tries = 0
                while tries < max_tries:
                    mouth_val = viewpoint_cam.talking_dict.get('mouth_bound', [0, 1, 0.5])[2]
                    if mouth_lb <= mouth_val <= mouth_ub:
                        break
                    if not viewpoint_stack:
                        viewpoint_stack = scene.getTrainCameras().copy()
                    viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
                    tries += 1
        
        # Load image on the fly
        if viewpoint_cam.original_image is None:
            viewpoint_cam = loadCamOnTheFly(copy.deepcopy(viewpoint_cam))
        
        # Get masks
        face_mask = torch.as_tensor(viewpoint_cam.talking_dict["face_mask"]).cuda()
        hair_mask = torch.as_tensor(viewpoint_cam.talking_dict.get("hair_mask", torch.zeros_like(face_mask))).cuda()
        mouth_mask = torch.as_tensor(viewpoint_cam.talking_dict["mouth_mask"]).cuda()
        head_mask = face_mask | hair_mask
        
        # Erode mouth mask for LPIPS
        if iteration > lpips_start_iter:
            max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            mouth_mask = (-max_pool(-max_pool(mouth_mask[None].float())))[0].bool()
        
        # Render
        if iteration < warmup_iter:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        else:
            render_pkg = render_motion_blendshape(
                viewpoint_cam, gaussians, motion_net, pipe, background, return_attn=True
            )
        
        image_render = render_pkg["render"]
        alpha = render_pkg["alpha"]
        viewspace_points = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]
        
        # Ground truth
        gt_image = viewpoint_cam.original_image.cuda() / 255.0
        gt_image_masked = gt_image * head_mask + background[:, None, None] * ~head_mask
        
        # Freeze parameters in later iterations
        if iteration > motion_stop_iter:
            for param in motion_net.parameters():
                param.requires_grad = False
        
        # === Loss computation ===
        # Mask out mouth interior
        gt_image_masked[:, mouth_mask] = background[:, None]
        
        # Hair mask handling (train less frequently)
        hair_mask_iter = (warmup_iter < iteration < lpips_start_iter - 1000) and iteration % 7 != 0
        if hair_mask_iter:
            image_render[:, hair_mask] = background[:, None]
            gt_image_masked[:, hair_mask] = background[:, None]
        
        # L1 + SSIM loss
        loss_l1 = l1_loss(image_render, gt_image_masked)
        loss = loss_l1 + config.lambda_ssim * (1.0 - ssim(image_render, gt_image_masked))
        
        # Motion regularization
        if iteration > warmup_iter and 'motion' in render_pkg:
            motion = render_pkg['motion']
            loss += config.lambda_motion_reg * motion['d_xyz'].abs().mean()
            loss += config.lambda_motion_reg * motion['d_rot'].abs().mean()
            loss += config.lambda_motion_reg * motion['d_opa'].abs().mean()
            loss += config.lambda_motion_reg * motion['d_scale'].abs().mean()
            
            # Alpha regularization
            loss += config.lambda_alpha_reg * (
                ((1 - alpha) * head_mask.float()).mean() +
                (alpha * (~head_mask).float()).mean()
            )
            
            # Attention regularization for lips region
            if 'attn' in render_pkg:
                lips_rect = viewpoint_cam.talking_dict.get('lips_rect', None)
                if lips_rect is not None:
                    xmin, xmax, ymin, ymax = lips_rect
                    # Expression attention should be low in lip region
                    loss += 1e-4 * render_pkg["attn"][1, xmin:xmax, ymin:ymax].mean()
        
        # LPIPS loss
        if iteration > lpips_start_iter:
            lips_rect = viewpoint_cam.talking_dict.get('lips_rect', None)
            if lips_rect is not None:
                xmin, xmax, ymin, ymax = lips_rect
                loss += 0.01 * lpips_criterion(
                    image_render.clone()[:, xmin:xmax, ymin:ymax].unsqueeze(0) * 2 - 1,
                    gt_image_masked.clone()[:, xmin:xmax, ymin:ymax].unsqueeze(0) * 2 - 1
                ).mean()
            
            image_t = image_render.clone()
            gt_image_t = gt_image_masked.clone()
            
            if lips_rect is not None:
                image_t[:, xmin:xmax, ymin:ymax] = background[:, None, None]
                gt_image_t[:, xmin:xmax, ymin:ymax] = background[:, None, None]
            
            patch_size = random.randint(32, 48) * 2
            loss += 0.2 * lpips_criterion(
                patchify(image_t[None, ...] * 2 - 1, patch_size),
                patchify(gt_image_t[None, ...] * 2 - 1, patch_size)
            ).mean()
        
        # Lip opacity constraint
        if iteration > warmup_iter and config.lambda_lip_opacity > 0:
            lip_landmarks = viewpoint_cam.talking_dict.get('lip_landmarks_3d', None)
            if lip_landmarks is not None and 'motion' in render_pkg:
                lip_loss = compute_lip_opacity_loss(
                    gaussians,
                    render_pkg['motion'],
                    lip_landmarks.cuda(),
                    min_opacity=0.8,
                )
                loss += config.lambda_lip_opacity * lip_loss
        
        # Backward
        loss.backward()
        
        # Logging
        ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
        
        if iteration % 10 == 0:
            progress_bar.set_postfix({
                "Loss": f"{ema_loss:.5f}",
                "Points": gaussians.get_xyz.shape[0]
            })
            progress_bar.update(10)
        
        with torch.no_grad():
            # Tensorboard
            if tb_writer and iteration % 100 == 0:
                tb_writer.add_scalar('stage2/loss', loss.item(), iteration)
                tb_writer.add_scalar('stage2/num_points', gaussians.get_xyz.shape[0], iteration)
            
            # Validation
            if iteration in testing_iterations:
                validate_face_gaussian(
                    gaussians, motion_net, scene, render_motion_blendshape,
                    tb_writer, iteration, pipe, background
                )
            
            # Save
            if iteration in saving_iterations:
                print(f"\n[Stage 2 - Iter {iteration}] Saving Checkpoint")
                scene.save(str(iteration) + '_face')
                
                checkpoint = {
                    'gaussians': gaussians.capture(),
                    'motion_net': motion_net.state_dict(),
                    'motion_optimizer': motion_optimizer.state_dict(),
                    'iteration': iteration,
                }
                torch.save(checkpoint, os.path.join(scene.model_path, f"chkpnt_face_{iteration}.pth"))
                torch.save(checkpoint, os.path.join(scene.model_path, "chkpnt_face_latest.pth"))
            
            # Densification
            if iteration < densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_points, visibility_filter)
                
                if iteration > densify_from_iter and iteration % densification_interval == 0:
                    size_threshold = 20 if iteration > opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        0.0002,
                        0.05 + 0.25 * iteration / densify_until_iter,
                        scene.cameras_extent,
                        size_threshold
                    )
            
            # Prune background-colored Gaussians
            if iteration > densify_from_iter and iteration % densification_interval == 0:
                shs_view = gaussians.get_features.transpose(1, 2).view(-1, 3, (gaussians.max_sh_degree + 1) ** 2)
                dir_pp = gaussians.get_xyz - viewpoint_cam.camera_center.repeat(gaussians.get_features.shape[0], 1)
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(gaussians.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                
                bg_color_mask = (
                    (colors_precomp[..., 0] < 30/255) &
                    (colors_precomp[..., 1] > 225/255) &
                    (colors_precomp[..., 2] < 30/255)
                )
                gaussians.prune_points(bg_color_mask.squeeze())
            
            # Optimizer step
            if iteration < iterations:
                motion_optimizer.step()
                gaussians.optimizer.step()
                
                motion_optimizer.zero_grad()
                gaussians.optimizer.zero_grad(set_to_none=True)
                
                scheduler.step()
    
    progress_bar.close()
    print("\nStage 2: Face Gaussian Training Complete.")
    
    return gaussians, motion_net


def validate_face_gaussian(gaussians, motion_net, scene, render_func, tb_writer, iteration, pipe, background):
    """Validation for face Gaussians."""
    torch.cuda.empty_cache()
    
    val_cameras = scene.getTestCameras()[:5]
    
    for idx, viewpoint in enumerate(val_cameras):
        if viewpoint.original_image is None:
            viewpoint = loadCamOnTheFly(copy.deepcopy(viewpoint))
        
        with torch.no_grad():
            render_pkg = render_func(viewpoint, gaussians, motion_net, pipe, background, return_attn=True)
        
        image = torch.clamp(render_pkg["render"], 0.0, 1.0)
        alpha = render_pkg["alpha"]
        
        # Composite with background
        if hasattr(viewpoint, 'background') and viewpoint.background is not None:
            bg = viewpoint.background.cuda() / 255.0
            image = image * alpha + bg * (1.0 - alpha)
        
        gt_image = viewpoint.original_image.cuda() / 255.0
        
        if tb_writer:
            tb_writer.add_images(f"stage2_val/render_{idx}", image.unsqueeze(0), global_step=iteration)
            tb_writer.add_images(f"stage2_val/gt_{idx}", gt_image.unsqueeze(0), global_step=iteration)
            
            if 'attn' in render_pkg:
                attn = render_pkg['attn']
                tb_writer.add_images(
                    f"stage2_val/attn_aud_{idx}",
                    (attn[0] / (attn[0].max() + 1e-6)).unsqueeze(0).unsqueeze(0),
                    global_step=iteration
                )
                tb_writer.add_images(
                    f"stage2_val/attn_exp_{idx}",
                    (attn[1] / (attn[1].max() + 1e-6)).unsqueeze(0).unsqueeze(0),
                    global_step=iteration
                )
    
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
    parser = argparse.ArgumentParser(description="ST-Gauss Stage 2: Face Gaussian Training")
    
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument('--config', type=str, default='default')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--quiet', action='store_true')
    
    args = parser.parse_args()
    
    config = get_config(args.config)
    
    print(f"Training ST-Gauss Stage 2: Face Gaussians")
    
    safe_state(args.quiet)
    
    training_stage2(
        dataset=lp.extract(args),
        config=config,
        pipe=pp.extract(args),
        checkpoint_path=args.checkpoint,
    )
    
    print("\nStage 2 Training Complete.")

