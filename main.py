"""
SyncFace - Enhanced Audio-Driven Talking Head Generation

Main training and inference script with enhanced audio encoders and emotion support.

Usage:
    # Train
    python main.py data/Macron --workspace output/Macron_hybrid --use_enhanced_encoder --iters 100000 -O

    # Test
    python main.py data/Macron --workspace output/Macron_hybrid --test -O

    # GUI
    python main.py data/Macron --workspace output/Macron_hybrid --test --gui -O
"""

import argparse
import os
import torch
import numpy as np

from nerf_triplane.provider import NeRFDataset
from nerf_triplane.utils import *
from nerf_triplane.network import NeRFNetwork

# ==============================================================================
# CONFIGURATION SETUP
# ==============================================================================

def setup_torch_precision():
    """Configure PyTorch numerical precision settings."""
    # torch.autograd.set_detect_anomaly(True)  # Enable for debugging NaN gradients

    # Disable tf32 for better numerical accuracy on RTX30xx GPUs
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except AttributeError:
        print('[INFO] This PyTorch version does not support tf32 settings.')


def create_argument_parser():
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(description='SyncFace - Enhanced Talking Head Generation')

    # ==========================================================================
    # BASIC ARGUMENTS
    # ==========================================================================
    parser.add_argument('path', type=str, help='Path to dataset directory')
    parser.add_argument('-O', action='store_true',
                       help='Optimization preset: equals --fp16 --cuda_ray --exp_eye')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: load model and run inference')
    parser.add_argument('--test_train', action='store_true',
                       help='Test mode: load model and test on training dataset')
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1],
                       help='Data range to use [start, end]')
    parser.add_argument('--workspace', type=str, default='workspace',
                       help='Output workspace directory')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed for reproducibility')

    # ==========================================================================
    # TRAINING OPTIONS
    # ==========================================================================
    parser.add_argument('--iters', type=int, default=200000,
                       help='Total number of training iterations')
    parser.add_argument('--lr', type=float, default=1e-2,
                       help='Initial learning rate for geometry')
    parser.add_argument('--lr_net', type=float, default=1e-3,
                       help='Initial learning rate for network parameters')
    parser.add_argument('--ckpt', type=str, default='latest',
                       help='Checkpoint to load (latest, best, or specific file)')
    parser.add_argument('--num_rays', type=int, default=4096 * 16,
                       help='Number of rays sampled per training step')
    parser.add_argument('--cuda_ray', action='store_true',
                       help='Use CUDA raymarching instead of PyTorch')
    parser.add_argument('--max_steps', type=int, default=16,
                       help='Max steps sampled per ray (CUDA raymarching only)')
    parser.add_argument('--num_steps', type=int, default=16,
                       help='Number of steps sampled per ray (PyTorch raymarching only)')
    parser.add_argument('--upsample_steps', type=int, default=0,
                       help='Upsampling steps per ray (PyTorch raymarching only)')
    parser.add_argument('--update_extra_interval', type=int, default=16,
                       help='Iteration interval for updating extra status (CUDA only)')
    parser.add_argument('--max_ray_batch', type=int, default=4096,
                       help='Batch size of rays at inference (PyTorch only)')

    # ==========================================================================
    # LOSS OPTIONS
    # ==========================================================================
    parser.add_argument('--warmup_step', type=int, default=10000,
                       help='Number of warmup steps before applying full loss')
    parser.add_argument('--amb_aud_loss', type=int, default=1,
                       help='Use ambient audio loss (0=off, 1=on)')
    parser.add_argument('--amb_eye_loss', type=int, default=1,
                       help='Use ambient eye loss (0=off, 1=on)')
    parser.add_argument('--unc_loss', type=int, default=1,
                       help='Use uncertainty loss (0=off, 1=on)')
    parser.add_argument('--lambda_amb', type=float, default=1e-4,
                       help='Weight for ambient loss')
    parser.add_argument('--pyramid_loss', type=int, default=0,
                       help='Use perceptual pyramid loss (0=off, >0=on)')

    # ==========================================================================
    # NETWORK BACKBONE OPTIONS
    # ==========================================================================
    parser.add_argument('--fp16', action='store_true',
                       help='Use mixed precision training (FP16)')

    # ==========================================================================
    # FACE MODELING OPTIONS
    # ==========================================================================
    parser.add_argument('--bg_img', type=str, default='',
                       help='Background image path')
    parser.add_argument('--fbg', action='store_true',
                       help='Use frame-wise background')
    parser.add_argument('--exp_eye', action='store_true',
                       help='Explicitly control eye movements')
    parser.add_argument('--fix_eye', type=float, default=-1,
                       help='Fixed eye area (-1 to disable, 0-0.3 for reasonable eye)')
    parser.add_argument('--smooth_eye', action='store_true',
                       help='Smooth eye area sequence')
    parser.add_argument('--bs_area', type=str, default="upper",
                       help='Blendshape area to use (upper or eye)')
    parser.add_argument('--au45', action='store_true',
                       help='Use OpenFace AU45 blendshapes')
    parser.add_argument('--torso_shrink', type=float, default=0.8,
                       help='Shrink torso coordinates for deformation flexibility')

    # ==========================================================================
    # DATASET OPTIONS
    # ==========================================================================
    parser.add_argument('--color_space', type=str, default='srgb',
                       help='Color space (linear, srgb)')
    parser.add_argument('--preload', type=int, default=0,
                       help='Data loading mode: 0=on-demand, 1=CPU preload, 2=GPU preload')
    parser.add_argument('--bound', type=float, default=1,
                       help='Scene bound for box [-bound, bound]^3')
    parser.add_argument('--scale', type=float, default=4,
                       help='Camera location scale into bound box')
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0],
                       help='Camera location offset [x, y, z]')
    parser.add_argument('--dt_gamma', type=float, default=1/256,
                       help='Adaptive ray marching gamma (0=disable, >0=accelerate)')
    parser.add_argument('--min_near', type=float, default=0.05,
                       help='Minimum camera near distance')
    parser.add_argument('--density_thresh', type=float, default=10,
                       help='Density threshold for occupied grid cells')
    parser.add_argument('--density_thresh_torso', type=float, default=0.01,
                       help='Density threshold for torso occupied grid cells')
    parser.add_argument('--patch_size', type=int, default=1,
                       help='Patch size for LPIPS loss (1=disabled)')

    # ==========================================================================
    # LIP MODELING OPTIONS
    # ==========================================================================
    parser.add_argument('--init_lips', action='store_true',
                       help='Initialize lip region')
    parser.add_argument('--finetune_lips', action='store_true',
                       help='Fine-tune lips using LPIPS and landmarks')
    parser.add_argument('--smooth_lips', action='store_true',
                       help='Smooth audio features with exponential decay')

    # ==========================================================================
    # TORSO MODELING OPTIONS
    # ==========================================================================
    parser.add_argument('--torso', action='store_true',
                       help='Train torso (fix head and train torso)')
    parser.add_argument('--head_ckpt', type=str, default='',
                       help='Head model checkpoint path')

    # ==========================================================================
    # GUI OPTIONS
    # ==========================================================================
    parser.add_argument('--gui', action='store_true',
                       help='Launch interactive GUI for real-time control')
    parser.add_argument('--W', type=int, default=450,
                       help='GUI window width')
    parser.add_argument('--H', type=int, default=450,
                       help='GUI window height')
    parser.add_argument('--radius', type=float, default=3.35,
                       help='Default GUI camera radius from center')
    parser.add_argument('--fovy', type=float, default=21.24,
                       help='Default GUI camera field of view')
    parser.add_argument('--max_spp', type=int, default=1,
                       help='GUI maximum samples per pixel')

    # ==========================================================================
    # AUDIO PROCESSING OPTIONS
    # ==========================================================================
    parser.add_argument('--att', type=int, default=2,
                       help='Audio attention mode (0=off, 1=left-direction, 2=bi-direction)')
    parser.add_argument('--aud', type=str, default='',
                       help='Audio source path (empty uses default)')
    parser.add_argument('--emb', action='store_true',
                       help='Use audio class embedding instead of logits')
    parser.add_argument('--portrait', action='store_true',
                       help='Render only face (no background)')
    
    # ==========================================================================
    # DUAL AUDIO ENCODER OPTIONS
    # ==========================================================================
    parser.add_argument('--use_dual_encoder', action='store_true',
                       help='Use dual audio encoder (content + sync branches)')
    parser.add_argument('--dual_encoder_fusion', type=str, default='cross_attention',
                       choices=['concat', 'add', 'cross_attention', 'gated'],
                       help='Fusion mode for dual encoder')
    
    # ==========================================================================
    # SYNC LOSS OPTIONS
    # ==========================================================================
    parser.add_argument('--use_sync_loss', action='store_true',
                       help='Use multi-scale sync loss (SyncNet, LSE-C, LSE-D)')
    parser.add_argument('--syncnet_checkpoint', type=str, default='',
                       help='Path to pretrained SyncNet checkpoint')
    parser.add_argument('--sync_loss_weight', type=float, default=1.0,
                       help='Weight for sync loss')

    # ==========================================================================
    # INDIVIDUAL CODE OPTIONS
    # ==========================================================================
    parser.add_argument('--ind_dim', type=int, default=4,
                       help='Individual code dimension (0=disable)')
    parser.add_argument('--ind_num', type=int, default=20000,
                       help='Number of individual codes (> training dataset size)')
    parser.add_argument('--ind_dim_torso', type=int, default=8,
                       help='Individual code dimension for torso')

    # ==========================================================================
    # AMBIENT LIGHTING OPTIONS
    # ==========================================================================
    parser.add_argument('--amb_dim', type=int, default=2,
                       help='Ambient lighting dimension')

    # ==========================================================================
    # DATA SUBSAMPLING OPTIONS
    # ==========================================================================
    parser.add_argument('--part', action='store_true',
                       help='Use partial training data (1/10)')
    parser.add_argument('--part2', action='store_true',
                       help='Use partial training data (first 15 seconds)')

    # ==========================================================================
    # ENHANCED AUDIO ENCODER OPTIONS
    # ==========================================================================
    parser.add_argument('--use_enhanced_encoder', action='store_true',
                       help='Use enhanced audio encoder with foundation models')
    parser.add_argument('--enhanced_encoder_type', type=str, default='whisper',
                       choices=['whisper', 'speecht5', 'encodec', 'ensemble', 'hybrid'],
                       help='Type of enhanced encoder')
    parser.add_argument('--use_prosody', action='store_true',
                       help='Extract and use prosodic features (pitch, energy, rhythm)')
    parser.add_argument('--use_contrastive', action='store_true',
                       help='Use CLIP-like contrastive audio-video alignment')
    parser.add_argument('--freeze_audio_backbone', action='store_true',
                       help='Freeze pretrained foundation model weights')
    parser.add_argument('--foundation_model_type', type=str, default='whisper',
                       help='Foundation model type for hybrid mode')

    # ==========================================================================
    # EMOTION RECOGNITION OPTIONS
    # ==========================================================================
    parser.add_argument('--use_emotion', action='store_true',
                       help='Enable emotion-aware facial expressions')
    parser.add_argument('--emotion_model', type=str, default='wav2vec2',
                       choices=['wav2vec2', 'cnn', 'prosody'],
                       help='Emotion recognition model type')
    parser.add_argument('--emotion_checkpoint', type=str, default='',
                       help='Path to trained emotion model checkpoint')
    parser.add_argument('--emotion_dim', type=int, default=64,
                       help='Emotion embedding dimension')
    parser.add_argument('--emotion_strength', type=float, default=0.7,
                       help='Emotion influence strength (0-1)')
    parser.add_argument('--emotion_smoothing', type=str, default='ema',
                       choices=['ema', 'conv'],
                       help='Temporal smoothing method for emotions')
    parser.add_argument('--emotion_blend_mode', type=str, default='add',
                       choices=['add', 'multiply', 'replace'],
                       help='Emotion blendshape blend mode')

    # ==========================================================================
    # CAMERA OPTIMIZATION OPTIONS
    # ==========================================================================
    parser.add_argument('--train_camera', action='store_true',
                       help='Optimize camera pose during training')
    parser.add_argument('--smooth_path', action='store_true',
                       help='Smooth camera pose trajectory')
    parser.add_argument('--smooth_path_window', type=int, default=7,
                       help='Smoothing window size for camera path')

    # ==========================================================================
    # ASR (AUTOMATIC SPEECH RECOGNITION) OPTIONS
    # ==========================================================================
    parser.add_argument('--asr', action='store_true',
                       help='Load ASR for real-time applications')
    parser.add_argument('--asr_wav', type=str, default='',
                       help='WAV file path for ASR input')
    parser.add_argument('--asr_play', action='store_true',
                       help='Play audio output in ASR mode')
    parser.add_argument('--asr_model', type=str, default='deepspeech',
                       help='ASR model type')
    parser.add_argument('--asr_save_feats', action='store_true',
                       help='Save ASR features to file')

    # ==========================================================================
    # AUDIO PROCESSING PARAMETERS
    # ==========================================================================
    parser.add_argument('--fps', type=int, default=50,
                       help='Audio processing FPS')
    parser.add_argument('-l', type=int, default=10,
                       help='ASR sliding window left context (20ms units)')
    parser.add_argument('-m', type=int, default=50,
                       help='ASR sliding window middle context (20ms units)')
    parser.add_argument('-r', type=int, default=10,
                       help='ASR sliding window right context (20ms units)')

    return parser

# ==============================================================================
# MAIN EXECUTION LOGIC
# ==============================================================================

def setup_options(opt):
    """Apply option presets and validation."""
    # Apply -O preset (optimization)
    if opt.O:
        opt.fp16 = True
        opt.exp_eye = True

    # Apply test mode presets (disabled for now)
    if opt.test and False:
        opt.smooth_path = True
        opt.smooth_eye = True
        opt.smooth_lips = True

    # Force CUDA raymarching
    opt.cuda_ray = True

    # Validate patch size
    if opt.patch_size > 1:
        assert opt.patch_size ** 2 <= opt.num_rays, "patch_size ** 2 should be <= num_rays."


def load_model_checkpoint(model, opt):
    """Load model checkpoint for torso training."""
    if not (opt.torso and opt.head_ckpt):
        return

    print(f"[INFO] Loading head checkpoint: {opt.head_ckpt}")

    try:
        checkpoint = torch.load(opt.head_ckpt, map_location='cpu')

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            model_dict = checkpoint['model']
        else:
            model_dict = checkpoint

        missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)

        if missing_keys:
            print(f"[WARN] Missing keys: {missing_keys}")
        if unexpected_keys:
            print(f"[WARN] Unexpected keys: {unexpected_keys}")

        # Freeze loaded parameters
        for name, param in model.named_parameters():
            if name in model_dict:
                param.requires_grad = False
                print(f"[INFO] Froze parameter: {name} {param.shape}")

    except Exception as e:
        print(f"[ERROR] Failed to load checkpoint: {e}")
        raise


def create_model_and_criterion(opt):
    """Create model and loss criterion."""
    # Extract only the necessary parameters for NeRFNetwork
    model = NeRFNetwork(
        # Core network parameters
        emb=getattr(opt, 'emb', False),
        asr_model=getattr(opt, 'asr_model', 'deepspeech'),
        att=getattr(opt, 'att', 2),

        # Face modeling parameters
        au45=getattr(opt, 'au45', False),
        bs_area=getattr(opt, 'bs_area', "upper"),
        exp_eye=getattr(opt, 'exp_eye', False),
        individual_dim=getattr(opt, 'ind_dim', 4),
        individual_dim_torso=getattr(opt, 'ind_dim_torso', 8),

        # Training parameters
        torso=getattr(opt, 'torso', False),
        train_camera=getattr(opt, 'train_camera', False),
        unc_loss=getattr(opt, 'unc_loss', 1),

        # Enhanced audio encoder parameters
        use_enhanced_encoder=getattr(opt, 'use_enhanced_encoder', False),
        enhanced_encoder_type=getattr(opt, 'enhanced_encoder_type', 'whisper'),
        use_prosody=getattr(opt, 'use_prosody', True),
        foundation_model_type=getattr(opt, 'foundation_model_type', 'whisper'),
        use_contrastive=getattr(opt, 'use_contrastive', False),
        freeze_audio_backbone=getattr(opt, 'freeze_audio_backbone', True),

        # Dual audio encoder parameters
        use_dual_encoder=getattr(opt, 'use_dual_encoder', False),
        dual_encoder_fusion=getattr(opt, 'dual_encoder_fusion', 'cross_attention'),
        
        # Sync loss parameters
        use_sync_loss=getattr(opt, 'use_sync_loss', False),
        syncnet_checkpoint=getattr(opt, 'syncnet_checkpoint', ''),
        sync_loss_weight=getattr(opt, 'sync_loss_weight', 1.0),

        # Emotion recognition parameters
        use_emotion=getattr(opt, 'use_emotion', False),
        emotion_model=getattr(opt, 'emotion_model', 'wav2vec2'),
        emotion_checkpoint=getattr(opt, 'emotion_checkpoint', ''),
        emotion_strength=getattr(opt, 'emotion_strength', 0.7),
    )

    # Update model with scene parameters from opt
    model.bound = getattr(opt, 'bound', 1.0)
    model.min_near = getattr(opt, 'min_near', 0.05)
    model.density_thresh = getattr(opt, 'density_thresh', 10.0)
    model.density_thresh_torso = getattr(opt, 'density_thresh_torso', 0.01)
    model.cuda_ray = getattr(opt, 'cuda_ray', True)

    # Load checkpoint if torso training
    load_model_checkpoint(model, opt)

    # Use L1 loss (more robust than MSE for images)
    criterion = torch.nn.L1Loss(reduction='none')

    return model, criterion


def run_test_mode(opt, model, criterion, device):
    """Run inference/test mode."""
    print("=" * 60)
    print("TEST MODE")
    print("=" * 60)

    # Setup metrics
    if opt.gui:
        metrics = []  # Disable metrics for faster GUI initialization
    else:
        metrics = [PSNRMeter(), LPIPSMeter(device=device), LMDMeter(backend='fan')]

    # Create trainer
    trainer = Trainer('ngp', opt, model, device=device,
                     workspace=opt.workspace, criterion=criterion,
                     fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)

    # Load audio features for test dataset
    if opt.test_train:
        # Test on training dataset
        test_set = NeRFDataset(opt, device=device, type='train')
        test_set.training = False
        test_set.num_rays = -1
        test_loader = test_set.dataloader()
    else:
        # Test on test dataset
        test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

    # Update model with audio features
    model.aud_features = test_loader._data.auds
    model.eye_areas = test_loader._data.eye_area

    if opt.gui:
        # Launch interactive GUI
        from nerf_triplane.gui import NeRFGUI
        with NeRFGUI(opt, trainer, test_loader) as gui:
            gui.render()
    else:
        # Run inference and save video
        trainer.test(test_loader)

        # Run evaluation if ground truth available
        if test_loader.has_gt:
            trainer.evaluate(test_loader)


def run_training_mode(opt, model, criterion, device):
    """Run training mode."""
    print("=" * 60)
    print("TRAINING MODE")
    print("=" * 60)

    # Create optimizer
    optimizer_func = lambda model: torch.optim.AdamW(
        model.get_params(opt.lr, opt.lr_net),
        betas=(0, 0.99),
        eps=1e-8
    )

    # Create dataset and dataloader
    train_loader = NeRFDataset(opt, device=device, type='train').dataloader()

    # Validate dataset size
    assert len(train_loader) < opt.ind_num, (
        f"[ERROR] Dataset too large: {len(train_loader)} frames, "
        f"increase --ind_num to at least this value!"
    )

    # Update model with training data
    model.aud_features = train_loader._data.auds
    model.eye_area = train_loader._data.eye_area
    model.poses = train_loader._data.poses

    # Setup learning rate scheduler
    if opt.finetune_lips:
        scheduler_func = lambda optimizer: optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.05 ** (iter / opt.iters)
        )
    else:
        scheduler_func = lambda optimizer: optim.lr_scheduler.LambdaLR(
            optimizer, lambda iter: 0.5 ** (iter / opt.iters)
        )

    # Setup metrics and evaluation
    metrics = [PSNRMeter(), LPIPSMeter(device=device), LMDMeter(backend='fan')]
    eval_interval = max(1, int(5000 / len(train_loader)))

    # Create trainer
    trainer = Trainer(
        'ngp', opt, model, device=device,
        workspace=opt.workspace, optimizer=optimizer_func,
        criterion=criterion, ema_decay=0.95, fp16=opt.fp16,
        lr_scheduler=scheduler_func, scheduler_update_every_step=True,
        metrics=metrics, use_checkpoint=opt.ckpt,
        eval_interval=eval_interval
    )

    # Save options to workspace
    with open(os.path.join(opt.workspace, 'opt.txt'), 'w') as f:
        f.write(str(opt))

    if opt.gui:
        # Launch training GUI
        from nerf_triplane.gui import NeRFGUI
        with NeRFGUI(opt, trainer, train_loader) as gui:
            gui.render()
    else:
        # Run training
        valid_loader = NeRFDataset(opt, device=device, type='val', downscale=1).dataloader()
        max_epochs = np.ceil(opt.iters / len(train_loader)).astype(np.int32)

        print(f"[INFO] Training for {max_epochs} epochs ({opt.iters} total iterations)")

        trainer.train(train_loader, valid_loader, max_epochs)

        # Cleanup memory
        del train_loader, valid_loader
        torch.cuda.empty_cache()

        # Run final test
        test_loader = NeRFDataset(opt, device=device, type='test').dataloader()

        if test_loader.has_gt:
            trainer.evaluate(test_loader)

        trainer.test(test_loader)


def main():
    """Main entry point."""
    # Setup PyTorch precision
    setup_torch_precision()

    # Parse arguments
    parser = create_argument_parser()
    opt = parser.parse_args()

    # Apply option presets and validation
    setup_options(opt)

    # Print configuration
    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(opt)
    print("=" * 60 + "\n")

    # Set random seed
    seed_everything(opt.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    # Create model and criterion
    model, criterion = create_model_and_criterion(opt)

    # Run appropriate mode
    if opt.test:
        run_test_mode(opt, model, criterion, device)
    else:
        run_training_mode(opt, model, criterion, device)


if __name__ == '__main__':
    main()
