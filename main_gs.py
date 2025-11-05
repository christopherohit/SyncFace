"""
Main training script for SyncTalk with 3D Gaussian Splatting
This replaces the original NeRF-based main.py with 3DGS components
"""

import argparse
import os
import torch
import numpy as np
from nerf_triplane.provider_gs import create_gaussian_data_loader
from nerf_triplane.trainer_gs import GaussianTrainer
from nerf_triplane.network_gs import DynamicGaussianNetwork

# Disable TF32 for better numerical accuracy
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except AttributeError:
    print('[INFO] This PyTorch version does not support TF32 settings.')

def configure_loss_weights(args):
    """
    Configure optimal loss weights for different training stages
    """
    if args.stage == 'canonical':
        # Canonical stage: Focus on geometric reconstruction
        if not hasattr(args, 'gs_lambda_l1') or getattr(args, 'gs_lambda_l1', 1.0) == 1.0:  # Only set if not explicitly specified
            args.gs_lambda_l1 = 1.0      # Primary reconstruction
            args.gs_lambda_l2 = 0.1      # Secondary L2 for stability
            args.gs_lambda_ssim = 0.2    # Perceptual quality
            args.gs_lambda_lpips = 0.05  # High-level features
            args.gs_lambda_temporal = 0.0  # No temporal for static
            args.gs_lambda_smoothness = 0.001  # Minor regularization

    elif args.stage == 'deform':
        # Deformation stage: Focus on motion and temporal consistency
        if not hasattr(args, 'gs_lambda_l1') or getattr(args, 'gs_lambda_l1', 1.0) == 1.0:  # Only set if not explicitly specified
            args.gs_lambda_l1 = 1.0      # Primary reconstruction
            args.gs_lambda_l2 = 0.0      # Less L2 for motion
            args.gs_lambda_ssim = 0.3    # Higher perceptual weight
            args.gs_lambda_lpips = 0.1   # More perceptual for motion
            args.gs_lambda_temporal = 0.05  # Temporal consistency
            args.gs_lambda_smoothness = 0.01  # Motion smoothness

    elif args.stage in ['finetune', 'all']:
        # Fine-tuning: Balanced approach with all losses
        if not hasattr(args, 'gs_lambda_l1') or getattr(args, 'gs_lambda_l1', 1.0) == 1.0:  # Only set if not explicitly specified
            args.gs_lambda_l1 = 0.8      # Slightly reduced
            args.gs_lambda_l2 = 0.0      # Minimal L2
            args.gs_lambda_ssim = 0.4    # Higher SSIM for quality
            args.gs_lambda_lpips = 0.1   # Perceptual quality
            args.gs_lambda_temporal = 0.1  # Good temporal consistency
            args.gs_lambda_smoothness = 0.02  # Motion regularization

    print(f'[INFO] Loss weights for {args.stage} stage:')
    print(f'  L1: {getattr(args, "gs_lambda_l1", "N/A")}, L2: {getattr(args, "gs_lambda_l2", "N/A")}, SSIM: {getattr(args, "gs_lambda_ssim", "N/A")}')
    print(f'  LPIPS: {getattr(args, "gs_lambda_lpips", "N/A")}, Temporal: {getattr(args, "gs_lambda_temporal", "N/A")}, Smoothness: {getattr(args, "gs_lambda_smoothness", "N/A")}')

def create_argument_parser():
    """Create argument parser with all original SyncTalk arguments plus 3DGS ones"""
    parser = argparse.ArgumentParser(description='SyncTalk with 3D Gaussian Splatting')

    # Basic options
    parser.add_argument('path', type=str, help='path to dataset')
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --exp_eye")
    parser.add_argument('--test', action='store_true', help="test mode (load model and test dataset)")
    parser.add_argument('--test_train', action='store_true', help="test mode (load model and train dataset)")
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1], help="data range to use")
    parser.add_argument('--workspace', type=str, default='workspace_gs')
    parser.add_argument('--seed', type=int, default=0)

    # Training options
    parser.add_argument('--iters', type=int, default=200000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--lr_net', type=float, default=1e-3, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=-1, help="num rays sampled per image for each training step (-1 for full images)")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=16, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=16, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")

    # Loss options
    parser.add_argument('--warmup_step', type=int, default=10000, help="warm up steps")
    parser.add_argument('--amb_aud_loss', type=int, default=1, help="use ambient aud loss")
    parser.add_argument('--amb_eye_loss', type=int, default=1, help="use ambient eye loss")
    parser.add_argument('--unc_loss', type=int, default=1, help="use uncertainty loss")
    parser.add_argument('--lambda_amb', type=float, default=1e-4, help="lambda for ambient loss")
    parser.add_argument('--pyramid_loss', type=int, default=0, help="use perceptual loss")

    # Network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--bg_img', type=str, default='', help="background image")
    parser.add_argument('--fbg', action='store_true', help="frame-wise bg")
    parser.add_argument('--exp_eye', action='store_true', help="explicitly control the eyes")
    parser.add_argument('--fix_eye', type=float, default=-1, help="fixed eye area, negative to disable, set to 0-0.3 for a reasonable eye")
    parser.add_argument('--smooth_eye', action='store_true', help="smooth the eye area sequence")
    parser.add_argument('--bs_area', type=str, default="upper", help="upper or eye")
    parser.add_argument('--au45', action='store_true', help="use openface au45")
    parser.add_argument('--torso_shrink', type=float, default=0.8, help="shrink bg coords to allow more flexibility in deform")

    # Dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', type=int, default=0, help="0 means load data from disk on-the-fly, 1 means preload to CPU, 2 means GPU.")
    parser.add_argument('--bound', type=float, default=1, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=4, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/256, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.05, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied (sigma)")
    parser.add_argument('--density_thresh_torso', type=float, default=0.01, help="threshold for density grid to be occupied (alpha)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    parser.add_argument('--init_lips', action='store_true', help="init lips region")
    parser.add_argument('--finetune_lips', action='store_true', help="use LPIPS and landmarks to fine tune lips region")
    parser.add_argument('--smooth_lips', action='store_true', help="smooth the enc_a in a exponential decay way...")

    parser.add_argument('--torso', action='store_true', help="fix head and train torso")
    parser.add_argument('--head_ckpt', type=str, default='', help="head model")

    # GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=450, help="GUI width")
    parser.add_argument('--H', type=int, default=450, help="GUI height")
    parser.add_argument('--radius', type=float, default=3.35, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=21.24, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=1, help="GUI rendering max sample per pixel")

    # Audio options
    parser.add_argument('--att', type=int, default=2, help="audio attention mode (0 = turn off, 1 = left-direction, 2 = bi-direction)")
    parser.add_argument('--aud', type=str, default='', help="audio source (empty will load the default, else should be a path to a npy file)")
    parser.add_argument('--emb', action='store_true', help="use audio class + embedding instead of logits")
    parser.add_argument('--portrait', action='store_true', help="only render face")
    parser.add_argument('--ind_dim', type=int, default=4, help="individual code dim, 0 to turn off")
    parser.add_argument('--ind_num', type=int, default=20000, help="number of individual codes, should be larger than training dataset size")

    parser.add_argument('--ind_dim_torso', type=int, default=8, help="individual code dim, 0 to turn off")
    parser.add_argument('--amb_dim', type=int, default=2, help="ambient dimension")
    parser.add_argument('--part', action='store_true', help="use partial training data (1/10)")
    parser.add_argument('--part2', action='store_true', help="use partial training data (first 15s)")

    # Enhanced Audio Encoder options
    parser.add_argument('--use_enhanced_encoder', action='store_true', help="use enhanced audio encoder with foundation models")
    parser.add_argument('--enhanced_encoder_type', type=str, default='whisper', choices=['whisper', 'speecht5', 'encodec', 'ensemble', 'hybrid'], help="type of enhanced encoder")
    parser.add_argument('--use_prosody', action='store_true', help="extract and use prosodic features (pitch, energy, rhythm)")
    parser.add_argument('--use_contrastive', action='store_true', help="use CLIP-like contrastive audio-video alignment")
    parser.add_argument('--freeze_audio_backbone', action='store_true', help="freeze pretrained foundation model weights")
    parser.add_argument('--foundation_model_type', type=str, default='whisper', help="foundation model type for hybrid mode")

    # Emotion Recognition options
    parser.add_argument('--use_emotion', action='store_true', help="enable emotion-aware facial expressions")
    parser.add_argument('--emotion_model', type=str, default='wav2vec2', choices=['wav2vec2', 'cnn', 'prosody'], help="emotion recognition model type")
    parser.add_argument('--emotion_checkpoint', type=str, default='', help="path to trained emotion model checkpoint")
    parser.add_argument('--emotion_dim', type=int, default=64, help="emotion embedding dimension")
    parser.add_argument('--emotion_strength', type=float, default=0.7, help="emotion influence strength (0-1)")
    parser.add_argument('--emotion_smoothing', type=str, default='ema', choices=['ema', 'conv'], help="temporal smoothing method")
    parser.add_argument('--emotion_blend_mode', type=str, default='add', choices=['add', 'multiply', 'replace'], help="emotion blendshape blend mode")

    parser.add_argument('--train_camera', action='store_true', help="optimize camera pose")
    parser.add_argument('--smooth_path', action='store_true', help="brute-force smooth camera pose trajectory with a window size")
    parser.add_argument('--smooth_path_window', type=int, default=7, help="smoothing window size")

    # ASR options
    parser.add_argument('--asr', action='store_true', help="load asr for real-time app")
    parser.add_argument('--asr_wav', type=str, default='', help="load the wav and use as input")
    parser.add_argument('--asr_play', action='store_true', help="play out the audio")
    parser.add_argument('--asr_model', type=str, default='deepspeech')
    parser.add_argument('--asr_save_feats', action='store_true')
    parser.add_argument('--fps', type=int, default=50)

    # 3DGS specific options
    parser.add_argument('--sh_degree', type=int, default=0, help='spherical harmonics degree (0 for DC only)')
    parser.add_argument('--canonical_ply', type=str, default='', help='path to pre-trained canonical Gaussians PLY file')
    parser.add_argument('--lambda_ssim', type=float, default=0.2, help='weight for SSIM loss')
    parser.add_argument('--init_num_gaussians', type=int, default=10000, help='initial number of Gaussians')

    # Training options
    parser.add_argument('--lr_gaussian', type=float, default=0.0, help='learning rate for Gaussian parameters (0 to freeze)')
    parser.add_argument('--lr_decay_steps', type=int, default=10000, help='lr decay steps')
    parser.add_argument('--lr_decay_gamma', type=float, default=0.5, help='lr decay gamma')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size (usually 1 for 3DGS)')
    parser.add_argument('--downscale', type=int, default=1, help='image downscale factor')

    # Loss options
    parser.add_argument('--gs_lambda_l1', type=float, default=1.0, help='weight for L1 reconstruction loss')
    parser.add_argument('--gs_lambda_l2', type=float, default=0.0, help='weight for L2 reconstruction loss')
    parser.add_argument('--gs_lambda_ssim', type=float, default=0.2, help='weight for SSIM loss')
    parser.add_argument('--gs_lambda_lpips', type=float, default=0.0, help='weight for LPIPS perceptual loss')
    parser.add_argument('--gs_lambda_temporal', type=float, default=0.0, help='weight for temporal consistency loss')
    parser.add_argument('--gs_lambda_smoothness', type=float, default=0.0, help='weight for deformation smoothness regularization')
    parser.add_argument('--max_grad_norm', type=float, default=0.1, help='maximum gradient norm for clipping')

    # Model options
    parser.add_argument('--eye_dim', type=int, default=6, help='eye/expression feature dimension')
    parser.add_argument('--audio_dim', type=int, default=32, help='audio feature dimension')

    # Training stages
    parser.add_argument('--stage', type=str, default='all', choices=['canonical', 'deform', 'finetune', 'all'],
                       help='training stage: canonical (static head), deform (deformation only), finetune (all), all (full pipeline)')
    parser.add_argument('--canonical_iters', type=int, default=30000, help='iterations for canonical training')
    parser.add_argument('--deform_iters', type=int, default=20000, help='iterations for deformation training')
    parser.add_argument('--gs_init_num_gaussians', type=int, default=50000, help='number of Gaussians to initialize for canonical training')

    # Evaluation options
    parser.add_argument('--eval_interval', type=int, default=1, help='evaluation interval (epochs)')
    parser.add_argument('--save_interval', type=int, default=1000, help='checkpoint save interval (steps)')

    return parser

def main():
    parser = create_argument_parser()

    # Parse arguments
    args = parser.parse_args()


    # Apply optimization presets
    if args.O:
        args.fp16 = True
        args.exp_eye = True

    # Auto-configure loss weights based on training stage
    configure_loss_weights(args)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    print(f'[INFO] Using device: {device}')
    print(f'[INFO] Workspace: {args.workspace}')
    print(f'[INFO] Training stage: {args.stage}')
    
    # Create model
    print('[INFO] Creating DynamicGaussianNetwork...')
    model = DynamicGaussianNetwork(args)
    
    # Load or initialize Gaussians
    if args.canonical_ply and os.path.exists(args.canonical_ply):
        print(f'[INFO] Loading canonical Gaussians from {args.canonical_ply}')
        model.load_canonical_gaussians(args.canonical_ply)
    elif args.stage == 'canonical':
        print(f'[INFO] Initializing random Gaussians for canonical training')
        init_num = getattr(args, 'gs_init_num_gaussians', 50000)
        model.initialize_gaussians(num_gaussians=init_num)
    elif args.stage in ['deform', 'finetune']:
        print('[ERROR] Deformation/finetune stage requires canonical PLY file!')
        print('[HINT] First run with --stage canonical to create the canonical model')
        return
    
    # Create data loaders
    print('[INFO] Loading dataset...')
    
    if args.test or args.test_train:
        # Test mode
        if args.test_train:
            test_loader = create_gaussian_data_loader(args, device, type='train', downscale=args.downscale)
        else:
            test_loader = create_gaussian_data_loader(args, device, type='test', downscale=args.downscale)
        
        # Create trainer for testing
        trainer = GaussianTrainer(
            name=args.workspace,
            opt=args,
            model=model,
            device=device,
            workspace=args.workspace,
            fp16=args.fp16,
            use_checkpoint=args.ckpt
        )
        
        # Run test
        test_save_path = os.path.join(args.workspace, 'results')
        trainer.test(test_loader, test_save_path, write_video=True)
        
    else:
        # Training mode
        train_loader = create_gaussian_data_loader(args, device, type='train', downscale=args.downscale)
        val_loader = create_gaussian_data_loader(args, device, type='val', downscale=args.downscale)
        
        # Configure optimizer based on training stage
        def get_optimizer(model):
            lr_dict = {}
            
            if args.stage == 'canonical':
                # Only train Gaussians for canonical model - use higher learning rate
                lr_dict['gaussians'] = 0.01  # Higher learning rate for Gaussians
                lr_dict['audio_net'] = 0.0
                lr_dict['exp_net'] = 0.0
                lr_dict['deform_net'] = 0.0
            elif args.stage == 'deform':
                # Only train deformation network
                lr_dict['gaussians'] = 0.0
                lr_dict['audio_net'] = args.lr
                lr_dict['exp_net'] = args.lr
                lr_dict['deform_net'] = args.lr_net
            elif args.stage in ['finetune', 'all']:
                # Train everything - use much higher learning rates for training from scratch
                if args.stage == 'all':
                    lr_dict['gaussians'] = 0.01  # Much higher for Gaussians
                    lr_dict['audio_net'] = 0.001  # Higher for audio network
                    lr_dict['exp_net'] = 0.001  # Higher for expression network
                    lr_dict['deform_net'] = 0.005  # Higher for deformation network
                else:
                    lr_dict['gaussians'] = args.lr_gaussian
                    lr_dict['audio_net'] = args.lr
                    lr_dict['exp_net'] = args.lr
                    lr_dict['deform_net'] = args.lr_net
            
            params = model.get_params(lr_dict)
            return torch.optim.Adam(params, betas=(0.9, 0.999))
        
        # Create trainer
        trainer = GaussianTrainer(
            name=f"{args.workspace}_{args.stage}",
            opt=args,
            model=model,
            optimizer=get_optimizer,
            device=device,
            workspace=args.workspace,
            fp16=args.fp16,
            eval_interval=args.eval_interval,
            use_checkpoint=args.ckpt if args.stage != 'canonical' else None,
            scheduler_update_every_step=True
        )
        
        # Determine number of iterations based on stage
        if args.stage == 'canonical':
            max_iters = args.canonical_iters
        elif args.stage == 'deform':
            max_iters = args.deform_iters
        else:
            max_iters = args.iters
        
        # Convert iterations to epochs (approximate)
        train_size = len(train_loader)
        max_epochs = max(1, max_iters // train_size)
        
        print(f'[INFO] Training for {max_epochs} epochs ({max_iters} iterations)')
        
        # Train model
        trainer.train(
            train_loader=train_loader,
            valid_loader=val_loader,
            max_epochs=max_epochs
        )
        
        # Save final checkpoint
        if args.stage == 'canonical':
            # Save canonical Gaussians
            canonical_path = os.path.join(args.workspace, 'canonical.ply')
            model.save_canonical_gaussians(canonical_path)
            print(f'[INFO] Saved canonical Gaussians to {canonical_path}')
            
        # Test on validation set
        print('[INFO] Running final evaluation...')
        test_save_path = os.path.join(args.workspace, f'results_{args.stage}')
        trainer.test(val_loader, test_save_path, write_video=True)
        
    print('[INFO] Training complete!')
    
    # Print instructions for next stage
    if args.stage == 'canonical':
        print('\n[NEXT STEP] Train deformation network:')
        print(f'python main_gs.py {args.path} --stage deform --canonical_ply {os.path.join(args.workspace, "canonical.ply")} --workspace {args.workspace}_deform')
    elif args.stage == 'deform':
        print('\n[NEXT STEP] Fine-tune all parameters:')
        print(f'python main_gs.py {args.path} --stage finetune --canonical_ply {args.canonical_ply} --ckpt {os.path.join(args.workspace, "checkpoints", args.workspace + "_deform", "latest.pth")} --workspace {args.workspace}_finetune')


if __name__ == '__main__':
    main()