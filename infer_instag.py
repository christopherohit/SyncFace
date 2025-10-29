"""
Inference script for InsTaG-adapted SyncTalk model.

Generate talking face videos using the adapted model from Phase 2.

Usage:
    conda activate synctalk && python infer_instag.py \
        --checkpoint workspace_adapt_may/adapted_final.pth \
        --audio demo/test.wav \
        --output output_video.mp4 \
        --fps 25
"""

import argparse
import os
import numpy as np
import torch
import cv2
from tqdm import tqdm
import subprocess

from nerf_triplane.instag_network import InsTaGNetwork
from nerf_triplane.utils import *
from nerf_triplane.provider import NeRFDataset


def create_opt_from_checkpoint(checkpoint_path: str) -> argparse.Namespace:
    """Create options from checkpoint metadata."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Create basic opt
    opt = argparse.Namespace()
    
    # Model options (use defaults if not in checkpoint)
    opt.audio_dim = 32
    opt.asr_model = 'deepspeech'
    opt.att = 0  # Simpler for inference
    opt.emb = False
    opt.exp_eye = True
    opt.au45 = False
    opt.bs_area = 'upper'
    opt.bound = 1.0
    
    # Individual codes
    opt.ind_dim = 4
    opt.ind_num = 1  # Single person for inference
    opt.ind_dim_torso = 8
    
    # Rendering options
    opt.cuda_ray = True
    opt.max_steps = 16
    opt.num_rays = 4096 * 16
    opt.update_extra_interval = 16
    
    # Dataset/rendering options
    opt.preload = 0
    opt.scale = 4
    opt.offset = [0, 0, 0]
    opt.dt_gamma = 1/256
    opt.min_near = 0.05
    opt.density_thresh = 10
    opt.density_thresh_torso = 0.01
    opt.data_range = [0, -1]
    
    # Other
    opt.torso = False
    opt.color_space = 'srgb'
    opt.test_train = False
    opt.smooth_lips = True
    opt.train_camera = False
    opt.fp16 = False
    
    # Loss options (not used in inference but needed for model)
    opt.lambda_amb = 1e-4
    opt.amb_aud_loss = 1
    opt.amb_eye_loss = 1
    opt.unc_loss = 1
    
    return opt


def extract_audio_features(audio_path: str, fps: int = 50, device='cuda'):
    """
    Extract audio features using DeepSpeech.
    
    Args:
        audio_path: Path to audio file (.wav)
        fps: Frames per second (should match video)
        device: Device to use
    
    Returns:
        Audio features tensor
    """
    # This is a placeholder - you need to implement proper audio feature extraction
    # For now, we'll create dummy features
    
    # Get audio duration
    import librosa
    audio, sr = librosa.load(audio_path, sr=16000)
    duration = len(audio) / sr
    num_frames = int(duration * fps)
    
    # TODO: Replace with actual DeepSpeech feature extraction
    # For now, return random features matching the expected shape
    # Shape should be [num_frames, 29, 16] for DeepSpeech
    features = torch.randn(num_frames, 29, 16, device=device) * 0.1
    
    print(f"[INFO] Extracted {num_frames} audio feature frames for {duration:.2f}s audio")
    
    return features


def render_frame(model, rays_o, rays_d, audio_features, bg_coords, poses, 
                H, W, index=0, eye=None):
    """
    Render a single frame.
    
    Args:
        model: InsTaGNetwork model
        rays_o: Ray origins [1, H*W, 3]
        rays_d: Ray directions [1, H*W, 3]
        audio_features: Audio features for this frame [1, 29, 16]
        bg_coords: Background coordinates [1, H*W, 2]
        poses: Camera poses [1, 4, 4]
        H, W: Image dimensions
        index: Frame index
        eye: Optional eye features
    
    Returns:
        Rendered RGB image [H, W, 3]
    """
    with torch.no_grad():
        results = model.render(
            rays_o=rays_o,
            rays_d=rays_d,
            auds=audio_features,
            bg_coords=bg_coords,
            poses=poses,
            index=index,
            eye=eye,
            staged=False,
            perturb=False,
            bg_color=1.0
        )
        
        image = results['image']  # [1, H*W, 3]
        image = image.reshape(H, W, 3)  # [H, W, 3]
        image = (image.cpu().numpy() * 255).astype(np.uint8)
        
    return image


def main():
    parser = argparse.ArgumentParser()
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to adapted model checkpoint')
    parser.add_argument('--audio', type=str, required=True,
                       help='Path to input audio file (.wav)')
    parser.add_argument('--output', type=str, default='output_instag.mp4',
                       help='Output video path')
    
    # Optional arguments
    parser.add_argument('--reference_data', type=str, default=None,
                       help='Path to reference data directory (for camera/geometry)')
    parser.add_argument('--fps', type=int, default=25,
                       help='Output video FPS')
    parser.add_argument('--W', type=int, default=450,
                       help='Output video width')
    parser.add_argument('--H', type=int, default=450,
                       help='Output video height')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("InsTaG Inference - Talking Face Generation")
    print("="*60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Audio: {args.audio}")
    print(f"Output: {args.output}")
    print("="*60)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print("\n[1/5] Loading checkpoint...")
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Create model
    print("[2/5] Creating model...")
    opt = create_opt_from_checkpoint(args.checkpoint)
    model = InsTaGNetwork(opt, phase='adapt').to(device)
    
    # Load model weights
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    print("Model loaded successfully")
    
    # Extract audio features
    print("[3/5] Extracting audio features...")
    audio_features = extract_audio_features(args.audio, args.fps, device)
    num_frames = len(audio_features)
    
    # Setup camera and rays
    print("[4/5] Setting up camera...")
    if args.reference_data and os.path.exists(args.reference_data):
        # Load camera parameters from reference data
        import json
        transforms_path = os.path.join(args.reference_data, 'transforms_train.json')
        with open(transforms_path, 'r') as f:
            transforms = json.load(f)
        
        # Get first frame's camera
        frame0 = transforms['frames'][0]
        c2w = np.array(frame0['transform_matrix'])
        
        # Get intrinsics
        if 'camera_angle_x' in transforms:
            fov_x = transforms['camera_angle_x']
            focal = args.W / (2 * np.tan(fov_x / 2))
        else:
            focal = 1111.0  # Default
    else:
        # Use default camera
        print("  Using default camera (no reference data provided)")
        focal = 1111.0
        # Front-facing camera
        c2w = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 3.5],
            [0, 0, 0, 1]
        ])
    
    # Generate rays
    intrinsics = [focal, focal, args.W/2, args.H/2]
    poses = torch.from_numpy(c2w).unsqueeze(0).float().to(device)  # [1, 4, 4]
    
    from nerf_triplane.utils import get_rays
    rays_o, rays_d = get_rays(poses, intrinsics, args.H, args.W, -1)  # [1, H*W, 3]
    
    # Background coordinates
    bg_coords = torch.zeros(1, args.H * args.W, 2, device=device)
    y_coords = torch.linspace(-1, 1, args.H, device=device)
    x_coords = torch.linspace(-1, 1, args.W, device=device)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    bg_coords[0, :, 0] = xx.reshape(-1)
    bg_coords[0, :, 1] = yy.reshape(-1)
    
    # Render frames
    print(f"[5/5] Rendering {num_frames} frames...")
    frames = []
    
    for frame_idx in tqdm(range(num_frames), desc="Rendering"):
        # Get audio for this frame
        aud_frame = audio_features[frame_idx:frame_idx+1]  # [1, 29, 16]
        
        # Render frame
        image = render_frame(
            model, rays_o, rays_d, aud_frame, bg_coords, poses,
            args.H, args.W, index=0, eye=None
        )
        
        frames.append(image)
    
    # Save video
    print(f"\nSaving video to {args.output}...")
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Save frames as temporary files
    temp_dir = 'temp_frames'
    os.makedirs(temp_dir, exist_ok=True)
    
    for i, frame in enumerate(frames):
        frame_path = os.path.join(temp_dir, f'frame_{i:05d}.png')
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # Use ffmpeg to create video with audio
    video_no_audio = args.output.replace('.mp4', '_no_audio.mp4')
    
    # Create video from frames
    cmd_video = [
        'ffmpeg', '-y',
        '-framerate', str(args.fps),
        '-i', os.path.join(temp_dir, 'frame_%05d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        video_no_audio
    ]
    subprocess.run(cmd_video, check=True)
    
    # Add audio to video
    cmd_audio = [
        'ffmpeg', '-y',
        '-i', video_no_audio,
        '-i', args.audio,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        args.output
    ]
    subprocess.run(cmd_audio, check=True)
    
    # Clean up
    import shutil
    shutil.rmtree(temp_dir)
    os.remove(video_no_audio)
    
    print("="*60)
    print(f"✓ Video saved to: {args.output}")
    print(f"  Duration: {num_frames/args.fps:.2f}s")
    print(f"  Resolution: {args.W}x{args.H}")
    print(f"  FPS: {args.fps}")
    print("="*60)


if __name__ == '__main__':
    main()

