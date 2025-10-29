"""
Preprocessing script to compute monocular depth and normal estimates.

This script processes video frames and generates:
- Depth estimates using a pre-trained monocular depth estimator
- Normal estimates computed from depth or using a normal prediction network

Supported models:
- MiDaS (depth): Robust general-purpose depth estimation
- ZoeDepth (depth): High-quality metric depth
- Omnidata (depth + normals): Joint prediction
- DPT (depth): Vision transformer-based depth

Usage:
    conda activate synctalk && python preprocess_geometry.py \
        --data /path/to/person_data \
        --method midas \
        --device cuda:0
"""

import argparse
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
from typing import Tuple, Optional


def load_midas_model(model_type: str = "DPT_Large", device='cuda'):
    """
    Load MiDaS depth estimation model.
    
    Args:
        model_type: "DPT_Large", "DPT_Hybrid", or "MiDaS_small"
        device: Device to load model on
    
    Returns:
        model, transform
    """
    try:
        import torch.hub
        # Load from torch hub
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        midas.to(device)
        midas.eval()
        
        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform
        
        print(f"[INFO] Loaded MiDaS model: {model_type}")
        return midas, transform
    
    except Exception as e:
        print(f"[ERROR] Failed to load MiDaS: {e}")
        print("[INFO] Install with: pip install timm")
        return None, None


def estimate_depth_midas(image: np.ndarray, model, transform, device='cuda') -> np.ndarray:
    """
    Estimate depth using MiDaS.
    
    Args:
        image: [H, W, 3] RGB image (0-255)
        model: MiDaS model
        transform: MiDaS transform
        device: Device
    
    Returns:
        depth: [H, W] depth map (inverse depth, larger = closer)
    """
    # Prepare input
    input_batch = transform(image).to(device)
    
    # Predict
    with torch.no_grad():
        prediction = model(input_batch)
        
        # Resize to original resolution
        prediction = F.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    depth = prediction.cpu().numpy()
    
    # Normalize to [0, 1]
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    
    return depth


def compute_normals_from_depth(depth: np.ndarray, K: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute surface normals from depth map.
    
    Args:
        depth: [H, W] depth map
        K: [3, 3] camera intrinsic matrix (optional)
    
    Returns:
        normals: [H, W, 3] surface normals
    """
    H, W = depth.shape
    
    # Compute gradients using Sobel filters
    depth_dx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
    depth_dy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
    
    # If intrinsics available, compute in 3D
    if K is not None:
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        
        # Create coordinate grids
        y, x = np.mgrid[0:H, 0:W]
        
        # Back-project to 3D
        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        Z = depth
        
        # Compute normals from 3D gradients
        dX_dx = np.gradient(X, axis=1)
        dY_dx = np.gradient(Y, axis=1)
        dZ_dx = np.gradient(Z, axis=1)
        
        dX_dy = np.gradient(X, axis=0)
        dY_dy = np.gradient(Y, axis=0)
        dZ_dy = np.gradient(Z, axis=0)
        
        # Tangent vectors
        T_x = np.stack([dX_dx, dY_dx, dZ_dx], axis=-1)
        T_y = np.stack([dX_dy, dY_dy, dZ_dy], axis=-1)
        
        # Normal = T_x × T_y
        normals = np.cross(T_x, T_y)
    else:
        # Simplified normal computation without intrinsics
        # Assume canonical camera
        normal_x = -depth_dx
        normal_y = -depth_dy
        normal_z = np.ones_like(depth)
        
        normals = np.stack([normal_x, normal_y, normal_z], axis=-1)
    
    # Normalize
    norm = np.linalg.norm(normals, axis=-1, keepdims=True) + 1e-8
    normals = normals / norm
    
    return normals


def load_camera_intrinsics(data_path: str) -> Optional[np.ndarray]:
    """
    Load camera intrinsics from transforms.json
    
    Returns:
        K: [3, 3] intrinsic matrix or None
    """
    import json
    
    transforms_path = os.path.join(data_path, 'transforms_train.json')
    if not os.path.exists(transforms_path):
        return None
    
    with open(transforms_path, 'r') as f:
        transforms = json.load(f)
    
    # Extract intrinsics
    if 'camera_angle_x' in transforms:
        # Compute from FOV
        w = transforms.get('w', 512)
        h = transforms.get('h', 512)
        fov_x = transforms['camera_angle_x']
        
        fx = w / (2 * np.tan(fov_x / 2))
        fy = fx  # Assume square pixels
        cx = w / 2
        cy = h / 2
        
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ])
        return K
    
    return None


def process_dataset(data_path: str, method: str, device: str, output_normals: bool = True):
    """
    Process all frames in a dataset and save geometry estimates.
    
    Args:
        data_path: Path to person data directory
        method: Estimation method ("midas", "zoedepth", etc.)
        device: Device to use
        output_normals: Whether to compute and save normals
    """
    print("="*60)
    print(f"Geometry Estimation Preprocessing")
    print(f"  Data: {data_path}")
    print(f"  Method: {method}")
    print(f"  Device: {device}")
    print("="*60)
    
    # Setup output directories
    depth_dir = os.path.join(data_path, 'depth_estimates')
    normal_dir = os.path.join(data_path, 'normal_estimates')
    os.makedirs(depth_dir, exist_ok=True)
    if output_normals:
        os.makedirs(normal_dir, exist_ok=True)
    
    # Load camera intrinsics (optional, for better normal estimation)
    K = load_camera_intrinsics(data_path)
    if K is not None:
        print(f"[INFO] Loaded camera intrinsics:\n{K}")
    else:
        print("[WARN] Camera intrinsics not found, using default")
    
    # Load depth model
    if method == 'midas':
        model, transform = load_midas_model("DPT_Large", device=device)
        if model is None:
            print("[ERROR] Failed to load depth model")
            return
    else:
        print(f"[ERROR] Unsupported method: {method}")
        print("[INFO] Supported methods: midas")
        return
    
    # Find image files
    img_dir = os.path.join(data_path, 'gt_imgs')
    if not os.path.exists(img_dir):
        img_dir = os.path.join(data_path, 'ori_imgs')
    if not os.path.exists(img_dir):
        print(f"[ERROR] Image directory not found: {data_path}/gt_imgs or ori_imgs")
        return
    
    image_files = sorted([f for f in os.listdir(img_dir) 
                         if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"[INFO] Found {len(image_files)} images")
    
    # Process each frame
    for img_file in tqdm(image_files, desc="Processing frames"):
        frame_id = os.path.splitext(img_file)[0]
        img_path = os.path.join(img_dir, img_file)
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] Failed to load {img_path}")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Estimate depth
        if method == 'midas':
            depth = estimate_depth_midas(image_rgb, model, transform, device)
        
        # Save depth
        depth_path = os.path.join(depth_dir, f'{frame_id}.npy')
        np.save(depth_path, depth.astype(np.float32))
        
        # Compute and save normals
        if output_normals:
            normals = compute_normals_from_depth(depth, K)
            normal_path = os.path.join(normal_dir, f'{frame_id}.npy')
            np.save(normal_path, normals.astype(np.float32))
    
    print(f"\n[INFO] Saved depth estimates to {depth_dir}")
    if output_normals:
        print(f"[INFO] Saved normal estimates to {normal_dir}")
    
    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Preprocess geometry estimates for adaptation")
    parser.add_argument('--data', type=str, required=True,
                       help="Path to person data directory")
    parser.add_argument('--method', type=str, default='midas',
                       choices=['midas', 'zoedepth', 'dpt'],
                       help="Depth estimation method")
    parser.add_argument('--device', type=str, default='cuda:0',
                       help="Device to use (cuda:0, cpu, etc.)")
    parser.add_argument('--no_normals', action='store_true',
                       help="Skip normal estimation")
    
    args = parser.parse_args()
    
    # Process dataset
    process_dataset(
        data_path=args.data,
        method=args.method,
        device=args.device,
        output_normals=not args.no_normals
    )


if __name__ == '__main__':
    main()


# Fallback strategy if MiDaS not available
"""
FALLBACK STRATEGY: If monocular geometry estimator is not available

Option 1: Use smoothed depth proxies
    - Render depth from NeRF itself (bootstrap)
    - Apply temporal smoothing
    - Use as weak supervision

Option 2: Skip geometry prior
    - Set --use_geometry_prior False in adapt_identity.py
    - Rely on photometric loss only
    - May result in degraded 3D consistency

Option 3: Use pre-computed estimates
    - Run depth estimation separately (e.g., with online services)
    - Save as .npy files in depth_estimates/ folder
    - Format: [H, W] float32 arrays

To install MiDaS dependencies:
    pip install timm
    pip install opencv-python
"""


