"""
Modified data provider for 3D Gaussian Splatting
This replaces ray-based sampling with full image loading for 3DGS rendering
"""

import os
import cv2
import glob
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation
import trimesh

from .utils import get_audio_features, AudDataset
from .provider import NeRFDataset as BaseNeRFDataset  # Import the base class
from .network import AudioEncoder


class GaussianDataset(BaseNeRFDataset):
    """
    Modified dataset for 3D Gaussian Splatting
    Inherits most functionality from NeRFDataset but modifies collate for full image loading
    """
    
    def __init__(self, opt, device, type='train', downscale=1):
        # Initialize parent class
        super().__init__(opt, device, type, downscale)
        
        # Override ray-based settings for 3DGS
        self.num_rays = -1  # Always use full images for 3DGS
        
        print(f'[INFO] GaussianDataset initialized for {type} with {len(self.poses)} frames')
        
    def collate(self, index):
        """
        Modified collate function for 3DGS that loads full images instead of rays
        
        Args:
            index: List of indices to load
            
        Returns:
            Dictionary with:
                - Full images instead of sampled rays
                - Camera parameters (intrinsics, extrinsics)
                - Audio and expression features
        """
        B = len(index)  # Batch size (usually 1)
        results = {}
        
        # 1. Load audio features (same as original)
        if self.auds is not None:
            auds = get_audio_features(self.auds, self.opt.att, index[0]).to(self.device)
            results['auds'] = auds
        else:
            results['auds'] = None
            
        # 2. Handle mirroring for pose (same as original)
        index[0] = self.mirror_index(index[0])
        
        # 3. Load camera poses
        poses = self.poses[index].to(self.device)  # [B, 4, 4]
        results['poses'] = poses
        
        # 4. Store image dimensions
        results['H'] = self.H
        results['W'] = self.W
        results['index'] = index
        
        # 5. Load expression/eye features (same as original)
        if self.opt.exp_eye:
            results['eye'] = self.eye_area[index].to(self.device)
        else:
            results['eye'] = None
            
        # 6. Load full ground truth image (NOT ray-sampled)
        images = self.images[index]  # [B, H, W, 3/4]
        if self.preload == 0:  # Load from disk
            images = cv2.imread(images[0], cv2.IMREAD_UNCHANGED)  # [H, W, 3/4]
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
            images = images.astype(np.float32) / 255.0  # Normalize to [0, 1]
            images = torch.from_numpy(images).unsqueeze(0)  # Add batch dimension
        
        images = images.to(self.device)  # [B, H, W, 3/4]
        
        # For 3DGS, we need images in [B, C, H, W] format
        results['images'] = images.permute(0, 3, 1, 2)  # [B, 3/4, H, W]
        results['images_hw'] = images  # Keep [B, H, W, C] format as well
        
        # 7. Load background/torso image if needed
        bg_torso_img = self.torso_img[index]
        if self.preload == 0:  # Load from disk
            bg_torso_img = cv2.imread(bg_torso_img[0], cv2.IMREAD_UNCHANGED)  # [H, W, 4]
            bg_torso_img = cv2.cvtColor(bg_torso_img, cv2.COLOR_BGRA2RGBA)
            bg_torso_img = bg_torso_img.astype(np.float32) / 255.0
            bg_torso_img = torch.from_numpy(bg_torso_img).unsqueeze(0)
        
        # Blend torso with background
        bg_torso_img = bg_torso_img[..., :3] * bg_torso_img[..., 3:] + self.bg_img * (1 - bg_torso_img[..., 3:])
        bg_torso_img = bg_torso_img.to(self.device)  # [B, H, W, 3]
        
        if not self.opt.torso:
            bg_img = bg_torso_img
        else:
            bg_img = self.bg_img.unsqueeze(0).repeat(B, 1, 1, 1).to(self.device)
            
        results['bg_color'] = bg_img  # [B, H, W, 3]
        results['bg_torso_color'] = bg_torso_img
        
        # 8. Load face masks and rectangles for loss computation
        if self.training:
            idx = index[0]

            # Check bounds for all rectangle arrays
            max_idx = len(self.poses) - 1
            if idx > max_idx:
                print(f"[WARNING] Index {idx} out of bounds (max: {max_idx}), using index 0")
                idx = 0

            # Store face rectangles for masked loss computation with bounds checking
            if hasattr(self, 'face_rect') and self.face_rect is not None and len(self.face_rect) > idx:
                results['face_rect'] = self.face_rect[idx]
            else:
                results['face_rect'] = [0, self.W, 0, self.H]  # Default to full image

            if hasattr(self, 'lips_rect') and self.lips_rect is not None and len(self.lips_rect) > idx:
                results['lips_rect'] = self.lips_rect[idx]
            else:
                results['lips_rect'] = None

            if hasattr(self, 'lhalf_rect') and self.lhalf_rect is not None and len(self.lhalf_rect) > idx:
                results['lhalf_rect'] = self.lhalf_rect[idx]
            else:
                results['lhalf_rect'] = None

            if hasattr(self, 'upface_rect') and self.upface_rect is not None and len(self.upface_rect) > idx:
                results['upface_rect'] = self.upface_rect[idx]
            else:
                results['upface_rect'] = [0, self.W, 0, self.H//2]  # Default to upper half

            if hasattr(self, 'lowface_rect') and self.lowface_rect is not None and len(self.lowface_rect) > idx:
                results['lowface_rect'] = self.lowface_rect[idx]
            else:
                results['lowface_rect'] = None

            if self.opt.exp_eye and hasattr(self, 'eye_rect') and self.eye_rect is not None and len(self.eye_rect) > idx:
                results['eye_rect'] = self.eye_rect[idx]

            # Create full image masks instead of ray masks
            face_mask = self.create_rect_mask(results['face_rect'], self.H, self.W)
            results['face_mask'] = face_mask.to(self.device)  # [H, W]

            if results['lips_rect'] is not None:
                lips_mask = self.create_rect_mask(results['lips_rect'], self.H, self.W)
                results['lips_mask'] = lips_mask.to(self.device)

            if results['lhalf_rect'] is not None:
                lhalf_mask = self.create_rect_mask(results['lhalf_rect'], self.H, self.W)
                results['lhalf_mask'] = lhalf_mask.to(self.device)

            if results['upface_rect'] is not None:
                upface_mask = self.create_rect_mask(results['upface_rect'], self.H, self.W)
                results['upface_mask'] = upface_mask.to(self.device)

            if results['lowface_rect'] is not None:
                lowface_mask = self.create_rect_mask(results['lowface_rect'], self.H, self.W)
                results['lowface_mask'] = lowface_mask.to(self.device)

            if self.opt.exp_eye and 'eye_rect' in results:
                eye_mask = self.create_rect_mask(results['eye_rect'], self.H, self.W)
                results['eye_mask'] = eye_mask.to(self.device)
        
        # 9. Handle portrait mode data if needed
        if self.opt.portrait:
            bg_gt_images = self.gt_images[index]
            if self.preload == 0:
                bg_gt_images = cv2.imread(bg_gt_images[0], cv2.IMREAD_UNCHANGED)
                bg_gt_images = cv2.cvtColor(bg_gt_images, cv2.COLOR_BGR2RGB)
                bg_gt_images = bg_gt_images.astype(np.float32) / 255.0
                bg_gt_images = torch.from_numpy(bg_gt_images).unsqueeze(0)
            bg_gt_images = bg_gt_images.to(self.device)
            results['bg_gt_images'] = bg_gt_images.permute(0, 3, 1, 2)  # [B, C, H, W]
            
            bg_face_mask = self.face_mask_imgs[index]
            if self.preload == 0:
                bg_face_mask = (255 - cv2.imread(bg_face_mask[0])[:, :, 1]) / 255.0
                bg_face_mask = torch.from_numpy(bg_face_mask).unsqueeze(0)
            bg_face_mask = bg_face_mask.to(self.device)
            results['bg_face_mask'] = bg_face_mask  # [B, H, W]
        
        # 10. Camera intrinsics
        results['intrinsics'] = self.intrinsics  # [fx, fy, cx, cy]
        
        # Build K matrix for 3DGS
        K = torch.eye(4, device=self.device)
        K[0, 0] = self.intrinsics[0]  # fx
        K[1, 1] = self.intrinsics[1]  # fy
        K[0, 2] = self.intrinsics[2]  # cx
        K[1, 2] = self.intrinsics[3]  # cy
        K[2, 2] = 1
        K[3, 3] = 1
        results['K'] = K.unsqueeze(0).repeat(B, 1, 1)  # [B, 4, 4]
        
        # Calculate FoV for 3DGS
        results['FovX'] = 2 * np.arctan(self.W / (2 * self.intrinsics[0]))
        results['FovY'] = 2 * np.arctan(self.H / (2 * self.intrinsics[1]))
        
        return results
    
    def create_rect_mask(self, rect, H, W):
        """
        Create a binary mask from a rectangle
        
        Args:
            rect: [xmin, xmax, ymin, ymax]
            H, W: Image height and width
            
        Returns:
            Binary mask [H, W]
        """
        xmin, xmax, ymin, ymax = rect
        mask = torch.zeros(H, W, dtype=torch.bool)
        mask[ymin:ymax, xmin:xmax] = True
        return mask
    
    def dataloader(self):
        """
        Create dataloader for 3DGS training/testing
        """
        if self.training:
            size = self.poses.shape[0]
        else:
            if self.auds is not None:
                size = self.auds.shape[0]
            else:
                size = 2 * self.poses.shape[0]
        
        loader = DataLoader(
            list(range(size)), 
            batch_size=1,  # 3DGS typically uses batch size of 1
            collate_fn=self.collate, 
            shuffle=self.training, 
            num_workers=0
        )
        
        loader._data = self  # Store reference to dataset
        loader.has_gt = (self.opt.aud == '')  # Check if we have ground truth
        
        return loader


# Helper function to create camera object for 3DGS
class Camera:
    """
    Camera object for 3DGS rendering
    """
    def __init__(self, C2W, W, H, K, FovX=None, FovY=None):
        """
        Args:
            C2W: Camera-to-world transformation matrix [4, 4]
            W, H: Image width and height
            K: Intrinsic matrix [4, 4]
            FovX, FovY: Field of view angles (optional)
        """
        self.C2W = C2W
        self.W2C = torch.inverse(C2W)  # World-to-camera
        self.image_width = W
        self.image_height = H
        self.K = K
        
        # Extract camera center from C2W
        self.camera_center = C2W[:3, 3]
        
        # Calculate FoV if not provided
        if FovX is None:
            FovX = 2 * torch.atan(W / (2 * K[0, 0]))
        if FovY is None:
            FovY = 2 * torch.atan(H / (2 * K[1, 1]))
            
        self.FoVx = FovX
        self.FoVy = FovY
        
        # Create projection matrix
        self.projection_matrix = self.get_projection_matrix()
        
        # World view transform is W2C
        self.world_view_transform = self.W2C.transpose(0, 1)  # Transpose for 3DGS
        
        # Full projection matrix
        self.full_proj_matrix = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        
    def get_projection_matrix(self):
        """
        Create OpenGL-style projection matrix
        """
        tanHalfFovY = torch.tan(torch.tensor(self.FoVy / 2, dtype=torch.float32))
        tanHalfFovX = torch.tan(torch.tensor(self.FoVx / 2, dtype=torch.float32))
        
        top = tanHalfFovY * 1.0  # near = 1.0
        bottom = -top
        right = tanHalfFovX * 1.0
        left = -right
        
        P = torch.zeros(4, 4, device=self.K.device)
        z_near = 0.1
        z_far = 100.0
        
        P[0, 0] = 2.0 / (right - left)
        P[1, 1] = 2.0 / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[2, 2] = z_far / (z_far - z_near)
        P[2, 3] = -(z_far * z_near) / (z_far - z_near)
        P[3, 2] = 1.0
        
        return P.transpose(0, 1)


def create_gaussian_data_loader(opt, device, type='train', downscale=1):
    """
    Factory function to create GaussianDataset and DataLoader
    
    Args:
        opt: Options/config object
        device: torch device
        type: 'train', 'val', 'test', or 'all'
        downscale: Image downscaling factor
        
    Returns:
        DataLoader for 3DGS training/testing
    """
    dataset = GaussianDataset(opt, device, type, downscale)
    return dataset.dataloader()
