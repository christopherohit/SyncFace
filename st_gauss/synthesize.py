"""
ST-Gauss Synthesis/Inference

Render talking head videos using trained ST-Gauss model.
Supports:
- Inference with custom audio
- Batch rendering for evaluation
- Real-time preview (optional)
"""

import os
import sys
import argparse
import copy
import torch
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from st_gauss.mouth_nerf import MouthNeRFNetwork
from st_gauss.face_motion import BlendshapeMotionNetwork
from st_gauss.hybrid_renderer import HybridRenderer
from st_gauss.config import get_config

from TalkingGaussian.scene import Scene, GaussianModel
from TalkingGaussian.utils.camera_utils import loadCamOnTheFly
from TalkingGaussian.arguments import ModelParams, PipelineParams


class STGaussSynthesizer:
    """ST-Gauss inference wrapper."""
    
    def __init__(
        self,
        checkpoint_path: str,
        config_preset: str = 'default',
        device: str = 'cuda',
    ):
        self.device = device
        self.config = get_config(config_preset)
        
        # Initialize models
        self.gaussians = GaussianModel(3)  # SH degree 3
        
        self.motion_net = BlendshapeMotionNetwork(
            audio_type=self.config.audio_type,
            audio_dim=self.config.audio_dim,
            blendshape_dim=self.config.blendshape_dim,
            bound=self.config.face_bound,
            use_facial_attention=self.config.use_facial_aware_attention,
        ).to(device)
        
        self.mouth_nerf = MouthNeRFNetwork(
            audio_type=self.config.audio_type,
            audio_dim=self.config.audio_dim,
            bound=self.config.mouth_bound,
            num_hash_levels=self.config.mouth_hash_levels,
        ).to(device)
        
        # Load checkpoint
        self.load_checkpoint(checkpoint_path)
        
        # Set to eval mode
        self.motion_net.eval()
        self.mouth_nerf.eval()
        self.mouth_nerf.testing = True
        
        # Renderer
        self.renderer = HybridRenderer(
            background_color=self.config.background_color,
        ).to(device)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load trained model checkpoint."""
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'gaussians' in checkpoint:
            self.gaussians.restore(checkpoint['gaussians'], None)
        if 'motion_net' in checkpoint:
            self.motion_net.load_state_dict(checkpoint['motion_net'])
        if 'mouth_nerf' in checkpoint:
            self.mouth_nerf.load_state_dict(checkpoint['mouth_nerf'])
        
        print("Checkpoint loaded successfully")
    
    @torch.no_grad()
    def render_frame(
        self,
        viewpoint_camera,
        pipe,
        background: torch.Tensor = None,
    ):
        """Render a single frame."""
        if background is None:
            background = torch.tensor(
                self.config.background_color,
                dtype=torch.float32,
                device=self.device
            )
        
        mouth_mask = torch.as_tensor(
            viewpoint_camera.talking_dict.get("mouth_mask", 
                torch.zeros((512, 512), dtype=torch.bool))
        ).to(self.device)
        
        render_pkg = self.renderer(
            viewpoint_camera=viewpoint_camera,
            gaussian_model=self.gaussians,
            face_motion_network=self.motion_net,
            mouth_nerf=self.mouth_nerf,
            pipe=pipe,
            mouth_mask=mouth_mask,
            render_mouth=True,
        )
        
        return render_pkg
    
    @torch.no_grad()
    def synthesize_video(
        self,
        scene: Scene,
        pipe,
        output_path: str,
        fps: int = 25,
        use_test_cameras: bool = False,
    ):
        """Synthesize full video."""
        cameras = scene.getTestCameras() if use_test_cameras else scene.getTrainCameras()
        
        # Get image dimensions from first camera
        first_cam = cameras[0]
        if first_cam.original_image is None:
            first_cam = loadCamOnTheFly(copy.deepcopy(first_cam))
        H, W = first_cam.image_height, first_cam.image_width
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
        
        background = torch.tensor(
            self.config.background_color,
            dtype=torch.float32,
            device=self.device
        )
        
        print(f"Rendering {len(cameras)} frames...")
        
        for viewpoint_cam in tqdm(cameras, desc="Rendering"):
            if viewpoint_cam.original_image is None:
                viewpoint_cam = loadCamOnTheFly(copy.deepcopy(viewpoint_cam))
            
            render_pkg = self.render_frame(viewpoint_cam, pipe, background)
            
            image = render_pkg['render']
            alpha = render_pkg.get('face_alpha', torch.ones_like(image[:1]))
            
            # Composite with background
            if hasattr(viewpoint_cam, 'background') and viewpoint_cam.background is not None:
                bg = viewpoint_cam.background.to(self.device) / 255.0
                image = image * alpha + bg * (1.0 - alpha)
            
            # Convert to numpy/OpenCV format
            image_np = (image.clamp(0, 1) * 255).byte()
            image_np = image_np.permute(1, 2, 0).cpu().numpy()
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            writer.write(image_np)
        
        writer.release()
        print(f"Video saved to {output_path}")
        
        return output_path


def synthesize(args):
    """Main synthesis function."""
    # Load config
    config = get_config(args.config)
    
    # Determine checkpoint path
    checkpoint_path = args.checkpoint
    if checkpoint_path is None:
        checkpoint_path = os.path.join(args.model_path, "chkpnt_joint_latest.pth")
    
    if not os.path.exists(checkpoint_path):
        # Try face checkpoint if joint not available
        checkpoint_path = os.path.join(args.model_path, "chkpnt_face_latest.pth")
    
    # Initialize synthesizer
    synthesizer = STGaussSynthesizer(
        checkpoint_path=checkpoint_path,
        config_preset=args.config,
    )
    
    # Load scene
    from TalkingGaussian.arguments import ModelParams
    lp = ModelParams(argparse.ArgumentParser())
    dataset = lp.extract(args)
    
    scene = Scene(dataset, synthesizer.gaussians, load_iteration=None, shuffle=False)
    
    # Create pipe object
    class PipeConfig:
        debug = False
        compute_cov3D_python = False
        convert_SHs_python = False
    
    pipe = PipeConfig()
    
    # Output path
    output_path = args.output
    if output_path is None:
        output_path = os.path.join(args.model_path, "synthesis.mp4")
    
    # Synthesize
    synthesizer.synthesize_video(
        scene=scene,
        pipe=pipe,
        output_path=output_path,
        fps=args.fps,
        use_test_cameras=args.use_test,
    )
    
    print(f"\nSynthesis complete: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ST-Gauss Synthesis")
    
    parser.add_argument('-s', '--source_path', type=str, required=True,
                       help='Path to preprocessed data')
    parser.add_argument('-m', '--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Specific checkpoint to load')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path')
    parser.add_argument('--config', type=str, default='default',
                       help='Config preset')
    parser.add_argument('--fps', type=int, default=25,
                       help='Output video FPS')
    parser.add_argument('--use_test', action='store_true',
                       help='Use test cameras instead of train')
    
    args = parser.parse_args()
    
    synthesize(args)

