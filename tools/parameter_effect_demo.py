#!/usr/bin/env python3
"""
Interactive BFM Parameter Effect Demonstration
Modify exp, euler, trans, and focal parameters to understand their effects on face rendering
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os
import cv2
from data_utils.face_tracking.facemodel import Face_3DMM
from data_utils.face_tracking.render_3dmm import Render_3DMM
from data_utils.face_tracking.util import euler2rot, rot_trans_pts, forward_transform
import argparse

class ParameterEffectDemo:
    def __init__(self, model_path="data_utils/face_tracking/3DMM", device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialize BFM model
        id_dim, exp_dim, tex_dim, point_num = 100, 79, 100, 34650
        self.model_3dmm = Face_3DMM(model_path, id_dim, exp_dim, tex_dim, point_num)

        # Initialize renderer
        self.renderer = Render_3DMM(focal=1015, img_h=512, img_w=512, device=self.device)

        # Load topology info for landmarks
        topo_info = np.load(os.path.join(model_path, "topology_info.npy"), allow_pickle=True).item()
        self.lands_info = torch.as_tensor(np.loadtxt(os.path.join(model_path, "lands_info.txt"), dtype=np.int32)).to(self.device)

        # Default camera parameters
        self.cxy = torch.tensor((256.0, 256.0), dtype=torch.float).to(self.device)

    def load_tracked_parameters(self, param_path):
        """Load tracked parameters from saved file"""
        params = torch.load(param_path, map_location=self.device)
        self.id_params = params['id']  # Already has batch dimension
        self.exp_params = params['exp']
        self.euler_params = params['euler']
        self.trans_params = params['trans']
        self.focal_params = params['focal']

        print(f"Loaded parameters from {param_path}")
        print(f"Frames: {self.exp_params.shape[0]}")
        print(f"Expression dims: {self.exp_params.shape[1]}")
        print(f"Identity dims: {self.id_params.shape[1]}")

    def modify_parameters(self, frame_idx=0, exp_mod=None, euler_mod=None, trans_mod=None, focal_mod=None):
        """Modify parameters for a specific frame"""

        # Start with original parameters
        id_p = self.id_params.clone()
        exp_p = self.exp_params[frame_idx:frame_idx+1].clone()
        euler_p = self.euler_params[frame_idx:frame_idx+1].clone()
        trans_p = self.trans_params[frame_idx:frame_idx+1].clone()
        focal_p = self.focal_params.clone()

        # Apply modifications
        if exp_mod is not None:
            exp_p += exp_mod.to(self.device)
            print(f"Modified expression parameters by: {exp_mod}")

        if euler_mod is not None:
            euler_p += euler_mod.to(self.device)
            print(f"Modified Euler angles by: {euler_mod} (radians)")

        if trans_mod is not None:
            trans_p += trans_mod.to(self.device)
            print(f"Modified translation by: {trans_mod}")

        if focal_mod is not None:
            focal_p += focal_mod
            print(f"Modified focal length by: {focal_mod}")

        return id_p, exp_p, euler_p, trans_p, focal_p

    def render_face(self, id_p, exp_p, euler_p, trans_p, focal_p, tex_p=None):
        """Render face with given parameters"""

        # Generate geometry
        geometry = self.model_3dmm.forward_geo(id_p, exp_p)

        # Generate texture (use neutral if not provided)
        if tex_p is None:
            tex_p = torch.zeros((1, 100), device=self.device)
        texture = self.model_3dmm.forward_tex(tex_p)

        # Apply pose transformation
        rot = euler2rot(euler_p)
        rott_geometry = rot_trans_pts(geometry, rot, trans_p)

        # Create neutral lighting
        diffuse_sh = torch.zeros((1, 27), device=self.device)
        diffuse_sh[:, 0] = 1.0  # Ambient lighting

        # Render
        rendered = self.renderer(rott_geometry, texture, diffuse_sh)

        return rendered.squeeze(0).cpu().numpy()

    def create_comparison_visualization(self, frame_idx=0, modifications=None):
        """Create side-by-side comparison of original vs modified faces"""

        if modifications is None:
            modifications = [
                {"name": "Original", "exp": None, "euler": None, "trans": None, "focal": None},
                {"name": "Happy Expression", "exp": torch.randn(1, 79) * 0.5, "euler": None, "trans": None, "focal": None},
                {"name": "Turn Head Right", "exp": None, "euler": torch.tensor([0.0, 0.3, 0.0]), "trans": None, "focal": None},
                {"name": "Move Closer", "exp": None, "euler": None, "trans": torch.tensor([0.0, 0.0, 100.0]), "focal": None},
                {"name": "Wide Angle Lens", "exp": None, "euler": None, "trans": None, "focal": -300.0},
            ]

        fig, axes = plt.subplots(1, len(modifications), figsize=(5*len(modifications), 5))
        if len(modifications) == 1:
            axes = [axes]

        for i, mod in enumerate(modifications):
            # Get modified parameters
            id_p, exp_p, euler_p, trans_p, focal_p = self.modify_parameters(
                frame_idx=frame_idx,
                exp_mod=mod["exp"],
                euler_mod=mod["euler"],
                trans_mod=mod["trans"],
                focal_mod=mod["focal"]
            )

            # Render
            rendered = self.render_face(id_p, exp_p, euler_p, trans_p, focal_p)

            # Display
            axes[i].imshow(rendered.astype(np.uint8))
            axes[i].set_title(mod["name"], fontsize=12, fontweight='bold')
            axes[i].axis('off')

        plt.tight_layout()
        return fig

    def demonstrate_parameter_ranges(self, frame_idx=0):
        """Show effects of parameter ranges"""

        param_types = [
            ("Expression Coefficient 0", "exp", 0, [-2, -1, 0, 1, 2]),
            ("Expression Coefficient 10", "exp", 10, [-2, -1, 0, 1, 2]),
            ("Pitch (Up/Down)", "euler", 0, [-0.5, -0.25, 0, 0.25, 0.5]),
            ("Yaw (Left/Right)", "euler", 1, [-0.5, -0.25, 0, 0.25, 0.5]),
            ("Z Translation (Distance)", "trans", 2, [-200, -100, 0, 100, 200]),
            ("Focal Length", "focal", 0, [-400, -200, 0, 200, 400]),
        ]

        for param_name, param_type, param_idx, values in param_types:
            fig, axes = plt.subplots(1, len(values), figsize=(4*len(values), 4))

            for i, val in enumerate(values):
                # Create modification
                if param_type == "exp":
                    mod = torch.zeros(1, 79)
                    mod[0, param_idx] = val
                    exp_mod, euler_mod, trans_mod, focal_mod = mod, None, None, None
                elif param_type == "euler":
                    mod = torch.zeros(1, 3)
                    mod[0, param_idx] = val
                    exp_mod, euler_mod, trans_mod, focal_mod = None, mod, None, None
                elif param_type == "trans":
                    mod = torch.zeros(1, 3)
                    mod[0, param_idx] = val
                    exp_mod, euler_mod, trans_mod, focal_mod = None, None, mod, None
                elif param_type == "focal":
                    exp_mod, euler_mod, trans_mod, focal_mod = None, None, None, val

                # Get parameters and render
                id_p, exp_p, euler_p, trans_p, focal_p = self.modify_parameters(
                    frame_idx=frame_idx,
                    exp_mod=exp_mod,
                    euler_mod=euler_mod,
                    trans_mod=trans_mod,
                    focal_mod=focal_mod
                )

                rendered = self.render_face(id_p, exp_p, euler_p, trans_p, focal_p)

                # Display
                axes[i].imshow(rendered.astype(np.uint8))
                axes[i].set_title(f"{param_name}\n{val}", fontsize=10)
                axes[i].axis('off')

            plt.suptitle(f"Effect of {param_name}", fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'parameter_effect_{param_type}_{param_idx}.png', dpi=150, bbox_inches='tight')
            plt.close()  # Close the figure to free memory

    def interactive_parameter_modification(self, frame_idx=0):
        """Interactive parameter modification interface"""

        print("\n" + "="*60)
        print("INTERACTIVE PARAMETER MODIFICATION")
        print("="*60)
        print("You can modify the following parameters:")
        print("• exp: Expression coefficients (79 values, range: -2 to +2)")
        print("• euler: Head rotation (3 values in radians: pitch, yaw, roll)")
        print("• trans: Translation (3 values: x, y, z position)")
        print("• focal: Camera focal length (scalar, typical range: 500-1500)")
        print("\nEnter modifications as comma-separated values, or 'quit' to exit")

        while True:
            try:
                print(f"\nCurrent frame: {frame_idx}")
                param_input = input("Modify parameters (format: exp=0.5,10:1.0;euler=0,0.2,0;trans=0,0,50;focal=100): ").strip()

                if param_input.lower() == 'quit':
                    break

                # Parse input
                modifications = {}
                parts = param_input.split(';')

                for part in parts:
                    if '=' not in part:
                        continue
                    param_type, values = part.split('=', 1)

                    if param_type == 'exp':
                        # Handle expression modifications like "exp=0:0.5,10:1.0"
                        if ':' in values:
                            exp_mods = {}
                            for exp_part in values.split(','):
                                if ':' in exp_part:
                                    idx, val = exp_part.split(':')
                                    exp_mods[int(idx)] = float(val)
                            modifications['exp'] = exp_mods
                        else:
                            # Single value for all coefficients
                            modifications['exp'] = float(values)
                    elif param_type in ['euler', 'trans']:
                        values_list = [float(x) for x in values.split(',')]
                        modifications[param_type] = torch.tensor(values_list)
                    elif param_type == 'focal':
                        modifications[param_type] = float(values)

                # Apply modifications
                exp_mod = None
                if 'exp' in modifications:
                    if isinstance(modifications['exp'], dict):
                        exp_mod = torch.zeros(1, 79)
                        for idx, val in modifications['exp'].items():
                            exp_mod[0, idx] = val
                    else:
                        exp_mod = torch.full((1, 79), modifications['exp'])

                id_p, exp_p, euler_p, trans_p, focal_p = self.modify_parameters(
                    frame_idx=frame_idx,
                    exp_mod=exp_mod,
                    euler_mod=modifications.get('euler'),
                    trans_mod=modifications.get('trans'),
                    focal_mod=modifications.get('focal')
                )

                # Render and display
                rendered = self.render_face(id_p, exp_p, euler_p, trans_p, focal_p)
                plt.figure(figsize=(8, 8))
                plt.imshow(rendered.astype(np.uint8))
                plt.title("Modified Face")
                plt.axis('off')
                plt.show()

            except Exception as e:
                print(f"Error: {e}")
                print("Please check your input format")

def main():
    parser = argparse.ArgumentParser(description="BFM Parameter Effect Demonstration")
    parser.add_argument('--param_path', type=str, default='track_params.pt',
                       help='Path to tracked parameters file')
    parser.add_argument('--frame', type=int, default=0,
                       help='Frame index to use for demonstration')
    parser.add_argument('--interactive', action='store_true',
                       help='Enable interactive parameter modification')

    args = parser.parse_args()

    # Initialize demo
    demo = ParameterEffectDemo()

    # Load parameters
    if os.path.exists(args.param_path):
        demo.load_tracked_parameters(args.param_path)
    else:
        print(f"Parameter file {args.param_path} not found. Please run face tracking first.")
        return

    if args.interactive:
        # Interactive mode
        demo.interactive_parameter_modification(args.frame)
    else:
        # Demonstration mode
        print("\nGenerating parameter effect demonstrations...")

        # Create comparison visualizations
        fig = demo.create_comparison_visualization(args.frame)
        fig.savefig('parameter_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory

        # Demonstrate parameter ranges
        demo.demonstrate_parameter_ranges(args.frame)

        print("\nDemonstration complete!")
        print("Generated files:")
        print("- parameter_comparison.png: Side-by-side comparisons")
        print("- parameter_effect_*.png: Individual parameter range effects")

if __name__ == "__main__":
    main()
