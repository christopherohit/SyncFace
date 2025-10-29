#!/usr/bin/env python3
"""
Example: How to modify BFM parameters to understand their effects
This script demonstrates basic usage of the parameter effect demo
"""

import torch
import numpy as np
from parameter_effect_demo import ParameterEffectDemo
import matplotlib.pyplot as plt

def create_sample_parameters():
    """Create sample parameters for demonstration when track_params.pt doesn't exist"""

    # Create sample parameters
    sample_params = {
        'id': torch.randn(1, 100) * 0.1,  # Identity parameters
        'exp': torch.zeros(10, 79),       # Expression parameters for 10 frames
        'euler': torch.zeros(10, 3),      # Euler angles for 10 frames
        'trans': torch.zeros(10, 3),      # Translation for 10 frames
        'focal': torch.tensor(1015.0)     # Focal length
    }

    # Add some variation to make it interesting
    sample_params['trans'][:, 2] = -600  # Default distance
    sample_params['exp'][0, 0] = 0.5     # Slight expression change

    # Save sample parameters
    torch.save(sample_params, 'sample_track_params.pt')
    print("Created sample parameters file: sample_track_params.pt")

    return sample_params

def demonstrate_expression_effects():
    """Show how expression parameters affect the face"""

    print("\n" + "="*50)
    print("EXPRESSION PARAMETER EFFECTS")
    print("="*50)

    demo = ParameterEffectDemo()

    # Load or create parameters
    try:
        demo.load_tracked_parameters('track_params.pt')
    except:
        print("Using sample parameters for demonstration...")
        create_sample_parameters()
        demo.load_tracked_parameters('sample_track_params.pt')

    # Create expression modifications
    modifications = [
        {"name": "Neutral", "exp": None, "euler": None, "trans": None, "focal": None},
        {"name": "Happy (Exp 0:+1)", "exp": torch.zeros(1, 79).scatter_(1, torch.tensor([[0]]), 1.0), "euler": None, "trans": None, "focal": None},
        {"name": "Sad (Exp 1:-1)", "exp": torch.zeros(1, 79).scatter_(1, torch.tensor([[1]]), -1.0), "euler": None, "trans": None, "focal": None},
        {"name": "Surprised (Exp 2:+1)", "exp": torch.zeros(1, 79).scatter_(1, torch.tensor([[2]]), 1.0), "euler": None, "trans": None, "focal": None},
    ]

    fig = demo.create_comparison_visualization(modifications=modifications)
    plt.savefig('expression_effects.png', dpi=150, bbox_inches='tight')
    plt.show()

def demonstrate_pose_effects():
    """Show how pose (Euler) parameters affect head orientation"""

    print("\n" + "="*50)
    print("POSE PARAMETER EFFECTS")
    print("="*50)

    demo = ParameterEffectDemo()

    try:
        demo.load_tracked_parameters('track_params.pt')
    except:
        create_sample_parameters()
        demo.load_tracked_parameters('sample_track_params.pt')

    # Create pose modifications
    modifications = [
        {"name": "Front View", "exp": None, "euler": None, "trans": None, "focal": None},
        {"name": "Turn Right\n(Yaw: +0.5)", "exp": None, "euler": torch.tensor([0.0, 0.5, 0.0]), "trans": None, "focal": None},
        {"name": "Turn Left\n(Yaw: -0.5)", "exp": None, "euler": torch.tensor([0.0, -0.5, 0.0]), "trans": None, "focal": None},
        {"name": "Look Up\n(Pitch: +0.3)", "exp": None, "euler": torch.tensor([0.3, 0.0, 0.0]), "trans": None, "focal": None},
        {"name": "Look Down\n(Pitch: -0.3)", "exp": None, "euler": torch.tensor([-0.3, 0.0, 0.0]), "trans": None, "focal": None},
        {"name": "Tilt Head\n(Roll: +0.2)", "exp": None, "euler": torch.tensor([0.0, 0.0, 0.2]), "trans": None, "focal": None},
    ]

    fig = demo.create_comparison_visualization(modifications=modifications)
    plt.savefig('pose_effects.png', dpi=150, bbox_inches='tight')
    plt.show()

def demonstrate_translation_effects():
    """Show how translation parameters affect position"""

    print("\n" + "="*50)
    print("TRANSLATION PARAMETER EFFECTS")
    print("="*50)

    demo = ParameterEffectDemo()

    try:
        demo.load_tracked_parameters('track_params.pt')
    except:
        create_sample_parameters()
        demo.load_tracked_parameters('sample_track_params.pt')

    # Create translation modifications
    modifications = [
        {"name": "Center Position", "exp": None, "euler": None, "trans": None, "focal": None},
        {"name": "Move Right\n(X: +50)", "exp": None, "euler": None, "trans": torch.tensor([50.0, 0.0, 0.0]), "focal": None},
        {"name": "Move Left\n(X: -50)", "exp": None, "euler": None, "trans": torch.tensor([-50.0, 0.0, 0.0]), "focal": None},
        {"name": "Move Up\n(Y: +30)", "exp": None, "euler": None, "trans": torch.tensor([0.0, 30.0, 0.0]), "focal": None},
        {"name": "Closer\n(Z: +100)", "exp": None, "euler": None, "trans": torch.tensor([0.0, 0.0, 100.0]), "focal": None},
        {"name": "Farther\n(Z: -100)", "exp": None, "euler": None, "trans": torch.tensor([0.0, 0.0, -100.0]), "focal": None},
    ]

    fig = demo.create_comparison_visualization(modifications=modifications)
    plt.savefig('translation_effects.png', dpi=150, bbox_inches='tight')
    plt.show()

def demonstrate_focal_effects():
    """Show how focal length affects perspective"""

    print("\n" + "="*50)
    print("FOCAL LENGTH EFFECTS")
    print("="*50)

    demo = ParameterEffectDemo()

    try:
        demo.load_tracked_parameters('track_params.pt')
    except:
        create_sample_parameters()
        demo.load_tracked_parameters('sample_track_params.pt')

    # Create focal length modifications
    modifications = [
        {"name": "Normal Lens\n(Focal: 1015)", "exp": None, "euler": None, "trans": None, "focal": None},
        {"name": "Wide Angle\n(Focal: 715)", "exp": None, "euler": None, "trans": None, "focal": -300.0},
        {"name": "Telephoto\n(Focal: 1315)", "exp": None, "euler": None, "trans": None, "focal": 300.0},
        {"name": "Very Wide\n(Focal: 515)", "exp": None, "euler": None, "trans": None, "focal": -500.0},
    ]

    fig = demo.create_comparison_visualization(modifications=modifications)
    plt.savefig('focal_effects.png', dpi=150, bbox_inches='tight')
    plt.show()

def parameter_guide():
    """Print a guide on how to interpret BFM parameters"""

    print("\n" + "="*80)
    print("BFM PARAMETER GUIDE")
    print("="*80)

    print("""
PARAMETERS OVERVIEW:
• exp (79 coefficients): Control facial expressions and movements
  - Range: typically -2.0 to +2.0
  - Different coefficients control different facial muscles
  - exp[0]: often mouth opening/smile
  - exp[1]: often sadness/frown
  - exp[2]: often surprise/eyebrow raise

• euler (3 angles): Control head orientation in radians
  - euler[0] (pitch): up/down head tilt
  - euler[1] (yaw): left/right head rotation
  - euler[2] (roll): head tilt side-to-side
  - Range: typically -π/2 to +π/2

• trans (3 values): Control head position
  - trans[0] (X): left/right position
  - trans[1] (Y): up/down position
  - trans[2] (Z): distance from camera (negative = farther)
  - Units: typically millimeters

• focal (1 value): Camera focal length
  - Controls perspective distortion
  - Higher focal = telephoto (less distortion)
  - Lower focal = wide angle (more distortion)
  - Typical range: 500-1500 pixels

USAGE EXAMPLES:
1. Make face smile: exp=0:1.0 (set coefficient 0 to 1.0)
2. Turn head right: euler=0,0.5,0 (yaw by 0.5 radians)
3. Move face closer: trans=0,0,100 (increase Z by 100)
4. Wide angle view: focal=-300 (decrease focal length)

For interactive modification, run:
python parameter_effect_demo.py --interactive
""")

if __name__ == "__main__":
    print("BFM Parameter Effect Examples")
    print("This script demonstrates how different parameters affect face rendering")

    # Show parameter guide
    parameter_guide()

    # Run demonstrations
    try:
        demonstrate_expression_effects()
        demonstrate_pose_effects()
        demonstrate_translation_effects()
        demonstrate_focal_effects()

        print("\n" + "="*50)
        print("DEMONSTRATION COMPLETE!")
        print("="*50)
        print("Generated visualization files:")
        print("- expression_effects.png")
        print("- pose_effects.png")
        print("- translation_effects.png")
        print("- focal_effects.png")
        print("\nFor interactive exploration, run:")
        print("python parameter_effect_demo.py --interactive")

    except Exception as e:
        print(f"Error during demonstration: {e}")
        print("Make sure you have matplotlib and required dependencies installed")

