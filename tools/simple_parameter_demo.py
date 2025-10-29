#!/usr/bin/env python3
"""
Simple BFM Parameter Modification Demo
Shows how to modify exp, euler, trans, and focal parameters and understand their effects
"""

import torch
import numpy as np
import os

def create_sample_parameters():
    """Create sample BFM parameters for demonstration"""

    # Create realistic sample parameters
    sample_params = {
        'id': torch.randn(1, 100) * 0.1,      # Identity parameters (stable)
        'exp': torch.zeros(5, 79),             # Expression parameters (per frame)
        'euler': torch.zeros(5, 3),            # Euler angles (pitch, yaw, roll)
        'trans': torch.zeros(5, 3),            # Translation (x, y, z)
        'focal': torch.tensor(1015.0)          # Focal length
    }

    # Set realistic default values
    sample_params['trans'][:, 2] = -600  # Default distance from camera
    sample_params['focal'] = torch.tensor(1015.0)  # Default focal length

    # Add some variation to make demonstrations interesting
    sample_params['exp'][0, 0] = 0.5     # Slight smile on first frame
    sample_params['euler'][1, 1] = 0.2   # Slight head turn on second frame
    sample_params['trans'][2, 0] = 30    # Slight movement on third frame

    return sample_params

def demonstrate_parameter_modification():
    """Show how to modify different parameters and their effects"""

    print("BFM Parameter Modification Demo")
    print("=" * 50)

    # Create sample parameters
    params = create_sample_parameters()

    print(f"Original parameters shape:")
    print(f"  Identity: {params['id'].shape}")
    print(f"  Expression: {params['exp'].shape}")
    print(f"  Euler angles: {params['euler'].shape}")
    print(f"  Translation: {params['trans'].shape}")
    print(f"  Focal length: {params['focal']}")

    # Demonstrate expression parameter modification
    print("\n1. EXPRESSION PARAMETERS (exp)")
    print("-" * 30)
    print("Expression parameters control facial movements and emotions")
    print("Each of the 79 coefficients controls different facial muscles")
    print("Typical range: -2.0 to +2.0")

    # Show original expression
    original_exp = params['exp'][0].clone()
    print(f"Original expression (first 5 coeffs): {original_exp[:5].tolist()[:5]}")

    # Modify expression to create a smile
    smile_exp = original_exp.clone()
    smile_exp[0] = 1.0  # Often controls mouth opening/smile
    print(f"Smile expression (first 5 coeffs): {smile_exp[:5].tolist()[:5]}")
    print("Effect: Face appears to be smiling")

    # Modify expression to create surprise
    surprise_exp = original_exp.clone()
    surprise_exp[2] = 1.2  # Often controls eyebrow raise
    surprise_exp[0] = 0.8  # Mouth slightly open
    print(f"Surprise expression (first 5 coeffs): {surprise_exp[:5].tolist()[:5]}")
    print("Effect: Eyebrows raised, mouth slightly open")

    # Demonstrate pose parameter modification
    print("\n2. POSE PARAMETERS (euler)")
    print("-" * 30)
    print("Euler angles control head orientation in 3D space")
    print("euler[0] = pitch (up/down tilt)")
    print("euler[1] = yaw (left/right rotation)")
    print("euler[2] = roll (side-to-side tilt)")
    print("Units: radians, typical range: -π/2 to +π/2")

    original_euler = params['euler'][0].clone()
    print(f"Original pose: {original_euler.tolist()}")

    # Turn head right
    turn_right = original_euler.clone()
    turn_right[1] = 0.5  # Yaw
    print(f"Turn head right: {turn_right.tolist()}")
    print("Effect: Head rotates to the right")

    # Look up
    look_up = original_euler.clone()
    look_up[0] = 0.3  # Pitch
    print(f"Look up: {look_up.tolist()}")
    print("Effect: Head tilts upward")

    # Tilt head
    tilt_head = original_euler.clone()
    tilt_head[2] = 0.2  # Roll
    print(f"Tilt head: {tilt_head.tolist()}")
    print("Effect: Head tilts to the side")

    # Demonstrate translation parameter modification
    print("\n3. TRANSLATION PARAMETERS (trans)")
    print("-" * 30)
    print("Translation controls head position in camera space")
    print("trans[0] = X (left/right position)")
    print("trans[1] = Y (up/down position)")
    print("trans[2] = Z (distance from camera)")
    print("Units: typically millimeters")

    original_trans = params['trans'][0].clone()
    print(f"Original position: {original_trans.tolist()}")

    # Move face right
    move_right = original_trans.clone()
    move_right[0] = 50.0
    print(f"Move right: {move_right.tolist()}")
    print("Effect: Face moves to the right in the frame")

    # Move face closer
    move_closer = original_trans.clone()
    move_closer[2] = -500.0  # Less negative = closer
    print(f"Move closer: {move_closer.tolist()}")
    print("Effect: Face appears larger (closer to camera)")

    # Move face up
    move_up = original_trans.clone()
    move_up[1] = 30.0
    print(f"Move up: {move_up.tolist()}")
    print("Effect: Face moves upward in the frame")

    # Demonstrate focal length modification
    print("\n4. FOCAL LENGTH PARAMETER (focal)")
    print("-" * 30)
    print("Focal length controls camera perspective")
    print("Higher focal = telephoto lens (less distortion)")
    print("Lower focal = wide angle lens (more distortion)")
    print("Typical range: 500-2000 pixels")

    original_focal = params['focal']
    print(f"Original focal length: {original_focal}")

    # Wide angle
    wide_angle = original_focal - 300
    print(f"Wide angle lens: {wide_angle}")
    print("Effect: More perspective distortion, face appears more rounded")

    # Telephoto
    telephoto = original_focal + 300
    print(f"Telephoto lens: {telephoto}")
    print("Effect: Less perspective distortion, more natural appearance")

def show_modification_examples():
    """Show practical examples of parameter combinations"""

    print("\n5. PRACTICAL MODIFICATION EXAMPLES")
    print("-" * 40)

    params = create_sample_parameters()

    examples = [
        {
            "name": "Happy and Turned",
            "description": "Face smiling and looking to the side",
            "modifications": {
                "exp[0]": 1.0,      # Smile
                "euler[1]": 0.3,    # Turn head right
            }
        },
        {
            "name": "Surprised and Close",
            "description": "Surprised expression, face moved closer",
            "modifications": {
                "exp[2]": 1.0,      # Eyebrow raise
                "exp[0]": 0.5,      # Mouth open
                "trans[2]": -400,   # Closer
            }
        },
        {
            "name": "Sad and Looking Down",
            "description": "Sad expression with downward gaze",
            "modifications": {
                "exp[1]": -0.8,     # Frown
                "euler[0]": -0.2,   # Look down
            }
        },
        {
            "name": "Angry Profile",
            "description": "Angry expression in profile view",
            "modifications": {
                "exp[3]": -0.6,     # Angry eyebrows
                "euler[1]": 0.8,    # Strong turn
                "euler[2]": 0.1,    # Slight tilt
            }
        },
        {
            "name": "Wide Angle Portrait",
            "description": "Face with wide angle perspective distortion",
            "modifications": {
                "focal": 700,       # Wide angle
                "trans[2]": -550,   # Slightly closer
            }
        }
    ]

    for example in examples:
        print(f"\n{example['name']}:")
        print(f"  {example['description']}")
        print("  Modifications:")
        for mod, value in example['modifications'].items():
            print(f"    {mod} = {value}")

def interactive_parameter_guide():
    """Show how to use the interactive modification interface"""

    print("\n6. INTERACTIVE MODIFICATION GUIDE")
    print("-" * 40)
    print("For interactive exploration, use the parameter_effect_demo.py script:")
    print("python parameter_effect_demo.py --interactive")
    print("\nInteractive commands:")
    print("• exp=0:1.0,5:0.5          - Set expression coeffs 0 and 5")
    print("• euler=0,0.3,0            - Turn head up (pitch)")
    print("• euler=0.2,0.5,0.1        - Complex head pose")
    print("• trans=50,20,-550         - Move position")
    print("• focal=800                - Change focal length")
    print("• exp=0:2.0;euler=0,0.8,0 - Multiple modifications")
    print("• quit                     - Exit interactive mode")

def save_parameter_summary():
    """Save a summary of parameter effects to a file"""

    summary = """
BFM PARAMETER MODIFICATION GUIDE
=================================

This guide shows how to modify Basel Face Model (BFM) parameters to understand their effects.

PARAMETERS:
-----------

1. Expression (exp) - 79 coefficients, range: -2.0 to +2.0
   - Controls facial muscles and expressions
   - exp[0]: Often mouth opening/smile
   - exp[1]: Often sadness/frown
   - exp[2]: Often surprise/eyebrow raise

2. Euler Angles (euler) - 3 values in radians, range: -π/2 to +π/2
   - euler[0]: Pitch (up/down head tilt)
   - euler[1]: Yaw (left/right head rotation)
   - euler[2]: Roll (side-to-side head tilt)

3. Translation (trans) - 3 values in world units
   - trans[0]: X position (left/right)
   - trans[1]: Y position (up/down)
   - trans[2]: Z position (distance, negative = farther)

4. Focal Length (focal) - 1 value, range: ~500-2000
   - Controls camera perspective
   - Lower = wide angle (more distortion)
   - Higher = telephoto (less distortion)

MODIFICATION EXAMPLES:
---------------------

Happy Face: exp=0:1.0
Sad Face: exp=1:-1.0
Surprised: exp=2:1.2,0:0.5
Turn Right: euler=0,0.5,0
Look Up: euler=0.3,0,0
Move Closer: trans=0,0,-500
Wide Angle: focal=700
Telephoto: focal=1300

COMBINED EFFECTS:
----------------

Happy + Turned: exp=0:1.0;euler=0,0.3,0
Surprised + Close: exp=2:1.0,0:0.5;trans=0,0,-400
Sad + Down: exp=1:-0.8;euler=-0.2,0,0

The effects are additive and can be combined for complex expressions.
"""

    with open('parameter_modification_guide.txt', 'w') as f:
        f.write(summary)

    print("\nParameter guide saved to: parameter_modification_guide.txt")

if __name__ == "__main__":
    demonstrate_parameter_modification()
    show_modification_examples()
    interactive_parameter_guide()
    save_parameter_summary()

    print("\n" + "="*50)
    print("DEMO COMPLETE!")
    print("="*50)
    print("To run interactive parameter modification:")
    print("python parameter_effect_demo.py --interactive")
    print("\nGenerated file:")
    print("- parameter_modification_guide.txt")
