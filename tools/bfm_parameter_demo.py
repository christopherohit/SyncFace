#!/usr/bin/env python3
"""
BFM Parameter Demonstration - Interactive parameter explanation
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, Polygon
import math

def create_base_face(ax, x_offset=0, y_offset=0, scale=1.0):
    """Create a simple base face outline"""
    # Face oval
    face_center = (5 + x_offset, 5 + y_offset)
    face_width = 3 * scale
    face_height = 4 * scale

    face_oval = patches.Ellipse(face_center, face_width, face_height,
                               fill=False, color='black', linewidth=2)
    ax.add_patch(face_oval)

    # Eyes
    left_eye = Circle((face_center[0] - 0.8 * scale, face_center[1] + 0.5 * scale),
                     0.3 * scale, fill=True, color='black', alpha=0.7)
    right_eye = Circle((face_center[0] + 0.8 * scale, face_center[1] + 0.5 * scale),
                      0.3 * scale, fill=True, color='black', alpha=0.7)
    ax.add_patch(left_eye)
    ax.add_patch(right_eye)

    # Nose
    nose_points = [
        (face_center[0], face_center[1] + 0.2 * scale),
        (face_center[0] - 0.2 * scale, face_center[1] - 0.3 * scale),
        (face_center[0] + 0.2 * scale, face_center[1] - 0.3 * scale)
    ]
    nose = Polygon(nose_points, fill=False, color='black', linewidth=2)
    ax.add_patch(nose)

    # Mouth
    mouth = patches.Arc((face_center[0], face_center[1] - 0.8 * scale),
                       1.2 * scale, 0.8 * scale, angle=0, theta1=0, theta2=180,
                       fill=False, color='black', linewidth=2)
    ax.add_patch(mouth)

    return face_center, face_width, face_height

def demonstrate_identity_parameters():
    """Show how identity parameters affect face shape"""
    print("\n" + "="*60)
    print("IDENTITY PARAMETERS - Facial Structure")
    print("="*60)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Identity Parameters: How They Shape Facial Structure', fontsize=16, fontweight='bold')

    variations = [
        {'name': 'Neutral Face', 'width_scale': 1.0, 'height_scale': 1.0, 'eye_spacing': 1.0, 'chin_shape': 'normal'},
        {'name': 'Wide Face\n(Identity coeff +2)', 'width_scale': 1.4, 'height_scale': 1.0, 'eye_spacing': 1.3, 'chin_shape': 'normal'},
        {'name': 'Narrow Face\n(Identity coeff -2)', 'width_scale': 0.7, 'height_scale': 1.0, 'eye_spacing': 0.8, 'chin_shape': 'normal'},
        {'name': 'Round Face\n(Identity coeff +1.5)', 'width_scale': 1.2, 'height_scale': 0.8, 'eye_spacing': 1.1, 'chin_shape': 'round'},
        {'name': 'Long Face\n(Identity coeff -1)', 'width_scale': 0.9, 'height_scale': 1.3, 'eye_spacing': 0.95, 'chin_shape': 'pointed'},
        {'name': 'Square Jaw\n(Identity coeff +1)', 'width_scale': 1.1, 'height_scale': 1.0, 'eye_spacing': 1.0, 'chin_shape': 'square'}
    ]

    for i, (ax, var) in enumerate(zip(axes.flat, variations)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(var['name'], fontsize=12, fontweight='bold')

        # Create custom face based on parameters
        face_center = (5, 5)
        face_width = 3 * var['width_scale']
        face_height = 4 * var['height_scale']

        # Face shape
        if var['chin_shape'] == 'round':
            face_oval = patches.Ellipse(face_center, face_width, face_height,
                                       fill=False, color='blue', linewidth=3)
        elif var['chin_shape'] == 'square':
            face_oval = patches.Rectangle((face_center[0] - face_width/2, face_center[1] - face_height/2),
                                         face_width, face_height, fill=False, color='green', linewidth=3)
        else:
            face_oval = patches.Ellipse(face_center, face_width, face_height,
                                       fill=False, color='black', linewidth=2)
        ax.add_patch(face_oval)

        # Eyes with adjusted spacing
        eye_spacing = var['eye_spacing']
        left_eye = Circle((face_center[0] - 0.8 * eye_spacing, face_center[1] + 0.5),
                         0.3, fill=True, color='black', alpha=0.7)
        right_eye = Circle((face_center[0] + 0.8 * eye_spacing, face_center[1] + 0.5),
                          0.3, fill=True, color='black', alpha=0.7)
        ax.add_patch(left_eye)
        ax.add_patch(right_eye)

        # Nose
        nose = patches.Ellipse((face_center[0], face_center[1] - 0.1), 0.3, 0.5,
                              fill=False, color='black', linewidth=2)
        ax.add_patch(nose)

        # Mouth
        mouth = patches.Arc((face_center[0], face_center[1] - 0.8),
                           1.2, 0.8, angle=0, theta1=0, theta2=180,
                           fill=False, color='black', linewidth=2)
        ax.add_patch(mouth)

    plt.tight_layout()
    plt.savefig('bfm_identity_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Identity parameters control permanent facial features:")
    print("• Face width/narrowness")
    print("• Eye spacing and size")
    print("• Nose shape and prominence")
    print("• Jawline and chin shape")
    print("• Cheekbone prominence")
    print("• Overall facial proportions")

def demonstrate_expression_parameters():
    """Show how expression parameters affect facial expressions"""
    print("\n" + "="*60)
    print("EXPRESSION PARAMETERS - Facial Movements")
    print("="*60)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Expression Parameters: How They Create Emotions', fontsize=16, fontweight='bold')

    expressions = [
        {'name': 'Neutral', 'eyebrows': 'normal', 'mouth': 'closed', 'eyes': 'normal'},
        {'name': 'Happy/Smile\n(Exp coeffs +0.8, +0.6)', 'eyebrows': 'raised', 'mouth': 'smile', 'eyes': 'normal'},
        {'name': 'Surprised\n(Exp coeffs +1.2, +0.8)', 'eyebrows': 'raised', 'mouth': 'open', 'eyes': 'wide'},
        {'name': 'Sad\n(Exp coeffs -0.5, -0.8)', 'eyebrows': 'lowered', 'mouth': 'frown', 'eyes': 'normal'},
        {'name': 'Angry\n(Exp coeffs -0.8, -0.6)', 'eyebrows': 'lowered', 'mouth': 'tight', 'eyes': 'narrowed'},
        {'name': 'Confused\n(Mixed coeffs)', 'eyebrows': 'asymmetrical', 'mouth': 'slight_open', 'eyes': 'normal'}
    ]

    for i, (ax, expr) in enumerate(zip(axes.flat, expressions)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(expr['name'], fontsize=12, fontweight='bold')

        face_center = (5, 5)
        face_width, face_height = 3, 4

        # Base face
        face_oval = patches.Ellipse(face_center, face_width, face_height,
                                   fill=False, color='black', linewidth=2)
        ax.add_patch(face_oval)

        # Eyes
        eye_y = face_center[1] + 0.5
        if expr['eyes'] == 'wide':
            eye_size = 0.4
        elif expr['eyes'] == 'narrowed':
            eye_size = 0.2
        else:
            eye_size = 0.3

        left_eye = Circle((face_center[0] - 0.8, eye_y), eye_size,
                         fill=True, color='black', alpha=0.7)
        right_eye = Circle((face_center[0] + 0.8, eye_y), eye_size,
                          fill=True, color='black', alpha=0.7)
        ax.add_patch(left_eye)
        ax.add_patch(right_eye)

        # Eyebrows
        brow_y = eye_y + 0.3
        brow_length = 0.6

        if expr['eyebrows'] == 'raised':
            brow_offset = 0.2
            color = 'red'
        elif expr['eyebrows'] == 'lowered':
            brow_offset = -0.2
            color = 'blue'
        elif expr['eyebrows'] == 'asymmetrical':
            # Left raised, right lowered
            left_brow = patches.Rectangle((face_center[0] - 1.1, brow_y + 0.2),
                                         brow_length, 0.1, fill=True, color='red', alpha=0.7)
            right_brow = patches.Rectangle((face_center[0] + 0.5, brow_y - 0.2),
                                          brow_length, 0.1, fill=True, color='blue', alpha=0.7)
            ax.add_patch(left_brow)
            ax.add_patch(right_brow)
        else:
            brow_offset = 0
            color = 'black'

        if expr['eyebrows'] != 'asymmetrical':
            left_brow = patches.Rectangle((face_center[0] - 1.1, brow_y + brow_offset),
                                         brow_length, 0.1, fill=True, color=color, alpha=0.7)
            right_brow = patches.Rectangle((face_center[0] + 0.5, brow_y + brow_offset),
                                          brow_length, 0.1, fill=True, color=color, alpha=0.7)
            ax.add_patch(left_brow)
            ax.add_patch(right_brow)

        # Mouth
        mouth_y = face_center[1] - 0.8
        if expr['mouth'] == 'smile':
            mouth = patches.Arc((face_center[0], mouth_y), 1.2, 0.8,
                               angle=0, theta1=0, theta2=180, fill=False, color='red', linewidth=3)
        elif expr['mouth'] == 'open':
            mouth = patches.Ellipse((face_center[0], mouth_y), 0.8, 0.6,
                                   fill=True, color='orange', alpha=0.5)
        elif expr['mouth'] == 'frown':
            mouth = patches.Arc((face_center[0], mouth_y), 1.2, 0.8,
                               angle=180, theta1=0, theta2=180, fill=False, color='blue', linewidth=3)
        elif expr['mouth'] == 'tight':
            mouth = patches.Ellipse((face_center[0], mouth_y), 1.0, 0.3,
                                   fill=False, color='purple', linewidth=3)
        else:  # closed/slight_open
            mouth_width = 0.8 if expr['mouth'] == 'slight_open' else 0.6
            mouth = patches.Ellipse((face_center[0], mouth_y), mouth_width, 0.2,
                                   fill=True, color='black', alpha=0.7)
        ax.add_patch(mouth)

        # Nose
        nose = patches.Ellipse((face_center[0], face_center[1] - 0.1), 0.3, 0.5,
                              fill=False, color='black', linewidth=2)
        ax.add_patch(nose)

    plt.tight_layout()
    plt.savefig('bfm_expression_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Expression parameters control temporary facial movements:")
    print("• Eyebrow positioning (raised/lowered)")
    print("• Eye opening/closing")
    print("• Mouth shape and opening")
    print("• Cheek puffing")
    print("• Nose wrinkling")
    print("• Jaw dropping")

def demonstrate_pose_parameters():
    """Show how pose parameters affect head orientation"""
    print("\n" + "="*60)
    print("POSE PARAMETERS - Head Orientation")
    print("="*60)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Pose Parameters: How They Orient the Head', fontsize=16, fontweight='bold')

    poses = [
        {'name': 'Front View\n(Euler: 0°, 0°, 0°)', 'pitch': 0, 'yaw': 0, 'roll': 0},
        {'name': 'Left Turn\n(Euler: 0°, +30°, 0°)', 'pitch': 0, 'yaw': 30, 'roll': 0},
        {'name': 'Right Turn\n(Euler: 0°, -30°, 0°)', 'pitch': 0, 'yaw': -30, 'roll': 0},
        {'name': 'Looking Up\n(Euler: +20°, 0°, 0°)', 'pitch': 20, 'yaw': 0, 'roll': 0},
        {'name': 'Looking Down\n(Euler: -20°, 0°, 0°)', 'pitch': -20, 'yaw': 0, 'roll': 0},
        {'name': 'Tilted Head\n(Euler: +10°, +15°, +10°)', 'pitch': 10, 'yaw': 15, 'roll': 10}
    ]

    for i, (ax, pose) in enumerate(zip(axes.flat, poses)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')

        # Create coordinate system hints
        ax.arrow(1, 1, 2, 0, head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.5, label='X')
        ax.arrow(1, 1, 0, 2, head_width=0.1, head_length=0.1, fc='green', ec='green', alpha=0.5, label='Y')
        ax.arrow(1, 1, 1.4, 1.4, head_width=0.1, head_length=0.1, fc='blue', ec='blue', alpha=0.5, label='Z')

        ax.text(3.2, 0.8, 'X (Pitch)', color='red', fontsize=8, alpha=0.7)
        ax.text(0.8, 3.2, 'Y (Yaw)', color='green', fontsize=8, alpha=0.7)
        ax.text(2.5, 2.5, 'Z (Roll)', color='blue', fontsize=8, alpha=0.7)

        # Apply pose transformation
        face_center = (5, 5)
        pitch_rad = math.radians(pose['pitch'])
        yaw_rad = math.radians(pose['yaw'])
        roll_rad = math.radians(pose['roll'])

        # Simple 2D rotation approximation for visualization
        total_rotation = yaw_rad  # Dominant effect for 2D view

        # Create rotated face
        face_width, face_height = 3, 4

        # Face oval (rotated)
        face_oval = patches.Ellipse(face_center, face_width, face_height, angle=math.degrees(total_rotation),
                                   fill=False, color='black', linewidth=2)
        ax.add_patch(face_oval)

        # Rotate eye positions
        cos_rot = math.cos(total_rotation)
        sin_rot = math.sin(total_rotation)

        def rotate_point(x, y, cx, cy):
            dx = x - cx
            dy = y - cy
            new_x = cx + dx * cos_rot - dy * sin_rot
            new_y = cy + dx * sin_rot + dy * cos_rot
            return new_x, new_y

        left_eye_x, left_eye_y = rotate_point(face_center[0] - 0.8, face_center[1] + 0.5, *face_center)
        right_eye_x, right_eye_y = rotate_point(face_center[0] + 0.8, face_center[1] + 0.5, *face_center)

        left_eye = Circle((left_eye_x, left_eye_y), 0.3, fill=True, color='black', alpha=0.7)
        right_eye = Circle((right_eye_x, right_eye_y), 0.3, fill=True, color='black', alpha=0.7)
        ax.add_patch(left_eye)
        ax.add_patch(right_eye)

        # Nose
        nose_x, nose_y = rotate_point(face_center[0], face_center[1] - 0.1, *face_center)
        nose = patches.Ellipse((nose_x, nose_y), 0.3, 0.5, angle=math.degrees(total_rotation),
                              fill=False, color='black', linewidth=2)
        ax.add_patch(nose)

        # Mouth
        mouth_x, mouth_y = rotate_point(face_center[0], face_center[1] - 0.8, *face_center)
        mouth = patches.Arc((mouth_x, mouth_y), 1.2, 0.8, angle=math.degrees(total_rotation),
                           theta1=0, theta2=180, fill=False, color='black', linewidth=2)
        ax.add_patch(mouth)

        ax.set_title(pose['name'], fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig('bfm_pose_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Pose parameters control head orientation in 3D space:")
    print("• Pitch (X): Up/down head tilt")
    print("• Yaw (Y): Left/right head rotation")
    print("• Roll (Z): Head tilt side-to-side")
    print("• Measured in radians or degrees")
    print("• Affects how the face appears from different angles")

def demonstrate_translation_focal():
    """Show translation and focal length effects"""
    print("\n" + "="*60)
    print("TRANSLATION & FOCAL LENGTH - Position & Perspective")
    print("="*60)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Translation & Focal Length: Position and Perspective Effects', fontsize=16, fontweight='bold')

    effects = [
        {'name': 'Center Position\n(Trans: 0, 0, -600)', 'x_offset': 0, 'y_offset': 0, 'scale': 1.0},
        {'name': 'Moved Right\n(Trans: +50, 0, -600)', 'x_offset': 0.5, 'y_offset': 0, 'scale': 1.0},
        {'name': 'Moved Left\n(Trans: -50, 0, -600)', 'x_offset': -0.5, 'y_offset': 0, 'scale': 1.0},
        {'name': 'Closer to Camera\n(Trans: 0, 0, -400)', 'x_offset': 0, 'y_offset': 0, 'scale': 1.3},
        {'name': 'Farther from Camera\n(Trans: 0, 0, -800)', 'x_offset': 0, 'y_offset': 0, 'scale': 0.7},
        {'name': 'Wide Angle Lens\n(Focal: 800)', 'x_offset': 0, 'y_offset': 0, 'scale': 0.9, 'distortion': 'barrel'},
    ]

    for i, (ax, effect) in enumerate(zip(axes.flat, effects)):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(effect['name'], fontsize=12, fontweight='bold')

        # Camera frame
        camera_frame = patches.Rectangle((0.5, 0.5), 9, 9, fill=False, color='gray', linestyle='--', alpha=0.5)
        ax.add_patch(camera_frame)
        ax.text(1, 9.2, 'Camera View', fontsize=10, color='gray', alpha=0.7)

        # Create face with translation and scaling
        face_center = (5 + effect['x_offset'], 5 + effect['y_offset'])
        face_width = 3 * effect['scale']
        face_height = 4 * effect['scale']

        # Apply perspective distortion hint
        if 'distortion' in effect and effect['distortion'] == 'barrel':
            # Simulate barrel distortion
            distortion_factor = 1.2
            face_oval = patches.Ellipse(face_center, face_width * distortion_factor,
                                       face_height, fill=False, color='purple', linewidth=2)
        else:
            face_oval = patches.Ellipse(face_center, face_width, face_height,
                                       fill=False, color='black', linewidth=2)
        ax.add_patch(face_oval)

        # Eyes
        eye_y = face_center[1] + 0.5 * effect['scale']
        left_eye = Circle((face_center[0] - 0.8 * effect['scale'], eye_y),
                         0.3 * effect['scale'], fill=True, color='black', alpha=0.7)
        right_eye = Circle((face_center[0] + 0.8 * effect['scale'], eye_y),
                          0.3 * effect['scale'], fill=True, color='black', alpha=0.7)
        ax.add_patch(left_eye)
        ax.add_patch(right_eye)

        # Nose
        nose = patches.Ellipse((face_center[0], face_center[1] - 0.1 * effect['scale']),
                              0.3 * effect['scale'], 0.5 * effect['scale'],
                              fill=False, color='black', linewidth=2)
        ax.add_patch(nose)

        # Mouth
        mouth = patches.Arc((face_center[0], face_center[1] - 0.8 * effect['scale']),
                           1.2 * effect['scale'], 0.8 * effect['scale'], angle=0, theta1=0, theta2=180,
                           fill=False, color='black', linewidth=2)
        ax.add_patch(mouth)

    plt.tight_layout()
    plt.savefig('bfm_translation_focal_demo.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("Translation parameters control position:")
    print("• X, Y: Horizontal/vertical position in camera frame")
    print("• Z: Distance from camera (negative = farther away)")
    print("• Measured in millimeters or world units")
    print("\nFocal length controls perspective:")
    print("• Higher focal length = less distortion (telephoto)")
    print("• Lower focal length = more distortion (wide angle)")
    print("• Measured in pixels or millimeters")

def create_parameter_summary():
    """Create a comprehensive parameter summary"""
    print("\n" + "="*80)
    print("BFM PARAMETER SUMMARY")
    print("="*80)

    summary_data = {
        'Parameter Group': ['Identity (100 coeffs)', 'Expression (79 coeffs)', 'Pose (3 Euler angles)', 'Translation (3 values)', 'Focal Length (1 value)'],
        'What it controls': [
            'Permanent facial structure',
            'Temporary facial movements',
            'Head orientation in 3D',
            'Head position in space',
            'Camera perspective'
        ],
        'Typical range': [
            '-3.0 to +3.0',
            '-2.0 to +2.0',
            '-π/2 to +π/2 radians',
            '-∞ to +∞ (units)',
            '500-2000 pixels'
        ],
        'Frame variation': [
            'Stable across video',
            'Changes every frame',
            'Smooth head movement',
            'Smooth position change',
            'Usually constant'
        ],
        'Visual effect': [
            'Face shape, features',
            'Emotions, speech',
            'Viewing angle',
            'Position in frame',
            'Perspective distortion'
        ]
    }

    print(f"{'Parameter Group':<25} {'Controls':<25} {'Range':<20} {'Variation':<20} {'Effect':<20}")
    print("-" * 110)

    for i in range(len(summary_data['Parameter Group'])):
        print(f"{summary_data['Parameter Group'][i]:<25} "
              f"{summary_data['What it controls'][i]:<25} "
              f"{summary_data['Typical range'][i]:<20} "
              f"{summary_data['Frame variation'][i]:<20} "
              f"{summary_data['Visual effect'][i]:<20}")

    print("\n" + "="*80)

if __name__ == "__main__":
    print("BFM Parameter Visualization Demo")
    print("This script creates visual demonstrations of how each BFM parameter affects face appearance")

    # Run all demonstrations
    demonstrate_identity_parameters()
    demonstrate_expression_parameters()
    demonstrate_pose_parameters()
    demonstrate_translation_focal()
    create_parameter_summary()

    print("\nDemo complete! Check the generated PNG files:")
    print("- bfm_identity_demo.png")
    print("- bfm_expression_demo.png")
    print("- bfm_pose_demo.png")
    print("- bfm_translation_focal_demo.png")
    print("\nThese visualizations show how BFM parameters create different facial appearances!")

