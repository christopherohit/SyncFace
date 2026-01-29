import torch
import math
import sys

# Try importing the package
try:
    from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
    print("[1/3] Import successful.")
except ImportError:
    print("Error: Could not import 'diff_gauss'. Check if the installation finished successfully.")
    sys.exit(1)

def simple_verification():
    # Check for CUDA
    if not torch.cuda.is_available():
        print("Error: CUDA not available.")
        return

    device = torch.device("cuda")
    
    # --- 1. Create Dummy Inputs (1 Gaussian) ---
    print("[2/3] Setting up dummy tensors on GPU...")
    
    # A single Gaussian at the origin
    means3D = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=device, requires_grad=True)
    
    # Scales (log space, so 0 is scale 1.0)
    scales = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    
    # Rotations (quaternion [w, x, y, z], identity)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
    
    # Opacity (sigmoid space, huge value = opaque)
    opacities = torch.tensor([[10.0]], dtype=torch.float32, device=device)
    
    # Spherical Harmonics (Degree 0, just RGB)
    shs = torch.rand((1, 1, 3), dtype=torch.float32, device=device)

    # --- 2. Create Dummy Camera ---
    H, W = 256, 256
    fov_x = math.pi / 2.0
    tan_fovx = math.tan(fov_x * 0.5)
    tan_fovy = math.tan(fov_x * 0.5) # Square aspect
    
    # View Matrix (Identity - Camera at 0,0,0 looking forward)
    # We move the camera back slightly so it sees the point
    view_matrix = torch.eye(4, device=device)
    view_matrix[2, 3] = 2.0 # Translate Z
    
    # Projection Matrix (Simple perspective)
    proj_matrix = view_matrix.clone() # Simplified for test
    
    # Camera Center
    campos = view_matrix[:3, 3]

    # --- 3. Configure Rasterizer ---
    settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tan_fovx,
        tanfovy=tan_fovy,
        bg=torch.tensor([0.0, 0.0, 0.0], device=device),
        scale_modifier=1.0,
        viewmatrix=view_matrix,
        projmatrix=proj_matrix,
        sh_degree=0,
        campos=campos,
        prefiltered=False,
        debug=True # Enable debug to catch CUDA errors early
    )

    rasterizer = GaussianRasterizer(settings)

    # --- 4. Run Forward Pass ---
    print("[3/3] Running forward pass (CUDA execution)...")
    try:
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=torch.zeros_like(means3D),
            shs=shs,
            colors_precomp=None,
            opacities=opacities,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None
        )
        
        # Force a synchronization to catch async CUDA errors
        torch.cuda.synchronize()
        
        if rendered_image.shape == (3, H, W):
            print("\n✅ SUCCESS: Rasterization complete!")
            print(f"   Output Image Shape: {rendered_image.shape}")
            print(f"   Non-zero radii count: {(radii > 0).sum().item()}")
        else:
            print("\n❌ FAILURE: Output shape mismatch.")
            
    except Exception as e:
        print(f"\n❌ CRASH: The CUDA kernel failed to execute.\nError: {e}")

if __name__ == "__main__":
    simple_verification()