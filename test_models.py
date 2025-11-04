#!/usr/bin/env python3

import sys
import os

# Add the backend path to Python path
backend_path = "/mnt/2T/nhanhuynh/project/SynthFace/real-video-enhancer/backend"
sys.path.insert(0, backend_path)

try:
    # Test loading the upscale model
    print("Testing upscale model loading...")
    from src.pytorch.spandrel import ModelLoader
    model_loader = ModelLoader()
    upscale_model = model_loader.load_from_file("real-video-enhancer/models/2x_ModernSpanimationV2.pth")
    print(f"Upscale model loaded successfully: {upscale_model}")

    # Test loading the interpolation model
    print("Testing interpolation model loading...")
    import torch
    interpolate_state = torch.load("real-video-enhancer/models/rife4.25.pkl", map_location="cpu")
    print(f"Interpolation model loaded successfully: {type(interpolate_state)}")

    print("All models loaded successfully!")

except Exception as e:
    print(f"Error loading models: {e}")
    import traceback
    traceback.print_exc()
