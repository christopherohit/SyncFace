#!/usr/bin/env python3
"""
Test script for enhanced audio encoder comparison.

This script helps you quickly compare the original LRS2 encoder
with the enhanced foundation model encoders.

Usage:
    python scripts/test_enhanced_encoder.py --encoder whisper --audio demo/test.wav
    python scripts/test_enhanced_encoder.py --encoder hybrid --audio demo/test.wav
    python scripts/test_enhanced_encoder.py --compare all --audio demo/test.wav
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_encoder(encoder_type, audio_path, output_dir):
    """Test a specific encoder type."""
    print(f"\n{'='*60}")
    print(f"Testing {encoder_type.upper()} encoder")
    print(f"{'='*60}\n")
    
    # Import after path setup
    from nerf_triplane.network import AudioEncoder
    from nerf_triplane.enhanced_audio_encoder import EnhancedAudioEncoder
    from nerf_triplane.audio_encoder_adapter import HybridAudioEncoder
    
    import librosa
    
    # Load audio
    print(f"Loading audio from {audio_path}...")
    audio, sr = librosa.load(audio_path, sr=16000)
    audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
    
    if torch.cuda.is_available():
        audio_tensor = audio_tensor.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    
    print(f"Audio shape: {audio_tensor.shape}, Duration: {len(audio)/sr:.2f}s")
    
    # Initialize encoder
    print(f"\nInitializing {encoder_type} encoder...")
    start_time = time.time()
    
    if encoder_type == 'original':
        encoder = AudioEncoder().to(device)
        # Load checkpoint if available
        ckpt_path = './nerf_triplane/checkpoints/audio_visual_encoder.pth'
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device)
            encoder.load_state_dict({f'audio_encoder.{k}': v for k, v in ckpt.items()})
            print("Loaded pretrained LRS2 checkpoint")
    
    elif encoder_type == 'whisper':
        encoder = EnhancedAudioEncoder(
            encoder_type='whisper',
            output_dim=512,
            use_prosody=True,
            use_contrastive=False,
            freeze_backbone=True
        ).to(device)
    
    elif encoder_type == 'speecht5':
        encoder = EnhancedAudioEncoder(
            encoder_type='speecht5',
            output_dim=512,
            use_prosody=True,
            use_contrastive=False,
            freeze_backbone=True
        ).to(device)
    
    elif encoder_type == 'encodec':
        encoder = EnhancedAudioEncoder(
            encoder_type='encodec',
            output_dim=512,
            use_prosody=True,
            use_contrastive=False,
            freeze_backbone=True
        ).to(device)
    
    elif encoder_type == 'ensemble':
        encoder = EnhancedAudioEncoder(
            encoder_type='ensemble',
            output_dim=512,
            use_prosody=True,
            use_contrastive=False,
            freeze_backbone=True
        ).to(device)
    
    elif encoder_type == 'hybrid':
        encoder = HybridAudioEncoder(
            use_lrs2=True,
            use_foundation=True,
            foundation_type='whisper',
            output_dim=512
        ).to(device)
    
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    init_time = time.time() - start_time
    print(f"Initialization time: {init_time:.2f}s")
    
    # Count parameters
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test inference
    print(f"\nRunning inference...")
    encoder.eval()
    
    # Prepare input
    if encoder_type == 'original':
        # Original encoder expects mel spectrogram [B, 1, H, W]
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=80)
        mel = librosa.power_to_db(mel, ref=np.max)
        mel_tensor = torch.from_numpy(mel).float().unsqueeze(0).unsqueeze(0).to(device)
        input_tensor = mel_tensor
    else:
        # Enhanced encoders can work with raw audio
        input_tensor = audio_tensor
    
    # Warm up
    with torch.no_grad():
        if encoder_type == 'original':
            _ = encoder(input_tensor)
        else:
            _ = encoder(input_tensor)
    
    # Time inference
    num_runs = 5
    inference_times = []
    
    for i in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            if encoder_type == 'original':
                output = encoder(input_tensor)
            else:
                output = encoder(input_tensor)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
        print(f"Run {i+1}/{num_runs}: {inference_time:.4f}s")
    
    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    
    # Extract features
    with torch.no_grad():
        if encoder_type == 'original':
            features = encoder(input_tensor)
        else:
            result = encoder(input_tensor)
            if isinstance(result, dict):
                features = result.get('audio_features_with_prosody', result['audio_features'])
            else:
                features = result
    
    # Analyze features
    print(f"\nFeature Analysis:")
    print(f"  Shape: {features.shape}")
    print(f"  Mean: {features.mean().item():.4f}")
    print(f"  Std: {features.std().item():.4f}")
    print(f"  Min: {features.min().item():.4f}")
    print(f"  Max: {features.max().item():.4f}")
    
    # Check for prosody features if available
    if encoder_type != 'original' and isinstance(result, dict) and 'prosody_features' in result:
        prosody = result['prosody_features']
        print(f"\nProsody Features:")
        print(f"  Shape: {prosody.shape}")
        print(f"  Mean: {prosody.mean().item():.4f}")
        print(f"  Std: {prosody.std().item():.4f}")
    
    # Performance summary
    print(f"\n{'='*60}")
    print(f"SUMMARY - {encoder_type.upper()}")
    print(f"{'='*60}")
    print(f"Initialization: {init_time:.2f}s")
    print(f"Inference (avg): {avg_time:.4f}s ± {std_time:.4f}s")
    print(f"Real-time factor: {avg_time / (len(audio)/sr):.2f}x")
    print(f"Parameters: {total_params:,} ({trainable_params:,} trainable)")
    print(f"Feature dimension: {features.shape[-1]}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    result_file = os.path.join(output_dir, f'{encoder_type}_results.npz')
    np.savez(
        result_file,
        features=features.cpu().numpy(),
        prosody=result.get('prosody_features', None).cpu().numpy() if isinstance(result, dict) and 'prosody_features' in result else None,
        inference_time=avg_time,
        total_params=total_params,
        trainable_params=trainable_params
    )
    print(f"\nResults saved to {result_file}")
    
    return {
        'encoder': encoder_type,
        'init_time': init_time,
        'inference_time': avg_time,
        'inference_std': std_time,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'feature_dim': features.shape[-1],
        'rtf': avg_time / (len(audio)/sr)
    }


def compare_encoders(encoders, audio_path, output_dir):
    """Compare multiple encoders."""
    results = []
    
    for encoder_type in encoders:
        try:
            result = test_encoder(encoder_type, audio_path, output_dir)
            results.append(result)
        except Exception as e:
            print(f"\n[ERROR] Failed to test {encoder_type}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPARISON TABLE")
    print(f"{'='*80}\n")
    
    print(f"{'Encoder':<15} {'Init(s)':<10} {'Infer(s)':<12} {'RTF':<8} {'Params':<12} {'Dim':<6}")
    print(f"{'-'*80}")
    
    for r in results:
        print(f"{r['encoder']:<15} {r['init_time']:<10.2f} {r['inference_time']:<12.4f} "
              f"{r['rtf']:<8.2f} {r['total_params']:<12,} {r['feature_dim']:<6}")
    
    print(f"\n{'='*80}")
    print("Legend:")
    print("  Init(s): Initialization time in seconds")
    print("  Infer(s): Average inference time in seconds")
    print("  RTF: Real-time factor (lower is better)")
    print("  Params: Total number of parameters")
    print("  Dim: Feature dimension")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Test enhanced audio encoders')
    parser.add_argument('--encoder', type=str, default='whisper',
                       choices=['original', 'whisper', 'speecht5', 'encodec', 'ensemble', 'hybrid'],
                       help='Encoder type to test')
    parser.add_argument('--compare', type=str, default=None,
                       help='Compare multiple encoders: "all", "foundation", or comma-separated list')
    parser.add_argument('--audio', type=str, default='demo/test.wav',
                       help='Path to test audio file')
    parser.add_argument('--output', type=str, default='output/encoder_test',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Check audio file exists
    if not os.path.exists(args.audio):
        print(f"[ERROR] Audio file not found: {args.audio}")
        sys.exit(1)
    
    # Determine which encoders to test
    if args.compare:
        if args.compare == 'all':
            encoders = ['original', 'whisper', 'speecht5', 'encodec', 'hybrid']
        elif args.compare == 'foundation':
            encoders = ['whisper', 'speecht5', 'encodec']
        else:
            encoders = [e.strip() for e in args.compare.split(',')]
        
        compare_encoders(encoders, args.audio, args.output)
    else:
        test_encoder(args.encoder, args.audio, args.output)


if __name__ == '__main__':
    main()


