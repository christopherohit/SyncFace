#!/usr/bin/env python3
"""
Simple training script for Macron with enhanced features
No complex documentation - just training code
"""

import sys
import os

# Configuration
DATA_PATH = "data/Macron"
WORKSPACE = "output/Macron_enhanced"
ITERS = 100000
ASR_MODEL = "ave"

def train_basic():
    """Train with basic setup"""
    cmd = f"""python main.py {DATA_PATH} \
        --workspace {WORKSPACE}_basic \
        --asr_model {ASR_MODEL} \
        --iters {ITERS} \
        -O"""
    os.system(cmd)

def train_enhanced():
    """Train with enhanced audio encoder"""
    cmd = f"""python main.py {DATA_PATH} \
        --workspace {WORKSPACE}_whisper \
        --use_enhanced_encoder \
        --enhanced_encoder_type whisper \
        --use_prosody \
        --freeze_audio_backbone \
        --asr_model {ASR_MODEL} \
        --iters {ITERS} \
        -O"""
    os.system(cmd)

def train_hybrid():
    """Train with hybrid mode (recommended)"""
    cmd = f"""python main.py {DATA_PATH} \
        --workspace {WORKSPACE}_hybrid \
        --use_enhanced_encoder \
        --enhanced_encoder_type hybrid \
        --use_prosody \
        --freeze_audio_backbone \
        --asr_model {ASR_MODEL} \
        --iters {ITERS} \
        -O"""
    os.system(cmd)

def train_quick_test():
    """Quick test with 10k iterations"""
    cmd = f"""python main.py {DATA_PATH} \
        --workspace {WORKSPACE}_test \
        --use_enhanced_encoder \
        --enhanced_encoder_type whisper \
        --use_prosody \
        --freeze_audio_backbone \
        --asr_model {ASR_MODEL} \
        --iters 10000 \
        -O"""
    os.system(cmd)

if __name__ == '__main__':
    print("=" * 60)
    print("SyncTalk Training - Macron")
    print("=" * 60)
    print("\nSelect training mode:")
    print("1. Basic (original)")
    print("2. Enhanced (Whisper + Prosody)")
    print("3. Hybrid (LRS2 + Whisper) [RECOMMENDED]")
    print("4. Quick Test (10k iterations)")
    print()
    
    choice = input("Enter choice [1-4]: ").strip()
    
    if choice == '1':
        print("\n[INFO] Training with basic setup...")
        train_basic()
    elif choice == '2':
        print("\n[INFO] Training with enhanced audio encoder...")
        train_enhanced()
    elif choice == '3':
        print("\n[INFO] Training with hybrid mode (recommended)...")
        train_hybrid()
    elif choice == '4':
        print("\n[INFO] Running quick test...")
        train_quick_test()
    else:
        print("[ERROR] Invalid choice")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("[INFO] Training completed!")
    print("=" * 60)

