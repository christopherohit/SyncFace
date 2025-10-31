#!/usr/bin/env python3
"""
SyncFace Setup and Run Script
Interactive setup for beginners
"""

import os
import sys
import subprocess
import shutil

def run_command(cmd, description=""):
    """Run shell command with error handling."""
    print(f"\n[EXECUTING] {description}")
    print(f"[COMMAND] {cmd}")
    print("-" * 60)

    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Command failed: {e}")
        if e.stderr:
            print(f"[STDERR] {e.stderr}")
        return False

def check_environment():
    """Check if environment is properly set up."""
    print("=" * 60)
    print("CHECKING ENVIRONMENT")
    print("=" * 60)

    # Check conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    if conda_env != 'synctalk':
        print("⚠️  WARNING: Not in 'synctalk' conda environment")
        print("   Run: conda activate synctalk")
        return False

    print(f"✅ Conda environment: {conda_env}")

    # Check PyTorch
    try:
        import torch
        print(f"✅ PyTorch: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name()}")
    except ImportError:
        print("❌ PyTorch not found")
        return False

    # Check required packages
    required_packages = ['numpy', 'scipy', 'tqdm', 'librosa']
    missing_packages = []

    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"✅ {pkg}")
        except ImportError:
            missing_packages.append(pkg)
            print(f"❌ {pkg}")

    if missing_packages:
        print(f"\n[INSTALLING] Missing packages: {', '.join(missing_packages)}")
        run_command(f"pip install {' '.join(missing_packages)}", "Installing missing packages")

    return True

def check_data():
    """Check if data exists."""
    print("\n" + "=" * 60)
    print("CHECKING DATA")
    print("=" * 60)

    data_paths = [
        "data/Macron/Macron.mp4",
        "data/Macron/aud.wav",
        "data/Macron/ori_imgs/0.jpg",
        "data/Macron/face_mask/0.png"
    ]

    all_exist = True
    for path in data_paths:
        if os.path.exists(path):
            print(f"✅ {path}")
        else:
            print(f"❌ {path}")
            all_exist = False

    if not all_exist:
        print("\n[INFO] Some data files missing. Run data processing first.")
        return False

    return True

def main_menu():
    """Main interactive menu."""
    while True:
        print("\n" + "=" * 60)
        print("SYNCFACE ENHANCED - MAIN MENU")
        print("=" * 60)
        print("1. Check Environment Setup")
        print("2. Process Macron Data")
        print("3. Train Basic Model (Original)")
        print("4. Train Enhanced Model (RECOMMENDED)")
        print("5. Train Hybrid Model (Best Balance)")
        print("6. Train Ensemble Model (Max Quality)")
        print("7. Train Emotion Model")
        print("8. Test Trained Model")
        print("9. Run GUI Mode")
        print("10. View All Commands")
        print("0. Exit")
        print("=" * 60)

        choice = input("Select option [0-10]: ").strip()

        if choice == '0':
            print("\nGoodbye! 🎭")
            break
        elif choice == '1':
            check_environment()
        elif choice == '2':
            process_data()
        elif choice == '3':
            train_basic()
        elif choice == '4':
            train_enhanced()
        elif choice == '5':
            train_hybrid()
        elif choice == '6':
            train_ensemble()
        elif choice == '7':
            train_emotion()
        elif choice == '8':
            test_model()
        elif choice == '9':
            run_gui()
        elif choice == '10':
            show_all_commands()
        else:
            print("❌ Invalid choice. Try again.")

        input("\nPress Enter to continue...")

def process_data():
    """Process Macron data."""
    if not os.path.exists("data/Macron/Macron.mp4"):
        print("❌ Macron.mp4 not found in data/Macron/")
        print("   Place your video file at: data/Macron/Macron.mp4")
        return

    print("\n[INFO] Processing Macron video...")
    print("This will extract: audio, images, face landmarks, optical flow")
    confirm = input("Continue? [y/N]: ").strip().lower()

    if confirm == 'y':
        cmd = "python data_utils/process.py data/Macron/Macron.mp4 --task -1 --asr ave"
        run_command(cmd, "Processing Macron data")

def train_basic():
    """Train basic model."""
    print("\n[INFO] Training basic model (original LRS2)")
    confirm = input("This will train for 100k iterations. Continue? [y/N]: ").strip().lower()

    if confirm == 'y':
        cmd = "python main.py data/Macron --workspace output/Macron_basic --asr_model ave --iters 100000 -O"
        run_command(cmd, "Training basic model")

def train_enhanced():
    """Train enhanced model."""
    print("\n[INFO] Training enhanced model (Whisper encoder)")
    confirm = input("This will train for 100k iterations with enhanced features. Continue? [y/N]: ").strip().lower()

    if confirm == 'y':
        cmd = "python main.py data/Macron --workspace output/Macron_enhanced --use_enhanced_encoder --enhanced_encoder_type whisper --use_prosody --freeze_audio_backbone --asr_model ave --iters 100000 -O"
        run_command(cmd, "Training enhanced model")

def train_hybrid():
    """Train hybrid model."""
    print("\n[INFO] Training hybrid model (LRS2 + Whisper, RECOMMENDED)")
    confirm = input("This will train for 100k iterations with best quality/speed balance. Continue? [y/N]: ").strip().lower()

    if confirm == 'y':
        cmd = "python main.py data/Macron --workspace output/Macron_hybrid --use_enhanced_encoder --enhanced_encoder_type hybrid --use_prosody --freeze_audio_backbone --asr_model ave --iters 100000 -O"
        run_command(cmd, "Training hybrid model")

def train_ensemble():
    """Train ensemble model."""
    print("\n[INFO] Training ensemble model (Maximum quality)")
    print("⚠️  WARNING: Requires ~12GB GPU memory and slower training")
    confirm = input("Continue? [y/N]: ").strip().lower()

    if confirm == 'y':
        cmd = "python main.py data/Macron --workspace output/Macron_ensemble --use_enhanced_encoder --enhanced_encoder_type ensemble --use_prosody --freeze_audio_backbone --asr_model ave --iters 100000 --batch_size 2 -O"
        run_command(cmd, "Training ensemble model")

def train_emotion():
    """Train emotion model."""
    ravdess_path = input("Enter path to RAVDESS dataset: ").strip()

    if not ravdess_path or not os.path.exists(ravdess_path):
        print("❌ Invalid RAVDESS path")
        return

    print("\n[INFO] Training emotion recognition model")
    cmd = f"python scripts/train_emotion_recognition.py --dataset ravdess --data_path {ravdess_path} --emotion_model wav2vec2 --use_pretrained --epochs 50 --output_dir output/emotion_training"
    run_command(cmd, "Training emotion model")

def test_model():
    """Test trained model."""
    workspace = input("Enter workspace path (e.g., output/Macron_hybrid): ").strip()

    if not workspace or not os.path.exists(workspace):
        print("❌ Invalid workspace path")
        return

    cmd = f"python main.py data/Macron --workspace {workspace} --test -O"
    run_command(cmd, "Testing model")

def run_gui():
    """Run GUI mode."""
    workspace = input("Enter workspace path (e.g., output/Macron_hybrid): ").strip()

    if not workspace or not os.path.exists(workspace):
        print("❌ Invalid workspace path")
        return

    cmd = f"python main.py data/Macron --workspace {workspace} --test --gui -O"
    run_command(cmd, "Running GUI mode")

def show_all_commands():
    """Show all available commands."""
    print("\n[INFO] Showing comprehensive command reference...")
    cmd = "bash run_all_commands.sh"
    run_command(cmd, "Displaying all commands")

if __name__ == '__main__':
    # Check environment on startup
    if not check_environment():
        print("\n[WARNING] Environment issues detected. Please fix before proceeding.")

    # Check data on startup
    check_data()

    # Run main menu
    main_menu()
