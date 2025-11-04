#!/bin/bash
# ==============================================================================
# SyncFace - Complete Command Reference
# Enhanced Audio-Driven Talking Head Generation with Foundation Models
# ==============================================================================
# Author: SyncFace Enhanced Team
# Date: 2025
# Description: All possible commands for training, testing, and running SyncFace

# Activate conda environment
echo "=========================================="
echo "SyncFace - Enhanced Commands Reference"
echo "=========================================="
echo "Activating conda environment: synctalk"
echo ""

# Setup environment
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
if ! conda activate synctalk; then
    echo "[ERROR] Failed to activate synctalk environment"
    echo "Please run: conda activate synctalk"
    exit 1
fi

echo "[INFO] Environment activated: $CONDA_DEFAULT_ENV"
echo ""

# Configuration variables
DATA_DIR="data/Macron"
BASIC_WORKSPACE="output/Macron_basic"
ENHANCED_WORKSPACE="output/Macron_enhanced"
HYBRID_WORKSPACE="output/Macron_hybrid"
EMOTION_WORKSPACE="output/Macron_emotion"
ENSEMBLE_WORKSPACE="output/Macron_ensemble"
TEST_WORKSPACE="output/Macron_test"

ITERS_100K="100000"
ITERS_50K="50000"
ITERS_10K="10000"

# ==============================================================================
# 1. DATA PROCESSING COMMANDS
# ==============================================================================

echo "=========================================="
echo "1. DATA PROCESSING COMMANDS"
echo "=========================================="
echo ""

echo "# Process Macron video (extract audio, images, faces, flow)"
echo "python data_utils/process.py data/Macron/Macron.mp4 --task -1 --asr ave"
echo ""

echo "# Process specific tasks only"
echo "python data_utils/process.py data/Macron/Macron.mp4 --task 1  # Audio only"
echo "python data_utils/process.py data/Macron/Macron.mp4 --task 2  # Images only"
echo "python data_utils/process.py data/Macron/Macron.mp4 --task 3  # Semantics only"
echo ""

echo "# Process with different ASR models"
echo "python data_utils/process.py data/Macron/Macron.mp4 --asr hubert     # HuBERT features"
echo "python data_utils/process.py data/Macron/Macron.mp4 --asr deepspeech # DeepSpeech features"
echo "python data_utils/process.py data/Macron/Macron.mp4 --asr esperanto  # Esperanto features"
echo ""

echo "# Process May dataset (existing)"
echo "python data_utils/process.py data/May/May.mp4 --task -1 --asr ave"
echo ""

# ==============================================================================
# 2. BASIC TRAINING COMMANDS (Original LRS2)
# ==============================================================================

echo "=========================================="
echo "2. BASIC TRAINING COMMANDS (Original)"
echo "=========================================="
echo ""

echo "# Basic training with default settings"
echo "python main.py $DATA_DIR --workspace $BASIC_WORKSPACE --asr_model ave --iters $ITERS_100K -O"
echo ""

echo "# Basic training with different configurations"
echo "python main.py $DATA_DIR --workspace ${BASIC_WORKSPACE}_lips --asr_model ave --finetune_lips --iters $ITERS_50K -O"
echo "python main.py $DATA_DIR --workspace ${BASIC_WORKSPACE}_eye --asr_model ave --exp_eye --iters $ITERS_100K -O"
echo "python main.py $DATA_DIR --workspace ${BASIC_WORKSPACE}_au45 --asr_model ave --au45 --iters $ITERS_100K -O"
echo ""

echo "# Different audio models"
echo "python main.py $DATA_DIR --workspace ${BASIC_WORKSPACE}_hubert --asr_model hubert --iters $ITERS_100K -O"
echo "python main.py $DATA_DIR --workspace ${BASIC_WORKSPACE}_dspeech --asr_model deepspeech --iters $ITERS_100K -O"
echo ""

echo "# Training with different loss weights"
echo "python main.py $DATA_DIR --workspace ${BASIC_WORKSPACE}_loss --asr_model ave --lambda_amb 0.01 --iters $ITERS_100K -O"
echo ""

# ==============================================================================
# 3. ENHANCED AUDIO ENCODER TRAINING COMMANDS
# ==============================================================================

echo "=========================================="
echo "3. ENHANCED AUDIO ENCODER TRAINING"
echo "=========================================="
echo ""

echo "# 3.1 WHISPER ENCODER (Recommended)"
echo "python main.py $DATA_DIR --workspace ${ENHANCED_WORKSPACE}_whisper --use_enhanced_encoder --enhanced_encoder_type whisper --use_prosody --freeze_audio_backbone --asr_model ave --iters $ITERS_100K -O"
echo ""

echo "# 3.2 SPEECHT5 ENCODER (Fast)"
echo "python main.py $DATA_DIR --workspace ${ENHANCED_WORKSPACE}_speecht5 --use_enhanced_encoder --enhanced_encoder_type speecht5 --use_prosody --freeze_audio_backbone --asr_model ave --iters $ITERS_100K -O"
echo ""

echo "# 3.3 ENCODEC ENCODER (Audio Quality)"
echo "python main.py $DATA_DIR --workspace ${ENHANCED_WORKSPACE}_encodec --use_enhanced_encoder --enhanced_encoder_type encodec --use_prosody --freeze_audio_backbone --asr_model ave --iters $ITERS_100K -O"
echo ""

echo "# 3.4 HYBRID ENCODER (LRS2 + Whisper, RECOMMENDED)"
echo "python main.py $DATA_DIR --workspace $HYBRID_WORKSPACE --use_enhanced_encoder --enhanced_encoder_type hybrid --foundation_model_type whisper --use_prosody --freeze_audio_backbone --asr_model ave --iters $ITERS_100K -O"
echo ""

echo "# 3.5 ENSEMBLE ENCODER (Maximum Quality, Slower)"
echo "python main.py $DATA_DIR --workspace $ENSEMBLE_WORKSPACE --use_enhanced_encoder --enhanced_encoder_type ensemble --use_prosody --freeze_audio_backbone --asr_model ave --iters $ITERS_100K --batch_size 2 -O"
echo ""

echo "# 3.6 Fine-tune backbone (Higher quality, slower)"
echo "python main.py $DATA_DIR --workspace ${ENHANCED_WORKSPACE}_finetune --use_enhanced_encoder --enhanced_encoder_type whisper --use_prosody --no-freeze_audio_backbone --asr_model ave --iters $ITERS_50K -O"
echo ""

echo "# 3.7 With contrastive learning"
echo "python main.py $DATA_DIR --workspace ${ENHANCED_WORKSPACE}_contrastive --use_enhanced_encoder --enhanced_encoder_type whisper --use_prosody --use_contrastive --contrastive_temperature 0.07 --freeze_audio_backbone --asr_model ave --iters $ITERS_100K -O"
echo ""

# ==============================================================================
# 4. EMOTION RECOGNITION TRAINING COMMANDS
# ==============================================================================

echo "=========================================="
echo "4. EMOTION RECOGNITION TRAINING"
echo "=========================================="
echo ""

echo "# 4.1 Train emotion recognition model (requires RAVDESS dataset)"
echo "python scripts/train_emotion_recognition.py --dataset ravdess --data_path /path/to/RAVDESS --emotion_model wav2vec2 --use_pretrained --epochs 50 --batch_size 16 --output_dir output/emotion_training"
echo ""

echo "# 4.2 Train with different emotion models"
echo "python scripts/train_emotion_recognition.py --dataset ravdess --data_path /path/to/RAVDESS --emotion_model cnn --epochs 100 --output_dir output/emotion_cnn"
echo "python scripts/train_emotion_recognition.py --dataset ravdess --data_path /path/to/RAVDESS --emotion_model prosody --epochs 100 --output_dir output/emotion_prosody"
echo ""

# ==============================================================================
# 5. FULL EMOTION-AWARE TRAINING COMMANDS
# ==============================================================================

echo "=========================================="
echo "5. FULL EMOTION-AWARE TRAINING"
echo "=========================================="
echo ""

echo "# 5.1 Enhanced + Emotion (requires emotion checkpoint)"
echo "python main.py $DATA_DIR --workspace $EMOTION_WORKSPACE --use_enhanced_encoder --enhanced_encoder_type hybrid --use_prosody --freeze_audio_backbone --use_emotion --emotion_model wav2vec2 --emotion_checkpoint output/emotion_training/best_emotion_model.pth --emotion_strength 0.7 --asr_model ave --iters $ITERS_100K -O"
echo ""

echo "# 5.2 Different emotion strengths"
echo "python main.py $DATA_DIR --workspace ${EMOTION_WORKSPACE}_strong --use_enhanced_encoder --enhanced_encoder_type hybrid --use_prosody --use_emotion --emotion_strength 1.0 --asr_model ave --iters $ITERS_100K -O"
echo "python main.py $DATA_DIR --workspace ${EMOTION_WORKSPACE}_subtle --use_enhanced_encoder --enhanced_encoder_type hybrid --use_prosody --use_emotion --emotion_strength 0.3 --asr_model ave --iters $ITERS_100K -O"
echo ""

echo "# 5.3 Different emotion smoothing"
echo "python main.py $DATA_DIR --workspace ${EMOTION_WORKSPACE}_smooth --use_enhanced_encoder --enhanced_encoder_type hybrid --use_emotion --emotion_smoothing ema --asr_model ave --iters $ITERS_100K -O"
echo "python main.py $DATA_DIR --workspace ${EMOTION_WORKSPACE}_conv_smooth --use_enhanced_encoder --enhanced_encoder_type hybrid --use_emotion --emotion_smoothing conv --asr_model ave --iters $ITERS_100K -O"
echo ""

# ==============================================================================
# 6. TESTING AND INFERENCE COMMANDS
# ==============================================================================

echo "=========================================="
echo "6. TESTING AND INFERENCE COMMANDS"
echo "=========================================="
echo ""

echo "# 6.1 Test trained model"
echo "python main.py $DATA_DIR --workspace $HYBRID_WORKSPACE --test -O"
echo ""

echo "# 6.2 Test on training dataset"
echo "python main.py $DATA_DIR --workspace $HYBRID_WORKSPACE --test --test_train -O"
echo ""

echo "# 6.3 Test with different audio"
echo "python main.py $DATA_DIR --workspace $HYBRID_WORKSPACE --test --aud demo/test.wav -O"
echo ""

echo "# 6.4 GUI mode for real-time interaction"
echo "python main.py $DATA_DIR --workspace $HYBRID_WORKSPACE --test --gui -O"
echo ""

echo "# 6.5 ASR real-time mode"
echo "python main.py $DATA_DIR --workspace $HYBRID_WORKSPACE --test --asr --asr_wav demo/test.wav -O"
echo ""

# ==============================================================================
# 7. TORSO TRAINING COMMANDS
# ==============================================================================

echo "=========================================="
echo "7. TORSO TRAINING COMMANDS"
echo "=========================================="
echo ""

echo "# 7.1 Train torso (requires head model)"
echo "python main.py $DATA_DIR --workspace output/Macron_torso --torso --head_ckpt output/Macron_hybrid/checkpoints/ngp.pth --asr_model ave --iters $ITERS_50K -O"
echo ""

echo "# 7.2 Enhanced torso training"
echo "python main.py $DATA_DIR --workspace output/Macron_torso_enhanced --torso --head_ckpt output/Macron_hybrid/checkpoints/ngp.pth --use_enhanced_encoder --enhanced_encoder_type hybrid --asr_model ave --iters $ITERS_50K -O"
echo ""

# ==============================================================================
# 8. TESTING AND VALIDATION COMMANDS
# ==============================================================================

echo "=========================================="
echo "8. TESTING AND VALIDATION COMMANDS"
echo "=========================================="
echo ""

echo "# 8.1 Test network refactoring"
echo "python test_network_refactor.py"
echo ""

echo "# 8.2 Test enhanced encoders"
echo "python scripts/test_enhanced_encoder.py --encoder whisper --audio demo/test.wav"
echo "python scripts/test_enhanced_encoder.py --compare all --audio demo/test.wav"
echo ""

echo "# 8.3 Quick training test"
echo "python train_macron.py  # Interactive menu"
echo ""

echo "# 8.4 Validate training data"
echo "python main.py $DATA_DIR --workspace output/validation --test --test_train --data_range 0,100 -O"
echo ""

# ==============================================================================
# 9. CONFIGURATION EXAMPLES
# ==============================================================================

echo "=========================================="
echo "9. CONFIGURATION EXAMPLES"
echo "=========================================="
echo ""

echo "# 9.1 Memory-efficient training (6GB GPU)"
echo "python main.py $DATA_DIR --workspace ${HYBRID_WORKSPACE}_lowmem --use_enhanced_encoder --enhanced_encoder_type whisper --freeze_audio_backbone --batch_size 1 --asr_model ave --iters $ITERS_50K -O"
echo ""

echo "# 9.2 High-quality training (24GB GPU)"
echo "python main.py $DATA_DIR --workspace ${HYBRID_WORKSPACE}_highq --use_enhanced_encoder --enhanced_encoder_type ensemble --use_prosody --use_contrastive --batch_size 4 --asr_model ave --iters $ITERS_100K -O"
echo ""

echo "# 9.3 Fast iteration training"
echo "python main.py $DATA_DIR --workspace ${TEST_WORKSPACE} --use_enhanced_encoder --enhanced_encoder_type hybrid --asr_model ave --iters $ITERS_10K -O"
echo ""

echo "# 9.4 Resume training from checkpoint"
echo "python main.py $DATA_DIR --workspace $HYBRID_WORKSPACE --ckpt ngp_ep0050 --iters $ITERS_100K -O"
echo ""

echo "# 9.5 Custom learning rates"
echo "python main.py $DATA_DIR --workspace ${HYBRID_WORKSPACE}_lr --lr 5e-3 --lr_net 5e-4 --asr_model ave --iters $ITERS_100K -O"
echo ""

# ==============================================================================
# 10. UTILITY COMMANDS
# ==============================================================================

echo "=========================================="
echo "10. UTILITY COMMANDS"
echo "=========================================="
echo ""

echo "# 10.1 Check GPU memory usage"
echo "nvidia-smi"
echo "watch -n 1 nvidia-smi"
echo ""

echo "# 10.2 Monitor training progress"
echo "tail -f output/Macron_hybrid/training.log"
echo "tensorboard --logdir output/Macron_hybrid/"
echo ""

echo "# 10.3 Clean up old checkpoints"
echo "find output/ -name '*.pth' -mtime +7 -delete"
echo ""

echo "# 10.4 Compare model sizes"
echo "ls -lh output/*/checkpoints/*.pth"
echo ""

echo "# 10.5 Backup important models"
echo "cp output/Macron_hybrid/checkpoints/ngp.pth backups/model_backup_$(date +%Y%m%d).pth"
echo ""

echo "# 10.6 View generated videos"
echo "ls -la output/*/results/*.mp4"
echo "vlc output/Macron_hybrid/results/ngp_ep0100.mp4"
echo ""

# ==============================================================================
# 11. TROUBLESHOOTING COMMANDS
# ==============================================================================

echo "=========================================="
echo "11. TROUBLESHOOTING COMMANDS"
echo "=========================================="
echo ""

echo "# 11.1 Check environment"
echo "conda list | grep torch"
echo "python -c 'import torch; print(f\"PyTorch: {torch.__version__}\", f\"CUDA: {torch.cuda.is_available()}\")'"
echo ""

echo "# 11.2 Install missing dependencies"
echo "pip install transformers accelerate torchaudio librosa"
echo "conda install -c conda-forge ffmpeg"
echo ""

echo "# 11.3 Clear cache and restart"
echo "rm -rf ~/.cache/torch/hub/"
echo "rm -rf ~/.cache/huggingface/"
echo "conda deactivate && conda activate synctalk"
echo ""

echo "# 11.4 Debug memory issues"
echo "python -c 'import torch; print(f\"GPU memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB\")'"
echo "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512"
echo ""

# ==============================================================================
# SUMMARY
# ==============================================================================

echo "=========================================="
echo "COMMAND REFERENCE SUMMARY"
echo "=========================================="
echo ""

echo "📁 Data Processing:"
echo "  python data_utils/process.py data/Macron/Macron.mp4 --task -1 --asr ave"
echo ""

echo "🚀 Basic Training:"
echo "  python main.py data/Macron --workspace output/basic --asr_model ave --iters 100000 -O"
echo ""

echo "🎵 Enhanced Training (RECOMMENDED):"
echo "  python main.py data/Macron --workspace output/enhanced --use_enhanced_encoder --enhanced_encoder_type hybrid --use_prosody --asr_model ave --iters 100000 -O"
echo ""

echo "😊 Emotion Training:"
echo "  python scripts/train_emotion_recognition.py --dataset ravdess --data_path /path/to/RAVDESS --emotion_model wav2vec2 --output_dir output/emotion"
echo ""

echo "🎬 Full Emotion-Aware Training:"
echo "  python main.py data/Macron --workspace output/emotion --use_enhanced_encoder --enhanced_encoder_type hybrid --use_emotion --emotion_checkpoint output/emotion/best_emotion_model.pth --asr_model ave --iters 100000 -O"
echo ""

echo "🧪 Testing:"
echo "  python main.py data/Macron --workspace output/enhanced --test -O"
echo ""

echo "🎮 GUI Mode:"
echo "  python main.py data/Macron --workspace output/enhanced --test --gui -O"
echo ""

echo "=========================================="
echo "✅ ALL COMMANDS LISTED ABOVE"
echo "Copy and run any command as needed!"
echo "=========================================="



#
/mnt/2T/nhanhuynh/project/InsTaG/OpenFace/build/bin/FeatureExtraction -fdir /mnt/2T/nhanhuynh/project/SynthFace/data/May/May_upscaled.mp4 -out_dir /mnt/2T/nhanhuynh/project/SynthFace/data/May/result_OF