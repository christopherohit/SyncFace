#!/bin/bash
# Complete Training Script for Macron with Enhanced Audio + Emotion
# Author: SyncTalk Enhanced
# Date: 2025

echo "=========================================="
echo "SyncTalk Enhanced Training - Macron"
echo "=========================================="

# Activate conda environment
echo "[INFO] Activating conda environment: synctalk"
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate synctalk

# Check if environment is activated
if [ "$CONDA_DEFAULT_ENV" != "synctalk" ]; then
    echo "[ERROR] Failed to activate synctalk environment"
    echo "Please activate manually: conda activate synctalk"
    exit 1
fi

echo "[INFO] Environment activated: $CONDA_DEFAULT_ENV"
echo ""

# Configuration
WORKSPACE="output/Macron_enhanced_final"
DATA_PATH="data/Macron"
ITERS=100000

echo "[INFO] Configuration:"
echo "  Workspace: $WORKSPACE"
echo "  Data Path: $DATA_PATH"
echo "  Iterations: $ITERS"
echo ""

# Option Selection
echo "Select training mode:"
echo "  1) Enhanced Audio Only (Whisper + Prosody)"
echo "  2) Hybrid Mode (LRS2 + Whisper) [RECOMMENDED]"
echo "  3) Full Ensemble (Maximum Quality)"
echo "  4) Quick Test (10k iterations)"
echo ""
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo "[INFO] Training with Enhanced Audio (Whisper + Prosody)"
        python main.py $DATA_PATH \
            --workspace $WORKSPACE \
            --use_enhanced_encoder \
            --enhanced_encoder_type whisper \
            --use_prosody \
            --freeze_audio_backbone \
            --asr_model ave \
            --iters $ITERS \
            -O
        ;;
    2)
        echo "[INFO] Training with Hybrid Mode (RECOMMENDED)"
        python main.py $DATA_PATH \
            --workspace $WORKSPACE \
            --use_enhanced_encoder \
            --enhanced_encoder_type hybrid \
            --use_prosody \
            --freeze_audio_backbone \
            --asr_model ave \
            --iters $ITERS \
            -O
        ;;
    3)
        echo "[INFO] Training with Full Ensemble (Maximum Quality)"
        python main.py $DATA_PATH \
            --workspace $WORKSPACE \
            --use_enhanced_encoder \
            --enhanced_encoder_type ensemble \
            --use_prosody \
            --freeze_audio_backbone \
            --asr_model ave \
            --iters $ITERS \
            --batch_size 2 \
            -O
        ;;
    4)
        echo "[INFO] Quick Test Mode (10k iterations)"
        python main.py $DATA_PATH \
            --workspace output/Macron_test \
            --use_enhanced_encoder \
            --enhanced_encoder_type whisper \
            --use_prosody \
            --freeze_audio_backbone \
            --asr_model ave \
            --iters 10000 \
            -O
        ;;
    *)
        echo "[ERROR] Invalid choice"
        exit 1
        ;;
esac

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "[SUCCESS] Training completed!"
    echo "=========================================="
    echo ""
    echo "To test the trained model, run:"
    echo "  conda activate synctalk"
    echo "  python main.py $DATA_PATH \\"
    echo "    --workspace $WORKSPACE \\"
    echo "    --use_enhanced_encoder \\"
    echo "    --enhanced_encoder_type [same as training] \\"
    echo "    --asr_model ave \\"
    echo "    -O --test"
    echo ""
else
    echo ""
    echo "[ERROR] Training failed with exit code $?"
    exit 1
fi

