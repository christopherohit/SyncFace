#!/bin/bash
# Training Macron with Enhanced Audio Encoder (Whisper)
# This captures prosody, emotion, and speaking style beyond just lip sync

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate synctalk

# Option 1: Whisper Encoder (Recommended - Best prosody modeling)
python main.py data/Macron \
  --workspace output/Macron_whisper \
  --use_enhanced_encoder \
  --enhanced_encoder_type whisper \
  --use_prosody \
  --freeze_audio_backbone \
  --asr_model ave \
  --iters 100000 \
  -O

# Option 2: Hybrid Mode (Best balance - LRS2 lip sync + Whisper prosody)
# python main.py data/Macron \
#   --workspace output/Macron_hybrid \
#   --use_enhanced_encoder \
#   --enhanced_encoder_type hybrid \
#   --foundation_model_type whisper \
#   --use_prosody \
#   --asr_model ave \
#   --iters 100000 \
#   -O

# Option 3: Ensemble (Maximum quality - slower but best results)
# python main.py data/Macron \
#   --workspace output/Macron_ensemble \
#   --use_enhanced_encoder \
#   --enhanced_encoder_type ensemble \
#   --use_prosody \
#   --freeze_audio_backbone \
#   --asr_model ave \
#   --iters 100000 \
#   --batch_size 2 \
#   -O

# After training, test with:
# python main.py data/Macron \
#   --workspace output/Macron_whisper \
#   --use_enhanced_encoder \
#   --enhanced_encoder_type whisper \
#   --use_prosody \
#   --asr_model ave \
#   -O --test

