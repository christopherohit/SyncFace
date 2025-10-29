#!/bin/bash
################################################################################
# Complete InsTaG-SyncTalk Pipeline: Training + Inference
#
# This script runs the entire pipeline:
# 1. Geometry preprocessing (optional)
# 2. Pre-training UMF (if you have multiple people)
# 3. Adaptation to target person
# 4. Inference to generate video
#
# Usage:
#   bash run_full_pipeline.sh
################################################################################

set -e  # Exit on error

# ============================================================================
# CONFIGURATION - EDIT THESE PATHS
# ============================================================================

# Path to your data
DATA_ROOT="/home/springer/Project/SyncTalk/data"
TARGET_PERSON="May"
TARGET_DATA="${DATA_ROOT}/${TARGET_PERSON}"

# Audio for inference
INFERENCE_AUDIO="demo/test.wav"

# Workspace directories
WORKSPACE_PRETRAIN="workspace_pretrain"
WORKSPACE_ADAPT="workspace_adapt_${TARGET_PERSON}"
OUTPUT_DIR="outputs"

# Training parameters
PRETRAIN_ITERS=200000
ADAPT_ITERS=20000

# Inference parameters
OUTPUT_VIDEO="${OUTPUT_DIR}/output_${TARGET_PERSON}.mp4"
VIDEO_FPS=25
VIDEO_WIDTH=450
VIDEO_HEIGHT=450

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

echo "========================================================================"
echo "  InsTaG-SyncTalk Complete Pipeline"
echo "========================================================================"
echo "Target Person: ${TARGET_PERSON}"
echo "Data Path: ${TARGET_DATA}"
echo "========================================================================"

# Activate conda environment
source ~/miniconda3/bin/activate synctalk
echo "✓ Activated synctalk environment"

# Create output directory
mkdir -p ${OUTPUT_DIR}

# ============================================================================
# STEP 1: GEOMETRY PREPROCESSING (Optional but Recommended)
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 1: Geometry Preprocessing"
echo "========================================================================"

read -p "Run geometry preprocessing? (Recommended for better 3D quality) [y/N]: " run_geometry

if [[ "$run_geometry" =~ ^[Yy]$ ]]; then
    echo "Running geometry preprocessing..."
    python preprocess_geometry.py \
        --data ${TARGET_DATA} \
        --method midas \
        --device cuda:0
    
    echo "✓ Geometry preprocessing complete"
else
    echo "⊘ Skipping geometry preprocessing"
    echo "  (You can run it later: python preprocess_geometry.py --data ${TARGET_DATA})"
fi

# ============================================================================
# STEP 2: PRE-TRAINING (Multi-Person Phase)
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 2: Pre-training Universal Motion Field"
echo "========================================================================"

if [ -f "${WORKSPACE_PRETRAIN}/umf_final.pth" ]; then
    echo "✓ Found existing UMF checkpoint: ${WORKSPACE_PRETRAIN}/umf_final.pth"
    read -p "Use existing checkpoint? [Y/n]: " use_existing
    
    if [[ "$use_existing" =~ ^[Nn]$ ]]; then
        run_pretrain=true
    else
        run_pretrain=false
    fi
else
    echo "No existing UMF checkpoint found"
    read -p "Run pre-training? (Requires multiple people in ${DATA_ROOT}) [y/N]: " run_pretrain_input
    
    if [[ "$run_pretrain_input" =~ ^[Yy]$ ]]; then
        run_pretrain=true
    else
        run_pretrain=false
        echo ""
        echo "⚠️  WARNING: Skipping pre-training phase"
        echo "   This is OK if you only want to test with single-person training"
        echo "   or if you already have a pre-trained UMF checkpoint"
        echo ""
    fi
fi

if [ "$run_pretrain" = true ]; then
    echo "Running pre-training (this will take 12-24 hours)..."
    echo "  Data root: ${DATA_ROOT}"
    echo "  Iterations: ${PRETRAIN_ITERS}"
    echo ""
    
    python pretrain_umf.py \
        --data_root ${DATA_ROOT} \
        --workspace ${WORKSPACE_PRETRAIN} \
        --iters ${PRETRAIN_ITERS} \
        --lambda_C 0.01 \
        --lr 1e-2 \
        --lr_net 1e-3 \
        --fp16 \
        --cuda_ray
    
    echo "✓ Pre-training complete"
fi

# ============================================================================
# STEP 3: ADAPTATION (Few-Shot Learning)
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 3: Adaptation to Target Person"
echo "========================================================================"

# Check if we have a UMF checkpoint for adaptation
if [ -f "${WORKSPACE_PRETRAIN}/umf_final.pth" ]; then
    USE_INSTAG=true
    UMF_CHECKPOINT="${WORKSPACE_PRETRAIN}/umf_final.pth"
    echo "Using InsTaG method with UMF: ${UMF_CHECKPOINT}"
else
    USE_INSTAG=false
    echo "⚠️  No UMF checkpoint found - using standard training"
    echo "   (For InsTaG features, you need to run pre-training first)"
fi

if [ "$USE_INSTAG" = true ]; then
    # InsTaG adaptation
    echo "Running adaptation (1-3 hours)..."
    echo "  Target: ${TARGET_DATA}"
    echo "  Iterations: ${ADAPT_ITERS}"
    echo ""
    
    python adapt_identity.py \
        --umf_checkpoint ${UMF_CHECKPOINT} \
        --data ${TARGET_DATA} \
        --workspace ${WORKSPACE_ADAPT} \
        --iters ${ADAPT_ITERS} \
        --lambda_D 0.1 \
        --lambda_N 0.05 \
        --lr 2e-2 \
        --lr_net 5e-3 \
        --fp16 \
        --cuda_ray
    
    TRAINED_MODEL="${WORKSPACE_ADAPT}/adapted_final.pth"
    echo "✓ Adaptation complete"
else
    # Standard single-person training
    echo "Running standard training..."
    echo "  Using original SyncTalk (not InsTaG)"
    echo ""
    
    python main.py ${TARGET_DATA} \
        --workspace ${WORKSPACE_ADAPT} \
        --iters ${ADAPT_ITERS} \
        --O
    
    TRAINED_MODEL="${WORKSPACE_ADAPT}/checkpoints/ngp_ep0200.pth"
    echo "✓ Training complete"
fi

# ============================================================================
# STEP 4: INFERENCE (Generate Video)
# ============================================================================

echo ""
echo "========================================================================"
echo "STEP 4: Inference - Generate Talking Face Video"
echo "========================================================================"

if [ ! -f "${INFERENCE_AUDIO}" ]; then
    echo "⚠️  Audio file not found: ${INFERENCE_AUDIO}"
    echo "   Please provide a valid audio file"
    exit 1
fi

if [ "$USE_INSTAG" = true ]; then
    # InsTaG inference
    echo "Running InsTaG inference..."
    echo "  Model: ${TRAINED_MODEL}"
    echo "  Audio: ${INFERENCE_AUDIO}"
    echo "  Output: ${OUTPUT_VIDEO}"
    echo ""
    
    python infer_instag.py \
        --checkpoint ${TRAINED_MODEL} \
        --audio ${INFERENCE_AUDIO} \
        --output ${OUTPUT_VIDEO} \
        --reference_data ${TARGET_DATA} \
        --fps ${VIDEO_FPS} \
        --W ${VIDEO_WIDTH} \
        --H ${VIDEO_HEIGHT} \
        --device cuda
else
    # Standard SyncTalk inference
    echo "Running standard inference..."
    echo "  Using original SyncTalk inference"
    echo ""
    
    python main.py ${TARGET_DATA} \
        --workspace ${WORKSPACE_ADAPT} \
        --test \
        --aud ${INFERENCE_AUDIO} \
        --O
    
    # The output will be in workspace_adapt/results/
    echo "✓ Inference complete"
    echo "  Check ${WORKSPACE_ADAPT}/results/ for output"
fi

# ============================================================================
# SUMMARY
# ============================================================================

echo ""
echo "========================================================================"
echo "  Pipeline Complete!"
echo "========================================================================"
echo ""
echo "Summary:"
echo "  Method: $([ "$USE_INSTAG" = true ] && echo "InsTaG (Multi-person)" || echo "Standard (Single-person)")"
echo "  Target Person: ${TARGET_PERSON}"
echo "  Model: ${TRAINED_MODEL}"

if [ "$USE_INSTAG" = true ] && [ -f "${OUTPUT_VIDEO}" ]; then
    echo "  Output Video: ${OUTPUT_VIDEO}"
    echo ""
    echo "✓ You can now view your generated video:"
    echo "  vlc ${OUTPUT_VIDEO}"
fi

echo ""
echo "========================================================================"
