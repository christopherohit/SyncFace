#!/bin/bash

# ST-Gauss: Full Training Pipeline
# Usage: ./train_st_gauss.sh /path/to/data [config]

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  ST-Gauss Training Pipeline${NC}"
echo -e "${GREEN}============================================${NC}"

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: $0 <data_path> [config]"
    echo "Configs: default, high_quality, fast"
    exit 1
fi

DATA_PATH="$1"
CONFIG="${2:-default}"
MODEL_PATH="${DATA_PATH}/output"

echo -e "\n${YELLOW}Data: ${DATA_PATH}${NC}"
echo -e "${YELLOW}Config: ${CONFIG}${NC}"
echo -e "${YELLOW}Output: ${MODEL_PATH}${NC}\n"

# Navigate to project root
cd "$(dirname "$0")/.."

# ============================================
# Stage 1: Mouth NeRF Training
# ============================================
echo -e "\n${GREEN}Stage 1: Mouth NeRF Training${NC}"
echo "============================================"
python st_gauss/train_mouth_nerf.py \
    -s "$DATA_PATH" \
    -m "$MODEL_PATH" \
    --config "$CONFIG"

# ============================================
# Stage 2: Face Gaussian Training
# ============================================
echo -e "\n${GREEN}Stage 2: Face Gaussian Training${NC}"
echo "============================================"
python st_gauss/train_face_gaussian.py \
    -s "$DATA_PATH" \
    -m "$MODEL_PATH" \
    --config "$CONFIG"

# ============================================
# Stage 3: Joint Fine-tuning
# ============================================
echo -e "\n${GREEN}Stage 3: Joint Fine-tuning${NC}"
echo "============================================"
python st_gauss/train_joint.py \
    -s "$DATA_PATH" \
    -m "$MODEL_PATH" \
    --config "$CONFIG"

# ============================================
# Synthesis
# ============================================
echo -e "\n${GREEN}Synthesis${NC}"
echo "============================================"
python st_gauss/synthesize.py \
    -s "$DATA_PATH" \
    -m "$MODEL_PATH" \
    --output "${MODEL_PATH}/synthesis.mp4"

echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}Training Complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo -e "Model saved to: ${MODEL_PATH}"
echo -e "Video output: ${MODEL_PATH}/synthesis.mp4"

