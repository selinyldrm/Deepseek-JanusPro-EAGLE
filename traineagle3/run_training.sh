#!/bin/bash

# Eagle 3 Training Script for Image Generation Models
# Usage: ./run_training.sh

set -e

echo "🚀 Starting Eagle 3 Training for Image Generation Models"

# Default configuration
MODEL="lumina_mgpt"
BASE_PATH="ckpts/lumina_mgpt/Lumina-mGPT-7B-768"
DATA_DIR="/home/server44/sihwan_workspace/ssd/lumina_mgpt_eagle_mscoco2017train"
SAVE_DIR="ckpts/lumina_mgpt/trained_drafters_eagle3"

# Training parameters
NUM_EPOCHS=20
BATCH_SIZE=4
LEARNING_RATE=1e-4
LENGTH=7  # Eagle 3 multi-step length

# Create save directory if it doesn't exist
mkdir -p $SAVE_DIR

echo "📋 Training Configuration:"
echo "  Model: $MODEL"
echo "  Base Path: $BASE_PATH" 
echo "  Data Directory: $DATA_DIR"
echo "  Save Directory: $SAVE_DIR"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Eagle 3 Length: $LENGTH"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "❌ Error: Data directory not found: $DATA_DIR"
    echo "Please update DATA_DIR in this script or create the directory"
    exit 1
fi

# Check if base model exists
if [ ! -d "$BASE_PATH" ]; then
    echo "❌ Error: Base model not found: $BASE_PATH"
    echo "Please update BASE_PATH in this script or download the model"
    exit 1
fi

echo "🔥 Starting Eagle 3 Training..."

python main.py \
    --model $MODEL \
    --base_path $BASE_PATH \
    --config_path config.json \
    --data_dir $DATA_DIR \
    --save_dir $SAVE_DIR \
    --num_epochs $NUM_EPOCHS \
    --bs $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --length $LENGTH \
    --eagle3_weight_decay 0.8 \
    --gradient_accumulation_steps 1 \
    --warmup_steps_ratio 0.03 \
    --is_warmup \
    --p_w 0.1 \
    --embed_upscale 1.0 \
    --grad_clip 0.5 \
    --max_len 4096 \
    --eval_freq 1 \
    --save_freq 5 \
    --data_noise uniform \
    --std 0.2 \
    --train_data_ratio 0.95

echo "✅ Eagle 3 Training Completed!"
echo "📁 Model saved in: $SAVE_DIR"

