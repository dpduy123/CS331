#!/bin/bash

# YOLOv8 Coffee Bean Training - Docker Runner
# Usage: ./run.sh [cpu|gpu] [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
MODE="${1:-cpu}"
EPOCHS="${2:-100}"
BATCH="${3:-4}"
MODEL="${4:-yolov8n}"

echo "=============================================="
echo "COFFEE BEAN YOLOV8 TRAINING - DOCKER"
echo "=============================================="
echo "Mode: $MODE"
echo "Model: $MODEL"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH"
echo "=============================================="

# Check if dataset exists
if [ ! -d "$PROJECT_DIR/kaggle_dataset_round1/images" ]; then
    echo "ERROR: Dataset not found at $PROJECT_DIR/kaggle_dataset_round1"
    echo "Please download the dataset first."
    exit 1
fi

cd "$SCRIPT_DIR"

if [ "$MODE" == "gpu" ]; then
    echo "Starting GPU training..."

    # Check if nvidia-docker is available
    if ! command -v nvidia-smi &> /dev/null; then
        echo "WARNING: nvidia-smi not found. GPU may not be available."
    fi

    # Build GPU image
    docker build -t coffee-yolo-gpu -f Dockerfile.gpu .

    # Run with GPU
    docker run --rm \
        --gpus all \
        -v "$PROJECT_DIR/kaggle_dataset_round1:/app/data:ro" \
        -v "$SCRIPT_DIR/runs:/app/runs" \
        -v "$SCRIPT_DIR/weights:/app/weights" \
        -v "$SCRIPT_DIR/train.py:/app/train.py:ro" \
        coffee-yolo-gpu \
        python /app/train.py \
            --model "$MODEL" \
            --epochs "$EPOCHS" \
            --batch "$BATCH" \
            --device 0

else
    echo "Starting CPU training..."

    # Build CPU image
    docker build -t coffee-yolo-cpu -f Dockerfile .

    # Run with CPU
    docker run --rm \
        -v "$PROJECT_DIR/kaggle_dataset_round1:/app/data:ro" \
        -v "$SCRIPT_DIR/runs:/app/runs" \
        -v "$SCRIPT_DIR/weights:/app/weights" \
        -v "$SCRIPT_DIR/train.py:/app/train.py:ro" \
        coffee-yolo-cpu \
        python /app/train.py \
            --model "$MODEL" \
            --epochs "$EPOCHS" \
            --batch "$BATCH" \
            --device cpu
fi

echo ""
echo "=============================================="
echo "TRAINING COMPLETE!"
echo "Results saved to: $SCRIPT_DIR/runs/"
echo "=============================================="
