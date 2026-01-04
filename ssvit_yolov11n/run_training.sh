#!/bin/bash
# SSViT-YOLOv11n Training Script
# Run from the ssvit_yolov11n directory

echo "=========================================="
echo "SSViT-YOLOv11n Training Pipeline"
echo "=========================================="

# Step 1: Prepare dataset
echo ""
echo "Step 1: Preparing dataset..."
python prepare_dataset.py --source ../kaggle_dataset_label --output ./dataset

# Step 2: Train model
echo ""
echo "Step 2: Starting training..."
python train.py \
    --data-path ./dataset \
    --epochs 100 \
    --batch-size 8 \
    --lr 0.001 \
    --img-size 640 \
    --num-classes 5 \
    --output-dir ./runs \
    --patience 50

echo ""
echo "Training complete!"
echo "Check ./runs for results"
