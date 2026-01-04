#!/bin/bash
#SBATCH --job-name=ssvit_yolov11n
#SBATCH --output=logs/ssvit_%j.out
#SBATCH --error=logs/ssvit_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# ============================================================
# SSViT-YOLOv11n Training on SLURM
# Coffee Bean Ripeness Detection
# ============================================================

echo "========================================"
echo "SSViT-YOLOv11n Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "========================================"

# Activate conda environment from /datastore
source ~/.bashrc
conda activate /datastore/vit/envs/yolo

# Verify environment
echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Set working directory to CoffeeProject
cd /datastore/vit/dpduy123/CoffeeProject

# Create logs directory
mkdir -p logs

# ============================================================
# Training Configuration
# ============================================================
DATA_PATH="dataset"
EPOCHS=100
BATCH_SIZE=16
IMG_SIZE=640
WORKERS=8
DEVICE="cuda"
OUTPUT_DIR="runs/ssvit_train_${SLURM_JOB_ID}"

echo ""
echo "========================================"
echo "Training Configuration:"
echo "  Data path: $DATA_PATH"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Image size: $IMG_SIZE"
echo "  Workers: $WORKERS"
echo "  Device: $DEVICE"
echo "  Output: $OUTPUT_DIR"
echo "========================================"
echo ""

# ============================================================
# Run Training
# ============================================================
python ssvit_yolov11n/train.py \
    --data-path "$DATA_PATH" \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --img-size $IMG_SIZE \
    --workers $WORKERS \
    --device $DEVICE \
    --output-dir "$OUTPUT_DIR"

# ============================================================
# Post-training
# ============================================================
echo ""
echo "========================================"
echo "Training completed!"
echo "End time: $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================"

# Copy best weights to models folder
mkdir -p models
if [ -f "$OUTPUT_DIR/best.pt" ]; then
    cp "$OUTPUT_DIR/best.pt" "models/ssvit_yolov11n_best.pt"
    echo "Best weights copied to: models/ssvit_yolov11n_best.pt"
fi

# List results
echo ""
echo "Output files:"
ls -lh "$OUTPUT_DIR"
