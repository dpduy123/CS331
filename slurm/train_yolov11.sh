#!/bin/bash
# ============================================================
# Unified YOLOv11 Training Script for SLURM
# Coffee Bean Ripeness Detection
#
# Usage:
#   sbatch --export=MODEL=yolov11n slurm/train_yolov11.sh
#   sbatch --export=MODEL=yolov11s,EPOCHS=150 slurm/train_yolov11.sh
#   sbatch --export=MODEL=yolov11m,BATCH=8,EPOCHS=200 slurm/train_yolov11.sh
# ============================================================

#SBATCH --job-name=yolov11_coffee
#SBATCH --output=logs/yolov11_%j.out
#SBATCH --error=logs/yolov11_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --partition=gpu

# ============================================================
# Default Configuration (can be overridden via --export)
# ============================================================
MODEL=${MODEL:-"yolov11n"}
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-16}
IMG_SIZE=${IMG_SIZE:-640}
DATA_YAML=${DATA_YAML:-"dataset.yaml"}
WORKERS=${WORKERS:-8}
PATIENCE=${PATIENCE:-50}

# Auto-adjust settings based on model size
case $MODEL in
    "yolov11n")
        EPOCHS=${EPOCHS:-100}
        BATCH=${BATCH:-16}
        ;;
    "yolov11s")
        EPOCHS=${EPOCHS:-150}
        BATCH=${BATCH:-16}
        ;;
    "yolov11m")
        EPOCHS=${EPOCHS:-200}
        BATCH=${BATCH:-8}
        ;;
    "yolov11l")
        EPOCHS=${EPOCHS:-200}
        BATCH=${BATCH:-4}
        ;;
    "yolov11x")
        EPOCHS=${EPOCHS:-200}
        BATCH=${BATCH:-2}
        ;;
esac

PROJECT="runs/train"
NAME="${MODEL}_coffee_${SLURM_JOB_ID}"

echo "========================================"
echo "YOLOv11 Training on SLURM"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "========================================"

# Load modules (adjust based on your cluster)
# module load cuda/11.8
# module load cudnn/8.6

# Activate conda environment
source ~/.bashrc
conda activate yolo  # Change to your env name

# Verify environment
echo ""
echo "Environment:"
echo "  Python: $(which python)"
echo "  Python version: $(python --version)"
python -c "import torch; print(f'  PyTorch: {torch.__version__}')"
python -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
python -c "import ultralytics; print(f'  Ultralytics: {ultralytics.__version__}')"

# Set working directory
cd $SLURM_SUBMIT_DIR

# Create directories
mkdir -p logs
mkdir -p models

echo ""
echo "========================================"
echo "Training Configuration:"
echo "  Model: $MODEL"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH"
echo "  Image size: $IMG_SIZE"
echo "  Dataset: $DATA_YAML"
echo "  Workers: $WORKERS"
echo "  Patience: $PATIENCE"
echo "  Project: $PROJECT"
echo "  Name: $NAME"
echo "========================================"
echo ""

# ============================================================
# Run Training
# ============================================================
python scripts/train_yolov11.py \
    --model $MODEL \
    --data $DATA_YAML \
    --epochs $EPOCHS \
    --batch $BATCH \
    --imgsz $IMG_SIZE \
    --device 0 \
    --workers $WORKERS \
    --project $PROJECT \
    --name $NAME \
    --patience $PATIENCE

TRAIN_EXIT_CODE=$?

# ============================================================
# Post-training
# ============================================================
echo ""
echo "========================================"
echo "Training completed!"
echo "Exit code: $TRAIN_EXIT_CODE"
echo "End time: $(date)"
echo "Results saved to: $PROJECT/$NAME"
echo "========================================"

# Copy best weights to models folder
if [ -f "$PROJECT/$NAME/weights/best.pt" ]; then
    cp "$PROJECT/$NAME/weights/best.pt" "models/${MODEL}_best.pt"
    echo "Best weights copied to: models/${MODEL}_best.pt"
fi

# Print final metrics if available
if [ -f "$PROJECT/$NAME/results.csv" ]; then
    echo ""
    echo "Final Training Metrics:"
    tail -1 "$PROJECT/$NAME/results.csv"
fi

exit $TRAIN_EXIT_CODE
