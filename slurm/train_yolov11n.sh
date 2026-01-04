#!/bin/bash
#SBATCH --job-name=yolov11n_coffee
#SBATCH --output=logs/yolov11n_%j.out
#SBATCH --error=logs/yolov11n_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=gpu

# ============================================================
# YOLOv11n Training on SLURM
# Coffee Bean Ripeness Detection
# ============================================================

echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "========================================"

# Load modules (adjust based on your cluster)
# module load cuda/11.8
# module load cudnn/8.6

# Activate conda environment
# Option 1: If conda is in your path
source ~/.bashrc
conda activate yolo  # Change to your env name

# Option 2: Full path to conda
# source /path/to/miniconda3/etc/profile.d/conda.sh
# conda activate yolo

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Set working directory
cd $SLURM_SUBMIT_DIR
# Or use absolute path:
# cd /path/to/CoffeeBeanDataset

# Create logs directory
mkdir -p logs

# ============================================================
# Training Configuration
# ============================================================
MODEL="yolov11n"
EPOCHS=100
BATCH_SIZE=16
IMG_SIZE=640
DATA_YAML="dataset.yaml"
PROJECT="runs/train"
NAME="${MODEL}_coffee_${SLURM_JOB_ID}"

echo "========================================"
echo "Training Configuration:"
echo "  Model: $MODEL"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Image size: $IMG_SIZE"
echo "  Dataset: $DATA_YAML"
echo "========================================"

# ============================================================
# Run Training
# ============================================================
python scripts/train_yolov11.py \
    --model $MODEL \
    --data $DATA_YAML \
    --epochs $EPOCHS \
    --batch $BATCH_SIZE \
    --imgsz $IMG_SIZE \
    --device 0 \
    --workers 8 \
    --project $PROJECT \
    --name $NAME \
    --patience 50

# ============================================================
# Post-training
# ============================================================
echo "========================================"
echo "Training completed!"
echo "End time: $(date)"
echo "Results saved to: $PROJECT/$NAME"
echo "========================================"

# Copy best weights to models folder
if [ -f "$PROJECT/$NAME/weights/best.pt" ]; then
    cp "$PROJECT/$NAME/weights/best.pt" "models/${MODEL}_best.pt"
    echo "Best weights copied to: models/${MODEL}_best.pt"
fi
