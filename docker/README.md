# YOLOv8 Coffee Bean Training - Docker

Train YOLOv8 models for coffee bean ripeness detection using Docker.

## Files

```
docker/
├── Dockerfile          # CPU training image
├── Dockerfile.gpu      # GPU training image (NVIDIA)
├── docker-compose.yml  # Docker Compose config
├── train.py            # Training script
├── run.sh              # Easy runner script
└── README.md           # This file
```

## Quick Start

### Option 1: Using run.sh (Recommended)

```bash
cd docker

# CPU training
./run.sh cpu 100 4 yolov8n
#         │   │  │  └── model (yolov8n, yolov8s, yolov8m)
#         │   │  └── batch size
#         │   └── epochs
#         └── mode (cpu or gpu)

# GPU training (requires nvidia-docker)
./run.sh gpu 100 16 yolov8n
```

### Option 2: Using Docker Compose

```bash
cd docker

# CPU training
docker-compose up yolo-cpu

# GPU training
docker-compose up yolo-gpu
```

### Option 3: Manual Docker commands

```bash
cd docker

# Build image
docker build -t coffee-yolo -f Dockerfile .

# Run training
docker run --rm \
    -v $(pwd)/../kaggle_dataset_round1:/app/data:ro \
    -v $(pwd)/runs:/app/runs \
    -v $(pwd)/train.py:/app/train.py:ro \
    coffee-yolo \
    python /app/train.py --model yolov8n --epochs 100 --batch 4
```

## Training Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model` | YOLOv8 model (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x) | yolov8n |
| `--epochs` | Number of training epochs | 100 |
| `--batch` | Batch size (reduce if OOM) | 4 |
| `--imgsz` | Image size | 640 |
| `--device` | Device (cpu, 0, 1) | auto |
| `--patience` | Early stopping patience | 20 |
| `--no-amp` | Disable mixed precision | False |

## GPU Training (NVIDIA)

Requirements:
- NVIDIA GPU
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
# Install nvidia-container-toolkit
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU is accessible
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi

# Run GPU training
./run.sh gpu 100 16 yolov8n
```

## Output

Training results are saved to `docker/runs/`:

```
runs/
└── coffee_train/
    ├── weights/
    │   ├── best.pt      # Best model
    │   └── last.pt      # Last checkpoint
    ├── results.csv
    ├── confusion_matrix.png
    └── ...
```

## Memory Issues

If you encounter OOM (Out of Memory):

1. **Reduce batch size**: `./run.sh cpu 100 2 yolov8n`
2. **Use smaller model**: `yolov8n` instead of `yolov8s`
3. **Reduce image size**: Add `--imgsz 480` in train.py

## Troubleshooting

### "Dataset not found"
Make sure `kaggle_dataset_round1/` exists in the project root with this structure:
```
kaggle_dataset_round1/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### "Permission denied"
```bash
chmod +x run.sh
```

### GPU not detected
```bash
# Check if nvidia-docker is installed
docker run --rm --gpus all nvidia/cuda:12.1-base nvidia-smi
```
