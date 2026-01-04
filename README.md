# Coffee Bean Ripeness Detection

A complete machine learning pipeline for detecting and classifying coffee bean ripeness levels using state-of-the-art object detection models. This project integrates data annotation, model training, active learning, and deployment for real-world agricultural applications.

## Overview

This project automates coffee bean ripeness classification into 5 categories to optimize harvest timing:

| Class | Description |
|-------|-------------|
| `barely-riped` | Beans just starting to ripen, mostly green with slight color change |
| `semi-riped` | Partially ripe, yellow-orange color |
| `riped` | Fully ripe beans, optimal for harvesting (red/cherry color) |
| `over-riped` | Beans past optimal ripeness, dark/brown color |
| `unriped` | Completely unripe, green beans |

## Features

- **Multiple YOLO Models**: Support for YOLOv8 and YOLOv11 variants (nano, small, medium, large)
- **Custom SSViT-YOLOv11n**: Vision Transformer fusion architecture with 84.54% mAP@0.5
- **Active Learning Pipeline**: Iterative improvement workflow for efficient annotation
- **Label Studio Integration**: Professional annotation platform with pre-annotation support
- **Web Demo Application**: Flask-based pseudo-labeling tool with real-time inference
- **Docker Support**: Containerized training for CPU and GPU environments
- **SLURM Integration**: HPC cluster job submission scripts

## Project Structure

```
CoffeeBeanDataset/
├── dataset.yaml              # YOLO dataset configuration
├── requirements.txt          # Python dependencies
├── scripts/                  # Core workflow scripts
│   ├── train_yolov11.py      # Primary training script
│   ├── prepare_dataset.py    # Dataset splitting
│   ├── active_learning.py    # Active learning workflow
│   ├── auto_bbox_cv.py       # Automatic bounding box detection
│   ├── predict_and_import.py # Batch prediction for Label Studio
│   └── visualize_annotations.py
├── ssvit_yolov11n/           # Custom SSViT-YOLOv11n architecture
├── DemoApp/                  # Flask web application
├── docker/                   # Docker containerization
├── docs/                     # Comprehensive documentation
├── models/                   # Pre-trained model weights
└── runs/                     # Training outputs
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA 11.7+ (for GPU training)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd CoffeeBeanDataset

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Dataset

Organize your images and labels in the following structure:

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Or use the dataset preparation script:

```bash
python scripts/prepare_dataset.py --source ./raw_data --output ./dataset --train 0.8 --val 0.15 --test 0.05
```

### 2. Train a Model

```bash
# Train YOLOv11 Nano (recommended for quick experiments)
python scripts/train_yolov11.py --model yolo11n.pt --epochs 100 --batch 16

# Train YOLOv11 Small (better accuracy)
python scripts/train_yolov11.py --model yolo11s.pt --epochs 100 --batch 8
```

### 3. Run Inference

```bash
# Using trained model
from ultralytics import YOLO

model = YOLO('yolo11n_coffee/weights/best.pt')
results = model.predict('path/to/image.jpg', conf=0.25)
```

### 4. Launch Demo App

```bash
cd DemoApp
python app.py
# Open http://localhost:5000 in browser
```

## Training with Docker

```bash
cd docker

# CPU training
./run.sh cpu 100 4 yolo11n

# GPU training (requires NVIDIA Docker)
./run.sh gpu 100 16 yolo11n
```

## Model Performance

### SSViT-YOLOv11n (Custom Architecture)

| Metric | Value |
|--------|-------|
| Precision | 81.1% |
| Recall | 77.4% |
| mAP@0.5 | 84.54% |
| FPS | 23 |
| Parameters | 2.16M |

## Active Learning Workflow

1. **Initial Annotation**: Manually annotate 50-100 images in Label Studio
2. **Train Model**: Train on annotated data
3. **Pre-annotate**: Generate predictions for unlabeled images
4. **Review & Correct**: Fix predictions in Label Studio
5. **Retrain**: Improve model with corrected annotations
6. **Repeat**: Continue until desired accuracy

```bash
# Run active learning round
python scripts/run_active_round.py --round 1 --batch_size 50
```

## Label Studio Setup

```bash
# Start Label Studio
./start_label_studio_python.sh

# Or manually
label-studio start --port 8080
```

Import predictions to Label Studio:

```bash
python scripts/predict_and_import.py --model yolo11n_coffee/weights/best.pt --images ./unlabeled --output ./predictions
```

## Documentation

Detailed guides are available in the `docs/` directory:

- [Quick Start Guide](docs/QUICK_START.md)
- [Complete YOLO Workflow](docs/YOLO_WORKFLOW.md)
- [Active Learning Guide](docs/ACTIVE_LEARNING_GUIDE.md)
- [Label Studio Setup](docs/LABEL_STUDIO_SETUP.md)
- [Auto Detection Guide](docs/AUTO_DETECTION_GUIDE.md)
- [Kaggle Dataset Guide](docs/KAGGLE_GUIDE.md)
- [SLURM Configuration](docs/slurm_server_config.md)

## Technologies

- **Object Detection**: Ultralytics YOLOv8/v11
- **Deep Learning**: PyTorch
- **Annotation**: Label Studio
- **Computer Vision**: OpenCV, Pillow
- **Web Framework**: Flask
- **Containerization**: Docker

## License

This project is for educational and research purposes.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO implementations
- [Label Studio](https://labelstud.io/) for annotation platform
- SSViT-YOLOv11 architecture inspired by recent Vision Transformer research
