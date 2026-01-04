# YOLOv8 Training Docker Image
# Supports both CPU and GPU (NVIDIA)

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    ultralytics \
    opencv-python-headless \
    kagglehub \
    matplotlib \
    pyyaml \
    numpy

# Create directories
RUN mkdir -p /app/data /app/runs /app/weights

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV YOLO_VERBOSE=true

# Default command
CMD ["python", "-c", "from ultralytics import YOLO; print('YOLOv8 ready!')"]
