"""
Flask App for Coffee Bean Pseudo-Labeling
Upload images, detect coffee beans using YOLOv8, display results with bounding boxes
"""

import os
import uuid
import zipfile
from io import BytesIO
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
import cv2
import numpy as np

# Try to import YOLO, handle if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. Running in demo mode.")

app = Flask(__name__)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.config['UPLOAD_FOLDER'] = Path(__file__).parent / 'static' / 'uploads'
app.config['RESULTS_FOLDER'] = Path(__file__).parent / 'static' / 'results'

# Ensure folders exist
app.config['UPLOAD_FOLDER'].mkdir(parents=True, exist_ok=True)
app.config['RESULTS_FOLDER'].mkdir(parents=True, exist_ok=True)

# Model configuration
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Available models configuration
AVAILABLE_MODELS = {
    "yolov8n": {
        "name": "YOLOv8n (Nano)",
        "path": MODELS_DIR / "yolov8n_best.pt",
        "description": "C2f Module - 3.2M params"
    },
    "yolov11n": {
        "name": "YOLOv11n (Nano)",
        "path": MODELS_DIR / "yolov11n_best.pt",
        "description": "Latest YOLO with C3K2 blocks - 2.6M params"
    },
    "ssvit": {
        "name": "SSViT-YOLOv11n",
        "path": MODELS_DIR / "ssvit_best.pt",
        "description": "Custom ViT + YOLO fusion - 2.16M params"
    }
}

CONF_THRESHOLD = 0.25

# Classes
CLASSES = ["barely-riped", "over-riped", "riped", "semi-riped", "unriped"]

# Colors for each class (BGR format for OpenCV)
# Must match CSS settings in style.css
COLORS = {
    "barely-riped": (0, 0, 255),     # Red (#FF0000) in BGR
    "over-riped": (16, 16, 117),     # Dark Red (#751010) in BGR
    "riped": (0, 165, 255),          # Orange (#FFA500) in BGR
    "semi-riped": (0, 255, 255),     # Yellow (#FFFF00) in BGR
    "unriped": (0, 255, 0),          # Green (#00FF00) in BGR
}

# RGB colors for frontend (same colors, RGB format)
# Must match CSS settings in style.css
COLORS_RGB = {
    "barely-riped": (255, 0, 0),     # Red (#FF0000)
    "over-riped": (117, 16, 16),     # Dark Red (#751010)
    "riped": (255, 165, 0),          # Orange (#FFA500)
    "semi-riped": (255, 255, 0),     # Yellow (#FFFF00)
    "unriped": (0, 255, 0),          # Green (#00FF00)
}

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

# Global model instances (cached)
loaded_models = {}
current_model_key = None
model = None


def get_available_models():
    """Get list of available models (only those with existing weight files)"""
    available = {}
    for key, info in AVAILABLE_MODELS.items():
        available[key] = {
            "name": info["name"],
            "description": info["description"],
            "available": info["path"].exists()
        }
    return available


def load_model(model_key=None):
    """Load YOLO model by key"""
    global model, current_model_key, loaded_models

    if not YOLO_AVAILABLE:
        return False, "Ultralytics not installed"

    # Default to first available model
    if model_key is None:
        for key, info in AVAILABLE_MODELS.items():
            if info["path"].exists():
                model_key = key
                break

    if model_key is None:
        return False, "No models available"

    # Check if model exists
    if model_key not in AVAILABLE_MODELS:
        return False, f"Unknown model: {model_key}"

    model_path = AVAILABLE_MODELS[model_key]["path"]

    if not model_path.exists():
        return False, f"Model file not found: {model_path}"

    # Check if already loaded
    if model_key in loaded_models:
        model = loaded_models[model_key]
        current_model_key = model_key
        return True, f"Loaded {AVAILABLE_MODELS[model_key]['name']}"

    # Load new model
    try:
        loaded_models[model_key] = YOLO(str(model_path))
        model = loaded_models[model_key]
        current_model_key = model_key
        return True, f"Loaded {AVAILABLE_MODELS[model_key]['name']}"
    except Exception as e:
        return False, f"Error loading model: {str(e)}"


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def process_image(image_path, session_id):
    """
    Process image with YOLO model and return results
    """
    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None, "Cannot read image"

    img_h, img_w = img.shape[:2]

    # Run inference
    if model is None:
        return None, None, "Model not loaded"

    results = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        verbose=False
    )

    result = results[0]
    boxes = result.boxes

    # Prepare detection data
    detections = []

    # Draw bounding boxes on image
    img_with_boxes = img.copy()

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        # Get class name (handle if using default model)
        if cls_id < len(CLASSES):
            cls_name = CLASSES[cls_id]
        else:
            cls_name = f"class_{cls_id}"

        # Get normalized coordinates (YOLO format)
        x_center, y_center, width, height = box.xywhn[0].tolist()

        # Get pixel coordinates for drawing
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Get color
        color = COLORS.get(cls_name, (255, 255, 255))

        # Draw rectangle
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label = f"{cls_name} {conf:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_with_boxes, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)

        # Draw label text
        cv2.putText(img_with_boxes, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Store detection info
        detections.append({
            'class_id': cls_id,
            'class_name': cls_name,
            'confidence': conf,
            'x_center': x_center,
            'y_center': y_center,
            'width': width,
            'height': height,
            'bbox': [x1, y1, x2, y2]
        })

    # Save result image
    result_filename = f"{session_id}_{Path(image_path).stem}_result.jpg"
    result_path = app.config['RESULTS_FOLDER'] / result_filename
    cv2.imwrite(str(result_path), img_with_boxes)

    # Generate YOLO format label
    label_content = ""
    for det in detections:
        label_content += f"{det['class_id']} {det['x_center']:.6f} {det['y_center']:.6f} {det['width']:.6f} {det['height']:.6f}\n"

    return {
        'result_image': result_filename,
        'detections': detections,
        'label_content': label_content,
        'num_detections': len(detections)
    }, result_path, None


@app.route('/')
def index():
    """Main page"""
    models = get_available_models()
    return render_template('index.html', classes=CLASSES, colors=COLORS,
                          models=models, current_model=current_model_key)


@app.route('/models', methods=['GET'])
def list_models():
    """List available models"""
    return jsonify({
        'models': get_available_models(),
        'current': current_model_key
    })


@app.route('/models/select', methods=['POST'])
def select_model():
    """Select and load a model"""
    data = request.json
    model_key = data.get('model')

    if not model_key:
        return jsonify({'error': 'No model specified'}), 400

    success, message = load_model(model_key)

    if success:
        return jsonify({
            'success': True,
            'message': message,
            'current': current_model_key
        })
    else:
        return jsonify({
            'success': False,
            'error': message
        }), 400


@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file upload and processing"""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files[]')

    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files selected'}), 400

    # Generate session ID for this batch
    session_id = str(uuid.uuid4())[:8]

    results = []

    for file in files:
        if file and allowed_file(file.filename):
            # Save uploaded file
            filename = secure_filename(file.filename)
            unique_filename = f"{session_id}_{filename}"
            upload_path = app.config['UPLOAD_FOLDER'] / unique_filename
            file.save(str(upload_path))

            # Process image
            result, result_path, error = process_image(upload_path, session_id)

            if error:
                results.append({
                    'filename': filename,
                    'error': error
                })
            else:
                results.append({
                    'filename': filename,
                    'original_image': unique_filename,
                    'result_image': result['result_image'],
                    'detections': result['detections'],
                    'label_content': result['label_content'],
                    'num_detections': result['num_detections']
                })
        else:
            results.append({
                'filename': file.filename,
                'error': 'Invalid file type'
            })

    return jsonify({
        'session_id': session_id,
        'results': results
    })


@app.route('/download_label/<filename>')
def download_label(filename):
    """Download single label file"""
    # Get label content from request args
    label_content = request.args.get('content', '')

    # Create temporary label file
    label_filename = Path(filename).stem + '.txt'
    label_path = app.config['RESULTS_FOLDER'] / label_filename

    with open(label_path, 'w') as f:
        f.write(label_content)

    return send_file(
        label_path,
        as_attachment=True,
        download_name=label_filename
    )


@app.route('/download_all', methods=['POST'])
def download_all():
    """Download all accepted labels as zip"""
    data = request.json

    if not data or 'labels' not in data:
        return jsonify({'error': 'No labels provided'}), 400

    labels = data['labels']
    session_id = data.get('session_id', str(uuid.uuid4())[:8])

    # Create zip file
    zip_filename = f"labels_{session_id}.zip"
    zip_path = app.config['RESULTS_FOLDER'] / zip_filename

    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for label in labels:
            filename = label.get('filename', 'unknown')
            content = label.get('content', '')

            label_filename = Path(filename).stem + '.txt'
            zipf.writestr(label_filename, content)

    return send_file(
        zip_path,
        as_attachment=True,
        download_name=zip_filename
    )


@app.route('/download_image/<filename>')
def download_image(filename):
    """Download result image with bounding boxes"""
    image_path = app.config['RESULTS_FOLDER'] / filename

    if not image_path.exists():
        return jsonify({'error': 'Image not found'}), 404

    return send_file(
        image_path,
        as_attachment=True,
        download_name=filename
    )


@app.route('/boxonly/<filename>')
def get_boxonly_image(filename):
    """
    Generate and return image with bounding boxes only (no text labels).
    Uses the original uploaded image and redraws boxes without labels.
    """
    # Parse filename to get session_id and original filename
    # Format: {session_id}_{session_id}_{original_stem}_result.jpg
    # Example: 882847b6_882847b6_14_result.jpg
    parts = filename.split('_')
    if len(parts) < 3:
        return jsonify({'error': 'Invalid filename format'}), 400

    session_id = parts[0]
    # The format is: {session_id}_{session_id}_{original_stem}_result.jpg
    # So we need to skip first two parts (both are session_id) and remove 'result' at end
    if len(parts) >= 4 and parts[0] == parts[1]:
        # Double session_id format
        original_stem = '_'.join(parts[2:-1])  # Skip both session_ids, remove 'result'
    else:
        # Single session_id format
        original_stem = '_'.join(parts[1:-1])  # Remove session_id and 'result'

    # Find the original uploaded image
    original_image = None
    for ext in ['jpg', 'jpeg', 'png', 'bmp', 'webp']:
        # Try with session_id prefix (how files are saved)
        potential_path = app.config['UPLOAD_FOLDER'] / f"{session_id}_{original_stem}.{ext}"
        if potential_path.exists():
            original_image = potential_path
            break

    if original_image is None:
        return jsonify({'error': 'Original image not found'}), 404

    # Read original image
    img = cv2.imread(str(original_image))
    if img is None:
        return jsonify({'error': 'Cannot read image'}), 500

    # Run inference to get boxes
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    results = model.predict(
        source=str(original_image),
        conf=CONF_THRESHOLD,
        verbose=False
    )

    result = results[0]
    boxes = result.boxes

    # Draw only bounding boxes (no labels, no text)
    for box in boxes:
        cls_id = int(box.cls[0])

        # Get class name
        if cls_id < len(CLASSES):
            cls_name = CLASSES[cls_id]
        else:
            cls_name = f"class_{cls_id}"

        # Get pixel coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # Get color
        color = COLORS.get(cls_name, (255, 255, 255))

        # Draw rectangle only (thicker for fullscreen view, NO TEXT)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

    # Encode image to bytes
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])

    return send_file(
        BytesIO(buffer.tobytes()),
        mimetype='image/jpeg'
    )


@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Clean up old files"""
    import time

    max_age = 3600  # 1 hour
    current_time = time.time()

    for folder in [app.config['UPLOAD_FOLDER'], app.config['RESULTS_FOLDER']]:
        for file in folder.glob('*'):
            if file.is_file():
                file_age = current_time - file.stat().st_mtime
                if file_age > max_age:
                    file.unlink()

    return jsonify({'status': 'cleaned'})


# Initialize model on startup
with app.app_context():
    if not load_model():
        print("Warning: Could not load model. Some features may not work.")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
