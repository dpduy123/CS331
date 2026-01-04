"""
Batch Pre-annotation Script for Label Studio
Uses trained YOLOv8 model to generate predictions and import to Label Studio
"""

import os
import json
import argparse
from pathlib import Path
from ultralytics import YOLO
from label_studio_sdk import Client
from PIL import Image
from tqdm import tqdm

# Class mapping - matches your Label Studio labels
CLASS_NAMES = {
    0: "riped",
    1: "unriped",
    2: "semi-riped",
    3: "over-riped",
    4: "barely-riped"
}


def predict_images(model_path, image_dir, confidence=0.25, output_dir='predictions'):
    """
    Run YOLOv8 predictions on all images in directory

    Args:
        model_path: Path to trained model weights
        image_dir: Directory containing images
        confidence: Confidence threshold
        output_dir: Directory to save predictions

    Returns:
        Dictionary mapping image paths to predictions
    """
    print(f"\n📸 Loading model: {model_path}")
    model = YOLO(model_path)

    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f'*{ext}'))
        image_files.extend(image_dir.glob(f'*{ext.upper()}'))

    print(f"📂 Found {len(image_files)} images")

    predictions_dict = {}

    print("\n🔍 Running predictions...")
    for img_path in tqdm(image_files):
        # Run prediction
        results = model.predict(
            source=str(img_path),
            conf=confidence,
            verbose=False
        )

        # Get image dimensions
        img = Image.open(img_path)
        img_width, img_height = img.size

        # Parse predictions
        predictions = []
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())

                x1, y1, x2, y2 = box
                predictions.append({
                    'class': cls,
                    'class_name': CLASS_NAMES.get(cls, f'class_{cls}'),
                    'confidence': conf,
                    'bbox': {
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2)
                    },
                    'bbox_normalized': {
                        'x': float(x1 / img_width),
                        'y': float(y1 / img_height),
                        'width': float((x2 - x1) / img_width),
                        'height': float((y2 - y1) / img_height)
                    }
                })

        predictions_dict[str(img_path)] = {
            'image_path': str(img_path),
            'image_name': img_path.name,
            'width': img_width,
            'height': img_height,
            'predictions': predictions
        }

    # Save predictions to JSON
    output_file = output_dir / 'predictions.json'
    with open(output_file, 'w') as f:
        json.dump(predictions_dict, f, indent=2)

    print(f"\n✓ Predictions saved to: {output_file}")
    return predictions_dict


def convert_to_label_studio_format(predictions_dict, image_url_prefix='/data/local-files/?d=', use_absolute_path=False, use_relative_path=False, base_dir=None, use_cloud_storage=False, storage_path=None):
    """
    Convert predictions to Label Studio import format

    Args:
        predictions_dict: Dictionary of predictions
        image_url_prefix: URL prefix for images in Label Studio
        use_absolute_path: If True, use absolute file paths instead of URL prefix
        use_relative_path: If True, use relative paths from base_dir
        base_dir: Base directory for relative paths
        use_cloud_storage: If True, use cloud storage path format
        storage_path: Cloud storage path prefix (e.g., 's3://bucket' or local path)

    Returns:
        List of tasks in Label Studio format
    """
    import urllib.parse

    tasks = []

    for img_path, pred_data in predictions_dict.items():
        # Determine image path format
        if use_cloud_storage and storage_path:
            # Use cloud storage format - reference images from configured storage
            # This tells Label Studio to load from cloud storage, not local files
            filename = pred_data['image_name']
            image_path = f"{storage_path}/{filename}" if not storage_path.endswith('/') else f"{storage_path}{filename}"
        elif use_relative_path and base_dir:
            # Use relative path from base directory
            abs_path = Path(pred_data['image_path']).absolute()
            base_path = Path(base_dir).absolute()
            try:
                rel_path = abs_path.relative_to(base_path)
                # URL encode the relative path for Label Studio
                image_path = f"{image_url_prefix}{urllib.parse.quote(str(rel_path))}"
            except ValueError:
                # Fallback to filename only if not relative
                encoded_name = urllib.parse.quote(pred_data['image_name'])
                image_path = f"{image_url_prefix}{encoded_name}"
        elif use_absolute_path:
            # Use absolute file path (requires Label Studio local file serving)
            image_path = str(Path(pred_data['image_path']).absolute())
        else:
            # Use URL format with proper encoding (default)
            encoded_name = urllib.parse.quote(pred_data['image_name'])
            image_path = f"{image_url_prefix}{encoded_name}"

        # Create task
        task = {
            'data': {
                'image': image_path
            },
            'predictions': [{
                'model_version': 'yolov8-preannotation',
                'result': []
            }]
        }

        # Add predictions
        for pred in pred_data['predictions']:
            bbox_norm = pred['bbox_normalized']

            annotation = {
                'from_name': 'label',
                'to_name': 'image',
                'type': 'rectanglelabels',
                'value': {
                    'rectanglelabels': [pred['class_name']],
                    'x': bbox_norm['x'] * 100,
                    'y': bbox_norm['y'] * 100,
                    'width': bbox_norm['width'] * 100,
                    'height': bbox_norm['height'] * 100
                },
                'score': pred['confidence']
            }

            task['predictions'][0]['result'].append(annotation)

        tasks.append(task)

    return tasks


def import_to_label_studio(tasks, ls_url, api_key, project_id):
    """
    Import pre-annotated tasks to Label Studio

    Args:
        tasks: List of tasks in Label Studio format
        ls_url: Label Studio URL
        api_key: API key
        project_id: Project ID
    """
    print(f"\n📤 Connecting to Label Studio: {ls_url}")
    ls = Client(url=ls_url, api_key=api_key)

    print(f"📋 Importing {len(tasks)} tasks to project {project_id}")

    project = ls.get_project(project_id)
    project.import_tasks(tasks)

    print(f"✓ Import complete!")


def main():
    parser = argparse.ArgumentParser(description='Pre-annotate images with YOLOv8')

    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained YOLOv8 model (.pt file)')
    parser.add_argument('--images', type=str, required=True,
                        help='Directory containing images to predict')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--output', type=str, default='predictions',
                        help='Output directory for predictions')

    # Label Studio integration
    parser.add_argument('--import-to-ls', action='store_true',
                        help='Import predictions to Label Studio')
    parser.add_argument('--ls-url', type=str, default='http://localhost:8080',
                        help='Label Studio URL')
    parser.add_argument('--ls-api-key', type=str,
                        help='Label Studio API key')
    parser.add_argument('--ls-project-id', type=int,
                        help='Label Studio project ID')
    parser.add_argument('--image-url-prefix', type=str,
                        default='/data/local-files/?d=',
                        help='URL prefix for images in Label Studio')
    parser.add_argument('--use-absolute-path', action='store_true',
                        help='Use absolute file paths (requires Label Studio local file serving)')
    parser.add_argument('--use-relative-path', action='store_true',
                        help='Use relative paths from base directory (recommended)')
    parser.add_argument('--base-dir', type=str,
                        help='Base directory for relative paths (default: current directory)')
    parser.add_argument('--use-cloud-storage', action='store_true',
                        help='Use cloud storage format (for Label Studio cloud storage)')
    parser.add_argument('--storage-path', type=str,
                        help='Storage path prefix (e.g., "/Users/path/folder" for local storage)')

    args = parser.parse_args()

    # Run predictions
    predictions_dict = predict_images(
        model_path=args.model,
        image_dir=args.images,
        confidence=args.conf,
        output_dir=args.output
    )

    # Determine base directory for relative paths
    base_dir = args.base_dir if args.base_dir else os.getcwd()

    # Convert to Label Studio format
    tasks = convert_to_label_studio_format(
        predictions_dict,
        image_url_prefix=args.image_url_prefix,
        use_absolute_path=args.use_absolute_path,
        use_relative_path=args.use_relative_path,
        base_dir=base_dir,
        use_cloud_storage=args.use_cloud_storage,
        storage_path=args.storage_path
    )

    # Save tasks to JSON
    tasks_file = Path(args.output) / 'label_studio_tasks.json'
    with open(tasks_file, 'w') as f:
        json.dump(tasks, f, indent=2)
    print(f"✓ Label Studio format saved to: {tasks_file}")

    # Import to Label Studio
    if args.import_to_ls:
        if not args.ls_api_key or not args.ls_project_id:
            print("❌ Error: --ls-api-key and --ls-project-id required for import")
            return

        import_to_label_studio(
            tasks=tasks,
            ls_url=args.ls_url,
            api_key=args.ls_api_key,
            project_id=args.ls_project_id
        )

    print("\n" + "="*60)
    print("🎉 Pre-annotation Complete!")
    print("="*60)
    print(f"Total images processed: {len(predictions_dict)}")
    print(f"Total detections: {sum(len(p['predictions']) for p in predictions_dict.values())}")
    print(f"\nNext steps:")
    print(f"1. Review predictions in Label Studio")
    print(f"2. Correct any errors")
    print(f"3. Export corrected annotations")
    print(f"4. Retrain model with improved data")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
