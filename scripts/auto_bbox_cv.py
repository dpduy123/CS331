"""
Auto-detect coffee bean bounding boxes using Computer Vision
Uses HSV color space + contour detection for fast, GPU-free detection
Generates Label Studio compatible pre-annotations
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


def auto_detect_coffee_beans(image_path,
                              min_area=50,
                              max_area=15000,
                              min_aspect_ratio=0.4,
                              max_aspect_ratio=2.5,
                              visualize=False):
    """
    Auto-detect coffee beans using traditional CV methods

    Args:
        image_path: Path to image
        min_area: Minimum contour area (pixels)
        max_area: Maximum contour area (pixels)
        min_aspect_ratio: Minimum width/height ratio
        max_aspect_ratio: Maximum width/height ratio
        visualize: Show detection results

    Returns:
        List of bounding boxes with confidence scores
    """

    # Read image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"⚠️  Could not read image: {image_path}")
        return []

    height, width = img.shape[:2]

    # Convert to different color spaces for better detection
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Create multiple masks for different ripeness stages
    masks = []

    # Mask 1: Green beans (unripe)
    lower_green = np.array([25, 20, 20])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    masks.append(mask_green)

    # Mask 2: Yellow/Orange beans (semi-ripe)
    lower_yellow = np.array([15, 40, 40])
    upper_yellow = np.array([35, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    masks.append(mask_yellow)

    # Mask 3: Red/Brown beans (ripe)
    lower_red1 = np.array([0, 30, 30])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 30, 30])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.bitwise_or(
        cv2.inRange(hsv, lower_red1, upper_red1),
        cv2.inRange(hsv, lower_red2, upper_red2)
    )
    masks.append(mask_red)

    # Mask 4: Dark brown/black beans (overripe)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 80])
    mask_dark = cv2.inRange(hsv, lower_dark, upper_dark)
    masks.append(mask_dark)

    # Combine all masks
    combined_mask = np.zeros_like(mask_green)
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Morphological operations to clean up mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Remove noise with median blur
    combined_mask = cv2.medianBlur(combined_mask, 5)

    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # Filter and extract bounding boxes
    bboxes = []
    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter by area
        if min_area < area < max_area:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Calculate aspect ratio
            aspect_ratio = w / h if h > 0 else 0

            # Filter by aspect ratio (coffee beans are roughly circular/oval)
            if min_aspect_ratio < aspect_ratio < max_aspect_ratio:
                # Calculate confidence based on circularity
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    confidence = min(circularity, 1.0)  # 1.0 = perfect circle
                else:
                    confidence = 0.5

                # Convert to percentage for Label Studio
                bbox = {
                    'x': (x / width) * 100,
                    'y': (y / height) * 100,
                    'width': (w / width) * 100,
                    'height': (h / height) * 100,
                    'confidence': float(confidence),
                    'area': int(area),
                    'aspect_ratio': float(aspect_ratio)
                }
                bboxes.append(bbox)

    # Visualize if requested
    if visualize:
        vis_img = img.copy()
        for bbox in bboxes:
            x = int(bbox['x'] / 100 * width)
            y = int(bbox['y'] / 100 * height)
            w = int(bbox['width'] / 100 * width)
            h = int(bbox['height'] / 100 * height)

            # Color based on confidence
            color = (0, int(255 * bbox['confidence']), int(255 * (1 - bbox['confidence'])))
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 2)
            cv2.putText(vis_img, f"{bbox['confidence']:.2f}", (x, y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Show result
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        plt.title(f'Detected {len(bboxes)} beans')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'detection_preview_{Path(image_path).stem}.png', dpi=150)
        plt.close()

    return bboxes


def create_labelstudio_tasks(image_dir,
                             output_file='preannotations.json',
                             min_confidence=0.3,
                             preview_count=5,
                             **detection_params):
    """
    Generate Label Studio pre-annotation tasks for all images

    Args:
        image_dir: Directory containing images
        output_file: Output JSON file path
        min_confidence: Minimum confidence to include
        preview_count: Number of images to generate preview for
        **detection_params: Parameters for auto_detect_coffee_beans
    """

    image_dir = Path(image_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

    # Get all images
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.glob(f'*{ext}'))

    print(f"\n📸 Found {len(image_files)} images")
    print(f"🔍 Starting auto-detection...\n")

    tasks = []
    total_detections = 0

    # Process each image
    for idx, img_path in enumerate(tqdm(image_files, desc="Processing images")):
        # Auto-detect bounding boxes
        visualize = idx < preview_count  # Create preview for first N images
        bboxes = auto_detect_coffee_beans(
            img_path,
            visualize=visualize,
            **detection_params
        )

        # Filter by confidence
        bboxes = [b for b in bboxes if b['confidence'] >= min_confidence]

        total_detections += len(bboxes)

        # Create Label Studio task
        task = {
            'data': {
                'image': f'/data/local-files/?d={img_path.name}'
            },
            'predictions': [{
                'model_version': 'auto-cv-detection-v1',
                'score': sum(b['confidence'] for b in bboxes) / len(bboxes) if bboxes else 0,
                'result': []
            }]
        }

        # Add each bbox as a prediction
        for bbox in bboxes:
            prediction = {
                'from_name': 'label',
                'to_name': 'image',
                'type': 'rectanglelabels',
                'value': {
                    'rectanglelabels': ['unclassified'],  # User will assign ripeness class
                    'x': bbox['x'],
                    'y': bbox['y'],
                    'width': bbox['width'],
                    'height': bbox['height']
                },
                'score': bbox['confidence']
            }
            task['predictions'][0]['result'].append(prediction)

        tasks.append(task)

    # Save to JSON
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)

    # Print statistics
    avg_detections = total_detections / len(image_files) if image_files else 0

    print(f"\n{'='*60}")
    print(f"✅ Auto-detection Complete!")
    print(f"{'='*60}")
    print(f"📊 Statistics:")
    print(f"  - Total images processed: {len(image_files)}")
    print(f"  - Total detections: {total_detections}")
    print(f"  - Average per image: {avg_detections:.1f} beans")
    print(f"  - Output file: {output_path.absolute()}")
    print(f"\n📝 Next steps:")
    print(f"  1. Review preview images: detection_preview_*.png")
    print(f"  2. Import {output_file} to Label Studio:")
    print(f"     - Go to your project")
    print(f"     - Click 'Import'")
    print(f"     - Upload {output_file}")
    print(f"  3. Review predictions (shown as light-colored boxes)")
    print(f"  4. For each image:")
    print(f"     - Accept good boxes ✓")
    print(f"     - Delete false positives ✗")
    print(f"     - Add missed beans manually")
    print(f"     - Assign ripeness labels")
    print(f"  5. Export corrected annotations")
    print(f"{'='*60}\n")

    return tasks


def tune_parameters(image_path):
    """
    Interactive parameter tuning for detection
    Shows different parameter combinations
    """

    print(f"\n🔧 Parameter Tuning Mode")
    print(f"Testing different parameters on: {image_path}\n")

    # Test different parameter combinations
    param_sets = [
        {'min_area': 50, 'max_area': 10000, 'name': 'Small beans'},
        {'min_area': 100, 'max_area': 15000, 'name': 'Medium beans'},
        {'min_area': 200, 'max_area': 20000, 'name': 'Large beans'},
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # Original image
    img = cv2.imread(str(image_path))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Test each parameter set
    for idx, params in enumerate(param_sets, 1):
        if idx >= len(axes):
            break

        bboxes = auto_detect_coffee_beans(image_path, **params)

        # Draw boxes
        vis_img = img.copy()
        height, width = img.shape[:2]
        for bbox in bboxes:
            x = int(bbox['x'] / 100 * width)
            y = int(bbox['y'] / 100 * height)
            w = int(bbox['width'] / 100 * width)
            h = int(bbox['height'] / 100 * height)
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        axes[idx].imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(f"{params['name']}: {len(bboxes)} detections")
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('parameter_tuning_results.png', dpi=150)
    print(f"✓ Results saved to parameter_tuning_results.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description='Auto-detect coffee bean bounding boxes using Computer Vision'
    )

    parser.add_argument('--images', type=str, required=True,
                        help='Directory containing images')
    parser.add_argument('--output', type=str, default='preannotations.json',
                        help='Output JSON file for Label Studio')

    # Detection parameters
    parser.add_argument('--min-area', type=int, default=50,
                        help='Minimum bean area in pixels (default: 50)')
    parser.add_argument('--max-area', type=int, default=15000,
                        help='Maximum bean area in pixels (default: 15000)')
    parser.add_argument('--min-confidence', type=float, default=0.3,
                        help='Minimum confidence threshold (default: 0.3)')
    parser.add_argument('--min-aspect', type=float, default=0.4,
                        help='Minimum aspect ratio (default: 0.4)')
    parser.add_argument('--max-aspect', type=float, default=2.5,
                        help='Maximum aspect ratio (default: 2.5)')

    # Utility options
    parser.add_argument('--preview', type=int, default=5,
                        help='Number of preview images to generate (default: 5)')
    parser.add_argument('--tune', type=str,
                        help='Path to single image for parameter tuning')

    args = parser.parse_args()

    # Parameter tuning mode
    if args.tune:
        tune_parameters(args.tune)
        return

    # Normal detection mode
    detection_params = {
        'min_area': args.min_area,
        'max_area': args.max_area,
        'min_aspect_ratio': args.min_aspect,
        'max_aspect_ratio': args.max_aspect,
    }

    create_labelstudio_tasks(
        image_dir=args.images,
        output_file=args.output,
        min_confidence=args.min_confidence,
        preview_count=args.preview,
        **detection_params
    )


if __name__ == '__main__':
    main()
