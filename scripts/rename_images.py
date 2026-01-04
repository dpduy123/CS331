"""
Rename all images in a folder to sequential numbers (1.jpg, 2.jpg, 3.jpg, ...)
"""

import os
import shutil
from pathlib import Path

def rename_images(source_dir: str, dry_run: bool = True):
    """
    Rename all images to sequential numbers

    Args:
        source_dir: Directory containing images
        dry_run: If True, only print what would happen without renaming
    """
    source_path = Path(source_dir)

    if not source_path.exists():
        print(f"ERROR: Directory not found: {source_dir}")
        return

    # Get all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    images = []

    for f in source_path.iterdir():
        if f.is_file() and f.suffix.lower() in image_extensions:
            images.append(f)

    print(f"Found {len(images)} images in {source_dir}")

    if len(images) == 0:
        return

    # Sort by name (try to preserve some order)
    images.sort(key=lambda x: x.name)

    print(f"\n{'DRY RUN - ' if dry_run else ''}Renaming plan:")
    print("=" * 60)

    # First pass: rename to temp names to avoid conflicts
    temp_names = []
    for i, img in enumerate(images, 1):
        ext = img.suffix.lower()
        if ext == '.jpeg':
            ext = '.jpg'
        temp_name = f"__temp_{i:04d}{ext}"
        temp_names.append((img, source_path / temp_name))

    # Second pass: rename to final names
    final_names = []
    for i, (original, temp) in enumerate(temp_names, 1):
        ext = temp.suffix
        final_name = source_path / f"{i}{ext}"
        final_names.append((original, temp, final_name))

    # Show plan
    for i, (original, temp, final) in enumerate(final_names[:20], 1):
        print(f"  {original.name:50} -> {final.name}")

    if len(final_names) > 20:
        print(f"  ... and {len(final_names) - 20} more")

    print("=" * 60)
    print(f"Total: {len(final_names)} files will be renamed")

    if dry_run:
        print("\nThis is a DRY RUN. No files were changed.")
        print("Run with --execute to actually rename files.")
        return

    # Execute renaming
    print("\nExecuting rename...")

    # Step 1: Rename to temp names
    print("Step 1: Renaming to temporary names...")
    for original, temp, _ in final_names:
        shutil.move(str(original), str(temp))

    # Step 2: Rename to final names
    print("Step 2: Renaming to final names...")
    for _, temp, final in final_names:
        shutil.move(str(temp), str(final))

    print(f"\nDone! Renamed {len(final_names)} files.")
    print(f"Files are now named: 1.jpg, 2.jpg, ... {len(final_names)}.jpg")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rename images to sequential numbers")
    parser.add_argument("--dir", type=str,
                        default="/Users/nguyenloan/Desktop/CS331/CoffeeBeanDataset/Hình cà phê",
                        help="Directory containing images")
    parser.add_argument("--execute", action="store_true",
                        help="Actually rename files (default is dry-run)")

    args = parser.parse_args()

    rename_images(args.dir, dry_run=not args.execute)
