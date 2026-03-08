#!/usr/bin/env python3
"""
Batch resize all images to 512x512 using direct method.
OVERWRITES ORIGINAL IMAGES - no separate output folder.
"""

from pathlib import Path
from PIL import Image
from tqdm import tqdm
import sys

def resize_to_512_direct(image_path):
    """
    Simple resize to 512x512 - overwrites original image
    """
    img = Image.open(image_path)
    img_resized = img.resize((512, 512), Image.Resampling.LANCZOS)
    img_resized.save(image_path)  # Overwrite original

def batch_resize_dataset(directory: Path):
    """
    Resize all images in directory to 512x512 IN-PLACE
    WARNING: Overwrites original images!
    """
    directory = Path(directory)
    
    if not directory.exists():
        print(f"✗ Directory not found: {directory}")
        return False
    
    # Find all image files
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = sorted([
        f for f in directory.iterdir() 
        if f.is_file() and f.suffix.lower() in valid_exts
    ])
    
    if not image_files:
        print(f"✗ No images found in {directory}")
        return False
    
    print(f"Resizing {len(image_files)} images to 512×512 IN-PLACE...")
    print(f"WARNING: This will OVERWRITE original images!")
    
    # Safety confirmation
    response = input("\nContinue? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled.")
        return False
    
    failed = 0
    for img_path in tqdm(image_files, desc=f"Resizing {directory.name}"):
        try:
            resize_to_512_direct(img_path)
        except Exception as e:
            print(f"\n✗ Error processing {img_path.name}: {e}")
            failed += 1
    
    if failed > 0:
        print(f"\n⚠ {failed} images failed to process")
    
    success_count = len(image_files) - failed
    print(f"\n✓ Resized {success_count}/{len(image_files)} images")
    return True

# ===== CONFIGURATION =====
# Edit these paths - images will be resized IN-PLACE
RESIZE_DIRS = {
    "Real Images": Path("shared/datasets/coco/metricsDataset/images/"),
    "Canny Edges": Path("shared/datasets/coco/metricsDataset/edges/"),
    "ControlNet Generated": Path("shared/datasets/coco/metricsDataset/generated_images_ControlNet/"),
    "SD1.5 Generated": Path("shared/datasets/coco/metricsDataset/generated_images_SD15/"),
}

def main():
    print("=" * 70)
    print("BATCH RESIZE TO 512×512 (OVERWRITES ORIGINAL IMAGES)")
    print("=" * 70)
    print("\n⚠️  WARNING: This will PERMANENTLY modify your original images!")
    print("   Make sure you have backups if needed.\n")
    
    all_success = True
    
    for name, directory in RESIZE_DIRS.items():
        print(f"\n{name}")
        print(f"Path: {directory}")
        print("-" * 70)
        
        success = batch_resize_dataset(directory)
        if not success:
            all_success = False
    
    print("\n" + "=" * 70)
    if all_success:
        print("✓ ALL RESIZING COMPLETE")
        print("\n✓ Images are now 512×512")
        print("✓ No path changes needed in qualityMetrics.py")
        print("\nNext: Run: python Quality_Metrics/qualityMetrics.py")
    else:
        print("✗ SOME RESIZING FAILED - Check errors above")
    print("=" * 70)

if __name__ == "__main__":
    main()