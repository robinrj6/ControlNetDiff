#!/usr/bin/env python3
"""
Dataset Validation and Cleaning Script for ControlNet Training

This script:
1. Reads metadata.jsonl from your dataset directory
2. Validates that all referenced image files exist
3. Removes entries with missing files
4. Creates a backup of the original metadata
5. Saves the cleaned metadata

Usage:
    python validate_and_fix_dataset.py \
        --dataset_dir /path/to/coco/canny/ \
        --metadata_file metadata.jsonl
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple


def validate_dataset(
    dataset_dir: Path,
    metadata_file: Path,
    dry_run: bool = False,
    keep_one_caption_per_image: bool = True
) -> Tuple[List[Dict], List[Dict]]:
    """
    Validate dataset by checking if all referenced files exist.
    
    Args:
        dataset_dir: Path to dataset directory
        metadata_file: Path to metadata.jsonl file
        dry_run: If True, don't save changes (just report)
        keep_one_caption_per_image: If True, only keep first valid entry per image
    
    Returns:
        (valid_entries, invalid_entries)
    """
    
    print(f"📂 Dataset directory: {dataset_dir}")
    print(f"📋 Metadata file: {metadata_file}")
    print("-" * 60)
    
    if not dataset_dir.exists():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")
    
    if not metadata_file.exists():
        raise ValueError(f"Metadata file not found: {metadata_file}")
    
    valid_entries = []
    invalid_entries = []
    seen_valid_images = set()
    
    with open(metadata_file, 'r') as f:
        for line_num, line in enumerate(f, start=1):
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError as e:
                print(f"❌ Line {line_num}: Invalid JSON format")
                invalid_entries.append({"error": "JSON decode error", "line": line_num})
                continue
            
            # Check for required fields
            if 'image' not in entry or 'conditioning' not in entry or 'text' not in entry:
                print(f"❌ Line {line_num}: Missing required fields (image/conditioning/text)")
                invalid_entries.append(entry)
                continue
            
            # Check if files exist
            image_path = dataset_dir / entry['image']
            conditioning_path = dataset_dir / entry['conditioning']
            
            missing_files = []
            if not image_path.exists():
                missing_files.append(f"image: {entry['image']}")
            if not conditioning_path.exists():
                missing_files.append(f"conditioning: {entry['conditioning']}")
            
            if missing_files:
                print(f"⚠️  Line {line_num}: Missing files:")
                for missing in missing_files:
                    print(f"     - {missing}")
                invalid_entries.append(entry)
            else:
                image_key = entry["image"]
                if keep_one_caption_per_image and image_key in seen_valid_images:
                    print(f"⚠️  Line {line_num}: Duplicate caption for image '{image_key}' (keeping first, removing this one)")
                    invalid_entries.append({
                        "error": "duplicate_image",
                        "line": line_num,
                        "image": image_key,
                        "conditioning": entry.get("conditioning", ""),
                        "text": entry.get("text", "")
                    })
                    continue

                seen_valid_images.add(image_key)
                valid_entries.append(entry)
    
    print("-" * 60)
    print(f"\n✅ Valid entries: {len(valid_entries)}")
    print(f"❌ Invalid entries: {len(invalid_entries)}")
    print(f"📊 Total: {len(valid_entries) + len(invalid_entries)}")
    if keep_one_caption_per_image:
        duplicate_count = sum(1 for e in invalid_entries if isinstance(e, dict) and e.get("error") == "duplicate_image")
        print(f"🧹 Duplicate captions removed: {duplicate_count}")
    
    if invalid_entries:
        print(f"\n⚠️  {len(invalid_entries)} entries will be removed")
        if len(invalid_entries) <= 10:
            print("\nInvalid entries details:")
            for entry in invalid_entries:
                print(f"  - {entry}")
    
    return valid_entries, invalid_entries


def save_cleaned_metadata(
    valid_entries: List[Dict],
    metadata_file: Path,
    create_backup: bool = True
) -> None:
    """
    Save cleaned metadata to file, optionally creating a backup.
    
    Args:
        valid_entries: List of valid metadata entries
        metadata_file: Path to metadata.jsonl file
        create_backup: Whether to create backup of original
    """
    
    # Create backup
    if create_backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = metadata_file.parent / f"metadata_backup_{timestamp}.jsonl"
        
        import shutil
        shutil.copy(metadata_file, backup_file)
        print(f"\n💾 Backup created: {backup_file}")
    
    # Write cleaned metadata
    with open(metadata_file, 'w') as f:
        for entry in valid_entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"✅ Cleaned metadata saved: {metadata_file}")


def print_summary(valid_entries: List[Dict], invalid_entries: List[Dict]) -> None:
    """Print a summary of the validation results."""
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    
    total_entries = len(valid_entries) + len(invalid_entries)
    if total_entries == 0:
        print("⚠️  No entries found in metadata file!")
        return
    
    kept_percentage = (len(valid_entries) / total_entries) * 100
    removed_percentage = (len(invalid_entries) / total_entries) * 100
    
    print(f"\nDataset Statistics:")
    print(f"  Total entries:   {total_entries}")
    print(f"  Kept entries:    {len(valid_entries):3d} ({kept_percentage:5.1f}%)")
    print(f"  Removed entries: {len(invalid_entries):3d} ({removed_percentage:5.1f}%)")
    
    if invalid_entries:
        print(f"\n⚠️  Dataset quality: {kept_percentage:.1f}% valid")
        if kept_percentage < 90:
            print("    ⚠️  WARNING: Less than 90% of data is valid!")
            print("    Consider investigating your dataset creation process.")
    else:
        print("\n✅ All entries are valid!")
    
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Validate and clean ControlNet training dataset"
    )
    
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Path to dataset directory containing images and metadata.jsonl"
    )
    
    parser.add_argument(
        "--metadata_file",
        type=str,
        default="metadata.jsonl",
        help="Name of metadata file (default: metadata.jsonl)"
    )
    
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't save changes, just report what would be removed"
    )
    
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Don't create backup of original metadata file"
    )

    parser.add_argument(
        "--allow_multiple_captions",
        action="store_true",
        help="Keep multiple captions per image (default behavior is one caption per image)"
    )
    
    args = parser.parse_args()
    
    dataset_dir = Path(args.dataset_dir)
    metadata_file = dataset_dir / args.metadata_file
    
    print("\n" + "=" * 60)
    print("ControlNet Dataset Validator")
    print("=" * 60 + "\n")
    
    try:
        # Validate dataset
        valid_entries, invalid_entries = validate_dataset(
            dataset_dir,
            metadata_file,
            dry_run=args.dry_run,
            keep_one_caption_per_image=not args.allow_multiple_captions
        )
        
        # Print summary
        print_summary(valid_entries, invalid_entries)
        
        # Save cleaned metadata (unless dry_run)
        if args.dry_run:
            print("🔍 DRY RUN MODE - No changes made")
            print("   Run without --dry_run to save cleaned metadata")
        else:
            if invalid_entries:
                save_cleaned_metadata(
                    valid_entries,
                    metadata_file,
                    create_backup=not args.no_backup
                )
                print("\n✅ Dataset cleaning complete!")
                print(f"   Ready to resume training with {len(valid_entries)} valid samples")
            else:
                print("✅ Dataset is already clean - no changes needed!")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())