#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

"""
Run : python create_metadata_jsonl.py --input-json captions_val2017.json --output-jsonl metadata.jsonl 
Optional:
--images-dir /path/to/images to check existence
--skip-missing to skip missing files
--image-prefix images (default)
--conditioning-prefix edges (default)
--one-caption-per-image
--skip-missing
"""


def load_coco_pairs(coco_json_path: Path, one_caption_per_image: bool = False) -> List[Tuple[str, str]]:
    """
    Return (file_name, caption) pairs from a COCO captions JSON.
    By default, produces one output row per annotation.
    If one_caption_per_image=True, keeps only the first caption per image.
    
    """
    with coco_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
        raise ValueError(
            "Input JSON must be a COCO-style captions file with 'images' and 'annotations' keys."
        )

    id_to_name: Dict[int, str] = {}
    for img in data["images"]:
        if not isinstance(img, dict):
            continue
        image_id = img.get("id")
        file_name = img.get("file_name")
        if image_id is not None and file_name:
            id_to_name[image_id] = file_name

    pairs: List[Tuple[str, str]] = []
    seen_image_ids = set()
    for ann in data["annotations"]:
        if not isinstance(ann, dict):
            continue
        image_id = ann.get("image_id")
        caption = ann.get("caption")
        file_name = id_to_name.get(image_id)

        if one_caption_per_image and image_id in seen_image_ids:
            continue

        if file_name and isinstance(caption, str) and caption.strip():
            pairs.append((file_name, caption.strip()))
            if one_caption_per_image:
                seen_image_ids.add(image_id)

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create ControlNet metadata JSONL from a COCO captions JSON."
    )
    parser.add_argument(
        "--input-json",
        required=True,
        type=Path,
        help="Path to COCO captions JSON (e.g. captions_val2017.json)",
    )
    parser.add_argument(
        "--output-jsonl",
        required=True,
        type=Path,
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Optional local images directory to validate that image files exist",
    )
    parser.add_argument(
        "--image-prefix",
        default="images",
        help="Prefix to write in 'image' field (default: images)",
    )
    parser.add_argument(
        "--conditioning-prefix",
        default="edges",
        help="Prefix to write in 'conditioning_image' field (default: edges)",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="If set, skip records whose image file is missing in --images-dir",
    )
    parser.add_argument(
        "--one-caption-per-image",
        action="store_true",
        help="If set, keep only one caption per image (COCO has multiple captions/image by default)",
    )

    args = parser.parse_args()

    pairs = load_coco_pairs(
        args.input_json,
        one_caption_per_image=args.one_caption_per_image,
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_missing = 0

    with args.output_jsonl.open("w", encoding="utf-8") as out_f:
        for file_name, caption in pairs:
            if args.images_dir is not None:
                if not (args.images_dir / file_name).is_file():
                    skipped_missing += 1
                    if args.skip_missing:
                        continue

            row = {
                "text": caption,
                # Required by newer `datasets` imagefolder loader
                "file_name": f"{args.image_prefix}/{file_name}",
                # Additional image column, exposed as `conditioning` in the dataset
                "conditioning_file_name": f"{args.conditioning_prefix}/{file_name}",
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(f"Done. Wrote {written} rows to {args.output_jsonl}")
    if args.images_dir is not None:
        print(f"Missing images found in {args.images_dir}: {skipped_missing}")


if __name__ == "__main__":
    main()
