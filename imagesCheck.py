# Check if the images in metadata JSONL exist in the images directory
import argparse
import json
from pathlib import Path

def main(args):
    with open(args.metadata_jsonl, "r") as f:
        lines = f.readlines()

    missing_images = []
    for line in lines:
        try:
            data = json.loads(line)
            image_path = data.get("image")
            if image_path:
                full_image_path = args.images_dir / image_path
                if not full_image_path.is_file():
                    missing_images.append(image_path)
        except json.JSONDecodeError:
            print(f"Warning: Skipping invalid JSON line: {line.strip()}")

    if missing_images:
        print("Missing images:")
        for img in missing_images:
            print(img)
    else:
        print("All images exist.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check if images in metadata JSONL exist in the images directory.")
    parser.add_argument("metadata_jsonl", type=Path, help="Path to the metadata JSONL file.")
    parser.add_argument("images_dir", type=Path, help="Path to the images directory.")
    args = parser.parse_args()
    main(args)