#!/bin/bash
# Download CLIP model to shared cache directory
# Run this on a machine with internet access before submitting jobs to the cluster

set -e  # Exit on error

CLIP_CACHE_DIR="shared/models/huggingface_cache"
mkdir -p "$CLIP_CACHE_DIR"

echo "Downloading CLIP model to: $CLIP_CACHE_DIR"
echo "This may take a few minutes on first run..."
echo ""

export HF_HOME="$CLIP_CACHE_DIR"

python3 << 'EOF'
import os
from transformers import CLIPModel, CLIPProcessor
from pathlib import Path

cache_dir = Path(os.environ.get("HF_HOME", "."))
print(f"Using cache directory: {cache_dir}")
print("")

print("Downloading CLIPProcessor (openai/clip-vit-base-patch32)...")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("✓ CLIPProcessor downloaded successfully")
print("")

print("Downloading CLIPModel (openai/clip-vit-base-patch32)...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
print("✓ CLIPModel downloaded successfully")
print("")

# List what was downloaded
import glob
print("Cache contents:")
for f in sorted(glob.glob(str(cache_dir / "**/*"), recursive=True)):
    rel_path = os.path.relpath(f, cache_dir)
    if os.path.isfile(f):
        size_mb = os.path.getsize(f) / (1024**2)
        print(f"  {rel_path:<60} {size_mb:>8.2f} MB")

print("")
print(f"✓ CLIP model cached successfully at: {cache_dir}")
EOF

echo ""
echo "Done! You can now run quality metrics on the cluster."
echo "Make sure HF_HOME is set to the cache directory in your SLURM script."
