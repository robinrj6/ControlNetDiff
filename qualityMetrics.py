from torchmetrics.image.fid import FrechetInceptionDistance
from transformers import CLIPModel, CLIPProcessor
import cv2
import numpy as np
import torch
from PIL import Image
import os
from glob import glob

def compute_iou(mask1, mask2):
    """Compute Intersection over Union between two binary masks"""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return intersection / union

def load_images_from_folder(folder_path):
    """Load all JPG images from a folder"""
    images = []
    image_paths = sorted(glob(os.path.join(folder_path, "*.jpg")))
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        images.append(img)
    return images, image_paths

def pil_to_tensor(images):
    """Convert PIL images to tensor format for FID (N, 3, H, W) uint8"""
    tensors = []
    for img in images:
        img_array = np.array(img)  # (H, W, 3)
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # (3, H, W)
        tensors.append(img_tensor)
    return torch.stack(tensors).to(torch.uint8)

# Define test case prompts
test_cases = [
    ("no_prompt", ""),
    ("insufficient_prompt", "a high-quality image"),
    ("conflicting_prompt", "delicious cake"),
    ("perfect_prompt", "create a realistic mountains with sunlight on background")
]

print("=" * 60)

# 1. FID Score (compare generated images to COCO validation set)
print("\n1. FID Score (Frechet Inception Distance)")
print("-" * 60)
coco_val_folder = "./shared/datasets/coco/val2017"  # Adjust path to your COCO validation folder
if os.path.exists(coco_val_folder) and len(controlnet_images) > 0:
    try:
        print(f"Loading COCO validation images from {coco_val_folder}...")
        coco_images, _ = load_images_from_folder(coco_val_folder)
        
        # Limit to reasonable number for FID computation
        max_images = min(5000, len(coco_images))
        coco_images = coco_images[:max_images]
        print(f"✓ Loaded {len(coco_images)} COCO validation images")
        
        # Convert to tensors for FID
        controlnet_tensors = pil_to_tensor(controlnet_images)
        coco_tensors = pil_to_tensor(coco_images)
        
        # Compute FID
        fid = FrechetInceptionDistance(normalize=True)
        fid.update(controlnet_tensors, real=False)
        fid.update(coco_tensors, real=True)
        fid_score = fid.compute().item()
        
        print(f"\n  FID Score: {fid_score:.2f}")
        print("  Note: Lower = better. Typical range: 20-100 (lower is closer to real images)")
        
        # Also compute for SD15 if available
        if len(sd15_images) > 0:
            print("\n  Computing FID for SD15 baseline...")
            sd15_tensors = pil_to_tensor(sd15_images)
            fid_sd15 = FrechetInceptionDistance(normalize=True)
            fid_sd15.update(sd15_tensors, real=False)
            fid_sd15.update(coco_tensors, real=True)
            fid_score_sd15 = fid_sd15.compute().item()
            print(f"  FID Score (SD15): {fid_score_sd15:.2f}")
            
    except Exception as e:
        print(f"  Error computing FID: {e}")
else:
    if not os.path.exists(coco_val_folder):
        print(f"  Skipped: COCO validation folder not found at {coco_val_folder}")
        print(f"  Please set the correct path to your COCO val2017 folder")
    else:
        print(f"  Skipped: No generated images found")

# 2. CLIP Score (text-image alignment)

# Load control image and compute its Canny edges
try:
    control_img_path = "./mountain.jpg"
    control_img = cv2.imread(control_img_path)
    if control_img is None:
        print(f"Warning: Control image not found at {control_img_path}")
        input_canny = None
    else:
        gray = cv2.cvtColor(control_img, cv2.COLOR_BGR2GRAY)
        input_canny = cv2.Canny(gray, 100, 200)
        print(f"✓ Loaded control image: {control_img_path}")
except Exception as e:
    print(f"Error loading control image: {e}")
    input_canny = None

# Load ControlNet generated images
controlnet_folder = "./results/controlnet"
if os.path.exists(controlnet_folder):
    controlnet_images, controlnet_paths = load_images_from_folder(controlnet_folder)
    print(f"✓ Loaded {len(controlnet_images)} ControlNet images")
else:
    controlnet_images = []
    print(f"Warning: ControlNet folder not found: {controlnet_folder}")

# Load SD15 generated images (if available)
sd15_folder = "./results/sd15"
if os.path.exists(sd15_folder):
    sd15_images, sd15_paths = load_images_from_folder(sd15_folder)
    print(f"✓ Loaded {len(sd15_images)} SD15 images")
else:
    sd15_images = []
    print(f"Warning: SD15 folder not found: {sd15_folder}")

print("=" * 60)

# 2. CLIP Score (text-image alignment)
if len(controlnet_images) > 0:
    print("\n2. CLIP Score (text-image alignment)")
    print("-" * 60)
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        
        # Evaluate each test case
        for (name, prompt), img in zip(test_cases, controlnet_images):
            if prompt == "":
                prompt = "an image"  # CLIP needs non-empty text
            
            inputs = processor(text=[prompt], images=[img], return_tensors="pt", padding=True)
            outputs = model(**inputs)
            clip_score = outputs.logits_per_image[0, 0].item()
            print(f"  {name:25s}: {clip_score:.4f} (prompt: '{prompt[:40]}')")
        
        print("  Note: Higher = better alignment. Typical range: 0.2-0.35")
    except Exception as e:
        print(f"  Error computing CLIP scores: {e}")
else:
    print("\n2. CLIP Score: Skipped (no images found)")

# 3. Canny Edge Preservation (Condition Fidelity)
if len(controlnet_images) > 0 and input_canny is not None:
    print("\n3. Canny Edge Preservation (Condition Fidelity)")
    print("-" * 60)
    try:
        edge_fidelity = []
        for (name, prompt), img in zip(test_cases, controlnet_images):
            # Convert PIL to numpy array
            img_array = np.array(img)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            detected_edges = cv2.Canny(gray, 100, 200)
            
            # Compare with input Canny (IoU)
            iou = compute_iou(input_canny, detected_edges)
            edge_fidelity.append(iou)
            print(f"  {name:25s}: IoU = {iou:.4f}")
        
        print(f"\n  Mean Edge Fidelity (IoU): {np.mean(edge_fidelity):.4f}")
        print("  Note: Higher = better structure preservation. Range: 0.0-1.0")
    except Exception as e:
        print(f"  Error computing edge fidelity: {e}")
else:
    print("\n3. Canny Edge Preservation: Skipped (missing images or control)")

print("\n" + "=" * 60)
print("Evaluation complete!")
print("=" * 60)