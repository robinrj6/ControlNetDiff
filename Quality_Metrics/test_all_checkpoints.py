#!/usr/bin/env python3
"""
Test all ControlNet checkpoints and compute FID for each.
Uses local SD1.5 model for faster inference.
"""

import os
import json
from pathlib import Path
from datetime import datetime
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import cv2
import numpy as np

# Imports for FID
from pytorch_fid.fid_score import calculate_fid_given_paths

# =========================
# CONFIG
# =========================
OUTPUT_DIR = Path("output/canny_model")
SD15_PATH = Path("/home/woody/rlvl/rlvl165v/ControlNetDiff/shared/models/sd15/")  # ← LOCAL SD1.5
CANNY_DIR = Path("shared/datasets/coco/metricsDataset/edges")
REAL_IMAGES_DIR = Path("shared/datasets/coco/metricsDataset/images")
METADATA_FILE = Path("shared/datasets/coco/metricsDataset/metadata.jsonl")
RESULTS_FILE = Path("checkpoint_fid_results.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5

# =========================
# FUNCTIONS
# =========================

def verify_sd15_path():
    """Verify SD1.5 model exists at the expected location"""
    if not SD15_PATH.exists():
        raise FileNotFoundError(f"SD1.5 not found at {SD15_PATH}")
    
    required_files = ["tokenizer", "text_encoder", "vae", "unet", "scheduler"]
    for file_or_dir in required_files:
        if not (SD15_PATH / file_or_dir).exists():
            raise FileNotFoundError(f"Missing {file_or_dir} in {SD15_PATH}")
    
    print(f"✓ SD1.5 verified at {SD15_PATH}")

def get_all_checkpoints():
    """Get all checkpoint directories sorted by step"""
    checkpoints = []
    for item in OUTPUT_DIR.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            step = int(item.name.split("-")[1])
            checkpoints.append((step, item))
    
    return sorted(checkpoints, key=lambda x: x[0])

def verify_checkpoint(checkpoint_dir):
    """Verify checkpoint has required files"""
    controlnet_dir = checkpoint_dir / "controlnet"
    config_file = controlnet_dir / "config.json"
    weights_file = controlnet_dir / "diffusion_pytorch_model.safetensors"
    
    if not config_file.exists():
        return False, f"Missing config.json"
    if not weights_file.exists():
        return False, f"Missing diffusion_pytorch_model.safetensors"
    
    return True, "OK"

def load_prompts_from_metadata(metadata_file):
    """Load prompts from metadata.jsonl, indexed by filename stem"""
    stem_to_prompt = {}
    
    try:
        with open(metadata_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    text = row.get("text", "").strip()
                    
                    if text and "conditioning_file_name" in row:
                        stem = Path(row["conditioning_file_name"]).stem
                        stem_to_prompt[stem] = text
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"  ⚠ Warning: Metadata file not found: {metadata_file}")
        return {}
    
    return stem_to_prompt

def generate_images_for_checkpoint(
    checkpoint_dir, 
    canny_dir, 
    output_dir, 
    metadata_file,
    num_images=100
):
    """Generate images using a specific checkpoint WITH ACTUAL PROMPTS"""
    
    # Load prompts from metadata
    stem_to_prompt = load_prompts_from_metadata(metadata_file)
    print(f"  Loaded {len(stem_to_prompt)} prompts from metadata")
    
    # Load ControlNet checkpoint
    controlnet_path = checkpoint_dir / "controlnet"
    
    try:
        controlnet = ControlNetModel.from_pretrained(str(controlnet_path))
    except Exception as e:
        print(f"  ✗ Failed to load ControlNet: {e}")
        return False
    
    # Use fp32 for stability (works with RTX 2080 Ti)
    weight_dtype = torch.float32
    
    # Load pipeline
    try:
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            str(SD15_PATH),
            controlnet=controlnet,
            torch_dtype=weight_dtype,  # Use float32
            safety_checker=None,
        )
        
        # Explicitly move all components to device and dtype
        pipeline.to(DEVICE)
        pipeline.unet = pipeline.unet.to(weight_dtype)
        pipeline.vae = pipeline.vae.to(weight_dtype)
        pipeline.text_encoder = pipeline.text_encoder.to(weight_dtype)
        pipeline.controlnet = pipeline.controlnet.to(weight_dtype)
        
        # Disable safety checker
        pipeline.safety_checker = None
        
    except Exception as e:
        print(f"  ✗ Failed to create pipeline: {e}")
        return False
    
    # Get Canny images
    canny_images = sorted(list(canny_dir.glob("*.png")))[:num_images]
    
    if not canny_images:
        print(f"  ✗ No Canny images found in {canny_dir}")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Generating {len(canny_images)} images with actual prompts...")
    
    matched_count = 0
    unmatched_count = 0
    
    for canny_path in tqdm(canny_images, desc="Generating"):
        try:
            # Get prompt for this image
            prompt = stem_to_prompt.get(canny_path.stem)
            
            if prompt is None:
                prompt = "a high-quality image"
                unmatched_count += 1
            else:
                matched_count += 1
            
            # Load and prepare Canny image
            canny_img = Image.open(canny_path).convert("L")
            if canny_img.size != (512, 512):
                canny_img = canny_img.resize((512, 512), Image.Resampling.LANCZOS)
            
            # Convert to tensor and ensure correct dtype
            canny_tensor = torch.tensor(np.array(canny_img), dtype=weight_dtype).unsqueeze(0).unsqueeze(0)
            canny_tensor = canny_tensor / 255.0  # Normalize to 0-1
            canny_tensor = canny_tensor.to(DEVICE)
            
            # Generate with ACTUAL PROMPT
            with torch.no_grad():
                image = pipeline(
                    prompt=prompt,
                    image=canny_img,  # Let pipeline handle it
                    num_inference_steps=NUM_INFERENCE_STEPS,
                    guidance_scale=GUIDANCE_SCALE,
                    controlnet_conditioning_scale=1.0,
                ).images[0]
            
            # Save
            image.save(output_dir / f"{canny_path.stem}.png", quality=95)
        
        except Exception as e:
            print(f"  ✗ Error generating for {canny_path.name}: {e}")
            continue
    
    print(f"  ✓ Used {matched_count} actual prompts, {unmatched_count} defaults")
    
    del pipeline
    torch.cuda.empty_cache()
    
    return True

def compute_fid(real_dir, generated_dir):
    """Compute FID between real and generated images"""
    
    try:
        fid_value = float(
            calculate_fid_given_paths(
                [str(real_dir), str(generated_dir)],
                batch_size=32,
                device=DEVICE,
                dims=2048,
                num_workers=4,
            )
        )
        return fid_value
    except Exception as e:
        print(f"  ✗ FID computation failed: {e}")
        return None

def main():
    print("=" * 70)
    print("TESTING ALL CHECKPOINTS FOR BEST FID")
    print("=" * 70)
    
    # Verify SD1.5
    try:
        verify_sd15_path()
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return
    
    # Get all checkpoints
    checkpoints = get_all_checkpoints()
    
    if not checkpoints:
        print("✗ No checkpoints found!")
        return
    
    print(f"\nFound {len(checkpoints)} checkpoints")
    for step, cp_dir in checkpoints:
        valid, msg = verify_checkpoint(cp_dir)
        status = "✓" if valid else "✗"
        print(f"  {status} {cp_dir.name}: {msg}")
    
    # Results storage
    results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
    
    # Test each checkpoint
    print("\n" + "=" * 70)
    print("COMPUTING FID FOR EACH CHECKPOINT")
    print("=" * 70)
    
    best_fid = float('inf')
    best_checkpoint = None
    
    for step, checkpoint_dir in checkpoints:
        checkpoint_name = checkpoint_dir.name
        
        # Skip if already computed
        if checkpoint_name in results and results[checkpoint_name].get('fid') is not None:
            fid_value = results[checkpoint_name]['fid']
            print(f"\n{checkpoint_name}: FID = {fid_value:.4f} (cached)")
            if fid_value < best_fid:
                best_fid = fid_value
                best_checkpoint = checkpoint_name
            continue
        
        print(f"\n{checkpoint_name}")
        print("-" * 70)
        
        # Verify checkpoint
        valid, msg = verify_checkpoint(checkpoint_dir)
        if not valid:
            print(f"  ✗ Checkpoint invalid: {msg}")
            results[checkpoint_name] = {'fid': None, 'error': msg}
            continue
        
        # Generate images
        gen_dir = Path(f"temp_generated/{checkpoint_name}")
        print(f"  Generating images...")
        success = generate_images_for_checkpoint(
            checkpoint_dir, 
            CANNY_DIR, 
            gen_dir,
            METADATA_FILE,
            num_images=100  # Use 100 for quick screening
        )
        
        if not success:
            print(f"  ✗ Generation failed")
            results[checkpoint_name] = {'fid': None, 'error': 'Generation failed'}
            continue
        
        # Compute FID
        print(f"  Computing FID...")
        fid_value = compute_fid(REAL_IMAGES_DIR, gen_dir)
        
        if fid_value is None:
            results[checkpoint_name] = {'fid': None, 'error': 'FID computation failed'}
            continue
        
        print(f"  ✓ FID = {fid_value:.4f}")
        
        results[checkpoint_name] = {
            'step': step,
            'fid': fid_value,
            'timestamp': datetime.now().isoformat()
        }
        
        # Track best
        if fid_value < best_fid:
            best_fid = fid_value
            best_checkpoint = checkpoint_name
        
        # Save results after each checkpoint
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    # Sort by FID
    sorted_results = sorted(
        [(name, data['fid']) for name, data in results.items() if data['fid'] is not None],
        key=lambda x: x[1]
    )
    
    print("\nCheckpoints ranked by FID (lower is better):")
    for rank, (name, fid) in enumerate(sorted_results, 1):
        marker = "🏆 BEST" if name == best_checkpoint else ""
        step = int(name.split("-")[1])
        print(f"  {rank}. {name} (step {step}): FID = {fid:.4f} {marker}")
    
    if best_checkpoint:
        best_step = int(best_checkpoint.split("-")[1])
        print(f"\n✓ Best checkpoint: {best_checkpoint}")
        print(f"  Best FID: {best_fid:.4f}")
        print(f"  Path: {OUTPUT_DIR / best_checkpoint / 'controlnet'}")
        print(f"\nFor final metrics with FULL dataset, use:")
        print(f"  CONTROLNET_CHECKPOINT = Path('{OUTPUT_DIR / best_checkpoint / 'controlnet'}')")
    
    print("\n✓ Results saved to: " + str(RESULTS_FILE))

if __name__ == "__main__":
    main()