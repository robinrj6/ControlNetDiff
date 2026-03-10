#!/usr/bin/env python3
"""
Test all ControlNet checkpoints and compute FID for each.
Uses local SD1.5 model and cached inception weights for faster inference.
"""

import os
import json
from pathlib import Path
from datetime import datetime
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import numpy as np

# Imports for FID
from pytorch_fid.fid_score import calculate_fid_given_paths

# =========================
# TORCH CACHE SETUP
# =========================
# Set TORCH_HOME to use cached inception weights (no internet needed)
TORCH_HOME = Path("/home/woody/rlvl/rlvl165v/ControlNetDiff/shared/models/torch_cache")
os.environ["TORCH_HOME"] = str(TORCH_HOME)

print(f"✓ Using cached models from: {TORCH_HOME}")

# =========================
# CONFIG
# =========================
OUTPUT_DIR = Path("output/canny_model")
SD15_PATH = Path("/home/woody/rlvl/rlvl165v/ControlNetDiff/shared/models/sd15/")
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

def verify_paths():
    """Verify all required paths exist"""
    paths_to_check = {
        "SD1.5": SD15_PATH,
        "Canny edges": CANNY_DIR,
        "Real images": REAL_IMAGES_DIR,
        "Metadata": METADATA_FILE,
        "Torch cache": TORCH_HOME,
    }
    
    for name, path in paths_to_check.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} not found at {path}")
        print(f"  ✓ {name}: {path}")

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
    num_images=100,
    generate_controlnet=True,
):
    """Generate images using a specific checkpoint WITH ACTUAL PROMPTS"""
    
    # Load prompts from metadata
    stem_to_prompt = load_prompts_from_metadata(metadata_file)
    print(f"  Loaded {len(stem_to_prompt)} prompts from metadata")
    
    # Load ControlNet checkpoint (only if generating ControlNet)
    controlnet = None
    if generate_controlnet:
        controlnet_path = checkpoint_dir / "controlnet"
        try:
            controlnet = ControlNetModel.from_pretrained(str(controlnet_path))
        except Exception as e:
            print(f"  ✗ Failed to load ControlNet: {e}")
            return False
    
    # Load pipeline
    try:
        if generate_controlnet:
            pipeline = StableDiffusionControlNetPipeline.from_pretrained(
                str(SD15_PATH),
                controlnet=controlnet,
                torch_dtype=torch.float32,
                safety_checker=None,
            ).to(DEVICE)
            print(f"  ✓ ControlNet pipeline loaded")
        else:
            # For SD1.5 baseline (no ControlNet)
            from diffusers import StableDiffusionPipeline
            pipeline = StableDiffusionPipeline.from_pretrained(
                str(SD15_PATH),
                torch_dtype=torch.float32,
                safety_checker=None,
            ).to(DEVICE)
            print(f"  ✓ SD1.5 baseline pipeline loaded")
        
    except Exception as e:
        print(f"  ✗ Failed to create pipeline: {e}")
        return False
    
    # Get Canny images
    canny_images = sorted(list(canny_dir.glob("*.png")))[:num_images]
    
    if not canny_images:
        print(f"  ✗ No Canny images found in {canny_dir}")
        return False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mode = "ControlNet" if generate_controlnet else "SD1.5 (baseline)"
    print(f"  Generating {len(canny_images)} images ({mode}) with actual prompts...")
    
    matched_count = 0
    unmatched_count = 0
    
    for canny_path in tqdm(canny_images, desc=f"Generating ({mode})"):
        try:
            # Get prompt for this image
            prompt = stem_to_prompt.get(canny_path.stem)
            
            if prompt is None:
                prompt = "a high-quality image"
                unmatched_count += 1
            else:
                matched_count += 1
            
            # Load Canny image
            canny_img = Image.open(canny_path).convert("L")
            canny_img_rgb = Image.new("RGB", canny_img.size)
            canny_img_rgb.paste(canny_img)
            if canny_img.size != (512, 512):
                canny_img_rgb = canny_img_rgb.resize((512, 512), Image.Resampling.LANCZOS)
            
            # Generate
            with torch.no_grad():
                if generate_controlnet:
                    # ControlNet with conditioning
                    image = pipeline(
                        prompt=prompt,
                        image=canny_img_rgb,
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                        controlnet_conditioning_scale=1.0,
                    ).images[0]
                else:
                    # SD1.5 baseline (NO conditioning)
                    image = pipeline(
                        prompt=prompt,
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        guidance_scale=GUIDANCE_SCALE,
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

def compute_fid_both(real_dir, controlnet_gen_dir, sd15_gen_dir):
    """Compute FID for both ControlNet and SD1.5 baseline"""
    
    results = {}
    
    # FID for ControlNet
    try:
        fid_controlnet = float(
            calculate_fid_given_paths(
                [str(real_dir), str(controlnet_gen_dir)],
                batch_size=32,
                device=DEVICE,
                dims=2048,
                num_workers=4,
            )
        )
        results['controlnet'] = fid_controlnet
        print(f"  ✓ ControlNet FID: {fid_controlnet:.4f}")
    except Exception as e:
        print(f"  ✗ ControlNet FID failed: {e}")
        results['controlnet'] = None
    
    # FID for SD1.5 baseline
    try:
        fid_sd15 = float(
            calculate_fid_given_paths(
                [str(real_dir), str(sd15_gen_dir)],
                batch_size=32,
                device=DEVICE,
                dims=2048,
                num_workers=4,
            )
        )
        results['sd15'] = fid_sd15
        print(f"  ✓ SD1.5 FID: {fid_sd15:.4f}")
    except Exception as e:
        print(f"  ✗ SD1.5 FID failed: {e}")
        results['sd15'] = None
    
    return results

def main():
    print("=" * 70)
    print("TESTING ALL CHECKPOINTS FOR BEST FID")
    print("=" * 70)
    
    # Verify all paths
    print("\nVerifying paths...")
    try:
        verify_paths()
    except FileNotFoundError as e:
        print(f"✗ {e}")
        return
    
    # Get all checkpoints
    print(f"\nScanning checkpoints...")
    checkpoints = get_all_checkpoints()
    
    if not checkpoints:
        print("✗ No checkpoints found!")
        return
    
    print(f"Found {len(checkpoints)} checkpoints")
    for step, cp_dir in checkpoints[:5]:  # Show first 5
        valid, msg = verify_checkpoint(cp_dir)
        status = "✓" if valid else "✗"
        print(f"  {status} {cp_dir.name}: {msg}")
    if len(checkpoints) > 5:
        print(f"  ... and {len(checkpoints) - 5} more")
    
    # Results storage
    results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
    
    # Test each checkpoint
    print("\n" + "=" * 70)
    print("COMPUTING FID FOR EACH CHECKPOINT")
    print("=" * 70)
    
    for step, checkpoint_dir in checkpoints:
        checkpoint_name = checkpoint_dir.name
        
        # Skip if already computed
        if checkpoint_name in results and results[checkpoint_name].get('fid_controlnet') is not None:
            fid_cn = results[checkpoint_name]['fid_controlnet']
            fid_sd = results[checkpoint_name].get('fid_sd15', None)
            print(f"\n{checkpoint_name}: ControlNet FID = {fid_cn:.4f}, SD1.5 FID = {fid_sd:.4f if fid_sd else 'N/A'} (cached)")
            continue
    
        print(f"\n{checkpoint_name}")
        print("-" * 70)
        
        # Verify checkpoint
        valid, msg = verify_checkpoint(checkpoint_dir)
        if not valid:
            print(f"  ✗ Checkpoint invalid: {msg}")
            results[checkpoint_name] = {'fid_controlnet': None, 'error': msg}
            continue
        
        # Generate ControlNet images
        gen_dir_cn = Path(f"temp_generated/{checkpoint_name}/controlnet")
        print(f"  [1/2] Generating ControlNet images...")
        success_cn = generate_images_for_checkpoint(
            checkpoint_dir, 
            CANNY_DIR, 
            gen_dir_cn,
            METADATA_FILE,
            num_images=100,
            generate_controlnet=True,
        )
        
        # Generate SD1.5 baseline images (once, then reuse)
        gen_dir_sd15 = Path(f"temp_generated/sd15_baseline")
        if not gen_dir_sd15.exists() or not list(gen_dir_sd15.glob("*.png")):
            print(f"  [2/2] Generating SD1.5 baseline images...")
            success_sd15 = generate_images_for_checkpoint(
                checkpoint_dir,
                CANNY_DIR, 
                gen_dir_sd15,
                METADATA_FILE,
                num_images=100,
                generate_controlnet=False,
            )
        else:
            print(f"  [2/2] Reusing cached SD1.5 baseline images")
            success_sd15 = True
        
        if not success_cn:
            print(f"  ✗ ControlNet generation failed")
            results[checkpoint_name] = {'fid_controlnet': None, 'error': 'Generation failed'}
            continue
        
        # Compute FID
        print(f"  Computing FID (using cached inception weights)...")
        fid_results = compute_fid_both(REAL_IMAGES_DIR, gen_dir_cn, gen_dir_sd15)
        
        if fid_results['controlnet'] is None:
            results[checkpoint_name] = {'fid_controlnet': None, 'error': 'FID computation failed'}
            continue
        
        results[checkpoint_name] = {
            'step': step,
            'fid_controlnet': fid_results['controlnet'],
            'fid_sd15': fid_results['sd15'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results after each checkpoint
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    sorted_results = sorted(
        [(name, data['fid_controlnet']) for name, data in results.items() 
         if data['fid_controlnet'] is not None],
        key=lambda x: x[1]
    )

    print(f"\nCheckpoints ranked by ControlNet FID (lower is better):")
    print(f"{'Rank':<5} {'Checkpoint':<20} {'ControlNet FID':<15} {'SD1.5 FID':<15} {'Difference':<15}")
    print("-" * 70)

    for rank, (name, fid_cn) in enumerate(sorted_results, 1):
        fid_sd = results[name].get('fid_sd15', None)
        diff = f"{fid_cn - fid_sd:.4f}" if fid_sd else "N/A"
        marker = "🏆 BEST" if rank == 1 else ""
        step = int(name.split("-")[1])

        print(f"{rank:<5} {name:<20} {fid_cn:<15.4f} {fid_sd:<15.4f} {diff:<15} {marker}")

    if sorted_results:
        best_checkpoint = sorted_results[0][0]
        best_fid = sorted_results[0][1]
        
        print("\n" + "=" * 70)
        print("BEST CHECKPOINT IDENTIFIED")
        print("=" * 70)
        print(f"\n✓ Best checkpoint: {best_checkpoint}")
        print(f"  ControlNet FID: {best_fid:.4f}")
        print(f"  Path: {OUTPUT_DIR / best_checkpoint / 'controlnet'}")
        
        print("\n" + "=" * 70)
        print("PAPER COMPARISON (Table 4)")
        print("=" * 70)
        print(f"Paper SD1.5 baseline FID:  6.09")
        print(f"Paper ControlNet FID:      15.27")
        print(f"Your best ControlNet FID:  {best_fid:.4f}")
        print(f"\nRatio (your / paper):      {best_fid / 15.27:.2f}x")
        
        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print(f"\n1. To regenerate with FULL 2000 images for final metrics:")
        print(f"   python << 'EOF'")
        print(f"   from pathlib import Path")
        print(f"   from test_all_checkpoints import generate_images_for_checkpoint")
        print(f"")
        print(f"   # ControlNet images")
        print(f"   generate_images_for_checkpoint(")
        print(f"       checkpoint_dir=Path('{OUTPUT_DIR / best_checkpoint}'),")
        print(f"       canny_dir=Path('{CANNY_DIR}'),")
        print(f"       output_dir=Path('shared/datasets/coco/metricsDataset/generated_images_ControlNet'),")
        print(f"       metadata_file=Path('{METADATA_FILE}'),")
        print(f"       num_images=2000,")
        print(f"       generate_controlnet=True,")
        print(f"   )")
        print(f"")
        print(f"   # SD1.5 baseline")
        print(f"   generate_images_for_checkpoint(")
        print(f"       checkpoint_dir=Path('{OUTPUT_DIR / best_checkpoint}'),")
        print(f"       canny_dir=Path('{CANNY_DIR}'),")
        print(f"       output_dir=Path('shared/datasets/coco/metricsDataset/generated_images_SD15'),")
        print(f"       metadata_file=Path('{METADATA_FILE}'),")
        print(f"       num_images=2000,")
        print(f"       generate_controlnet=False,")
        print(f"   )")
        print(f"   EOF")
        print(f"\n2. Then run final metrics:")
        print(f"   python Quality_Metrics/qualityMetrics.py")

    print("\n✓ Results saved to: " + str(RESULTS_FILE))

if __name__ == "__main__":
    main()