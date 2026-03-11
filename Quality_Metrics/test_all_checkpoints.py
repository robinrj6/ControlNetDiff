#!/usr/bin/env python3
"""
Test all ControlNet checkpoints and compute FID for each.
Uses local SD1.5 model and cached inception weights for faster inference.
FIXED VERSION: Better error handling, memory cleanup, detailed logging
"""

import os
import json
import gc
import sys
from pathlib import Path
from datetime import datetime
import torch
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import numpy as np
import traceback

# Imports for FID
from pytorch_fid.fid_score import calculate_fid_given_paths

# =========================
# TORCH CACHE SETUP
# =========================
TORCH_HOME = Path("/home/woody/rlvl/rlvl165v/ControlNetDiff/shared/models/torch_cache")
os.environ["TORCH_HOME"] = str(TORCH_HOME)

print(f"✓ Using cached models from: {TORCH_HOME}")

# =========================
# CONFIG
# =========================
OUTPUT_DIR = Path("output/depth_model")
SD15_PATH = Path("/home/woody/rlvl/rlvl165v/ControlNetDiff/shared/models/sd15/")
depth_DIR = Path("shared/datasets/coco/metricsDataset/edges")
REAL_IMAGES_DIR = Path("shared/datasets/coco/metricsDataset/images")
METADATA_FILE = Path("shared/datasets/coco/metricsDataset/metadata.jsonl")
RESULTS_FILE = Path("checkpoint_fid_results.json")
LOG_FILE = Path("checkpoint_testing.log")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
NUM_INFERENCE_STEPS = 50
GUIDANCE_SCALE = 7.5

# =========================
# LOGGING
# =========================
def log_message(msg, level="INFO"):
    """Log to both console and file"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_msg = f"[{timestamp}] [{level}] {msg}"
    print(full_msg)
    
    # Also log to file
    with open(LOG_FILE, 'a') as f:
        f.write(full_msg + "\n")
    
    sys.stdout.flush()

# =========================
# MEMORY MANAGEMENT
# =========================
def cleanup_memory():
    """Clean up GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def get_gpu_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    return 0

# =========================
# FUNCTIONS
# =========================

def verify_paths():
    """Verify all required paths exist"""
    paths_to_check = {
        "SD1.5": SD15_PATH,
        "depth edges": depth_DIR,
        "Real images": REAL_IMAGES_DIR,
        "Metadata": METADATA_FILE,
        "Torch cache": TORCH_HOME,
    }
    
    for name, path in paths_to_check.items():
        if not path.exists():
            raise FileNotFoundError(f"{name} not found at {path}")
        log_message(f"✓ {name}: {path}")

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
        log_message(f"Warning: Metadata file not found: {metadata_file}", "WARN")
        return {}
    
    return stem_to_prompt

def generate_images_for_checkpoint(
    checkpoint_dir, 
    depth_dir, 
    output_dir, 
    metadata_file,
    num_images=100,
    generate_controlnet=True,
):
    """Generate images using a specific checkpoint WITH ACTUAL PROMPTS"""
    
    try:
        # Load prompts from metadata
        stem_to_prompt = load_prompts_from_metadata(metadata_file)
        log_message(f"  Loaded {len(stem_to_prompt)} prompts from metadata")
        
        # Load ControlNet checkpoint (only if generating ControlNet)
        controlnet = None
        if generate_controlnet:
            controlnet_path = checkpoint_dir / "controlnet"
            try:
                controlnet = ControlNetModel.from_pretrained(str(controlnet_path))
            except Exception as e:
                log_message(f"  Failed to load ControlNet: {e}", "ERROR")
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
                log_message(f"  ✓ ControlNet pipeline loaded (GPU memory: {get_gpu_memory_usage():.2f}GB)")
            else:
                from diffusers import StableDiffusionPipeline
                pipeline = StableDiffusionPipeline.from_pretrained(
                    str(SD15_PATH),
                    torch_dtype=torch.float32,
                    safety_checker=None,
                ).to(DEVICE)
                log_message(f"  ✓ SD1.5 baseline pipeline loaded (GPU memory: {get_gpu_memory_usage():.2f}GB)")
            
        except Exception as e:
            log_message(f"  Failed to create pipeline: {e}", "ERROR")
            log_message(f"  Traceback: {traceback.format_exc()}", "ERROR")
            return False
        
        # Get depth images
        depth_images = sorted(list(depth_dir.glob("*.png")))[:num_images]
        
        if not depth_images:
            log_message(f"  No depth images found in {depth_dir}", "ERROR")
            return False
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mode = "ControlNet" if generate_controlnet else "SD1.5 (baseline)"
        log_message(f"  Generating {len(depth_images)} images ({mode}) with actual prompts...")
        
        matched_count = 0
        unmatched_count = 0
        
        for depth_path in tqdm(depth_images, desc=f"Generating ({mode})", leave=False):
            try:
                # Get prompt for this image
                prompt = stem_to_prompt.get(depth_path.stem)
                
                if prompt is None:
                    prompt = "a high-quality image"
                    unmatched_count += 1
                else:
                    matched_count += 1
                
                # Load depth image
                depth_img = Image.open(depth_path).convert("L")
                depth_img_rgb = Image.new("RGB", depth_img.size)
                depth_img_rgb.paste(depth_img)
                if depth_img.size != (512, 512):
                    depth_img_rgb = depth_img_rgb.resize((512, 512), Image.Resampling.LANCZOS)
                
                # Generate
                with torch.no_grad():
                    if generate_controlnet:
                        image = pipeline(
                            prompt=prompt,
                            image=depth_img_rgb,
                            num_inference_steps=NUM_INFERENCE_STEPS,
                            guidance_scale=GUIDANCE_SCALE,
                            controlnet_conditioning_scale=1.0,
                        ).images[0]
                    else:
                        image = pipeline(
                            prompt=prompt,
                            num_inference_steps=NUM_INFERENCE_STEPS,
                            guidance_scale=GUIDANCE_SCALE,
                        ).images[0]
                
                # Save
                image.save(output_dir / f"{depth_path.stem}.png", quality=95)
            
            except Exception as e:
                log_message(f"    Error generating for {depth_path.name}: {e}", "WARN")
                continue
        
        log_message(f"  ✓ Used {matched_count} actual prompts, {unmatched_count} defaults")
        
        # Cleanup
        del pipeline
        cleanup_memory()
        log_message(f"  GPU memory after cleanup: {get_gpu_memory_usage():.2f}GB")
        
        return True
        
    except Exception as e:
        log_message(f"Unexpected error in generate_images_for_checkpoint: {e}", "ERROR")
        log_message(f"Traceback: {traceback.format_exc()}", "ERROR")
        cleanup_memory()
        return False

def compute_fid_both(real_dir, controlnet_gen_dir, sd15_gen_dir):
    """Compute FID for both ControlNet and SD1.5 baseline"""
    
    results = {}
    
    try:
        # FID for ControlNet
        log_message(f"  Computing ControlNet FID...")
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
            log_message(f"  ✓ ControlNet FID: {fid_controlnet:.4f}")
        except Exception as e:
            log_message(f"  ControlNet FID failed: {e}", "ERROR")
            log_message(f"  Traceback: {traceback.format_exc()}", "ERROR")
            results['controlnet'] = None
        
        cleanup_memory()
        
        # FID for SD1.5 baseline
        log_message(f"  Computing SD1.5 FID...")
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
            log_message(f"  ✓ SD1.5 FID: {fid_sd15:.4f}")
        except Exception as e:
            log_message(f"  SD1.5 FID failed: {e}", "ERROR")
            log_message(f"  Traceback: {traceback.format_exc()}", "ERROR")
            results['sd15'] = None
        
        cleanup_memory()
        
    except Exception as e:
        log_message(f"Unexpected error in compute_fid_both: {e}", "ERROR")
        log_message(f"Traceback: {traceback.format_exc()}", "ERROR")
    
    return results

def main():
    log_message("=" * 70)
    log_message("TESTING ALL CHECKPOINTS FOR BEST FID")
    log_message("=" * 70)
    
    # Verify all paths
    log_message("\nVerifying paths...")
    try:
        verify_paths()
    except FileNotFoundError as e:
        log_message(f"Path verification failed: {e}", "ERROR")
        return
    
    # Get all checkpoints
    log_message(f"\nScanning checkpoints...")
    checkpoints = get_all_checkpoints()
    
    if not checkpoints:
        log_message("No checkpoints found!", "ERROR")
        return
    
    log_message(f"Found {len(checkpoints)} checkpoints")
    for step, cp_dir in checkpoints[:5]:
        valid, msg = verify_checkpoint(cp_dir)
        status = "✓" if valid else "✗"
        log_message(f"  {status} {cp_dir.name}: {msg}")
    if len(checkpoints) > 5:
        log_message(f"  ... and {len(checkpoints) - 5} more")
    
    # Results storage
    results = {}
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'r') as f:
            results = json.load(f)
    
    # Test each checkpoint
    log_message("\n" + "=" * 70)
    log_message("COMPUTING FID FOR EACH CHECKPOINT")
    log_message("=" * 70)
    
    for step, checkpoint_dir in checkpoints:
        checkpoint_name = checkpoint_dir.name
        
        # Skip if already computed
        if checkpoint_name in results and results[checkpoint_name].get('fid_controlnet') is not None:
            fid_cn = results[checkpoint_name]['fid_controlnet']
            fid_sd = results[checkpoint_name].get('fid_sd15', None)
            fid_sd_str = f"{fid_sd:.4f}" if fid_sd is not None else "N/A"
            log_message(f"\n{checkpoint_name}: ControlNet FID = {fid_cn:.4f}, SD1.5 FID = {fid_sd_str} (cached)")
            continue
    
        log_message(f"\n{checkpoint_name}")
        log_message("-" * 70)
        
        try:
            # Verify checkpoint
            valid, msg = verify_checkpoint(checkpoint_dir)
            if not valid:
                log_message(f"  Checkpoint invalid: {msg}", "ERROR")
                results[checkpoint_name] = {'fid_controlnet': None, 'error': msg}
                continue
            
            # Generate ControlNet images
            gen_dir_cn = Path(f"temp_generated/{checkpoint_name}/controlnet")
            log_message(f"  [1/2] Generating ControlNet images...")
            success_cn = generate_images_for_checkpoint(
                checkpoint_dir, 
                depth_DIR, 
                gen_dir_cn,
                METADATA_FILE,
                num_images=100,
                generate_controlnet=True,
            )
            
            if not success_cn:
                log_message(f"  ControlNet generation failed", "ERROR")
                results[checkpoint_name] = {'fid_controlnet': None, 'error': 'Generation failed'}
                continue
            
            # Generate SD1.5 baseline images (once, then reuse)
            gen_dir_sd15 = Path(f"temp_generated/sd15_baseline")
            if not gen_dir_sd15.exists() or not list(gen_dir_sd15.glob("*.png")):
                log_message(f"  [2/2] Generating SD1.5 baseline images...")
                success_sd15 = generate_images_for_checkpoint(
                    checkpoint_dir,
                    depth_DIR, 
                    gen_dir_sd15,
                    METADATA_FILE,
                    num_images=100,
                    generate_controlnet=False,
                )
            else:
                log_message(f"  [2/2] Reusing cached SD1.5 baseline images")
                success_sd15 = True
            
            # Compute FID
            log_message(f"  Computing FID (using cached inception weights)...")
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
            
            log_message(f"  ✓ Checkpoint {checkpoint_name} complete - GPU memory: {get_gpu_memory_usage():.2f}GB")
            
        except Exception as e:
            log_message(f"Unexpected error processing {checkpoint_name}: {e}", "ERROR")
            log_message(f"Traceback: {traceback.format_exc()}", "ERROR")
            results[checkpoint_name] = {'fid_controlnet': None, 'error': f'Unexpected error: {str(e)}'}
            cleanup_memory()
            continue
    
    # Summary
    log_message("\n" + "=" * 70)
    log_message("SUMMARY")
    log_message("=" * 70)

    sorted_results = sorted(
        [(name, data['fid_controlnet']) for name, data in results.items() 
         if data['fid_controlnet'] is not None],
        key=lambda x: x[1]
    )

    log_message(f"\nCheckpoints ranked by ControlNet FID (lower is better):")
    log_message(f"{'Rank':<5} {'Checkpoint':<20} {'ControlNet FID':<15} {'SD1.5 FID':<15}")
    log_message("-" * 70)

    for rank, (name, fid_cn) in enumerate(sorted_results, 1):
        fid_sd = results[name].get('fid_sd15', None)
        marker = "🏆 BEST" if rank == 1 else ""
        
        if fid_sd is not None:
            log_message(f"{rank:<5} {name:<20} {fid_cn:<15.4f} {fid_sd:<15.4f} {marker}")
        else:
            log_message(f"{rank:<5} {name:<20} {fid_cn:<15.4f} {'N/A':<15} {marker}")

    if sorted_results:
        best_checkpoint = sorted_results[0][0]
        best_fid = sorted_results[0][1]
        
        log_message("\n" + "=" * 70)
        log_message("BEST CHECKPOINT IDENTIFIED")
        log_message("=" * 70)
        log_message(f"\n✓ Best checkpoint: {best_checkpoint}")
        log_message(f"  ControlNet FID: {best_fid:.4f}")

    log_message(f"\n✓ Results saved to: {RESULTS_FILE}")
    log_message(f"✓ Log saved to: {LOG_FILE}")

if __name__ == "__main__":
    main()