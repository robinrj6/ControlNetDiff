from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import os
import json
from pathlib import Path

base_model_path = "/home/woody/rlvl/rlvl165v/ControlNetDiff/shared/models/sd15/"
# Point directly to the output directory which contains config.json and safetensors file
controlnet_path = "/home/woody/rlvl/rlvl165v/ControlNetDiff/output/canny_model/"

# Load with local_files_only=True to avoid trying to connect to huggingface.co
controlnet = ControlNetModel.from_pretrained(
    controlnet_path,
    torch_dtype=torch.float16,
    local_files_only=True
)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    local_files_only=True
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# remove following line if xformers is not installed or when using Torch 2.0.
pipe.enable_xformers_memory_efficient_attention()
# memory optimization.
pipe.enable_model_cpu_offload()

# Load metadata with prompts
metadata_path = "shared/datasets/coco/metricsDataset/metadata.jsonl"
metadata_dict = {}
with open(metadata_path, 'r') as f:
    for line in f:
        entry = json.loads(line)
        conditioning_file = entry['conditioning_file_name']
        prompt = entry['text']
        metadata_dict[Path(conditioning_file).name] = prompt

# Load canny edge images from folder and generate with prompts
control_images_dir = "shared/datasets/coco/metricsDataset/edges/"
output_dir = "shared/datasets/coco/metricsDataset/generated_images_ControlNet/"
Path(output_dir).mkdir(parents=True, exist_ok=True)

for img_file in sorted(os.listdir(control_images_dir)):
    if img_file.endswith((".png")):
        img_path = os.path.join(control_images_dir, img_file)
        control_image = load_image(img_path).convert("RGB")
        
        # Get the prompt from metadata
        prompt = metadata_dict.get(img_file, "a photo")
        
        # Generate image with ControlNet
        generator = torch.Generator(device="cuda").manual_seed(42)
        output = pipe(
            prompt=prompt,
            image=control_image,
            generator=generator,
            num_inference_steps=20
        )
        
        # Save the generated image
        output_path = os.path.join(output_dir, img_file)
        output.images[0].save(output_path)
        print(f"Generated: {img_file} with prompt: {prompt}")
