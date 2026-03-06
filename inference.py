from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch
import os

base_model_path = "/home/woody/rlvl/rlvl165v/ControlNetDiff/shared/models/sd15/"
# Point directly to the output directory which contains config.json and safetensors file
controlnet_path = "/home/woody/rlvl/rlvl165v/ControlNetDiff/output/"

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

control_image = load_image("./mountain.jpg")

# Create output directory if it doesn't exist
os.makedirs("./results/controlnet", exist_ok=True)

# Test cases with different prompts
test_cases = [
    {
        "name": "no_prompt",
        "prompt": "",
        "description": "No prompt (empty string)"
    },
    {
        "name": "insufficient_prompt",
        "prompt": "a high-quality image",
        "description": "Insufficient prompt (doesn't describe content)"
    },
    {
        "name": "conflicting_prompt",
        "prompt": "delicious cake",
        "description": "Conflicting prompt (cake for mountain image)"
    },
    {
        "name": "perfect_prompt",
        "prompt": "create a realistic mountains with sunlight on background",
        "description": "Perfect prompt (accurate description)"
    }
]

# Generate images for each test case
for test_case in test_cases:
    print(f"\nGenerating: {test_case['description']}")
    print(f"Prompt: '{test_case['prompt']}'")
    
    generator = torch.manual_seed(0)
    image = pipe(
        test_case['prompt'], 
        num_inference_steps=20, 
        generator=generator, 
        image=control_image
    ).images[0]
    
    output_path = f"./results/controlnet/output_{test_case['name']}.png"
    image.save(output_path)
    print(f"Saved: {output_path}")