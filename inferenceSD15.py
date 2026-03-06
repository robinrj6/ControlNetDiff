from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch
import os

base_model_path = "/home/woody/rlvl/rlvl165v/ControlNetDiff/shared/models/sd15/"

# Load only the base Stable Diffusion model
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    local_files_only=True
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

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

# Create output directory if it doesn't exist
os.makedirs("./results/sd15", exist_ok=True)

# Generate images for each test case
for test_case in test_cases:
    print(f"\nGenerating: {test_case['description']}")
    print(f"Prompt: '{test_case['prompt']}'")
    
    generator = torch.manual_seed(0)
    image = pipe(
        test_case['prompt'], 
        num_inference_steps=20, 
        generator=generator
    ).images[0]
    
    output_path = f"./results/sd15/output_sd15_{test_case['name']}.png"
    image.save(output_path)
    print(f"Saved: {output_path}")