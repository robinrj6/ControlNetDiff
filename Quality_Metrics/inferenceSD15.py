from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from diffusers.utils import load_image
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

# Load 5 canny edge images from folder
control_images_dir = "./results/canny/"
control_images = []
for i in range(1, 6):
    img_path = os.path.join(control_images_dir, f"{i}.jpg")
    if os.path.exists(img_path):
        img = load_image(img_path)
        # Resize to dimensions divisible by 8 (required by diffusion pipeline)
        width, height = img.size
        new_width = (width // 8) * 8
        new_height = (height // 8) * 8
        img = img.resize((new_width, new_height))
        control_images.append(img)
    else:
        print(f"Warning: {img_path} not found")

if len(control_images) != 5:
    raise ValueError(f"Expected 5 control images, but found {len(control_images)}")

# Define prompt sets for each image (4 prompt styles per image)
prompt_sets = [
    [  # Image 1
        ("no_prompt", ""),
        ("insufficient_prompt", "a high-quality image"),
        ("conflicting_prompt", "delicious cake"),
        ("perfect_prompt", "A big burly grizzly bear is show with grass in the background.")
    ],
    [  # Image 2
        ("no_prompt", ""),
        ("insufficient_prompt", "a high-quality image"),
        ("conflicting_prompt", "delicious cake"),
        ("perfect_prompt", "a woman standing on skiis while posing for the camera")
    ],
    [  # Image 3
        ("no_prompt", ""),
        ("insufficient_prompt", "a high-quality image"),
        ("conflicting_prompt", "delicious cake"),
        ("perfect_prompt", "A bus that is sitting on the street.")
    ],
    [  # Image 4
        ("no_prompt", ""),
        ("insufficient_prompt", "a high-quality image"),
        ("conflicting_prompt", "delicious cake"),
        ("perfect_prompt", "A herd of elephants walking away from a watering hole.")
    ],
    [  # Image 5
        ("no_prompt", ""),
        ("insufficient_prompt", "a high-quality image"),
        ("conflicting_prompt", "delicious cake"),
        ("perfect_prompt", "A little girl with a big, blue umbrella.")
    ]
]

# Create output directory if it doesn't exist
os.makedirs("./results/sd15", exist_ok=True)

# Generate images for each of 5 images with 4 prompts each
for img_idx, prompts in enumerate(prompt_sets, 1):
    print(f"\n{'='*60}")
    print(f"Generating Image {img_idx}/5")
    print(f"{'='*60}")
    
    control_image = control_images[img_idx - 1]  # Get corresponding control image
    
    for name, prompt in prompts:
        print(f"\n  Prompt Type: {name}")
        print(f"  Prompt: '{prompt}'")
        
        generator = torch.manual_seed(img_idx)  # Different seed for each image
        image = pipe(
            prompt, 
            num_inference_steps=20, 
            generator=generator,
            height=control_image.height,
            width=control_image.width
        ).images[0]
        
        output_path = f"./results/sd15/{img_idx:06d}_{name}.jpg"
        image.save(output_path)
        print(f"  Saved: {output_path}")

print(f"\n{'='*60}")
print("Completed generating 5 images with 4 prompts each!")
print(f"{'='*60}")