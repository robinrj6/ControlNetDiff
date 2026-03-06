from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
import torch

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

prompt = "create a realistic mountains with sunlight on background"

# generate image (no control image needed)
generator = torch.manual_seed(0)
image = pipe(
    prompt, num_inference_steps=20, generator=generator
).images[0]
image.save("./output_sd_only.png")