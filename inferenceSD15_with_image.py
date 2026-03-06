from diffusers import StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch

base_model_path = "/home/woody/rlvl/rlvl165v/ControlNetDiff/shared/models/sd15/"

# Load Stable Diffusion Image-to-Image pipeline
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    local_files_only=True
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

# Load the control image as the initial image
init_image = load_image("./mountain.jpg")
prompt = "create a realistic mountains with sunlight on background"

# generate image using img2img (strength controls how much to transform the input image)
# strength=0.75 means 75% transformation, higher = more change, lower = more similar to input
generator = torch.manual_seed(0)
image = pipe(
    prompt, 
    image=init_image,
    num_inference_steps=20, 
    strength=0.75,
    generator=generator
).images[0]
image.save("./output_sd_only_with_image.png")