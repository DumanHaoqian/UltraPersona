import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
from PIL import Image
import numpy as np


def generate(input_image_array: np.ndarray, prompt: str):
    pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
        "/home/haoqian/PI/three_methods/Image_Styling/ft_sdxl/animate.safetensors", 
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    negative_prompt = "photographic, photo, worst quality, bad anatomy, comics, cropped, cross-eyed, ugly, deformed, glitch, mutated, watermark, unprofessional, jpeg artifacts, low quality"
    init_image = Image.fromarray(input_image_array).convert("RGB")
    image = pipe(
        prompt, 
        image=init_image, 
        negative_prompt=negative_prompt, 
        strength=0.4, 
        guidance_scale=7, 
        num_inference_steps=50
    ).images[0]
    logo_path = "/home/haoqian/PI/ultraPersona/ImageStyling/animation/jojo_logo.png"
    logo = Image.open(logo_path).convert("RGBA")
    logo_size = (350, 150)  
    logo = logo.resize(logo_size, Image.LANCZOS)  
    image.paste(logo, (100, image.height - logo.height - 10), logo)
    return image

