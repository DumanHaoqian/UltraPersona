import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
from PIL import Image
import numpy as np
import os
import cv2

def generate(input_image_array: np.ndarray, prompt: str):
    negative_prompt = "photographic, photo, worst quality, bad anatomy, comics, cropped, cross-eyed, ugly, deformed, glitch, mutated, watermark, unprofessional, jpeg artifacts, low quality"
    pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
        "/home/haoqian/PI/ultraPersona/ImageStyling/protovision/protovisionXLHighFidelity3D_releaseV660Bakedvae.safetensors", 
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    init_image = Image.fromarray(input_image_array).convert("RGB")
    image = pipe(
        prompt, 
        image=init_image, 
        negative_prompt=negative_prompt, 
        strength=0.4, 
        guidance_scale=7, 
        num_inference_steps=50
    ).images[0]
    return image
    
