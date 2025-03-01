import os
import cv2
import PIL.Image as Image
import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

# Initialize the Stable Diffusion model
model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)


def style_transfer(image, prompt="Make it like a animation", num_inference_steps=10, image_guidance_scale=1):

    images = pipe(prompt, image=image, num_inference_steps=num_inference_steps, image_guidance_scale=image_guidance_scale).images
    return images[0]  # Return the first generated image

def generate(image, prompt="Make it like a animation"):
    image=Image.fromarray(image)
    styled_image = style_transfer(image, prompt=prompt)
    return styled_image