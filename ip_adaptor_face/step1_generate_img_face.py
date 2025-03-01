import cv2
import torch
from insightface.app import FaceAnalysis
from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from . import face_img
import numpy as np
import os
def generate_image_from_face(image: np.ndarray, prompt: str = "", seed: int = 2025):
    """
    Generate an image from a face in the given CV2 image array using a prompt.

    :param image: Input image containing a face as a NumPy array (CV2 format).
    :param prompt: Text prompt for image generation.
    :param seed: Random seed for reproducibility.
    :return: Generated image as a NumPy array.
    """
    app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    faces = app.get(image)
    if len(faces) == 0:
        print("No faces found in the image.")
        return None
    faceid_embeds = torch.from_numpy(faces[0].normed_embedding).unsqueeze(0)
    base_model_path = "SG161222/RealVisXL_V3.0"
    #########################################################################################
    ip_ckpt = "/home/haoqian/PI/ultraPersona/ip_adaptor_face/ip-adapter-faceid_sdxl.bin"
    #########################################################################################
    device = "cuda"
    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    pipe = StableDiffusionXLPipeline.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        add_watermarker=False,
    )
    ip_model = face_img.IPAdapterFaceIDXL(pipe, ip_ckpt, device)
    if not prompt:
        prompt = "Wide shot, best quality, high quality,Create a highly detailed portrait of a person with a natural expression. The individual should be well-dressed in a smart outfit, showcasing their entire face. The background should be softly blurred to emphasize the subject, with warm lighting highlighting their features. Aim for a realistic and vibrant color palette that enhances the overall quality of the image."
    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
    images = ip_model.generate(
        prompt=prompt+"Wide shot, best quality, high quality,Create a highly detailed portrait of a person with a natural expression. The individual should be well-dressed in a smart outfit, showcasing their entire face. The background should be softly blurred to emphasize the subject, with warm lighting highlighting their features. Aim for a realistic and vibrant color palette that enhances the overall quality of the image.",
        negative_prompt=negative_prompt,
        faceid_embeds=faceid_embeds,
        num_samples=1,
        num_inference_steps=30,
        guidance_scale=7.5,
        seed=seed
    )
    
    generated_image = images[0]
    
    # Ensure the generated image is in the correct format for OpenV
    generated_image = np.array(generated_image)
    if generated_image.shape[-1] == 3:  # Check if it's RGB
        generated_image = cv2.cvtColor(generated_image, cv2.COLOR_RGB2BGR)

    return generated_image

if __name__ == "__main__":
    img_folder_path = "/home/haoqian/PI/demo_imgs/people/"
    result_folder_path = "/home/haoqian/PI/three_methods/ip_adaptor_face/faceid_result/"
    os.makedirs(result_folder_path, exist_ok=True)

    for i in range(1, 13):
        img_path = os.path.join(img_folder_path, f"{i}.jpeg")  # or .jpeg
        img = cv2.imread(img_path)
        if img is not None:
            face_id_img = generate_image_from_face(img, "")
            face_id_img_img_path = os.path.join(result_folder_path, f"{i}_face.png")
            
            success = cv2.imwrite(face_id_img_img_path, face_id_img)

            print(f"Processed {img_path} and saved to {face_id_img_img_path}")
        else:
            print(f"Image {img_path} not found or could not be read.")

