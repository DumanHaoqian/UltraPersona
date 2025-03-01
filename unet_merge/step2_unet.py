import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import albumentations as albu
from iglovikov_helper_functions.utils.image_utils import load_rgb, pad, unpad
from iglovikov_helper_functions.dl.pytorch.utils import tensor_from_rgb_image
from people_segmentation.pre_trained_models import create_model

model = create_model("Unet_2020-07-20")
model.eval()

def segment_people(image):
    """Segment people from the image using a pre-trained model."""
    transform = albu.Compose([albu.Normalize(p=1)], p=1)
    
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    padded_image, pads = pad(image, factor=32, border=cv2.BORDER_CONSTANT)
    x = transform(image=padded_image)["image"]
    x = torch.unsqueeze(tensor_from_rgb_image(x), 0)
    
    with torch.no_grad():
        prediction = model(x)[0][0]
    
    mask = (prediction > 0).cpu().numpy().astype(np.uint8)
    mask = unpad(mask, pads)
    return mask, image

def load_background(choice):
    """Load the background image based on the user's choice."""
    background_paths = {
        1: "/home/haoqian/PI/ultraPersona/backgrounds/Y.jpg",
        2: "/home/haoqian/PI/ultraPersona/backgrounds/lab.jpeg",
        3: "/home/haoqian/PI/ultraPersona/backgrounds/grass.jpeg",
        4: "/home/haoqian/PI/ultraPersona/backgrounds/a_garden.jpeg",
        5: "/home/haoqian/PI/ultraPersona/backgrounds/clock.jpeg",
        6: "/home/haoqian/PI/ultraPersona/backgrounds/lib.jpeg",
        7: "/home/haoqian/PI/ultraPersona/backgrounds/lib_g.jpeg",
        8: "/home/haoqian/PI/ultraPersona/backgrounds/newcan.jpeg",
    }
    
    if choice not in background_paths:
        raise ValueError("Invalid choice for background image.")
    
    return load_rgb(background_paths[choice])

def extract_and_place_person(original_image, choice):
    """Extract the person from the image and place them in a chosen background."""
    mask, person_image = segment_people(original_image)
    
    if person_image.shape[2] == 3:
        person_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
    
    person_extracted = cv2.bitwise_and(person_image, person_image, mask=mask)
    indices = np.argwhere(mask > 0)
    
    if indices.size == 0:
        raise ValueError("No person detected in the image.")
    
    # Crop the person based on mask
    y_min, y_max = indices[:, 0].min(), indices[:, 0].max()
    x_min, x_max = indices[:, 1].min(), indices[:, 1].max()
    cropped_person = person_extracted[y_min:y_max + 1, x_min:x_max + 1]
    cropped_mask = mask[y_min:y_max + 1, x_min:x_max + 1]  

    background_image = load_background(choice)
    background_image=cv2.cvtColor(background_image,cv2.COLOR_BGR2RGB)
    
    # Resize person for background
    target_height = int(background_image.shape[0] * 0.5)
    scale = target_height / cropped_person.shape[0]
    
    person_resized = cv2.resize(cropped_person, (int(cropped_person.shape[1] * scale), target_height))
    mask_resized = cv2.resize(cropped_mask, (int(cropped_mask.shape[1] * scale), target_height), interpolation=cv2.INTER_NEAREST)

    # Define offsets for placing the person
    offsets = {
        1: (638, 750), 
        2: (525, 500),
        3: (400, 500),
        4: (638, 500),
        5: (638, 600),
        6: (638, 300),
        7: (638, 500),
        8: (411, 300),
    }

    if choice not in offsets:
        raise ValueError("Invalid choice for offsets.")
    
    y_offset, x_offset = offsets[choice]

    # Check if the person fits in the background
    if (y_offset < 0 or x_offset < 0 or 
        y_offset + person_resized.shape[0] > background_image.shape[0] or 
        x_offset + person_resized.shape[1] > background_image.shape[1]):
        raise ValueError("The resized person exceeds the background dimensions.")
    
    # Place the person on the background
    for i in range(person_resized.shape[0]):
        for j in range(person_resized.shape[1]):
            if mask_resized[i, j] > 0:
                background_image[y_offset + i, x_offset + j] = person_resized[i, j]
    
    return background_image

if __name__ == "__main__":
    img_path = "/home/haoqian/PI/demo_imgs/peoples/dhq.jpeg"
    img = cv2.imread(img_path)
    
    if img is None:
        raise FileNotFoundError("Image not found. Please check the file path.")
    
    mask, img = segment_people(img)
    background_choice = 1  
    final_image = extract_and_place_person(img, background_choice)
    cv2.imwrite("final_image.jpg", cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))