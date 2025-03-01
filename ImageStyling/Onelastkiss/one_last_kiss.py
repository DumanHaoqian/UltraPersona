import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from PIL import Image
from colorsys import hsv_to_rgb

def render(img_rgb):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (25, 25), 0, 0)
    img_blend = cv2.divide(img_gray, img_blur, scale=256)
    img_blend = cv2.cvtColor(img_blend, cv2.COLOR_GRAY2BGR)
    return img_blend

def rainbow_color_stops(height, width, start=5/10, end=9/10, theta=np.pi/16):
    colors = []
    for i in range(height):
        color = []
        for j in range(width):
            i_slope = i * np.cos(theta) - j * np.sin(theta)
            h = start + (end - start) * i_slope / height
            j_slope = i * np.sin(theta) + j * np.cos(theta)
            s = start + (end - start) * j_slope / width
            color.append(hsv_to_rgb(h, s, 0.8))
        colors.append(color)
    colors = np.array(colors, dtype=np.float32)  # Ensure the colors are float32
    return colors
def one_last_kiss(img_cv2):
    if img_cv2 is None:
        raise ValueError("Input image is None. Ensure the image is correctly loaded.")
    if not isinstance(img_cv2, np.ndarray):
        raise ValueError("Input image is not a valid numpy array.")
    
    height, width, _ = img_cv2.shape
    img_render = render(img_cv2)
    colors = rainbow_color_stops(height, width)
    
    img_render2 = img_render.copy().astype(np.float32)  # Ensure the image is float32
    img_render2 = img_render2 / 255.0
    for i in range(height):
        for j in range(width):
            if img_render2[i, j, :].sum() < 2.5:
                img_render2[i, j] = colors[i, j]
            else:
                img_render2[i, j] = [1, 1, 1]
    
    img_render2 = (img_render2 * 255).astype(np.uint8)  # Convert back to uint8 for PIL
    img = Image.fromarray(img_render2)
    #out=f"Pipelines/generated_imgs/final_img_{random.randint(1, 10000000000)}.png"
    #img.save(out)
    return img