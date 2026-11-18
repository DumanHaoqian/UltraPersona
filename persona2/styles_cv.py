"""Non-model styles. Same look as the v1 One Last Kiss style, but vectorised (the old per-pixel loops
took tens of seconds on a 1700x1276 image)."""
import cv2
import numpy as np
from PIL import Image


def one_last_kiss(img: Image.Image) -> Image.Image:
    rgb = np.asarray(img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blend = cv2.divide(gray, cv2.GaussianBlur(gray, (25, 25), 0), scale=256)
    h, w = gray.shape
    i, j = np.mgrid[0:h, 0:w].astype(np.float32)
    theta, start, end = np.pi / 16, 0.5, 0.9
    hue = start + (end - start) * (i * np.cos(theta) - j * np.sin(theta)) / h
    sat = start + (end - start) * (i * np.sin(theta) + j * np.cos(theta)) / w
    hsv = np.stack([np.mod(hue, 1.0) * 180, np.clip(sat, 0, 1) * 255, np.full_like(hue, 0.8 * 255)], -1)
    colors = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    lines = (blend.astype(np.float32) * 3 / 255.0) < 2.5          # same test as the old code
    out = np.where(lines[..., None], colors, 255).astype(np.uint8)
    return Image.fromarray(out)
