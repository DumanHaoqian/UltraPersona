"""Person segmentation and background restoration.

Uses Mask2Former (COCO panoptic, built into transformers, no remote code) and keeps the "person" class.

Qwen-Image-Edit redraws the whole picture, so PolyU buildings and signs can change shape. restore_background()
cuts the people out of the generated picture and puts them back onto the original background photo.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
import torch
from PIL import Image

SEG_MODEL = "facebook/mask2former-swin-large-coco-panoptic"


@lru_cache(maxsize=1)
def _model():
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    proc = AutoImageProcessor.from_pretrained(SEG_MODEL)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(SEG_MODEL).eval().to("cuda")
    person_id = next(i for i, n in model.config.id2label.items() if n == "person")
    return proc, model, person_id


@torch.no_grad()
def person_mask(img: Image.Image) -> np.ndarray:
    """Float mask in {0, 1} of all people, same size as img."""
    proc, model, person_id = _model()
    inputs = proc(images=img.convert("RGB"), return_tensors="pt").to("cuda")
    out = model(**inputs)
    seg = proc.post_process_panoptic_segmentation(out, target_sizes=[img.size[::-1]], threshold=0.5)[0]
    ids = [s["id"] for s in seg["segments_info"] if s["label_id"] == person_id]
    m = torch.isin(seg["segmentation"], torch.tensor(ids, device=seg["segmentation"].device)) if ids else         torch.zeros_like(seg["segmentation"], dtype=torch.bool)
    return m.float().cpu().numpy()


def restore_background(generated: Image.Image, background: Image.Image, feather_px: float = 1.5,
                       dilate_px: int = -1) -> Image.Image:
    """Keep the generated people, take everything else from the original background photo resized to the
    generated picture's size. Trade-off: landmarks stay exact, but light and shadows around the people
    no longer match as well as in the full generated picture."""
    gen = np.asarray(generated.convert("RGB")).astype(np.float32)
    bg = np.asarray(background.convert("RGB").resize(generated.size, Image.LANCZOS)).astype(np.float32)
    m = person_mask(generated)
    m = (m > 0.5).astype(np.uint8)
    k = np.ones((abs(dilate_px) * 2 + 1,) * 2, np.uint8)
    if dilate_px > 0:
        m = cv2.dilate(m, k)
    elif dilate_px < 0:
        m = cv2.erode(m, k)  # pull the edge inwards so no generated background shows as a halo
    m = m.astype(np.float32)
    if feather_px:
        m = cv2.GaussianBlur(m, (0, 0), feather_px)
    m = m[..., None]
    return Image.fromarray((gen * m + bg * (1 - m)).clip(0, 255).astype(np.uint8))


def free():
    _model.cache_clear()
    torch.cuda.empty_cache()
