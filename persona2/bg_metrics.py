"""How well the background of a generated picture keeps the original PolyU background.

Only the area outside the people is scored (people mask from Mask2Former, grown by a margin).

- ssim: structural similarity of the grey images, averaged over the background area. 1 = identical.
  Good for the photo style; drops for any art style even when the buildings are kept.
- sift_inliers / sift_ratio: SIFT key points of the original background matched to the output with a
  ratio test, then a RANSAC homography. Inliers are matches that agree with one geometric mapping.
  This still works when the texture is repainted in a style, so it measures "are the same buildings
  in the same place". ratio = inliers / key points found in the original background area.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a * a) - ma ** 2, blur(b * b) - mb ** 2, blur(a * b) - ma * mb
    return ((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))


def background_scores(output: Image.Image, background: Image.Image, people: np.ndarray | None = None,
                      margin_px: int = 15) -> dict:
    """people: HxW mask of the people in `output` (1 = person). Computed with Mask2Former if None."""
    if people is None:
        from .matting import person_mask
        people = person_mask(output)
    out = cv2.cvtColor(np.asarray(output.convert("RGB")), cv2.COLOR_RGB2GRAY)
    bg = cv2.cvtColor(np.asarray(background.convert("RGB").resize(output.size, Image.LANCZOS)), cv2.COLOR_RGB2GRAY)
    m = cv2.dilate((people > 0.5).astype(np.uint8), np.ones((2 * margin_px + 1,) * 2, np.uint8))
    keep = (m == 0)
    ssim = float(_ssim_map(out, bg)[keep].mean()) if keep.any() else float("nan")

    sift = cv2.SIFT_create(nfeatures=4000)
    kb, db = sift.detectAndCompute(bg, keep.astype(np.uint8) * 255)
    ko, do = sift.detectAndCompute(out, keep.astype(np.uint8) * 255)
    inliers = 0
    if db is not None and do is not None and len(kb) >= 8 and len(ko) >= 8:
        pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(db, do, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        if len(good) >= 8:
            src = np.float32([kb[g.queryIdx].pt for g in good])
            dst = np.float32([ko[g.trainIdx].pt for g in good])
            _, inl = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            inliers = int(inl.sum()) if inl is not None else 0
    return {"ssim": ssim, "sift_inliers": inliers, "sift_ratio": inliers / max(1, len(kb)),
            "bg_area": float(keep.mean())}
