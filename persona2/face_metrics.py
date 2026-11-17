"""Face identity metrics.

A face recognition model turns each face into a 512-number vector. Two photos of
the same person give vectors that point in almost the same direction; photos of
different people do not. The metric is the cosine similarity of the two vectors
(1 = identical, lower = less alike).

For several people: detect all faces in the input and output, then pair them
one-to-one with the Hungarian algorithm on the similarity matrix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@lru_cache(maxsize=4)
def get_face_app(name: str = "buffalo_l", det_thresh: float = 0.5, det_size: int = 640):
    """InsightFace detector + recognizer, CPU only (no VRAM, no CUDA build issues)."""
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(name=name, providers=["CPUExecutionProvider"],
                       allowed_modules=["detection", "recognition", "landmark_3d_68"])
    app.prepare(ctx_id=-1, det_thresh=det_thresh, det_size=(det_size, det_size))
    return app


def to_bgr(img) -> np.ndarray:
    """Accept a PIL image (RGB) or a numpy array (assumed BGR) and return BGR uint8."""
    if hasattr(img, "convert"):
        return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    return img


def detect(img, det_thresh: float = 0.5, model: str = "buffalo_l"):
    """Faces sorted left to right."""
    faces = get_face_app(model, det_thresh).get(to_bgr(img))
    return sorted(faces, key=lambda f: f.bbox[0])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


@dataclass
class FaceMatch:
    n_input: int
    n_output: int
    count_ok: bool
    per_person: list[float] = field(default_factory=list)   # input order (left to right); missing = 0.0
    worst: float = 0.0
    mean: float = 0.0
    swaps: int = 0          # matched output face is more similar to another input person
    detected_ratio: float = 0.0

    def as_dict(self):
        return self.__dict__.copy()


def match_faces(input_faces, output_faces) -> FaceMatch:
    n_in, n_out = len(input_faces), len(output_faces)
    if n_in == 0:
        return FaceMatch(0, n_out, n_out == 0)
    if n_out == 0:
        return FaceMatch(n_in, 0, False, [0.0] * n_in, 0.0, 0.0, 0, 0.0)
    a = np.stack([f.normed_embedding for f in input_faces])
    b = np.stack([f.normed_embedding for f in output_faces])
    sim = a @ b.T                                   # n_in x n_out
    rows, cols = linear_sum_assignment(-sim)
    per = [0.0] * n_in
    swaps = 0
    for r, c in zip(rows, cols):
        per[r] = float(sim[r, c])
        if n_in > 1 and sim[:, c].argmax() != r:
            swaps += 1
    return FaceMatch(
        n_input=n_in, n_output=n_out, count_ok=(n_in == n_out), per_person=per,
        worst=float(min(per)), mean=float(np.mean(per)), swaps=swaps,
        detected_ratio=len(rows) / n_in,
    )


def compare(input_img, output_img, model: str = "buffalo_l", out_det_thresh: float = 0.3) -> FaceMatch:
    """Lower detection threshold on the output: painted faces score lower."""
    return match_faces(detect(input_img, 0.5, model), detect(output_img, out_det_thresh, model))
