"""Input analysis: look at the uploaded photo before generation and decide how to handle it.

Rules only (no model judgement), so the result is stable and testable.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .face_metrics import detect, to_bgr


@dataclass
class FaceInfo:
    bbox: list[float]
    size_px: int            # shorter side of the face box
    sharpness: float        # variance of Laplacian on the face crop
    yaw: float              # degrees, 0 = frontal
    det_score: float


@dataclass
class Analysis:
    people: int
    faces: list[FaceInfo]
    framing: str            # closeup | half_body | wide
    status: str             # ok | warn | reject
    messages: list[str] = field(default_factory=list)
    risk: str = "low"       # low | medium | high (drives retries and face repair)

    def as_dict(self):
        d = self.__dict__.copy()
        d["faces"] = [f.__dict__ for f in self.faces]
        return d


def _sharpness(bgr: np.ndarray, bbox) -> float:
    x0, y0, x1, y1 = [int(max(0, v)) for v in bbox]
    crop = bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (112, 112))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def analyze(img, cfg: dict) -> Analysis:
    a_cfg = cfg["analysis"]
    bgr = to_bgr(img)
    h = bgr.shape[0]
    raw = detect(bgr, det_thresh=0.5)
    faces = []
    for f in raw:
        x0, y0, x1, y1 = f.bbox
        yaw = float(f.pose[1]) if getattr(f, "pose", None) is not None else 0.0
        faces.append(FaceInfo([float(v) for v in f.bbox], int(min(x1 - x0, y1 - y0)),
                              _sharpness(bgr, f.bbox), yaw, float(f.det_score)))

    n = len(faces)
    msgs: list[str] = []
    status = "ok"
    if n == 0:
        return Analysis(0, [], "unknown", "reject", ["没有检测到人脸，请换一张能看清正脸的照片。"], "high")
    if n > a_cfg["max_people"]:
        status = "reject"
        msgs.append(f"检测到 {n} 个人，目前最多支持 {a_cfg['max_people']} 个人。")

    small = [i for i, f in enumerate(faces) if f.size_px < a_cfg["min_face_px"]]
    blurry = [i for i, f in enumerate(faces) if f.sharpness < a_cfg["min_sharpness"]]
    side = [i for i, f in enumerate(faces) if abs(f.yaw) > 45]
    if small:
        status = "warn" if status == "ok" else status
        msgs.append(f"从左数第 {', '.join(str(i + 1) for i in small)} 个人的脸太小，生成后可能不像本人。")
    if blurry:
        status = "warn" if status == "ok" else status
        msgs.append(f"从左数第 {', '.join(str(i + 1) for i in blurry)} 个人的脸不够清楚。")
    if side:
        status = "warn" if status == "ok" else status
        msgs.append(f"从左数第 {', '.join(str(i + 1) for i in side)} 个人是侧脸，生成后可能不像本人。")

    median_face = float(np.median([f.size_px for f in faces]))
    ratio = median_face / h
    framing = "closeup" if ratio > 0.22 else ("half_body" if ratio > 0.08 else "wide")

    risk = "low" if n == 1 else ("medium" if n <= 3 else "high")
    if small or blurry or side:
        risk = "high"
    return Analysis(n, faces, framing, status, msgs, risk)
