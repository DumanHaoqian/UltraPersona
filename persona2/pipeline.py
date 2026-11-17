"""UltraPersona v2 pipeline.

  analyze photo -> build prompt -> Qwen-Image-Edit (one step, or compose then stylize)
  -> face check against the input photo -> retry with a new seed / repair single faces -> result
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import cv2
import numpy as np
import yaml
from PIL import Image

from . import prompts
from .analyze import analyze
from .face_metrics import FaceMatch, compare, detect
from .qwen_edit import edit, load_pipeline
from .styles_cv import one_last_kiss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: str | None = None) -> dict:
    return yaml.safe_load(open(path or os.path.join(ROOT, "persona2", "config.yaml"), encoding="utf-8"))


@dataclass
class Result:
    image: Image.Image | None
    composite: Image.Image | None
    analysis: dict
    faces: dict | None = None
    attempts: list[dict] = field(default_factory=list)
    prompts: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _score(m: FaceMatch):
    return (m.count_ok, m.worst)


class Persona:
    def __init__(self, cfg: dict | None = None, preset: str | None = None, pipe=None):
        self.cfg = cfg or load_config()
        mc = self.cfg["model"]
        self.preset = preset or mc["default_preset"]
        self.pipe = pipe or load_pipeline(mc["path"], self.preset, mc.get("quantized_dir"))

    # -- helpers ----------------------------------------------------------------------------
    def background(self, bg_id: int) -> tuple[dict, Image.Image]:
        bg = self.cfg["backgrounds"][int(bg_id)]
        return bg, Image.open(os.path.join(ROOT, bg["file"])).convert("RGB")

    def _edit(self, images, prompt, seed):
        mc = self.cfg["model"]
        return edit(self.pipe, images, prompt, prompts.NEGATIVE, seed, mc["steps"], mc["true_cfg_scale"])

    # -- main entry -------------------------------------------------------------------------
    def generate(self, photo: Image.Image, bg_id: int, style_id: int, mode: str = "twostep",
                 seed: int = 0, gate: bool = True) -> Result:
        """mode: 'onestep' (compose + style in one edit) or 'twostep' (compose, then stylize)."""
        photo = photo.convert("RGB")
        an = analyze(photo, self.cfg)
        res = Result(None, None, an.as_dict(), warnings=list(an.messages))
        if an.status == "reject":
            return res
        bg, bg_img = self.background(bg_id)
        if an.people > bg["max_people"]:
            res.warnings.append(f"{bg['name']} 最多放 {bg['max_people']} 个人，请换一个更开阔的背景。")
            return res
        style = self.cfg["styles"][int(style_id)]
        q = self.cfg["quality"]
        retries = q["max_retries"] if gate else 0
        if an.risk == "high" and gate:
            retries += 1

        best = None
        for k in range(retries + 1):
            s = seed + k
            composite, final, used = self._run_once(photo, an, bg, bg_img, style, mode, s)
            m = compare(photo, final if style["engine"] == "qwen" else composite)
            res.attempts.append({"seed": s, **m.as_dict()})
            if best is None or _score(m) > _score(best[2]):
                best = (composite, final, m, used)
            if not gate or (m.count_ok and m.worst >= q["face_sim_min"]):
                break

        composite, final, m, used = best
        if gate and an.people > 1 and m.count_ok and m.worst < q["face_sim_min"]:
            final = self.repair_faces(photo, final, style, m, seed)
            m = compare(photo, final)
            res.attempts.append({"seed": "repair", **m.as_dict()})
        if style["engine"] == "opencv":
            final = one_last_kiss(composite)
        if gate and m.worst < q["face_sim_min"]:
            res.warnings.append("生成结果和本人的相似度偏低，可以换一张更清楚的正脸照片再试。")
        res.image, res.composite, res.faces, res.prompts = final, composite, m.as_dict(), used
        return res

    def _compose(self, photo, an, bg, bg_img, seed):
        p = prompts.compose_prompt(an.people, an.framing, bg)
        comp = self._edit([photo, bg_img], p, seed)
        if self.cfg["model"].get("restore_background"):
            from .matting import restore_background
            comp = restore_background(comp, bg_img)
        return comp, p

    def _run_once(self, photo, an, bg, bg_img, style, mode, seed):
        n, fr = an.people, an.framing
        if style["engine"] != "qwen":
            comp, p = self._compose(photo, an, bg, bg_img, seed)
            return comp, comp, [p]
        if mode == "onestep":
            p = prompts.onestep_prompt(n, fr, bg, style)
            out = self._edit([photo, bg_img], p, seed)
            return None, out, [p]
        comp, p1 = self._compose(photo, an, bg, bg_img, seed)
        p2 = prompts.stylize_prompt(n, style)
        out = self._edit([comp], p2, seed)
        return comp, out, [p1, p2]

    # -- single-face repair -----------------------------------------------------------------
    def repair_faces(self, photo, out_img, style, m: FaceMatch, seed: int) -> Image.Image:
        """Re-draw each face whose similarity is below the threshold, using a larger crop."""
        thr = self.cfg["quality"]["face_sim_min"]
        in_faces = detect(photo, 0.5)
        out_faces = detect(out_img, 0.3)
        if len(in_faces) != len(out_faces):
            return out_img
        out = np.asarray(out_img).copy()
        style_txt = f" Keep the {style['prompt']} look." if style["engine"] == "qwen" else ""
        for i, (fi, fo) in enumerate(zip(in_faces, out_faces)):
            if m.per_person[i] >= thr:
                continue
            ref = _crop(np.asarray(photo), fi.bbox, 1.8)
            box = _expand(fo.bbox, 2.2, out.shape)
            x0, y0, x1, y1 = box
            region = Image.fromarray(out[y0:y1, x0:x1]).resize((768, 768), Image.LANCZOS)
            p = ("Redraw the face in Picture 2 so it is clearly the same person as in Picture 1: same face "
                 "shape, eyes, nose, mouth and hairstyle. Keep the pose, head angle, lighting and "
                 f"everything outside the face unchanged.{style_txt}")
            fixed = self._edit([Image.fromarray(ref), region], p, seed)
            fixed = np.asarray(fixed.resize((x1 - x0, y1 - y0), Image.LANCZOS))
            mask = np.zeros((y1 - y0, x1 - x0), np.float32)
            cv2.ellipse(mask, ((x1 - x0) // 2, (y1 - y0) // 2), (int((x1 - x0) * 0.32), int((y1 - y0) * 0.4)),
                        0, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (0, 0), (x1 - x0) * 0.06)[..., None]
            out[y0:y1, x0:x1] = (fixed * mask + out[y0:y1, x0:x1] * (1 - mask)).astype(np.uint8)
        return Image.fromarray(out)


def _expand(bbox, scale, shape):
    x0, y0, x1, y1 = bbox
    cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * scale / 2
    h, w = shape[:2]
    return int(max(0, cx - s)), int(max(0, cy - s)), int(min(w, cx + s)), int(min(h, cy + s))


def _crop(arr, bbox, scale):
    x0, y0, x1, y1 = _expand(bbox, scale, arr.shape)
    return arr[y0:y1, x0:x1]
