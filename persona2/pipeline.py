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
        lt = mc.get("lightning")
        self.fast = mc["lightning_models"][lt] if lt else None    # step-distilled weights, few steps, no CFG
        self.pipe = pipe or load_pipeline(mc["path"], self.preset, mc.get("quantized_dir"),
                                          transformer_path=self.fast["transformer"] if self.fast else None)

    # -- helpers ----------------------------------------------------------------------------
    def background(self, bg_id: int) -> tuple[dict, Image.Image]:
        bg = self.cfg["backgrounds"][int(bg_id)]
        return bg, Image.open(os.path.join(ROOT, bg["file"])).convert("RGB")

    def _edit(self, images, prompt, seed, lock=None, guide=None, steps=None):
        mc = self.cfg["model"]
        if self.fast:   # distilled: fixed short schedule, no classifier-free guidance (one pass per step)
            return edit(self.pipe, images, prompt, prompts.NEGATIVE, seed, self.fast["steps"], 1.0,
                        lock=lock, guide=guide)
        return edit(self.pipe, images, prompt, prompts.NEGATIVE, seed, steps or mc["steps"], mc["true_cfg_scale"],
                    lock=lock, guide=guide)

    @property
    def ctl(self) -> dict:
        return self.cfg.get("control") or {}

    def make_guide(self, boxes, refs):
        from .control import Guide

        c = self.ctl
        return Guide(list(boxes), list(refs), scale=c.get("guide_scale", 0.05),
                     sigma_range=tuple(c.get("guide_sigma", (0.4, 0.85))), blur=c.get("guide_blur", 1.5),
                     n_aug=c.get("guide_aug", 4))

    def styled_background(self, bg_id: int, style_id: int, seed: int = 0) -> Image.Image:
        """The background alone, redrawn in the style (no people). Cached on disk; used as the lock target
        for art styles, so the lock keeps the layout without pulling the picture back to a photo."""
        style = self.cfg["styles"][int(style_id)]
        d = os.path.join(ROOT, "persona2", "cache", "styled_bg")
        path = os.path.join(d, f"bg{bg_id}_st{style_id}_s{seed}.png")
        if os.path.exists(path):
            return Image.open(path).convert("RGB")
        _, bg_img = self.background(bg_id)
        p = (f"Turn this photo into {style['prompt']}. Keep every building, sign, staircase, tree and lawn in "
             f"exactly the same shape and position; keep the whole layout. Do not add any people.")
        img = self._edit([bg_img], p, seed)
        os.makedirs(d, exist_ok=True)
        img.save(path)
        return img

    def _controls(self, photo, composite, bg_img, style, target=None):
        """Background lock and identity guidance for the final edit, planned from the unlocked composite:
        the people mask says where the model may draw, the composite's faces say where to guide."""
        from .control import Lock, reference_embedding, soft_mask
        from .matting import person_mask

        c = self.ctl
        lock = guide = None
        if c.get("lock"):
            mask = soft_mask(person_mask(composite), c.get("mask_dilate_px", 24), c.get("mask_feather_px", 6))
            release = style.get("lock_release", c["lock_release_photo"] if style["engine"] != "qwen"
                                else c["lock_release_style"])
            # people from the free first edit anchor size and place for the first steps
            lock = Lock(target if target is not None else bg_img, mask, release, inner=composite,
                        inner_release=c.get("lock_inner_release", 0.2))
        if c.get("id_guide"):
            pairs = matched_faces(photo, composite)
            if pairs:
                guide = self.make_guide([fo.bbox for _, fo in pairs],
                                        [reference_embedding(photo, fi.bbox) for fi, _ in pairs])
        return lock, guide

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
        repair_on = self.ctl.get("repair", True)
        if style["engine"] == "qwen" and not self.ctl.get("repair_art_styles"):
            repair_on = False
        if gate and repair_on and (an.people > 1 or self.ctl.get("repair_single")) and m.worst < q["face_sim_min"]:
            final = self.repair_faces(photo, final, style, seed)
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
        if self.ctl.get("lock") or self.ctl.get("id_guide"):
            return self._run_controlled(photo, an, bg, bg_img, style, mode, seed)
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

    def _run_controlled(self, photo, an, bg, bg_img, style, mode, seed):
        """A free first edit with the same prompt gives the people mask and face positions; the second edit
        runs under control. The plan must come from the same prompt: a plain composite draws people at a
        different size than a styled edit, and the lock would then force the styled people into it."""
        n, fr = an.people, an.framing
        if style["engine"] != "qwen":
            p1 = prompts.compose_prompt(n, fr, bg)
            first = self._edit([photo, bg_img], p1, seed)
            lock, guide = self._controls(photo, first, bg_img, style)
            out = self._keep_count(photo, first, self._edit([photo, bg_img], p1, seed, lock, guide))
            return out, out, [p1]
        if mode == "onestep":
            p = prompts.onestep_prompt(n, fr, bg, style)
            first = self._edit([photo, bg_img], p, seed)
            target = self._lock_target(bg, style, seed)
            lock, guide = self._controls(photo, first, bg_img, style, target)
            return None, self._keep_count(photo, first, self._edit([photo, bg_img], p, seed, lock, guide)), [p]
        p1 = prompts.compose_prompt(n, fr, bg)
        comp = self._edit([photo, bg_img], p1, seed)
        lock, guide = self._controls(photo, comp, bg_img, style, self._lock_target(bg, style, seed))
        p2 = prompts.stylize_prompt(n, style)
        return comp, self._edit([comp], p2, seed, lock, guide), [p1, p2]

    @staticmethod
    def _keep_count(photo, free, locked):
        """The lock can add a stray figure in the background. If the locked picture has the wrong number of
        faces and the free one has the right number, use the free one."""
        if compare(photo, locked).count_ok or not compare(photo, free).count_ok:
            return locked
        return free

    def _lock_target(self, bg, style, seed):
        if not self.ctl.get("lock") or self.ctl.get("lock_target_style", "styled") != "styled":
            return None
        bg_id = next(k for k, v in self.cfg["backgrounds"].items() if v is bg)
        style_id = next(k for k, v in self.cfg["styles"].items() if v is style)
        return self.styled_background(bg_id, style_id, seed)

    # -- single-face repair -----------------------------------------------------------------
    def repair_faces(self, photo, out_img, style, seed: int, thr: float | None = None,
                     log: list | None = None) -> Image.Image:
        """Re-draw each face whose similarity is below the threshold at a larger scale.

        The face region is cut out and enlarged to 1024x1024, so a 40-pixel face gets many more latent
        cells. The edit sees [reference face from the photo, enlarged region]. Inside the region,
        everything outside an ellipse around the face is locked to the region itself, and identity
        guidance pulls the face towards the reference. The result is pasted back with a soft ellipse.
        """
        from .control import Lock, reference_embedding

        c = self.ctl
        thr = self.cfg["quality"]["face_sim_min"] if thr is None else thr
        out = np.asarray(out_img.convert("RGB")).copy()
        pairs = matched_faces(photo, out_img, with_sim=True)
        style_txt = f" Keep the {style['prompt']} look." if style["engine"] == "qwen" else ""
        p = ("The face of the person in Picture 2 is blank. Draw the face of the person in Picture 1 there: "
             "same face shape, eyes, nose, mouth, skin tone and expression, turned to fit the head in Picture 2. "
             f"Keep everything outside the blank area unchanged.{style_txt}")
        scale = c.get("repair_region_scale", 3.0)
        for fi, fo, sim in pairs:
            if sim >= thr:
                continue
            x0, y0, x1, y1 = _expand(fo.bbox, scale, out.shape)
            rw, rh = x1 - x0, y1 - y0
            if min(rw, rh) < 16:
                continue
            R = 1024  # the pipeline generates ~1 megapixel, so a square region comes back at 1024x1024
            region = Image.fromarray(out[y0:y1, x0:x1]).resize((R, R), Image.LANCZOS)
            kx, ky = R / rw, R / rh
            fb = [(fo.bbox[0] - x0) * kx, (fo.bbox[1] - y0) * ky, (fo.bbox[2] - x0) * kx, (fo.bbox[3] - y0) * ky]
            lock = guide = None
            if c.get("repair_lock", True):
                lock = Lock(region, _ellipse(R, R, fb, grow=1.35, feather=R / 40), c.get("repair_lock_release", 0.8))
            if c.get("repair_guide", True):
                guide = self.make_guide([fb], [reference_embedding(photo, fi.bbox)])
            ref = Image.fromarray(_crop(np.asarray(photo.convert("RGB")), fi.bbox, 1.8))
            # Blank the old face, so the model cannot copy it and must take the face from the reference.
            hole = _ellipse(R, R, fb, grow=1.05, feather=R / 80)[..., None]
            arr = np.asarray(region).astype(np.float32)
            blank = cv2.GaussianBlur(arr, (0, 0), R / 25) * 0.3 + 128 * 0.7
            region_in = Image.fromarray((arr * (1 - hole) + blank * hole).astype(np.uint8))
            fixed = self._edit([ref, region_in], p, seed, lock, guide, steps=c.get("repair_steps"))
            fixed = np.asarray(fixed.resize((rw, rh), Image.LANCZOS)).astype(np.float32)
            local = [fo.bbox[0] - x0, fo.bbox[1] - y0, fo.bbox[2] - x0, fo.bbox[3] - y0]
            paste = _ellipse(rw, rh, local, grow=1.2, feather=max(2.0, rw * 0.03))[..., None]
            fixed = _match_colour(fixed, out[y0:y1, x0:x1].astype(np.float32), paste[..., 0] > 0.5)
            cand = out.copy()
            cand[y0:y1, x0:x1] = (fixed * paste + out[y0:y1, x0:x1] * (1 - paste)).astype(np.uint8)
            # Keep the new face only if it is closer to the person than the old one (the redraw can fail).
            after = _face_sim_at(cand, fo.bbox, fi.normed_embedding)
            kept = after is not None and after > sim
            if kept:
                out = cand
            if log is not None:
                log.append({"face": [round(float(v)) for v in fo.bbox], "before": round(sim, 3),
                            "after": None if after is None else round(after, 3), "kept": kept,
                            "guide": guide.log[-1] if guide and guide.log else None})
        return Image.fromarray(out)


def _expand(bbox, scale, shape):
    x0, y0, x1, y1 = bbox
    cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * scale / 2
    h, w = shape[:2]
    return int(max(0, cx - s)), int(max(0, cy - s)), int(min(w, cx + s)), int(min(h, cy + s))


def _crop(arr, bbox, scale):
    x0, y0, x1, y1 = _expand(bbox, scale, arr.shape)
    return arr[y0:y1, x0:x1]


def _face_sim_at(arr, bbox, ref_emb):
    """ArcFace similarity of the face in arr that overlaps bbox most; None if no face is found there."""
    x0, y0, x1, y1 = bbox
    best, best_iou = None, 0.3
    for f in detect(Image.fromarray(arr), 0.3):
        a0, b0, a1, b1 = f.bbox
        iw, ih = max(0, min(x1, a1) - max(x0, a0)), max(0, min(y1, b1) - max(y0, b0))
        inter = iw * ih
        iou = inter / ((x1 - x0) * (y1 - y0) + (a1 - a0) * (b1 - b0) - inter + 1e-6)
        if iou > best_iou:
            best, best_iou = float(np.dot(f.normed_embedding, ref_emb)), iou
    return best


def _ellipse(w: int, h: int, bbox, grow: float = 1.3, feather: float = 8.0) -> np.ndarray:
    """Soft ellipse (1 inside) around a face box, in a w x h canvas."""
    x0, y0, x1, y1 = bbox
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (int((x0 + x1) / 2), int((y0 + y1) / 2 - (y1 - y0) * 0.05)),
                (int((x1 - x0) / 2 * grow), int((y1 - y0) / 2 * grow * 1.15)), 0, 0, 360, 1.0, -1)
    return cv2.GaussianBlur(m, (0, 0), feather) if feather > 0 else m


def matched_faces(photo, img, with_sim: bool = False):
    """Pair input faces with faces in img (Hungarian on ArcFace similarity).
    Returns [(input_face, output_face)] or [(input_face, output_face, similarity)]."""
    from scipy.optimize import linear_sum_assignment

    fi, fo = detect(photo, 0.5), detect(img, 0.3)
    if not fi or not fo:
        return []
    sim = np.stack([f.normed_embedding for f in fi]) @ np.stack([f.normed_embedding for f in fo]).T
    rows, cols = linear_sum_assignment(-sim)
    return [(fi[r], fo[c], float(sim[r, c])) if with_sim else (fi[r], fo[c]) for r, c in zip(rows, cols)]


def _match_colour(src: np.ndarray, ref: np.ndarray, area: np.ndarray) -> np.ndarray:
    """Give src the mean and spread of ref inside `area`, per LAB channel (the redrawn face can come back
    with a different skin tone from the rest of the picture)."""
    if area.sum() < 16:
        return src
    a = cv2.cvtColor(np.clip(src, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    b = cv2.cvtColor(np.clip(ref, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    for ch in range(3):
        ma, sa = a[..., ch][area].mean(), a[..., ch][area].std() + 1e-6
        mb, sb = b[..., ch][area].mean(), b[..., ch][area].std() + 1e-6
        a[..., ch] = (a[..., ch] - ma) * (sb / sa) + mb
    return cv2.cvtColor(np.clip(a, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
