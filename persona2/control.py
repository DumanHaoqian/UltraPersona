"""Control the denoising of Qwen-Image-Edit from outside the model.

Two controls, both applied after every scheduler step (no change to the model weights):

1. Background lock (latent blending, as in RePaint / Blended Latent Diffusion).
   The pipeline denoises a packed latent z_t. Flow matching uses z_t = (1 - s) * x0 + s * noise.
   Outside the people mask we replace z_t with the target background latent at the same noise level s,
   using the same starting noise. The model only generates the people and a band around them, so it fits
   them to a background that is held fixed. The lock is released after `release` of the steps; the last
   steps then refine the whole picture together, which removes cut-out edges.
   For art styles a small `release` (e.g. 0.4) keeps the layout of the buildings (set in early steps)
   but lets the model repaint their texture in the style.

2. Identity guidance.
   From the model output v we get the predicted clean latent x0 = z_t - s * v. For each face we decode
   a small latent crop around it with the VAE, embed it with a face recognition network, and take the
   gradient of (1 - cosine to the reference face) with respect to x0. The next latent is moved against
   that gradient. The 20B transformer is not part of the gradient, so the extra memory is small.
   The guidance network (FaceNet, VGGFace2) is a different model from the one used for scoring
   (ArcFace, InsightFace), so the score cannot be raised by fitting the scorer.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ----------------------------------------------------------------------------------------------------
# geometry helpers
# ----------------------------------------------------------------------------------------------------
def output_size(pipe, last_image: Image.Image) -> tuple[int, int]:
    """(width, height) the pipeline will generate for this last input image."""
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import calculate_dimensions

    w, h = calculate_dimensions(1024 * 1024, last_image.width / last_image.height)
    m = pipe.vae_scale_factor * 2
    return w // m * m, h // m * m


def square_box(bbox, scale: float, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    cx, cy, s = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * scale / 2
    return int(max(0, cx - s)), int(max(0, cy - s)), int(min(w, cx + s)), int(min(h, cy + s))


def soft_mask(mask: np.ndarray, dilate_px: int, feather_px: float) -> np.ndarray:
    """mask: HxW, 1 = generate freely. Grow it so the model has room to blend, then soften the edge."""
    m = (mask > 0.5).astype(np.uint8)
    if dilate_px > 0:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2))
    m = m.astype(np.float32)
    if feather_px > 0:
        m = cv2.GaussianBlur(m, (0, 0), feather_px)
    return np.clip(m, 0, 1)


# ----------------------------------------------------------------------------------------------------
# face embedding network used for guidance
# ----------------------------------------------------------------------------------------------------
@lru_cache(maxsize=1)
def guide_net():
    from facenet_pytorch import InceptionResnetV1

    net = InceptionResnetV1(pretrained="vggface2").eval().to("cuda")
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _embed(x: torch.Tensor) -> torch.Tensor:
    """x: (N, 3, H, W) in [-1, 1], already cropped to the face box."""
    x = F.interpolate(x.float(), size=(160, 160), mode="bilinear", align_corners=False)
    return F.normalize(guide_net()(x), dim=-1)


def face_crop_tensor(img: Image.Image, bbox, scale: float = 1.2) -> torch.Tensor:
    x0, y0, x1, y1 = square_box(bbox, scale, img.width, img.height)
    a = np.asarray(img.convert("RGB").crop((x0, y0, x1, y1)).resize((160, 160), Image.BILINEAR), np.float32)
    return torch.from_numpy(a / 127.5 - 1).permute(2, 0, 1)[None].cuda()


@torch.no_grad()
def reference_embedding(img: Image.Image, bbox) -> torch.Tensor:
    return _embed(face_crop_tensor(img, bbox))[0]


def guide_similarity(img: Image.Image, bbox, ref: torch.Tensor) -> float:
    """FaceNet cosine of a face in a finished image to its reference (for logging)."""
    with torch.no_grad():
        return float(_embed(face_crop_tensor(img, bbox))[0] @ ref)


# ----------------------------------------------------------------------------------------------------
# controls
# ----------------------------------------------------------------------------------------------------
@dataclass
class Lock:
    target: Image.Image            # picture the free-of-people area should keep
    mask: np.ndarray               # HxW float in [0, 1] at output size, 1 = generate freely
    release: float = 0.85          # fraction of steps that are locked
    inner: Image.Image | None = None  # picture for the people area in the first steps (fixes their size and place)
    inner_release: float = 0.0     # fraction of steps where the whole picture is locked to target + inner


@dataclass
class Guide:
    boxes: list                    # face boxes (x0, y0, x1, y1) in output pixels
    refs: list                     # reference embeddings (torch, 512), same order as boxes
    scale: float = 0.02            # step size, in units of the latent's typical value
    clip: float = 2.0              # clip the normalised gradient per element (unclipped steps pile up around
                                   # the eyes and destroy the face; see experiments/guide_sanity.py)
    sigma_range: tuple = (0.4, 0.85)  # middle noise levels only: later model steps clean up what guidance adds
    blur: float = 1.5              # Gaussian blur of the gradient, in latent cells (removes high-frequency
                                   # adversarial patterns that fool the face network but are not a face)
    n_aug: int = 4                 # jittered / flipped face crops averaged per face
    crop_scale: float = 1.2        # face crop used for the embedding (same as the reference crop)
    margin: float = 1.8            # latent crop decoded around the face (VAE context)
    max_latent: int = 56           # largest latent crop side decoded with gradients (memory limit)
    log: list = field(default_factory=list)


@torch.no_grad()
def _encode(pipe, img: Image.Image, w: int, h: int) -> torch.Tensor:
    t = pipe.image_processor.preprocess(img.convert("RGB"), h, w).unsqueeze(2)
    t = t.to(device=pipe.vae.device, dtype=pipe.vae.dtype)
    lat = pipe._encode_vae_image(t, None)                     # (1, C, 1, h/8, w/8), normalised
    return pipe._pack_latents(lat, 1, lat.shape[1], h // pipe.vae_scale_factor, w // pipe.vae_scale_factor)


def _mask_tokens(mask: np.ndarray, w: int, h: int) -> torch.Tensor:
    m = cv2.resize(mask.astype(np.float32), (w // 16, h // 16), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(m.reshape(1, -1, 1)).cuda()


def _vae_stats(pipe, dtype):
    c = pipe.vae.config.z_dim
    mean = torch.tensor(pipe.vae.config.latents_mean).view(1, c, 1, 1, 1).to("cuda", dtype)
    std = torch.tensor(pipe.vae.config.latents_std).view(1, c, 1, 1, 1).to("cuda", dtype)
    return mean, std


def _guide_grad(pipe, x0_packed: torch.Tensor, g: Guide, w: int, h: int):
    """Gradient of the identity loss w.r.t. the packed x0 latent, and the per-face cosine."""
    vsf = pipe.vae_scale_factor
    lh, lw = h // vsf, w // vsf
    mean, std = _vae_stats(pipe, pipe.vae.dtype)
    with torch.enable_grad():
        x0 = x0_packed.detach().float().requires_grad_(True)
        lat = pipe._unpack_latents(x0, h, w, vsf)              # (1, C, 1, lh, lw)
        loss, cos = 0.0, []
        for box, ref in zip(g.boxes, g.refs):
            fx0, fy0, fx1, fy1 = square_box(box, g.crop_scale, w, h)
            side = max(box[2] - box[0], box[3] - box[1])
            margin = min(g.margin, max(g.crop_scale, g.max_latent * vsf / max(side, 1)))
            cx0, cy0, cx1, cy1 = square_box(box, margin, w, h)
            # latent crop, at least 16 latents wide so the VAE has context
            a0, b0 = cx0 // vsf, cy0 // vsf
            a1, b1 = max(a0 + 16, math.ceil(cx1 / vsf)), max(b0 + 16, math.ceil(cy1 / vsf))
            a1, b1 = min(a1, lw, a0 + g.max_latent), min(b1, lh, b0 + g.max_latent)
            z = lat[:, :, :, b0:b1, a0:a1].to(pipe.vae.dtype) * std + mean
            img = pipe.vae.decode(z, return_dict=False)[0][:, :, 0]          # (1, 3, H', W') in [-1, 1]
            ox, oy = a0 * vsf, b0 * vsf
            H, W = img.shape[-2:]
            gen = torch.Generator().manual_seed(len(g.log) * 131 + len(cos))
            crops = []
            for k in range(max(1, g.n_aug)):
                # random shift of up to 6% and scale of up to 8% of the face crop, every second one flipped
                j = (torch.rand(3, generator=gen) * 2 - 1).tolist() if k else [0.0, 0.0, 0.0]
                side = (fx1 - fx0) * (1 + 0.08 * j[2])
                cx, cy = (fx0 + fx1) / 2 + 0.06 * side * j[0] - ox, (fy0 + fy1) / 2 + 0.06 * side * j[1] - oy
                x0_, y0_ = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
                x1_, y1_ = int(min(W, cx + side / 2)), int(min(H, cy + side / 2))
                f = F.interpolate(img[:, :, y0_:y1_, x0_:x1_].float(), size=(160, 160), mode="bilinear",
                                  align_corners=False)
                crops.append(f.flip(-1) if k % 2 else f)
            e = _embed(torch.cat(crops).clamp(-1, 1))
            c = (e @ ref).mean()
            cos.append(float(e[0] @ ref))
            loss = loss + (1 - c)
        grad = torch.autograd.grad(loss, x0)[0]
    if g.blur > 0:
        grad = _blur_packed(pipe, grad, w, h, g.blur)
    return grad, cos


def _blur_packed(pipe, packed: torch.Tensor, w: int, h: int, sigma: float) -> torch.Tensor:
    vsf = pipe.vae_scale_factor
    lat = pipe._unpack_latents(packed, h, w, vsf)[:, :, 0]               # (1, C, lh, lw)
    r = max(1, int(3 * sigma))
    x = torch.arange(-r, r + 1, device=lat.device, dtype=lat.dtype)
    k = torch.exp(-x ** 2 / (2 * sigma ** 2))
    k = k / k.sum()
    C = lat.shape[1]
    lat = F.conv2d(F.pad(lat, (r, r, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).repeat(C, 1, 1, 1), groups=C)
    lat = F.conv2d(F.pad(lat, (0, 0, r, r), mode="replicate"), k.view(1, 1, -1, 1).repeat(C, 1, 1, 1), groups=C)
    return pipe._pack_latents(lat.unsqueeze(2), 1, C, lat.shape[-2], lat.shape[-1])


@contextmanager
def controls(pipe, last_image: Image.Image, steps: int, lock: Lock | None = None, guide: Guide | None = None):
    """Patch pipe.scheduler.step for one pipeline call."""
    if lock is None and guide is None:
        yield
        return
    w, h = output_size(pipe, last_image)
    sched = pipe.scheduler
    orig_step = sched.step
    z_bg = m_tok = None
    if lock is not None:
        mask = lock.mask
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        z_bg = _encode(pipe, lock.target, w, h).float()
        m_tok = _mask_tokens(mask, w, h)
        z_in = _encode(pipe, lock.inner, w, h).float() if lock.inner is not None and lock.inner_release > 0 else None
    state = {"i": 0, "noise": None}

    def step(model_output, timestep, sample, *args, **kwargs):
        i = state["i"]
        sig = sched.sigmas.to(sample.device)
        s, s_next = float(sig[i]), float(sig[i + 1])
        if state["noise"] is None:
            state["noise"] = sample.detach().float().clone()    # s_0 = 1, so z_0 is the pure starting noise
        out = orig_step(model_output, timestep, sample, *args, **kwargs)
        prev = out[0] if isinstance(out, tuple) else out.prev_sample
        new = prev.float()

        if guide is not None and guide.sigma_range[0] <= s <= guide.sigma_range[1]:
            x0 = sample.float() - s * model_output.float()
            grad, cos = _guide_grad(pipe, x0, guide, w, h)
            nz = grad.abs() > 0
            if nz.any():
                d = grad / (grad[nz].pow(2).mean().sqrt() + 1e-12)
                if guide.clip:
                    d = d.clamp(-guide.clip, guide.clip)
                # moving x0 by -d moves z_next by -(1 - s_next) * d (Euler step of flow matching)
                new = new - (1 - s_next) * guide.scale * d
            guide.log.append({"step": i, "sigma": round(s, 3), "cos": [round(c, 3) for c in cos]})

        if lock is not None and z_in is not None and i / steps < lock.inner_release:
            # whole picture: people from `inner`, background from `target` (a collage), at this noise level
            col = m_tok * z_in + (1 - m_tok) * z_bg
            new = (1 - s_next) * col + s_next * state["noise"]
        elif lock is not None and i / steps < lock.release:
            ref = (1 - s_next) * z_bg + s_next * state["noise"]
            new = m_tok * new + (1 - m_tok) * ref

        state["i"] = i + 1
        new = new.to(prev.dtype)
        return (new,) if isinstance(out, tuple) else type(out)(prev_sample=new)

    sched.step = step
    try:
        yield
    finally:
        sched.step = orig_step
