"""Identity guidance without the diffusion model: does a gradient step on the clean latent raise face
similarity without damaging the face?

Encode a finished picture with the VAE, take N steps of the same gradient used during generation
(persona2.control._guide_grad), decode, and score with FaceNet (the guided network) and ArcFace (independent).
  python -m experiments.guide_sanity --image <png> --photo <input photo> --steps 10 --lr 0.05 --clip 2
"""
import argparse
import json
import os

import torch
from PIL import Image

from persona2.control import Guide, _guide_grad, output_size, reference_embedding
from persona2.face_metrics import compare
from persona2.pipeline import ROOT, load_config, matched_faces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--photo", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--lr", nargs="+", type=float, default=[0.02, 0.05])
    ap.add_argument("--clip", nargs="+", type=float, default=[0, 2])
    ap.add_argument("--blur", type=float, default=1.5)
    ap.add_argument("--out", default=os.path.join(ROOT, "experiments", "results", "guide_sanity"))
    args = ap.parse_args()

    from diffusers import AutoencoderKLQwenImage, QwenImageEditPlusPipeline

    cfg = load_config()
    vae = AutoencoderKLQwenImage.from_pretrained(cfg["model"]["path"], subfolder="vae", torch_dtype=torch.bfloat16).cuda()
    pipe = QwenImageEditPlusPipeline(scheduler=None, vae=vae, text_encoder=None, tokenizer=None, processor=None,
                                     transformer=None)
    img = Image.open(args.image).convert("RGB")
    photo = Image.open(args.photo).convert("RGB")
    w, h = output_size(pipe, img)
    img = img.resize((w, h), Image.LANCZOS)
    pairs = matched_faces(photo, img)
    boxes = [fo.bbox for _, fo in pairs]
    refs = [reference_embedding(photo, fi.bbox) for fi, _ in pairs]
    from persona2.control import _encode

    x_start = _encode(pipe, img, w, h).float()
    os.makedirs(args.out, exist_ok=True)
    rows = []

    def decode(x):
        mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).cuda()
        std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).cuda()
        lat = pipe._unpack_latents(x, h, w, 8) * std + mean
        with torch.no_grad():
            t = vae.decode(lat.to(vae.dtype), return_dict=False)[0][:, :, 0]
        return pipe.image_processor.postprocess(t, output_type="pil")[0]

    base = decode(x_start)
    m0 = compare(photo, base)
    rows.append({"lr": 0, "clip": 0, "step": 0, "arcface": m0.per_person})
    print("start arcface", [round(v, 3) for v in m0.per_person])
    for lr in args.lr:
        for clip in args.clip:
            g = Guide(boxes, refs, scale=lr, blur=args.blur)
            x = x_start.clone()
            for s in range(1, args.steps + 1):
                grad, cos = _guide_grad(pipe, x, g, w, h)
                nz = grad.abs() > 0
                d = grad / (grad[nz].pow(2).mean().sqrt() + 1e-12)
                if clip:
                    d = d.clamp(-clip, clip)
                x = x - lr * d
                g.log.append(1)
                if s in (1, args.steps // 2, args.steps):
                    out = decode(x)
                    m = compare(photo, out)
                    rows.append({"lr": lr, "clip": clip, "step": s, "facenet": cos, "arcface": m.per_person})
                    print(f"lr={lr} clip={clip} step={s} facenet={[round(c, 3) for c in cos]} "
                          f"arcface={[round(v, 3) for v in m.per_person]}", flush=True)
            out.save(os.path.join(args.out, f"lr{lr}_clip{clip}.png"))
    base.save(os.path.join(args.out, "start.png"))
    json.dump(rows, open(os.path.join(args.out, "log.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
