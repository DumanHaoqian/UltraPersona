"""Quick check that a quantization preset produces a clean image (no noise, no NaN / black output).

Quantizes on the fly (ignores saved weights), runs one single-person case at a reduced step count.
  python -m experiments.ablate_quant --preset nf4 --steps 20
"""
import argparse
import json
import os

import numpy as np
from PIL import Image

from persona2 import prompts
from persona2.analyze import analyze
from persona2.face_metrics import compare
from persona2.pipeline import ROOT, load_config
from persona2.qwen_edit import edit, load_pipeline, peak_vram, set_vram_cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--negative", default="long", choices=["long", "space"])
    args = ap.parse_args()
    cfg = load_config()
    mc = cfg["model"]
    set_vram_cap(mc["vram_cap_gb"])
    pipe = load_pipeline(mc["path"], args.preset, mc.get("quantized_dir"), use_saved=False)
    d = os.path.join(ROOT, "testcases", "selected", "people_1")
    photo = Image.open(os.path.join(d, sorted(os.listdir(d))[0])).convert("RGB")
    an = analyze(photo, cfg)
    bg = cfg["backgrounds"][1]
    bg_img = Image.open(os.path.join(ROOT, bg["file"])).convert("RGB")
    neg = prompts.NEGATIVE if args.negative == "long" else " "
    with peak_vram() as pv:
        img = edit(pipe, [photo, bg_img], prompts.compose_prompt(an.people, an.framing, bg), neg, 0, args.steps,
                   mc["true_cfg_scale"])
    arr = np.asarray(img).astype(np.float32)
    # high-frequency energy: noisy images have a large mean absolute Laplacian
    import cv2
    lap = float(np.abs(cv2.Laplacian(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY), cv2.CV_32F)).mean())
    out_dir = os.path.join(ROOT, "experiments", "results", "ablate_quant")
    os.makedirs(out_dir, exist_ok=True)
    img.save(os.path.join(out_dir, f"{args.preset}_s{args.steps}_{args.negative}.png"))
    m = compare(photo, img)
    row = {"preset": args.preset, "steps": args.steps, "negative": args.negative, "mean_pixel": float(arr.mean()),
           "laplacian": lap, "face": m.as_dict(), **pv}
    print(json.dumps(row))
    with open(os.path.join(out_dir, "ablate.jsonl"), "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
