"""Style strength: how much more an image matches its style description than "a realistic photograph".

Without this, a method could keep faces better only because it stylizes less.
  python -m experiments.score_style --run v1
Adds a "style_score" field to experiments/results/one_vs_two/<run>/style_scores.jsonl
"""
import argparse
import json
import os

import open_clip
import torch
from PIL import Image

from persona2.pipeline import ROOT, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v1")
    ap.add_argument("--dir", default=None, help="results folder with metrics.jsonl and images/ (overrides --run)")
    args = ap.parse_args()
    cfg = load_config()
    d = args.dir or os.path.join(ROOT, "experiments", "results", "one_vs_two", args.run)
    rows = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl"))]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, pre = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai", device=dev)
    tok = open_clip.get_tokenizer("ViT-L-14")
    texts = {k: v["prompt"] or "a photograph" for k, v in cfg["styles"].items()}
    with torch.no_grad():
        t = model.encode_text(tok(["a realistic photograph"] + [f"an image in the style of {texts[k]}" for k in texts]).to(dev))
        t = t / t.norm(dim=-1, keepdim=True)
    idx = {k: i + 1 for i, k in enumerate(texts)}
    out = open(os.path.join(d, "style_scores.jsonl"), "w")
    for r in rows:
        if r["method"] == "composite":
            continue
        img = pre(Image.open(os.path.join(d, "images", r["key"] + ".png")).convert("RGB")).unsqueeze(0).to(dev)
        with torch.no_grad():
            e = model.encode_image(img)
            e = e / e.norm(dim=-1, keepdim=True)
            s = (e @ t.T)[0]
        out.write(json.dumps({"key": r["key"], "style_score": float(s[idx[r["style"]]] - s[0])}) + "\n")
    out.close()


if __name__ == "__main__":
    main()
