"""Quantize Qwen-Image-Edit-2511 once and save the result, so deployment loads the small weights directly.

  python -m persona2.quantize --preset nf4
  python -m persona2.quantize --preset fp8

bitsandbytes NF4 saves as safetensors; torchao FP8 tensors need safe_serialization=False.
"""
import argparse
import os
import time

import torch
import yaml

from .qwen_edit import PRESETS, load_text_encoder, load_transformer


def _size_gb(p):
    return sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p)) / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), required=True)
    ap.add_argument("--config", default="persona2/config.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))["model"]
    tr_mode, te_mode = PRESETS[args.preset]
    out = os.path.join(cfg["quantized_dir"], args.preset)
    os.makedirs(out, exist_ok=True)

    t0 = time.time()
    tr = load_transformer(cfg["path"], tr_mode)
    print(f"transformer ({tr_mode}) quantized in {time.time() - t0:.0f}s, GPU {torch.cuda.memory_allocated() / 2**30:.1f} GB")
    tr.save_pretrained(os.path.join(out, "transformer"), safe_serialization=tr_mode.startswith("nf4"))
    del tr
    torch.cuda.empty_cache()

    te = load_text_encoder(cfg["path"], te_mode)
    te.save_pretrained(os.path.join(out, "text_encoder"), safe_serialization=te_mode == "nf4")
    del te
    for d in ("transformer", "text_encoder"):
        p = os.path.join(out, d)
        print(f"{d}: {_size_gb(p):.1f} GB -> {p}")


if __name__ == "__main__":
    main()
