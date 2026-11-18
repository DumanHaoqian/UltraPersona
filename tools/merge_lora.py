"""Merge a Lightning LoRA into the BF16 transformer weights, shard by shard.

The merged transformer is then quantized like the original (NF4 / FP8), so inference has no LoRA
overhead and the LoRA is not applied on top of 4-bit weights. Peak RAM is about one shard (~10GB).

  python tools/merge_lora.py --base /mnt/d/hf_models/Qwen-Image-Edit-2511 \
      --lora /mnt/d/hf_models/Qwen-Image-Edit-2511-Lightning/Qwen-Image-Edit-2511-Lightning-8steps-V1.0-bf16.safetensors \
      --out /mnt/d/hf_models/Qwen-Image-Edit-2511-L8
Writes <out>/transformer/ (config, index, shards). The pipeline still takes the text encoder, VAE and
processor from the base folder.
"""
import argparse
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def lora_table(path):
    """{target weight name: (up, down, scale)}; LoRA key 'blocks.0.attn.to_q.lora_up.weight' targets
    'blocks.0.attn.to_q.weight'."""
    f = safe_open(path, "pt")
    keys = set(f.keys())
    out = {}
    for k in keys:
        if not k.endswith(".lora_up.weight"):
            continue
        base = k[: -len(".lora_up.weight")]
        up, down = f.get_tensor(k), f.get_tensor(base + ".lora_down.weight")
        rank = down.shape[0]
        alpha = float(f.get_tensor(base + ".alpha")) if base + ".alpha" in keys else rank
        out[base + ".weight"] = (up, down, alpha / rank)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--strength", type=float, default=1.0)
    args = ap.parse_args()

    src = os.path.join(args.base, "transformer")
    dst = os.path.join(args.out, "transformer")
    os.makedirs(dst, exist_ok=True)
    lora = lora_table(args.lora)
    scales = {round(s, 4) for _, _, s in lora.values()}
    print(f"{len(lora)} LoRA targets, alpha/rank = {sorted(scales)}")
    index = json.load(open(os.path.join(src, "diffusion_pytorch_model.safetensors.index.json")))
    merged = set()
    for shard in sorted(set(index["weight_map"].values())):
        f = safe_open(os.path.join(src, shard), "pt")
        tensors = {}
        for k in f.keys():
            w = f.get_tensor(k)
            if k in lora:
                up, down, s = lora[k]
                delta = (up.float().cuda() @ down.float().cuda()) * (s * args.strength)
                w = (w.float().cuda() + delta).to(w.dtype).cpu()
                merged.add(k)
            tensors[k] = w.contiguous()
        save_file(tensors, os.path.join(dst, shard), metadata={"format": "pt"})
        print(shard, len(tensors), "tensors", flush=True)
        del tensors
    missing = set(lora) - merged
    if missing:
        raise SystemExit(f"{len(missing)} LoRA targets not found in the base weights, e.g. {sorted(missing)[:3]}")
    shutil.copy(os.path.join(src, "config.json"), dst)
    shutil.copy(os.path.join(src, "diffusion_pytorch_model.safetensors.index.json"), dst)
    json.dump({"base": args.base, "lora": args.lora, "strength": args.strength, "targets": len(merged)},
              open(os.path.join(args.out, "merge_info.json"), "w"), indent=1)
    print(f"merged {len(merged)} weights into {dst}")


if __name__ == "__main__":
    main()
