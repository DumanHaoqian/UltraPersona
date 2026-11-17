"""Compare quantization presets on the RTX 5090: load time, peak VRAM, seconds per image, face similarity.

Run one preset per process so memory numbers do not mix:
  python -m experiments.bench_quant --preset nf4
  python -m experiments.bench_quant --preset fp8
Results append to experiments/results/bench_quant.jsonl, images to experiments/results/bench_quant/<preset>/.
"""
import argparse
import json
import os
import subprocess
import time

import torch
from PIL import Image

from persona2 import prompts
from persona2.analyze import analyze
from persona2.face_metrics import compare
from persona2.pipeline import ROOT, load_config
from persona2.qwen_edit import edit, load_pipeline, peak_vram, set_vram_cap

CASES = [  # (photo, background id); photos come from testcases/
    ("people_1", 1), ("people_1", 5), ("people_2", 3), ("people_3", 6), ("people_5", 3),
]


def gpu_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.strip().splitlines()[0])


def pick(bucket, k=0):
    d = os.path.join(ROOT, "testcases", "selected", bucket)
    files = sorted(os.listdir(d))
    return os.path.join(d, files[k % len(files)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", required=True)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lora", action="store_true", help="load the 8-step Lightning LoRA")
    ap.add_argument("--n", type=int, default=len(CASES), help="only run the first N cases")
    ap.add_argument("--tag", default="", help="suffix for the output folder")
    args = ap.parse_args()
    cfg = load_config()
    mc = cfg["model"]
    set_vram_cap(mc["vram_cap_gb"])
    steps = args.steps or (8 if args.lora else mc["steps"])
    tag = args.preset + ("_lightning" if args.lora else "") + args.tag
    outdir = os.path.join(ROOT, "experiments", "results", "bench_quant", tag)
    os.makedirs(outdir, exist_ok=True)

    t0 = time.time()
    lora = None
    if args.lora:
        lora_dir = mc["lightning_lora"]
        cands = sorted(f for f in os.listdir(lora_dir) if "8steps" in f and "bf16" in f and f.endswith(".safetensors"))
        lora = os.path.join(lora_dir, cands[0])
    pipe = load_pipeline(mc["path"], args.preset, mc.get("quantized_dir"), lora)
    load_s = time.time() - t0
    idle_mib = gpu_used_mib()

    rows = []
    for i, (bucket, bg_id) in enumerate(CASES[: args.n]):
        photo = Image.open(pick(bucket)).convert("RGB")
        an = analyze(photo, cfg)
        bg = cfg["backgrounds"][bg_id]
        bg_img = Image.open(os.path.join(ROOT, bg["file"])).convert("RGB")
        p = prompts.compose_prompt(max(an.people, 1), an.framing, bg)
        with peak_vram() as pv:
            img = edit(pipe, [photo, bg_img], p, prompts.NEGATIVE, seed=0, steps=steps,
                       true_cfg_scale=1.0 if args.lora else mc["true_cfg_scale"])
        smi = gpu_used_mib()
        img.save(os.path.join(outdir, f"{i}_{bucket}_bg{bg_id}.png"))
        m = compare(photo, img)
        row = {"preset": tag, "case": i, "bucket": bucket, "bg": bg_id, "steps": steps, **pv,
               "nvidia_smi_mib_after": smi, "face": m.as_dict(), "out_size": img.size}
        rows.append(row)
        print(json.dumps(row))
    summary = {"preset": tag, "summary": True, "load_seconds": load_s, "idle_nvidia_smi_mib": idle_mib,
               "max_peak_reserved_gb": max(r["peak_reserved_gb"] for r in rows),
               "mean_seconds_per_image": sum(r["seconds"] for r in rows[1:]) / max(1, len(rows) - 1),
               "mean_worst_face_sim": sum(r["face"]["worst"] for r in rows) / len(rows),
               "torch": torch.__version__}
    print(json.dumps(summary))
    with open(os.path.join(ROOT, "experiments", "results", "bench_quant.jsonl"), "a") as f:
        for r in rows + [summary]:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
