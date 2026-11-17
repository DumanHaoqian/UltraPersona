"""One step (compose + style in one edit) vs two steps (compose, then stylize): which keeps faces better?

Both methods use the same preset, seeds, steps and photos. The two-step composite of a
(photo, background, seed) is made once and reused for every style.

  python -m experiments.one_vs_two --per-bucket 2 --bgs 3 5 --styles 1 3 4 5
Writes experiments/results/one_vs_two/<run>/images/* and metrics.jsonl (resumable).
Style strength is scored afterwards by experiments/score_style.py.
"""
import argparse
import json
import os

from PIL import Image

from persona2 import prompts
from persona2.analyze import analyze
from persona2.face_metrics import compare
from persona2.pipeline import ROOT, load_config
from persona2.qwen_edit import edit, load_pipeline, peak_vram, set_vram_cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v1")
    ap.add_argument("--preset", default=None)
    ap.add_argument("--per-bucket", type=int, default=2)
    ap.add_argument("--buckets", nargs="+", default=["people_1", "people_2", "people_3", "people_4", "people_5"])
    ap.add_argument("--bgs", nargs="+", type=int, default=[3, 5])
    ap.add_argument("--styles", nargs="+", type=int, default=[1, 3, 4, 5])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = ap.parse_args()

    cfg = load_config()
    mc = cfg["model"]
    set_vram_cap(mc["vram_cap_gb"])
    preset = args.preset or mc["default_preset"]
    out = os.path.join(ROOT, "experiments", "results", "one_vs_two", args.run)
    img_dir = os.path.join(out, "images")
    os.makedirs(img_dir, exist_ok=True)
    log_path = os.path.join(out, "metrics.jsonl")
    done = set()
    if os.path.exists(log_path):
        done = {json.loads(l)["key"] for l in open(log_path)}
    log = open(log_path, "a")
    pipe = load_pipeline(mc["path"], preset, mc.get("quantized_dir"))

    def run(images, prompt, seed):
        with peak_vram() as pv:
            img = edit(pipe, images, prompt, prompts.NEGATIVE, seed, mc["steps"], mc["true_cfg_scale"])
        return img, pv

    for bucket in args.buckets:
        d = os.path.join(ROOT, "testcases", "selected", bucket)
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d))[: args.per_bucket]:
            photo = Image.open(os.path.join(d, fname)).convert("RGB")
            an = analyze(photo, cfg)
            pid = os.path.splitext(fname)[0][:40]
            for bg_id in args.bgs:
                bg = cfg["backgrounds"][bg_id]
                if an.people > bg["max_people"]:
                    continue
                bg_img = Image.open(os.path.join(ROOT, bg["file"])).convert("RGB")
                for seed in args.seeds:
                    base = f"{bucket}__{pid}__bg{bg_id}__s{seed}"
                    comp_path = os.path.join(img_dir, base + "__composite.png")
                    if os.path.exists(comp_path):
                        comp = Image.open(comp_path).convert("RGB")
                    else:
                        p = prompts.compose_prompt(an.people, an.framing, bg)
                        comp, pv = run([photo, bg_img], p, seed)
                        comp.save(comp_path)
                        m = compare(photo, comp)
                        log.write(json.dumps({"key": base + "__composite", "bucket": bucket, "photo": fname,
                                              "people": an.people, "bg": bg_id, "seed": seed, "style": 0,
                                              "method": "composite", "prompt": p, "face": m.as_dict(), **pv}) + "\n")
                        log.flush()
                    for st in args.styles:
                        style = cfg["styles"][st]
                        for method in ("onestep", "twostep"):
                            key = f"{base}__st{st}__{method}"
                            if key in done:
                                continue
                            if method == "onestep":
                                p = prompts.onestep_prompt(an.people, an.framing, bg, style)
                                img, pv = run([photo, bg_img], p, seed)
                            else:
                                p = prompts.stylize_prompt(an.people, style)
                                img, pv = run([comp], p, seed)
                            img.save(os.path.join(img_dir, key + ".png"))
                            m = compare(photo, img)
                            row = {"key": key, "bucket": bucket, "photo": fname, "people": an.people, "bg": bg_id,
                                   "seed": seed, "style": st, "method": method, "prompt": p,
                                   "face": m.as_dict(), **pv}
                            log.write(json.dumps(row) + "\n")
                            log.flush()
                            print(key, f"worst={m.worst:.3f} count_ok={m.count_ok} t={pv['seconds']:.0f}s")


if __name__ == "__main__":
    main()
