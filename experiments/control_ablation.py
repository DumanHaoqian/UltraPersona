"""Ablation of the three controls on top of the v1 one-step results.

Each variant is a spec "<kind>[:key=value,...]":
  base                        v1 output (one-step for art styles, the composite for the photo style)
  lock:r=0.25                 background lock (approach 1), locked for the first 25% of steps
  guide:g=0.2                 identity guidance only (approach 5), step size 0.2
  lock_guide:r=0.25,g=0.2     both
  repair                      face repair at a larger scale (approach 3) on the base output
  repair:g=0+src=lock_guide:r=0.25,g=0.2   face repair on another variant's output
Without r, the release comes from config (photo / style). The people mask and face positions come from the
base output, which was made with the same prompt and no control (see Persona._run_controlled).

  python -m experiments.control_ablation --run c1 --styles 0 1 4 5 --variants base "lock:r=0.25" ...
Writes experiments/results/control/<run>/images/* and metrics.jsonl (resumable).
Style strength afterwards: python -m experiments.score_style --dir experiments/results/control/<run>
"""
import argparse
import json
import os

from PIL import Image

from persona2 import prompts
from persona2.analyze import analyze
from persona2.bg_metrics import background_scores
from persona2.control import Lock
from persona2.face_metrics import compare
from persona2.matting import person_mask
from persona2.pipeline import ROOT, Persona, load_config
from persona2.qwen_edit import peak_vram, set_vram_cap

V1 = os.path.join(ROOT, "experiments", "results", "one_vs_two", "v1", "images")


def parse(spec: str):
    """lock:r=0.3,t=styled   guide:g=0.05   repair:g=0+src=lock:r=0.3,t=styled"""
    kind, _, rest = spec.partition(":")
    sep = "+" if kind == "repair" else ","   # "+" survives the Windows -> WSL command line; ";" does not
    opts = {}
    for kv in (rest.split(sep) if rest else []):
        k, _, v = kv.partition("=")
        try:
            opts[k] = float(v)
        except ValueError:
            opts[k] = v
    if kind == "repair":
        opts.setdefault("src", "base")
    return kind, opts


def tag(spec: str) -> str:
    return spec.replace(":", "-").replace("=", "").replace(",", "_").replace("+", "_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="c1")
    ap.add_argument("--per-bucket", type=int, default=1)
    ap.add_argument("--buckets", nargs="+", default=["people_1", "people_2", "people_3", "people_4", "people_5"])
    ap.add_argument("--styles", nargs="+", type=int, default=[0, 1, 4, 5])
    ap.add_argument("--variants", nargs="+", default=["base", "lock", "guide", "lock_guide", "repair"])
    ap.add_argument("--bg", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lightning", default=None, help="L8 / L4: fast weights; base is then generated, not v1")
    ap.add_argument("--repair-thr", type=float, default=None, help="repair faces below this ArcFace score")
    args = ap.parse_args()

    cfg = load_config()
    cfg["model"]["lightning"] = args.lightning
    c = cfg.setdefault("control", {})
    set_vram_cap(cfg["model"]["vram_cap_gb"])
    out = os.path.join(ROOT, "experiments", "results", "control", args.run)
    img_dir = os.path.join(out, "images")
    os.makedirs(img_dir, exist_ok=True)
    log_path = os.path.join(out, "metrics.jsonl")
    done = {json.loads(l)["key"] for l in open(log_path, encoding="utf-8")} if os.path.exists(log_path) else set()
    log = open(log_path, "a", encoding="utf-8")
    json.dump({"control": c, "argv": vars(args)}, open(os.path.join(out, "settings.json"), "w"), indent=1)

    per = Persona(cfg)
    c.update(lock=True, id_guide=True)          # _controls plans both; each variant picks what it uses
    bg = cfg["backgrounds"][args.bg]
    bg_img = Image.open(os.path.join(ROOT, bg["file"])).convert("RGB")
    thr = args.repair_thr if args.repair_thr is not None else cfg["quality"]["face_sim_min"]

    for bucket in args.buckets:
        d = os.path.join(ROOT, "testcases", "selected", bucket)
        for fname in sorted(os.listdir(d))[: args.per_bucket]:
            photo = Image.open(os.path.join(d, fname)).convert("RGB")
            an = analyze(photo, cfg)
            pid = os.path.splitext(fname)[0][:40]
            base = f"{bucket}__{pid}__bg{args.bg}__s{args.seed}"
            for st in args.styles:
                style = cfg["styles"][st]
                if style["engine"] == "qwen":
                    p = prompts.onestep_prompt(an.people, an.framing, bg, style)
                    v1_path = os.path.join(V1, f"{base}__st{st}__onestep.png")
                else:
                    p = prompts.compose_prompt(an.people, an.framing, bg)
                    v1_path = os.path.join(V1, base + "__composite.png")
                base_seconds = None
                if args.lightning:   # free edit with the fast weights, same prompt and seed as v1
                    bpath = os.path.join(img_dir, f"{base}__st{st}__base.png")
                    if os.path.exists(bpath):
                        base_img = Image.open(bpath).convert("RGB")
                    else:
                        with peak_vram() as bpv:
                            base_img = per._edit([photo, bg_img], p, args.seed)
                        base_seconds = bpv
                        base_img.save(bpath)
                else:
                    base_img = Image.open(v1_path).convert("RGB")
                made = {"base": base_img}
                plan = None
                for spec in args.variants:
                    kind, o = parse(spec)
                    key = f"{base}__st{st}__{tag(spec)}"
                    path = os.path.join(img_dir, key + ".png")
                    if key in done:
                        made[spec] = Image.open(path).convert("RGB")
                        continue
                    if plan is None and kind != "base":
                        plan = per._controls(photo, base_img, bg_img, style)
                    extra = {"spec": spec}
                    with peak_vram() as pv:
                        if kind == "base":
                            img = base_img
                        elif kind in ("lock", "guide", "lock_guide"):
                            lock = guide = None
                            if "lock" in kind and plan[0] is not None:
                                target = plan[0].target
                                if o.get("t") == "styled" and style["engine"] == "qwen":
                                    target = per.styled_background(args.bg, st, args.seed)
                                # r applies to art styles; the photo style keeps its own (late) release unless rp is given
                                r = o.get("r", plan[0].release) if style["engine"] == "qwen" else o.get("rp", plan[0].release)
                                lock = Lock(target, plan[0].mask, r, inner=base_img, inner_release=o.get("ri", 0.0))
                            if "guide" in kind and plan[1] is not None:
                                guide = per.make_guide(plan[1].boxes, plan[1].refs)
                                guide.scale = o.get("g", guide.scale)
                            img = per._edit([photo, bg_img], p, args.seed, lock, guide)
                            if guide:
                                extra["guide_log"] = guide.log
                        elif kind == "repair":
                            src = made.get(o["src"])
                            if src is None:
                                print("skip", spec, "(source not made)")
                                continue
                            rlog = []
                            saved = per.cfg["control"].get("repair_guide", True)
                            per.cfg["control"]["repair_guide"] = bool(o.get("g", 1))
                            try:
                                img = per.repair_faces(photo, src, style, args.seed, thr=thr, log=rlog)
                            finally:
                                per.cfg["control"]["repair_guide"] = saved
                            extra["repaired"] = rlog
                        else:
                            raise ValueError(spec)
                    img.save(path)
                    made[spec] = img
                    m = compare(photo, img)
                    pm = person_mask(img)
                    b = background_scores(img, bg_img, pm)
                    if style["engine"] == "qwen":   # layout kept relative to the styled background
                        extra["bgm_styled"] = background_scores(img, per.styled_background(args.bg, st, args.seed), pm)
                    row = {"key": key, "bucket": bucket, "photo": fname, "people": an.people, "bg": args.bg,
                           "seed": args.seed, "style": st, "method": tag(spec), "face": m.as_dict(), "bgm": b,
                           **(base_seconds or {} if kind == "base" else pv), **extra}
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    print(f"{key}  worst={m.worst:.3f} mean={m.mean:.3f} n={m.n_output}/{m.n_input} "
                          f"ssim={b['ssim']:.3f} sift={b['sift_inliers']} t={row.get('seconds', 0):.0f}s "
                          f"vram={row.get('peak_reserved_gb', 0):.1f}GB", flush=True)


if __name__ == "__main__":
    main()
