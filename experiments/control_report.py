"""Tables for experiments/control_ablation.py results.

  python -m experiments.control_report --run c1 > experiments/results/control/c1/report.md
Run experiments.score_style --dir ... first to include style strength.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

from persona2.pipeline import ROOT, load_config

COLS = ["张数", "最差那个人", "平均", "人数正确", "背景 SSIM", "建筑匹配点", "风格化背景匹配点", "风格强度", "秒"]


def load(run):
    d = os.path.join(ROOT, "experiments", "results", "control", run)
    rows = list({r["key"]: r for r in map(json.loads, open(os.path.join(d, "metrics.jsonl"), encoding="utf-8"))}.values())
    sp = os.path.join(d, "style_scores.jsonl")
    if os.path.exists(sp):
        ss = {json.loads(l)["key"]: json.loads(l)["style_score"] for l in open(sp, encoding="utf-8")}
        for r in rows:
            r["style_score"] = ss.get(r["key"])
    return rows


def agg(rs):
    f = lambda k: np.nanmean([x for x in k if x is not None]) if any(x is not None for x in k) else float("nan")
    return [len(rs), f([r["face"]["worst"] for r in rs]), f([r["face"]["mean"] for r in rs]),
            f([float(r["face"]["count_ok"]) for r in rs]), f([r["bgm"]["ssim"] for r in rs]),
            f([r["bgm"]["sift_inliers"] for r in rs]), f([r.get("bgm_styled", {}).get("sift_inliers") for r in rs]),
            f([r.get("style_score") for r in rs]), f([r.get("seconds") for r in rs])]


def fmt(v, i):
    if isinstance(v, float) and np.isnan(v):
        return "–"
    return {0: "{:d}", 3: "{:.0%}", 5: "{:.0f}", 6: "{:.0f}", 8: "{:.0f}"}.get(i, "{:.3f}").format(v)


def table(title, groups):
    print(f"\n### {title}\n")
    print("| 组 | 方法 | " + " | ".join(COLS) + " |")
    print("|---|---|" + "---|" * len(COLS))
    for (g, m), rs in groups:
        print(f"| {g} | {m} | " + " | ".join(fmt(v, i) for i, v in enumerate(agg(rs))) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="c1")
    args = ap.parse_args()
    cfg = load_config()
    rows = load(args.run)
    methods = list(dict.fromkeys(r["method"] for r in rows))
    by = defaultdict(list)
    for r in rows:
        by[(r["style"], r["method"])].append(r)
    table("按风格", [((cfg["styles"][s]["name_zh"], m), by[(s, m)]) for s in sorted({r["style"] for r in rows})
                   for m in methods if by[(s, m)]])
    byp = defaultdict(list)
    for r in rows:
        byp[(r["people"], r["method"])].append(r)
    table("按人数", [((f"{p} 人", m), byp[(p, m)]) for p in sorted({r["people"] for r in rows}) for m in methods
                   if byp[(p, m)]])
    table("全部", [(("全部", m), [r for r in rows if r["method"] == m]) for m in methods])

    base = {(r["photo"], r["style"]): r for r in rows if r["method"] == "base"}
    print("\n### 和 base 配对比较（同一照片、风格）\n")
    print("| 方法 | 对数 | 最差那个人 平均变化 | 变好的比例 | 平均相似度 平均变化 |")
    print("|---|---|---|---|---|")
    for m in methods:
        if m == "base":
            continue
        d = [(r["face"]["worst"] - base[(r["photo"], r["style"])]["face"]["worst"],
              r["face"]["mean"] - base[(r["photo"], r["style"])]["face"]["mean"])
             for r in rows if r["method"] == m and (r["photo"], r["style"]) in base]
        if d:
            a = np.array(d)
            print(f"| {m} | {len(a)} | {a[:, 0].mean():+.3f} | {(a[:, 0] > 0).mean():.0%} | {a[:, 1].mean():+.3f} |")


if __name__ == "__main__":
    main()
