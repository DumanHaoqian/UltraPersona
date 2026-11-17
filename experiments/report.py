"""Summarise one-step vs two-step results as Markdown tables.

  python -m experiments.report --run v1  > experiments/results/one_vs_two/v1/report.md
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np

from persona2.pipeline import ROOT, load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="v1")
    args = ap.parse_args()
    cfg = load_config()
    d = os.path.join(ROOT, "experiments", "results", "one_vs_two", args.run)
    rows = [json.loads(l) for l in open(os.path.join(d, "metrics.jsonl"))]
    ss_path = os.path.join(d, "style_scores.jsonl")
    ss = {}
    if os.path.exists(ss_path):
        ss = {json.loads(l)["key"]: json.loads(l)["style_score"] for l in open(ss_path)}

    def summarize(group):
        w = [r["face"]["worst"] for r in group]
        return {
            "n": len(group),
            "worst": np.mean(w),
            "mean": np.mean([r["face"]["mean"] for r in group]),
            "count_ok": np.mean([r["face"]["count_ok"] for r in group]),
            "detected": np.mean([r["face"]["detected_ratio"] for r in group]),
            "swaps": np.mean([r["face"]["swaps"] for r in group]),
            "style": np.mean([ss[r["key"]] for r in group if r["key"] in ss]) if ss else float("nan"),
            "sec": np.mean([r["seconds"] for r in group]),
        }

    head = "| {} | 方法 | 张数 | 最差那个人的相似度 | 平均相似度 | 人数正确率 | 检出人脸比例 | 认错人次数 | 风格强度 | 秒/张 |"
    sep = "|---|---|---|---|---|---|---|---|---|---|"

    def table(key_name, keyf, label):
        g = defaultdict(list)
        for r in rows:
            g[(keyf(r), r["method"])].append(r)
        print(f"\n### 按{label}\n")
        print(head.format(label))
        print(sep)
        for (k, m) in sorted(g):
            s = summarize(g[(k, m)])
            print(f"| {k} | {m} | {s['n']} | {s['worst']:.3f} | {s['mean']:.3f} | {s['count_ok']:.0%} | "
                  f"{s['detected']:.0%} | {s['swaps']:.2f} | {s['style']:.3f} | {s['sec']:.0f} |")

    table("style", lambda r: cfg["styles"][r["style"]]["name"] if r["style"] else "写实合成", "风格")
    table("people", lambda r: r["people"], "人数")
    # Paired comparison: same photo/bg/seed/style, onestep vs twostep.
    by = defaultdict(dict)
    for r in rows:
        if r["method"] in ("onestep", "twostep"):
            by[r["key"].rsplit("__", 1)[0]][r["method"]] = r["face"]["worst"]
    diffs = [v["twostep"] - v["onestep"] for v in by.values() if len(v) == 2]
    if diffs:
        diffs = np.array(diffs)
        print(f"\n配对比较（同一照片、背景、种子、风格）：{len(diffs)} 对；两步减一步的“最差那个人相似度”平均差 "
              f"{diffs.mean():+.3f}，两步更好的比例 {np.mean(diffs > 0):.0%}。")


if __name__ == "__main__":
    main()
