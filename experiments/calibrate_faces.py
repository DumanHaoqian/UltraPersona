"""Give the face similarity numbers a meaning: what do same-person and different-person pairs score?

Uses the public LFW pairs set (sklearn downloads it once). Prints both distributions and the
threshold that best separates them; put a value near it into config.yaml quality.face_sim_min.

  python -m experiments.calibrate_faces
"""
import json
import os

import numpy as np
from sklearn.datasets import fetch_lfw_pairs

from persona2.face_metrics import detect
from persona2.pipeline import ROOT


def emb(img_float):
    img = (img_float * 255).astype(np.uint8) if img_float.max() <= 1.0 else img_float.astype(np.uint8)
    bgr = np.ascontiguousarray(img[..., ::-1])
    pad = np.pad(bgr, ((60, 60), (60, 60), (0, 0)), mode="edge")   # detector needs context around the face
    faces = detect(pad, det_thresh=0.3)
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])).normed_embedding


def main():
    data = fetch_lfw_pairs(subset="test", color=True, resize=1.0, funneled=True)
    same, diff = [], []
    for (a, b), y in zip(data.pairs, data.target):
        ea, eb = emb(a), emb(b)
        if ea is None or eb is None:
            continue
        (same if y == 1 else diff).append(float(np.dot(ea, eb)))
    same, diff = np.array(same), np.array(diff)
    ths = np.linspace(0, 1, 201)
    acc = [((same >= t).sum() + (diff < t).sum()) / (len(same) + len(diff)) for t in ths]
    best = float(ths[int(np.argmax(acc))])
    out = {
        "pairs_used": int(len(same) + len(diff)),
        "same_person": {"mean": float(same.mean()), "p10": float(np.percentile(same, 10)), "p50": float(np.median(same))},
        "different_people": {"mean": float(diff.mean()), "p90": float(np.percentile(diff, 90)), "p99": float(np.percentile(diff, 99))},
        "best_threshold": best, "accuracy_at_best": float(max(acc)),
    }
    print(json.dumps(out, indent=2))
    os.makedirs(os.path.join(ROOT, "experiments", "results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, "experiments", "results", "face_calibration.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
