"""Sort downloaded test photos by the number of usable faces.

A face counts when its box is at least --min-face pixels on the shorter side and the detector
score is >= 0.6. Photos that also contain smaller faces (background crowd) go to 'mixed' and are
not used, because the expected people count would be ambiguous.

  python tools/bucket_testcases.py testcases
writes testcases/people_<n>/ (n = 1..5), testcases/people_many/, testcases/manifest.csv
"""
import csv
import os
import shutil
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from persona2.face_metrics import detect  # noqa: E402


def main():
    root = sys.argv[1]
    min_face = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    raw = os.path.join(root, "raw")
    src = {r["file"]: r for r in csv.DictReader(open(os.path.join(root, "sources.csv"), encoding="utf-8"))}
    rows = []
    for f in sorted(os.listdir(raw)):
        img = cv2.imread(os.path.join(raw, f))
        if img is None:
            continue
        faces = detect(img, det_thresh=0.5)
        big = [x for x in faces if min(x.bbox[2] - x.bbox[0], x.bbox[3] - x.bbox[1]) >= min_face and x.det_score >= 0.6]
        small = len(faces) - len(big)
        n = len(big)
        if n == 0:
            bucket = "no_face"
        elif small > 0:
            bucket = "mixed"
        elif n <= 5:
            bucket = f"people_{n}"
        else:
            bucket = "people_many"
        if bucket.startswith("people_"):
            os.makedirs(os.path.join(root, bucket), exist_ok=True)
            shutil.copy2(os.path.join(raw, f), os.path.join(root, bucket, f))
        meta = src.get(f, {})
        rows.append({"file": f, "bucket": bucket, "faces_big": n, "faces_small": small,
                     "license": meta.get("license", ""), "author": meta.get("author", ""),
                     "page_url": meta.get("page_url", "")})
        print(f"{bucket:12s} big={n} small={small} {f}")
    with open(os.path.join(root, "manifest.csv"), "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(Counter(r["bucket"] for r in rows))


if __name__ == "__main__":
    main()
