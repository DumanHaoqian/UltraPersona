"""One comparison sheet per test photo: input, realistic composite, then one-step vs two-step per style.

  python tools/demo_sheets.py <out_dir> [--run v1]
Run on Windows or WSL from the repo root; needs a CJK font (Microsoft YaHei or Noto CJK).
"""
import argparse
import json
import os

from PIL import Image, ImageDraw, ImageFont, ImageOps

STYLES = {1: "梵高《星空》", 3: "3D 动画", 4: "JoJo 漫画", 5: "赛博朋克"}
W, H, LAB = 420, 315, 34


def load_font(size):
    for f in ("C:/Windows/Fonts/msyh.ttc", "/mnt/c/Windows/Fonts/msyh.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"):
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--run", default="v1")
    a = ap.parse_args()
    d = f"experiments/results/one_vs_two/{a.run}"
    rows = [json.loads(l) for l in open(f"{d}/metrics.jsonl", encoding="utf-8")]
    font, small = load_font(22), load_font(18)
    os.makedirs(a.out, exist_ok=True)

    def tile(img, text):
        t = Image.new("RGB", (W, H + LAB), "white")
        t.paste(ImageOps.pad(img.convert("RGB"), (W, H), color=(235, 235, 235)), (0, LAB))
        ImageDraw.Draw(t).text((8, 4), text, fill="black", font=small)
        return t

    def face(r):
        w = r["face"].get("worst")
        return f"最差 {w:.2f}" if w is not None else "未检出"

    photos = sorted({(r["bucket"], r["photo"]) for r in rows})
    for n, (bucket, photo) in enumerate(photos, 1):
        rs = {(r["style"], r["method"]): r for r in rows if r["photo"] == photo and r["bucket"] == bucket}
        comp = rs[(0, "composite")]
        grid = [[tile(Image.open(f"testcases/selected/{bucket}/{photo}"), "输入照片"),
                 tile(Image.open(f"{d}/images/{comp['key']}.png"), f"写实合成  {face(comp)}")]]
        for st, name in STYLES.items():
            grid.append([tile(Image.open(f"{d}/images/{r['key']}.png"), f"{name} · {mz}  {face(r)}")
                         for m, mz in (("onestep", "一步"), ("twostep", "两步"))
                         if (r := rs.get((st, m)))])
        sheet = Image.new("RGB", (2 * W + 30, len(grid) * (H + LAB + 10) + 60), "white")
        people = bucket.replace("people_", "")
        ImageDraw.Draw(sheet).text((10, 12), f"{people} 人 · {photo[:60]}", fill="black", font=font)
        for i, line in enumerate(grid):
            for j, t in enumerate(line):
                sheet.paste(t, (10 + j * (W + 10), 60 + i * (H + LAB + 10)))
        name = f"对比_{n:02d}_{people}人.jpg"
        sheet.save(os.path.join(a.out, name), quality=88)
        print("saved", name, photo)


if __name__ == "__main__":
    main()
