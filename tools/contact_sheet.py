"""Make a numbered contact sheet of a folder: python tools/contact_sheet.py <dir> <out.jpg>"""
import os, sys
from PIL import Image, ImageDraw
d, out = sys.argv[1], sys.argv[2]
files = sorted(os.listdir(d)); cols = 6; w = 260
rows = (len(files) + cols - 1) // cols
sheet = Image.new("RGB", (cols * w, rows * w), "white"); dr = ImageDraw.Draw(sheet)
for i, f in enumerate(files):
    im = Image.open(os.path.join(d, f)).convert("RGB"); im.thumbnail((w - 6, w - 24))
    x, y = (i % cols) * w, (i // cols) * w
    sheet.paste(im, (x + 3, y + 20)); dr.text((x + 4, y + 4), f"{i} {f[:34]}", fill="black")
sheet.save(out, quality=80); print(len(files))
