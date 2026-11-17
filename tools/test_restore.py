"""Try background restoration on one generated picture:
python tools/test_restore.py <generated.png> <background.jpg> <out.png>"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from PIL import Image  # noqa: E402

from persona2.matting import restore_background  # noqa: E402

gen, bg = Image.open(sys.argv[1]), Image.open(sys.argv[2])
restore_background(gen, bg).save(sys.argv[3])
print("saved", sys.argv[3])
