#!/bin/bash
# Screenshot the page (no model loaded) at desktop and phone width: tools/screenshot_page.sh <out_dir>
cd /mnt/c/Users/admin/CC/UltraPersona
OUT=${1:-/mnt/c/Users/admin/CC/UltraPersona/experiments/results/page}
mkdir -p "$OUT"
export PERSONA_NO_MODEL=1
PY=$HOME/Projects/ultrapersona/.venv/bin/python
$PY -m uvicorn persona2.server:app --host 127.0.0.1 --port 7862 > /tmp/shot_server.log 2>&1 &
PID=$!
for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:7862/api/options && break; sleep 1; done
PHOTO=$(ls -d testcases/selected/people_3/* | head -1)
$PY - "$OUT" "$PHOTO" <<'EOF'
import sys
from playwright.sync_api import sync_playwright
out, photo = sys.argv[1], sys.argv[2]
with sync_playwright() as p:
    b = p.chromium.launch()
    for name, vp in [("desktop", {"width": 1440, "height": 1000}), ("phone", {"width": 390, "height": 844})]:
        pg = b.new_page(viewport=vp)
        pg.goto("http://127.0.0.1:7862/")
        pg.wait_for_selector("#bgs .opt")
        pg.set_input_files("#file", photo)
        pg.wait_for_function("document.getElementById('found').textContent.includes('照片里有')", timeout=60000)
        pg.click("#bgs .opt:nth-child(3)")
        pg.click("#styles .opt:nth-child(2)")
        pg.wait_for_timeout(800)
        pg.screenshot(path=f"{out}/{name}.png", full_page=True)
        print("saved", name)
    b.close()
EOF
kill $PID
