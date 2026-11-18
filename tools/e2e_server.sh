#!/bin/bash
# Start the server with the model and run one real generation through the HTTP API.
cd /mnt/c/Users/admin/CC/UltraPersona
export HF_HOME=/mnt/d/hf_models/hf_home
PY=$HOME/Projects/ultrapersona/.venv/bin/python
OUT=experiments/results/e2e; mkdir -p $OUT
$PY -m uvicorn persona2.server:app --host 127.0.0.1 --port 7863 > $OUT/server.log 2>&1 &
PID=$!
for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:7863/api/options && break; sleep 1; done
for i in $(seq 1 120); do curl -s http://127.0.0.1:7863/api/options | grep -q '"model_ready":true' && break; sleep 10; done
echo "model ready after ~$((i*10)) s"
CASES=("$@"); [ ${#CASES[@]} -eq 0 ] && CASES=("people_2 4 1" "people_1 1 3")   # "bucket background style"
for case in "${CASES[@]}"; do
  set -- $case
  f=$(ls testcases/selected/$1 | head -1)
  /usr/bin/time -f "%e s" curl -s -F "photo=@testcases/selected/$1/$f" -F background=$2 -F style=$3 http://127.0.0.1:7863/api/generate > $OUT/$1_bg$2_st$3.json
  $PY - $OUT/$1_bg$2_st$3.json <<'PY'
import sys, json, base64
p = sys.argv[1]; r = json.load(open(p))
print(p, "ok" if r.get("ok") else "FAIL", r.get("people"), r.get("face_similarity"), r.get("messages"))
if r.get("image"): open(p[:-5] + ".jpg", "wb").write(base64.b64decode(r["image"]))
PY
done
kill $PID
