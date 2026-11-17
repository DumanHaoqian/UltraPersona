#!/bin/bash
# Start the web server without the model and check the endpoints that do not need the GPU.
cd /mnt/c/Users/admin/CC/UltraPersona
export PERSONA_NO_MODEL=1 HF_HOME=/mnt/d/hf_models/hf_home
PY=$HOME/Projects/ultrapersona/.venv/bin/python
$PY -m uvicorn persona2.server:app --host 127.0.0.1 --port 7861 > /tmp/smoke_server.log 2>&1 &
PID=$!
for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:7861/api/options && break; sleep 1; done
echo "--- options"; curl -s http://127.0.0.1:7861/api/options | head -c 600; echo
echo "--- index"; curl -s -o /dev/null -w "%{http_code} %{size_download}\n" http://127.0.0.1:7861/
echo "--- background"; curl -s -o /dev/null -w "%{http_code} %{size_download}\n" http://127.0.0.1:7861/backgrounds/Y.jpg
for b in people_1 people_3 people_many; do
  f=$(ls testcases/selected/$b | head -1)
  echo "--- analyze $b"; curl -s -F "photo=@testcases/selected/$b/$f" http://127.0.0.1:7861/api/analyze; echo
done
echo "--- generate (expect 503, no model)"; curl -s -o /dev/null -w "%{http_code}\n" -F "photo=@testcases/selected/people_1/$(ls testcases/selected/people_1 | head -1)" -F background=1 -F style=1 http://127.0.0.1:7861/api/generate
kill $PID
tail -5 /tmp/smoke_server.log
