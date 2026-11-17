#!/bin/bash
R=/mnt/c/Users/admin/CC/UltraPersona/tools/run.sh
for p in "$@"; do
  bash $R -m experiments.ablate_quant --preset $p --steps 20 2>&1 | tr '\r' '\n' | grep -aE '^\{|Error|Traceback|\[qwen' | cut -c1-500
done
