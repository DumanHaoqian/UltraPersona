#!/bin/bash
# Run a python module/script from the repo inside the WSL venv: tools/run.sh <args to python>
cd /mnt/c/Users/admin/CC/UltraPersona
export HF_HOME=/mnt/d/hf_models/hf_home PYTHONUNBUFFERED=1
exec $HOME/Projects/ultrapersona/.venv/bin/python "$@"
