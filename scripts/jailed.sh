#!/usr/bin/env bash
# Run a Python module of the repo inside the kernel sandbox of scripts/jail.sb (macOS).
#   scripts/jailed.sh NAME -m qita.bench run ...
# QITA_PY picks another interpreter inside the repo (for a model the main venv cannot load); the default
# is the repo venv. QITA_MODELS is the folder of local models (default ~/qita-models).
# Writes go only to runs/jail/NAME (TMPDIR included); no network; reads only repo, models, Python.
set -euo pipefail
name=$1; shift
repo="$(cd "$(dirname "$0")/.." && pwd)"
run="$repo/runs/jail/$name"
mkdir -p "$run/tmp"
cd "$repo"
exec env TMPDIR="$run/tmp" HOME="$HOME" sandbox-exec -f "$repo/scripts/jail.sb" \
  -D REPO="$repo" -D RUN="$run" -D MODELS="${QITA_MODELS:-$HOME/qita-models}" \
  -D PYHOME="$HOME/.local/share/uv" -D USERHOME="$HOME" \
  "$repo/${QITA_PY:-.venv/bin/python}" "$@"
