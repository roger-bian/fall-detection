#!/bin/bash
# Pose extraction over the whole test set for the model comparison.
# Output: data/pose/<tag>/<video_id>.{tracks.parquet,meta.json}; skips videos already done.
set -e
cd "$(dirname "$0")/.."
PY=.venv-gpu/bin/python
VIDEOS=(data/videos/caucafall/*.mp4 data/videos/gmdcsa24/*.mp4)
run() { tag=$1; shift; echo "=== $tag $(date +%T)"; $PY -m falldet.pose "${VIDEOS[@]}" --out-dir data/pose/$tag "$@" 2>&1 | grep --line-buffered '^\['; }
run rtmo_t0.3      --model rtmo     --det-score-thr 0.3
run rtmo_t0.1      --model rtmo     --det-score-thr 0.1
run balanced_t0.3  --model balanced --det-score-thr 0.3
echo "=== ALL DONE $(date +%T)"
