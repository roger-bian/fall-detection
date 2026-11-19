#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv-gpu/bin/python
$PY -m falldet.pose data/videos/*/*.mp4        --out-dir data/pose/rtmo_ts          --tracker-timestamps
$PY -m falldet.pose data/videos_10fps/*/*.mp4  --out-dir data/pose/rtmo_10fps
$PY -m falldet.pose data/videos_10fps/*/*.mp4  --out-dir data/pose/rtmo_10fps_trk30    --tracker-fps 30
$PY -m falldet.pose data/videos_10fps/*/*.mp4  --out-dir data/pose/rtmo_10fps_trk30_ts --tracker-fps 30 --tracker-timestamps
echo ALL_DONE
