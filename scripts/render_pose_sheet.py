"""Side-by-side skeleton overlays of several pose configs on chosen frames.

    .venv-gpu/bin/python scripts/render_pose_sheet.py OUT.jpg \
        --tags rtmo_t0.3 balanced_t0.3 --item caucafall_S03_fall_sitting:40,120 ...

One row per (video, frame), one column per tag. Keypoints with conf < 0.3 are not drawn.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
COLORS = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 165, 255), (0, 255, 0)]


def read_frame(path: str, idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(ROOT / path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, f = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"cannot read frame {idx} of {path}")
    return f


def draw(img, df, frame, thr=0.3):
    img = img.copy()
    scale = max(1, img.shape[0] // 360)
    for _, r in df[df.frame == frame].iterrows():
        col = COLORS[int(r.track_id) % len(COLORS)]
        kp = np.asarray(r.kp17_xyc).reshape(17, 3)
        for a, b in EDGES:
            if kp[a, 2] >= thr and kp[b, 2] >= thr:
                cv2.line(img, tuple(map(int, kp[a, :2])), tuple(map(int, kp[b, :2])), col, 2 * scale)
        for x, y, c in kp:
            if c >= thr:
                cv2.circle(img, (int(x), int(y)), 3 * scale, col, -1)
        cv2.putText(img, f"#{int(r.track_id)} {r.kp_mean_conf:.2f}",
                    (int(r.bbox_x), max(15, int(r.bbox_y) - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * scale, col, scale)
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--item", action="append", required=True, help="video_id:f1,f2,...")
    ap.add_argument("--width", type=int, default=480, help="tile width")
    args = ap.parse_args()

    manifest = {r["video_id"]: r for r in csv.DictReader(open(ROOT / "data/annotations/manifest.csv"))}
    rows = []
    for item in args.item:
        vid, frames = item.split(":")
        dfs = {t: pd.read_parquet(ROOT / f"data/pose/{t}/{vid}.tracks.parquet") for t in args.tags}
        for f in map(int, frames.split(",")):
            base = read_frame(manifest[vid]["path"], f)
            tiles = []
            for t in args.tags:
                tile = draw(base, dfs[t], f)
                h = int(tile.shape[0] * args.width / tile.shape[1])
                tile = cv2.resize(tile, (args.width, h))
                cv2.rectangle(tile, (0, 0), (args.width, 22), (0, 0, 0), -1)
                cv2.putText(tile, f"{t} | {vid} f{f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (255, 255, 255), 1)
                tiles.append(tile)
            rows.append(np.hstack(tiles))
    w = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
    cv2.imwrite(args.out, np.vstack(rows))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
