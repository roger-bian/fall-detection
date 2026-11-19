"""Make a reduced-frame-rate copy of the test set, to simulate a slow camera / CPU.

    python scripts/make_lowfps.py --fps 10
    -> data/videos_10fps/{caucafall,gmdcsa24}/*.mp4 + data/annotations/manifest_10fps.csv

Frame-level GT (CAUCAFall, 20 fps) needs an integer frame ratio: every k-th frame is kept
(k = 20 / 10 = 2) and GT frame N maps to N // k. Clip-level GT (GMDCSA24, variable
frame rate) is resampled with ffmpeg's fps filter; its labels need no mapping.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def n_frames(path: Path) -> int:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, check=True).stdout
    return int(out.strip())


def convert(r: dict, fps: float, out_dir: Path) -> dict:
    src = ROOT / r["path"]
    dst = out_dir / r["dataset"] / f"{r['video_id']}.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)
    src_fps = float(r["fps"])
    r = dict(r)
    if r["gt_level"] == "frame":
        k = src_fps / fps
        if abs(k - round(k)) > 1e-6:
            raise ValueError(f"{r['video_id']}: frame-level GT needs an integer ratio, got {k}")
        k = int(round(k))
        vf = f"select=not(mod(n\\,{k})),setpts=N/{fps:g}/TB"
        for c in ("gt_fall_start", "gt_fall_end"):
            if r[c]:
                r[c] = str(int(r[c]) // k)
        if r["gt_fall_start"]:
            r["gt_fall_frames"] = str(int(r["gt_fall_end"]) - int(r["gt_fall_start"]) + 1)
    else:
        vf = f"fps={fps:g}"
    if not dst.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, "-r", f"{fps:g}",
                        "-c:v", "libx264", "-crf", "18", "-preset", "veryfast",
                        "-pix_fmt", "yuv420p", "-an", str(dst)], check=True)
    r["path"] = str(dst.relative_to(ROOT))
    r["fps"] = f"{fps:g}"
    r["n_frames"] = str(n_frames(dst))
    r["duration_s"] = f"{int(r['n_frames']) / fps:.2f}"
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    tag = f"{args.fps:g}fps"
    out_dir = ROOT / "data" / f"videos_{tag}"
    man = list(csv.DictReader(open(ROOT / "data/annotations/manifest.csv")))
    with ThreadPoolExecutor(args.workers) as ex:
        rows = list(ex.map(lambda r: convert(r, args.fps, out_dir), man))
    dst = ROOT / "data/annotations" / f"manifest_{tag}.csv"
    with open(dst, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(man[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} videos to {out_dir.relative_to(ROOT)} and {dst.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
