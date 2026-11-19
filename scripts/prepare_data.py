"""Build data/videos/ (uniform mp4) and data/annotations/manifest.csv from data/raw/.

    .venv-gpu/bin/python scripts/prepare_data.py            # or .venv-cpu
    .venv-gpu/bin/python scripts/prepare_data.py --no-transcode   # manifest only

Videos are re-encoded to H.264 mp4 at their native resolution and frame rate
(no resampling), so frame indices in the manifest are the same as in the raw files.

Ground truth:
  * CAUCAFall  -- per-frame YOLO labels (class 1 = fall, 0 = nofall; classes.txt).
                  label file N (1-based, last 5 digits of the name) -> frame N-1.
                  The authors' frame export has one extra file per video, a duplicate
                  of the last frame (verified by pixel-matching their PNGs to the
                  decoded .avi); indices >= n_frames are dropped.
                  gt_fall_start/end = first/last frame labelled fall.
  * GMDCSA24   -- clip-level only (Fall/ vs ADL/ folder + a text description).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
VIDEOS = ROOT / "data" / "videos"
ANN = ROOT / "data" / "annotations"

LICENSES = {"caucafall": "CC BY 4.0", "gmdcsa24": "MIT (GitHub) / CC BY 4.0 (Zenodo)"}

FIELDS = ["video_id", "dataset", "subject", "activity", "label", "path", "source_path",
          "width", "height", "fps", "n_frames", "duration_s",
          "gt_level", "gt_fall_start", "gt_fall_end", "gt_fall_frames",
          "label_files", "description", "license"]


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_read_packets",
         "-of", "json", str(path)], capture_output=True, text=True, check=True)
    s = json.loads(out.stdout)["streams"][0]
    # avg_frame_rate, not r_frame_rate: GMDCSA24 is variable frame rate with a 60 fps
    # timebase (r_frame_rate 60/1) but ~29.6 fps of actual frames.
    num, den = (int(x) for x in s["avg_frame_rate"].split("/"))
    fps = num / den
    n = int(s["nb_read_packets"])
    return {"width": s["width"], "height": s["height"], "fps": round(fps, 3),
            "n_frames": n, "duration_s": round(n / fps, 2)}


def transcode(src: Path, dst: Path, overwrite: bool) -> None:
    if dst.exists() and not overwrite:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    # -fps_mode passthrough: keep every source frame, never duplicate/drop.
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-map", "0:v:0",
         "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p",
         "-fps_mode", "passthrough", "-an", str(dst)], check=True)


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def caucafall_rows() -> list[dict]:
    rows = []
    base = RAW / "caucafall"
    for avi in sorted(base.glob("Subject.*/*/*.avi")):
        act_dir = avi.parent
        subj = int(act_dir.parent.name.split(".")[1])
        activity = act_dir.name
        labels = {}
        for t in act_dir.glob("*.txt"):
            m = re.search(r"(\d{5})\.txt$", t.name)
            if not m or t.name == "classes.txt":
                continue
            line = t.read_text().split()
            labels[int(m.group(1)) - 1] = int(line[0]) if line else None
        is_fall = activity.lower().startswith("fall")
        rows.append({
            "video_id": f"caucafall_S{subj:02d}_{slug(activity)}",
            "dataset": "caucafall", "subject": subj, "activity": activity,
            "label": "fall" if is_fall else "adl",
            "source": avi, "gt_level": "frame", "_labels": labels,
            "description": "", "license": LICENSES["caucafall"],
        })
    return rows


def gmdcsa24_rows() -> list[dict]:
    rows = []
    base = RAW / "gmdcsa24"
    for subj_dir in sorted(base.glob("Subject */")):
        subj = int(subj_dir.name.split()[1])
        for kind in ("Fall", "ADL"):
            desc = {}
            csv_path = subj_dir / f"{kind}.csv"
            if csv_path.exists():
                with open(csv_path, newline="", encoding="utf-8-sig") as fh:
                    r = csv.reader(fh)
                    next(r, None)
                    for rec in r:
                        if rec:
                            desc[rec[0].strip()] = ",".join(rec[1:]).strip()
            for mp4 in sorted((subj_dir / kind).glob("*.mp4")):
                rows.append({
                    "video_id": f"gmdcsa24_S{subj:02d}_{kind.lower()}_{mp4.stem}",
                    "dataset": "gmdcsa24", "subject": subj, "activity": kind,
                    "label": "fall" if kind == "Fall" else "adl",
                    "source": mp4, "gt_level": "clip",
                    "gt_fall_start": "", "gt_fall_end": "", "gt_fall_frames": "",
                    "label_files": "", "description": desc.get(mp4.name, ""),
                    "license": LICENSES["gmdcsa24"],
                })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-transcode", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--datasets", nargs="+", default=["caucafall", "gmdcsa24"],
                    help="subset to process (the manifest then lists only these)")
    args = ap.parse_args()

    builders = {"caucafall": caucafall_rows, "gmdcsa24": gmdcsa24_rows}
    rows = [r for d in args.datasets for r in builders[d]()]
    ANN.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(rows, 1):
        src = r.pop("source")
        dst = VIDEOS / r["dataset"] / f"{r['video_id']}.mp4"
        if not args.no_transcode:
            transcode(src, dst, args.overwrite)
        info = probe(dst if dst.exists() else src)
        r.update(info, path=str(dst.relative_to(ROOT)), source_path=str(src.relative_to(ROOT)))
        labels = r.pop("_labels", None)
        if labels is not None:
            n = info["n_frames"]
            extra = sorted(f for f in labels if f >= n)
            missing = n - sum(f < n for f in labels)
            if extra != [n] or missing:
                print(f"  ! {r['video_id']}: {len(labels)} label files for {n} frames "
                      f"(beyond end: {extra}, missing: {missing})", flush=True)
            fall = sorted(f for f, c in labels.items() if c == 1 and f < n)
            r.update(gt_fall_start=fall[0] if fall else "", gt_fall_end=fall[-1] if fall else "",
                     gt_fall_frames=len(fall), label_files=len(labels))
        if i % 20 == 0 or i == len(rows):
            print(f"[{i}/{len(rows)}] {r['video_id']}", flush=True)

    with open(ANN / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    n_fall = sum(r["label"] == "fall" for r in rows)
    print(f"wrote {ANN / 'manifest.csv'}: {len(rows)} videos ({n_fall} fall, "
          f"{len(rows) - n_fall} adl)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
