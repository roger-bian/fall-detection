"""Frame-rate / tracker-timing comparison: does tracking (and the v2 rule) survive a slow
camera, and does the tracker need to be told the real frame rate?

    .venv-gpu/bin/python scripts/compare_lowfps.py

Configs (pose from data/pose/<tag>, made by scripts/run_lowfps_pose.sh):
  rtmo_t0.3_dedup      original frame rate, tracker counts frames (current batch results)
  rtmo_ts              original frame rate, tracker driven by timestamps
  rtmo_10fps           10 fps video, tracker told 10 fps
  rtmo_10fps_trk30     10 fps video, tracker told 30 fps (camera reports 30, delivers 10)
  rtmo_10fps_trk30_ts  as above, but tracker driven by timestamps
  ...+stitch           the original-fps and 10 fps runs with track stitching (falldet.stitch)

Each is scored with the fixed v2 thresholds (scripts/run_rules_v2.py --fixed) into
results/lowfps/<tag>/. Tracking metrics:
  ids_after   distinct track ids from fall onset to the end (1 = one identity throughout;
              CAUCAFall: GT onset, GMDCSA24: last third of the clip)
  restarts    new track ids that start within 2 s of another track ending (the person
              lost and picked up again under a new id), summed over clips
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from compare_pose import per_video  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
from falldet.stitch import stitch_tracks  # noqa: E402

# (name, pose tag, manifest, stitch, description)
M, M10 = "data/annotations/manifest.csv", "data/annotations/manifest_10fps.csv"
CONFIGS = [
    ("rtmo_t0.3_dedup", "rtmo_t0.3_dedup", M, False, "original fps, frame counting"),
    ("rtmo_ts", "rtmo_ts", M, False, "original fps, timestamps"),
    ("rtmo_10fps", "rtmo_10fps", M10, False, "10 fps, tracker told 10"),
    ("rtmo_10fps_trk30", "rtmo_10fps_trk30", M10, False, "10 fps, tracker told 30"),
    ("rtmo_10fps_trk30_ts", "rtmo_10fps_trk30_ts", M10, False,
     "10 fps, tracker told 30 + timestamps"),
    ("rtmo_t0.3_dedup+stitch", "rtmo_t0.3_dedup", M, True, "original fps + stitching"),
    ("rtmo_10fps+stitch", "rtmo_10fps", M10, True, "10 fps + stitching"),
]
GROUPS = [("tune/caucafall", "CAUCAFall"), ("tune/gmdcsa24_S1-2", "GMDCSA24 S1-2"),
          ("test/gmdcsa24_S3-4", "GMDCSA24 S3-4")]


def restarts(df: pd.DataFrame, fps: float, gap_s: float = 2.0) -> int:
    if df.empty:
        return 0
    span = df.groupby("track_id")["frame"].agg(["min", "max"]).sort_values("min")
    n = 0
    for tid, s in span.iterrows():
        ended = span[(span.index != tid) & (span["max"] < s["min"])
                     & (s["min"] - span["max"] <= gap_s * fps)]
        n += int(len(ended) > 0)
    return n


def tracking(tag: str, man: list[dict], stitch: bool) -> pd.DataFrame:
    rows = []
    for r in man:
        base = ROOT / "data/pose" / tag / r["video_id"]
        meta = json.load(open(f"{base}.meta.json"))
        df = pd.read_parquet(f"{base}.tracks.parquet") if meta["rows"] else pd.DataFrame(
            columns=["frame", "track_id", "kp_mean_conf"])
        if stitch and len(df):
            df = stitch_tracks(df, float(r["fps"]))
            meta = {**meta, "tracks": int(df["track_id"].nunique())}
        v = per_video(df, meta, r)
        v["restarts"] = restarts(df, float(r["fps"]))
        rows.append(v)
    return pd.DataFrame(rows)


def main() -> int:
    out = ROOT / "results/lowfps"
    out.mkdir(parents=True, exist_ok=True)
    table, per_clip = [], {}
    for name, tag, manifest, stitch, desc in CONFIGS:
        man = list(csv.DictReader(open(ROOT / manifest)))
        res = out / name
        subprocess.run([sys.executable, str(ROOT / "scripts/run_rules_v2.py"), "--fixed",
                        "--pose", tag, "--manifest", manifest, "--out", str(res)]
                       + ([] if stitch else ["--no-stitch"]), check=True, capture_output=True)
        summ = json.load(open(res / "summary.json"))
        pv = pd.read_csv(res / "per_video.csv")
        tr = tracking(tag, man, stitch)
        falls = tr[tr.label == "fall"]
        row = {"config": name, "desc": desc,
               "fall_clips_one_id": f"{int((falls.ids_after == 1).sum())}/{len(falls)}",
               "restarts_fall": int(falls.restarts.sum()),
               "restarts_adl": int(tr[tr.label == "adl"].restarts.sum()),
               "tracks_per_clip": round(float(tr.tracks.mean()), 2)}
        for key, label in GROUPS:
            s = summ[key]
            row[label] = f"{s['TP']}/{s['falls']} TP, {s['FP']}/{s['adl']} FP"
        table.append(row)
        per_clip[name] = pv.set_index("video_id")["result"]
        print(f"done {name}", flush=True)

    t = pd.DataFrame(table)
    # each config vs its unstitched counterpart (or the original-fps baseline)
    ref_of = {"rtmo_t0.3_dedup+stitch": "rtmo_t0.3_dedup", "rtmo_10fps+stitch": "rtmo_10fps"}
    changes = {}
    for name in per_clip:
        if name == "rtmo_t0.3_dedup":
            continue
        ref = ref_of.get(name, "rtmo_t0.3_dedup")
        base = per_clip[ref]
        d = per_clip[name][per_clip[name] != base]
        changes[f"{name} vs {ref}"] = {v: f"{base[v]} -> {d[v]}" for v in d.index}
    json.dump({"table": table, "changed_vs_original": changes}, open(out / "compare.json", "w"),
              indent=2)
    pd.set_option("display.width", 250, "display.max_colwidth", 60)
    print(t.to_string(index=False))
    for tag, c in changes.items():
        print(f"\n{tag}: {len(c)} clips changed vs original")
        for v, s in c.items():
            print(f"  {v}: {s}")
    print(f"\nwrote {out}/compare.json and results/lowfps/<config>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
