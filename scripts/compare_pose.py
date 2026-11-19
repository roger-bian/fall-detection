"""Compare pose configs on the test set: does the person stay detected and tracked
through the fall and while lying on the floor?

    .venv-gpu/bin/python scripts/compare_pose.py rtmo_t0.3 rtmo_t0.1 balanced_t0.3

Reads data/pose/<tag>/*.tracks.parquet + data/annotations/manifest.csv and writes
data/pose/compare.json (plus a table on stdout).

Phases (CAUCAFall fall clips, frame-level GT):
  before   [0, onset)                 standing / walking
  falling  [onset, onset + 2 s)       the fall itself
  lying    [onset + 2 s, end]         on the floor
GMDCSA24 has clip labels only; there "lying" is the last third of a fall clip.

All CAUCAFall and GMDCSA24 clips show one person, so:
  coverage   = share of frames with >= 1 tracked person
  multi      = share of frames with >= 2 tracked persons (duplicates / false positives)
  ids_after  = distinct track ids from onset to the end (1 = one unbroken identity;
               every switch resets per-track baselines such as `sink`)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def phase_masks(n: int, onset: int, fps: float, label: str, gt_level: str):
    idx = np.arange(n)
    if label == "fall" and gt_level == "frame":
        fall_end = onset + int(round(2 * fps))
        return {"before": idx < onset, "falling": (idx >= onset) & (idx < fall_end),
                "lying": idx >= fall_end, "after_onset": idx >= onset}
    if label == "fall":
        return {"all": np.ones(n, bool), "lying": idx >= int(n * 2 / 3),
                "after_onset": idx >= int(n * 2 / 3)}
    return {"all": np.ones(n, bool)}


def per_video(df: pd.DataFrame, meta: dict, row: dict) -> dict:
    n = int(row["n_frames"])
    fps = float(row["fps"])
    onset = int(row["gt_fall_start"]) if row["gt_fall_start"] else 0
    per_frame = np.zeros(n, int)
    conf = np.full(n, np.nan)
    if len(df):
        counts = df.groupby("frame").size()
        f = counts.index.to_numpy()
        per_frame[f[f < n]] = counts.to_numpy()[f < n]
        best = df.groupby("frame")["kp_mean_conf"].max()
        b = best.index.to_numpy()
        conf[b[b < n]] = best.to_numpy()[b < n]
    out = {"video_id": row["video_id"], "dataset": row["dataset"], "label": row["label"],
           "ms_per_frame": meta["ms_per_frame"], "tracks": meta["tracks"]}
    for name, m in phase_masks(n, onset, fps, row["label"], row["gt_level"]).items():
        if not m.any():
            continue
        if name == "after_onset":
            ids = df[df["frame"] >= np.argmax(m)]["track_id"].nunique() if len(df) else 0
            out["ids_after"] = int(ids)
            continue
        out[f"cov_{name}"] = float((per_frame[m] >= 1).mean())
        out[f"multi_{name}"] = float((per_frame[m] >= 2).mean())
        c = conf[m]
        out[f"conf_{name}"] = float(np.nanmean(c)) if np.isfinite(c).any() else np.nan
    return out


def main(tags: list[str]) -> int:
    manifest = list(csv.DictReader(open(ROOT / "data/annotations/manifest.csv")))
    results = {}
    for tag in tags:
        rows = []
        for r in manifest:
            base = ROOT / "data/pose" / tag / r["video_id"]
            if not Path(f"{base}.meta.json").exists():
                continue
            meta = json.load(open(f"{base}.meta.json"))
            df = pd.read_parquet(f"{base}.tracks.parquet") if meta["rows"] else pd.DataFrame(
                columns=["frame", "track_id", "kp_mean_conf"])
            rows.append(per_video(df, meta, r))
        results[tag] = pd.DataFrame(rows)

    groups = [("caucafall", "fall"), ("caucafall", "adl"), ("gmdcsa24", "fall"),
              ("gmdcsa24", "adl")]
    cols = ["cov_before", "cov_falling", "cov_lying", "cov_all", "conf_lying", "conf_all",
            "multi_falling", "multi_lying", "multi_all", "ids_after", "tracks", "ms_per_frame"]
    summary = {}
    for ds, lab in groups:
        print(f"\n## {ds} / {lab}")
        table = {}
        for tag, d in results.items():
            g = d[(d.dataset == ds) & (d.label == lab)]
            if g.empty:
                continue
            s = {"n": len(g)}
            for c in cols:
                if c in g and g[c].notna().any():
                    s[c] = round(float(g[c].mean()), 3)
            if "ids_after" in g and g["ids_after"].notna().any():
                s["clips_ids_after_1"] = int((g["ids_after"] == 1).sum())
                s["clips_lying_cov_lt_50pct"] = int((g.get("cov_lying", pd.Series()) < 0.5).sum())
            table[tag] = s
        summary[f"{ds}/{lab}"] = table
        print(pd.DataFrame(table).to_string())

    out = ROOT / "data/pose/compare.json"
    json.dump({"summary": summary,
               "per_video": {t: d.to_dict(orient="records") for t, d in results.items()}},
              open(out, "w"), indent=1, default=float)
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["rtmo_t0.3", "rtmo_t0.1", "balanced_t0.3"]))
