"""Apply the fall rules to extracted keypoints, score against GT, explain each error.

    .venv-gpu/bin/python scripts/run_rules.py --pose rtmo_t0.3_dedup --out results/rules_v1

Scoring
  CAUCAFall fall clips (frame GT): HIT if any alarm in [onset - 0.5 s, end].
      Alarms before onset - 0.5 s are counted separately as early false alarms.
  GMDCSA24 fall clips (clip GT): HIT if any alarm anywhere in the clip.
  ADL clips (both): FALSE POSITIVE if any alarm anywhere.

For every miss / false positive, the per-video row records what the rule saw
(max |angle|, max sink, max sink while still, pose coverage, track ids, ...) and a
`why` string naming the failing condition.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from falldet.rules import RuleParams, compute_features  # noqa: E402

TOL_S = 0.5


def frame_alarms(feat: pd.DataFrame) -> pd.DataFrame:
    """One row per alarm frame: which rule(s) fired, and the values on the firing track."""
    a = feat[feat["fall"]]
    if a.empty:
        return a
    return a.sort_values("frame")


def diagnose(feat: pd.DataFrame, lo: int, hi: int, n_window: int, p: RuleParams) -> dict:
    w = feat[(feat.frame >= lo) & (feat.frame <= hi)]
    v = w.dropna(subset=["angle"])
    d = {"cov": round(v.frame.nunique() / max(n_window, 1), 3),
         "ids": int(v.track_id.nunique())}
    if v.empty:
        return d
    absang = v.angle.abs()
    i = absang.idxmax()
    d.update(max_abs_angle=round(float(absang.max()), 1), max_angle_frame=int(v.frame[i]),
             p90_abs_angle=round(float(absang.quantile(0.9)), 1))
    s = v.dropna(subset=["sink"])
    if not s.empty:
        j = s.sink.idxmax()
        d.update(max_sink=round(float(s.sink.max()), 3), max_sink_frame=int(s.frame[j]),
                 motion_at_max_sink=round(float(s.motion[j]), 3) if pd.notna(s.motion[j]) else None)
        st = s[s.still]
        d["max_sink_still"] = round(float(st.sink.max()), 3) if len(st) else None
        d["frac_still"] = round(float(s.still.mean()), 3)
        hi_sink = s[s.sink >= p.sink_thr]
        d["min_motion_when_sink_hi"] = (round(float(hi_sink.motion.min()), 3)
                                        if len(hi_sink) and hi_sink.motion.notna().any() else None)
    return d


def why_miss(d: dict, p: RuleParams) -> str:
    if d.get("cov", 0) < 0.2:
        return f"no pose: person tracked in {d.get('cov', 0):.0%} of post-onset frames"
    parts = [f"max |angle| {d.get('max_abs_angle')}° < {p.angle_thr:g}"]
    ms = d.get("max_sink")
    if ms is None or ms < p.sink_thr:
        parts.append(f"max sink {ms} < {p.sink_thr:g}")
    else:
        parts.append(f"sink reached {ms} but never while still "
                     f"(min motion {d.get('min_motion_when_sink_hi')} > {p.still_tol:g})")
    return "; ".join(parts)


def why_alarm(alarms: pd.DataFrame) -> str:
    first = alarms.iloc[0]
    by = []
    if alarms.fall_angle.any():
        aa = alarms[alarms.fall_angle]
        by.append(f"angle ({len(aa)} fr, |a| max {aa.angle.abs().max():.0f}°, "
                  f"first f{int(aa.frame.iloc[0])} a={aa.angle.iloc[0]:+.0f}°)")
    if alarms.fall_sink.any():
        ss = alarms[alarms.fall_sink]
        by.append(f"sink+still ({len(ss)} fr, sink max {ss.sink.max():.2f}, "
                  f"first f{int(ss.frame.iloc[0])} s={ss.sink.iloc[0]:.2f})")
    return f"first alarm f{int(first.frame)}: " + " + ".join(by)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default="rtmo_t0.3_dedup")
    ap.add_argument("--out", default="results/rules_v1")
    args = ap.parse_args()
    p = RuleParams()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    manifest = list(csv.DictReader(open(ROOT / "data/annotations/manifest.csv")))
    rows, feats = [], []
    for r in manifest:
        vid = r["video_id"]
        base = ROOT / "data/pose" / args.pose / vid
        meta = json.load(open(f"{base}.meta.json"))
        tracks = pd.read_parquet(f"{base}.tracks.parquet") if meta["rows"] else pd.DataFrame()
        fps, n = float(r["fps"]), int(r["n_frames"])
        feat = compute_features(tracks, fps, p)
        feat.insert(0, "video_id", vid)
        feats.append(feat)
        alarms = frame_alarms(feat)
        alarm_frames = sorted(set(alarms.frame)) if len(alarms) else []

        row = {"video_id": vid, "dataset": r["dataset"], "activity": r["activity"],
               "label": r["label"], "n_frames": n, "fps": fps,
               "alarm_frames": len(alarm_frames)}
        tol = int(round(TOL_S * fps))
        if r["label"] == "fall" and r["gt_level"] == "frame":
            onset = int(r["gt_fall_start"])
            hits = alarms[alarms.frame >= onset - tol] if len(alarms) else alarms
            early = alarms[alarms.frame < onset - tol] if len(alarms) else alarms
            row.update(onset=onset, result="TP" if len(hits) else "FN",
                       early_alarm_frames=int(early.frame.nunique()) if len(early) else 0)
            if len(hits):
                row["latency_s"] = round((hits.frame.iloc[0] - onset) / fps, 2)
                row["first_rule"] = "angle" if hits.fall_angle.iloc[0] else "sink+still"
            d = diagnose(feat, onset, n - 1, n - onset, p)
            row.update(d)
            if not len(hits):
                row["why"] = why_miss(d, p)
            if len(early):
                row["early_why"] = why_alarm(early)
        elif r["label"] == "fall":
            row["result"] = "TP" if alarm_frames else "FN"
            d = diagnose(feat, 0, n - 1, n, p)
            row.update(d)
            if alarm_frames:
                row["first_rule"] = "angle" if alarms.fall_angle.iloc[0] else "sink+still"
                row["first_alarm_pos"] = round(alarm_frames[0] / n, 2)
            else:
                row["why"] = why_miss(d, p)
        else:
            row["result"] = "FP" if alarm_frames else "TN"
            row.update(diagnose(feat, 0, n - 1, n, p))
            if alarm_frames:
                row["why"] = why_alarm(alarms)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out / "per_video.csv", index=False)
    pd.concat(feats, ignore_index=True).to_parquet(out / "frame_features.parquet", index=False)

    summary = {"pose": args.pose, "params": p.as_dict(), "tolerance_s": TOL_S}
    for ds in ("caucafall", "gmdcsa24"):
        g = df[df.dataset == ds]
        f, a = g[g.label == "fall"], g[g.label == "adl"]
        s = {"falls": len(f), "TP": int((f.result == "TP").sum()), "FN": int((f.result == "FN").sum()),
             "adl": len(a), "FP": int((a.result == "FP").sum()), "TN": int((a.result == "TN").sum())}
        s["recall"] = round(s["TP"] / max(s["falls"], 1), 3)
        s["fp_rate"] = round(s["FP"] / max(s["adl"], 1), 3)
        tp = f[f.result == "TP"]
        s["first_rule"] = tp.first_rule.value_counts().to_dict()
        if "latency_s" in tp and tp.latency_s.notna().any():
            s["latency_s_median"] = float(tp.latency_s.median())
            s["latency_s_max"] = float(tp.latency_s.max())
        if "early_alarm_frames" in f:
            s["fall_clips_with_early_alarm"] = int((f.early_alarm_frames > 0).sum())
        s["fp_by_activity"] = a[a.result == "FP"].activity.value_counts().to_dict()
        summary[ds] = s
    json.dump(summary, open(out / "summary.json", "w"), indent=2, default=str)

    print(json.dumps({k: v for k, v in summary.items() if k != "params"}, indent=2, default=str))
    pd.set_option("display.width", 250, "display.max_colwidth", 140)
    for res in ("FN", "FP"):
        e = df[df.result == res]
        print(f"\n### {res}: {len(e)}")
        if len(e):
            print(e[["video_id", "cov", "ids", "max_abs_angle", "max_sink", "max_sink_still",
                     "why"]].to_string(index=False))
    early = df[df.get("early_alarm_frames", pd.Series(0, index=df.index)).fillna(0) > 0]
    print(f"\n### fall clips with alarms > {TOL_S}s before GT onset: {len(early)}")
    if len(early):
        print(early[["video_id", "onset", "early_alarm_frames", "early_why"]].to_string(index=False))
    print(f"\nwrote {out}/per_video.csv, frame_features.parquet, summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
