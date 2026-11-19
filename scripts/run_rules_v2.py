"""v2 rules: lying AND (hip drop OR fast torso rotation). Tune on one split, test on another.

    .venv-gpu/bin/python scripts/run_rules_v2.py --pose rtmo_t0.3_dedup --out results/rules_v2

Splits:
    tune  = CAUCAFall (all) + GMDCSA24 subjects 1-2
    test  = GMDCSA24 subjects 3-4   (thresholds are not fitted here, but the v2 features were
                                     designed after inspecting per-clip values from all GMDCSA24
                                     subjects, so test numbers are optimistic, not held-out)

drop_thr and rot_thr are grid-searched on `tune` (maximise recall - FP rate, ties -> higher
recall). lie_thr / lie_s were fixed from the v1 analysis on CAUCAFall.
Scoring is the same as scripts/run_rules.py (hit window [onset - 0.5 s, end] for
frame-level GT, any alarm for clip-level GT, any alarm on ADL = false positive).
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from falldet.rules import RuleParamsV2, compute_features_v2  # noqa: E402

TOL_S = 0.5
DROP_GRID = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, np.inf]
ROT_GRID = [80, 90, 100, 110, 120, 140, 160, np.inf]


def split_of(r: dict) -> str:
    if r["dataset"] == "gmdcsa24" and int(r["subject"]) >= 3:
        return "test"
    return "tune"


def score(man: list[dict], feats: dict[str, pd.DataFrame], d: float, rot: float) -> pd.DataFrame:
    rows = []
    for r in man:
        f = feats[r["video_id"]]
        fd = f["lying"] & (f["drop"].fillna(-1) >= d)
        fr = f["lying"] & (f["rot"].fillna(-1) >= rot)
        al = f[fd | fr].assign(by_drop=fd[fd | fr], by_rot=fr[fd | fr]).sort_values("frame")
        fps, n = float(r["fps"]), int(r["n_frames"])
        row = {"video_id": r["video_id"], "dataset": r["dataset"], "split": split_of(r),
               "activity": r["activity"], "label": r["label"], "description": r["description"][:90]}
        if r["label"] == "fall" and r["gt_level"] == "frame":
            onset = int(r["gt_fall_start"])
            hits = al[al.frame >= onset - int(round(TOL_S * fps))]
            row["result"] = "TP" if len(hits) else "FN"
            row["early_alarm"] = int(len(al) > len(hits))
            if len(hits):
                row["latency_s"] = round((hits.frame.iloc[0] - onset) / fps, 2)
        elif r["label"] == "fall":
            row["result"] = "TP" if len(al) else "FN"
        else:
            row["result"] = "FP" if len(al) else "TN"
        if len(al):
            row["first_alarm"] = int(al.frame.iloc[0])
            row["fired_by"] = "+".join(k for k, c in (("drop", "by_drop"), ("rot", "by_rot"))
                                       if al[c].iloc[0])
        # what the rule saw
        ly = f[f["lying"]]
        row["lying_frames"] = int(f["lying"].sum())
        row["max_tilt_min"] = round(float(np.nanmax(np.minimum(f["angle"].abs(), f["leg"].abs())))
                                    if len(f) and f["angle"].notna().any() else np.nan, 1)
        row["max_drop_lying"] = round(float(ly["drop"].max()), 3) if len(ly) else np.nan
        row["max_rot_lying"] = round(float(ly["rot"].max()), 1) if len(ly) else np.nan
        row["pose_cov"] = round(f["frame"].nunique() / n, 3) if len(f) else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def metrics(df: pd.DataFrame) -> dict:
    f, a = df[df.label == "fall"], df[df.label == "adl"]
    tp, fp = int((f.result == "TP").sum()), int((a.result == "FP").sum())
    return {"falls": len(f), "TP": tp, "recall": round(tp / max(len(f), 1), 3),
            "adl": len(a), "FP": fp, "fp_rate": round(fp / max(len(a), 1), 3)}


def why(row, p: RuleParamsV2, d: float, rot: float) -> str:
    if row.result == "FN":
        if row.pose_cov < 0.2:
            return f"no pose ({row.pose_cov:.0%} of frames)"
        if row.lying_frames == 0:
            return (f"never lying: max min(|torso|,|leg|) tilt {row.max_tilt_min}° "
                    f"(need >= {p.lie_thr:g}° for {p.lie_s:g} s)")
        return (f"lying {row.lying_frames} fr but drop {row.max_drop_lying} < {d:g} and "
                f"rot {row.max_rot_lying}°/s < {rot:g}")
    if row.result == "FP":
        return (f"lying {row.lying_frames} fr; fired by {row.fired_by} "
                f"(max drop {row.max_drop_lying}, max rot {row.max_rot_lying}°/s)")
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", default="rtmo_t0.3_dedup")
    ap.add_argument("--out", default="results/rules_v2")
    args = ap.parse_args()
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    p = RuleParamsV2()

    man = list(csv.DictReader(open(ROOT / "data/annotations/manifest.csv")))
    feats = {}
    for r in man:
        base = ROOT / "data/pose" / args.pose / r["video_id"]
        meta = json.load(open(f"{base}.meta.json"))
        t = pd.read_parquet(f"{base}.tracks.parquet") if meta["rows"] else pd.DataFrame()
        feats[r["video_id"]] = compute_features_v2(t, float(r["fps"]), p)

    tune = [r for r in man if split_of(r) == "tune"]
    grid = []
    for d, rot in itertools.product(DROP_GRID, ROT_GRID):
        m = metrics(score(tune, feats, d, rot))
        grid.append({"drop_thr": d, "rot_thr": rot, **m, "j": m["recall"] - m["fp_rate"]})
    g = pd.DataFrame(grid).sort_values(["j", "recall"], ascending=False)
    g.to_csv(out / "grid_tune.csv", index=False)
    best = g.iloc[0]
    d, rot = float(best.drop_thr), float(best.rot_thr)
    p = replace(p, drop_thr=d, rot_thr=rot)
    print("top of tune grid:\n", g.head(6).to_string(index=False))

    df = score(man, feats, d, rot)
    df["why"] = [why(r, p, d, rot) for r in df.itertuples()]
    df.to_csv(out / "per_video.csv", index=False)
    pd.concat([f.assign(video_id=v) for v, f in feats.items() if len(f)],
              ignore_index=True).to_parquet(out / "frame_features.parquet", index=False)

    summary = {"pose": args.pose, "params": p.as_dict()}
    for name, sel in [("tune/caucafall", (df.split == "tune") & (df.dataset == "caucafall")),
                      ("tune/gmdcsa24_S1-2", (df.split == "tune") & (df.dataset == "gmdcsa24")),
                      ("test/gmdcsa24_S3-4", df.split == "test")]:
        s = df[sel]
        summary[name] = metrics(s)
        summary[name]["fired_by"] = s[s.result == "TP"].fired_by.value_counts().to_dict()
        if "latency_s" in s and s.latency_s.notna().any():
            summary[name]["latency_s_median"] = float(s.latency_s.median())
            summary[name]["early_alarm_clips"] = int(s.early_alarm.fillna(0).sum())
        summary[name]["fp_by_activity"] = s[s.result == "FP"].activity.value_counts().to_dict()
    json.dump(summary, open(out / "summary.json", "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in summary.items() if k != "params"}, indent=2, default=float))
    print("params:", p.as_dict())

    pd.set_option("display.width", 260, "display.max_colwidth", 110)
    for res in ("FN", "FP"):
        e = df[df.result == res]
        print(f"\n### {res}: {len(e)}")
        if len(e):
            print(e[["video_id", "split", "max_tilt_min", "max_drop_lying", "max_rot_lying",
                     "why", "description"]].to_string(index=False))
    print(f"\nwrote {out}/per_video.csv, grid_tune.csv, frame_features.parquet, summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
