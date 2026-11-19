"""Render v2 result videos in the style of sample_result/.

    .venv-gpu/bin/python scripts/render_v2_videos.py            # all 260 videos
    .venv-gpu/bin/python scripts/render_v2_videos.py --only caucafall_S03_fall_left

Output: results/rules_v2/videos/<TP|FN|FP|TN>/<video_id>.mp4 (640 px wide, H.264)
        results/rules_v2/videos/index.html  (open locally in a browser)

Overlay:
  header    f<frame> <t>s  GT fall f<a>-<b> | GT fall (clip label) | GT no fall   + result
  2nd line  the rule with its thresholds
  skeleton  per track colour; left limbs blue, right limbs green, torso box orange,
            white hip-mid -> shoulder-mid line; label  #id a<torso> l<leg> d<drop> r<rot>
            (label turns red-filled while that track is "lying")
  banner    red bar + border from the first alarm on: FALL #id [drop|rot|drop+rot] f<frame>
  timeline  grey bar, orange = GT fall frames, red ticks = alarm frames, white = now
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from falldet.stitch import stitch_tracks  # noqa: E402
from falldet.viz import RED, WHITE, draw_alarm, draw_person, feature_label, rule_text, text  # noqa: E402
RES = ROOT / "results/rules_v2"
OUT = RES / "videos"
W_OUT = 640

def render(job: dict) -> str:
    r, res = job["row"], job["res"]
    vid = r["video_id"]
    n, fps = int(r["n_frames"]), float(r["fps"])
    tracks = pd.read_parquet(ROOT / f"data/pose/{job['pose']}/{vid}.tracks.parquet") \
        if job["has_rows"] else pd.DataFrame()
    if job["stitch"] and len(tracks):
        tracks = stitch_tracks(tracks, fps)   # same ids as the scored features
    feats = pd.read_parquet(job["feat_path"]) if job["feat_path"] else pd.DataFrame()
    kp_by = {}
    if len(tracks):
        for row in tracks.itertuples():
            kp_by.setdefault(row.frame, []).append(
                (row.track_id, np.asarray(row.kp17_xyc, np.float32).reshape(17, 3)))
    f_by = {(int(x.frame), int(x.track_id)): x for x in feats.itertuples()} if len(feats) else {}
    alarms = feats[feats["fall"]].sort_values("frame") if len(feats) else feats
    alarm_frames = set(alarms["frame"].astype(int)) if len(alarms) else set()

    gt_a = int(r["gt_fall_start"]) if r["gt_fall_start"] else None
    gt_b = int(r["gt_fall_end"]) if r["gt_fall_end"] else None
    if gt_a is not None:
        gt_txt = f"GT fall f{gt_a}-{gt_b}"
    elif r["label"] == "fall":
        gt_txt = "GT fall (clip label)"
    else:
        gt_txt = "GT no fall"
    rule_txt = rule_text(job["params"])

    cap = cv2.VideoCapture(str(ROOT / r["path"]))
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    s = W_OUT / w0
    h_img = int(round(h0 * s / 2) * 2)
    tl_h = 12
    H = h_img + tl_h
    dst = OUT / res / f"{vid}.mp4"
    dst.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W_OUT}x{H}", "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-crf", "23",
         "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)],
        stdin=subprocess.PIPE)

    # static timeline
    tl = np.full((tl_h, W_OUT, 3), 60, np.uint8)
    x_of = lambda f: int(round(f / max(n - 1, 1) * (W_OUT - 1)))  # noqa: E731
    if gt_a is not None:
        tl[2:-2, x_of(gt_a):x_of(gt_b) + 1] = (0, 140, 255)
    elif r["label"] == "fall":
        tl[2:-2, :] = (0, 90, 160)  # clip-level fall: dim orange over the whole clip
    for f in alarm_frames:
        x = x_of(f)
        tl[:, max(0, x - 1):x + 2] = RED

    last_alarm = None
    fi = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        img = cv2.resize(frame, (W_OUT, h_img), interpolation=cv2.INTER_AREA)

        for tid, kp in kp_by.get(fi, []):
            ft = f_by.get((fi, tid))
            label, lying = None, False
            if ft is not None and not np.isnan(ft.angle):
                lying = bool(ft.lying)
                label = feature_label(tid, ft.angle, ft.leg, lying, ft.drop, ft.rot)
            draw_person(img, kp, tid, label, lying, scale=s)

        if fi in alarm_frames:
            a = alarms[alarms.frame == fi].iloc[0]
            why = "+".join(k for k, c in (("drop", "fall_drop"), ("rot", "fall_rot")) if a[c])
            last_alarm = f"FALL #{int(a.track_id)} [{why}] f{fi}"

        top = draw_alarm(img, last_alarm) if last_alarm else 0
        text(img, f"f{fi} {fi / fps:.2f}s  {gt_txt}", (8, top + 20), 0.55, WHITE, 1)
        text(img, rule_txt, (8, top + 38), 0.42, (230, 230, 230), 1)
        rcol = {"TP": (0, 200, 0), "TN": (0, 200, 0), "FN": (0, 0, 230), "FP": (0, 0, 230)}[res]
        text(img, f"v2 {res}", (W_OUT - 70, top + 20), 0.55, WHITE, 1, bg=rcol)

        bar = tl.copy()
        cx = x_of(fi)
        bar[:, max(0, cx - 1):cx + 2] = WHITE
        ff.stdin.write(np.vstack([img, bar]).tobytes())
        fi += 1
    cap.release()
    ff.stdin.close()
    ff.wait()
    return f"{res}/{vid}.mp4"


def write_index(rows: list[dict], summary: dict) -> None:
    order = ["FN", "FP", "TP", "TN"]
    titles = {"FN": "Missed falls (FN)", "FP": "False alarms (FP)", "TP": "Detected falls (TP)",
              "TN": "Correctly quiet (TN)"}
    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Fall rules v2 videos</title>",
             "<style>body{font-family:system-ui,sans-serif;margin:16px;background:#111;color:#eee}"
             "h2{margin-top:28px}.g{display:flex;flex-wrap:wrap;gap:12px}"
             ".c{width:400px;background:#1d1d1d;padding:6px;border-radius:6px}"
             "video{width:100%}.m{font-size:12px;color:#bbb;margin-top:4px}"
             ".id{font-weight:600;font-size:13px}nav a{color:#9cf;margin-right:12px}</style></head><body>",
             "<h1>Fall rules v2: result videos</h1><p>Rule: lying (torso &amp; leg tilt ≥ "
             f"{summary['params']['lie_thr']:g}° for {summary['params']['lie_s']:g} s) AND "
             f"(hip drop ≥ {summary['params']['drop_thr']:g} OR rotation ≥ "
             f"{summary['params']['rot_thr']:g}°/s). Pose: {summary['pose']}.</p><nav>"]
    for k in order:
        parts.append(f"<a href='#{k}'>{titles[k]} ({sum(r['result'] == k for r in rows)})</a>")
    parts.append("</nav>")
    for k in order:
        parts.append(f"<h2 id='{k}'>{titles[k]}</h2><div class='g'>")
        for r in sorted((r for r in rows if r["result"] == k), key=lambda r: r["video_id"]):
            meta = r.get("why") or r.get("description") or ""
            vals = (f"lying {r['lying_frames']} fr · max min-tilt {r['max_tilt_min']}° · "
                    f"drop {r['max_drop_lying']} · rot {r['max_rot_lying']}°/s")
            parts.append(
                f"<div class='c'><div class='id'>{html.escape(r['video_id'])} "
                f"<span style='color:#888'>[{r['split']}]</span></div>"
                f"<video controls preload='none' src='{k}/{html.escape(r['video_id'])}.mp4'></video>"
                f"<div class='m'>{html.escape(vals)}</div>"
                f"<div class='m'>{html.escape(str(meta))}</div></div>")
        parts.append("</div>")
    parts.append("</body></html>")
    (OUT / "index.html").write_text("\n".join(parts))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    summary = json.load(open(RES / "summary.json"))
    params, pose = summary["params"], summary["pose"]
    per = pd.read_csv(RES / "per_video.csv", keep_default_na=False)
    res_of = dict(zip(per.video_id, per.result))
    man = list(csv.DictReader(open(ROOT / "data/annotations/manifest.csv")))

    # split per-video features into small files so workers don't each load the whole table
    feats = pd.read_parquet(RES / "frame_features.parquet")
    tmp = OUT / "_feat"
    tmp.mkdir(parents=True, exist_ok=True)
    feat_paths = {}
    for vid, g in feats.groupby("video_id"):
        pth = tmp / f"{vid}.parquet"
        g.to_parquet(pth, index=False)
        feat_paths[vid] = str(pth)

    jobs = []
    for r in man:
        if args.only and r["video_id"] not in args.only:
            continue
        meta = json.load(open(ROOT / f"data/pose/{pose}/{r['video_id']}.meta.json"))
        jobs.append({"row": r, "res": res_of[r["video_id"]], "pose": pose, "params": params,
                     "has_rows": meta["rows"] > 0, "feat_path": feat_paths.get(r["video_id"]),
                     "stitch": summary.get("stitch", False)})
    done = 0
    with ProcessPoolExecutor(args.workers) as ex:
        for out in ex.map(render, jobs):
            done += 1
            if done % 20 == 0 or done == len(jobs):
                print(f"[{done}/{len(jobs)}] {out}", flush=True)
    for pth in tmp.glob("*.parquet"):
        pth.unlink()
    tmp.rmdir()
    write_index(per.to_dict(orient="records"), summary)
    print(f"wrote {OUT}/index.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
