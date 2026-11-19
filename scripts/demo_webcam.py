"""Live fall-detection demo: webcam -> RTMO pose -> ByteTrack -> v2 rules -> overlay.

CPU by default; add --gpu for CUDA (needs the GPU venv):

    .venv-cpu/bin/python scripts/demo_webcam.py                 # webcam 0, CPU
    .venv-gpu/bin/python scripts/demo_webcam.py --gpu           # webcam 0, GPU
    .venv-gpu/bin/python scripts/demo_webcam.py --source 1      # another camera
    .venv-cpu/bin/python scripts/demo_webcam.py --source data/videos/caucafall/caucafall_S01_fall_forward.mp4

Keys: q / Esc quit, r reset tracks and alarms, f toggle mirror.
A webcam is mirrored by default (selfie view; --no-flip to turn off); video files are not.

Rule (results/rules_v2/REPORT.md):
    FALL = lying (torso AND legs tilted >= 50 deg for 1 s)
           AND (hip drop >= 0.25 x body height OR torso rotation >= 110 deg/s)
It needs the WHOLE body in view (hips and knees/ankles), and it was tuned on wall/ceiling
cameras 2-5 m from the person. A laptop webcam at desk height that only sees your upper body
will never fire: put the camera a few metres away so you are visible head to feet.

Timing: with a camera the rule runs on wall-clock time, so it works at whatever frame rate
the pose model manages (CPU RTMO-m ~ 8-12 fps). With a video file it uses the file's
timestamps and processes every frame, which reproduces the batch v2 results exactly.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# The imports below take a few seconds (30 s+ on the first run after install).
print("[demo] importing libraries ...", flush=True)

from falldet.pose import MODELS, detect_and_track, make_backend, make_tracker  # noqa: E402
from falldet.rules import OnlineFallRuleV2, RuleParamsV2  # noqa: E402
from falldet.stitch import TrackStitcher  # noqa: E402
from falldet.runtime import require_cuda_ep  # noqa: E402
from falldet.viz import draw_alarm, draw_person, feature_label, rule_text, text  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

ort.set_default_logger_severity(3)  # hide onnxruntime's CUDA graph-placement info messages


class LatestFrame:
    """Background reader that keeps only the newest camera frame, so a slow model
    never works on a growing backlog (which would add seconds of latency)."""

    def __init__(self, cap: cv2.VideoCapture):
        self.cap, self.frame, self.ok = cap, None, True
        self.running = True
        self.cond = threading.Condition()
        self.seq = 0          # number of frames read so far; callers start from 0
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            ok, f = self.cap.read()
            with self.cond:
                self.ok, self.frame = ok, f
                self.seq += 1
                self.cond.notify_all()
            if not ok:
                break

    def read(self, last_seq: int, timeout: float = 5.0):
        """Block until a frame newer than `last_seq` exists. Returns (ok, frame, seq);
        ok is False if the camera failed or nothing new arrived within `timeout`."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq or not self.ok, timeout)
            fresh = self.seq != last_seq
            return fresh and self.ok and self.frame is not None, self.frame, self.seq

    def stop(self):
        """Stop the reader before the capture is released: releasing it while the thread
        is inside read() races the V4L2 stream setup (VIDIOC_REQBUFS -> EBUSY)."""
        self.running = False
        self.thread.join(timeout=2.0)


def open_source(src: str, width: int, height: int):
    if src.isdigit():
        cap = cv2.VideoCapture(int(src))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        live = True
    else:
        cap = cv2.VideoCapture(src)
        live = False
    if not cap.isOpened():
        sys.exit(f"cannot open video source {src!r}"
                 + (" (is a webcam connected? try --source 1)" if src.isdigit() else ""))
    return cap, live


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index (default 0) or video file / URL")
    ap.add_argument("--gpu", action="store_true", help="run pose on CUDA (default: CPU)")
    ap.add_argument("--model", default="rtmo", choices=MODELS,
                    help="pose model (default rtmo; v2 thresholds were tuned on it)")
    ap.add_argument("--det-score-thr", type=float, default=0.3)
    ap.add_argument("--width", type=int, default=1280, help="requested camera width")
    ap.add_argument("--height", type=int, default=720, help="requested camera height")
    ap.add_argument("--display-width", type=int, default=960)
    ap.add_argument("--fullscreen", action="store_true",
                    help="fill the monitor the window opens on; the overlay is drawn at that "
                         "monitor's size (read back from the window), so --display-width is "
                         "only the starting size")
    ap.add_argument("--no-stitch", action="store_true",
                    help="use raw tracker ids (default: a person who gets a new tracker id "
                         "mid-fall keeps their history, falldet.stitch)")
    ap.add_argument("--no-flip", action="store_true",
                    help="do not mirror the webcam image (webcam is mirrored by default; files never are)")
    ap.add_argument("--alarm-hold", type=float, default=5.0,
                    help="seconds the FALL banner stays up after the last alarm frame")
    ap.add_argument("--record", help="also write the annotated video to this .mp4")
    ap.add_argument("--events", help="append fall events as JSON lines to this file")
    ap.add_argument("--no-display", action="store_true", help="headless (for testing)")
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if args.gpu else "cpu"
    print(f"[demo] loading {args.model} on {device} ...", flush=True)
    net = make_backend(args.model, device, det_score_thr=args.det_score_thr)
    if args.gpu:
        try:
            require_cuda_ep(net)
        except RuntimeError as e:
            sys.exit(f"--gpu requested but CUDA is not available in this environment ({e}).\n"
                     "Run with .venv-gpu/bin/python, or drop --gpu to use the CPU.")

    cap, live = open_source(args.source, args.width, args.height)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker_update, _ = make_tracker(src_fps)
    params = RuleParamsV2()
    rule = OnlineFallRuleV2(params)
    stitcher = None if args.no_stitch else TrackStitcher()
    reader = LatestFrame(cap) if live else None
    flip = live and not args.no_flip
    rtxt = rule_text(asdict(params))

    win = "fall demo v2"
    if not args.no_display:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        if args.fullscreen:
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    disp_w = args.display_width
    writer = None
    events_fh = open(args.events, "a") if args.events else None
    log: deque[str] = deque(maxlen=4)
    falling: dict[int, bool] = {}
    last_alarm_t, last_alarm_msg = None, ""
    t_start = time.monotonic()
    seq, idx = 0, 0
    reason = "end of video" if not live else "camera stopped delivering frames"
    fps_ema, ms_ema = None, None
    t_prev = time.perf_counter()
    print("[demo] running - q/Esc quit, r reset, f mirror", flush=True)

    while True:
        if live:
            ok, frame, seq = reader.read(seq)
            t = time.monotonic() - t_start
        else:
            ok, frame = cap.read()
            t = idx / src_fps
        if not ok:
            if live and idx == 0:
                print(f"[demo] camera {args.source} opened but delivered no frames within 5 s. "
                      "Check that no other app is streaming from it, or try --source 1.",
                      flush=True)
            break
        if flip:
            frame = cv2.flip(frame, 1)

        t0 = time.perf_counter()
        # frame counting, not timestamps: at 10 fps, telling the tracker the wrong rate made
        # no difference in frame-count mode but tripled ID restarts in timestamp mode
        # (results/lowfps/REPORT.md)
        people = detect_and_track(net, tracker_update, frame)
        if stitcher:
            people = stitcher.update(t, people)
        ms = 1000 * (time.perf_counter() - t0)
        rule.tick(t)

        s = disp_w / frame.shape[1]
        img = cv2.resize(frame, (disp_w, int(round(frame.shape[0] * s))),
                         interpolation=cv2.INTER_AREA)
        for p in people:
            tid = p["track_id"]
            r = rule.update(tid, t, p["kp17_xyc"], p["bbox_h"])
            label = None if np.isnan(r["angle"]) else feature_label(
                tid, r["angle"], r["leg"], r["lying"], r["drop"], r["rot"])
            draw_person(img, p["kp17_xyc"], tid, label, r["lying"], scale=s)
            if r["fall"]:
                why = "+".join(k for k in ("drop", "rot") if r[f"fall_{k}"])
                stamp = time.strftime("%H:%M:%S") if live else f"{t:.2f}s f{idx}"
                last_alarm_t, last_alarm_msg = t, f"FALL #{tid} [{why}] {stamp}"
                if not falling.get(tid):            # rising edge -> one event per fall
                    ev = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "t": round(t, 2),
                          "frame": idx, "track_id": tid, "fired_by": why,
                          "drop": None if np.isnan(r["drop"]) else round(float(r["drop"]), 3),
                          "rot": None if np.isnan(r["rot"]) else round(float(r["rot"]), 1)}
                    print(f"[demo] FALL {json.dumps(ev)}", flush=True)
                    log.appendleft(f"{stamp}  FALL #{tid} [{why}]")
                    if events_fh:
                        events_fh.write(json.dumps(ev) + "\n")
                        events_fh.flush()
            falling[tid] = r["fall"]

        top = 0
        if last_alarm_t is not None and t - last_alarm_t <= args.alarm_hold:
            top = draw_alarm(img, last_alarm_msg)
        now = time.perf_counter()
        dt = now - t_prev
        t_prev = now
        fps_ema = 1 / dt if fps_ema is None else 0.9 * fps_ema + 0.1 / max(dt, 1e-6)
        ms_ema = ms if ms_ema is None else 0.9 * ms_ema + 0.1 * ms
        text(img, f"{device.upper()} {args.model}  {fps_ema:4.1f} fps  pose {ms_ema:4.0f} ms  "
                  f"people {len(people)}", (8, top + 20), 0.55, (255, 255, 255), 1)
        text(img, rtxt, (8, top + 38), 0.42, (230, 230, 230), 1)
        for i, line in enumerate(log):
            text(img, line, (8, img.shape[0] - 10 - 20 * i), 0.5, (255, 255, 255), 1, bg=(0, 0, 150))

        if args.record:
            if writer is None:
                fps_out = src_fps if not live else 15.0
                rec_size = (img.shape[1], img.shape[0])
                writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), fps_out,
                                         rec_size)
            # the recording keeps its first size (in fullscreen the overlay size can change)
            writer.write(img if img.shape[1::-1] == rec_size else cv2.resize(img, rec_size))
        if not args.no_display:
            cv2.imshow(win, img)
            if args.fullscreen:
                # draw the next frame at the monitor's size (fit, keeping the aspect ratio),
                # so OpenCV doesn't upscale a small overlay. Some backends can't report it;
                # then OpenCV just stretches the image to the screen.
                try:
                    _, _, sw, sh = cv2.getWindowImageRect(win)
                    if sw > 0 and sh > 0:
                        disp_w = max(320, int(min(sw, sh * frame.shape[1] / frame.shape[0])))
                except cv2.error:
                    pass
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                reason = f"key {'Esc' if key == 27 else 'q'} pressed"
                break
            if key == ord("r"):
                rule.reset()
                tracker_update, _ = make_tracker(src_fps)
                if stitcher:
                    stitcher.reset()
                falling.clear()
                log.clear()
                last_alarm_t = None
            if key == ord("f"):
                flip = not flip
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                reason = "window closed"
                break
        idx += 1
        if args.max_frames and idx >= args.max_frames:
            reason = "--max-frames reached"
            break

    if reader:
        reader.stop()
    cap.release()
    if writer:
        writer.release()
    if events_fh:
        events_fh.close()
    cv2.destroyAllWindows()
    print(f"[demo] stopped ({reason}): {idx} frames, avg pose {ms_ema or 0:.0f} ms/frame",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
