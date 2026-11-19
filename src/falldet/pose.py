"""Video -> per-frame tracked COCO-17 keypoints (`<name>.tracks.parquet`).

Adapted from shoplifting-detection `sld.pose.extract` (same author). Output schema,
one row per (frame, tracked person):

    frame         int32     0-based frame index in the source video
    track_id      int32     stable across frames, from ByteTrack
    bbox_x/y/w/h  float32   person box in NATIVE video pixels
    det_conf      float32   detector confidence for that box
    kp17_xyc      list<float32>[51]  COCO-17 as (x, y, conf) x 17, native pixels
    kp_mean_conf  float32   mean of the 17 confidences

Default model is RTMO-m (one-stage) with keypoint-box dedup, decided by
scripts/compare_pose.py on all 260 test videos (data/pose/compare.json, 2026-11-19):
the YOLOX detector of the top-down path misses people lying on the floor (CAUCAFall
falls: 78% of lying frames covered vs 99.9% for RTMO; 12/50 clips under 50% vs 0/50),
and RTMO is ~1.8x faster. The top-down models stay available via --model.
Unlike shoplifting-detection, no track filters are applied: a short track can be the fall.

Run:
    .venv-gpu/bin/python -m falldet.pose data/videos/caucafall/*.mp4 --out-dir data/pose/rtmo
"""

from __future__ import annotations

# Must run before rtmlib imports onnxruntime.
from falldet.runtime import preload_cuda_libs, require_cuda_ep

preload_cuda_libs()

import argparse
import json
import os
import time
import warnings

# supervision 0.32 emits a FutureWarning per decorated function at import
# ("`template_mgs` is deprecated"): harmless and noisy.
warnings.filterwarnings("ignore", message=".*template_mgs.*", category=FutureWarning)

import cv2
import numpy as np
import pandas as pd
import rtmlib
import supervision as sv
from rtmlib.tools.object_detection.post_processings import multiclass_nms

COCO17_NAMES = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)

RTMO_M = ("https://download.openmmlab.com/mmpose/v1/projects/rtmo/onnx_sdk/"
          "rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.zip")


class ScoringYOLOX(rtmlib.YOLOX):
    """YOLOX that also returns detection scores (rtmlib drops them; ByteTrack's
    two-pass association needs them)."""

    def postprocess(self, outputs: np.ndarray, ratio: float):
        if outputs.shape[-1] == 5:  # NMS baked into the graph: [x1, y1, x2, y2, score]
            boxes = outputs[0, :, :4] / ratio
            scores = outputs[0, :, 4]
            keep = scores > self.score_thr
            return boxes[keep].astype(np.float32), scores[keep].astype(np.float32)

        strides = (8, 16, 32)
        grids, expanded = [], []
        for stride in strides:
            h = self.model_input_size[0] // stride
            w = self.model_input_size[1] // stride
            xv, yv = np.meshgrid(np.arange(w), np.arange(h))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            expanded.append(np.full((*grid.shape[:2], 1), stride))
        grids = np.concatenate(grids, 1)
        expanded = np.concatenate(expanded, 1)

        outputs = outputs.copy()
        outputs[..., :2] = (outputs[..., :2] + grids) * expanded
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded

        pred = outputs[0]
        cxcywh = pred[:, :4]
        cls_scores = pred[:, 4:5] * pred[:, 5:]
        xyxy = np.empty_like(cxcywh)
        xyxy[:, 0] = cxcywh[:, 0] - cxcywh[:, 2] / 2.0
        xyxy[:, 1] = cxcywh[:, 1] - cxcywh[:, 3] / 2.0
        xyxy[:, 2] = cxcywh[:, 0] + cxcywh[:, 2] / 2.0
        xyxy[:, 3] = cxcywh[:, 1] + cxcywh[:, 3] / 2.0
        xyxy /= ratio

        dets, _ = multiclass_nms(xyxy, cls_scores, nms_thr=self.nms_thr, score_thr=self.score_thr)
        if dets is None:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        boxes, scores, cls = dets[:, :4], dets[:, 4], dets[:, 5].astype(int)
        keep = cls == 0
        return boxes[keep].astype(np.float32), scores[keep].astype(np.float32)


class TopDownBackend:
    """YOLOX detector + RTMPose. `balanced` = YOLOX-m + RTMPose-m,
    `lightweight` = YOLOX-tiny + RTMPose-s, `performance` = YOLOX-x + RTMPose-x."""

    def __init__(self, mode: str = "balanced", device: str = "cuda",
                 backend: str = "onnxruntime", det_score_thr: float = 0.3):
        cfg = rtmlib.Body.MODE[mode]
        self.det = ScoringYOLOX(cfg["det"], model_input_size=cfg["det_input_size"],
                                mode="human", score_thr=det_score_thr,
                                backend=backend, device=device)
        self.pose = rtmlib.RTMPose(cfg["pose"], model_input_size=cfg["pose_input_size"],
                                   backend=backend, device=device)

    def run(self, img: np.ndarray):
        xyxy, scores = self.det(img)
        if len(xyxy) == 0:
            return xyxy, scores, np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32)
        kp, kpc = self.pose(img, bboxes=list(xyxy))
        return xyxy, scores, np.asarray(kp, np.float32), np.asarray(kpc, np.float32)


class OneStageBackend:
    """RTMO-m. Boxes are derived from confident keypoints (rtmlib discards RTMO's own)."""

    def __init__(self, device: str = "cuda", backend: str = "onnxruntime",
                 det_score_thr: float = 0.3, kpt_thr: float = 0.3, dedup_iou: float = 0.6):
        self.pose = rtmlib.RTMO(RTMO_M, model_input_size=(640, 640), score_thr=det_score_thr,
                                backend=backend, device=device)
        self.kpt_thr = kpt_thr
        self.dedup_iou = dedup_iou

    @staticmethod
    def _dedup(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
        """Greedy NMS on keypoint-derived boxes.

        rtmlib already runs NMS (IoU 0.45) on RTMO's own boxes, but on GMDCSA24 RTMO
        still emitted two near-identical skeletons for one seated person in ~38% of
        ADL frames (keypoint-box IoU 0.8-0.9). Each duplicate becomes a second track.
        """
        order = np.argsort(-scores)
        keep: list[int] = []
        for i in order:
            ok = True
            for j in keep:
                x1 = max(boxes[i, 0], boxes[j, 0]); y1 = max(boxes[i, 1], boxes[j, 1])
                x2 = min(boxes[i, 2], boxes[j, 2]); y2 = min(boxes[i, 3], boxes[j, 3])
                inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                ai = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
                aj = (boxes[j, 2] - boxes[j, 0]) * (boxes[j, 3] - boxes[j, 1])
                if inter / max(ai + aj - inter, 1e-6) >= thr:
                    ok = False
                    break
            if ok:
                keep.append(int(i))
        return sorted(keep)

    def run(self, img: np.ndarray):
        kp, kpc = self.pose(img)
        kp = np.asarray(kp, np.float32).reshape(-1, 17, 2)
        kpc = np.asarray(kpc, np.float32).reshape(-1, 17)
        boxes, scores, keep = [], [], []
        for i in range(len(kp)):
            vis = kpc[i] >= self.kpt_thr
            if vis.sum() < 2:
                continue
            pts = kp[i][vis]
            boxes.append((pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()))
            scores.append(float(kpc[i][vis].mean()))
            keep.append(i)
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        scores = np.asarray(scores, np.float32)
        kp, kpc = kp[keep], kpc[keep]
        if self.dedup_iou and len(boxes) > 1:
            k = self._dedup(boxes, scores, self.dedup_iou)
            boxes, scores, kp, kpc = boxes[k], scores[k], kp[k], kpc[k]
        return boxes, scores, kp, kpc


MODELS = ("rtmo", "balanced", "lightweight", "performance")


def make_backend(model: str = "rtmo", device: str = "cuda", backend: str = "onnxruntime",
                 det_score_thr: float = 0.3):
    preload_cuda_libs()
    if model == "rtmo":
        return OneStageBackend(device=device, backend=backend, det_score_thr=det_score_thr)
    if model in ("balanced", "lightweight", "performance"):
        return TopDownBackend(mode=model, device=device, backend=backend,
                              det_score_thr=det_score_thr)
    raise ValueError(f"unknown model {model!r}")


def make_tracker(fps: float, activation: float = 0.35, min_frames: int = 3):
    """ByteTrack. Returns (update, name); call update(detections, timestamp=None).

    Without timestamps the tracker counts frames: a lost track survives `fps` frames and
    the motion model steps once per frame, so `fps` must be the rate frames are actually
    processed at. With timestamps (seconds, increasing) it uses real elapsed time
    instead: a lost track survives 1 s and the motion model steps by the actual gap,
    whatever the frame rate.

    Prefer frame counting. On the test set at 10 fps (results/lowfps/REPORT.md), frame
    counting gave identical tracks whether the tracker was told 10 or 30 fps (the
    lost-track buffer length did not matter), while timestamps with fps=30 stretched every
    motion prediction to 3 frame-steps and raised ID restarts in fall clips from 11 to 38.
    Timestamps only behave when `fps` matches the real processing rate."""
    try:
        from trackers import ByteTrackTracker

        t = ByteTrackTracker(frame_rate=fps, track_activation_threshold=activation,
                             minimum_consecutive_frames=min_frames)
        return (lambda dets, timestamp=None: t.update(dets, timestamp=timestamp),
                "trackers.ByteTrackTracker")
    except ImportError:
        t = sv.ByteTrack(frame_rate=round(fps), track_activation_threshold=activation,
                         minimum_consecutive_frames=min_frames)
        # supervision's ByteTrack has no timestamp support: frame counting only
        return (lambda dets, timestamp=None: t.update_with_detections(dets),
                "supervision.ByteTrack")


def detect_and_track(net, tracker_update, frame: np.ndarray, upscale: float = 1.0,
                     timestamp: float | None = None) -> list[dict]:
    """Pose + tracking for one frame. Returns confirmed tracks only, as dicts with
    track_id, bbox_x/y/w/h (clamped to the frame), det_conf, kp17_xyc (17x3 float32,
    native pixels) and kp_mean_conf. Shared by extract() and the live demo.
    `timestamp` (seconds) switches the tracker to elapsed-time mode, see make_tracker."""
    height, width = frame.shape[:2]
    img = frame if upscale == 1.0 else cv2.resize(
        frame, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    xyxy, dscore, kp, kpc = net.run(img)
    if upscale != 1.0 and len(xyxy):
        xyxy, kp = xyxy / upscale, kp / upscale

    dets = sv.Detections(
        xyxy=xyxy.astype(np.float32), confidence=dscore.astype(np.float32),
        class_id=np.zeros(len(xyxy), int), data={"idx": np.arange(len(xyxy))},
    ) if len(xyxy) else sv.Detections.empty()
    tracked = tracker_update(dets, timestamp=timestamp)

    out = []
    for j in range(len(tracked)):
        tid = int(tracked.tracker_id[j])
        if tid < 0:
            continue  # not yet confirmed
        src = int(tracked.data["idx"][j])
        x1, y1, x2, y2 = tracked.xyxy[j]
        x1, x2 = (min(max(v, 0.0), width) for v in (x1, x2))
        y1, y2 = (min(max(v, 0.0), height) for v in (y1, y2))
        out.append({
            "track_id": tid,
            "bbox_x": float(x1), "bbox_y": float(y1),
            "bbox_w": float(x2 - x1), "bbox_h": float(y2 - y1),
            "det_conf": float(tracked.confidence[j]),
            "kp17_xyc": np.concatenate([kp[src], kpc[src][:, None]], axis=1).astype(np.float32),
            "kp_mean_conf": float(kpc[src].mean()),
        })
    return out


def extract(video: str, net, *, model: str, upscale: float = 1.0, max_frames: int = 0,
            timestamps: bool = False, tracker_fps: float | None = None
            ) -> tuple[pd.DataFrame, dict]:
    """Run pose + tracking over one video. `net` is reused across videos;
    the tracker is fresh per video so ids never leak between clips.

    timestamps   feed the tracker frame index / fps as timestamps (elapsed-time mode)
    tracker_fps  tell the tracker this frame rate instead of the file's (to reproduce a
                 camera that reports one rate but delivers another)"""
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tracker_update, tracker_name = make_tracker(tracker_fps or fps)

    rows: list[dict] = []
    idx = 0
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        ts = idx / fps if timestamps else None
        for d in detect_and_track(net, tracker_update, frame, upscale, timestamp=ts):
            d["kp17_xyc"] = d["kp17_xyc"].reshape(-1).tolist()
            rows.append({"frame": idx, **d})
        idx += 1
        if max_frames and idx >= max_frames:
            break
    cap.release()

    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.astype({"frame": "int32", "track_id": "int32", "bbox_x": "float32",
                        "bbox_y": "float32", "bbox_w": "float32", "bbox_h": "float32",
                        "det_conf": "float32", "kp_mean_conf": "float32"})
    meta = {
        "video": os.path.abspath(video), "width": width, "height": height, "fps": fps,
        "frames_in_file": n_frames, "frames_processed": idx,
        "model": model, "upscale": upscale, "tracker": tracker_name,
        "tracker_fps": tracker_fps or fps, "tracker_timestamps": timestamps,
        "keypoint_format": "coco17", "keypoint_names": list(COCO17_NAMES),
        "coordinate_space": "native_video_pixels",
        "rows": len(df), "tracks": int(df["track_id"].nunique()) if len(df) else 0,
        "seconds": round(elapsed, 2), "ms_per_frame": round(1000 * elapsed / max(idx, 1), 2),
    }
    return df, meta


def main() -> int:
    ap = argparse.ArgumentParser(description="RTMPose + ByteTrack keypoint extraction")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="rtmo", choices=MODELS)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--upscale", type=float, default=1.0,
                    help="resize factor before inference (e.g. 2.0 for 320x240 video)")
    ap.add_argument("--det-score-thr", type=float, default=0.3)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--tracker-timestamps", action="store_true",
                    help="drive the tracker by timestamps (frame / fps) instead of frame counts")
    ap.add_argument("--tracker-fps", type=float,
                    help="frame rate to tell the tracker (default: the file's)")
    args = ap.parse_args()

    net = make_backend(args.model, args.device, det_score_thr=args.det_score_thr)
    if args.device == "cuda":
        require_cuda_ep(net)
    os.makedirs(args.out_dir, exist_ok=True)

    for i, video in enumerate(args.videos, 1):
        name = os.path.splitext(os.path.basename(video))[0]
        out = os.path.join(args.out_dir, name)
        if not args.overwrite and os.path.exists(f"{out}.meta.json"):
            print(f"[{i}/{len(args.videos)}] skip {name} (exists)", flush=True)
            continue
        df, meta = extract(video, net, model=args.model, upscale=args.upscale,
                           max_frames=args.max_frames, timestamps=args.tracker_timestamps,
                           tracker_fps=args.tracker_fps)
        df.to_parquet(f"{out}.tracks.parquet", index=False)
        with open(f"{out}.meta.json", "w") as fh:
            json.dump(meta, fh, indent=2)
        print(f"[{i}/{len(args.videos)}] {name}: {meta['frames_processed']} frames, "
              f"{meta['ms_per_frame']} ms/frame, {meta['tracks']} tracks", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
