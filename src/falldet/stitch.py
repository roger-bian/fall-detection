"""Track stitching: keep one person id when the tracker loses a person and picks them up
again under a new id.

Why: during a fall the box changes from tall (standing) to wide (lying) within a few
frames and the pose often drops out briefly. ByteTrack then fails to match the returning
box to the predicted one and starts a new id, even after a gap of 0-0.3 s. The fall rule
keeps history per id, so the new id never sees the hip drop or the rotation that
happened under the old one, and the fall is missed (results/rules_v2/REPORT.md,
results/lowfps/REPORT.md).

Rule: a new tracker id inherits the stable id of a track that
  * is not visible in the current frame (never steal from a visible person),
  * was last seen at most `max_gap_s` ago,
  * lasted at least `min_old_s` (skips 1-2 frame ghost detections), and
  * ended within `max_dist` x its last box size (max of width, height), centre to centre.
The nearest such track wins; each old track can be inherited once per frame.

Measured on the test set (data/pose/rtmo_t0.3_dedup and rtmo_10fps): restarts have gap
median 0.1-0.2 s (95% < 1.5 s) and move median 0.1 box sizes (falls up to 1.9).

With several people near each other a returning person can inherit the wrong history.
The gates keep that to people who appear where someone vanished moments ago.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class TrackStitcher:
    def __init__(self, max_gap_s: float = 1.5, max_dist: float = 2.0, min_old_s: float = 0.5):
        self.max_gap_s, self.max_dist, self.min_old_s = max_gap_s, max_dist, min_old_s
        self.reset()

    def reset(self) -> None:
        self.raw2stable: dict[int, int] = {}
        self.tracks: dict[int, dict] = {}   # stable id -> first_t, last_t, cx, cy, size
        self.next_id = 0
        self.n_stitched = 0

    def _new_id(self) -> int:
        sid = self.next_id
        self.next_id += 1
        return sid

    def update(self, t: float, people: list[dict]) -> list[dict]:
        """Rewrite each person's `track_id` to a stable id (the tracker's id is kept as
        `raw_track_id`). `people` are dicts with track_id and bbox_x/y/w/h, as returned by
        falldet.pose.detect_and_track. Call once per processed frame, t in seconds."""
        claimed: set[int] = set()
        known = [p for p in people if p["track_id"] in self.raw2stable]
        new = [p for p in people if p["track_id"] not in self.raw2stable]
        for p in known:
            sid = self.raw2stable[p["track_id"]]
            if sid in claimed:                       # two raw ids on one stable id: split
                sid = self._new_id()
                self.raw2stable[p["track_id"]] = sid
            claimed.add(sid)
            p["_sid"] = sid
        # new raw ids: nearest eligible vanished track, greedy over all pairs
        pairs = []
        for i, p in enumerate(new):
            cx, cy = p["bbox_x"] + p["bbox_w"] / 2, p["bbox_y"] + p["bbox_h"] / 2
            for sid, s in self.tracks.items():
                if sid in claimed or s["last_t"] >= t:
                    continue
                if t - s["last_t"] > self.max_gap_s or s["last_t"] - s["first_t"] < self.min_old_s:
                    continue
                d = np.hypot(cx - s["cx"], cy - s["cy"]) / max(s["size"], 1.0)
                if d <= self.max_dist:
                    pairs.append((d, i, sid))
        taken: set[int] = set()
        for d, i, sid in sorted(pairs):
            if i in taken or sid in claimed:
                continue
            new[i]["_sid"] = sid
            taken.add(i)
            claimed.add(sid)
            self.n_stitched += 1
        for p in new:
            if "_sid" not in p:
                p["_sid"] = self._new_id()
                claimed.add(p["_sid"])
            self.raw2stable[p["track_id"]] = p["_sid"]

        for p in people:
            sid = p.pop("_sid")
            s = self.tracks.setdefault(sid, {"first_t": t})
            s.update(last_t=t, cx=p["bbox_x"] + p["bbox_w"] / 2, cy=p["bbox_y"] + p["bbox_h"] / 2,
                     size=max(p["bbox_w"], p["bbox_h"]))
            p["raw_track_id"], p["track_id"] = p["track_id"], sid
        for sid in [k for k, s in self.tracks.items() if t - s["last_t"] > 2 * self.max_gap_s]:
            del self.tracks[sid]
        return people


def stitch_tracks(tracks: pd.DataFrame, fps: float, **kw) -> pd.DataFrame:
    """Batch version: run TrackStitcher over a tracks table in frame order (exactly what
    the live demo does frame by frame). Returns a copy with stable `track_id` and the
    tracker's id in `raw_track_id`."""
    if tracks.empty:
        return tracks.assign(raw_track_id=tracks.get("track_id"))
    st = TrackStitcher(**kw)
    df = tracks.sort_values(["frame", "track_id"]).reset_index(drop=True)
    sid = np.empty(len(df), np.int64)
    cols = ["track_id", "bbox_x", "bbox_y", "bbox_w", "bbox_h"]
    for frame, g in df.groupby("frame", sort=True):
        people = [dict(zip(cols, map(float, v))) for v in g[cols].to_numpy()]
        for p in people:
            p["track_id"] = int(p["track_id"])
        out = st.update(frame / fps, people)
        sid[g.index.to_numpy()] = [p["track_id"] for p in out]
    return df.assign(raw_track_id=df["track_id"], track_id=sid.astype("int32"))
