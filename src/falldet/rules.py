"""Per-frame, per-track fall rules on COCO-17 keypoints.

    FALL = |angle| >= angle_thr  OR  (sink >= sink_thr AND still)

Reproduces the rule deduced from sample_result/ (`angle>=35 or sink>=0.40+still`).
The overlays fixed the structure, the two thresholds, per-frame firing (no confirmation
window, no cooldown) and that the angle is the torso tilt. They did NOT fix how sink and
stillness are computed, so those are our definitions, chosen to be simple and stated here:

angle  Torso tilt from vertical, degrees. Vector hip-midpoint -> shoulder-midpoint
       (each midpoint from whichever of L/R has conf >= kp_thr). 0 = upright,
       +/-90 = horizontal, sign + = shoulders to the right of the hips in the image.
       The rule uses |angle|.

sink   How far the hip midpoint has dropped below its highest point in the trailing
       `sink_window_s` of the same track, divided by the person's box height at that
       highest point (≈ standing height). 0.40 = dropped 40% of body height. The reference
       is per track, so a new track id restarts it (this matches the sample overlays,
       where a new track showed s=0.00).

still  Over the trailing `still_window_s` of the track, the hip and shoulder midpoints
       each stay within `still_tol` × the current box's longer side (range of x and y).
       Needs at least `still_min_frames` valid frames in the window.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12


@dataclass
class RuleParams:
    angle_thr: float = 35.0
    sink_thr: float = 0.40
    kp_thr: float = 0.3
    sink_window_s: float = 2.0
    still_window_s: float = 0.5
    still_tol: float = 0.10
    still_min_frames: int = 3

    def as_dict(self) -> dict:
        return asdict(self)


def _mid(kp: np.ndarray, a: int, b: int, thr: float) -> np.ndarray:
    """(N, 2) midpoint of keypoints a/b using whichever are confident; NaN if neither."""
    pa, pb = kp[:, a, :2], kp[:, b, :2]
    ca, cb = kp[:, a, 2] >= thr, kp[:, b, 2] >= thr
    out = np.full((len(kp), 2), np.nan)
    both = ca & cb
    out[both] = (pa[both] + pb[both]) / 2
    out[ca & ~cb] = pa[ca & ~cb]
    out[~ca & cb] = pb[~ca & cb]
    return out


def compute_features(tracks: pd.DataFrame, fps: float, p: RuleParams) -> pd.DataFrame:
    """Add angle / sink / still / fall columns to a tracks dataframe (one row per
    frame x track, as written by falldet.pose)."""
    cols = ["frame", "track_id", "angle", "sink", "still", "motion", "fall_angle",
            "fall_sink", "fall", "hip_y", "ref_h"]
    if tracks.empty:
        return pd.DataFrame(columns=cols)
    out = []
    w_sink = p.sink_window_s * fps
    w_still = p.still_window_s * fps
    for tid, g in tracks.sort_values("frame").groupby("track_id"):
        kp = np.stack(g["kp17_xyc"].to_numpy()).reshape(-1, 17, 3)
        frames = g["frame"].to_numpy()
        sh = _mid(kp, L_SH, R_SH, p.kp_thr)
        hip = _mid(kp, L_HIP, R_HIP, p.kp_thr)
        v = sh - hip
        angle = np.degrees(np.arctan2(v[:, 0], -v[:, 1]))
        bh = g["bbox_h"].to_numpy()
        blong = np.maximum(g["bbox_w"].to_numpy(), bh)

        n = len(g)
        sink = np.full(n, np.nan)
        ref_h = np.full(n, np.nan)
        motion = np.full(n, np.nan)
        still = np.zeros(n, bool)
        valid = ~np.isnan(hip[:, 1]) & ~np.isnan(sh[:, 1])
        for i in range(n):
            if not valid[i]:
                continue
            # sink: highest hip (min y) in the trailing window, normalised by box height there
            win = (frames > frames[i] - w_sink) & (frames <= frames[i]) & valid
            j = np.flatnonzero(win)[np.argmin(hip[win, 1])]
            if bh[j] > 0:
                sink[i] = (hip[i, 1] - hip[j, 1]) / bh[j]
                ref_h[i] = bh[j]
            # still: range of hip & shoulder midpoints in the trailing window
            sw = (frames > frames[i] - w_still) & (frames <= frames[i]) & valid
            if sw.sum() >= p.still_min_frames and blong[i] > 0:
                rng = max(np.ptp(hip[sw], axis=0).max(), np.ptp(sh[sw], axis=0).max())
                motion[i] = rng / blong[i]
                still[i] = motion[i] <= p.still_tol
        fa = valid & (np.abs(angle) >= p.angle_thr)
        fs = valid & (np.nan_to_num(sink, nan=-1) >= p.sink_thr) & still
        out.append(pd.DataFrame({
            "frame": frames, "track_id": tid, "angle": np.where(valid, angle, np.nan),
            "sink": sink, "still": still, "motion": motion,
            "fall_angle": fa, "fall_sink": fs, "fall": fa | fs,
            "hip_y": hip[:, 1], "ref_h": ref_h,
        }))
    return pd.concat(out, ignore_index=True)[cols]


# --------------------------------------------------------------------------------------
# v2: lying + (drop OR fast rotation)
# --------------------------------------------------------------------------------------

L_KNEE, R_KNEE, L_ANK, R_ANK = 13, 14, 15, 16


@dataclass
class RuleParamsV2:
    """FALL = lying AND (drop >= drop_thr OR rot >= rot_thr).

    lying  torso tilt |angle| AND leg tilt (hip -> ankle midpoint, knee if no ankle;
           0 = feet straight below hips) both >= lie_thr, in >= lie_frac of the
           frames of the trailing lie_s window. Bending, kneeling and sitting keep the
           legs under the hips; lying does not (v1 analysis, results/rules_v1).
    drop   largest hip descent within the trailing desc_s window (from the window's
           highest hip point), divided by the largest box height in that window. Separates
           falling to the floor from lying back on a bed (hips barely move).
    rot    peak torso rotation speed, deg/s over rot_dt_s steps, in the trailing
           lie_s + rot_pre_s window. Separates uncontrolled falls (incl. onto a bed or from
           sitting) from lying down on purpose.
    """
    lie_thr: float = 50.0
    lie_s: float = 1.0
    lie_frac: float = 0.8
    drop_thr: float = 0.25
    desc_s: float = 3.0
    rot_thr: float = 110.0
    rot_dt_s: float = 0.25
    rot_pre_s: float = 1.5
    kp_thr: float = 0.3

    def as_dict(self) -> dict:
        return asdict(self)


def compute_features_v2(tracks: pd.DataFrame, fps: float, p: RuleParamsV2) -> pd.DataFrame:
    cols = ["frame", "track_id", "angle", "leg", "lie_raw", "lying", "drop", "rot",
            "fall_drop", "fall_rot", "fall"]
    if tracks.empty:
        return pd.DataFrame(columns=cols)
    w_lie = max(1, int(round(p.lie_s * fps)))
    w_desc = p.desc_s * fps
    q = max(1, int(round(p.rot_dt_s * fps)))
    w_rot = (p.lie_s + p.rot_pre_s) * fps
    out = []
    for tid, g in tracks.sort_values("frame").groupby("track_id"):
        kp = np.stack(g["kp17_xyc"].to_numpy()).reshape(-1, 17, 3)
        fr = g["frame"].to_numpy()
        sh = _mid(kp, L_SH, R_SH, p.kp_thr)
        hip = _mid(kp, L_HIP, R_HIP, p.kp_thr)
        low = _mid(kp, L_ANK, R_ANK, p.kp_thr)
        knee = _mid(kp, L_KNEE, R_KNEE, p.kp_thr)
        low = np.where(np.isnan(low), knee, low)
        v = sh - hip
        angle = np.degrees(np.arctan2(v[:, 0], -v[:, 1]))
        lv = low - hip
        leg = np.degrees(np.arctan2(lv[:, 0], lv[:, 1]))
        lie_raw = np.nan_to_num(np.minimum(np.abs(angle), np.abs(leg)), nan=0.0) >= p.lie_thr
        bh = g["bbox_h"].to_numpy()
        hy = hip[:, 1]

        # torso rotation speed on a gap-filled per-frame series
        full = np.arange(fr[0], fr[-1] + 1)
        A = pd.Series(np.abs(angle), index=fr).reindex(full).interpolate(limit=3)
        rate_full = (A.diff(q).abs() / (q / fps)).to_numpy()

        n = len(g)
        lying = np.zeros(n, bool)
        drop = np.full(n, np.nan)
        rot = np.full(n, np.nan)
        for i in range(n):
            wl = (fr > fr[i] - w_lie) & (fr <= fr[i])
            if wl.sum() < max(2, w_lie // 2) or lie_raw[wl].mean() < p.lie_frac:
                continue
            lying[i] = True
            wd = (fr > fr[i] - w_desc) & (fr <= fr[i]) & ~np.isnan(hy)
            if wd.sum() >= 2:
                idx = np.flatnonzero(wd)
                j = idx[np.argmin(hy[idx])]
                after = idx[idx >= j]
                ref = np.nanmax(bh[idx])
                if ref > 0:
                    drop[i] = (np.nanmax(hy[after]) - hy[j]) / ref
            lo = np.searchsorted(full, fr[i] - w_rot)
            hi = np.searchsorted(full, fr[i], side="right")
            seg = rate_full[lo:hi]
            if np.isfinite(seg).any():
                rot[i] = np.nanmax(seg)
        fd = lying & (np.nan_to_num(drop, nan=-1) >= p.drop_thr)
        frot = lying & (np.nan_to_num(rot, nan=-1) >= p.rot_thr)
        out.append(pd.DataFrame({
            "frame": fr, "track_id": tid, "angle": angle, "leg": leg, "lie_raw": lie_raw,
            "lying": lying, "drop": drop, "rot": rot,
            "fall_drop": fd, "fall_rot": frot, "fall": fd | frot,
        }))
    return pd.concat(out, ignore_index=True)[cols]


class OnlineFallRuleV2:
    """Streaming version of compute_features_v2 for live video.

    Same definitions, but driven by timestamps (seconds) instead of frame indices, because
    a live camera has no fixed frame rate (CPU inference runs well below camera fps).
    Feed one update per (track, frame); it returns that frame's features and decision.

    Differences from the batch version, both small:
      * lying needs >= half the samples expected for lie_s at the measured rate
        (batch: half of lie_s * video fps).
      * rotation speed compares each sample to the newest sample at least rot_dt_s older
        (batch: exactly rot_dt_s * fps frames back on a gap-interpolated series).
    """

    def __init__(self, p: RuleParamsV2 | None = None, forget_s: float = 5.0):
        self.p = p or RuleParamsV2()
        self.forget_s = forget_s
        self.hist: dict[int, list[tuple]] = {}   # tid -> [(t, abs_angle, lie_raw, hip_y, bbox_h, rate)]
        self.dt_ema: float | None = None
        self._last_t: float | None = None

    def reset(self) -> None:
        self.hist.clear()
        self.dt_ema = None
        self._last_t = None

    def tick(self, t: float) -> None:
        """Call once per processed frame (before the updates) to track the frame rate
        and forget tracks that have not been seen for forget_s."""
        if self._last_t is not None and t > self._last_t:
            dt = t - self._last_t
            self.dt_ema = dt if self.dt_ema is None else 0.9 * self.dt_ema + 0.1 * dt
        self._last_t = t
        for tid in [k for k, h in self.hist.items() if t - h[-1][0] > self.forget_s]:
            del self.hist[tid]

    def update(self, tid: int, t: float, kp17_xyc: np.ndarray, bbox_h: float) -> dict:
        p = self.p
        kp = np.asarray(kp17_xyc, np.float32).reshape(1, 17, 3)
        sh = _mid(kp, L_SH, R_SH, p.kp_thr)[0]
        hip = _mid(kp, L_HIP, R_HIP, p.kp_thr)[0]
        low = _mid(kp, L_ANK, R_ANK, p.kp_thr)[0]
        if np.isnan(low).any():
            low = _mid(kp, L_KNEE, R_KNEE, p.kp_thr)[0]
        v, lv = sh - hip, low - hip
        angle = float(np.degrees(np.arctan2(v[0], -v[1])))
        leg = float(np.degrees(np.arctan2(lv[0], lv[1])))
        m = min(abs(angle), abs(leg))
        lie_raw = bool(np.nan_to_num(m, nan=0.0) >= p.lie_thr)

        h = self.hist.setdefault(tid, [])
        # rotation rate vs the newest earlier sample at least rot_dt_s back
        rate = np.nan
        a_abs = abs(angle)
        for s in reversed(h):
            if t - s[0] >= p.rot_dt_s:
                if not (np.isnan(a_abs) or np.isnan(s[1])) and t - s[0] <= 3 * p.rot_dt_s:
                    rate = abs(a_abs - s[1]) / (t - s[0])
                break
        h.append((t, a_abs, lie_raw, float(hip[1]), float(bbox_h), rate))
        keep_s = max(p.desc_s, p.lie_s + p.rot_pre_s) + 0.5
        while h and t - h[0][0] > keep_s:
            h.pop(0)

        dt = self.dt_ema or (1 / 15)
        win = [s for s in h if s[0] > t - p.lie_s]
        need = max(2, int(p.lie_s / dt) // 2)
        lying = len(win) >= need and np.mean([s[2] for s in win]) >= p.lie_frac

        drop = rot = np.nan
        if lying:
            wd = [s for s in h if s[0] > t - p.desc_s and not np.isnan(s[3])]
            if len(wd) >= 2:
                j = int(np.argmin([s[3] for s in wd]))
                ref = max(s[4] for s in wd)
                if ref > 0:
                    drop = (max(s[3] for s in wd[j:]) - wd[j][3]) / ref
            rates = [s[5] for s in h if s[0] > t - (p.lie_s + p.rot_pre_s) and not np.isnan(s[5])]
            if rates:
                rot = max(rates)
        fall_drop = bool(lying and np.nan_to_num(drop, nan=-1) >= p.drop_thr)
        fall_rot = bool(lying and np.nan_to_num(rot, nan=-1) >= p.rot_thr)
        return {"angle": angle, "leg": leg, "lying": bool(lying), "drop": drop, "rot": rot,
                "fall_drop": fall_drop, "fall_rot": fall_rot, "fall": fall_drop or fall_rot}
