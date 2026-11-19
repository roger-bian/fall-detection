"""Overlay drawing shared by the result-video renderer and the live demo
(style follows sample_result/: per-track skeleton + value label, red FALL banner + border)."""

from __future__ import annotations

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
LEFT = [(5, 7), (7, 9), (11, 13), (13, 15)]
RIGHT = [(6, 8), (8, 10), (12, 14), (14, 16)]
TORSO = [(5, 6), (11, 12), (5, 11), (6, 12)]
BLUE, GREEN, ORANGE, WHITE, RED = (255, 128, 0), (0, 220, 0), (0, 140, 255), (255, 255, 255), (0, 0, 220)
TRACK_COLORS = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 200, 255), (180, 105, 255),
                (0, 255, 128)]
BANNER_H = 24


OUTLINE = [(dx, dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy]


def text(img, s, org, scale=0.5, color=WHITE, thick=1, bg=None, outline=(0, 0, 0)):
    """Text with a dark outline, or on a filled box when `bg` is given.

    The outline is the same text stamped at 1 px offsets in 8 directions, at the same
    thickness as the text itself. (A thicker outline stroke fills the gaps between letters
    and makes the text hard to read.)"""
    (w, h), base = cv2.getTextSize(s, FONT, scale, thick)
    x, y = org
    if bg is not None:
        cv2.rectangle(img, (x - 2, y - h - 3), (x + w + 2, y + base), bg, -1)
    elif outline is not None:
        for dx, dy in OUTLINE:
            cv2.putText(img, s, (x + dx, y + dy), FONT, scale, outline, thick, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, color, thick, cv2.LINE_AA)


def mid(kp, a, b, thr=0.3):
    pts = [kp[i, :2] for i in (a, b) if kp[i, 2] >= thr]
    return np.mean(pts, axis=0) if pts else None


def track_color(tid: int):
    return TRACK_COLORS[tid % len(TRACK_COLORS)]


def feature_label(tid: int, angle: float, leg: float, lying: bool, drop: float, rot: float) -> str:
    lab = f"#{tid} a{angle:+.0f} l{leg:+.0f}" if not np.isnan(leg) else f"#{tid} a{angle:+.0f} l-"
    if lying and not np.isnan(drop):
        lab += f" d{drop:.2f} r{rot:.0f}" if not np.isnan(rot) else f" d{drop:.2f}"
    return lab


def draw_person(img, kp17_xyc, tid: int, label: str | None = None, lying: bool = False,
                scale: float = 1.0, kp_thr: float = 0.3) -> None:
    """Skeleton (left blue, right green, torso orange), white hip->shoulder line, mid-point
    dots in the track colour, and the value label (red tag while lying)."""
    k = np.asarray(kp17_xyc, np.float32).reshape(17, 3).copy()
    k[:, :2] *= scale
    col = track_color(tid)
    for edges, c in ((LEFT, BLUE), (RIGHT, GREEN), (TORSO, ORANGE)):
        for a, b in edges:
            if k[a, 2] >= kp_thr and k[b, 2] >= kp_thr:
                cv2.line(img, tuple(map(int, k[a, :2])), tuple(map(int, k[b, :2])), c, 2,
                         cv2.LINE_AA)
    sh, hp = mid(k, 5, 6, kp_thr), mid(k, 11, 12, kp_thr)
    if sh is not None and hp is not None:
        cv2.line(img, tuple(map(int, hp)), tuple(map(int, sh)), WHITE, 2, cv2.LINE_AA)
        cv2.circle(img, tuple(map(int, hp)), 4, col, -1, cv2.LINE_AA)
    if sh is not None:
        cv2.circle(img, tuple(map(int, sh)), 4, col, -1, cv2.LINE_AA)
        if label:
            x = int(min(sh[0] + 6, img.shape[1] - 8))
            y = int(max(sh[1] - 6, 40))
            if lying:
                text(img, label, (x, y), 0.42, WHITE, 1, bg=(0, 0, 170))
            else:
                text(img, label, (x, y), 0.42, col, 1)


def draw_alarm(img, msg: str) -> int:
    """Red border + top banner. Returns the banner height (offset for the header)."""
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), RED, 4)
    cv2.rectangle(img, (0, 0), (w, BANNER_H), (0, 0, 200), -1)
    text(img, msg, (8, 18), 0.6, WHITE, 1, outline=(0, 0, 90))
    return BANNER_H


def rule_text(p: dict) -> str:
    return (f"lying(torso&leg>={p['lie_thr']:g}deg {p['lie_s']:g}s) & "
            f"(drop>={p['drop_thr']:g} or rot>={p['rot_thr']:g}deg/s)")
