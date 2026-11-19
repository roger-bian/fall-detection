# Rules v2: lying AND (hip drop OR fast rotation)

Pose: RTMO-m with keypoint-box dedup (`data/pose/rtmo_t0.3_dedup`). Code: `src/falldet/rules.py`
(`compute_features_v2`), `scripts/run_rules_v2.py`. Per-video results with measured values:
`per_video.csv`. Threshold grid: `grid_tune.csv`.

## Rule

```
lying = |torso tilt| >= 50°  AND  |leg tilt| >= 50°  in >= 80% of the last 1 s
        (torso = hip-mid -> shoulder-mid from vertical; leg = hip-mid -> ankle-mid (knee if
         no ankle), 0° = feet straight below the hips)
drop  = largest hip descent in the last 3 s / largest box height in that window
rot   = peak torso rotation speed (°/s, 0.25 s steps) in the last 2.5 s
FALL  = lying AND (drop >= 0.25 OR rot >= 110°/s)
```

Why each part (measured on v1 errors, `results/rules_v1`):
- **Legs as well as torso.** Bending to pick up, kneeling and sitting down tilt the torso 35–180° (all v1 angle false alarms) but keep the legs under the hips. With both required in 100% of a 1 s window, every CAUCAFall ADL stays under 50° (max: kneel 48°, sit down 42°). The rule uses 80% of the window, and one kneel (S08) gets through.
- **Drop.** Lying down on a bed lowers the hips 0.05–0.22 × body height. Falls lower them by a median of 0.45–1.0.
- **Rotation speed.** Falling onto a bed or backwards from sitting barely lowers the hips, but the torso turns at 115–212°/s. Lying down on purpose turns it at 40–149°/s (median 75).
- The v1 sink + still rule was dropped. Its false alarms were walking toward the ceiling camera (perspective) and kneeling.

## Protocol

`drop_thr` and `rot_thr` were grid-searched on **tune** = CAUCAFall + GMDCSA24 subjects 1–2
(maximise recall − FP rate). **Test** = GMDCSA24 subjects 3–4, scored with the tuned values.

**The test split is not a clean held-out set.** No threshold was fitted on S3–4, but the
features were designed after printing per-clip values for all GMDCSA24 subjects, S3–4
included: bed ADLs, push-ups and low-drop falls. The defaults 0.25 / 110 were also written
before the grid ran. Treat the S3–4 numbers as **optimistic**. A clean estimate needs data
none of this was designed on, e.g. the care-home clips or a third dataset.

## Results

| | v1 falls detected | v1 false alarms (ADL) | **v2 falls detected** | **v2 false alarms** |
|---|---|---|---|---|
| CAUCAFall (tune) | 49/50 | 40/50 | 43/50 | **1/50** |
| GMDCSA24 S1–2 (tune) | 40/41 | 30/39 | 37/41 | **1/39** |
| GMDCSA24 S3–4 (test, optimistic) | 38/38 | 23/42 | **31/38** | **3/42** |

What each branch adds on the tune split (`grid_tune.csv`): drop ≥ 0.25 alone gives 77/91 falls
and 2/89 FP. Rotation ≥ 110°/s alone gives 69/91 falls and 1/89 FP. Both together give 80/91
falls and 2/89 FP. Rotation thresholds of 110, 120 and 140 tie, so the data doesn't pin
`rot_thr` down.

CAUCAFall alarms now come a median 0.8 s after the labelled onset (v1: 0.4 s before),
because v2 waits for 1 s of lying. No alarms come before the onset.

## Remaining errors (all 23, values measured after the GT onset)

**False alarms (5)**
- Push-ups, 3 clips (GMDCSA24 S02 adl 22, S04 adl 06/07): really lying prone on the floor with the hips lowered (drop 0.62–0.75). Indistinguishable from a fall by posture and drop.
- S04 adl 10, walking then lying on a bed: rotation 149°/s, above the 110°/s threshold (drop 0.17).
- CAUCAFall S08 kneel: 11 frames lying (min tilt 72°, drop 0.66, rotation 176°/s). Verified on the frame: night IR, the person is low on hands and knees.

**Misses (18)**

| Cause | Clips | What was measured |
|---|---|---|
| Torso lying, legs not | CF S10 fall sitting, CF S04 fall left, GM S03 fall 02, GM S04 fall 08 / 09 / 16 / 17 | median torso 52–94°, leg 8–49°. Verified for GM S04 fall 08 / 09: lying across the bed with legs hanging off the edge. The others are not checked |
| Legs lying, torso not | CF S02 fall backwards, CF S04 fall right, GM S04 fall 03 | median torso 41–46°, legs 55–60°. Verified for the two CF clips: the body lies **diagonally** in the image, so the torso measures ~45° from vertical, just under 50°. A threshold measured in the image is orientation-dependent |
| Ends sitting upright | CF S02 / S04 fall sitting, GM S01 fall 13 / 14 | torso 12–31°, legs 11–49°. Neither posture rule can see these |
| Track ID changes at the fall | CF S03 fall left, GM S03 fall 19 | lying held (47 / 22 frames), but the person is lost during the fall (13 / 2 frames) and re-appears with a new ID. Drop and rotation are per-track, so the descent is not seen (drop 0.19 / 0.08, rotation 58 / 31°/s) |
| No usable pose | GM S01 fall 01 / 02 | the body is mostly out of frame ("full body not visible"): pose in 9% of frames / no confident shoulders and hips |

## Next steps (not done)

1. **Stitch tracks across a fall.** If a track appears within ~1 s and ~1 body length of one that just ended, carry its history over. This fixes the track-ID cause without loosening any threshold.
2. **Loosen the leg part.** For example, legs ≥ 30–40° while the torso is ≥ 60°. It must be re-checked against sit down (legs reach 72°) and kneel. Tune on the tune split only.
3. **Sitting-upright falls** need a different signal: fast hip descent to floor level, with the torso upright, then no movement. It must not fire on sitting down onto a chair or on kneeling.
4. **Push-ups** cannot be separated by posture. Accept them, or add a "repetitive motion" check.
5. **`drop` still has the camera-angle problem.** It measures hip descent in image y, the same quantity that broke v1 on walking toward the ceiling camera. It is now only gated by "lying", so walking can't trigger it, but its size still depends on where in the room the person falls. A perspective-robust version (e.g. hip descent relative to the head–feet extent) is not done.
