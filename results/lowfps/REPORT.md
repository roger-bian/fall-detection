# Slow camera (10 fps): tracker timing and v2 rule

Question: if the CPU/webcam only manages 10 fps, does ByteTrack's lost-track buffer
(1 s by default) need adjusting, and how much does 10 fps cost the v2 rule?

Setup: every test video converted to 10 fps (`scripts/make_lowfps.py`; CAUCAFall keeps
every 2nd frame so frame-level GT maps to frame // 2, GMDCSA24 is resampled). RTMO pose +
ByteTrack re-run (`scripts/run_lowfps_pose.sh`), scored with the **fixed** v2
thresholds (`run_rules_v2.py --fixed`, no re-tuning). Table: `scripts/compare_lowfps.py`.

| config | fall clips one id after onset | id restarts (fall / ADL clips) | CAUCAFall | GMDCSA24 S1-2 | GMDCSA24 S3-4 |
|---|---|---|---|---|---|
| original fps, frame counting (current) | 123/129 | 16 / 11 | 43/50, 1 FP | 37/41, 1 FP | 31/38, 3 FP |
| original fps, timestamps | 123/129 | 16 / 11 | 43/50, 1 FP | 37/41, 1 FP | 31/38, 3 FP |
| 10 fps, tracker told 10 fps | 127/129 | 11 / 11 | 42/50, 1 FP | 36/41, 2 FP | 30/38, 3 FP |
| 10 fps, tracker told 30 fps | 127/129 | 11 / 11 | 42/50, 1 FP | 36/41, 2 FP | 30/38, 3 FP |
| 10 fps, told 30 fps + timestamps | 119/129 | 38 / 13 | 42/50, 1 FP | 35/41, 2 FP | 27/38, 3 FP |
| **original fps + stitching** (now default) | 124/129 | 6 / 4 | 44/50, 1 FP | 37/41, 1 FP | 32/38, 3 FP |
| **10 fps + stitching** | 127/129 | 1 / 4 | 44/50, 1 FP | 38/41, 2 FP | 32/38, 3 FP |

"Restarts" = new track ids that start within 2 s of another track ending (the person lost
and picked up again under a new id).

## Findings

1. **The lost-track buffer does not need adjusting.** At 10 fps, telling the tracker
   10 fps (buffer = 10 frames = 1 s) or 30 fps (buffer = 30 frames = 3 s) gave identical
   tracks on all 260 clips. The ID changes that matter happen after gaps of 0-0.3 s, well
   inside either buffer: the returning box (wide, on the floor) does not overlap the
   predicted box (tall, standing) enough to be matched. That is an association failure,
   not an expiry.

2. **Timestamp mode is worse unless its fps is right.** `trackers` scales the Kalman step
   by elapsed time x the fps it was given. Told 30 fps while frames arrive at 10 fps, it
   predicts 3 frame-steps ahead with 3x the process noise: restarts in fall clips rose
   from 11 to 38 and 4 more falls were missed (104 vs 108 of 129). At the original fps, timestamps and frame
   counting are equivalent (259/260 clips identical). The live demo therefore stays on
   frame counting.

3. **10 fps costs 3 of 129 falls and adds 1 false alarm**, with the same thresholds:
   - `gmdcsa24_S02_fall_03`, `gmdcsa24_S04_fall_11`: the track id changes in the middle of
     the fall (gap about 0.3 s). The new id only sees the person already low, so drop
     0.30 -> 0.04 and 0.88 -> 0.13. At the original fps both kept one id through the fall.
   - `caucafall_S08_fall_forward`: the id changes in both versions, but at 10 fps the new
     id starts slightly later, so the drop it sees falls just under the threshold
     (0.375 -> 0.238, threshold 0.25).
   - `gmdcsa24_S02_adl_03` (lying down): max rotation 103.5 -> 110.3 deg/s, just over the
     110 threshold. At 10 fps rotation is measured over 0.2 s steps instead of 0.25 s.
   Across the falls caught in both versions, the medians barely move (max rotation 210 vs
   214 deg/s, max drop 0.63 vs 0.62, CAUCAFall latency 0.80 vs 0.85 s).

4. **Track stitching fixes it** (`falldet.stitch`, now on by default in the evaluation,
   renderer and demo). A new id inherits the history of a track that vanished <= 1.5 s
   earlier within 2 box sizes. With the same thresholds:
   - original fps: +2 falls (`caucafall_S03_fall_left`, `gmdcsa24_S03_fall_19`), no new FP.
   - 10 fps: +6 falls vs unstitched 10 fps (the 3 lost to 10 fps plus
     `caucafall_S03_fall_left`, `gmdcsa24_S01_fall_02`, `gmdcsa24_S03_fall_19`), no new FP.
     10 fps + stitching (114/129) catches one more fall than original fps + stitching
     (113/129); the one extra false alarm at 10 fps (`gmdcsa24_S02_adl_03`, rotation
     110.3 deg/s) remains.
