# fall_detection

Rule-based fall detection on RTMPose keypoints. Rules being reproduced (from `sample_result/`):
`FALL = torso tilt ≥ 35° OR (sink ≥ 0.40 AND still)`, evaluated per tracked person per frame.

## Environments

Two venvs from one uv project. The `gpu` and `cpu` extras conflict, so each lives in its own venv:

```bash
UV_PROJECT_ENVIRONMENT=.venv-gpu uv sync --extra gpu   # onnxruntime-gpu + CUDA 12/cuDNN 9 wheels (~3.3 GB)
UV_PROJECT_ENVIRONMENT=.venv-cpu uv sync --extra cpu   # onnxruntime CPU (~1 GB)

.venv-gpu/bin/python -m falldet.runtime    # providers should include CUDAExecutionProvider
.venv-cpu/bin/python -m falldet.runtime    # providers: CPUExecutionProvider
```

Run things with `.venv-gpu/bin/python ...` (or `.venv-cpu`). Plain `uv run` would create
a third `.venv` without either runtime; to use it, set `UV_PROJECT_ENVIRONMENT=.venv-gpu`
and pass `--extra gpu`.

`pyproject.toml` overrides rtmlib's `onnxruntime` dependency so the CPU build is only ever
installed with the `cpu` extra. If both packages are installed, the CPU one can overwrite the
GPU one and the CUDA provider silently disappears. That happened in
shoplifting-detection's `.venv`.

## Pose extraction

```bash
.venv-gpu/bin/python -m falldet.pose data/videos/caucafall/*.mp4 --out-dir data/pose/rtmo
.venv-cpu/bin/python -m falldet.pose VIDEO --device cpu --out-dir OUT
```

Models (rtmlib, ONNX; weights cached in `~/.cache/rtmlib`):

| `--model` | detector + pose | GPU ms/frame (720×480) |
|---|---|---|
| `rtmo` (default) | RTMO-m, one-stage, plus keypoint-box dedup (IoU ≥ 0.6) | ~10.5 |
| `balanced` | YOLOX-m + RTMPose-m | ~18 |
| `lightweight` | YOLOX-tiny + RTMPose-s | |
| `performance` | YOLOX-x + RTMPose-x | ~44 |

CPU (`balanced`, one clip) is ~146 ms/frame, with keypoints matching GPU to 0.06 px mean.

### Why RTMO is the default

`scripts/compare_pose.py` over all 260 test videos (`data/pose/compare.json`):

| | `rtmo` (dedup) | `balanced` (YOLOX) |
|---|---|---|
| CAUCAFall falls: lying frames with a person | **99.9%** | 78.4% |
| CAUCAFall falls: clips with < 50% of lying frames covered | **0 / 50** | 12 / 50 |
| CAUCAFall falls: frames during the fall (first 2 s) | **98.9%** | 70.5% |
| CAUCAFall falls: clips with one track id from onset to end | **50 / 50** | 42 / 50 |
| GMDCSA24 falls: last third of clip covered | **98.2%** | 87.6% |
| ADL frames with a person (CAUCAFall / GMDCSA24) | 98.7% / 98.7% | 88.5% / 98.2% |
| ADL frames with a duplicate 2nd person | 0.0% / 0.3% | 0.0% / 3.5% |
| GPU ms/frame | **~10.5** | ~18.6 |

YOLOX misses clearly visible people lying on the floor. RTMO's one problem was
duplicate skeletons on one seated person: 38% of GMDCSA24 ADL frames, which split one
person into 2–3 tracks. That's why the dedup was added in `OneStageBackend`. RTMO at
detection threshold 0.1 instead of 0.3 adds little coverage but many false and duplicate
people, so 0.3 stays.

Not yet tested: small or far-away people at low resolution. Like `sample_result/` at
640×360, shoplifting-detection found top-down better there at 320×240.

Output per video: `<name>.tracks.parquet` (frame, track_id, bbox, det_conf, COCO-17
`kp17_xyc`, kp_mean_conf) + `<name>.meta.json`. Adapted from shoplifting-detection's
`sld.pose.extract`. Unlike that version it applies no track filters, because a short
track can be the fall itself.

## Data

See [data/README.md](data/README.md).

## Live demo (webcam)

```bash
.venv-cpu/bin/python scripts/demo_webcam.py            # webcam 0 on CPU (~13 fps with RTMO-m)
.venv-gpu/bin/python scripts/demo_webcam.py --gpu      # webcam 0 on GPU (~9 ms/frame pose)
.venv-gpu/bin/python scripts/demo_webcam.py --source data/videos/caucafall/caucafall_S01_fall_forward.mp4
```

Or with make: `make webcam` (CPU), `make webcam-gpu`, `make webcam CAM=1 ARGS="--no-flip"`.

Keys: `q`/`Esc` quit, `r` reset tracks and alarms, `f` mirror. Options: `--source N|file`,
`--no-flip` (the webcam is mirrored by default, files never), `--record out.mp4`, `--events events.jsonl` (one JSON line per fall), `--alarm-hold`
(banner seconds), `--model`, `--no-display`.

- Runs the v2 rules (`results/rules_v2/REPORT.md`) through `falldet.rules.OnlineFallRuleV2`,
  a streaming version on wall-clock time. Fed a file's timestamps, it agrees with the batch
  v2 results on all 260 test videos: same alarm / no-alarm per clip, alarm-frame IoU median
  1.00, min 0.83.
- `--gpu` needs `.venv-gpu`. In `.venv-cpu` it exits with a message instead of silently
  running on CPU.
- **Camera placement matters.** The rule needs the hips and knees/ankles in view, and was
  tuned on wall/ceiling cameras 2–5 m away. A webcam on the desk that only sees your upper
  body will never fire. Stand back so you are visible head to feet, and fall onto something soft.
