# data/

Test videos for the RTMPose fall-detection rules.

```
data/
├── README.md
├── raw/                      downloads as published (do not edit)
│   ├── caucafall/            Subject.N/<Activity>/{<Name>SN.avi, per-frame *.txt, classes.txt}
│   │   └── _mendeley_file_index_v5.json   file list + sha256 from the Mendeley API
│   └── gmdcsa24/             Subject N/{Fall,ADL}/NN.mp4, {Fall,ADL}.csv, LICENSE, README.md
│       └── GMDCSA24-v2.0.zip original archive
├── videos/                   test set: H.264 mp4, native resolution + fps, every frame kept
│   ├── caucafall/            caucafall_S03_fall_sitting.mp4 ...
│   └── gmdcsa24/             gmdcsa24_S01_fall_01.mp4 ...
├── annotations/
│   └── manifest.csv          one row per video (see below)
├── pose/                     (later) RTMPose outputs: <model>/<video_id>.tracks.parquet + .meta.json
└── logs/                     download logs
```

Rebuild `videos/` and `annotations/` from `raw/` with:

```bash
.venv-gpu/bin/python scripts/prepare_data.py          # or --no-transcode for the manifest only
```

## Datasets

| | CAUCAFall | GMDCSA24 |
|---|---|---|
| Videos | 100 (10 subjects × 5 falls + 5 ADL) | 160 (4 subjects; 79 falls, 81 ADL) |
| Setting | One home room, ceiling-corner CCTV camera; day, low light, night/IR | Indoor home rooms (bed, chair, floor); day and night recordings |
| Format | 720×480 @ 20 fps, .avi | 1280×720 (122+20 clips) or 640×480 (18), **variable frame rate**, ~30 or ~15 fps average, .mp4 |
| Ground truth | **Per-frame** fall / nofall (YOLO box + class) | **Per-clip** fall / ADL + free-text description |
| Fall types | backwards, forward, left, right, sitting | from chair / bed / standing, various sides |
| ADL | hop, kneel, pick up object, sit down, walk | lying down on bed, sitting, picking up, etc. |
| Licence | **CC BY 4.0** | **MIT** (GitHub) / CC BY 4.0 (Zenodo record) |
| Source | [Mendeley Data 10.17632/7w7fccy7ky.5](https://data.mendeley.com/datasets/7w7fccy7ky) | [Zenodo 10.5281/zenodo.12921216](https://doi.org/10.5281/zenodo.12921216), [GitHub](https://github.com/ekramalam/GMDCSA24-A-Dataset-for-Human-Fall-Detection-in-Videos) |

Both licences allow commercial use with attribution. Cite:

- CAUCAFall: Eraso, J.C., Muñoz, E., Muñoz, M., Pinto, J. (2022). *Dataset CAUCAFall*. Mendeley Data, V5. doi:10.17632/7w7fccy7ky.5. Paper: *Data in Brief* (2022), https://www.sciencedirect.com/science/article/pii/S2352340922008162
- GMDCSA24: Alam, E., Sufian, A., Dutta, P., Leo, M., Hameed, I.A. (2024). *GMDCSA24: A Dataset for Human Fall Detection in Videos*. Data in Brief. doi:10.5281/zenodo.11489420

All participants are healthy adults performing **staged** falls. Expect real care-home
falls (like `sample_result/`) to be harder: slower, partly occluded, frail subjects.

## manifest.csv

| column | meaning |
|---|---|
| `video_id` | `<dataset>_S<subject>_<activity>`; also the file stem under `videos/` |
| `label` | `fall` or `adl` (clip level) |
| `path` / `source_path` | normalised mp4 / original raw file, relative to the project root |
| `width,height,fps,n_frames,duration_s` | from ffprobe on the mp4 |
| `gt_level` | `frame` (CAUCAFall) or `clip` (GMDCSA24) |
| `gt_fall_start,gt_fall_end` | 0-based first/last frame labelled fall (CAUCAFall only) |
| `gt_fall_frames` | number of frames labelled fall |
| `label_files` | number of per-frame label files found in raw/ |
| `description` | GMDCSA24 clip description |

### CAUCAFall label caveats

- Label file `xxxSNnnnnn.txt` → video frame `nnnnn − 1`. The authors' frame export
  has one more file than the video has frames: a duplicate of the last frame.
  Verified by pixel-matching their PNGs to the decoded video (Subject 3, Fall sitting:
  PNG 55/58/61/64 → frames 54/57/60/63). `prepare_data.py` drops it.
- "Fall" in CAUCAFall labels covers the fall **and the time lying on the floor
  afterwards**. In 49 of the 50 fall clips it runs to the last frame. Use `gt_fall_start` as
  the fall onset; an alarm anywhere in `[gt_fall_start, gt_fall_end]` is a hit. Onsets
  are at 1.2–6.5 s (median 3.0 s). No ADL clip has a fall-labelled frame.
- Clips that do not have exactly one extra label file (`prepare_data.py` prints them):
  - `S02_fall_backwards` (+2) and `S06_fall_backwards` (+23): the export runs past the
    end of the .avi. Pixel-matched: label N → frame N−1 holds through the fall
    (S02 labels 20–90, S06 labels 20–234), so GT is usable.
  - One label file missing in `S05_fall_left`, `S05_kneel`, `S08_hop` (2), `S09_sit_down`
    (failed downloads, deliberately not retried) and `S01_sit_down`, `S10_walk`,
    `S06_walk`, `S08_pick_up_object`, `S09_walk`. Single-frame gaps, no effect on GT.
- The upload also contains a stray `Subject.2/Fall backwards/cas200091 - copia.png`.
  It is ignored.
- PNG frames (8 GB) were not downloaded; they are the video frames.
