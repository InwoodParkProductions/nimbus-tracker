# Nimbus Tracker

Automatic camera tracking and CG background compositing for greenscreen /
locked-off plates. Point it at an edited clip; it splits the clip into shots,
masks out the people, solves the camera, renders your CG scene through the
solved camera, and composites the CG behind the actors — one command, or the
app's render queue.

by Inwood Park Productions.

---

## What it does, stage by stage

```
clip ─► split into shots ─► person masks ─► camera solve ─► QC overlay
     ─► bake camera into your scene ─► render CG ─► soft matte ─► COMPOSITE
```

| Stage | File | Notes |
|------|------|-------|
| 0  split | `split_shots.py` | scene-cut detection + 1080p tracking proxies |
| 1  mask | `segment_people.py` | SAM2 silhouettes + YOLO11, temporal hysteresis |
| 2  solve | `auto_track.py` + `auto_track_stage2.py` | classic KLT front-end, then a learned (CoTracker3) retry; three solve attempts per front-end each in its own Blender process; every solve validated for real 3D geometry, not just reprojection error |
| 2a learned tracks | `cotrack_points.py` | background-seeded CoTracker3, streamed (flat VRAM) |
| 2c 2D fallback | `flow_solve.py` | rotation/zoom motion-match when 3D can't solve |
| QC | `dump_solve.py` + `qc_render.py` | overlays solved points on the footage so you can *see* if the track holds |
| 3  handoff | `apply_track_stage3.py` | bakes the solved camera into your `.blend` |
| 4  render | `render_stage4.py` | GPU Cycles/Eevee, resumable PNG sequences |
| 5a matte | `matte_people.py` | RobustVideoMatting soft alpha |
| 5  comp | `comp_stage5.py` | CG warped through the solved lens, composited behind the actors |

Run from source:

```
python auto_track.py "clip.mov"              # list shots
python auto_track.py "clip.mov" --shot 3 \
    --scene myscene.blend --start 0,-8,2 \
    --render out/shot03 --engine cycles --samples 128
```

or `python ui.py` for the windowed app. Build a standalone exe with
`python -m PyInstaller "Nimbus Tracker.spec"` (see `DEVELOPER_README.txt`).

---

## License

Nimbus Tracker is released under the **GNU General Public License v3.0**
(see `LICENSE`). This is required by its dependencies as much as chosen:
Blender (bundled) and every script that imports `bpy` are GPL, and Ultralytics
YOLO11 is AGPL-3.0.

### Two AI models are downloaded at runtime, NOT shipped in this repo

They have their own, more restrictive terms. The pipeline treats both as
best-effort and falls back without them, but if you use them, their licenses
govern:

| Model | Downloaded by | License | Plain-English |
|-------|---------------|---------|---------------|
| **CoTracker3** | `cotrack_points.py` | **non-commercial** (Meta research license — verify current terms) | fine for personal / research use; **not** for a commercial product |
| **RobustVideoMatting** | `matte_people.py` | GPL-3.0 | fine here; review before bundling into anything you sell |

### If you want to SELL a closed-source product

GPL software *can* be sold, but you must give buyers the source and they may
redistribute it. To ship a **proprietary** paid build instead you would need
to, at minimum:

1. Buy a commercial license from Ultralytics (to lift AGPL on YOLO11), **or**
   swap YOLO for a permissively-licensed detector.
2. Replace CoTracker3 (non-commercial) with a permissive point tracker.
3. Replace RobustVideoMatting (GPL) with a permissive matting model — e.g.
   BiRefNet (MIT) — verify at swap time.
4. Note that the `bpy` scripts and bundled Blender remain GPL regardless;
   Blender is load-bearing here.

None of this is legal advice — for a commercial launch, get a lawyer to review
the dependency stack.

---

## Requirements

Windows 10/11 64-bit, an NVIDIA GPU (falls back to CPU, slowly). Everything
else — Blender, Python, the AI models — is bundled in the packaged app, or set
up per `DEVELOPER_README.txt` when running from source.
