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
| 2  solve | `auto_track.py` + `auto_track_stage2.py` | classic KLT front-end, then a learned retry; three solve attempts per front-end each in its own Blender process; every solve validated for real 3D geometry, not just reprojection error |
| 2a learned tracks | `cotrack_points.py` | background-seeded point tracking, streamed (flat VRAM). Default backend **BootsTAPIR** (Apache-2.0, commercial-safe); `--tracker cotracker` for Meta CoTracker (non-commercial, slightly tighter on hard shots) |
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

**First-time setup:** after installing the Python deps, run
`python setup_tracker.py` once to fetch the default tracker (BootsTAPIR,
Apache-2.0 — its code + ~209 MB checkpoint aren't vendored in this repo).

---

## License

Nimbus Tracker as a whole is **AGPL-3.0** (see `LICENSE` and
[THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md)). This is set by its
dependencies as much as chosen: Ultralytics YOLO11 is AGPL-3.0, Blender and
every `bpy` script are GPL. **The default tracker is now BootsTAPIR
(Apache-2.0), so nothing in the default pipeline is non-commercial** — the
program is legitimately shareable open-source.

Crucially: AGPL/GPL govern *distributing the software*, **not the output you
make with it.** Plates, renders and comps you produce are yours — commercial
work included. You can use this to make paid VFX shots today.

### The one non-commercial option (off by default)

| Model | Selected by | License | Plain-English |
|-------|---------------|---------|---------------|
| **CoTracker3** | `--tracker cotracker` | **CC-BY-NC (non-commercial)** | slightly tighter on the hardest shots; personal / non-monetized only, never in a distributed build |

`RobustVideoMatting` (matting, GPL-3.0) is fine for output use; swap it for
BiRefNet (MIT) if you want to tidy that corner for a bundled build.

### If you want to SELL a closed-source (proprietary) product

Using it to make commercial work is already fine. To ship a *proprietary paid
build of the software itself* you'd still need to:

1. Buy a commercial license from Ultralytics (to lift AGPL on YOLO11), **or**
   swap YOLO for a permissively-licensed detector.
2. Keep `--tracker bootstapir` (done) — never bundle CoTracker.
3. Optionally replace RVM (GPL) with BiRefNet (MIT).
4. `bpy` scripts and bundled Blender remain GPL; Blender is load-bearing.

None of this is legal advice — for a commercial launch, get a lawyer to review
the dependency stack.

---

## Requirements

Windows 10/11 64-bit, an NVIDIA GPU (falls back to CPU, slowly). Everything
else — Blender, Python, the AI models — is bundled in the packaged app, or set
up per `DEVELOPER_README.txt` when running from source.
