# Third-Party Notices

Nimbus Tracker uses the following third-party software. Each is the property of
its respective owners and is used under its own license.

## Default runtime dependencies (all OSI open-source)

| Component | Role | License |
|---|---|---|
| Blender 5.0 | camera solve, render, comp geometry | GPL-2.0-or-later |
| Ultralytics YOLO11 | person detection for masking | **AGPL-3.0** |
| RobustVideoMatting | soft person mattes (stage 5) | **GPL-3.0** |
| SAM2 (Segment Anything 2) | person silhouettes | Apache-2.0 |
| BootsTAPIR / tapnet (default tracker) | learned point tracking | Apache-2.0 |
| PyTorch, TorchVision | inference runtime | BSD-3-Clause |
| OpenCV | image/video I/O, classic tracking | Apache-2.0 |
| NumPy | arrays | BSD-3-Clause |
| Flask, Werkzeug, Jinja2 | local UI server | BSD-3-Clause |
| PySceneDetect | shot splitting | BSD-3-Clause |
| pywebview | native window | BSD-3-Clause |
| pythonnet | native window bridge | MIT |

**The strongest copyleft term wins: this project as a whole is AGPL-3.0**
(Ultralytics YOLO11). AGPL and GPL govern *distributing the software*, not the
*output you create with it* — plates, comps and renders you produce are yours,
including for commercial work.

## Optional dependency — NOT used by default

| Component | Role | License |
|---|---|---|
| CoTracker3 (Meta) | alternative tracker (`--tracker cotracker`) | **CC-BY-NC 4.0 (non-commercial)** |

CoTracker is **non-commercial** and is therefore *not* part of the default
pipeline (the default tracker is BootsTAPIR, Apache-2.0). Selecting
`--tracker cotracker` pulls in a non-commercial restriction — use it only for
personal / non-monetized work, never in a distributed build or paid pipeline.

## Fetched at setup / first run

`setup_tracker.py` clones **ibaiGorordo/Tapir-Pytorch-Inference** (Apache-2.0)
into `third_party/tapir/` and downloads the **dm-tapnet causal BootsTAPIR
checkpoint** (Apache-2.0). Neither is vendored in this repository. Ultralytics,
SAM2 and RVM weights download on first use via their libraries.

---
*This is an informational summary, not legal advice. Before any commercial
distribution, have the license stack reviewed — the AGPL-3.0 obligations and
the CoTracker non-commercial term are the ones that carry real consequences.*
