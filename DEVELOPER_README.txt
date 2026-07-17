================================================================
  NIMBUS TRACKER — SOURCE KIT (for editing the program)
  by Inwood Park Productions
================================================================

This kit contains everything needed to modify Nimbus Tracker and
rebuild the standalone app on a Windows machine.

----------------------------------------------------------------
1. ONE-TIME SETUP on the editing machine
----------------------------------------------------------------
a) Install Python 3.13 (python.org — check "Add to PATH").
b) In this folder, open a terminal and run:
       pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
       pip install -r requirements.txt
   (No NVIDIA GPU? Use plain `pip install torch torchvision` instead
   of the first line — everything falls back to CPU automatically.)
c) Blender 5.0: either install it from blender.org, OR copy the
   "blender" folder out of the packaged app ("Nimbus Tracker/blender")
   into this folder. The pipeline finds it automatically.

----------------------------------------------------------------
2. WHAT EACH FILE DOES
----------------------------------------------------------------
ui.py                  The app: window, pages, render queue, all UI.
                       The look lives in the CSS string near the top.
aerotrack_main.py      Entry point of the packaged .exe.
auto_track.py          Pipeline orchestrator: static-or-moving decision,
                       the solve chain, stage timing, render. The solve
                       chain per shot is:
                         classic KLT front-end  (precise on sharp footage)
                           -> learned front-end (robust on soft footage)
                             -> 2D flow -> static hold
                       and within each front-end, three solve attempts run
                       in SEPARATE Blender processes (auto keyframes,
                       manual keyframes, tripod) — Blender's solver is
                       stateful within a session and second solves can
                       return garbage; one attempt per process is the only
                       configuration found reliable. Best validated result
                       wins (perspective > tripod > none, then lower error).
split_shots.py         Stage 0 — cuts the clip into shots (+1080p proxies).
segment_people.py      Stage 1 — AI person masking (SAM2 silhouettes +
                       YOLO union + motion attach + gear pass + hysteresis).
cotrack_points.py      Stage 2a — learned point tracking (CoTracker3,
                       streamed so VRAM is flat in shot length). Seeds a
                       grid on the BACKGROUND (person masks say where) and
                       tracks through blur KLT can't hold. Weights download
                       on first use (~100MB); without them the pipeline
                       falls back to the classic front-end. NOTE: check the
                       CoTracker license before commercial distribution.
auto_track_stage2.py   Stage 2 — builds tracks (KLT, or from cotrack json)
                       and solves ONE configured attempt per process
                       (settings["solve_attempt"]). Solves are validated for
                       geometry, not just reprojection error: a homography
                       test plus bundle flatness/depth checks reject
                       "perspective" solves with no real 3D structure —
                       on greenscreen plates the only trackable points are
                       a flat backdrop, and a low-error fake-3D solve bakes
                       a confidently wrong camera into the scene.
flow_solve.py          Stage 2c — 2D motion-match fallback camera.
apply_track_stage3.py  Stage 3 — bakes the solved camera into your scene
                       (runs inside Blender).
render_stage4.py       Stage 4 — render, live-window mode, GPU denoise
                       (runs inside Blender).
place_static.py        Locked-off camera placement for static shots.
blender_setup.py       The in-Blender "choose camera + render settings"
                       panel used during shot setup.
preview_track.py, export_setup.py, export_camera.py
                       Aux Blender scripts (preview, exports).
Nimbus Tracker.spec    PyInstaller build recipe (bundled files list).
template.blend         Default scene; nagai_bg.svg = the app artwork.
*.pt                   AI models (YOLO n/x + SAM2) — keep next to the code.

----------------------------------------------------------------
3. RUN FROM SOURCE while editing (no build needed)
----------------------------------------------------------------
       python ui.py
   opens the app window using your edited code directly.
   Pipeline-only test of one shot:
       python auto_track.py "C:\path\clip.mov" --shot 2 --scene template.blend

----------------------------------------------------------------
4. REBUILD THE STANDALONE APP
----------------------------------------------------------------
       python -m PyInstaller "Nimbus Tracker.spec" --noconfirm
   Output lands in dist/Nimbus Tracker/.
   Then copy a Blender 5.0 folder INTO the output as:
       dist/Nimbus Tracker/blender/     (so blender.exe is at
       dist/Nimbus Tracker/blender/blender.exe)
   That makes the build all-in-one. Zip the "Nimbus Tracker"
   folder (plus READ ME FIRST.txt at the zip root) to share it.

   NOTE: PyInstaller DELETES the output folder each build — if you
   rebuild over a previous output, move the blender/ folder out
   first and put it back after.

----------------------------------------------------------------
5. GOTCHAS LEARNED THE HARD WAY
----------------------------------------------------------------
* If `python` on the machine is a different version without the
  packages, use the full path to the right python.exe.
* Blender 5 requires media_type='VIDEO' to be set BEFORE
  file_format='FFMPEG' (see render_stage4.py).
* SAM2's video predictor needs a real video file as input, not a
  list of frames (see segment_people.py chunk/trim handling).
* torch.cuda.is_available() can be True with zero devices — always
  check device_count() too (see segment_people.pick_device).
* App settings/queue live in %APPDATA%\NimbusTracker, not next to
  the exe.
