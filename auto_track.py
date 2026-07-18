"""
auto_track.py — one command for the whole pipeline.
====================================================
Run with the SYSTEM python from the auto-track folder:

  Step 1: see what shots are in your footage (fast, nothing heavy runs):
      python auto_track.py "path\\to\\footage.mov"

  Step 2: track one shot (and optionally hand off + render):
      python auto_track.py "path\\to\\footage.mov" --shot 3
          [--scene my_scene.blend --start 0,-8,2]     -> bakes camera into your scene
          [--render out.mp4 --engine cycles --samples 64 --percent 50]
          [--rotation rx,ry,rz] [--scale s]           -> orient/scale the path
          [--tracking-settings '{"motion_model": "LocRotScale"}']

Everything lands in a work folder next to the footage:
    <footage_dir>/<footage_name>_autotrack/
        shots/            shot_NN.mp4 + shots.json
        shot_NN_masks/    person masks
        shot_NN_out/      *_track_log.json (solve error!) + tracked .blend
        scene_tracked.blend, render output       (if requested)
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

from split_shots import shot_file_for  # noqa: E402


def find_blender():
    """Locate blender.exe across the places it commonly installs, so the app
    works on a fresh machine without hand-configuring anything:
      1. AEROTRACK_BLENDER env var (explicit override)
      2. blender on PATH
      3. Program Files / Steam / per-user install folders (newest version)
    Returns 'blender' as a last resort (may be on PATH under another name)."""
    import shutil
    env = os.environ.get("AEROTRACK_BLENDER")
    if env and os.path.exists(env):
        return env
    # bundled Blender ships inside the app folder — the all-in-one install
    # needs nothing from the machine
    exe_dir = (os.path.dirname(sys.executable)
               if getattr(sys, "frozen", False) else HERE)
    for cand in (os.path.join(exe_dir, "blender", "blender.exe"),
                 os.path.join(os.path.dirname(exe_dir),
                              "blender", "blender.exe")):
        if os.path.exists(cand):
            return cand
    on_path = shutil.which("blender")
    if on_path:
        return on_path
    roots = [
        r"C:\Program Files\Blender Foundation",
        r"C:\Program Files (x86)\Blender Foundation",
        r"C:\Program Files\Steam\steamapps\common\Blender",
        r"C:\Program Files (x86)\Steam\steamapps\common\Blender",
        os.path.join(os.environ.get("LOCALAPPDATA", ""),
                     r"Programs\Blender Foundation"),
    ]
    candidates = []
    for root in roots:
        if not root:
            continue
        candidates += glob.glob(os.path.join(root, "Blender *", "blender.exe"))
        candidates += glob.glob(os.path.join(root, "blender.exe"))
    if candidates:
        return sorted(candidates)[-1]  # newest version
    return "blender"  # hope it's on PATH


DEFAULT_BLENDER = find_blender()

# A shot counts as locked-off below this much median feature displacement
# (px, measured at 1024px wide by shot_motion). Real handheld/dolly work sits
# well above it — the shot that triggered this being named was ~21px — so the
# gap between "static" and "moving" is wide and this needs no tuning.
STATIC_MOTION_PX = 2.0


def py_cmd(module):
    """Command prefix to run one of our pipeline modules as a subprocess.
    Works from source AND inside a frozen (PyInstaller) app, where there is
    no python.exe — the app re-invokes itself with --run <module>."""
    if getattr(sys, "frozen", False):
        return [sys.executable, "--run", module]
    return [sys.executable, os.path.join(HERE, module + ".py")]


def parse_args():
    p = argparse.ArgumentParser(description="Clip -> person-masked camera track -> your Blender scene -> render.")
    p.add_argument("footage", help="Video file (edited sequences are fine; they get split)")
    p.add_argument("--shot", type=int, help="Which shot number to track (omit to just list shots)")
    p.add_argument("--scene", help="Your .blend to put the tracked camera into")
    p.add_argument("--start", default="0,0,0", help="Camera position at first frame, e.g. 0,-8,2")
    p.add_argument("--rotation", default="0,0,0", help="Extra path rotation in degrees")
    p.add_argument("--scale", default="1.0", help="Scale of the camera motion")
    p.add_argument("--render", help="Render output (.mp4 = video, no extension = PNG sequence)")
    p.add_argument("--engine", default=None,
                   help="cycles | eevee | workbench. Default: whatever the "
                        ".blend is set to — forcing an engine breaks lighting "
                        "(a Cycles-lit scene rendered in Eevee comes out ~2x "
                        "dark). Eevee is still the automatic fallback if the "
                        "scene's engine can't run.")
    p.add_argument("--samples", help="Render samples")
    p.add_argument("--percent", help="Resolution percentage")
    p.add_argument("--frames", help="Render frame range A-B")
    p.add_argument("--transparent", action="store_true", help="Transparent background (PNG output)")
    p.add_argument("--tracking-settings", default='{"motion_model": "LocRotScale", "pattern_size": 31, "clean_error_threshold": 3.0}',
                   help="JSON tracking settings passed to stage 2")
    p.add_argument("--lens-mm", help="Known lens focal length (mm). If set, "
                   "the solver uses it as a fixed known value instead of "
                   "refining focal length.")
    p.add_argument("--focus-distance", help="Camera depth-of-field focus "
                   "distance (scene units) for the final camera.")
    p.add_argument("--static", action="store_true",
                   help="Hint that the shot is locked-off: place a static "
                        "camera at --start/--rotation instead of solving. "
                        "The shot's motion is measured either way, and this "
                        "hint is refused if the footage clearly moves — pass "
                        "--force-static to override that check. Without this "
                        "flag the static path is chosen automatically when "
                        "there is no camera movement.")
    p.add_argument("--force-static", action="store_true",
                   help="Place a static camera even if the shot measures as "
                        "moving. Escape hatch for footage the motion "
                        "estimate reads wrong; implies --static.")
    p.add_argument("--live", action="store_true",
                   help="Render in a visible Blender window instead of "
                        "headless, so you can watch frames appear. Costs GPU "
                        "time and ~0.8GB of VRAM; off by default.")
    p.add_argument("--solve-timeout", type=int, default=1200,
                   help="Seconds before stage 2 is considered hung and the "
                        "shot falls back to the 2D flow path (default 1200). "
                        "Blender's bundle adjustment can thrash on a "
                        "degenerate problem without ever converging or "
                        "erroring — measured at 8.5 hours on a 34-frame shot. "
                        "A real solve finishes in minutes.")
    p.add_argument("--stage-timeout", type=int, default=3600,
                   help="Seconds before a masking/tracking stage is treated "
                        "as hung (default 3600). Rendering is deliberately "
                        "unbounded — long renders are legitimate.")
    p.add_argument("--no-comp", action="store_true",
                   help="Skip stage 5 (compositing CG behind the actors "
                        "after the render). On by default because the "
                        "composited shot IS the deliverable.")
    p.add_argument("--tracker", default="cotracker",
                   help="learned-tracking backend (default cotracker). "
                        "cotracker is non-commercial; permissive Apache-2.0 "
                        "backends (tapnext/bootstapir/locotrack) get wired in "
                        "for commercial use. Reverting is just --tracker "
                        "cotracker — no code change.")
    p.add_argument("--no-cotracker", action="store_true",
                   help="Skip the learned tracking front-end and use the "
                        "classic detect+KLT one. The learned front-end is "
                        "worth a tripod-vs-perspective solve on soft footage, "
                        "so this is for debugging or comparison.")
    p.add_argument("--cotracker-max-frames", type=int, default=400,
                   help="Shots longer than this use the classic front-end: "
                        "the offline tracker holds the whole clip in VRAM, "
                        "and windowing it would break tracks across the "
                        "boundary the solver's keyframes need to span "
                        "(default 400).")
    p.add_argument("--masking-model", default="best",
                   help="Person-masking model: 'fast' (yolo11n, quick) or "
                        "'best' (yolo11x, detects costumes/unusual figures — "
                        "slower but far more reliable). Default best.")
    p.add_argument("--blender", default=DEFAULT_BLENDER, help="Path to blender.exe")
    return p.parse_args()


def resolve_masking_model(choice):
    """Map fast/best to the bundled model file (falls back to name so
    ultralytics downloads it if the file isn't present)."""
    name = {"fast": "yolo11n-seg.pt", "best": "yolo11x-seg.pt"}.get(
        choice, choice)
    local = os.path.join(HERE, name)
    return local if os.path.exists(local) else name


def _inherit_io():
    """Explicit stdout/stderr handles for child processes (Blender, the
    masking/flow steps). In the frozen windowed app the default handle
    inheritance drops them, so child output would vanish and the render
    queue's live progress parser would see nothing. Passing our own streams
    (they wrap real OS fds — the queue's pipe, or the log file) makes every
    child's output flow to whoever is watching."""
    kw = {}
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name)
        try:
            stream.flush()
            stream.fileno()          # must be a real OS handle
            kw[name] = stream
        except Exception:
            pass                     # no usable handle — let the child default
    return kw


def run(cmd, what):
    print(f"\n=== {what} ===")
    result = subprocess.run(cmd, **_inherit_io())
    if result.returncode != 0:
        sys.exit(f"FAILED at: {what} (exit {result.returncode}). See output above.")


def render_landed(render_path):
    """Did stage 4 actually produce output?

    An .mp4 lands at render_path itself, but a PNG sequence lands at
    "<render_path>_0001.png", "..._0002.png" — render_path never exists as a
    file. Testing os.path.exists(render_path) therefore marks every
    SUCCESSFUL png render as a failure, which re-renders the whole shot in
    Eevee and then reports that rendering did not complete.
    """
    if os.path.splitext(render_path)[1]:      # .mp4 and friends: one file
        return os.path.exists(render_path)
    return bool(glob.glob(render_path + "_*.png"))


def do_comp(args, shot, masks_dir, solve_json=None, solve_mode=None):
    """Stage 5: composite the rendered CG behind the actors — the deliverable.

    Best-effort by design, like the learned tracker: soft alpha mattes come
    from RobustVideoMatting when it can run (GPU, weights — note its GPL-3.0
    license before commercial distribution), and the comp falls back to the
    grown/feathered SAM2 masks otherwise. Measured on shot 19: the binary
    masks leave a visible halo band around the silhouette where the grown
    matte drags backdrop in; RVM's soft alpha has no halo and keeps fabric
    fringe. Output lands next to the render:  <render_dir>/comp/
    """
    if not args.render or args.no_comp:
        return
    cg_dir = os.path.dirname(os.path.abspath(args.render))
    import re as _re
    rels = [int(m.group(1)) for f in os.listdir(cg_dir)
            if (m := _re.search(r"_(\d{4})\.png$", f))]
    if not rels:
        print("[comp] no rendered PNG frames found — comp skipped "
              "(mp4 renders can't be composited per-frame)")
        return
    a = shot["frame_start"]
    b = a + max(rels) - 1
    comp_dir = os.path.join(cg_dir, "comp")
    alpha_dir = os.path.join(cg_dir, "alpha")
    have_alpha = run_ok(
        py_cmd("matte_people") + [args.footage_abs, alpha_dir,
                                  "--frames", f"{a}-{b}"],
        "Stage 5a: soft person mattes (RVM)", timeout=args.stage_timeout)
    cmd = py_cmd("comp_stage5") + [args.footage_abs, cg_dir,
                                   masks_dir or "", comp_dir,
                                   "--frames", f"{a}-{b}", "--preview"]
    # Warp the CG through the solve's lens model so it sits in the plate's
    # distorted image space — without it the comp misaligns toward the frame
    # edges (measured at ~400px in the corners on one real solve). The QC
    # step wrote the model next to the track log; no solve dump (static or
    # 2D-flow shots) means no warp, which is correct for those.
    #
    # Only warp when the ACCEPTED solve is a real 3D one (perspective/tripod).
    # A 2D-flow shot renders with a faux flow camera, but its tracked .blend
    # still holds the rejected 3D reconstruction stage 2 tried — and
    # dump_solve grabs THAT reconstruction's distortion. Applying a discarded
    # solve's lens model to a flow render bends the CG edges (visible bowing).
    # So gate the warp on the accepted mode, not on the mere existence of a
    # solve dump.
    warp_ok = solve_mode in ("perspective", "tripod")
    if solve_json and os.path.exists(solve_json) and warp_ok:
        cmd += ["--solve-json", solve_json]
    elif solve_json and os.path.exists(solve_json):
        print(f"[comp] solve mode is {solve_mode or 'none'} — not warping CG "
              "(the distortion dump belongs to a rejected 3D solve, not the "
              "camera that rendered)")
    if have_alpha:
        cmd += ["--alpha-dir", alpha_dir]
    elif masks_dir:
        print("[comp] RVM unavailable — using feathered SAM2 masks "
              "(expect a slight edge halo)")
    else:
        print("[comp] no matte source (RVM failed, no masks) — comp "
              "skipped; the render is still in place")
        return
    if run_ok(cmd, "Stage 5: compositing CG behind the actors",
              timeout=args.stage_timeout):
        print(f"Comp elements:   {comp_dir}")
        print(f"                 bg/ (CG) + fg/ (actor RGBA) + matte/ — in "
              "Resolve, fg on V2 over bg on V1")


def do_render(args, scene_out):
    """Stage 4 with an engine safety net: try the chosen engine, and if
    Cycles can't run on this machine (no GPU/driver, out of memory, …) fall
    back to Eevee so the shot still produces a video. Returns True when the
    output landed."""
    if not args.render:
        return True
    if not scene_out:
        print("[warn] --render needs a scene; skipping render")
        return False
    render_path = os.path.abspath(args.render)
    # Engine list. `None` means "use whatever the .blend is set to" — which is
    # the RIGHT default, because forcing an engine silently breaks lighting:
    # these scenes are lit for Cycles with an HDR world, and rendering them in
    # Eevee (no baked light probes = no GI) came out HALF as bright (measured
    # 43 vs 86 mean on shot 19). The old default of "eevee" turned every
    # Cycles scene dark. Eevee stays only as the automatic fallback if the
    # scene's own engine can't run at all.
    if args.engine:
        engines = [args.engine]
        if args.engine.lower() == "cycles":
            engines.append("eevee")
    else:
        engines = [None, "eevee"]      # None = the .blend's own engine
    for eng in engines:
        # --factory-startup: don't load the user's addons into the render
        # (asset-library addons poll servers and slow it down). It keeps the
        # SCENE's color management (view transform, etc.) — those live in the
        # .blend, not preferences — so AgX/Filmic still apply.
        cmd = [args.blender, "--factory-startup"]
        if not args.live:
            # -b (headless) is the right default for a queue: no window, no
            # GL context, and no render-display buffer — which at 4K is VRAM
            # the scene needs (measured: 6.6GB headless vs 7.4GB windowed, of
            # 8GB). --live puts the watch-it-happen window back.
            cmd.insert(1, "-b")
        cmd += [scene_out, "-P", os.path.join(HERE, "render_stage4.py"), "--",
                "--out", render_path]
        if eng:
            cmd += ["--engine", eng]
        for flag in ("samples", "percent", "frames"):
            if getattr(args, flag):
                cmd += ["--" + flag, getattr(args, flag)]
        if args.transparent:
            cmd.append("--transparent")
        label = eng or "scene's engine"
        if run_ok(cmd, f"Stage 4: rendering ({label})") and \
                render_landed(render_path):
            return True
        print(f"[warn] render with {label} failed"
              + (" — trying Eevee" if eng != engines[-1] else ""))
    print("[warn] rendering did not complete")
    return False


STAGE_TIMES = []  # (label, seconds) — printed as a breakdown at DONE


def run_ok(cmd, what, timeout=None):
    """Run a stage but DON'T abort on failure — return True/False so the
    caller can fall back. Used for every stage that has a safety net, so a
    single crashing shot never kills the batch: it degrades to the next
    approach (3D solve -> 2D flow -> static hold; Cycles -> Eevee).

    `timeout` (seconds) covers the case the fallbacks could not: a stage that
    neither crashes nor finishes. Blender's Ceres bundle adjustment can hit a
    degenerate problem and thrash forever — observed on a 34-frame shot that
    logged "Step failed to evaluate. Treating it as a step with infinite cost"
    for 8.5 HOURS without converging or giving up. A crash degrades gracefully;
    a hang stalls the whole queue overnight with nothing to show. Pass None
    only for stages that are legitimately open-ended (rendering).
    """
    print(f"\n=== {what} ===")
    t0 = time.time()
    try:
        ok = subprocess.run(cmd, timeout=timeout,
                            **_inherit_io()).returncode == 0
    except subprocess.TimeoutExpired:
        print(f"[warn] {what} exceeded {timeout}s and was stopped — this "
              "stage is not converging; falling back to the next approach")
        ok = False
    except Exception as e:
        print(f"[warn] {what} could not run: {e}")
        ok = False
    dt = time.time() - t0
    STAGE_TIMES.append((what, dt))
    print(f"[timing] {what}: {dt:.0f}s")
    return ok


def print_timing_summary():
    if not STAGE_TIMES:
        return
    total = sum(t for _, t in STAGE_TIMES)
    print("\nWhere the time went:")
    for label, t in STAGE_TIMES:
        print(f"  {t:6.0f}s  ({100 * t / max(total, 0.01):3.0f}%)  {label}")
    print(f"  {total:6.0f}s  total")


def shot_motion(footage, start, end, gap=12, windows=4):
    """Motion estimate inside a shot: median tracked-feature displacement
    over `gap` frames (px at 1024w), max across sample windows.

    Uses optical flow on feature points rather than whole-frame phase
    correlation — push-ins, pull-outs, zooms and rotations move features
    radially with near-zero NET shift, so phase correlation reports them
    as static (missed 2 of 4 moving shots on real footage)."""
    import cv2
    import numpy as np

    def grab(cap, idx, w=1024):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx - 1)
        ok, f = cap.read()
        if not ok:
            return None
        h = int(f.shape[0] * w / f.shape[1])
        return cv2.cvtColor(cv2.resize(f, (w, h)), cv2.COLOR_BGR2GRAY)

    n = end - start + 1
    if n < gap + 2:
        gap = max(2, n - 2)
    cap = cv2.VideoCapture(footage)
    positions = np.linspace(start, end - gap,
                            min(windows, max(1, n // gap))).astype(int)
    per_window = []
    for pos in positions:
        a = grab(cap, int(pos))
        b = grab(cap, int(pos) + gap)
        if a is None or b is None:
            continue
        pts = cv2.goodFeaturesToTrack(a, maxCorners=200, qualityLevel=0.01,
                                      minDistance=12)
        if pts is None or len(pts) < 10:
            continue
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None,
                                              winSize=(21, 21), maxLevel=3)
        good = st.ravel() == 1
        if good.sum() < 10:
            continue
        disp = np.linalg.norm((nxt - pts).reshape(-1, 2)[good], axis=1)
        per_window.append(float(np.median(disp)))
    cap.release()
    return max(per_window) if per_window else None


def main():
    args = parse_args()
    footage = os.path.abspath(args.footage)
    args.footage_abs = footage
    if not os.path.exists(footage):
        sys.exit(f"Footage not found: {footage}")

    base = os.path.splitext(os.path.basename(footage))[0].replace(" ", "_")
    workdir = os.path.join(os.path.dirname(footage), base + "_autotrack")
    shots_dir = os.path.join(workdir, "shots")
    shots_json = os.path.join(shots_dir, "shots.json")

    # ---- Stage 0: split into shots (reused if already done) ----
    if not os.path.exists(shots_json):
        run(py_cmd("split_shots") + [footage, shots_dir],
            "Stage 0: splitting footage into shots")
    with open(shots_json) as f:
        shots_data = json.load(f)
    shots = shots_data["shots"]
    source_size = shots_data.get("size")  # original plate resolution

    # ---- No --shot: list shots with a motion estimate and stop ----
    if args.shot is None:
        print(f"\n{len(shots)} shot(s) found in {os.path.basename(footage)}:\n")
        for s in shots:
            m = shot_motion(footage, s["frame_start"], s["frame_end"])
            move = "?" if m is None else f"{m:.1f}"
            verdict = ("  <- static, can't be 3D-tracked"
                       if m is not None and m < STATIC_MOTION_PX else "")
            print(f"  shot {s['shot']}: frames {s['frame_start']}-{s['frame_end']}"
                  f" ({s['num_frames']} frames), camera motion ~{move} px{verdict}")
        print(f"\nPick a moving shot and re-run with:  --shot N")
        return

    shot = next((s for s in shots if s["shot"] == args.shot), None)
    if shot is None:
        sys.exit(f"No shot {args.shot}; footage has shots 1-{len(shots)}")
    shot_file = shot.get("file") or shot_file_for(shots_dir, args.shot)
    tag = f"shot_{args.shot:02d}"

    # ---- First decision: does the camera move at all? ----
    # Static shot -> place a locked-off camera at the chosen pose, no solve.
    # Moving shot -> 3D track, falling back to the 2D motion match.
    # If static placement fails for any reason, fall through to the tracking
    # chain — its own fallbacks guarantee the shot still gets a camera.
    if args.scene:
        # Measure motion ALWAYS — including when --static was asked for.
        # A --static hint used to skip this measurement entirely, so a shot
        # mis-flagged in the UI silently rendered a frozen camera over moving
        # footage: 769 identical frames, ~21 hours, and a comp that could
        # never line up. Nothing downstream could catch it either, because
        # place_static.py reports solve_ok=true. The measurement is a few
        # seconds; a wrong static call costs a day.
        try:
            m = shot_motion(footage, shot["frame_start"], shot["frame_end"])
        except Exception as e:  # can't measure -> assume moving, track it
            print(f"[warn] motion measurement failed ({e}) — "
                  "treating the shot as moving")
            m = None

        is_static = args.static or args.force_static
        if m is not None and m < STATIC_MOTION_PX:
            if not is_static:
                print(f"\nShot {args.shot} has ~{m:.1f}px of camera motion — "
                      "treating it as a locked-off (static) shot.")
            is_static = True
        elif is_static and m is not None and not args.force_static:
            # The hint disagrees with the footage. Trust the footage.
            print(f"\n[REFUSED] Shot {args.shot} was flagged static, but it "
                  f"measures ~{m:.1f}px of camera motion "
                  f"(static is < {STATIC_MOTION_PX}px).")
            print("          A static camera here would render one frozen "
                  "image over a moving plate — the CG would slide against "
                  "the background and the comp would not hold.")
            print("          Tracking it instead. Pass --force-static if the "
                  "measurement is wrong for this shot.")
            is_static = False
        if is_static:
            scene = os.path.abspath(args.scene)
            base = os.path.splitext(os.path.basename(scene))[0]
            scene_out = os.path.join(workdir, f"{tag}_{base}_tracked.blend")
            out_dir = os.path.join(workdir, tag + "_out")
            os.makedirs(out_dir, exist_ok=True)
            place = [args.blender, "-b", scene,
                     "-P", os.path.join(HERE, "place_static.py"), "--",
                     "--footage", shot_file,
                     "--start", args.start, "--rotation", args.rotation,
                     "--frames", str(shot["num_frames"]),
                     "--out", scene_out,
                     "--log", os.path.join(out_dir,
                                           tag + "_masked_track_log.json")]
            if args.lens_mm:
                place += ["--focal-mm", args.lens_mm]
            if args.focus_distance:
                place += ["--focus-distance", args.focus_distance]
            if source_size:
                place += ["--render-size", f"{source_size[0]}x{source_size[1]}"]
            if run_ok(place, "Static shot: placing locked-off camera") and \
                    os.path.exists(scene_out):
                # One frame is the whole render. The camera is locked and
                # nothing in the scene animates, so every frame of a static
                # shot is the same image — rendering the full range just
                # writes that image N times. Hold the single frame for the
                # shot's length in the comp instead.
                if args.render:
                    n = shot["num_frames"]
                    if args.frames and args.frames != "1-1":
                        print(f"[info] static shot: ignoring --frames "
                              f"{args.frames}; one frame is all there is")
                    args.frames = "1-1"
                    print(f"[info] static shot: rendering 1 frame instead of "
                          f"{n} identical ones")
                do_render(args, scene_out)
                do_comp(args, shot, None)
                print("\n=== DONE ===")
                print(f"Work folder:     {workdir}")
                print("Solve:           locked-off static camera (no motion)")
                print(f"Your scene:      {scene_out}  (camera: TrackedCamera)")
                if args.render:
                    print(f"Render:          {os.path.abspath(args.render)}"
                          f"  (single frame — hold it for "
                          f"{shot['num_frames']} frames)")
                print_timing_summary()
                return
            print("[warn] static placement failed — falling back to the "
                  "tracking chain")

    # ---- Stage 1: person masks (reused if already done) ----
    # best = SAM2 silhouette tracking seeded+unioned with yolo11x (temporally
    # stable, catches costumes); fast = classic per-frame yolo11n.
    masks_dir = os.path.join(workdir, tag + "_masks")
    if not os.path.exists(os.path.join(masks_dir, "manifest.json")):
        engine = "yolo" if args.masking_model == "fast" else "sam2"
        run_ok(py_cmd("segment_people") + [shot_file, masks_dir,
                                           "--engine", engine,
                                           "--model", resolve_masking_model(args.masking_model)],
               "Stage 1: learning person vs background (AI masking)")
    use_masks = os.path.exists(os.path.join(masks_dir, "manifest.json"))
    if not use_masks:
        print("[warn] person masking unavailable — tracking the whole frame")

    # ---- Stage 2: track + solve ----
    out_dir = os.path.join(workdir, tag + "_out")
    tset = json.loads(args.tracking_settings)
    if args.lens_mm:  # known lens: use it fixed, don't refine focal length
        tset["focal_length_mm"] = float(args.lens_mm)
        tset["refine_focal"] = False
    # ---- Stage 2: classic first, learned retry -----------------------------
    # Two front-ends feed the same solver, and they win on different footage:
    #   classic KLT   more PRECISE on sharp footage (lab 2: 1.53px vs 3.45px —
    #                 CoTracker's accuracy floor is ~3.5px regardless of
    #                 processing width; measured at 512/768/1024)
    #   learned       more ROBUST on soft/blurred footage, where KLT finds
    #                 almost nothing off the actors (shot 09: no solve vs
    #                 tripod 1.91px on 200 tracks; shot 19: 6 tracks vs 200)
    # So: run the precise one, and only if its solve is missing or weak, retry
    # with the robust one and keep the better result. This extends the
    # existing fallback chain to: classic 3D -> learned 3D -> 2D flow ->
    # static hold.

    def stage2_run(dest, points_json=None, label="classic detect+KLT",
                   attempt="auto"):
        os.makedirs(dest, exist_ok=True)
        s = dict(tset)
        s["solve_attempt"] = attempt
        if points_json:
            s["points_json"] = points_json
        cmd = [args.blender, "-b", os.path.join(HERE, "template.blend"),
               "-P", os.path.join(HERE, "auto_track_stage2.py"), "--",
               shot_file, dest]
        if use_masks:
            cmd += ["--masks", masks_dir]
        cmd += ["--settings", json.dumps(s)]
        ok = run_ok(cmd, f"Stage 2: tracking camera ({label}, {attempt})",
                    timeout=args.solve_timeout)
        m = {}
        try:
            with open(os.path.join(dest, tag + "_masked_track_log.json")) as f:
                m = json.load(f)
        except Exception:
            ok = False
        return ok, m, m.get("average_solve_error"), m.get("solve_mode")

    def solve_rank(mode, e):
        """Orderable quality: perspective beats tripod beats nothing, and
        within a mode a lower error wins."""
        if e is None:
            return (0, 0.0)
        return (2 if mode == "perspective" else 1, -e)

    tripod_at = tset.get("tripod_fallback_error", 8.0)

    def stage2_attempts(base_dir, points_json=None, label="classic detect+KLT"):
        """auto -> manual -> tripod, ONE pristine Blender process each.

        Solving twice in one process is unreliable — Blender's solver carries
        hidden state between solves (the same markers that solved at 4.38px
        re-solved at 492.55px moments later, and a fifth in-process attempt
        returned 13 million px). First solves in a fresh process are
        consistently sane, so every attempt gets its own process, and the
        best result's files are copied into place.
        """
        best = (None, {}, None, None, None)   # rank-holder: ok,m,err,mode,dir
        for attempt in ("auto", "manual", "tripod"):
            adir = os.path.join(base_dir + "_attempts", attempt)
            ok, m, e, md = stage2_run(adir, points_json, label, attempt)
            if ok and solve_rank(md, e) > solve_rank(best[3], best[2]):
                best = (ok, m, e, md, adir)
            # A clean perspective solve is the ceiling — stop early. tripod
            # can't beat it, and manual only exists to rescue auto's failures.
            if md == "perspective" and e is not None and e <= tripod_at:
                break
        if best[4]:
            import shutil
            os.makedirs(base_dir, exist_ok=True)
            for fn in os.listdir(best[4]):
                shutil.copy2(os.path.join(best[4], fn),
                             os.path.join(base_dir, fn))
        return best[0] or False, best[1], best[2], best[3]

    stage2_ok, metrics, err, mode2 = stage2_attempts(out_dir)
    classic_good = (mode2 == "perspective" and err is not None
                    and err <= tripod_at)

    if not classic_good and not args.no_cotracker:
        print(f"[stage2] classic front-end result: "
              f"{mode2 or 'no solve'}"
              + (f" {err:.2f}px" if err is not None else "")
              + " — retrying with the learned front-end")
        points_json = os.path.join(out_dir, tag + "_cotrack.json")
        os.makedirs(out_dir, exist_ok=True)
        ct_cmd = py_cmd("cotrack_points") + [shot_file, points_json]
        if use_masks:
            ct_cmd += ["--masks", masks_dir]
        ct_cmd += ["--max-frames", str(args.cotracker_max_frames),
                   "--tracker", args.tracker]
        if run_ok(ct_cmd, "Stage 2a: learned point tracking (background)",
                  timeout=args.stage_timeout) and os.path.exists(points_json):
            ct_dir = out_dir + "_ct"
            ok2, m2, err2, mode22 = stage2_attempts(
                ct_dir, points_json, label="learned front-end")
            if ok2 and solve_rank(mode22, err2) > solve_rank(mode2, err):
                print(f"[stage2] learned front-end wins: "
                      f"{mode22} {err2:.2f}px vs "
                      f"{mode2 or 'no solve'}"
                      + (f" {err:.2f}px" if err is not None else ""))
                import shutil
                for fn in os.listdir(ct_dir):
                    shutil.copy2(os.path.join(ct_dir, fn),
                                 os.path.join(out_dir, fn))
                stage2_ok, metrics, err, mode2 = ok2, m2, err2, mode22
            else:
                print("[stage2] keeping the classic front-end's result")
        else:
            print("[warn] learned tracking unavailable for this shot — "
                  "keeping the classic result")

    log_path = os.path.join(out_dir, tag + "_masked_track_log.json")
    if not stage2_ok or err is None:
        print("[warn] 3D tracking didn't produce a usable solve — falling back "
              "to the 2D motion match")
    mode = metrics.get("solve_mode", "perspective")
    tracked_blend = os.path.join(out_dir, tag + "_masked_tracked.blend")

    # ---- QC overlay: see the solve, don't just read its number -------------
    # Reprojects the solved bundles over the footage (distortion-aware) next
    # to the tracked markers: dots riding their crosses = locked; sliding =
    # drift. Best-effort — a failed QC render never blocks the shot.
    if err is not None and os.path.exists(tracked_blend):
        qc_dir = os.path.join(out_dir, tag + "_qc")
        solve_json = os.path.join(qc_dir, "solve.json")
        os.makedirs(qc_dir, exist_ok=True)
        if run_ok([args.blender, "-b", tracked_blend, "-P",
                   os.path.join(HERE, "dump_solve.py"), "--", solve_json],
                  "QC: dumping solve", timeout=args.stage_timeout)                 and run_ok(py_cmd("qc_render") + [shot_file, solve_json,
                                                  qc_dir],
                           "QC: rendering overlay",
                           timeout=args.stage_timeout):
            print(f"QC overlay:      {os.path.join(qc_dir, 'qc.mp4')}")
    flow_json = None
    # A 3D solve worse than this jitters too much to trust; treat it as a
    # failure and hand the shot to the best-effort 2D motion match instead —
    # a faux camera that follows the movement beats a garbage 3D one.
    BAD_SOLVE_PX = 8.0
    solve_failed = err is None or err != err        # no 3D solve at all
    solve_unusable = solve_failed or err >= BAD_SOLVE_PX
    solve_3d_error = None if solve_failed else err   # keep for the log
    if solve_unusable:                               # try the 2D flow fallback
        reason = ("3D solve not possible" if solve_failed
                  else f"3D solve too rough ({err:.1f}px) to trust")
        print(f"\n{reason} — using 2D flow motion match instead…")
        flow_json = os.path.join(out_dir, tag + "_flow_solve.json")
        os.makedirs(out_dir, exist_ok=True)
        flow_cmd = py_cmd("flow_solve") + [shot_file, flow_json]
        # Pass the known lens. flow_solve defaults to 35mm, and it uses the
        # focal to build K — rot_from_homography divides measured image motion
        # by it, so a wrong focal scales every recovered pan/tilt by the same
        # factor (a 14mm plate solved as 35mm under-rotates ~2.5x). The focal
        # also lands in the json and becomes the baked camera's lens, so a
        # wrong guess gives the CG camera the wrong FOV as well. Every other
        # stage is already told the lens; this one was the exception.
        if args.lens_mm:
            flow_cmd += ["--focal-mm", str(args.lens_mm)]
        if use_masks:
            flow_cmd += ["--masks", masks_dir]
        flow_ok = run_ok(flow_cmd, "Stage 2c: 2D flow solve (approximate)")
        if not flow_ok or not os.path.exists(flow_json):
            # absolute last resort: a static hold that cannot fail, so the
            # shot still yields a usable (if motionless) camera + render
            print("[warn] flow solve failed — using a static hold camera")
            nf = shot["num_frames"]
            with open(flow_json, "w") as f:
                json.dump({"solver": "static-hold", "num_frames": nf,
                           "fps": 24.0, "size": source_size or [1920, 1080],
                           "focal_mm": float(args.lens_mm) if args.lens_mm else 35.0,
                           "sensor_width_mm": 36.0,
                           "quaternions_wxyz": [[1.0, 0.0, 0.0, 0.0]] * nf,
                           "scale_cum": [1.0] * nf,
                           "median_residual_px": None,
                           "frames_without_flow": nf}, f)
        with open(flow_json) as f:
            flow = json.load(f)
        residual = flow.get("median_residual_px")
        held = flow.get("frames_without_flow", 0)
        held_frac = held / max(flow["num_frames"], 1)
        # Every shot gets a camera; the label carries the honesty. Extreme
        # blur hides CG imprecision anyway — a loose match beats a refusal.
        if residual is None:
            tier = "static hold — motion was unmeasurable on this shot"
        elif held_frac > 0.5:
            tier = (f"partial — flow lost on {held_frac:.0%} of frames "
                    "(camera holds through the gaps)")
        elif residual > 25.0:
            tier = "very loose — extreme motion blur"
        elif residual > 10.0:
            tier = "loose"
        else:
            tier = "good"
        err = residual if residual is not None else 99.0
        mode = "2d-flow"
        metrics.update(solve_mode="2d-flow", average_solve_error=err,
                       solve_ok=True, flow_json=flow_json, flow_tier=tier,
                       fell_back_from_3d=(not solve_failed),
                       rejected_3d_error=solve_3d_error)
        with open(log_path, "w") as f:
            json.dump(metrics, f, indent=2)
    if mode == "2d-flow":
        quality = (f"APPROXIMATE 2D motion match ({metrics.get('flow_tier', '')}) "
                   "— CG follows the camera's movement feel; right for "
                   "blurred/close-up shots, not for locked floor contact")
    else:
        quality = ("EXCELLENT (production-grade)" if err < 1.0 else
                   "GOOD (small drift possible)" if err < 3.0 else
                   "ROUGH (visible drift likely — hard shot)" if err < 8.0 else
                   "BAD (don't use this solve)")
        if mode == "tripod":
            quality += " [rotation-only tripod solve — camera pans in place]"
    print(f"\nSolve error: {err:.2f} px -> {quality}")

    # ---- Stage 3: bake camera into the user's scene ----
    scene_out = None
    if args.scene:
        scene = os.path.abspath(args.scene)
        scene_out = os.path.join(workdir, tag + "_" +
                                 os.path.splitext(os.path.basename(scene))[0] + "_tracked.blend")
        cmd3 = [args.blender, "-b", scene,
                "-P", os.path.join(HERE, "apply_track_stage3.py"), "--"]
        if flow_json:
            cmd3 += ["--flow", flow_json, "--footage", shot_file]
        else:
            cmd3 += [tracked_blend]
        # stage3 uses manual opt() parsing (handles leading '-' fine), so
        # pass separate tokens here — NOT the =value form.
        cmd3 += ["--start", args.start, "--rotation", args.rotation,
                 "--scale", args.scale, "--out", scene_out]
        if args.lens_mm:
            cmd3 += ["--lens-mm", args.lens_mm]
        if args.focus_distance:
            cmd3 += ["--focus-distance", args.focus_distance]
        if source_size:  # tracking ran on a proxy; render at plate resolution
            cmd3 += ["--render-size", f"{source_size[0]}x{source_size[1]}"]
        if not run_ok(cmd3, "Stage 3: baking camera into your scene"):
            print("[warn] baking the camera into your scene failed")
            scene_out = None  # nothing to render

    # ---- Stage 4: render ----
    do_render(args, scene_out)
    do_comp(args, shot, masks_dir if use_masks else None,
            solve_json=os.path.join(out_dir, tag + "_qc", "solve.json"),
            solve_mode=mode)

    print("\n=== DONE ===")
    print(f"Work folder:     {workdir}")
    print(f"Solve:           {err:.2f} px ({quality})")
    print(f"Tracked blend:   {tracked_blend}")
    if scene_out:
        print(f"Your scene:      {scene_out}  (camera: TrackedCamera)")
    if args.render:
        print(f"Render:          {os.path.abspath(args.render)}")
    print_timing_summary()


if __name__ == "__main__":
    main()
