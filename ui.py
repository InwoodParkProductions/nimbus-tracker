"""
AeroTrack — desktop app for the camera-tracking pipeline.
=========================================================
Double-click the AeroTrack shortcut (or run `python ui.py`) — opens as a
native frameless window with its own title bar. Flags:
    --server-only   run the web server without a window (for testing)

Flow: open a clip (native file dialog) -> it finds the shots -> pick a shot
-> (optionally) give your scene .blend + camera start -> Track -> live
progress -> results.
"""

import glob
import json
import os
import re
import subprocess
import sys
import threading
import time

import cv2
from flask import Flask, request, redirect, send_file, abort, jsonify

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from auto_track import (find_blender, py_cmd, render_landed,  # noqa: E402
                        STATIC_MOTION_PX)
from split_shots import shot_file_for  # noqa: E402

BLENDER = None  # resolved after settings helpers are defined (see below)
# starting folder for file dialogs; falls back to Videos/home elsewhere
DIALOG_START_DIR = next(
    (p for p in (r"C:\Users\Inwoo\Documents\screenshots for vfx",
                 os.path.join(os.path.expanduser("~"), "Videos"),
                 os.path.expanduser("~"))
     if os.path.isdir(p)), "")
PORT = 8765
APP_NAME = "Nimbus Tracker"
APP_VERSION = "3.0"


def _data_dir():
    """A per-user writable folder for settings/recents, so the app never has
    to write inside its own install folder (which may be read-only, e.g. when
    unzipped into Program Files). Falls back to the app folder if AppData is
    somehow unavailable."""
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    d = os.path.join(base, "NimbusTracker")
    try:
        os.makedirs(d, exist_ok=True)
        probe = os.path.join(d, ".write_test")
        open(probe, "w").close()
        os.remove(probe)
        return d
    except Exception:
        return HERE


DATA_DIR = _data_dir()


def _cfg(name):
    """Config file path in the writable data dir; migrates a legacy copy that
    an older build may have written next to the app."""
    new = os.path.join(DATA_DIR, name)
    legacy = os.path.join(HERE, name)
    if not os.path.exists(new) and os.path.exists(legacy):
        try:
            import shutil
            shutil.copy2(legacy, new)
        except Exception:
            return legacy
    return new


SETTINGS_PATH = _cfg("nimbus_settings.json")
RECENTS_PATH = _cfg("recent_clips.json")
PLACEMENTS_PATH = _cfg("placements.json")


def load_settings():
    if os.path.exists(SETTINGS_PATH):
        with open(SETTINGS_PATH) as f:
            return json.load(f)
    return {}


def resolve_blender():
    override = load_settings().get("blender_path", "")
    if override and os.path.exists(override):
        return override
    return find_blender()


def save_settings(s):
    with open(SETTINGS_PATH, "w") as f:
        json.dump(s, f, indent=2)


def remember_last(scene, form):
    """Persist last-used scene + render settings to prefill next time."""
    s = load_settings()
    if scene:
        s["last_scene"] = scene
    s["last_engine"] = form.get("engine", "eevee")
    s["last_samples"] = form.get("samples", "64")
    s["last_percent"] = form.get("percent", "100")
    try:
        save_settings(s)
    except Exception:
        pass


BLENDER = resolve_blender()

app = Flask(__name__)
webview_window = None  # set when running as a native window


def notify(done_ok=True):
    """Audible cue + bring the window forward when a long job finishes, so
    the user can walk away and still be alerted."""
    try:
        import winsound
        winsound.MessageBeep(winsound.MB_ICONASTERISK if done_ok
                             else winsound.MB_ICONHAND)
    except Exception:
        pass
    if webview_window is not None:
        try:
            webview_window.on_top = True
            webview_window.on_top = False
        except Exception:
            pass

# ------------------------------------------------------------------ job runner
job = {"proc": None, "log": [], "status": "idle", "kind": None, "meta": {},
       "render_total": 0, "render_frame": 0, "render_t0": None,
       "render_spf": None}
job_lock = threading.Lock()

# Background (-b) Blender doesn't print "Fra:N" progress lines. It prints
# "…| Video append frame N" (video out) or "…| Saved: '…NNNN.png'" (image
# sequence), each prefixed with Blender's own elapsed time "MM:SS.ss" or
# "HH:MM:SS.ss". We parse the frame number AND Blender's elapsed clock so the
# ETA is based on Blender's real per-frame timing.
FRAME_RE = re.compile(r"Video append frame (\d+)|Saved:\s*'.*?(\d+)\.\w+'")
BTIME_RE = re.compile(r"^\s*(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)\s")
MASK_RE = re.compile(r"\[segment\] frame (\d+)")  # masking progress during track


def blender_elapsed(line):
    m = BTIME_RE.match(line)
    if not m:
        return None
    h = int(m.group(1)) if m.group(1) else 0
    return h * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def parse_frame(line):
    m = FRAME_RE.search(line)
    if m:
        return int(m.group(1) or m.group(2))
    return None


def start_job(cmd, kind, meta):
    with job_lock:
        if job["status"] == "running" or queue_state.get("running"):
            return False
        job.update(proc=None, log=[], status="running", kind=kind, meta=meta,
                   render_total=meta.get("render_total", 0), render_frame=0,
                   render_t0=None, render_spf=None)

    def run():
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, text=True,
                                    encoding="utf-8", errors="replace",
                                    cwd=HERE, bufsize=1)
            job["proc"] = proc
            first_frame = first_bt = None
            for line in proc.stdout:
                job["log"].append(line.rstrip())
                fr = parse_frame(line)
                if fr is not None:  # live render progress → sec/frame → ETA
                    bt = blender_elapsed(line)
                    if first_frame is None:
                        first_frame, first_bt = fr, bt
                        job["render_t0"] = time.time()
                    job["render_frame"] = fr
                    done = fr - first_frame
                    if done > 0:
                        # prefer Blender's own clock; fall back to wall time
                        if bt is not None and first_bt is not None:
                            job["render_spf"] = (bt - first_bt) / done
                        else:
                            job["render_spf"] = \
                                (time.time() - job["render_t0"]) / done
            proc.wait()
            job["status"] = "done" if proc.returncode == 0 else "failed"
        except Exception as e:
            job["log"].append(f"UI ERROR: {e}")
            job["status"] = "failed"
        if kind == "track" and meta.get("footage") and \
                meta.get("shot") is not None:
            # solves are ephemeral until the user saves them
            record_unsaved_track(meta["footage"], meta["shot"])
        notify(job["status"] == "done")

    threading.Thread(target=run, daemon=True).start()
    return True


@app.route("/job_render_status")
def job_render_status():
    total = job.get("render_total") or 0
    frame = job.get("render_frame") or 0
    spf = job.get("render_spf")
    rendering = (job["status"] == "running" and total > 0 and frame > 0)
    eta = None
    if rendering and spf:
        eta = max(0, round((total - frame) * spf))
    return jsonify({
        "rendering": rendering,
        "frame": frame, "total": total,
        "pct": round(100 * frame / total, 1) if total else 0,
        "eta_s": eta,
        "status": job["status"],
    })


# ------------------------------------------------------------- track-all batch
track_all_state = {"running": False, "footage": None, "total": 0, "done": 0,
                   "current": None, "results": []}


def start_track_all(footage, shots_todo):
    if track_all_state["running"] or job["status"] == "running":
        return False
    track_all_state.update(running=True, footage=footage,
                           total=len(shots_todo), done=0, current=None,
                           results=[])

    def run():
        for shot in shots_todo:
            if not track_all_state["running"]:
                break
            track_all_state["current"] = shot
            try:
                subprocess.run(py_cmd("auto_track") +
                               [footage, "--shot", str(shot),
                                "--blender", BLENDER, "--masking-model",
                                load_settings().get("masking_model", "best")],
                               capture_output=True, text=True, timeout=10800)
            except Exception:
                pass
            st = shot_status(footage, shot)
            track_all_state["results"].append(
                {"shot": shot, "label": st[1] if st else "failed",
                 "cls": st[0] if st else "bad"})
            track_all_state["done"] += 1
        track_all_state["running"] = False
        track_all_state["current"] = None
        notify(True)

    threading.Thread(target=run, daemon=True).start()
    return True


# ------------------------------------------------------------------ helpers
def workdir_for(footage):
    base = os.path.splitext(os.path.basename(footage))[0].replace(" ", "_")
    return os.path.join(os.path.dirname(footage), base + "_autotrack")


def flag(value):
    """Read a checkbox/flag out of a query string or form.

    Never use bool() for this. Query and form values arrive as STRINGS, and
    bool("0") is True — which is not a hypothetical: the setup page emits
    "&static=0" for a moving shot, that string round-tripped back into
    bool(request.args.get("static")), came back True, and the page re-emitted
    the shot as static=1. A moving shot was then rendered with a locked-off
    camera: 769 identical frames, ~21 hours, and a comp that could not hold.
    """
    if value is None:
        return False
    return str(value).strip().lower() not in ("", "0", "false", "no", "off")


def videos_dir():
    """The user's Videos folder (standard render destination)."""
    v = os.path.join(os.path.expanduser("~"), "Videos")
    try:
        os.makedirs(v, exist_ok=True)
    except Exception:
        v = os.path.expanduser("~")
    return v


def default_render_path(footage, shot, suffix=""):
    """Where a shot renders when the user doesn't name a path.

    A PNG sequence in a folder of its own — deliberately not an .mp4.
    Rendering straight to video means an interrupted render loses the entire
    shot (H.264 has no resume), which on a 20-hour render is the difference
    between losing a frame and losing a day. It also re-compresses every
    frame on the way out. Stage 4 writes "<this>_0001.png" and skips frames
    already on disk, so a crash or a power cut costs one frame. Encode to
    video once at the end, from the PNGs.
    """
    stem = os.path.splitext(os.path.basename(footage))[0].replace(" ", "_")
    name = f"{stem}_shot_{shot:02d}{suffix}"
    return os.path.join(videos_dir(), name, name)


def default_export_path(footage, shot, ext):
    """Camera-export path (.abc/.fbx). Separate from default_render_path so
    the export doesn't have to strip an extension off a render path that no
    longer has one."""
    stem = os.path.splitext(os.path.basename(footage))[0].replace(" ", "_")
    return os.path.join(videos_dir(), f"{stem}_shot_{shot:02d}_camera.{ext}")


def _shot_frames(footage, shot):
    """Frame count of a shot (for render ETA)."""
    sj = os.path.join(workdir_for(footage), "shots", "shots.json")
    if os.path.exists(sj):
        try:
            with open(sj) as f:
                data = json.load(f)
            s = next((x for x in data["shots"] if x["shot"] == shot), None)
            if s:
                return s["num_frames"]
        except Exception:
            pass
    return 0


def _shot_source_size(footage):
    """[w, h] of the source plate from shots.json, or None."""
    sj = os.path.join(workdir_for(footage), "shots", "shots.json")
    if os.path.exists(sj):
        try:
            with open(sj) as f:
                return json.load(f).get("size")
        except Exception:
            pass
    return None


def shot_status(footage, shot):
    """Result state for a shot: (badge_class, short_label) or None if not
    tracked yet. Reads the shot's track log."""
    tag = f"shot_{shot:02d}"
    logj = os.path.join(workdir_for(footage), tag + "_out",
                        tag + "_masked_track_log.json")
    if not os.path.exists(logj):
        return None
    try:
        with open(logj) as f:
            d = json.load(f)
    except Exception:
        return None
    err = d.get("average_solve_error")
    mode = d.get("solve_mode")
    if err is not None and err != err:  # NaN
        err = None
    if mode == "static":
        return ("ok", "✓ static camera")
    if err is None:
        return ("bad", "no solve")
    if mode == "2d-flow":
        return ("warn", f"✓ motion match {err:.1f}px")
    label = f"✓ {err:.1f}px" + ("" if mode != "tripod" else " tripod")
    cls = "ok" if err < 3 else ("warn" if err < 8 else "bad")
    return (cls, label)


def load_shots(footage):
    path = os.path.join(workdir_for(footage), "shots", "shots.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)["shots"]


def load_recents():
    if not os.path.exists(RECENTS_PATH):
        seeded = []
        if os.path.isdir(DIALOG_START_DIR):
            for wd in glob.glob(os.path.join(DIALOG_START_DIR, "*_autotrack")):
                stem = os.path.basename(wd)[:-len("_autotrack")]
                for ext in (".mov", ".mp4"):
                    cand = os.path.join(DIALOG_START_DIR,
                                        stem.replace("_", " ") + ext)
                    if os.path.exists(cand):
                        seeded.append({"path": cand, "ts": os.path.getmtime(wd)})
                        break
        save_recents(seeded)
        return seeded
    with open(RECENTS_PATH) as f:
        recents = json.load(f)
    return [r for r in recents if os.path.exists(r["path"])]


def save_recents(recents):
    with open(RECENTS_PATH, "w") as f:
        json.dump(recents[:20], f, indent=2)


def add_recent(footage):
    recents = [r for r in load_recents() if r["path"] != footage]
    recents.insert(0, {"path": footage, "ts": time.time()})
    save_recents(recents)


# ------------------------------------------------------------------ render queue
QUEUE_PATH = _cfg("render_queue.json")
render_queue = []
queue_state = {"running": False, "proc": None, "spf": None,  # sec/frame
               "started": None}
queue_lock = threading.Lock()


def save_queue():
    with open(QUEUE_PATH, "w") as f:
        json.dump(render_queue, f, indent=2)


def load_queue():
    global render_queue
    if os.path.exists(QUEUE_PATH):
        with open(QUEUE_PATH) as f:
            render_queue = json.load(f)
        for e in render_queue:  # a crash mid-render leaves stale state
            if e["status"] == "rendering":
                e["status"] = "queued"
                e["frames_done"] = 0


def _run_tracked_proc(cmd, entry, watch_stages=True,
                      render_base=0.45, render_span=0.55):
    """Run one pipeline/render subprocess, streaming its output to update the
    entry's stage and a real 0..1 progress fraction (entry['progress']).

    The whole shot is one bar: the pre-render work (masking + solve) fills up
    to ``render_base`` and the render fills the remaining ``render_span``. For
    a full track+render, masking is measurable per frame; the solve is a short
    plateau; then the render fills frame by frame. Render-only entries pass
    render_base=0 so the whole bar is the render."""
    t0 = time.time()
    total = max(entry["frames_total"], 1)

    def set_prog(p):
        entry["progress"] = max(entry.get("progress", 0.0), min(p, 1.0))

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", bufsize=1)
        queue_state["proc"] = proc
        first_frame = first_bt = None
        for line in proc.stdout:
            if watch_stages:
                mm = MASK_RE.search(line)
                if mm:  # masking pass — real per-frame progress up to 30%
                    entry["stage"] = "tracking"
                    set_prog(min(int(mm.group(1)) / total, 1.0) * render_base * 0.66)
                elif "=== Stage 2" in line:
                    entry["stage"] = "tracking"
                    set_prog(render_base * 0.66)
                elif "Solve error" in line or "=== Stage 3" in line:
                    set_prog(render_base * 0.9)
                elif "=== Stage 4" in line or "Render:" in line:
                    entry["stage"] = "rendering"
                    set_prog(render_base)
            fr = parse_frame(line)
            if fr is None:
                continue
            entry["stage"] = "rendering"
            done = min(fr, entry["frames_total"])
            if done > entry["frames_done"]:
                entry["frames_done"] = done
            set_prog(render_base + (done / total) * render_span)
            bt = blender_elapsed(line)
            if first_frame is None:
                first_frame, first_bt = fr, bt
            prog = fr - first_frame
            if prog > 0:  # sec/frame from Blender's own clock
                if bt is not None and first_bt is not None:
                    queue_state["spf"] = (bt - first_bt) / prog
                else:
                    queue_state["spf"] = (time.time() - t0) / max(done, 1)
        proc.wait()
        # render_landed, not os.path.exists: a PNG sequence lands at
        # "<render>_0001.png", so entry["render"] itself never exists and a
        # successful sequence would be scored a failure.
        ok = proc.returncode == 0 and render_landed(entry["render"])
        if ok:
            set_prog(1.0)
        return ok
    except Exception:
        return False


def _process_queue_entry(entry):
    """Do a queued shot's full pipeline: track (or place static) then render.
    Legacy entries that carry a pre-tracked 'blend' just render it."""
    # Legacy render-only entry (added by an older build)
    if entry.get("blend") and not entry.get("kind"):
        cmd = [BLENDER, "--factory-startup", entry["blend"], "-P",
               os.path.join(HERE, "render_stage4.py"), "--",
               "--out", entry["render"], "--engine", entry.get("engine", "eevee")]
        if entry.get("samples"):
            cmd += ["--samples", str(entry["samples"])]
        if entry.get("percent") and str(entry["percent"]) != "100":
            cmd += ["--percent", str(entry["percent"])]
        if entry.get("transparent"):
            cmd += ["--transparent"]
        entry["stage"] = "rendering"
        # whole bar is the render for a pre-tracked entry
        return _run_tracked_proc(cmd, entry, watch_stages=False,
                                 render_base=0.0, render_span=1.0)

    footage, shot, scene = entry["footage"], entry["shot"], entry["scene"]

    entry["stage"] = "tracking"
    cmd = py_cmd("auto_track") + [
        footage, "--shot", str(shot), "--blender", BLENDER,
        "--masking-model", load_settings().get("masking_model", "best"),
        "--scene", scene,
        f"--start={entry.get('start', '0,0,0')}",
        f"--rotation={entry.get('rotation', '0,0,0')}",
        f"--scale={entry.get('scale', '1.0')}",
        "--render", entry["render"], "--engine", entry.get("engine", "eevee")]
    if entry.get("lens"):
        cmd += [f"--lens-mm={entry['lens']}"]
    if entry.get("focus"):
        cmd += [f"--focus-distance={entry['focus']}"]
    if entry.get("samples"):
        cmd += ["--samples", str(entry["samples"])]
    if entry.get("percent") and str(entry["percent"]) != "100":
        cmd += ["--percent", str(entry["percent"])]
    if entry.get("transparent"):
        cmd += ["--transparent"]
    if entry.get("kind") == "static":
        # UI marked this shot locked-off; auto_track places a static camera
        # (and even without the flag it re-measures motion and decides).
        cmd += ["--static"]
    return _run_tracked_proc(cmd, entry)



def queue_runner():
    while True:
        with queue_lock:
            if not queue_state["running"]:
                break
            entry = next((e for e in render_queue if e["status"] == "queued"),
                         None)
            if entry is None:
                queue_state["running"] = False
                notify(True)
                break
            entry["status"] = "rendering"  # 'in progress' (track or render)
            entry["stage"] = "tracking"
            entry["frames_done"] = 0
            entry["progress"] = 0.0
            save_queue()
        ok = _process_queue_entry(entry)
        if entry.get("footage") and entry.get("shot") is not None:
            # tracking data is ephemeral until the user saves it
            record_unsaved_track(entry["footage"], entry["shot"])
        with queue_lock:
            if entry["status"] == "rendering":  # not reset by a stop
                entry["status"] = "done" if ok else "failed"
                entry["stage"] = "done" if ok else "failed"
                if ok:
                    entry["frames_done"] = entry["frames_total"]
            queue_state["proc"] = None
            save_queue()


def start_queue():
    with queue_lock:
        if queue_state["running"] or job["status"] == "running":
            return False
        if not any(e["status"] == "queued" for e in render_queue):
            return False
        queue_state["running"] = True
        queue_state["started"] = time.time()
    cleanup_unsaved_tracks()  # a new batch starts clean: unsaved solves go
    threading.Thread(target=queue_runner, daemon=True).start()
    return True


def kill_tree(proc):
    """Kill a subprocess AND every process it spawned (Blender, masking,
    flow-solve, ...). Every pipeline step we launch is really a chain of
    processes (this app re-invokes itself, which in turn shells out to
    Blender); on Windows proc.terminate() only kills the immediate process,
    leaving Blender running as an orphan that keeps eating GPU/CPU forever.
    taskkill /T walks the whole tree."""
    if proc is None:
        return
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                          capture_output=True)
        else:
            proc.terminate()
    except Exception:
        pass


def sweep_pipeline_processes():
    """Belt-and-braces stop: kill every process that is provably part of OUR
    pipeline — any Blender running one of Nimbus's stage scripts, and any
    worker copy of this app ('--run <module>'). Matched by command line, so
    the user's own Blender sessions and the app window itself are never
    touched. Catches any orphan that slipped out of the process tree."""
    if os.name != "nt":
        return
    markers = ("auto_track_stage2.py", "render_stage4.py", "place_static.py",
               "apply_track_stage3.py", "preview_track.py", "export_setup.py",
               "export_camera.py",
               # parent wrappers too — killing only the Blender child lets
               # the parent respawn the next stage a second later
               "auto_track.py", "segment_people.py", "flow_solve.py",
               "split_shots.py")
    ps = ("Get-CimInstance Win32_Process -Filter \""
          "Name='blender.exe' or Name='Nimbus Tracker.exe' or "
          "Name='python.exe'\" | "
          "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=25).stdout
    except Exception:
        return
    me = os.getpid()
    for line in out.splitlines():
        if "\t" not in line:
            continue
        pid_s, cmdline = line.split("\t", 1)
        if not pid_s.strip().isdigit():
            continue
        pid = int(pid_s)
        if pid == me:
            continue
        if any(m in cmdline for m in markers) or " --run " in cmdline:
            try:
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                               capture_output=True, timeout=20)
            except Exception:
                pass


def stop_queue():
    with queue_lock:
        queue_state["running"] = False
        proc = queue_state.get("proc")
        for e in render_queue:
            if e["status"] == "rendering":
                e["status"] = "queued"
                e["frames_done"] = 0
        save_queue()
    kill_tree(proc)
    sweep_pipeline_processes()  # nothing of ours survives a Stop


load_queue()


def load_placements():
    if not os.path.exists(PLACEMENTS_PATH):
        return []
    with open(PLACEMENTS_PATH) as f:
        return json.load(f)


def save_placement(name, loc, rot_deg, scale):
    """Remember a camera placement (TrackRoot transform) for reuse."""
    recs = load_placements()
    for r in recs:  # skip near-duplicates
        if (all(abs(a - b) < 1e-3 for a, b in zip(r["loc"], loc)) and
                all(abs(a - b) < 0.05 for a, b in zip(r["rot_deg"], rot_deg)) and
                abs(r["scale"] - scale) < 1e-3):
            r["ts"] = time.time()
            break
    else:
        recs.insert(0, {"name": name, "loc": loc, "rot_deg": rot_deg,
                        "scale": scale, "ts": time.time()})
    recs.sort(key=lambda r: -r["ts"])
    with open(PLACEMENTS_PATH, "w") as f:
        json.dump(recs[:30], f, indent=2)


def scene_profile_path(footage):
    """Per-clip 'scene profile': the Blender scene + render settings shared by
    every shot in that clip. Camera placement stays per-shot; this is the
    stuff that's the same across the whole scene."""
    return os.path.join(workdir_for(footage), "scene_profile.json")


def load_scene_profile(footage):
    p = scene_profile_path(footage)
    if os.path.exists(p):
        try:
            with open(p) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_scene_profile(footage, prof):
    try:
        os.makedirs(workdir_for(footage), exist_ok=True)
        with open(scene_profile_path(footage), "w") as f:
            json.dump(prof, f, indent=2)
    except Exception:
        pass


# ---- tracking data is EPHEMERAL unless explicitly saved -------------------
# A finished shot's solve stays on disk for the session (result page, camera
# export, repositioning all work), but unless the user presses "Save tracking
# data" it is forgotten — deleted at the next batch start / app launch. The
# render itself (the mp4 in Videos) is always kept; masks are kept too (they
# depend only on the footage and are expensive to recompute).
SAVED_TRACKS_PATH = _cfg("saved_tracks.json")
UNSAVED_REG_PATH = _cfg("unsaved_tracks.json")


def _load_pairs(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return [tuple(x) for x in json.load(f)]
        except Exception:
            pass
    return []


def _save_pairs(path, pairs):
    try:
        with open(path, "w") as f:
            json.dump(sorted(set(pairs)), f, indent=2)
    except Exception:
        pass


def _track_key(footage, shot):
    return (os.path.normpath(footage).lower(), int(shot))


def is_track_saved(footage, shot):
    return _track_key(footage, shot) in _load_pairs(SAVED_TRACKS_PATH)


def mark_track_saved(footage, shot):
    key = _track_key(footage, shot)
    _save_pairs(SAVED_TRACKS_PATH, _load_pairs(SAVED_TRACKS_PATH) + [key])
    _save_pairs(UNSAVED_REG_PATH,
                [p for p in _load_pairs(UNSAVED_REG_PATH) if p != key])


def record_unsaved_track(footage, shot):
    key = _track_key(footage, shot)
    if key not in _load_pairs(SAVED_TRACKS_PATH):
        _save_pairs(UNSAVED_REG_PATH, _load_pairs(UNSAVED_REG_PATH) + [key])


def forget_tracking(footage, shot):
    """Delete a shot's solve products. Keeps: the rendered video, the person
    masks (footage-derived, expensive) and the chosen camera pose."""
    import shutil
    wd = workdir_for(footage)
    tag = f"shot_{int(shot):02d}"
    try:
        shutil.rmtree(os.path.join(wd, tag + "_out"), ignore_errors=True)
        for p in (glob.glob(os.path.join(wd, tag + "_*_tracked.blend")) +
                  glob.glob(os.path.join(wd, tag + "_*_queued_*.blend"))):
            try:
                os.remove(p)
            except OSError:
                pass
    except Exception:
        pass


def cleanup_unsaved_tracks():
    """Forget every tracked shot the user never saved."""
    saved = set(_load_pairs(SAVED_TRACKS_PATH))
    remaining = []
    for footage, shot in _load_pairs(UNSAVED_REG_PATH):
        if (footage, shot) in saved:
            continue
        try:
            forget_tracking(footage, shot)
        except Exception:
            remaining.append((footage, shot))
    _save_pairs(UNSAVED_REG_PATH, remaining)


cleanup_unsaved_tracks()  # app start: forget last session's unsaved solves


def shot_motion_cached(footage, shots):
    """Camera-motion estimate per shot, cached in the work folder."""
    # motion_v2: optical-flow detector (v1 phase-correlation caches are wrong
    # for push-in/zoom shots and must not be reused)
    cache_path = os.path.join(workdir_for(footage), "motion_v2.json")
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return {int(k): v for k, v in json.load(f).items()}
    sys.path.insert(0, HERE)
    from auto_track import shot_motion
    motion = {}
    for s in shots:
        m = shot_motion(footage, s["frame_start"], s["frame_end"])
        motion[s["shot"]] = None if m is None else round(m, 1)
    with open(cache_path, "w") as f:
        json.dump(motion, f)
    return motion


def make_thumb(footage, shot):
    """Brightened mid-shot thumbnail; cached."""
    thumb_dir = os.path.join(workdir_for(footage), "thumbs")
    os.makedirs(thumb_dir, exist_ok=True)
    path = os.path.join(thumb_dir, f"shot_{shot['shot']:02d}.jpg")
    if not os.path.exists(path):
        cap = cv2.VideoCapture(footage)
        mid = (shot["frame_start"] + shot["frame_end"]) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, mid - 1)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            return None
        h = int(frame.shape[0] * 480 / frame.shape[1])
        frame = cv2.resize(frame, (480, h))
        frame = cv2.convertScaleAbs(frame, alpha=1.6)  # these plates are dark
        cv2.imwrite(path, frame)
    return path


# ------------------------------------------------------------------ chrome
ORB = """<svg width="{s}" height="{s}" viewBox="0 0 32 32">
<defs>
 <linearGradient id="cl{u}" x1="0" y1="0" x2="0" y2="1">
  <stop offset="0%" stop-color="#ffffff"/>
  <stop offset="100%" stop-color="#cfe6f8"/></linearGradient>
</defs>
<path d="M9 22 a5.5 5.5 0 0 1 -.6 -10.97 A7 7 0 0 1 21.9 9.6
  A5.8 5.8 0 0 1 23.5 22 Z" fill="url(#cl{u})"
  stroke="rgba(70,130,190,.55)" stroke-width="1.1"/>
<circle cx="16" cy="16.4" r="4.6" fill="none"
  stroke="#2b7bc2" stroke-width="1.6"/>
<circle cx="16" cy="16.4" r="1.5" fill="#2b7bc2"/>
<path d="M16 9.6 V12 M16 20.8 V23.2 M9.2 16.4 H11.6 M20.4 16.4 H22.8"
  stroke="#2b7bc2" stroke-width="1.6" stroke-linecap="round"/>
</svg>"""

I_LIB = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="3" y="5" width="18" height="14" rx="2.5"/><path d="M3 9h18M7 5v4M12 5v4M17 5v4"/></svg>"""
I_ACT = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h4l3-8 4 16 3-8h4"/></svg>"""
I_FOLDER = """<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2.5h8a2 2 0 0 1 2 2V17a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/></svg>"""
I_GEAR = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3.2"/><path d="M19 12a7 7 0 0 0-.1-1.2l2-1.5-2-3.4-2.3 1a7 7 0 0 0-2-1.2L14.2 3h-4l-.4 2.7a7 7 0 0 0-2 1.2l-2.3-1-2 3.4 2 1.5a7 7 0 0 0 0 2.4l-2 1.5 2 3.4 2.3-1a7 7 0 0 0 2 1.2l.4 2.7h4l.4-2.7a7 7 0 0 0 2-1.2l2.3 1 2-3.4-2-1.5c.06-.4.1-.8.1-1.2Z"/></svg>"""
I_QUEUE = """<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 7h11M4 12h11M4 17h7"/><path d="M18 8l3 4-3 4"/></svg>"""

CSS = """
  :root {
    --ink:#132c52; --ink2:#2c4a76; --dim:#5a749c; --faint:#8aa0bf;
    --panel:rgba(255,255,252,.93); --panel2:rgba(255,255,252,.98);
    --line:rgba(19,44,82,.10); --edge:rgba(19,44,82,.22);
    --coral:#ff6f61; --coral2:#e6503f; --teal:#12a5b8; --teal2:#0d8494;
    --sun:#ffd166;
    --green:#1fa860; --amber:#e0912f; --red:#e05545;
    --hard:6px 6px 0 rgba(16,42,90,.16);
    --hard-sm:3px 3px 0 rgba(16,42,90,.16);
  }
  * { box-sizing:border-box; }
  html, body { height:100%; }
  body { margin:0; color:var(--ink); font-size:13.5px;
      font-family:'Segoe UI',system-ui,sans-serif;
      -webkit-font-smoothing:antialiased;
      background:url('/bg') center/cover no-repeat fixed,
        linear-gradient(180deg,#0e4fa8 0%, #1f6fd0 55%, #5aa7e8 100%) fixed;
      display:flex; flex-direction:column; overflow:hidden; }
  ::selection { background:rgba(255,111,97,.30); }
  ::-webkit-scrollbar { width:12px; }
  ::-webkit-scrollbar-thumb { background:rgba(255,255,255,.75);
      border-radius:8px; border:3px solid transparent; background-clip:padding-box; }
  ::-webkit-scrollbar-thumb:hover { background:rgba(255,255,255,.95);
      border:3px solid transparent; background-clip:padding-box; }
  ::-webkit-scrollbar-track { background:transparent; }

  /* ---- title bar: thin white sunshade ---- */
  #titlebar { height:42px; min-height:42px; display:flex; align-items:center;
      padding:0 6px 0 14px; gap:10px; z-index:50; position:relative;
      background:rgba(255,255,252,.88); border-bottom:2px solid #132c52;
      backdrop-filter:blur(10px); user-select:none; }
  #titlebar .t { font-size:13.5px; font-weight:700; color:var(--ink);
      letter-spacing:.6px; }
  #titlebar .drag { flex:1; align-self:stretch; }
  .winbtn { width:34px; height:26px; border-radius:7px;
      border:2px solid var(--ink); background:#fff; color:var(--ink);
      font-size:12px; cursor:pointer; padding:0; font-weight:700;
      display:flex; align-items:center; justify-content:center;
      box-shadow:var(--hard-sm); }
  .winbtn:hover { background:var(--sun); }
  .winbtn.close:hover { background:var(--coral); color:#fff; }

  #shell { display:flex; flex:1; min-height:0; }

  /* ---- sidebar: white cabana ---- */
  nav { width:196px; min-width:196px; display:flex; flex-direction:column;
      padding:14px 10px; user-select:none; background:rgba(255,255,252,.88);
      border-right:2px solid #132c52; backdrop-filter:blur(10px); }
  nav a.item { display:flex; align-items:center; gap:9px; padding:9px 12px;
      border-radius:10px; color:var(--ink2); text-decoration:none;
      font-weight:650; font-size:13px; margin-bottom:6px;
      border:2px solid transparent; }
  nav a.item:hover { background:#fff; border-color:var(--edge); }
  nav a.item.active { background:var(--sun); border-color:var(--ink);
      color:var(--ink); box-shadow:var(--hard-sm); }
  nav .grow { flex:1; }
  nav .foot { padding:10px 12px 2px; border-top:2px solid var(--line);
      color:var(--dim); font-size:10.5px; line-height:1.7; }
  .dot { width:9px; height:9px; border-radius:50%; margin-left:auto;
      background:#d5dfeb; border:1.5px solid rgba(19,44,82,.25); }
  .dot.busy { background:var(--sun); border-color:var(--amber);
      animation:pulse 1.3s ease-in-out infinite; }
  .dot.ok { background:#4ad07f; border-color:var(--green); }
  .dot.err { background:var(--coral); border-color:var(--red); }
  @keyframes pulse { 50% { opacity:.4; } }

  main { flex:1; overflow-y:auto; min-height:0; }
  .wrap { max-width:880px; margin:0 auto; padding:24px 32px 70px; }
  .crumbs { color:#eaf3fd; font-size:12px; margin-bottom:10px;
      text-shadow:0 1px 3px rgba(10,40,90,.45); }
  .crumbs a { color:#eaf3fd; text-decoration:none; }
  .crumbs a:hover { color:#fff; text-decoration:underline; }
  .crumbs .sep { opacity:.7; margin:0 4px; }

  .pagehead { display:flex; align-items:flex-start; gap:16px; margin:0 0 18px; }
  .pagehead .titles { flex:1; }
  h1 { font-size:21px; font-weight:800; margin:0 0 3px; color:#ffffff;
      letter-spacing:.4px; text-shadow:0 2px 0 rgba(16,42,90,.35); }
  .sub { color:#eaf3fd; margin:0; font-size:13px; line-height:1.5;
      text-shadow:0 1px 3px rgba(10,40,90,.45); }

  /* ---- cards: crisp white with a hard 80s shadow ---- */
  .card { background:var(--panel); border:2px solid #132c52;
      border-radius:14px; padding:18px 20px; margin:0 0 16px;
      position:relative; box-shadow:var(--hard); }
  .card h3 { margin:0 0 3px; font-size:13.5px; font-weight:750; color:var(--ink); }
  .card .hint { color:var(--ink2); font-size:12.5px; margin:0 0 4px;
      line-height:1.5; }
  a { color:var(--teal2); }

  label { display:block; margin:13px 0 5px; color:var(--ink2);
      font-size:11.5px; font-weight:700; letter-spacing:.3px; }
  input[type=text], select { background:#ffffff;
      color:var(--ink); border:2px solid var(--edge); border-radius:9px;
      padding:8px 11px; width:100%; font-size:13px; outline:none;
      font-family:inherit; }
  input[type=text]:focus, select:focus { border-color:var(--teal);
      box-shadow:0 0 0 3px rgba(18,165,184,.18); }
  input::placeholder { color:#9fb2ca; }
  .browserow { display:flex; gap:8px; }
  .browserow input { flex:1; }

  /* ---- buttons: coral pop with hard shadow ---- */
  button { border-radius:10px; padding:8px 20px; font-size:13px;
      font-weight:750; cursor:pointer; font-family:inherit; color:#fff;
      background:var(--coral); border:2px solid #132c52;
      box-shadow:var(--hard-sm); letter-spacing:.2px; }
  button:hover { background:#ff8578; transform:translate(-1px,-1px);
      box-shadow:4px 4px 0 rgba(16,42,90,.16); }
  button:active { transform:translate(2px,2px); box-shadow:1px 1px 0 rgba(16,42,90,.16);
      background:var(--coral2); }
  button.small { padding:6px 14px; font-size:12px; }
  button.ghost, button.browse { color:var(--ink);
      background:#ffffff; border:2px solid #132c52; }
  button.ghost:hover, button.browse:hover { background:var(--sun); }
  button.browse { padding:8px 13px; display:inline-flex; align-items:center;
      gap:6px; white-space:nowrap; font-weight:650; }
  .linklike { background:none !important; border:none !important;
      box-shadow:none !important; color:#f0f6ff; font-size:12px;
      cursor:pointer; text-decoration:underline; padding:0; font-weight:600;
      transform:none !important; text-shadow:0 1px 3px rgba(10,40,90,.5); }
  .linklike:hover { color:#ffffff; }

  .badge { display:inline-flex; align-items:center; gap:6px; padding:3px 11px;
      border-radius:20px; font-size:11px; font-weight:750; letter-spacing:.2px;
      border:1.5px solid var(--ink); background:#fff; color:var(--ink2); }
  .badge::before { content:''; width:6px; height:6px; border-radius:50%;
      background:var(--dim); }
  .badge.ok { background:#e2f8ec; color:#137a44; border-color:#1fa860; }
  .badge.ok::before { background:var(--green); }
  .badge.bad { background:#ffe9e6; color:#b2372a; border-color:#e05545; }
  .badge.bad::before { background:var(--red); }
  .badge.warn { background:#fff4dd; color:#93610c; border-color:#e0912f; }
  .badge.warn::before { background:var(--amber); }

  .grid3 { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; }
  .shotgrid { display:grid;
      grid-template-columns:repeat(auto-fill,minmax(262px,1fr)); gap:17px; }
  .shotcard, .clipcard { background:var(--panel2); border:2px solid #132c52;
      border-radius:14px; overflow:hidden; box-shadow:var(--hard); }
  .shotcard:hover, .clipcard:hover { transform:translate(-2px,-2px);
      box-shadow:8px 8px 0 rgba(16,42,90,.18); }
  .shotcard .thumbwrap, .clipcard .thumbwrap { position:relative;
      aspect-ratio:16/9; background:#10263f;
      border-bottom:2px solid #132c52; }
  .shotcard img, .clipcard img { width:100%; height:100%; object-fit:cover;
      display:block; }
  .thumbwrap::after { content:''; position:absolute; inset:0;
      pointer-events:none; }
  .shotcard .thumbwrap .badge { position:absolute; top:8px; left:8px; z-index:2; }
  .shotcard .body { padding:11px 14px; display:flex;
      align-items:center; justify-content:space-between; gap:10px; }
  .muted { color:var(--dim); }

  .clipgrid { display:grid;
      grid-template-columns:repeat(auto-fill,minmax(225px,1fr)); gap:17px; }
  .clipcard { text-decoration:none; color:var(--ink); display:block;
      transition:none; }
  .clipcard .body { padding:10px 14px 11px; }
  .clipcard .name { font-weight:700; font-size:12.5px; margin-bottom:2px;
      white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
  .clipcard .meta { color:var(--dim); font-size:11.5px; }
  .sechead { font-size:11.5px; font-weight:800; color:#ffffff;
      text-transform:uppercase; letter-spacing:1.4px; margin:20px 0 12px;
      text-shadow:0 2px 0 rgba(16,42,90,.35); }

  pre { background:#10263f; border:2px solid #132c52;
      border-radius:12px; padding:13px 15px; overflow:auto;
      font-size:12px; line-height:1.6; max-height:380px; color:#cfe4f5;
      font-family:'Cascadia Mono','Consolas',monospace;
      box-shadow:var(--hard-sm); }

  .steps { display:flex; align-items:center; margin:2px 0 16px; flex-wrap:wrap;
      row-gap:8px; }
  .step { padding:5px 14px; border-radius:20px; font-size:12px; font-weight:700;
      background:#fff; color:var(--dim); border:1.5px solid var(--edge); }
  .step.active { background:var(--sun); color:var(--ink);
      border-color:var(--ink); box-shadow:var(--hard-sm); }
  .step.done { background:#e2f8ec; color:#137a44; border-color:#1fa860; }
  .step.done::before { content:'¹3 '; }
  .conn { width:16px; height:2px; background:rgba(255,255,255,.85);
      border-radius:1px; }

  .spinner { width:15px; height:15px; border:2.5px solid rgba(255,255,255,.7);
      border-top-color:var(--coral); border-radius:50%; display:inline-block;
      vertical-align:-2px; margin-right:10px; animation:spin .8s linear infinite; }
  @keyframes spin { to { transform:rotate(360deg); } }

  .verdict { border-radius:14px; padding:15px 19px; margin-bottom:16px;
      border:2px solid; box-shadow:var(--hard); background:var(--panel); }
  .verdict .h { font-size:15.5px; font-weight:800; margin-bottom:2px; }
  .verdict .s { font-size:12.5px; opacity:.9; }
  .verdict.ok { border-color:#1fa860; color:#137a44; }
  .verdict.warn { border-color:#e0912f; color:#93610c; }
  .verdict.bad { border-color:#e05545; color:#b2372a; }

  .tiles { display:grid; grid-template-columns:repeat(3,1fr); gap:14px;
      margin-bottom:16px; }
  .tile { background:var(--panel); border:2px solid #132c52;
      border-radius:14px; padding:13px 16px; box-shadow:var(--hard-sm); }
  .tile .k { color:var(--dim); font-size:10.5px; font-weight:800;
      text-transform:uppercase; letter-spacing:.8px; margin-bottom:5px; }
  .tile .v { font-size:20px; font-weight:800; font-variant-numeric:tabular-nums;
      color:var(--ink); }

  .pathrow { display:flex; align-items:center; gap:13px; padding:9px 0;
      border-top:1.5px solid var(--line); }
  .pathrow:first-of-type { border-top:none; }
  .pathrow .lbl { color:var(--ink2); font-size:12px; min-width:130px;
      font-weight:700; }
  .pathrow .val { flex:1; font-size:11.5px; word-break:break-all;
      color:var(--dim); font-family:'Cascadia Mono','Consolas',monospace; }
  .actions { display:flex; gap:11px; margin-top:16px; flex-wrap:wrap; }
  form.inline { display:inline; margin:0; }
  .checkrow { display:flex; align-items:center; gap:8px; margin-top:13px;
      color:var(--ink2); font-size:13px; }
  .checkrow input { accent-color:var(--teal2); width:14px; height:14px; }
  .vp-toolbar { display:flex; align-items:center; gap:8px; margin-bottom:10px;
      flex-wrap:wrap; }
  .vp-sep { width:1px; height:22px; background:var(--edge); }
  .vp-inline { display:flex; align-items:center; gap:6px; font-size:12px;
      color:var(--ink2); font-weight:650; }
  .vp-on { background:var(--teal) !important; color:#fff !important; }
  #vp { position:relative; height:62vh; min-height:420px; overflow:hidden;
      background:linear-gradient(180deg,#9fcdec,#eaf6fc);
      border-radius:12px; border:2px solid #132c52; }
  #vp canvas, #vp img { position:absolute; left:0; top:0; }
  #plate { object-fit:fill; visibility:hidden; pointer-events:none; }
  #vphint { position:absolute; bottom:8px; left:12px; font-size:11px;
      color:rgba(20,60,100,.6); pointer-events:none; }
  .vp-sliders { display:grid; grid-template-columns:1fr 1fr 1fr; gap:14px;
      margin-top:12px; }
  .slgroup { background:#fff; border:1.5px solid var(--edge);
      border-radius:10px; padding:10px 12px; }
  .slhead { font-size:11px; font-weight:800; color:var(--ink2);
      text-transform:uppercase; letter-spacing:.6px; margin-bottom:8px; }
  .slrow { display:flex; align-items:center; gap:8px; margin-bottom:6px; }
  .slrow span { width:12px; font-size:12px; font-weight:800; color:var(--dim); }
  .slrow input[type=range] { flex:1; accent-color:var(--teal2); min-width:0; }
  .slrow input[type=number] { width:62px; padding:4px 6px; font-size:12px; }
  .slhint { font-size:11px; color:var(--ink2); line-height:1.45; margin-top:4px; }

  /* ---- render queue: pool-water progress ---- */
  .qbar { height:22px; border-radius:11px; background:#e8f2f8;
      border:2px solid #132c52; overflow:hidden; position:relative; }
  .qfill { height:100%; width:0%; border-radius:8px; transition:width .6s;
      background:repeating-linear-gradient(-45deg,
        #28c3dd 0 14px, #17b3c1 14px 28px); }
  .qmeta { display:flex; justify-content:space-between; margin:7px 2px 2px;
      font-size:12.5px; color:var(--ink2); font-weight:700; }
  .qrow { display:flex; align-items:center; gap:12px; padding:9px 0;
      border-top:1.5px solid var(--line); font-size:12.5px; }
  .qrow .nm { flex:1; }
  .qmini { width:120px; height:10px; border-radius:6px;
      background:#e8f2f8; border:1.5px solid var(--edge); overflow:hidden; }
  .qmini > div { height:100%; border-radius:4px;
      background:var(--teal); transition:width .6s; }
  .qpctnum { width:46px; text-align:right; font-variant-numeric:tabular-nums;
      color:var(--ink2); font-weight:750; }
  .empty { text-align:center; color:var(--ink2); padding:52px 20px;
      border:2px solid #132c52; border-radius:14px;
      background:var(--panel); box-shadow:var(--hard); }
  .fallback { display:none; margin-top:0; }
  .fallback.show { display:block; }
"""

WINJS = """
<script>
function hasWin() { return typeof window.pywebview !== 'undefined'; }
document.addEventListener('DOMContentLoaded', () => {
  if (!hasWin()) document.getElementById('winbtns').style.display = 'none';
});
function wMin() { if (hasWin()) window.pywebview.api.minimize(); }
function wMax() { if (hasWin()) window.pywebview.api.toggle_max(); }
function wFull() { if (hasWin()) window.pywebview.api.toggle_fullscreen(); }
function wClose() { if (hasWin()) window.pywebview.api.close(); }
document.addEventListener('keydown', (e) => {
  if (e.key === 'F11') { e.preventDefault(); wFull(); }
});
</script>"""


def page(body, refresh=None, crumbs=None, active="library"):
    meta = f'<meta http-equiv="refresh" content="{refresh}">' if refresh else ""
    crumb_html = ""
    if crumbs:
        parts = [(f'<a href="{href}">{label}</a>' if href else f'<span>{label}</span>')
                 for label, href in crumbs]
        crumb_html = ('<div class="crumbs">' +
                      '<span class="sep">›</span>'.join(parts) + '</div>')
    dot = {"running": "busy", "done": "ok", "failed": "err"}.get(job["status"], "")
    qdot = "busy" if queue_state.get("running") else (
        "ok" if (render_queue and all(e["status"] in ("done", "failed")
                                      for e in render_queue)) else "")
    return f"""<!doctype html><html><head><title>{APP_NAME}</title>{meta}
<meta name="application-name" content="autotrack-app">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>{CSS}</style></head><body>
<div id="titlebar">
  {ORB.format(s=24, u='tb')}
  <span class="t">{APP_NAME}</span>
  <div class="drag pywebview-drag-region"></div>
  <div id="winbtns" style="display:flex;gap:5px">
    <button class="winbtn" onclick="wMin()" title="Minimize">&#8211;</button>
    <button class="winbtn" onclick="wMax()" title="Maximize">&#9633;</button>
    <button class="winbtn" onclick="wFull()" title="Full screen (F11)">&#x26F6;</button>
    <button class="winbtn close" onclick="wClose()" title="Close">&#10005;</button>
  </div>
</div>
<div id="shell">
<nav>
  <a class="item {'active' if active == 'library' else ''}" href="/">{I_LIB} Library</a>
  <a class="item {'active' if active == 'activity' else ''}" href="/job">{I_ACT} Activity
    <span class="dot {dot}"></span></a>
  <a class="item {'active' if active == 'queue' else ''}" href="/queue">{I_QUEUE} Render queue
    <span class="dot {qdot}"></span></a>
  <a class="item {'active' if active == 'settings' else ''}" href="/settings">{I_GEAR} Settings</a>
  <div class="grow"></div>
  <div class="foot">{APP_NAME} {APP_VERSION}<br>
    Made by Inwood Park Productions<br>
    All processing stays on this machine</div>
</nav>
<main><div class="wrap">
{crumb_html}
{body}
</div></main>
</div>
{WINJS}
</body></html>"""


# ------------------------------------------------------------------ routes
@app.route("/")
def home():
    recents = load_recents()
    cards = []
    for r in recents:
        f = r["path"]
        shots = load_shots(f)
        meta = f"{len(shots)} shots" if shots else "not analyzed yet"
        cards.append(f"""
<a class="clipcard" href="/analyze_get?footage={f}">
  <div class="thumbwrap"><img src="/clipthumb?footage={f}"
    onerror="this.remove()" loading="lazy"></div>
  <div class="body"><div class="name">{os.path.basename(f)}</div>
  <div class="meta">{meta}</div></div>
</a>""")
    clear_btn = ('<button class="linklike" style="float:right;margin-top:4px" '
                 'onclick="clearLibrary()">Clear library</button>'
                 if cards else "")
    recent_html = (f'<div class="sechead">Recent clips{clear_btn}</div>'
                   f'<div class="clipgrid">{"".join(cards)}</div>' if cards else
                   '<div class="empty">No clips yet. Open one to get started.</div>')
    body = f"""
<div class="pagehead">
  <div class="titles">
    <h1>Library</h1>
    <p class="sub">Open a clip to split it into camera shots and track one.
    People are masked automatically; only the background drives the solve.</p>
  </div>
  <button onclick="browse('video')" style="margin-top:2px">Open clip…</button>
</div>
<div class="fallback" id="fb">
  <form action="/analyze" method="post" class="card">
    <label style="margin-top:0">File path</label>
    <div class="browserow">
      <input type="text" name="footage_custom" placeholder="C:\\path\\to\\clip.mp4">
      <button class="small">Find shots</button>
    </div>
  </form>
</div>
{recent_html}
<p style="margin-top:16px"><button class="linklike"
  onclick="document.getElementById('fb').classList.toggle('show')">
  paste a file path instead</button></p>
<script>
async function browse(kind) {{
  const r = await fetch('/browse?kind=' + kind, {{method:'POST'}});
  const d = await r.json();
  if (d.unsupported) {{ document.getElementById('fb').classList.add('show'); return; }}
  if (!d.path) return;
  const form = document.createElement('form');
  form.method = 'POST'; form.action = '/analyze';
  const i = document.createElement('input');
  i.type = 'hidden'; i.name = 'footage_custom'; i.value = d.path;
  form.appendChild(i); document.body.appendChild(form); form.submit();
}}
function clearLibrary() {{
  if (confirm('Clear all clips from the library?\\n\\nThis only empties this '
    + 'list — your footage files and any tracking results on disk are kept.'))
    location.href = '/clear_library';
}}
</script>"""
    return page(body, active="library")


@app.route("/browse", methods=["POST"])
def browse():
    """Native file dialogs. kind: video | blend | render_save"""
    if webview_window is None:
        return jsonify({"unsupported": True})
    import webview
    kind = request.args.get("kind", "video")
    if kind == "render_save":
        picked = webview_window.create_file_dialog(
            webview.SAVE_DIALOG, directory=DIALOG_START_DIR,
            save_filename="render.mp4")
    else:
        types = {"video": ("Video files (*.mov;*.mp4;*.m4v;*.avi;*.mkv)",
                           "All files (*.*)"),
                 "blend": ("Blender files (*.blend)", "All files (*.*)"),
                 "exe": ("Programs (*.exe)", "All files (*.*)")}[kind]
        start = DIALOG_START_DIR if os.path.isdir(DIALOG_START_DIR) else ""
        picked = webview_window.create_file_dialog(
            webview.OPEN_DIALOG, directory=start, allow_multiple=False,
            file_types=types)
    if not picked:
        return jsonify({"path": None})
    path = picked[0] if isinstance(picked, (list, tuple)) else picked
    return jsonify({"path": path})


@app.route("/analyze_get")
def analyze_get():
    """Recent-clip click: straight to shots if analyzed, else analyze."""
    footage = request.args.get("footage", "")
    if load_shots(footage) is not None:
        add_recent(footage)
        return redirect("/shots?footage=" + footage)
    return _start_analyze(footage)


@app.route("/analyze", methods=["POST"])
def analyze():
    footage = request.form.get("footage_custom", "").strip().strip('"')
    return _start_analyze(footage)


def _start_analyze(footage):
    if not os.path.exists(footage):
        return page('<div class="card">That file does not exist.'
                    '<div class="actions"><a href="/"><button class="ghost">'
                    'Back</button></a></div></div>')
    add_recent(footage)
    if load_shots(footage) is not None:
        return redirect("/shots?footage=" + footage)
    shots_dir = os.path.join(workdir_for(footage), "shots")
    started = start_job(py_cmd("split_shots") + [footage, shots_dir],
                        "analyze", {"footage": footage})
    if not started:
        return page('<div class="card">Another job is already running — '
                    '<a href="/job">watch it</a>.</div>')
    return redirect("/job")


@app.route("/clipthumb")
def clipthumb():
    footage = request.args.get("footage", "")
    shots = load_shots(footage)
    if shots:
        thumb_dir = os.path.join(workdir_for(footage), "thumbs")
        existing = sorted(glob.glob(os.path.join(thumb_dir, "shot_*.jpg")))
        if existing:
            return send_file(existing[0], mimetype="image/jpeg")
        path = make_thumb(footage, shots[0])
        if path:
            return send_file(path, mimetype="image/jpeg")
    abort(404)


@app.route("/shots")
def shots_page():
    footage = request.args.get("footage", "")
    shots = load_shots(footage)
    if shots is None:
        return redirect("/")
    motion = shot_motion_cached(footage, shots)
    fps = 24.0
    # shots from this clip already sitting in the render queue
    queued_shots = {e["shot"] for e in render_queue
                    if e.get("footage") == footage
                    and e["status"] in ("queued", "rendering")}
    cards = []
    n_track = 0
    n_done = 0
    n_untracked_moving = 0
    for s in shots:
        m = motion.get(s["shot"])
        static = m is not None and m < STATIC_MOTION_PX
        n_track += 0 if static else 1
        motion_badge = ('<span class="badge bad">static</span>' if static
                        else f'<span class="badge ok">motion {m} px</span>')
        status = shot_status(footage, s["shot"])
        if status:
            n_done += 1
            status_badge = (f'<span class="badge {status[0]}" '
                            f'style="top:auto;bottom:8px">{status[1]}</span>')
        else:
            status_badge = ""
            if not static:
                n_untracked_moving += 1
        secs = s["num_frames"] / fps
        # static shots have no motion to solve, but can still be set up as a
        # locked-off camera matched to the plate (static=1)
        if status:
            action = f"""
          <form class="inline" action="/track_form" method="get">
            <input type="hidden" name="footage" value="{footage}">
            <input type="hidden" name="shot" value="{s['shot']}">
            {'<input type="hidden" name="static" value="1">' if static else ''}
            <button class="small ghost">Re-do</button></form>"""
        else:
            action = f"""
          <form class="inline" action="/track_form" method="get">
            <input type="hidden" name="footage" value="{footage}">
            <input type="hidden" name="shot" value="{s['shot']}">
            {'<input type="hidden" name="static" value="1">' if static else ''}
            <button class="small">Set up</button>
          </form>"""
        queued_badge = ('<span class="badge ok" style="top:8px;left:8px;'
                        'right:auto">✓ in queue</span>'
                        if s["shot"] in queued_shots else "")
        cards.append(f"""
<div class="shotcard">
  <div class="thumbwrap">
    <img src="/thumb?footage={footage}&shot={s['shot']}" loading="lazy">
    {motion_badge}
    {status_badge}
    {queued_badge}
  </div>
  <div class="body">
    <div><b>Shot {s['shot']}</b><br>
      <span class="muted" style="font-size:12px">frames
      {s['frame_start']}–{s['frame_end']} · {secs:.1f}s</span>
    </div>
    {action}
  </div>
</div>""")
    track_all = ""
    if n_untracked_moving > 0:
        track_all = f"""
  <form class="inline" action="/track_all" method="post" style="margin-top:2px">
    <input type="hidden" name="footage" value="{footage}">
    <button class="ghost">Auto-track all {n_untracked_moving} (default camera)</button></form>"""
    total_q = sum(1 for e in render_queue if e["status"] in ("queued", "rendering"))
    just = request.args.get("queued")
    banner = ""
    if total_q:
        clip_q = len(queued_shots)
        toast = (f'<span class="badge ok">✓ shot {just} added</span> &nbsp;'
                 if just else "")
        banner = f"""
<div class="card" style="display:flex;align-items:center;gap:14px;
    background:#0f2436;border-color:#1f4b6b">
  <div style="flex:1">{toast}<b>{total_q} shot{'s' if total_q != 1 else ''}
    in the render queue</b>{f' · {clip_q} from this clip' if clip_q else ''}.
    <span class="muted">Set up more shots, then render them all at once.</span></div>
  <a href="/queue"><button>Go to render queue →</button></a>
</div>"""
    body = f"""
<div class="pagehead"><div class="titles">
  <h1>{os.path.basename(footage)}</h1>
  <p class="sub">{len(shots)} shots · {n_track} with camera motion · {n_done}
  done. Click <b>Set up</b> to place the camera in Blender and add the shot to
  the render queue.</p></div>{track_all}</div>
{banner}
<div class="shotgrid">{"".join(cards)}</div>"""
    return page(body, crumbs=[("Library", "/"), (os.path.basename(footage), None)],
                active="library")


@app.route("/track_all", methods=["POST"])
def track_all():
    footage = request.form["footage"]
    shots = load_shots(footage) or []
    motion = shot_motion_cached(footage, shots)
    todo = [s["shot"] for s in shots
            if (motion.get(s["shot"]) or 0) >= 2.0
            and not shot_status(footage, s["shot"])]
    if not todo:
        return redirect("/shots?footage=" + footage)
    if not start_track_all(footage, todo):
        return page('<div class="card">Another job is already running — '
                    '<a href="/job">watch it</a>.</div>')
    return redirect("/job")


@app.route("/track_all_stop", methods=["POST"])
def track_all_stop():
    track_all_state["running"] = False
    return redirect("/job")


@app.route("/track_all_status")
def track_all_status():
    return jsonify(track_all_state)


@app.route("/thumb")
def thumb():
    footage = request.args.get("footage", "")
    shots = load_shots(footage) or []
    shot = next((s for s in shots if s["shot"] == int(request.args.get("shot", 0))), None)
    if shot is None:
        abort(404)
    path = make_thumb(footage, shot)
    if path is None:
        abort(404)
    return send_file(path, mimetype="image/jpeg")


@app.route("/track_form")
def track_form():
    footage = request.args.get("footage", "")
    shot = request.args.get("shot", "")
    name = os.path.basename(footage)
    st = load_settings()
    # The scene + render settings are the SAME for every shot in this clip and
    # live in the per-clip profile (set once in Blender). Only the camera
    # changes shot to shot. Arg values (just returned from Blender) win.
    prof = load_scene_profile(footage)
    prefill_scene = (request.args.get("scene", "") or prof.get("scene", "")
                     or st.get("last_scene", ""))
    pre_start = request.args.get("start", "0,0,0")
    pre_rotation = request.args.get("rotation", "0,0,0")
    pre_scale = request.args.get("scale", "1.0")
    pre_lens = request.args.get("lens", "") or st.get("default_lens", "")
    # render settings come from Blender (the profile); fall back to last/defaults
    # default to Cycles (GPU — fastest); a scene set up in Blender or your
    # last choice overrides, and you can change it in Advanced below
    pre_engine = prof.get("engine") or st.get("last_engine") or "cycles"
    pre_samples = str(prof.get("samples") or st.get("last_samples", "32"))
    pre_percent = str(prof.get("percent") or st.get("last_percent", "100"))
    pre_transparent = bool(prof.get("transparent"))
    have_render = bool(prof.get("engine"))
    is_static = flag(request.args.get("static"))
    placed = request.args.get("placed")
    render_summary = (f'{pre_engine} · {pre_samples} samples · {pre_percent}%'
                      + (' · transparent' if pre_transparent else ''))
    if placed:
        placed_chip = (
            '<div class="card" style="background:#12321a;border-color:#1f5b30">'
            '<span class="badge ok">✓ camera set in Blender</span>'
            + (f'&nbsp;&nbsp;<span class="badge ok">✓ render: {render_summary}'
               '</span>' if have_render else '')
            + '<p class="hint" style="margin:8px 0 0">This shot is ready. '
              '<b>Add it to the render queue</b>, then set up your other shots '
              'the same way — they reuse this scene &amp; render settings, so '
              'you only re-position the camera. Render them all at the end.</p>'
            '</div>')
    else:
        placed_chip = ""
    if is_static:
        head = f"Set up static shot {shot}"
        blurb = ("No camera motion here. Pick your scene, position the "
                 "locked-off camera and set your render settings in Blender, "
                 "then add it to the render queue.")
    else:
        head = f"Set up shot {shot}"
        blurb = ("Set the starting camera and your render settings in Blender "
                 "against the first frame, then add this shot to the render "
                 "queue. Track + render happen as a batch at the end.")
    setup_label = ("🎥 Re-open camera &amp; render in Blender…" if placed
                   else "🎥 Set up camera &amp; render in Blender…")
    reuse_hint = ('<p class="hint" style="margin:6px 0 0">Scene &amp; render '
                  'settings are reused from the other shots in this clip — '
                  'you only need to position the camera. Change them by '
                  'editing below or re-opening Blender.</p>'
                  if have_render and not placed else "")
    body = f"""
<div class="pagehead"><div class="titles">
  <h1>{head}</h1>
  <p class="sub">{blurb}</p>
</div></div>
<form action="/queue_add" method="post" id="setupform">
<input type="hidden" name="footage" value="{footage}">
<input type="hidden" name="shot" value="{shot}">
<input type="hidden" name="placed" id="placedflag" value="{1 if placed else 0}">
{'<input type="hidden" name="static" value="1">' if is_static else ''}
<div class="card">
  <h3>Your Blender scene</h3>
  <p class="hint">{'The camera is placed into a copy of this file.'
    if is_static else 'The solved camera is baked into a copy of this file '
    '(camera object: <b>TrackedCamera</b>).'} This scene and its render
  settings are shared by every shot in this clip.</p>
  <label>Scene .blend file</label>
  <div class="browserow">
    <input type="text" name="scene" id="scene" value="{prefill_scene}"
        placeholder="C:\\path\\to\\your_scene.blend">
    <button type="button" class="browse"
        onclick="pick('blend','scene')">{I_FOLDER} Browse</button>
  </div>
  {reuse_hint}
  <div class="actions" style="margin-top:12px">
    <button type="button" class="ghost" onclick="setupBlender()">
      {setup_label}</button>
  </div>
  <p class="hint" style="margin-top:6px">Opens your scene in Blender with the
  first frame on the camera. Frame the shot, set your render settings in the
  <b>Render</b> + <b>Output</b> tabs, then click <b>Choose Starting
  Position</b> in the Nimbus panel — it returns here automatically.</p>
</div>
{placed_chip}
<details class="card" style="cursor:pointer">
  <summary style="font-weight:650;font-size:13px">Advanced / override
    <span class="muted" style="font-weight:400;font-size:12px"> — filled in
    from Blender; edit only if you want</span></summary>
  <div style="cursor:auto">
  <div class="grid3">
    <div><label>Camera start x,y,z</label>
         <input type="text" name="start" id="fstart" value="{pre_start}"></div>
    <div><label>Rotation ° rx,ry,rz</label>
         <input type="text" name="rotation" id="frot" value="{pre_rotation}"></div>
    <div><label>Motion scale</label>
         <input type="text" name="scale" value="{pre_scale}"></div>
  </div>
  <div class="grid3">
    <div><label>Lens (mm)</label>
         <input type="text" name="lens" value="{pre_lens}"
           placeholder="auto — solver estimates"></div>
    <div><label>Focus distance</label>
         <input type="text" name="focus" value=""
           placeholder="off — scene units for DoF"></div>
    <div></div>
  </div>
  <p class="hint" style="margin-top:2px">Leave <b>Lens</b> blank to let the
  solver estimate focal length; set it to a known lens (e.g. 35) to lock it.</p>
  <label>Render output (automatic if empty)</label>
  <div class="browserow">
    <input type="text" name="render" id="render"
        placeholder="automatic — PNG sequence in its own folder under Videos">
    <button type="button" class="browse"
        onclick="pick('render_save','render')">{I_FOLDER} Browse</button>
  </div>
  <div class="grid3">
    <div><label>Engine</label>
      <select name="engine">
      {''.join(f'<option{" selected" if e == pre_engine else ""}>{e}</option>'
               for e in ('eevee', 'cycles', 'workbench'))}</select></div>
    <div><label>Samples</label><input type="text" name="samples" value="{pre_samples}"></div>
    <div><label>Resolution %</label><input type="text" name="percent" value="{pre_percent}"></div>
  </div>
  <p class="hint" style="margin-top:6px">Speed tip: render time scales with
  pixels, not samples — <b>Resolution 50%</b> is about <b>3× faster</b> than
  100% on 4K footage. Use 50% for drafts, 100% for the final.</p>
  <div class="checkrow"><input type="checkbox" name="transparent" id="tr"
    {'checked' if pre_transparent else ''}>
    <label for="tr" style="all:unset;cursor:pointer">transparent background
    (PNG sequences only)</label></div>
  </div>
</details>
<div class="actions">
  <button formaction="/queue_add" onclick="return needCam(event)">
    ＋ Add to render queue</button>
  <button formaction="/track" class="ghost" onclick="return needCam(event)"
    title="Track and bake this one shot right now instead of batching">
    Track this shot now</button>
  <a href="/shots?footage={footage}"><button type="button" class="ghost">Cancel</button></a>
</div>
</form>
<script>
async function pick(kind, field) {{
  const r = await fetch('/browse?kind=' + kind, {{method:'POST'}});
  const d = await r.json();
  if (d.path) document.getElementById(field).value = d.path;
}}
function setupBlender() {{
  const s = document.getElementById('scene').value.trim();
  if (!s) {{ alert('Pick your scene .blend first.'); return; }}
  window.location = '/blender_setup?footage=' +
    encodeURIComponent({json.dumps(footage)}) + '&shot={shot}' +
    '&static={1 if is_static else 0}&scene=' + encodeURIComponent(s);
}}
function needCam(ev) {{
  const scene = document.getElementById('scene').value.trim();
  if (!scene) {{ alert('Pick your scene .blend first.'); ev.preventDefault(); return false; }}
  if (document.getElementById('placedflag').value !== '1') {{
    if (!confirm('You haven\\'t set the camera in Blender for this shot yet. '
      + 'Add it anyway with a default camera?')) {{ ev.preventDefault(); return false; }}
  }}
  return true;
}}
</script>"""
    return page(body, crumbs=[("Library", "/"),
                              (name, f"/shots?footage={footage}"),
                              (f"Shot {shot}", None)], active="library")


@app.route("/track", methods=["POST"])
def track():
    f = request.form
    footage = f["footage"]
    scene0 = f.get("scene", "").strip().strip('"')

    # Static shot: no solve — just place a locked-off camera at the chosen
    # pose over the shot's frame range.
    if flag(f.get("static")):
        shot = int(f["shot"])
        if not scene0 or not os.path.exists(scene0):
            return page('<div class="card">A static shot needs a scene '
                        '.blend to place the camera into.'
                        '<div class="actions"><a href="javascript:history.back()">'
                        '<button class="ghost">Back</button></a></div></div>')
        return _run_static_place(footage, shot, scene0, f)

    cmd = py_cmd("auto_track") + [footage,
           "--shot", f["shot"], "--blender", BLENDER,
           "--masking-model", load_settings().get("masking_model", "best")]
    scene = f.get("scene", "").strip().strip('"')
    if scene:
        if not os.path.exists(scene):
            return page(f'<div class="card">Scene not found: {scene}'
                        f'<div class="actions"><a href="javascript:history.back()">'
                        f'<button class="ghost">Back</button></a></div></div>')
        start = f.get("start", "0,0,0")
        rotation = f.get("rotation", "0,0,0")
        scale = f.get("scale", "1.0")
        if f.get("placement"):  # reuse a saved camera placement
            recs = load_placements()
            i = int(f["placement"])
            if 0 <= i < len(recs):
                p = recs[i]
                start = ",".join(f"{v:.4f}" for v in p["loc"])
                rotation = ",".join(f"{v:.3f}" for v in p["rot_deg"])
                scale = f"{p['scale']:.4f}"
        # --flag=value form: start/rotation can be negative (e.g. -6.69,...),
        # and argparse mistakes a leading '-' for a new option otherwise.
        cmd += ["--scene", scene, f"--start={start}",
                f"--rotation={rotation}", f"--scale={scale}"]
        lens = f.get("lens", "").strip()
        if lens:
            cmd += [f"--lens-mm={lens}"]
        focus = f.get("focus", "").strip()
        if focus:
            cmd += [f"--focus-distance={focus}"]
    render = f.get("render", "").strip().strip('"')
    if render and not scene:
        return page('<div class="card">Rendering needs a scene .blend.'
                    '<div class="actions"><a href="javascript:history.back()">'
                    '<button class="ghost">Back</button></a></div></div>')
    if scene and not render:
        # default render destination: the user's Videos folder (standard)
        render = default_render_path(footage, int(f["shot"]))
    # Rendering is DEFERRED: first track + bake the camera, then the user
    # positions TrackRoot in Blender's viewport, then presses Render on the
    # result page (render settings are remembered in the job meta).
    meta = {"footage": footage, "shot": int(f["shot"]), "scene": scene,
            "render": render if scene else "",
            "engine": f.get("engine", "eevee"),
            "samples": f.get("samples", "64"),
            "percent": f.get("percent", "100"),
            "transparent": flag(f.get("transparent")),
            "rendered": False}
    remember_last(scene, f)  # prefill these next time
    started = start_job(cmd, "track", meta)
    if not started:
        return page('<div class="card">Another job is already running — '
                    '<a href="/job">watch it</a>.</div>')
    return redirect("/job")


def _run_static_place(footage, shot, scene, f):
    """Place a locked-off camera for a static shot (no tracking)."""
    tag = f"shot_{shot:02d}"
    wd = workdir_for(footage)
    base = os.path.splitext(os.path.basename(scene))[0]
    scene_out = os.path.join(wd, f"{tag}_{base}_tracked.blend")
    out_log = os.path.join(wd, tag + "_out", tag + "_masked_track_log.json")
    shot_file = shot_file_for(os.path.join(wd, "shots"), shot)
    sj = os.path.join(wd, "shots", "shots.json")
    frames, source_size = 1, None
    if os.path.exists(sj):
        with open(sj) as fp:
            data = json.load(fp)
        source_size = data.get("size")
        s = next((x for x in data["shots"] if x["shot"] == shot), None)
        if s:
            frames = s["num_frames"]
    cmd = [BLENDER, "-b", scene, "-P", os.path.join(HERE, "place_static.py"), "--",
           "--footage", shot_file, "--start", f.get("start", "0,0,0"),
           "--rotation", f.get("rotation", "0,0,0"), "--frames", str(frames),
           "--out", scene_out, "--log", out_log]
    if f.get("lens", "").strip():
        cmd += ["--focal-mm", f["lens"].strip()]
    if f.get("focus", "").strip():
        cmd += ["--focus-distance", f["focus"].strip()]
    if source_size:
        cmd += ["--render-size", f"{source_size[0]}x{source_size[1]}"]
    render = f.get("render", "").strip().strip('"') or \
        default_render_path(footage, shot)
    meta = {"footage": footage, "shot": shot, "scene": scene,
            "render": render, "engine": f.get("engine", "eevee"),
            "samples": f.get("samples", "64"), "percent": f.get("percent", "100"),
            "transparent": flag(f.get("transparent")), "rendered": False,
            "static": True}
    if not start_job(cmd, "track", meta):
        return page('<div class="card">Another job is already running — '
                    '<a href="/job">watch it</a>.</div>')
    return redirect("/job")


# ---- Blender camera positioning (real Blender, not the internal viewport) ----
@app.route("/clear_library")
def clear_library():
    save_recents([])  # empties the recents list; footage/results on disk kept
    return redirect("/")


def dlog(msg):
    """Append a timestamped line to nimbus_debug.log (handshake tracing)."""
    try:
        with open(os.path.join(HERE, "nimbus_debug.log"), "a",
                  encoding="utf-8") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except Exception:
        pass


@app.errorhandler(Exception)
def _nimbus_error(e):
    import traceback
    from werkzeug.exceptions import HTTPException
    if isinstance(e, HTTPException):
        return e
    tb = traceback.format_exc()
    dlog("EXCEPTION on " + request.path + "\n" + tb)
    return page('<div class="card"><h3>Something went wrong</h3>'
                '<p class="hint">The details were logged. Please tell Claude '
                'what you were doing.</p><pre style="font-size:11px">'
                + tb[-1600:] + '</pre>'
                '<div class="actions"><a href="/"><button class="ghost">'
                'Library</button></a></div></div>'), 500


@app.before_request
def _trace_request():
    if request.path in ("/track_form", "/track", "/blender_setup"):
        dlog(f"REQ {request.path} args={dict(request.args)} "
             f"form={dict(request.form)}")


@app.route("/blender_setup")
def blender_setup():
    footage = request.args.get("footage", "")
    shot = int(request.args.get("shot", 1))
    scene = request.args.get("scene", "").strip().strip('"')
    static = request.args.get("static", "0") == "1"
    dlog(f"blender_setup shot={shot} static={static} scene={scene}")
    if not scene or not os.path.exists(scene):
        return page('<div class="card">Pick your scene .blend first.'
                    '<div class="actions"><a href="javascript:history.back()">'
                    '<button class="ghost">Back</button></a></div></div>')
    tag = f"shot_{shot:02d}"
    wd = workdir_for(footage)
    os.makedirs(wd, exist_ok=True)
    frame_png = os.path.join(wd, tag + "_frame1.png")
    shot_file = shot_file_for(os.path.join(wd, "shots"), shot)
    cap = cv2.VideoCapture(shot_file if os.path.exists(shot_file) else footage)
    ok, img = cap.read()
    cap.release()
    if ok:
        cv2.imwrite(frame_png, img)
    out_json = os.path.join(wd, tag + "_startcam.json")
    if os.path.exists(out_json):
        os.remove(out_json)
    try:
        subprocess.Popen([BLENDER, scene, "-P",
                          os.path.join(HERE, "blender_setup.py"), "--",
                          "--frame-img", frame_png, "--out", out_json])
        dlog(f"launched Blender={BLENDER!r} out={out_json}")
    except Exception as e:
        dlog(f"Blender launch FAILED: {e}")
        return page(f'<div class="card">Could not launch Blender:<br>{e}'
                    f'<br><br>Set the Blender path in Settings if it is '
                    f'installed somewhere unusual.<div class="actions">'
                    f'<a href="javascript:history.back()"><button class="ghost">'
                    f'Back</button></a></div></div>')
    # Stateless handshake: the waiting page carries the shot params and polls
    # by the on-disk pose file — no in-memory token, so it survives restarts.
    from urllib.parse import urlencode
    q = urlencode({"footage": footage, "shot": shot, "scene": scene,
                   "static": 1 if static else 0})
    return page(f"""
<div class="pagehead"><div class="titles">
  <h1><span class="spinner"></span>Positioning camera in Blender…</h1>
  <p class="sub">Blender opened your scene with the first frame on the
  camera. Navigate to frame the shot (the camera follows the view), then
  click <b>Choose Starting Position</b> in the <b>Nimbus</b> panel
  (press N in Blender if you don't see it). This returns automatically.</p>
</div></div>
<div class="card">
  <p class="hint" style="margin:0">Waiting for Blender… if you closed it
  without choosing a position, go back and try again.</p>
  <div class="actions"><a href="/track_form?{urlencode({'footage': footage, 'shot': shot, 'scene': scene, 'static': 1 if static else 0})}">
    <button class="ghost">Back to the shot</button></a></div>
</div>
<script>
let nimbusDone = false;
async function checkReady() {{
  if (nimbusDone) return;
  try {{
    const r = await fetch('/blender_setup_ready?{q}');
    const d = await r.json();
    if (d.redirect) {{
      nimbusDone = true;
      try {{ if (window.pywebview) await window.pywebview.api.raise_window(); }}
      catch (e) {{}}
      location.href = d.redirect;
    }}
  }} catch (e) {{}}
}}
// Poll on a timer AND whenever this window regains focus (the moment Blender
// closes) — background windows throttle timers, so focus is the reliable one.
setInterval(checkReady, 1500);
window.addEventListener('focus', checkReady);
document.addEventListener('visibilitychange', () => {{
  if (!document.hidden) checkReady();
}});
checkReady();
</script>""", active="library")


@app.route("/blender_setup_ready")
def blender_setup_ready():
    footage = request.args.get("footage", "")
    shot = int(request.args.get("shot", 1))
    scene = request.args.get("scene", "")
    static = request.args.get("static", "0") == "1"
    out = os.path.join(workdir_for(footage), f"shot_{shot:02d}_startcam.json")
    if not os.path.exists(out):
        return jsonify({"waiting": True})
    try:
        with open(out) as f:
            d = json.load(f)
        loc, rot = d["loc"], d["rot_deg"]
    except Exception:
        return jsonify({"waiting": True})  # mid-write; poll again
    dlog(f"ready shot={shot} -> redirect (pose found)")
    # The render settings the user configured in Blender apply to the WHOLE
    # clip by default — save them into the per-clip scene profile so the next
    # shot doesn't ask again (only the camera changes shot to shot).
    prof = load_scene_profile(footage)
    prof["scene"] = scene
    rs = d.get("render") or {}
    if rs:
        prof["engine"] = rs.get("engine", prof.get("engine", "eevee"))
        prof["samples"] = str(rs.get("samples", prof.get("samples", "64")))
        prof["percent"] = str(rs.get("percent", prof.get("percent", "100")))
        prof["transparent"] = bool(rs.get("transparent",
                                          prof.get("transparent", False)))
    save_scene_profile(footage, prof)
    from urllib.parse import urlencode
    q = {"footage": footage, "shot": shot, "scene": scene, "placed": 1,
         "start": ",".join(f"{v:.4f}" for v in loc),
         "rotation": ",".join(f"{v:.3f}" for v in rot)}
    if d.get("focal_mm"):  # carry the lens the user set on NimbusStartCam
        q["lens"] = f"{d['focal_mm']:.2f}"
    if static:
        q["static"] = 1
    return jsonify({"redirect": "/track_form?" + urlencode(q)})


INTERESTING = re.compile(
    r"===|\[stage\d\]|\[segment\]|\[shots\]|\[preview\]|\[flow\]|\[static\]|"
    r"Solve error|usable features|Deleted|Muted|Fra:|Error|Traceback|FAILED",
    re.I)

STEPS = ["Split shots", "Person masks", "Camera track", "Your scene", "Render"]
STEP_PAT = re.compile(r"=== Stage (\d)")


def steps_html(meta):
    """Pipeline chips: done/active/pending, derived from the log."""
    wanted = [0, 1, 2] + ([3] if meta.get("scene") else []) + \
             ([4] if meta.get("rendered") else [])
    current = 0
    for line in job["log"]:
        m = STEP_PAT.search(line)
        if m:
            current = int(m.group(1))
    chips = []
    for n, i in enumerate(wanted):
        cls = ("done" if i < current else
               "active" if i == current and job["status"] == "running" else
               "done" if job["status"] == "done" else "")
        if n:
            chips.append('<span class="conn"></span>')
        chips.append(f'<span class="step {cls}">{STEPS[i]}</span>')
    return '<div class="steps">' + "".join(chips) + '</div>'


def queue_card():
    """Render-queue panel with overall progress bar, ETA and per-take rows."""
    if not render_queue:
        return ""
    badge = {"queued": ("waiting", "warn"), "rendering": ("rendering", "ok"),
             "done": ("done", "ok"), "failed": ("failed", "bad")}
    rows = []
    for i, e in enumerate(render_queue):
        label, cls = badge[e["status"]]
        if e["status"] == "rendering":  # in progress — show which stage
            label = "tracking…" if e.get("stage") == "tracking" else "rendering…"
        remove = ("" if e["status"] == "rendering" else f"""
          <form class="inline" action="/queue_remove" method="post">
            <input type="hidden" name="idx" value="{i}">
            <button class="small ghost">remove</button></form>""")
        save_btn = ""
        if e["status"] == "done" and e.get("footage") and \
                e.get("shot") is not None:
            if is_track_saved(e["footage"], e["shot"]):
                save_btn = '<span class="badge ok">tracking saved</span>'
            else:
                save_btn = f"""
          <form class="inline" action="/save_tracking" method="post"
            title="Keep this shot's camera solve — otherwise it is forgotten
when the next batch starts or the app closes (the rendered video is always kept)">
            <input type="hidden" name="footage" value="{e['footage']}">
            <input type="hidden" name="shot" value="{e['shot']}">
            <button class="small ghost">save tracking data</button></form>"""
        pct0 = round(100 * _entry_progress(e))
        rows.append(f"""
<div class="qrow"><div class="nm"><b>{e['name']}</b>
  <span class="muted">· {e['engine']} · {e['frames_total']} frames</span></div>
  <div class="qmini"><div id="qe{i}" style="width:{pct0}%"></div></div>
  <span class="qpctnum" id="qp{i}">{pct0}%</span>
  <span class="badge {cls}" id="qb{i}">{label}</span>{save_btn}{remove}</div>""")
    controls = ("""<form class="inline" action="/queue_stop" method="post">
        <button class="ghost">Stop queue</button></form>"""
                if queue_state["running"] else
                """<form class="inline" action="/queue_start" method="post">
        <button>Start queue</button></form>""")
    return f"""
<div class="card">
  <h3>Render queue</h3>
  <div class="qbar"><div class="qfill" id="qfill"></div></div>
  <div class="qmeta"><span id="qpct">0%</span><span id="qeta"></span></div>
  {"".join(rows)}
  <div class="actions">{controls}</div>
</div>
<script>
async function qpoll() {{
  try {{
    const r = await fetch('/queue_status'); const d = await r.json();
    document.getElementById('qfill').style.width = d.overall_pct + '%';
    document.getElementById('qpct').textContent =
        d.overall_pct.toFixed(1) + '% complete';
    let eta = '';
    if (d.eta_s !== null && d.eta_s !== undefined) {{
      const m = Math.floor(d.eta_s / 60), s = d.eta_s % 60;
      eta = 'about ' + (m > 0 ? m + ' min ' : '') + s + ' s left';
    }} else if (d.running) {{ eta = 'estimating…'; }}
    document.getElementById('qeta').textContent = eta;
    d.entries.forEach((e, i) => {{
      const el = document.getElementById('qe' + i);
      if (el) el.style.width = e.pct + '%';
      const pn = document.getElementById('qp' + i);
      if (pn) pn.textContent = Math.round(e.pct) + '%';
      const bd = document.getElementById('qb' + i);
      if (bd && e.status === 'rendering')
        bd.textContent = (e.stage === 'tracking') ? 'tracking…' : 'rendering…';
    }});
  }} catch (err) {{}}
}}
qpoll(); setInterval(qpoll, 2000);
</script>"""


@app.route("/queue")
def queue_page():
    card = queue_card()
    if not card:
        card = ('<div class="empty">The render queue is empty.<br><br>'
                '<span style="font-size:12.5px">On any tracked shot\'s result '
                'page, position the camera and press <b>add to render queue</b>. '
                'Stack several takes or shots, then start them here as a batch — '
                'renders save to your Videos folder.</span></div>')
    body = f"""
<div class="pagehead"><div class="titles">
  <h1>Render queue</h1>
  <p class="sub">Batch-render stacked shots and takes. Progress and ETA
  update live; finished videos land in your Videos folder.</p>
</div></div>
{card}"""
    return page(body, active="queue")


def track_all_card():
    """Progress panel for a 'Track all' batch."""
    if not (track_all_state["running"] or track_all_state["results"]):
        return ""
    ta = track_all_state
    pct = round(100 * ta["done"] / ta["total"], 0) if ta["total"] else 0
    rows = "".join(
        f'<div class="qrow"><span class="nm">Shot {r["shot"]}</span>'
        f'<span class="badge {r["cls"]}">{r["label"]}</span></div>'
        for r in ta["results"])
    cur = (f'<div class="qrow"><span class="nm">Shot {ta["current"]}</span>'
           f'<span class="badge warn">tracking…</span></div>'
           if ta["running"] and ta["current"] else "")
    ctrl = ('<form class="inline" action="/track_all_stop" method="post">'
            '<button class="ghost">Stop</button></form>'
            if ta["running"] else
            '<a href="/shots?footage=' + (ta["footage"] or "") +
            '"><button class="ghost">View shots</button></a>')
    head = (f'<span class="spinner"></span>Tracking all shots — '
            f'{ta["done"]}/{ta["total"]}' if ta["running"] else
            f'Track-all finished — {ta["done"]}/{ta["total"]} done')
    refresh = ('<script>setTimeout(()=>location.reload(),2500)</script>'
               if ta["running"] else "")
    return f"""
<div class="card">
  <h3>{head}</h3>
  <div class="qbar"><div class="qfill" style="width:{pct}%"></div></div>
  <div style="margin-top:10px">{rows}{cur}</div>
  <div class="actions">{ctrl}</div>
</div>{refresh}"""


@app.route("/job")
def job_page():
    if job["status"] == "idle":
        parts = [track_all_card(), queue_card()]
        body_extra = "".join(p for p in parts if p) or (
            '<div class="empty">Nothing running. '
            'Open a clip in the Library to start.</div>')
        return page(f"""
<div class="pagehead"><div class="titles">
  <h1>Activity</h1>
  <p class="sub">Live progress appears here while a job, a track-all batch,
  or the render queue runs.</p>
</div></div>
{body_extra}""",
                    active="activity")
    shown = [l for l in job["log"] if INTERESTING.search(l)]
    fra = [l for l in shown if l.startswith("Fra:")]
    if len(fra) > 3:
        shown = [l for l in shown if not l.startswith("Fra:")] + fra[-3:]
    log_tail = "\n".join(shown[-40:]) or "starting…"
    if job["status"] == "running":
        stage = "Working"
        for line in reversed(job["log"]):
            m = re.search(r"=== (.+) ===", line)
            if m:
                stage = m.group(1)
                break
        steps = steps_html(job["meta"]) if job["kind"] == "track" else ""
        render_eta = ""
        if job.get("render_total"):
            render_eta = """
<div class="card" id="rendercard" style="display:none">
  <h3 id="rhead">Rendering…</h3>
  <div class="qbar"><div class="qfill" id="rfill"></div></div>
  <div class="qmeta"><span id="rpct">0%</span><span id="reta"></span></div>
</div>
<script>
let rdone = false;
async function rpoll() {
  try {
    const d = await (await fetch('/job_render_status')).json();
    if (d.rendering) {
      document.getElementById('rendercard').style.display = 'block';
      document.getElementById('rfill').style.width = d.pct + '%';
      document.getElementById('rhead').textContent =
          'Rendering frame ' + d.frame + ' of ' + d.total;
      document.getElementById('rpct').textContent = d.pct.toFixed(1) + '%';
      let eta = 'estimating…';
      if (d.eta_s !== null && d.eta_s !== undefined) {
        const m = Math.floor(d.eta_s/60), s = d.eta_s%60;
        eta = 'about ' + (m>0 ? m+' min ' : '') + s + ' s left';
      }
      document.getElementById('reta').textContent = eta;
    } else if (!rdone && (d.status === 'done' || d.status === 'failed')) {
      rdone = true; location.reload();  // render finished → go to result
    }
  } catch (e) {}
}
rpoll(); setInterval(rpoll, 2000);
</script>"""
        body = f"""
<div class="pagehead"><div class="titles">
  <h1><span class="spinner"></span>{stage}</h1>
  <p class="sub">Updates automatically. Renders show a live ETA below —
  safe to leave in the background (keep the window open).</p>
</div></div>
{steps}
{render_eta}
<pre id="log">{log_tail}</pre>
<script>const l=document.getElementById('log'); l.scrollTop=l.scrollHeight;</script>
{queue_card()}"""
        # don't full-refresh while rendering (it would reset the live ETA);
        # the render card polls itself. Refresh only during non-render stages.
        rf = None if job.get("render_frame") else 3
        return page(body, refresh=rf, active="activity")
    if job["status"] == "failed":
        return page(f"""
<div class="pagehead"><div class="titles">
  <h1>Something went wrong</h1>
  <p class="sub">The log below usually says why. Static shots, extremely dark
  footage, or a wrong file path are the usual suspects.</p>
</div></div>
<pre>{log_tail}</pre>
<div class="actions"><a href="/"><button>Back to Library</button></a></div>""",
                    active="activity")
    if job["kind"] == "analyze":
        return redirect("/shots?footage=" + job["meta"]["footage"])
    return redirect("/result")


@app.route("/result")
def result():
    if job["kind"] != "track" or job["status"] != "done":
        return redirect("/")
    meta = job["meta"]
    footage, shot = meta["footage"], meta["shot"]
    name = os.path.basename(footage)
    tag = f"shot_{shot:02d}"
    out_dir = os.path.join(workdir_for(footage), tag + "_out")
    log_json = os.path.join(out_dir, tag + "_masked_track_log.json")
    err = tracks = frames = None
    mode = "perspective"
    if os.path.exists(log_json):
        with open(log_json) as fp:
            d = json.load(fp)
        err = d["average_solve_error"]
        if err is not None and err != err:  # NaN in old logs
            err = None
        tracks = d.get("num_tracks_final")
        frames = d.get("frame_duration")
        mode = d.get("solve_mode") or "perspective"
    if mode == "static":
        cls, head, sub = "ok", "Locked-off camera placed", \
            "No camera motion in this shot — a static camera was set at your " \
            "chosen position, matched to the plate. Ready to render."
    elif err is None:
        cls, head, sub = "bad", "No usable solve", \
            "The background is too flat or featureless for camera tracking " \
            "in this shot. Adding tracking markers to the set would fix it."
    elif err < 1.0:
        cls, head, sub = "ok", f"Excellent solve — {err:.2f} px", \
            "Production-grade. CG will sit rock-solid in this shot."
    elif err < 3.0:
        cls, head, sub = "ok", f"Good solve — {err:.2f} px", \
            "Small drift possible on tight close contact, fine for most uses."
    elif err < 8.0:
        cls, head, sub = "warn", f"Rough solve — {err:.2f} px", \
            "Usable for loose set extensions; visible drift if CG is locked to the floor."
    else:
        cls, head, sub = "bad", f"Bad solve — {err:.2f} px", \
            "Don't use this. The shot may lack parallax or background features."
    if err is not None and mode == "tripod":
        sub += (" Rotation-only (tripod) solve: the camera pans in place — "
                "correct for shots where the camera doesn't physically move.")
    if err is not None and mode == "2d-flow":
        cls = "warn"
        tier = d.get("flow_tier", "") if os.path.exists(log_json) else ""
        head = f"Approximate motion match — {err:.2f} px" + \
               (f" ({tier})" if tier else "")
        sub = ("This shot can't be 3D-solved (motion blur or featureless "
               "background), so the camera follows the shot's overall motion "
               "instead. Right for comping on blurred/close-up shots; don't "
               "lock CG to the floor with it.")

    tiles = f"""
<div class="tiles">
  <div class="tile"><div class="k">Solve error</div>
    <div class="v">{f"{err:.2f} px" if err is not None else "—"}</div></div>
  <div class="tile"><div class="k">Background tracks</div>
    <div class="v">{tracks if tracks is not None else "—"}</div></div>
  <div class="tile"><div class="k">Frames</div>
    <div class="v">{frames if frames is not None else "—"}</div></div>
</div>"""

    if is_track_saved(footage, shot):
        keep_html = '<span class="badge ok">saved</span>'
    else:
        keep_html = f"""
      <form class="inline" action="/save_tracking" method="post">
        <input type="hidden" name="footage" value="{footage}">
        <input type="hidden" name="shot" value="{shot}">
        <button class="small ghost" title="Without this, the solve is
forgotten when the next batch starts or the app closes (renders are always
kept)">save tracking data</button></form>"""
    rows = [f"""<div class="pathrow"><div class="lbl">Tracking data</div>
      <div class="val">{os.path.join(out_dir, tag + "_masked_tracked.blend")}</div>
      {keep_html}</div>"""]
    position_card = ""
    if meta["scene"]:
        base = os.path.splitext(os.path.basename(meta["scene"]))[0]
        scene_out = os.path.join(workdir_for(footage), f"{tag}_{base}_tracked.blend")
        rows.append(f"""<div class="pathrow"><div class="lbl">Scene + camera</div>
      <div class="val">{scene_out}</div></div>""")
        if meta.get("render") and not meta.get("rendered"):
            position_card = f"""
<div class="card">
  <h3>Position the shot, then render</h3>
  <p class="hint"><b>1.</b> Open the scene in Blender. Select the
  <b>TrackRoot</b> arrows and move / rotate / scale to place the whole camera
  path — the footage plays on the camera so you can line it up. The solved
  motion never changes. Save (Ctrl+S) and close Blender.<br>
  <b>2.</b> Press Render — it renders the scene exactly as you placed it,
  with your chosen settings ({meta.get('engine','eevee')},
  {meta.get('samples','64')} samples, {meta.get('percent','100')}%).
  Skipping straight to Render uses the placement as-is.</p>
  <div class="actions">
    <form class="inline" action="/open" method="post">
      <input type="hidden" name="path" value="{scene_out}">
      <button type="button" class="ghost" onclick="this.form.submit()">
      1. Open scene in Blender</button></form>
    <form class="inline" action="/render_final" method="post">
      <input type="hidden" name="footage" value="{footage}">
      <input type="hidden" name="shot" value="{shot}">
      <button>2. Render</button></form>
    <form class="inline" action="/queue_add_tracked" method="post">
      <input type="hidden" name="footage" value="{footage}">
      <input type="hidden" name="shot" value="{shot}">
      <button class="ghost">or add to render queue</button></form>
  </div>
  <p class="hint" style="margin-top:10px">Queue trick: position → add to
  queue → reposition in Blender → add again. Each queued take is a snapshot,
  so you can render several placements of the same shot in one batch from
  Activity.</p>
</div>"""
        elif meta.get("rendered") and meta.get("render"):
            rows.append(f"""<div class="pathrow"><div class="lbl">Render</div>
      <div class="val">{meta["render"]}</div>
      <form class="inline" action="/open" method="post">
        <input type="hidden" name="path" value="{meta["render"]}">
        <button class="small ghost">Play</button></form></div>""")

    preview_path = os.path.join(out_dir, tag + "_preview.mp4")
    if os.path.exists(preview_path):
        prev_btn = f"""<form class="inline" action="/open" method="post">
          <input type="hidden" name="path" value="{preview_path}">
          <button>Play track preview</button></form>"""
    else:
        prev_btn = f"""<form class="inline" action="/preview" method="post">
          <input type="hidden" name="footage" value="{footage}">
          <input type="hidden" name="shot" value="{shot}">
          <button>Render track preview</button></form>
          <span class="muted" style="font-size:12px"> — quick overlay video:
          glowing dots (3D solves) or a sky-grid (motion matches) that should
          stick to the background if the track is good</span>"""

    body = f"""
<div class="pagehead"><div class="titles">
  <h1>Shot {shot} · {name}</h1>
  <p class="sub">Tracking finished.</p>
</div></div>
<div class="verdict {cls}"><div class="h">{head}</div>
  <div class="s">{sub}</div></div>
{position_card}
<p>{prev_btn}</p>
{tiles}
<div class="card">{"".join(rows)}</div>
{('''<div class="card">
  <h3>Export camera</h3>
  <p class="hint">Send the solved camera to other software (After Effects,
  Nuke, Cinema 4D, Fusion…). Saves to your Videos folder.</p>
  <div class="actions">
    <form class="inline" action="/export_camera" method="post">
      <input type="hidden" name="footage" value="''' + footage + '''">
      <input type="hidden" name="shot" value="''' + str(shot) + '''">
      <input type="hidden" name="format" value="fbx">
      <button class="ghost">Export FBX</button></form>
    <form class="inline" action="/export_camera" method="post">
      <input type="hidden" name="footage" value="''' + footage + '''">
      <input type="hidden" name="shot" value="''' + str(shot) + '''">
      <input type="hidden" name="format" value="abc">
      <button class="ghost">Export Alembic (.abc)</button></form>
  </div>
</div>''') if meta.get('scene') else ''}
<div class="actions">
  <form class="inline" action="/reveal" method="post">
    <input type="hidden" name="path" value="{footage}">
    <button class="ghost">Show background plate in folder</button></form>
  <form class="inline" action="/open" method="post">
    <input type="hidden" name="path" value="{workdir_for(footage)}">
    <button class="ghost">Open results folder</button></form>
  <a href="/shots?footage={footage}"><button class="ghost">Other shots</button></a>
  <a href="/"><button class="ghost">Library</button></a>
</div>"""
    return page(body, crumbs=[("Library", "/"),
                              (name, f"/shots?footage={footage}"),
                              (f"Shot {shot}", None)], active="activity")


@app.route("/bg")
def background():
    art = os.path.join(HERE, "nagai_bg.svg")  # City-Pop seaside, ours
    if os.path.exists(art):
        return send_file(art, mimetype="image/svg+xml")
    return send_file(os.path.join(HERE, "aero_bg.svg"),
                     mimetype="image/svg+xml")


@app.route("/preview", methods=["POST"])
def preview():
    footage = request.form["footage"]
    shot = int(request.form["shot"])
    tag = f"shot_{shot:02d}"
    wd = workdir_for(footage)
    out_dir = os.path.join(wd, tag + "_out")
    log_json = os.path.join(out_dir, tag + "_masked_track_log.json")
    if not os.path.exists(log_json):
        return redirect("/")
    with open(log_json) as fp:
        d = json.load(fp)
    out = os.path.join(out_dir, tag + "_preview.mp4")
    if d.get("flow_json"):
        shot_file = shot_file_for(os.path.join(wd, "shots"), 1)
        cmd = [BLENDER, "-b", "-P", os.path.join(HERE, "preview_track.py"),
               "--", "--flow", d["flow_json"], "--footage", shot_file,
               "--out", out]
    else:
        tracked = os.path.join(out_dir, tag + "_masked_tracked.blend")
        cmd = [BLENDER, "-b", tracked,
               "-P", os.path.join(HERE, "preview_track.py"), "--",
               "--out", out]
    # keep prior meta so the result page still shows scene/render rows
    meta = dict(job["meta"]) if (job["meta"].get("footage") == footage and
                                 job["meta"].get("shot") == shot) else \
        {"footage": footage, "shot": shot, "scene": "", "render": ""}
    started = start_job(cmd, "track", meta)
    if not started:
        return page('<div class="card">Another job is already running — '
                    '<a href="/job">watch it</a>.</div>')
    return redirect("/job")


@app.route("/settings", methods=["GET", "POST"])
def settings_page():
    global BLENDER
    saved = ""
    if request.method == "POST":
        s = load_settings()
        path = request.form.get("blender_path", "").strip().strip('"')
        if path and not os.path.exists(path):
            saved = '<span class="badge bad">that file does not exist</span>'
        else:
            s["blender_path"] = path
            s["default_lens"] = request.form.get("default_lens", "").strip()
            s["masking_model"] = request.form.get("masking_model", "best")
            with open(SETTINGS_PATH, "w") as f:
                json.dump(s, f, indent=2)
            BLENDER = resolve_blender()
            saved = '<span class="badge ok">saved</span>'
    s = load_settings()
    detected = find_blender()
    body = f"""
<div class="pagehead"><div class="titles">
  <h1>Settings</h1>
  <p class="sub">Everything runs locally; these are the only knobs that
  live outside a shot.</p>
</div></div>
<form method="post" class="card">
  <h3>Blender</h3>
  <p class="hint">Auto-detected: <b>{detected}</b>. Set a path here only to
  use a different Blender install.</p>
  <label>Blender executable override (optional)</label>
  <div class="browserow">
    <input type="text" name="blender_path" id="blender_path"
      value="{s.get('blender_path', '')}"
      placeholder="leave empty to use the auto-detected Blender">
    <button type="button" class="browse"
      onclick="pickb()">{I_FOLDER} Browse</button>
  </div>
  <h3 style="margin-top:18px">Person masking</h3>
  <p class="hint">How people are removed from tracking. <b>Best</b> finds each
  person once and then <i>tracks their exact silhouette</i> through the whole
  shot (SAM2 + YOLO) — no flicker, full costume/prop coverage; ideal for
  cosplay and creatures. <b>Fast</b> re-detects every frame with a light
  model — quicker, fine for ordinary clothing.</p>
  <label>Masking quality</label>
  <select name="masking_model" style="max-width:320px">
    <option value="best"{' selected' if s.get('masking_model','best')=='best' else ''}>Best — tracks silhouettes through time (SAM2)</option>
    <option value="fast"{' selected' if s.get('masking_model','best')=='fast' else ''}>Fast — per-frame detection (quicker)</option>
  </select>
  <h3 style="margin-top:18px">Camera preset</h3>
  <p class="hint">A default lens for your usual camera — prefills the Lens
  field on the track form so you don't retype it.</p>
  <label>Default lens (mm, optional)</label>
  <input type="text" name="default_lens" value="{s.get('default_lens', '')}"
    placeholder="e.g. 35 — leave blank to let the solver estimate"
    style="max-width:220px">
  <div class="actions"><button>Save</button> {saved}</div>
</form>
<div class="card">
  <h3>About</h3>
  <p class="hint">{APP_NAME} {APP_VERSION} — automatic camera tracking for
  Blender, by Inwood Park Productions. Shot detection, AI person masking,
  three-tier camera solving (3D / tripod / motion-match), visual shot setup,
  render queue. All processing stays on this machine; footage never leaves
  it.</p>
</div>
<script>
async function pickb() {{
  const r = await fetch('/browse?kind=exe', {{method: 'POST'}});
  const d = await r.json();
  if (d.path) document.getElementById('blender_path').value = d.path;
}}
</script>"""
    return page(body, active="settings")


@app.route("/open", methods=["POST"])
def open_path():
    path = request.form.get("path", "")
    if os.path.exists(path):
        os.startfile(path)  # noqa — Windows only, local app
    return redirect(request.referrer or "/")


def setup_page(footage, shot, mode, scene):
    """The 3D shot-setup viewport."""
    from urllib.parse import urlencode
    q = urlencode({"footage": footage, "shot": shot, "mode": mode,
                   "scene": scene})
    title = ("Set the starting camera" if mode == "pre"
             else "Place the tracked shot")
    sub = ("Position the camera where the shot should begin — tracking will "
           "build the camera move from here." if mode == "pre" else
           "Move the whole solved camera path into place. The motion itself "
           "never changes — you're placing it in your world.")
    body = f"""
<div class="pagehead"><div class="titles">
  <h1>{title}</h1>
  <p class="sub">{sub} Drag the gizmo in the viewport; use <b>Camera view</b>
  to check the line-up against your footage, then Apply.</p>
</div></div>
<div class="card" style="padding:12px">
  <div class="vp-toolbar">
    <button type="button" class="small ghost" id="btnMove">Move</button>
    <button type="button" class="small ghost" id="btnRot">Rotate</button>
    <button type="button" class="small ghost" id="btnScale">Scale</button>
    <span class="vp-sep"></span>
    <button type="button" class="small ghost" id="btnFrame">Frame all</button>
    <button type="button" class="small ghost" id="btnPOV">Camera view</button>
    <label class="vp-inline">plate <input type="range" id="plateOp" min="0"
      max="100" value="55" style="width:90px"></label>
    <span class="vp-sep"></span>
    <label class="vp-inline" id="frameWrap">frame
      <input type="range" id="frame" min="1" max="1" value="1"
        style="width:160px"> <span id="frameNo">1</span></label>
    <span style="flex:1"></span>
    <button type="button" id="btnApply">{"Use this placement" if mode == "pre"
                                         else "Apply &amp; save"}</button>
    <a href="javascript:history.back()"><button type="button" class="ghost">
      Cancel</button></a>
  </div>
  <div id="vp">
    <img id="plate" alt="">
    <canvas id="c"></canvas>
    <div id="vphint">left-drag orbit · right-drag pan · wheel zoom</div>
  </div>
  <div class="vp-sliders">
    <div class="slgroup">
      <div class="slhead">Position</div>
      <div class="slrow"><span>X</span>
        <input type="range" id="pxr" step="0.01"><input type="number" id="pxn" step="0.1"></div>
      <div class="slrow"><span>Y</span>
        <input type="range" id="pyr" step="0.01"><input type="number" id="pyn" step="0.1"></div>
      <div class="slrow"><span>Z</span>
        <input type="range" id="pzr" step="0.01"><input type="number" id="pzn" step="0.1"></div>
    </div>
    <div class="slgroup">
      <div class="slhead">Rotation°</div>
      <div class="slrow"><span>X</span>
        <input type="range" id="rxr" min="-180" max="180" step="0.5"><input type="number" id="rxn" step="1"></div>
      <div class="slrow"><span>Y</span>
        <input type="range" id="ryr" min="-180" max="180" step="0.5"><input type="number" id="ryn" step="1"></div>
      <div class="slrow"><span>Z</span>
        <input type="range" id="rzr" min="-180" max="180" step="0.5"><input type="number" id="rzn" step="1"></div>
    </div>
    <div class="slgroup">
      <div class="slhead">Scale</div>
      <div class="slrow"><span></span>
        <input type="range" id="scr" min="0.05" max="20" step="0.01"><input type="number" id="scn" step="0.1"></div>
      <div class="slhint">Tip: turn on <b>Camera view</b> and nudge these to
      line the scene up against your footage.</div>
    </div>
  </div>
</div>
<script type="importmap">
{{"imports": {{"three": "/static/three.module.js"}}}}
</script>
<script id="cfg" type="application/json">{json.dumps({
    "q": q, "mode": mode, "footage": footage, "shot": shot,
    "scene": scene})}</script>
<script type="module" src="/static/setup_view.js"></script>"""
    return page(body, crumbs=[("Library", "/"),
                              (os.path.basename(footage),
                               f"/shots?footage={footage}"),
                              ("Shot setup", None)], active="library")


# ------------------------------------------------------------------ shot setup
def quat_to_euler_xyz_deg(w, x, y, z):
    """Quaternion -> Blender 'XYZ' euler (degrees). Blender XYZ order means
    R = Rz @ Ry @ Rx (x applied first, extrinsic)."""
    import math as m
    R20 = 2 * (x * z - w * y)
    R21 = 2 * (y * z + w * x)
    R22 = 1 - 2 * (x * x + y * y)
    R10 = 2 * (x * y + w * z)
    R00 = 1 - 2 * (y * y + z * z)
    ey = m.asin(max(-1.0, min(1.0, -R20)))
    ex = m.atan2(R21, R22)
    ez = m.atan2(R10, R00)
    return [m.degrees(ex), m.degrees(ey), m.degrees(ez)]


setup_exports = {}  # setup_dir -> "running" | "done" | "error"
setup_export_lock = threading.Lock()


def setup_export_fresh(source_blend, setup_dir):
    campath = os.path.join(setup_dir, "campath.json")
    glb = os.path.join(setup_dir, "setup.glb")
    return (os.path.exists(campath) and os.path.exists(glb) and
            os.path.getmtime(campath) >= os.path.getmtime(source_blend))


def start_setup_export(source_blend, setup_dir):
    """Kick off the Blender viewport export in the background (heavy scenes
    take ~a minute); returns immediately. Idempotent."""
    with setup_export_lock:
        if setup_exports.get(setup_dir) == "running":
            return
        setup_exports[setup_dir] = "running"

    def run():
        try:
            os.makedirs(setup_dir, exist_ok=True)
            subprocess.run([BLENDER, "-b", source_blend,
                            "-P", os.path.join(HERE, "export_setup.py"),
                            "--", "--out", setup_dir],
                           capture_output=True, text=True, timeout=900,
                           encoding="utf-8", errors="replace")
            ok = setup_export_fresh(source_blend, setup_dir)
            setup_exports[setup_dir] = "done" if ok else "error"
        except Exception:
            setup_exports[setup_dir] = "error"

    threading.Thread(target=run, daemon=True).start()


@app.route("/static/<path:name>")
def static_file(name):
    path = os.path.normpath(os.path.join(HERE, "static", name))
    if not path.startswith(os.path.join(HERE, "static")) or not os.path.exists(path):
        abort(404)
    return send_file(path)


@app.route("/plate_frame")
def plate_frame():
    """One frame of the shot proxy as JPEG (viewport background plate)."""
    footage = request.args.get("footage", "")
    shot = int(request.args.get("shot", 1))
    frame = max(1, int(request.args.get("frame", 1)))
    shot_file = shot_file_for(os.path.join(workdir_for(footage), "shots"), shot)
    if not os.path.exists(shot_file):
        abort(404)
    cap = cv2.VideoCapture(shot_file)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame - 1)
    ok, img = cap.read()
    cap.release()
    if not ok:
        abort(404)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 82])
    from flask import Response
    return Response(buf.tobytes(), mimetype="image/jpeg")


def setup_paths(footage, shot, mode, scene=""):
    tag = f"shot_{shot:02d}"
    wd = workdir_for(footage)
    if mode == "post":
        base = os.path.splitext(os.path.basename(scene))[0] if scene else ""
        if not base:  # find it from the job meta or existing files
            meta_scene = job["meta"].get("scene", "") if \
                job["meta"].get("footage") == footage else ""
            base = os.path.splitext(os.path.basename(meta_scene))[0]
        source = os.path.join(wd, f"{tag}_{base}_tracked.blend")
        setup_dir = os.path.join(wd, f"{tag}_setup")
    else:
        source = scene
        setup_dir = os.path.join(wd, f"{tag}_setup_pre")
    return source, setup_dir


@app.route("/setup")
def setup_view():
    footage = request.args.get("footage", "")
    shot = int(request.args.get("shot", 1))
    mode = request.args.get("mode", "post")
    scene = request.args.get("scene", "")
    source, setup_dir = setup_paths(footage, shot, mode, scene)
    if not source or not os.path.exists(source):
        return page('<div class="card">Scene file not found — pick your '
                    '.blend on the track page first.<div class="actions">'
                    '<a href="javascript:history.back()"><button class="ghost">'
                    'Back</button></a></div></div>')
    if setup_export_fresh(source, setup_dir):
        setup_exports[setup_dir] = "done"
        return setup_page(footage, shot, mode, scene)
    state = setup_exports.get(setup_dir)
    if state == "error":
        setup_exports.pop(setup_dir, None)
        return page('<div class="card">Could not read the scene for the '
                    'viewport. You can still position in Blender via the '
                    'TrackRoot empty.<div class="actions">'
                    '<a href="javascript:history.back()"><button class="ghost">'
                    'Back</button></a></div></div>')
    if state != "running":
        start_setup_export(source, setup_dir)
    from urllib.parse import urlencode
    q = urlencode({"footage": footage, "shot": shot, "mode": mode,
                   "scene": scene})
    return page(f"""
<div class="pagehead"><div class="titles">
  <h1><span class="spinner"></span>Preparing the 3D viewport…</h1>
  <p class="sub">Reading your scene and building a lightweight preview. The
  first time for a heavy scene can take up to a minute — it's cached after
  that.</p>
</div></div>
<script>
setInterval(async () => {{
  const r = await fetch('/setup_ready?' + {json.dumps(q)});
  const d = await r.json();
  if (d.ready) location.href = '/setup?' + {json.dumps(q)};
  else if (d.error) location.reload();
}}, 2000);
</script>""", active="library")


@app.route("/setup_ready")
def setup_ready():
    footage = request.args.get("footage", "")
    shot = int(request.args.get("shot", 1))
    mode = request.args.get("mode", "post")
    scene = request.args.get("scene", "")
    source, setup_dir = setup_paths(footage, shot, mode, scene)
    ready = source and os.path.exists(source) and \
        setup_export_fresh(source, setup_dir)
    return jsonify({"ready": bool(ready),
                    "error": setup_exports.get(setup_dir) == "error"})


@app.route("/setup_data")
def setup_data():
    footage = request.args.get("footage", "")
    shot = int(request.args.get("shot", 1))
    mode = request.args.get("mode", "post")
    scene = request.args.get("scene", "")
    fn = request.args.get("file", "")
    if fn not in ("campath.json", "setup.glb"):
        abort(404)
    _, setup_dir = setup_paths(footage, shot, mode, scene)
    path = os.path.join(setup_dir, fn)
    if not os.path.exists(path):
        abort(404)
    return send_file(path)


@app.route("/setup_apply", methods=["POST"])
def setup_apply():
    d = request.get_json(force=True)
    footage, shot = d["footage"], int(d["shot"])
    mode = d.get("mode", "post")
    loc = [float(v) for v in d["loc"]]
    quat = [float(v) for v in d["quat_wxyz"]]
    scale = float(d["scale"])
    eul = quat_to_euler_xyz_deg(*quat)
    shotname = os.path.splitext(os.path.basename(footage))[0]
    save_placement(f"{shotname} · shot {shot}", loc, eul, scale)

    if mode == "pre":
        from urllib.parse import urlencode
        q = urlencode({"footage": footage, "shot": shot,
                       "scene": d.get("scene", ""),
                       "start": ",".join(f"{v:.4f}" for v in loc),
                       "rotation": ",".join(f"{v:.3f}" for v in eul),
                       "scale": f"{scale:.4f}", "placed": 1})
        return jsonify({"redirect": "/track_form?" + q})

    source, _ = setup_paths(footage, shot, "post", d.get("scene", ""))
    expr = (f"import bpy\n"
            f"from mathutils import Vector, Quaternion, Matrix\n"
            f"root = bpy.data.objects['TrackRoot']\n"
            f"root.matrix_world = Matrix.LocRotScale(Vector({loc}), "
            f"Quaternion({quat}), Vector(({scale}, {scale}, {scale})))\n"
            f"bpy.ops.wm.save_mainfile()\n")
    r = subprocess.run([BLENDER, "-b", source, "--python-expr", expr],
                       capture_output=True, text=True, timeout=300,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        return jsonify({"error": "could not save placement"}), 500
    return jsonify({"redirect": "/result"})


@app.route("/queue_add", methods=["POST"])
def queue_add():
    """Snapshot a shot's setup (scene + camera pose + render settings) as a
    queued job. Nothing is tracked or rendered yet — that happens later when
    the queue runs — so adding is instant and you can batch many shots fast."""
    f = request.form
    footage = f["footage"]
    shot = int(f["shot"])
    scene = f.get("scene", "").strip().strip('"')
    if not scene or not os.path.exists(scene):
        return page('<div class="card">Pick a valid scene .blend before adding '
                    'to the render queue.<div class="actions">'
                    '<a href="javascript:history.back()"><button class="ghost">'
                    'Back</button></a></div></div>')
    static = flag(f.get("static"))
    engine = f.get("engine", "eevee")
    samples = f.get("samples", "64")
    percent = f.get("percent", "100")
    transparent = flag(f.get("transparent"))
    # persist the shared scene profile so the other shots in this clip reuse it
    prof = load_scene_profile(footage)
    prof.update(scene=scene, engine=engine, samples=samples, percent=percent,
                transparent=transparent)
    save_scene_profile(footage, prof)

    n = 1 + sum(1 for e in render_queue
                if e.get("footage") == footage and e.get("shot") == shot)
    shotname = os.path.splitext(os.path.basename(footage))[0]
    render = f.get("render", "").strip().strip('"') or \
        default_render_path(footage, shot, f"_take{n:02d}" if n > 1 else "")
    entry = {
        "name": f"{shotname} · shot {shot}" + (f" · take {n}" if n > 1 else ""),
        "kind": "static" if static else "track",
        "footage": footage, "shot": shot, "scene": scene,
        "start": f.get("start", "0,0,0"),
        "rotation": f.get("rotation", "0,0,0"),
        "scale": f.get("scale", "1.0"),
        "lens": f.get("lens", "").strip(),
        "focus": f.get("focus", "").strip(),
        "render": render, "engine": engine, "samples": samples,
        "percent": percent, "transparent": transparent,
        "frames_total": max(_shot_frames(footage, shot), 1), "frames_done": 0,
        "stage": "queued", "status": "queued",
    }
    with queue_lock:
        render_queue.append(entry)
        save_queue()
    from urllib.parse import quote
    return redirect(f"/shots?footage={quote(footage)}&queued={shot}")


@app.route("/queue_add_tracked", methods=["POST"])
def queue_add_tracked():
    """Add an ALREADY-tracked shot (from the result page, after 'Track this
    shot now') to the queue as a render-only take — snapshotting the current
    TrackRoot placement so you can render several placements in one batch."""
    footage = request.form["footage"]
    shot = int(request.form["shot"])
    tag = f"shot_{shot:02d}"
    meta = dict(job["meta"]) if (job["meta"].get("footage") == footage and
                                 job["meta"].get("shot") == shot) else {}
    scene = meta.get("scene", "")
    if not scene:
        return redirect(request.referrer or "/")
    base = os.path.splitext(os.path.basename(scene))[0]
    scene_out = os.path.join(workdir_for(footage), f"{tag}_{base}_tracked.blend")
    if not os.path.exists(scene_out):
        return redirect(request.referrer or "/")
    n = 1 + sum(1 for e in render_queue
                if e.get("footage") == footage and e.get("shot") == shot)
    blend_copy = os.path.join(workdir_for(footage),
                              f"{tag}_{base}_queued_{n:02d}.blend")
    import shutil
    shutil.copy2(scene_out, blend_copy)
    shotname = os.path.splitext(os.path.basename(footage))[0]
    entry = {
        "name": f"{shotname} · shot {shot} · take {n}",
        "footage": footage, "shot": shot, "blend": blend_copy,  # render-only
        "render": default_render_path(footage, shot, f"_take{n:02d}"),
        "engine": meta.get("engine", "eevee"),
        "samples": meta.get("samples", "64"),
        "percent": meta.get("percent", "100"),
        "transparent": bool(meta.get("transparent")),
        "frames_total": max(_shot_frames(footage, shot), 1), "frames_done": 0,
        "stage": "queued", "status": "queued",
    }
    with queue_lock:
        render_queue.append(entry)
        save_queue()
    _remember_placement(scene_out, footage, shot)
    return redirect("/job")


@app.route("/save_tracking", methods=["POST"])
def save_tracking_route():
    """Keep a shot's solve permanently (otherwise it is forgotten at the
    next batch / app start)."""
    footage = request.form.get("footage", "")
    shot = request.form.get("shot", "")
    if footage and shot:
        mark_track_saved(footage, int(shot))
    return redirect(request.referrer or "/queue")


@app.route("/queue_start", methods=["POST"])
def queue_start_route():
    start_queue()
    return redirect("/job")


@app.route("/queue_stop", methods=["POST"])
def queue_stop_route():
    stop_queue()
    return redirect("/job")


@app.route("/queue_remove", methods=["POST"])
def queue_remove():
    idx = int(request.form["idx"])
    with queue_lock:
        if 0 <= idx < len(render_queue) and \
                render_queue[idx]["status"] != "rendering":
            render_queue.pop(idx)
            save_queue()
    return redirect("/job")


def _entry_progress(e):
    """0..1 completion of one queue entry across its whole track+render job."""
    if e["status"] == "done":
        return 1.0
    if e["status"] == "queued":
        return 0.0
    if "progress" in e:  # live phase-weighted fraction (tracking + render)
        return max(0.0, min(e["progress"], 1.0))
    ft = max(e.get("frames_total", 1), 1)  # legacy fallback
    return min(e.get("frames_done", 0) / ft, 1.0)


@app.route("/queue_status")
def queue_status():
    active = [e for e in render_queue if e["status"] != "failed"]
    wsum = sum(e["frames_total"] for e in active)
    # weight each shot's progress by its length so the overall bar is fair
    done_w = sum(_entry_progress(e) * e["frames_total"] for e in active)
    # render-frame ETA (meaningful during the render phase)
    rem_frames = sum(e["frames_total"] - e["frames_done"] for e in active
                     if e["status"] != "done")
    spf = queue_state.get("spf")
    eta = round(rem_frames * spf) if (spf and queue_state["running"]) else None
    return jsonify({
        "running": queue_state["running"],
        "overall_pct": round(100 * done_w / wsum, 1) if wsum else 0,
        "eta_s": eta,
        "entries": [{"name": e["name"], "status": e["status"],
                     "stage": e.get("stage", ""),
                     "pct": round(100 * _entry_progress(e), 1)}
                    for e in render_queue],
    })


def _remember_placement(scene_out, footage, shot):
    """Probe TrackRoot from the saved scene and file it as a placement."""
    try:
        expr = ("import bpy, json, math\n"
                "o = bpy.data.objects.get('TrackRoot')\n"
                "if o:\n"
                "    l, r, s = o.matrix_world.decompose()\n"
                "    e = r.to_euler('XYZ')\n"
                "    print('PLACEMENT ' + json.dumps({'loc': list(l), "
                "'rot_deg': [math.degrees(a) for a in e], 'scale': s[0]}))\n")
        pr = subprocess.run([BLENDER, "-b", scene_out, "--python-expr", expr],
                            capture_output=True, text=True, timeout=120,
                            encoding="utf-8", errors="replace")
        for line in (pr.stdout or "").splitlines():
            if line.startswith("PLACEMENT "):
                d = json.loads(line[10:])
                shotname = os.path.splitext(os.path.basename(footage))[0]
                save_placement(f"{shotname} · shot {shot}",
                               d["loc"], d["rot_deg"], d["scale"])
                break
    except Exception:
        pass  # placement memory is best-effort


@app.route("/render_final", methods=["POST"])
def render_final():
    """Render the (possibly user-repositioned) tracked scene with the
    settings chosen on the track form."""
    footage = request.form["footage"]
    shot = int(request.form["shot"])
    tag = f"shot_{shot:02d}"
    meta = dict(job["meta"]) if (job["meta"].get("footage") == footage and
                                 job["meta"].get("shot") == shot) else {}
    scene = meta.get("scene", "")
    if not scene:
        return redirect("/")
    base = os.path.splitext(os.path.basename(scene))[0]
    scene_out = os.path.join(workdir_for(footage), f"{tag}_{base}_tracked.blend")
    render = meta.get("render") or default_render_path(footage, shot)

    # remember where the user placed TrackRoot, for reuse on similar shots
    _remember_placement(scene_out, footage, shot)
    cmd = [BLENDER, "--factory-startup", scene_out,
           "-P", os.path.join(HERE, "render_stage4.py"), "--",
           "--out", render, "--engine", meta.get("engine", "eevee")]
    if meta.get("samples"):
        cmd += ["--samples", str(meta["samples"])]
    if meta.get("percent") and str(meta["percent"]) != "100":
        cmd += ["--percent", str(meta["percent"])]
    if meta.get("transparent"):
        cmd += ["--transparent"]
    meta.update(footage=footage, shot=shot, render=render, rendered=True,
                render_total=_shot_frames(footage, shot))
    started = start_job(cmd, "track", meta)
    if not started:
        return page('<div class="card">Another job is already running — '
                    '<a href="/job">watch it</a>.</div>')
    return redirect("/job")


@app.route("/reveal", methods=["POST"])
def reveal():
    """Open Explorer with the file selected (its folder, file highlighted)."""
    path = request.form.get("path", "")
    if os.path.exists(path):
        subprocess.Popen(["explorer", "/select,", os.path.normpath(path)])
    return redirect(request.referrer or "/")


@app.route("/export_camera", methods=["POST"])
def export_camera():
    footage = request.form["footage"]
    shot = int(request.form["shot"])
    fmt = request.form.get("format", "fbx")
    meta = job["meta"] if job["meta"].get("footage") == footage else {}
    scene = meta.get("scene", "")
    tag = f"shot_{shot:02d}"
    wd = workdir_for(footage)
    # find the tracked scene .blend (has TrackedCamera)
    scene_out = None
    if scene:
        base = os.path.splitext(os.path.basename(scene))[0]
        cand = os.path.join(wd, f"{tag}_{base}_tracked.blend")
        if os.path.exists(cand):
            scene_out = cand
    if scene_out is None:  # fall back to any tracked blend for this shot
        hits = glob.glob(os.path.join(wd, f"{tag}_*_tracked.blend"))
        scene_out = hits[0] if hits else None
    if not scene_out:
        return page('<div class="card">No tracked scene to export — track '
                    'the shot with a scene first.<div class="actions">'
                    '<a href="javascript:history.back()"><button class="ghost">'
                    'Back</button></a></div></div>')
    ext = "abc" if fmt == "abc" else "fbx"
    out = default_export_path(footage, shot, ext)
    r = subprocess.run([BLENDER, "-b", scene_out, "-P",
                        os.path.join(HERE, "export_camera.py"), "--",
                        "--out", out, "--format", ext],
                       capture_output=True, text=True, timeout=300,
                       encoding="utf-8", errors="replace")
    if os.path.exists(out):
        subprocess.Popen(["explorer", "/select,", os.path.normpath(out)])
        return page(f'<div class="card"><h3>Camera exported</h3>'
                    f'<p class="hint">{out}</p><div class="actions">'
                    f'<a href="javascript:history.back()"><button class="ghost">'
                    f'Back to shot</button></a></div></div>')
    dlog("export_camera failed: " + (r.stderr or "")[-400:])
    return page('<div class="card">Export failed — see the log. '
                '<a href="javascript:history.back()">Back</a></div>')


# ------------------------------------------------------------------ startup
class WindowApi:
    """Window controls for the custom title bar (frameless window)."""

    def __init__(self):
        self._maximized = False

    def minimize(self):
        if webview_window:
            webview_window.minimize()

    def toggle_max(self):
        if not webview_window:
            return
        if self._maximized:
            webview_window.restore()
        else:
            webview_window.maximize()
        self._maximized = not self._maximized

    def toggle_fullscreen(self):
        if webview_window:
            webview_window.toggle_fullscreen()

    def close(self):
        if webview_window:
            webview_window.destroy()

    def raise_window(self):
        """Bring the app window to the front (after Blender closes)."""
        if not webview_window:
            return
        try:
            webview_window.restore()
        except Exception:
            pass
        try:
            webview_window.on_top = True
            webview_window.on_top = False  # flash to front without pinning
        except Exception:
            pass


def port_status(port):
    """'ours' if this app already serves the port, 'busy', or 'free'."""
    import urllib.request
    import socket
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as r:
            body = r.read(8192)
            return "ours" if b"autotrack-app" in body else "busy"
    except urllib.error.URLError:
        pass
    except Exception:
        return "busy"
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
            return "free"
        except OSError:
            return "busy"


def _open_in_browser_and_block(url, reason=None):
    """Fallback when no native-window backend is available (e.g. the WebView2
    runtime isn't installed): open the UI in the default browser and keep the
    server alive so the app is fully usable anywhere."""
    import webbrowser
    if reason:
        print(f"[{APP_NAME}] Native window unavailable ({reason}).")
    print(f"[{APP_NAME}] Opening in your web browser: {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    print(f"[{APP_NAME}] Running. Leave this window open. "
          f"If the browser didn't open, go to {url}")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        os._exit(0)


def run_window(url):
    """Native frameless window with our own title bar (WebView2 runtime).
    Falls back to the system browser if no webview backend exists, so the
    app runs on any machine — with or without WebView2 installed."""
    global webview_window
    try:
        import webview
        webview_window = webview.create_window(
            APP_NAME, url, width=1180, height=860, min_size=(940, 700),
            frameless=True, easy_drag=False, background_color="#8ecdf5",
            js_api=WindowApi())
        for backend in ("edgechromium", None):
            try:
                webview.start(gui=backend) if backend else webview.start()
                break
            except Exception:
                continue
        else:
            raise RuntimeError("no available webview backend")
    except Exception as e:
        _open_in_browser_and_block(url, e)
        return
    # window closed: stop any running pipeline subprocess (and its Blender
    # children) before exiting, so nothing keeps rendering as an orphan
    if job["status"] == "running":
        kill_tree(job.get("proc"))
    kill_tree(queue_state.get("proc"))
    sweep_pipeline_processes()   # closing the app stops ALL of our work
    cleanup_unsaved_tracks()     # unsaved solves don't outlive the session
    os._exit(0)


if __name__ == "__main__":
    server_only = "--server-only" in sys.argv or "--no-browser" in sys.argv

    port = PORT
    status = port_status(port)
    if status == "ours":
        url = f"http://127.0.0.1:{port}"
        print(f"{APP_NAME} is already running - attaching to {url}")
        if server_only:
            sys.exit(0)
        run_window(url)  # second window onto the same running app
    if status == "busy":  # something else owns the port; pick a free one
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        print(f"Port {PORT} is in use by another program; using {port} instead.")

    url = f"http://127.0.0.1:{port}"
    if server_only:
        app.run(host="127.0.0.1", port=port, debug=False)
    else:
        threading.Thread(target=lambda: app.run(host="127.0.0.1", port=port,
                                                debug=False),
                         daemon=True).start()
        run_window(url)
