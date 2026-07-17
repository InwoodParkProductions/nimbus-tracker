"""
Stage 2: Mask-aware Headless Blender Auto-Tracking
--------------------------------------------------
Run with:
    blender -b template.blend -P auto_track_stage2.py -- <footage> <output_dir> [--masks <masks_dir>]

Same flow as Stage 1 (detect -> track -> solve -> clean -> resolve), plus:
  - If --masks is given (a directory of mask_NNNNNN.png files from
    segment_people.py, white = person = exclude):
      1. Features detected inside the person mask on the detect frame are
         deleted before tracking starts.
      2. After tracking, every marker that lands inside the person mask on
         its frame is muted, so it can't feed the solve.
      3. Tracks left with too few live markers are deleted entirely.

Without --masks it behaves exactly like Stage 1.
"""

import bpy
import sys
import os
import json
import re
from datetime import datetime

import numpy as np


def get_cli_args():
    argv = sys.argv
    if "--" not in argv:
        raise ValueError("Usage: blender -b template.blend -P auto_track_stage2.py -- "
                         "<footage> <output_dir> [--masks <masks_dir>]")
    argv = argv[argv.index("--") + 1:]
    if len(argv) < 2:
        raise ValueError("Need <footage> and <output_dir> arguments")
    footage, output_dir = argv[0], argv[1]
    masks_dir = None
    if "--masks" in argv:
        masks_dir = argv[argv.index("--masks") + 1]
    settings = {}
    if "--settings" in argv:
        raw = argv[argv.index("--settings") + 1]
        if os.path.isfile(raw):
            with open(raw) as f:
                settings = json.load(f)
        else:
            settings = json.loads(raw)
    return footage, output_dir, masks_dir, settings


class MaskStack:
    """Per-frame boolean person masks, indexed by clip frame (1-based).

    Loads mask PNGs through bpy.data.images so no extra dependencies are
    needed inside Blender. Blender presents pixel rows bottom-up, which
    matches marker.co's bottom-left-origin normalized coordinates.
    """

    def __init__(self, masks_dir):
        self.masks = {}
        pat = re.compile(r"mask_(\d+)\.png$")
        for fn in sorted(os.listdir(masks_dir)):
            m = pat.match(fn)
            if not m:
                continue
            frame = int(m.group(1))
            img = bpy.data.images.load(os.path.join(masks_dir, fn))
            w, h = img.size
            buf = np.empty(w * h * 4, dtype=np.float32)
            img.pixels.foreach_get(buf)
            self.masks[frame] = buf.reshape(h, w, 4)[:, :, 0] > 0.5
            bpy.data.images.remove(img)
        if not self.masks:
            raise FileNotFoundError(f"No mask_NNNNNN.png files found in {masks_dir}")
        print(f"[stage2] Loaded {len(self.masks)} masks from {masks_dir}")

    def is_person(self, frame, x_norm, y_norm):
        """True if normalized clip coords (bottom-left origin) hit a person."""
        mask = self.masks.get(frame)
        if mask is None:
            return False
        h, w = mask.shape
        px = min(max(int(x_norm * w), 0), w - 1)
        py = min(max(int(y_norm * h), 0), h - 1)
        return bool(mask[py, px])


def setup_clip(footage_path):
    if not os.path.exists(footage_path):
        raise FileNotFoundError(f"Footage not found: {footage_path}")
    bpy.ops.clip.open(directory=os.path.dirname(footage_path),
                      files=[{"name": os.path.basename(footage_path)}])
    return bpy.data.movieclips[-1]


def tracks_from_points(clip, points_json, stats):
    """Build clip tracks from cotrack_points.py output instead of detecting
    and KLT-tracking them here.

    Only the front-end changes: the solve chain below is untouched and gets
    the same thing it always got — a clip full of tracks with per-frame
    markers. Coordinates in the json are already normalized with a bottom-left
    origin, which is marker.co's convention, so they drop straight in at any
    clip resolution.
    """
    with open(points_json) as f:
        data = json.load(f)
    fs = clip.frame_start
    n_marks = 0
    n_tracks = 0
    for i, pts in enumerate(data["tracks"]):
        # Insert markers ONLY on frames where the point is visible — never a
        # muted marker. mute_person_markers reads a pre-muted marker as
        # "this frame was inside a person mask" (it's how the KLT path passes
        # that information along), so muting for OCCLUSION here gets counted
        # as person-contact. On shots where people cover 57-71% of frame that
        # pushed every track past the "lives on a person" threshold and
        # deleted all of them: shot 09 went 152 tracks -> 0, shot 10 -> 1.
        # A gap in a track is expressed by the marker's absence.
        vis = [(t, x, y) for t, (x, y, v) in enumerate(pts) if v]
        if len(vis) < 2:
            continue
        tr = clip.tracking.tracks.new(name=f"ct_{i:04d}", frame=fs + vis[0][0])
        for t, x, y in vis:
            tr.markers.insert_frame(fs + t, co=(x, y))
            n_marks += 1
        n_tracks += 1
    stats["markers_detected"] = n_tracks
    stats["front_end"] = "cotracker"
    stats["cotracker_seed_frame"] = data.get("seed_frame")
    print(f"[stage2] cotracker front-end: {n_tracks} tracks, "
          f"{n_marks} markers from {os.path.basename(points_json)}")
    return n_tracks


def get_clip_editor_context(clip):
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'CLIP_EDITOR':
                area.spaces.active.clip = clip
                for region in area.regions:
                    if region.type == 'WINDOW':
                        return {
                            'window': window,
                            'screen': screen,
                            'area': area,
                            'region': region,
                            'space_data': area.spaces.active,
                            'edit_movieclip': clip,
                        }
    return None


def delete_selected_tracks_only(tracks, doomed):
    """Select exactly `doomed` and run the delete op."""
    if not doomed:
        return
    for t in tracks:
        t.select = t in doomed
    bpy.ops.clip.delete_track()


def filter_detected_markers(clip, mask_stack, detect_frame, stats):
    """Delete tracks whose detection-frame marker sits on a person."""
    doomed = []
    for track in clip.tracking.tracks:
        marker = track.markers.find_frame(detect_frame)
        if marker and mask_stack.is_person(detect_frame, marker.co[0], marker.co[1]):
            doomed.append(track)
    stats["markers_deleted_at_detect"] = len(doomed)
    delete_selected_tracks_only(clip.tracking.tracks, doomed)
    print(f"[stage2] Deleted {len(doomed)} detected features on people "
          f"({len(clip.tracking.tracks)} remain)")


def mute_person_markers(clip, mask_stack, min_live_markers, stats,
                        max_person_fraction=0.35):
    """Mute every tracked marker that lands inside a person mask, then delete
    whole tracks that are really *on a person*.

    Per-frame muting alone is not enough: person detection flickers, and on
    the frames where a person is momentarily undetected the marker un-mutes
    and leaks into the solve. So a point that sits inside a person mask for
    more than ``max_person_fraction`` of its life is a person point — its few
    "background" frames are just detection dropout — and the entire track is
    dropped, not merely muted. Tracks left with too few live markers after
    muting are also dropped."""
    muted = 0
    doomed = []
    dropped_person = 0
    for track in clip.tracking.tracks:
        live = 0
        inside = 0
        total = 0
        for marker in track.markers:
            total += 1
            if marker.mute:
                inside += 1  # already muted upstream = was inside a mask
                continue
            frame = marker.frame
            if mask_stack.is_person(frame, marker.co[0], marker.co[1]):
                marker.mute = True
                muted += 1
                inside += 1
            else:
                live += 1
        if total and inside / total > max_person_fraction:
            doomed.append(track)          # predominantly a person point
            dropped_person += 1
        elif live < min_live_markers:
            doomed.append(track)          # too little background signal left
    stats["markers_muted_in_masks"] = muted
    stats["tracks_dropped_on_person"] = dropped_person
    stats["tracks_deleted_too_short"] = len(doomed) - dropped_person
    delete_selected_tracks_only(clip.tracking.tracks, doomed)
    print(f"[stage2] Muted {muted} markers inside person masks; dropped "
          f"{dropped_person} tracks that lived on a person and "
          f"{len(doomed) - dropped_person} left too short "
          f"({len(clip.tracking.tracks)} background tracks remain)")


class NoSolveError(Exception):
    """Shot has no recoverable camera information — report, don't crash."""


def _solve_and_finish(clip, ts, mask_stack, settings, stats, fs, fd,
                      clean_error, min_live_markers):
    """The solve chain, shared by both front-ends.

    Extracted verbatim so the classic detect+KLT path and the cotracker
    path solve identically — the front-end is the only variable. Must be
    called inside the clip-editor temp_override; `ts` is
    clip.tracking.settings, which the tripod/keyframe fallbacks mutate.
    """
    if mask_stack:
        mute_person_markers(
            clip, mask_stack, min_live_markers, stats,
            max_person_fraction=settings.get("max_person_fraction", 0.35))
        bpy.ops.clip.select_all(action='SELECT')

    min_solved = settings.get("min_solved_tracks", 5)

    def try_solve(label):
        """Solve; return average error or None. solve_camera RAISES on
        shots it can't reconstruct (pans, flat backdrops) — never let
        that kill the pipeline. Degenerate 'solves' (NaN error, or fewer
        than min_solved contributing tracks) are rejected: a 1-track
        0.00px tripod solve is noise, not a camera path."""
        bpy.ops.clip.select_all(action='SELECT')
        try:
            bpy.ops.clip.solve_camera()
        except Exception as e:
            print(f"[stage2] {label} solve failed: {e}")
            return None
        rec = clip.tracking.reconstruction
        err = rec.average_error if rec.is_valid else None
        if err is not None and err != err:  # NaN
            print(f"[stage2] {label} solve rejected: NaN error")
            return None
        # A near-zero error is a collapsed solve, not a perfect one. Real
        # footage never reconstructs below ~0.01px — 8.7e-10px is a nanometre
        # on the sensor. It means the solver found a trivial solution (all
        # points effectively coplanar / at infinity / the camera not moving),
        # which reports as EXCELLENT (production-grade) on the quality scale
        # below and then puts CG confidently nowhere. min_solved doesn't catch
        # it: shot 06 collapsed to 8.7e-10px with 48 contributing tracks.
        min_believable = settings.get("min_believable_error", 1e-3)
        if err is not None and err < min_believable:
            print(f"[stage2] {label} solve rejected: {err:.2e} px is a "
                  f"collapse, not a solve (nothing real solves below "
                  f"{min_believable:g} px)")
            return None
        n_solved = sum(1 for t in clip.tracking.tracks
                       if t.average_error > 0)
        stats["solved_tracks"] = n_solved
        if err is not None and n_solved < min_solved:
            print(f"[stage2] {label} solve rejected: only {n_solved} "
                  f"contributing tracks (min {min_solved})")
            return None
        print(f"[stage2] {label} solve: "
              f"{'%.2f px' % err if err is not None else 'invalid'} "
              f"({n_solved} tracks)")
        return err

    err = try_solve("perspective")
    stats["solve_mode"] = "perspective"

    if err is not None:
        # Clean-and-resolve — but only on a solve that actually needs it.
        #
        # Deleting tracks is destructive and cannot be undone: once they're
        # gone the previous reconstruction is unrecoverable, so `err` has to
        # take whatever the re-solve gives, better or worse. That's fine when
        # rescuing a bad solve (693px -> 1.86px, measured), and actively
        # harmful on a good one. Measured on shot 19 with the learned
        # front-end: a 1.97px perspective solve cleaned to 108px, which then
        # blew past the tripod threshold and shipped a 2.75px ROTATION-ONLY
        # solve instead of the working 3D one that was already in hand.
        #
        # So: if the solve is already inside the "good" band, leave it alone.
        clean_above = settings.get("clean_if_error_above", 3.0)
        if err > clean_above:
            keep_min = settings.get("keep_min_tracks", 12)
            tracks = clip.tracking.tracks
            solved = sorted((t for t in tracks if t.average_error > 0),
                            key=lambda t: t.average_error, reverse=True)
            max_deletable = max(0, len(solved) - keep_min)
            doomed = [t for t in solved
                      if t.average_error > clean_error][:max_deletable]
            if doomed:
                delete_selected_tracks_only(tracks, doomed)
                new_err = try_solve("perspective (cleaned)")
                if new_err is not None and new_err > err:
                    print(f"[stage2] cleaning made it worse "
                          f"({err:.2f} -> {new_err:.2f} px); the pruned tracks "
                          "are gone, so that is what we have")
                err = new_err if new_err is not None else err
        else:
            print(f"[stage2] solve is {err:.2f} px — skipping the clean pass "
                  f"(only runs above {clean_above:g} px)")

    tripod_at = settings.get("tripod_fallback_error", 8.0)

    # Manual-keyframe retry: automatic keyframe selection often reports
    # "no good keyframes" on short shots. Pick the best-covered frame in
    # each half of the shot and solve between those.
    if err is None or err > tripod_at:
        def marker_count(f):
            return sum(1 for t in clip.tracking.tracks
                       if (m := t.markers.find_frame(f)) and not m.mute)
        half = fs + fd // 2
        ka = max(range(fs, half), key=marker_count, default=fs)
        kb = max(range(half, fs + fd), key=marker_count, default=fs + fd - 1)
        if kb > ka:
            ts.use_keyframe_selection = False
            obj = clip.tracking.objects.active
            obj.keyframe_a = ka
            obj.keyframe_b = kb
            merr = try_solve(f"perspective (manual keyframes {ka}-{kb})")
            if merr is not None and (err is None or merr < err):
                err = merr
                stats["solve_mode"] = "perspective"
                stats["manual_keyframes"] = [ka, kb]

    # Tripod fallback: pans/rotations have no parallax, so the full 3D
    # solve fails or produces garbage — a rotation-only solve is the
    # correct model for those shots.
    if err is None or err > tripod_at:
        ts.use_tripod_solver = True
        terr = try_solve("tripod (rotation-only)")
        if terr is not None and (err is None or terr < err):
            stats["solve_mode"] = "tripod"
            err = terr
        elif err is not None:
            ts.use_tripod_solver = False
            try_solve("perspective (restored)")

    # Quality ceiling: a "solve" with tens of pixels of error is worse
    # than no solve — it would place CG confidently in the wrong place.
    max_err = settings.get("max_acceptable_error", 20.0)
    if err is not None and err > max_err:
        print(f"[stage2] best solve {err:.2f}px exceeds {max_err}px "
              "ceiling — reporting no usable solve")
        stats["no_solve_reason"] = (f"best achievable solve was "
                                    f"{err:.1f}px — too inaccurate to use")
        err = None

    # what run_tracking actually vouches for (reconstruction.is_valid can
    # be true even for solves we rejected as degenerate)
    stats["accepted_error"] = err
    if err is None:
        stats["solve_mode"] = None
    return clip


def run_tracking(clip, mask_stack=None, settings=None, stats=None):
    settings = settings or {}
    stats = stats if stats is not None else {}
    detect_threshold = settings.get("detect_threshold", 0.3)
    # Detection distances were tuned on 4K plates; scale them to the actual
    # clip resolution (tracking usually runs on 1080p proxies now)
    res_scale = clip.size[0] / 4096.0
    detect_margin = max(6, round(settings.get("detect_margin", 16) * res_scale))
    detect_min_distance = max(12, round(
        settings.get("detect_min_distance", 50) * res_scale))

    if "focal_length_mm" in settings:
        clip.tracking.camera.focal_length = settings["focal_length_mm"]
    if "sensor_width_mm" in settings:
        clip.tracking.camera.sensor_width = settings["sensor_width_mm"]
    ts = clip.tracking.settings
    # If the lens is known (refine_focal False), keep the given focal fixed;
    # otherwise let the solver refine focal length for the best fit.
    refine = settings.get("refine_focal", True)
    if hasattr(ts, "refine_intrinsics_focal_length"):
        ts.refine_intrinsics_focal_length = refine
        # Real lenses have barrel/pincushion distortion; solving for radial
        # distortion (K1) alongside focal length meaningfully lowers
        # reprojection error on real footage. Principal-point and tangential
        # refine are left OFF — they overfit on short/noisy tracks.
        if hasattr(ts, "refine_intrinsics_radial_distortion"):
            ts.refine_intrinsics_radial_distortion = \
                settings.get("refine_distortion", True)
    else:  # older API (< 3.5)
        ts.refine_intrinsics = 'FOCAL_LENGTH' if refine else 'NONE'
    ts.use_keyframe_selection = True
    ts.default_pattern_size = settings.get("pattern_size", 21)
    # wider search radius keeps tracks alive through fast handheld moves
    ts.default_search_size = settings.get("search_size", 101)
    ts.default_motion_model = settings.get("motion_model", 'LocRot')
    # Normalize pattern brightness while tracking — big win on dark or
    # unevenly lit footage
    if hasattr(ts, "use_default_normalization"):
        ts.use_default_normalization = True
    clean_error = settings.get("clean_error_threshold", 1.5)
    min_live_markers = settings.get("min_live_markers", 8)

    ctx = get_clip_editor_context(clip)
    if ctx is None:
        raise RuntimeError(
            "No CLIP_EDITOR area found. Run from template.blend (see README)."
        )

    points_json = settings.get("points_json")

    with bpy.context.temp_override(**ctx):
        fs = clip.frame_start
        fd = clip.frame_duration

        if points_json:
            # Learned front-end: tracks come in pre-made from cotrack_points.py
            # (background-seeded, tracked through blur). Skip detection and
            # KLT entirely; everything from mute_person_markers down is shared.
            n = tracks_from_points(clip, points_json, stats)
            if n < 8:
                raise NoSolveError(
                    f"cotracker returned only {n} background tracks — the "
                    "visible background is too small or too occluded to solve.")
            stats["detect_frame"] = stats.get("cotracker_seed_frame")
            stats["detect_threshold_used"] = None
            return _solve_and_finish(clip, ts, mask_stack, settings, stats,
                                     fs, fd, clean_error, min_live_markers)

        # Frame 1 is often the WORST frame to detect on (motion blur, person
        # filling frame). Try several frames; tracking is bidirectional so a
        # mid-shot detect frame works fine.
        candidates = sorted({fs, fs + fd // 4, fs + fd // 2,
                             fs + (3 * fd) // 4, fs + fd - 1})

        def detect_on(frame, threshold):
            bpy.context.scene.frame_set(frame)
            bpy.ops.clip.select_all(action='SELECT')
            if clip.tracking.tracks:
                bpy.ops.clip.delete_track()
            bpy.ops.clip.detect_features(
                threshold=threshold,
                margin=detect_margin,
                min_distance=detect_min_distance,
            )
            stats["markers_detected"] = len(clip.tracking.tracks)
            if mask_stack:
                filter_detected_markers(clip, mask_stack, frame, stats)
            return len(clip.tracking.tracks)

        # Adaptive detection: step the threshold down and sweep candidate
        # frames until enough BACKGROUND features exist (masked don't count).
        min_features = settings.get("min_detect_features", 40)
        threshold = detect_threshold
        best = (0, candidates[0], threshold)  # (count, frame, threshold)
        detect_frame = None
        while True:
            for frame in candidates:
                n = detect_on(frame, threshold)
                print(f"[stage2] detect frame {frame} threshold "
                      f"{threshold:g}: {n} usable features")
                if n > best[0]:
                    best = (n, frame, threshold)
                if n >= min_features:
                    detect_frame = frame
                    break
            if detect_frame is not None or threshold <= 0.0001:
                break
            threshold /= 3.0
        if detect_frame is None:
            # settle for the best frame seen across the sweep
            n, detect_frame, threshold = best
            detect_on(detect_frame, threshold)
        stats["detect_threshold_used"] = threshold
        stats["detect_frame"] = detect_frame
        if len(clip.tracking.tracks) < 8:
            raise NoSolveError(
                f"Only {len(clip.tracking.tracks)} usable background features "
                "found on any frame — the visible background is too "
                "flat/featureless for camera tracking."
            )

        bpy.context.scene.frame_set(detect_frame)
        bpy.ops.clip.select_all(action='SELECT')
        bpy.ops.clip.track_markers(backwards=False, sequence=True)
        bpy.context.scene.frame_set(detect_frame)
        bpy.ops.clip.track_markers(backwards=True, sequence=True)

        return _solve_and_finish(clip, ts, mask_stack, settings, stats, fs, fd,
                                 clean_error, min_live_markers)


def collect_metrics(clip, stats):
    tracking = clip.tracking
    reconstruction = tracking.reconstruction
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "clip_name": clip.name,
        "frame_start": clip.frame_start,
        "frame_duration": clip.frame_duration,
        "masking": stats,
        "solve_mode": stats.get("solve_mode"),
        "num_tracks_final": len(tracking.tracks),
        "solved_tracks": stats.get("solved_tracks", 0),
        "solve_ok": stats.get("accepted_error") is not None,
        "average_solve_error": stats.get("accepted_error"),
        "no_solve_reason": stats.get("no_solve_reason"),
        "per_track_errors": [
            {"name": t.name, "avg_error": t.average_error}
            for t in tracking.tracks if t.average_error is not None
        ],
    }


def main():
    footage_path, output_dir, masks_dir, settings = get_cli_args()
    os.makedirs(output_dir, exist_ok=True)

    mask_stack = MaskStack(masks_dir) if masks_dir else None

    clip = setup_clip(footage_path)
    stats = {"masks_used": bool(mask_stack), "settings": settings}
    try:
        clip = run_tracking(clip, mask_stack=mask_stack, settings=settings,
                            stats=stats)
    except NoSolveError as e:
        # A featureless shot is a REPORTABLE OUTCOME, not a crash: write the
        # log with the reason so callers show a clear verdict.
        print(f"[stage2] no solve possible: {e}")
        stats["no_solve_reason"] = str(e)
        stats["accepted_error"] = None
        stats["solve_mode"] = None
    metrics = collect_metrics(clip, stats)

    base_name = os.path.splitext(os.path.basename(footage_path))[0]
    suffix = "_masked" if mask_stack else ""

    log_path = os.path.join(output_dir, f"{base_name}{suffix}_track_log.json")
    with open(log_path, "w") as f:
        json.dump(metrics, f, indent=2)

    blend_path = os.path.join(output_dir, f"{base_name}{suffix}_tracked.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    print(f"[stage2] Done. Solve error: {metrics['average_solve_error']}")
    print(f"[stage2] Log saved to: {log_path}")
    print(f"[stage2] Blend saved to: {blend_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)  # make failures visible to callers (blender may exit 0 otherwise)
