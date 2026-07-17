"""
Stage 1: Headless Blender Auto-Tracking Pipeline
--------------------------------------------------
Run with:
    blender -b -P auto_track_stage1.py -- /path/to/footage.mp4 /path/to/output_dir

What this does:
  1. Creates a new .blend, loads the footage as a movie clip
  2. Runs feature detection + tracking + camera solve
  3. Logs solve quality metrics to a JSON file (for later stages to consume)
  4. Saves the resulting .blend with tracking data baked in

This is intentionally simple — no masking, no quality gating, no parameter
tuning yet. It's the foundation the rest of the pipeline builds on.
"""

import bpy
import sys
import os
import json
from datetime import datetime


def get_cli_args():
    """Blender passes script args after '--'."""
    argv = sys.argv
    if "--" not in argv:
        raise ValueError("Usage: blender -b -P auto_track_stage1.py -- <footage_path> <output_dir>")
    argv = argv[argv.index("--") + 1:]
    if len(argv) < 2:
        raise ValueError("Usage: blender -b -P auto_track_stage1.py -- <footage_path> <output_dir>")
    return argv[0], argv[1]


def setup_clip(footage_path):
    """Load footage into a new movie clip in the current scene."""
    if not os.path.exists(footage_path):
        raise FileNotFoundError(f"Footage not found: {footage_path}")

    bpy.ops.clip.open(directory=os.path.dirname(footage_path),
                       files=[{"name": os.path.basename(footage_path)}])
    clip = bpy.data.movieclips[-1]
    return clip


def get_clip_editor_context(clip):
    """
    Tracking ops need a CLIP_EDITOR context override since we're headless
    (no actual editor area exists on screen). We fake one.
    """
    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type == 'CLIP_EDITOR':
                # poll() checks space_data.clip, not just the override dict —
                # the clip must actually be assigned to the editor's space
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


def run_tracking(clip, settings=None):
    """
    Run detect -> track -> solve.
    `settings` lets later stages pass tuned parameters; sane defaults for now.
    """
    settings = settings or {}
    detect_threshold = settings.get("detect_threshold", 0.3)
    detect_margin = settings.get("detect_margin", 16)
    detect_min_distance = settings.get("detect_min_distance", 50)

    # The clip's tracking camera defaults rarely match the real lens; let the
    # solver refine focal length (and optionally set knowns via settings).
    if "focal_length_mm" in settings:
        clip.tracking.camera.focal_length = settings["focal_length_mm"]
    if "sensor_width_mm" in settings:
        clip.tracking.camera.sensor_width = settings["sensor_width_mm"]
    ts = clip.tracking.settings
    if hasattr(ts, "refine_intrinsics_focal_length"):
        ts.refine_intrinsics_focal_length = True
    else:  # older API (< 3.5)
        ts.refine_intrinsics = 'FOCAL_LENGTH'
    # Let the solver pick the keyframe pair with the best parallax
    ts.use_keyframe_selection = True
    # Larger pattern = more context per marker = less drift
    ts.default_pattern_size = settings.get("pattern_size", 21)
    ts.default_motion_model = settings.get("motion_model", 'LocRot')
    clean_error = settings.get("clean_error_threshold", 1.5)

    ctx = get_clip_editor_context(clip)
    if ctx is None:
        # Headless fallback: no clip editor area exists yet.
        # We create a temporary screen layout with one to satisfy the op's poll().
        raise RuntimeError(
            "No CLIP_EDITOR area found. Run this from a .blend that already "
            "has a Movie Clip Editor area saved in its screen layout, or use "
            "the --factory-startup workaround documented in the README."
        )

    with bpy.context.temp_override(**ctx):
        bpy.ops.clip.detect_features(
            threshold=detect_threshold,
            margin=detect_margin,
            min_distance=detect_min_distance,
        )
        bpy.ops.clip.select_all(action='SELECT')
        bpy.ops.clip.track_markers(backwards=False, sequence=True)

        # Rewind and track backwards too, so features detected mid-clip
        # still get tracked across the full range
        bpy.context.scene.frame_set(clip.frame_start)
        bpy.ops.clip.track_markers(backwards=True, sequence=True)

        bpy.ops.clip.solve_camera()

        # Clean-and-resolve: drop tracks with high reprojection error, then
        # solve again on the survivors. Deleting is capped so we always keep
        # enough tracks for a stable second solve.
        keep_min = settings.get("keep_min_tracks", 12)
        tracks = clip.tracking.tracks
        solved = sorted((t for t in tracks if t.average_error > 0),
                        key=lambda t: t.average_error, reverse=True)
        max_deletable = max(0, len(solved) - keep_min)
        doomed = [t for t in solved if t.average_error > clean_error][:max_deletable]
        if doomed:
            for t in tracks:
                t.select = t in doomed
            bpy.ops.clip.delete_track()
            bpy.ops.clip.select_all(action='SELECT')
            bpy.ops.clip.solve_camera()

    return clip


def collect_metrics(clip):
    """Pull solve quality info out of the clip's tracking data."""
    tracking = clip.tracking
    reconstruction = tracking.reconstruction

    num_tracks = len(tracking.tracks)
    num_solved = sum(1 for t in tracking.tracks if t.average_error is not None)

    metrics = {
        "timestamp": datetime.utcnow().isoformat(),
        "clip_name": clip.name,
        "frame_start": clip.frame_start,
        "frame_duration": clip.frame_duration,
        "num_tracks_detected": num_tracks,
        "solve_ok": reconstruction.is_valid,
        "average_solve_error": reconstruction.average_error if reconstruction.is_valid else None,
        "per_track_errors": [
            {"name": t.name, "avg_error": t.average_error}
            for t in tracking.tracks if t.average_error is not None
        ],
    }
    return metrics


def main():
    footage_path, output_dir = get_cli_args()
    os.makedirs(output_dir, exist_ok=True)

    clip = setup_clip(footage_path)
    clip = run_tracking(clip)
    metrics = collect_metrics(clip)

    base_name = os.path.splitext(os.path.basename(footage_path))[0]

    # Save metrics log
    log_path = os.path.join(output_dir, f"{base_name}_track_log.json")
    with open(log_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save the .blend with tracking data
    blend_path = os.path.join(output_dir, f"{base_name}_tracked.blend")
    bpy.ops.wm.save_as_mainfile(filepath=blend_path)

    print(f"[auto_track] Done. Solve error: {metrics['average_solve_error']}")
    print(f"[auto_track] Log saved to: {log_path}")
    print(f"[auto_track] Blend saved to: {blend_path}")


if __name__ == "__main__":
    main()
