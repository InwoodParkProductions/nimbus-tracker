"""
Stage 3: Scene handoff — put the solved camera into the user's .blend.
----------------------------------------------------------------------
Run with:
    blender -b <your_scene.blend> -P apply_track_stage3.py -- <tracked.blend>
        [--start x,y,z]        camera position at the first frame (scene units)
        [--rotation rx,ry,rz]  extra rotation in degrees applied to the whole
                               solved path around the start point (default none)
        [--scale s]            scales the solved camera motion (solves have
                               arbitrary scale; default 1.0)
        [--name CamName]       name for the created camera (default TrackedCamera)
        [--out path.blend]     output file (default <scene>_tracked.blend)

What it does:
  1. Appends the movie clip (with its tracking + reconstruction) from the
     Stage-2 tracked .blend into your scene.
  2. Creates a camera driven by a Camera Solver constraint, evaluates it on
     every frame, and BAKES the result to plain location/rotation keyframes —
     so your file has an ordinary animated camera, no constraint magic left.
  3. Re-anchors the whole path so the first-frame camera sits exactly at
     --start (with optional --rotation / --scale), sets the camera's focal
     length and sensor width from the solve, and matches the scene frame
     range and resolution to the footage.
  4. Sets the footage as the camera's background movie (visible in the
     viewport through the camera) and saves the result.
"""

import bpy
import sys
import os
import json
import math
from mathutils import Matrix, Vector, Euler


def get_cli_args():
    argv = sys.argv
    if "--" not in argv:
        raise ValueError("Usage: blender -b scene.blend -P apply_track_stage3.py -- "
                         "<tracked.blend> [--start x,y,z] [--rotation rx,ry,rz] "
                         "[--scale s] [--name N] [--out path.blend]")
    argv = argv[argv.index("--") + 1:]
    if len(argv) < 1:
        raise ValueError("Need the Stage-2 tracked .blend (or --flow json) "
                         "as first argument")

    def opt(flag, default=None):
        return argv[argv.index(flag) + 1] if flag in argv else default

    flow_json = opt("--flow")
    footage = opt("--footage")  # for camera background in flow mode
    tracked_blend = None if flow_json else argv[0]
    start = opt("--start", "0,0,0")
    rotation = opt("--rotation", "0,0,0")
    scale = float(opt("--scale", "1.0"))
    name = opt("--name", "TrackedCamera")
    out = opt("--out")
    render_size = opt("--render-size")  # "WxH" of the ORIGINAL plate when
    # tracking ran on a downscaled proxy (the solve itself is
    # resolution-independent)
    if render_size:
        render_size = tuple(int(v) for v in render_size.lower().split("x"))
    lens_mm = opt("--lens-mm")
    focus_distance = opt("--focus-distance")
    start = Vector([float(v) for v in start.split(",")])
    rotation = Euler([math.radians(float(v)) for v in rotation.split(",")], 'XYZ')
    return (tracked_blend, start, rotation, scale, name, out, render_size,
            flow_json, footage,
            float(lens_mm) if lens_mm else None,
            float(focus_distance) if focus_distance else None)


def append_clip(tracked_blend):
    """Append the movie clip (tracking data rides along inside it)."""
    if not os.path.exists(tracked_blend):
        raise FileNotFoundError(f"Tracked blend not found: {tracked_blend}")
    before = set(bpy.data.movieclips)
    with bpy.data.libraries.load(tracked_blend, link=False) as (src, dst):
        if not src.movieclips:
            raise RuntimeError(f"No movie clip found in {tracked_blend}")
        dst.movieclips = src.movieclips
    new_clips = [c for c in bpy.data.movieclips if c not in before]
    clip = new_clips[-1]
    if not clip.tracking.reconstruction.is_valid:
        raise RuntimeError(f"Clip in {tracked_blend} has no valid camera solve")
    return clip


def solve_matrices(clip):
    """World matrix of the solver-driven camera on every clip frame."""
    scene = bpy.context.scene
    cam_data = bpy.data.cameras.new("_solver_tmp")
    cam = bpy.data.objects.new("_solver_tmp", cam_data)
    scene.collection.objects.link(cam)
    con = cam.constraints.new(type='CAMERA_SOLVER')
    con.use_active_clip = False
    con.clip = clip

    frame_start = clip.frame_start
    frame_end = clip.frame_start + clip.frame_duration - 1
    depsgraph = bpy.context.evaluated_depsgraph_get()
    matrices = {}
    for f in range(frame_start, frame_end + 1):
        scene.frame_set(f)
        depsgraph.update()
        matrices[f] = cam.evaluated_get(depsgraph).matrix_world.copy()

    bpy.data.objects.remove(cam)
    bpy.data.cameras.remove(cam_data)
    return matrices


def make_track_root(start, rotation, scale):
    """Control empty: grab/rotate/scale THIS in the viewport to place the
    whole camera path visually. --start/--rotation/--scale just set its
    initial transform."""
    root = bpy.data.objects.new("TrackRoot", None)
    root.empty_display_type = 'ARROWS'
    root.empty_display_size = 1.5
    bpy.context.scene.collection.objects.link(root)
    root.matrix_world = (Matrix.Translation(start)
                         @ rotation.to_matrix().to_4x4()
                         @ Matrix.Scale(scale, 4))
    return root


def bake_camera(clip, matrices, start, rotation, scale, name):
    """Create the final camera with plain keyframes, parented to a TrackRoot
    empty anchored at `start` — so the path can be repositioned visually in
    Blender afterwards by moving the empty."""
    frames = sorted(matrices)
    m0 = matrices[frames[0]]
    p0 = m0.to_translation()

    root = make_track_root(start, rotation, scale)

    cam_data = bpy.data.cameras.new(name)
    cam_data.sensor_width = clip.tracking.camera.sensor_width
    cam_data.lens = clip.tracking.camera.focal_length
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.rotation_mode = 'QUATERNION'
    cam.parent = root

    # keyframes are LOCAL to the root: path starts at the root's origin
    for f in frames:
        m = Matrix.Translation(-p0) @ matrices[f]
        loc, rot, _ = m.decompose()
        cam.location = loc
        cam.rotation_quaternion = rot
        cam.keyframe_insert(data_path="location", frame=f)
        cam.keyframe_insert(data_path="rotation_quaternion", frame=f)

    # Footage as camera background for eyeballing the match in the viewport
    bg = cam_data.background_images.new()
    bg.source = 'MOVIE_CLIP'
    bg.clip = clip
    bg.alpha = 1.0
    cam_data.show_background_images = True
    return cam


def bake_flow_camera(flow_json, footage, start, rotation, scale, name):
    """Faux camera from a 2D flow solve (see flow_solve.py): pan/tilt/roll
    from the rotation path, plus a forward/back dolly from the measured zoom
    so push-ins and pull-outs are followed too — not just swing."""
    from mathutils import Quaternion
    with open(flow_json) as f:
        flow = json.load(f)

    root = make_track_root(start, rotation, 1.0)

    cam_data = bpy.data.cameras.new(name)
    cam_data.sensor_width = flow.get("sensor_width_mm", 36.0)
    cam_data.lens = flow.get("focal_mm", 35.0)
    cam = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam)
    cam.rotation_mode = 'QUATERNION'
    cam.parent = root

    quats = flow["quaternions_wxyz"]
    # Assumed subject distance (Blender units): the image scale S means the
    # subject looks S x bigger, i.e. the camera sits at D0/S, so it dollied
    # forward (toward -Z) by D0*(1 - 1/S). --scale tunes the throw.
    subj_dist = 6.0 * (scale if scale else 1.0)
    scales = flow.get("scale_cum") or [1.0] * len(quats)

    # local to root: root carries position/orientation, the camera rotates
    # (pan/tilt/roll) and dollies along its view axis (push-in/out)
    for i, q in enumerate(quats, start=1):
        cam.rotation_quaternion = Quaternion(q)
        s = scales[i - 1] if i - 1 < len(scales) else scales[-1]
        cam.location = (0.0, 0.0, -subj_dist * (1.0 - 1.0 / s))
        cam.keyframe_insert(data_path="rotation_quaternion", frame=i)
        cam.keyframe_insert(data_path="location", frame=i)

    if footage and os.path.exists(footage):
        bpy.ops.clip.open(directory=os.path.dirname(footage),
                          files=[{"name": os.path.basename(footage)}])
        clip = bpy.data.movieclips[-1]
        bg = cam_data.background_images.new()
        bg.source = 'MOVIE_CLIP'
        bg.clip = clip
        bg.alpha = 1.0
        cam_data.show_background_images = True
    return cam, flow


def apply_lens(cam, lens_mm, focus_distance):
    """Optional manual camera settings: fixed lens and DoF focus distance."""
    if lens_mm:
        cam.data.lens = lens_mm
    if focus_distance is not None:
        cam.data.dof.use_dof = True
        cam.data.dof.focus_distance = focus_distance


def main():
    (tracked_blend, start, rotation, scale, name, out, render_size,
     flow_json, footage, lens_mm, focus_distance) = get_cli_args()
    scene = bpy.context.scene

    if flow_json:
        cam, flow = bake_flow_camera(flow_json, footage, start, rotation,
                                     scale, name)
        apply_lens(cam, lens_mm, focus_distance)
        scene.camera = cam
        scene.frame_start = 1
        scene.frame_end = flow["num_frames"]
        scene.render.resolution_x = render_size[0] if render_size else flow["size"][0]
        scene.render.resolution_y = render_size[1] if render_size else flow["size"][1]
        scene.render.fps = round(flow.get("fps") or 24)
        if not out:
            base = bpy.data.filepath or os.path.join(os.getcwd(), "scene.blend")
            out = os.path.splitext(base)[0] + "_tracked.blend"
        bpy.ops.wm.save_as_mainfile(filepath=out)
        print(f"[stage3] Flow camera '{cam.name}' baked over frames "
              f"1-{flow['num_frames']} (rotation-only 2D motion match)")
        print(f"[stage3] Start position: {[round(v, 4) for v in start]}")
        print(f"[stage3] Saved: {out}")
        return

    clip = append_clip(tracked_blend)
    matrices = solve_matrices(clip)
    cam = bake_camera(clip, matrices, start, rotation, scale, name)
    apply_lens(cam, lens_mm, focus_distance)
    scene.camera = cam

    # Match scene timing/format to the footage. If tracking ran on a proxy,
    # --render-size restores the original plate resolution for rendering.
    scene.frame_start = clip.frame_start
    scene.frame_end = clip.frame_start + clip.frame_duration - 1
    scene.render.resolution_x = render_size[0] if render_size else clip.size[0]
    scene.render.resolution_y = render_size[1] if render_size else clip.size[1]
    if clip.fps > 0:
        scene.render.fps = round(clip.fps)

    if not out:
        base = bpy.data.filepath or os.path.join(os.getcwd(), "scene.blend")
        out = os.path.splitext(base)[0] + "_tracked.blend"
    bpy.ops.wm.save_as_mainfile(filepath=out)

    scene.frame_set(min(matrices))
    summary = {
        "camera": cam.name,
        "frame_range": [scene.frame_start, scene.frame_end],
        "start_position": [round(v, 4) for v in cam.matrix_world.to_translation()],
        "focal_length_mm": clip.tracking.camera.focal_length,
        "solve_error_px": clip.tracking.reconstruction.average_error,
        "output_blend": out,
    }
    log_path = os.path.splitext(out)[0] + "_handoff.json"
    with open(log_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[stage3] Camera '{cam.name}' baked over frames "
          f"{scene.frame_start}-{scene.frame_end}")
    print(f"[stage3] Start position: {summary['start_position']} (requested {list(start)})")
    print(f"[stage3] Saved: {out}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
