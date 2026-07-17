"""
Place a locked-off (static) camera into a scene, matched to a plate.
--------------------------------------------------------------------
For shots with no camera motion: no solve needed, just a fixed camera at the
pose the user chose in Blender, over the shot's frame range, with the footage
as background. Produces the same TrackRoot/TrackedCamera rig + track log as a
solved shot, so the result / render / queue flow is identical.

    blender <scene.blend> -P place_static.py -- --footage <mp4>
        --start x,y,z --rotation rx,ry,rz --frames N
        --out <blend> --log <json> [--render-size WxH]
"""

import bpy
import json
import math
import os
import sys
from mathutils import Matrix, Vector, Euler

argv = sys.argv[sys.argv.index("--") + 1:]


def opt(flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def main():
    footage = opt("--footage")
    start = Vector([float(v) for v in opt("--start", "0,0,0").split(",")])
    rot = Euler([math.radians(float(v)) for v in
                 opt("--rotation", "0,0,0").split(",")], 'XYZ')
    frames = int(opt("--frames", "1"))
    out_blend = opt("--out")
    out_log = opt("--log")
    render_size = opt("--render-size")

    scene = bpy.context.scene

    root = bpy.data.objects.new("TrackRoot", None)
    root.empty_display_type = 'ARROWS'
    root.empty_display_size = 1.5
    scene.collection.objects.link(root)
    root.matrix_world = Matrix.LocRotScale(start, rot, Vector((1, 1, 1)))

    cam_data = bpy.data.cameras.new("TrackedCamera")
    focal = opt("--focal-mm")
    cam_data.lens = float(focal) if focal else 35.0
    focus = opt("--focus-distance")
    if focus:
        cam_data.dof.use_dof = True
        cam_data.dof.focus_distance = float(focus)
    cam = bpy.data.objects.new("TrackedCamera", cam_data)
    scene.collection.objects.link(cam)
    cam.parent = root  # camera identity-local to root; root carries the pose
    scene.camera = cam

    scene.frame_start = 1
    scene.frame_end = frames
    if render_size:
        w, h = (int(v) for v in render_size.lower().split("x"))
    else:
        w, h = scene.render.resolution_x, scene.render.resolution_y
    scene.render.resolution_x, scene.render.resolution_y = w, h

    if footage and os.path.exists(footage):
        bpy.ops.clip.open(directory=os.path.dirname(footage),
                          files=[{"name": os.path.basename(footage)}])
        clip = bpy.data.movieclips[-1]
        bg = cam_data.background_images.new()
        bg.source = 'MOVIE_CLIP'
        bg.clip = clip
        bg.alpha = 1.0
        cam_data.show_background_images = True

    os.makedirs(os.path.dirname(out_blend), exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=out_blend)

    os.makedirs(os.path.dirname(out_log), exist_ok=True)
    with open(out_log, "w") as f:
        json.dump({"solve_mode": "static", "average_solve_error": 0.0,
                   "solve_ok": True, "num_tracks_final": 0,
                   "frame_duration": frames,
                   "note": "locked-off camera placed manually (no solve)"}, f,
                  indent=2)
    print(f"[static] placed camera over {frames} frames -> {out_blend}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
