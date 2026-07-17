"""
Export the tracked camera for other software (After Effects, Nuke, C4D…).
------------------------------------------------------------------------
    blender <scene_tracked.blend> -P export_camera.py -- --out <path> --format fbx|abc

Exports the TrackedCamera (with its TrackRoot parent so the world placement
is preserved) and its animation. FBX and Alembic both carry the animated
camera into standard 3D/compositing pipelines.
"""

import bpy
import os
import sys

argv = sys.argv[sys.argv.index("--") + 1:]


def opt(flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def main():
    out = opt("--out")
    fmt = (opt("--format", "fbx")).lower()
    scene = bpy.context.scene

    cam = bpy.data.objects.get("TrackedCamera") or scene.camera
    root = bpy.data.objects.get("TrackRoot")
    if cam is None:
        raise RuntimeError("no TrackedCamera in this scene")

    for o in scene.objects:
        o.select_set(False)
    cam.select_set(True)
    if root is not None:
        root.select_set(True)
    bpy.context.view_layer.objects.active = cam

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    if fmt == "abc":
        bpy.ops.wm.alembic_export(
            filepath=out, selected=True, flatten=False,
            start=scene.frame_start, end=scene.frame_end)
    else:  # fbx
        bpy.ops.export_scene.fbx(
            filepath=out, use_selection=True,
            object_types={'CAMERA', 'EMPTY'}, bake_anim=True,
            bake_anim_use_all_bones=False, add_leaf_bones=False)
    print(f"[export] camera -> {out} ({fmt})")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
