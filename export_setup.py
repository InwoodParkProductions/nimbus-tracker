"""
Export shot-setup viewport data from a tracked scene .blend:
  setup.glb     — scene geometry proxy (glTF, no camera/rig)
  campath.json  — root-local camera path, lens, and current TrackRoot pose

Run: blender -b <scene_tracked.blend> -P export_setup.py -- --out <dir>
"""

import bpy
import json
import os
import sys


def main():
    argv = sys.argv[sys.argv.index("--") + 1:]
    out_dir = argv[argv.index("--out") + 1]
    os.makedirs(out_dir, exist_ok=True)

    scene = bpy.context.scene
    root = bpy.data.objects.get("TrackRoot")
    cam = scene.camera if root is not None else None

    # camera path in ROOT-LOCAL space (the animated local transforms).
    # Pre-track scenes have no rig yet: empty path, identity root — the
    # viewport then shows a single placeable start frustum.
    path = []
    if root is not None and cam is not None:
        for f in range(scene.frame_start, scene.frame_end + 1):
            scene.frame_set(f)
            path.append({"loc": list(cam.location),
                         "quat": list(cam.rotation_quaternion)})
        l, q, s = root.matrix_world.decompose()
        rootdata = {"loc": list(l), "quat": list(q), "scale": s[0]}
        focal, sensor = cam.data.lens, cam.data.sensor_width
    else:
        rootdata = {"loc": [0, 0, 0], "quat": [1, 0, 0, 0], "scale": 1.0}
        focal, sensor = 35.0, 36.0

    data = {
        "frame_start": scene.frame_start,
        "frame_end": scene.frame_end,
        "focal_mm": focal,
        "sensor_mm": sensor,
        "res_x": scene.render.resolution_x,
        "res_y": scene.render.resolution_y,
        "root": rootdata,
        "path": path,
    }
    # geometry proxy: the WHOLE scene, regardless of hide state or collection
    # nesting. Selection-based export silently drops hidden objects and
    # objects in excluded/nested collections — which real scenes are full of.
    def unhide_layer(layer_coll):
        layer_coll.exclude = False
        layer_coll.hide_viewport = False
        if layer_coll.collection:
            layer_coll.collection.hide_viewport = False
        for ch in layer_coll.children:
            unhide_layer(ch)

    view_layer = bpy.context.view_layer
    unhide_layer(view_layer.layer_collection)
    mesh_count = 0
    for o in scene.objects:
        o.hide_viewport = False
        try:
            o.hide_set(False)
        except RuntimeError:
            pass
        if o.type in {'MESH', 'CURVE', 'SURFACE', 'META', 'FONT'} \
                and o not in (cam, root):
            mesh_count += 1

    # Decimate heavy meshes for a light viewport proxy — shot setup needs
    # rough shapes, not millions of polys. export_apply=True bakes these.
    face_budget = 8000
    for o in scene.objects:
        if o.type != 'MESH' or o in (cam, root):
            continue
        n = len(o.data.polygons)
        if n > face_budget:
            mod = o.modifiers.new("vp_decimate", 'DECIMATE')
            mod.ratio = max(0.02, face_budget / n)

    data["mesh_count"] = mesh_count
    with open(os.path.join(out_dir, "campath.json"), "w") as fp:
        json.dump(data, fp)

    # export EVERYTHING (no selection filter); camera + empty come along as
    # harmless non-geometry nodes. use_visible left default (export all).
    # CRITICAL: materials/textures NONE — a textured scene embeds every image
    # at full res (one real scene = 360 MB, unloadable in a browser). Shot
    # setup only needs geometry shapes/positions; solid grey is ideal, like
    # a matchmove solid view. Also skip lights/cameras data to stay lean.
    bpy.ops.export_scene.gltf(
        filepath=os.path.join(out_dir, "setup.glb"),
        use_selection=False, export_apply=True, export_animations=False,
        export_materials='NONE', export_lights=False,
        export_cameras=False, export_normals=False,
        export_texcoords=False)
    sz = os.path.getsize(os.path.join(out_dir, "setup.glb"))
    print(f"[setup-export] wrote campath.json + setup.glb "
          f"({mesh_count} mesh objects, {sz // 1024} KB)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
