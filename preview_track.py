"""
Track preview render — eyeball a solve before committing a real render.
-----------------------------------------------------------------------
3D solves (perspective/tripod):
    blender -b <tracked.blend> -P preview_track.py -- --out preview.mp4
    Renders the footage with the reconstructed 3D track points as glowing
    dots. If the solve is good, every dot stays glued to its feature.

Flow solves:
    blender -b -P preview_track.py -- --flow flow.json --footage shot.mp4 --out preview.mp4
    Renders the footage with a wireframe sky-grid driven by the recovered
    rotation. If the match is good, the grid rotates in lockstep with the
    background.

Optional: --width 960 (default) proxy render width.
"""

import bpy
import json
import os
import sys


def get_args():
    argv = sys.argv[sys.argv.index("--") + 1:]

    def opt(flag, default=None):
        return argv[argv.index(flag) + 1] if flag in argv else default

    return {
        "out": opt("--out"),
        "flow": opt("--flow"),
        "footage": opt("--footage"),
        "width": int(opt("--width", "960")),
    }


def emissive(name, color):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    nt.nodes.clear()
    em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*color, 1.0)
    em.inputs["Strength"].default_value = 3.0
    outn = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs["Emission"], outn.inputs["Surface"])
    return mat


def setup_render(scene, clip_w, clip_h, fps, out, width):
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.film_transparent = True
    h = round(clip_h * width / clip_w)
    scene.render.resolution_x = width - width % 2
    scene.render.resolution_y = h - h % 2
    scene.render.resolution_percentage = 100
    scene.render.fps = round(fps) if fps else 24
    if hasattr(scene.render.image_settings, "media_type"):
        scene.render.image_settings.media_type = 'VIDEO'
    scene.render.image_settings.file_format = 'FFMPEG'
    scene.render.ffmpeg.format = 'MPEG4'
    scene.render.ffmpeg.codec = 'H264'
    scene.render.filepath = out


def composite_over_clip(scene, clip):
    """Blender 5.0 compositor: a node group on scene.compositing_node_group
    with a NodeGroupOutput (CompositorNodeComposite no longer exists)."""
    nt = bpy.data.node_groups.new("TrackPreviewComp", "CompositorNodeTree")
    nt.interface.new_socket("Image", in_out='OUTPUT',
                            socket_type='NodeSocketColor')
    scene.compositing_node_group = nt
    rl = nt.nodes.new("CompositorNodeRLayers")
    rl.scene = scene
    mc = nt.nodes.new("CompositorNodeMovieClip")
    mc.clip = clip
    sc = nt.nodes.new("CompositorNodeScale")
    # explicit relative factors — 'Render Size' letterboxes unpredictably
    sc.inputs["Type"].default_value = 'Relative'
    sc.inputs["X"].default_value = scene.render.resolution_x / clip.size[0]
    sc.inputs["Y"].default_value = scene.render.resolution_y / clip.size[1]
    ao = nt.nodes.new("CompositorNodeAlphaOver")
    out = nt.nodes.new("NodeGroupOutput")
    nt.links.new(mc.outputs["Image"], sc.inputs["Image"])
    nt.links.new(sc.outputs["Image"], ao.inputs["Background"])
    nt.links.new(rl.outputs["Image"], ao.inputs["Foreground"])
    nt.links.new(ao.outputs["Image"], out.inputs["Image"])


def preview_3d(out, width):
    """Runs inside the tracked .blend: bundles + solver camera over footage."""
    scene = bpy.context.scene
    clip = bpy.data.movieclips[-1]

    # the tracked .blend inherits template leftovers (default cube etc.)
    for o in list(scene.objects):
        bpy.data.objects.remove(o, do_unlink=True)

    cam_data = bpy.data.cameras.new("PreviewCam")
    cam_data.sensor_width = clip.tracking.camera.sensor_width
    cam_data.lens = clip.tracking.camera.focal_length
    cam = bpy.data.objects.new("PreviewCam", cam_data)
    scene.collection.objects.link(cam)
    con = cam.constraints.new(type='CAMERA_SOLVER')
    con.use_active_clip = False
    con.clip = clip
    scene.camera = cam

    bundles = [t.bundle.copy() for t in clip.tracking.tracks if t.has_bundle]
    if not bundles:
        raise RuntimeError("no reconstructed 3D points in this solve")

    scene.frame_set(clip.frame_start)
    deps = bpy.context.evaluated_depsgraph_get()
    deps.update()
    cam_pos = cam.evaluated_get(deps).matrix_world.to_translation()

    # constant APPARENT size: scale each dot by its own camera distance
    # (reconstructions often contain far-flung outlier points)
    mat = emissive("m_bundle", (0.1, 1.0, 0.75))
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=0.006)
    proto = bpy.context.active_object
    proto.data.materials.append(mat)
    for b in bundles:
        d = max((b - cam_pos).length, 1e-3)
        o = proto.copy()  # linked duplicate — shares mesh
        o.location = b
        o.scale = (d, d, d)
        scene.collection.objects.link(o)
    d0 = max((bundles[0] - cam_pos).length, 1e-3)
    proto.location = bundles[0]
    proto.scale = (d0, d0, d0)

    scene.frame_start = clip.frame_start
    scene.frame_end = clip.frame_start + clip.frame_duration - 1
    setup_render(scene, clip.size[0], clip.size[1], clip.fps, out, width)
    composite_over_clip(scene, clip)
    bpy.ops.render.render(animation=True)
    print(f"[preview] 3D preview ({len(bundles)} points): {out}")


def preview_flow(flow_path, footage, out, width):
    """Fresh scene: rotation-driven wireframe sky-grid over footage."""
    from mathutils import Quaternion
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    with open(flow_path) as f:
        flow = json.load(f)

    bpy.ops.clip.open(directory=os.path.dirname(footage),
                      files=[{"name": os.path.basename(footage)}])
    clip = bpy.data.movieclips[-1]

    cam_data = bpy.data.cameras.new("PreviewCam")
    cam_data.sensor_width = flow.get("sensor_width_mm", 36.0)
    cam_data.lens = flow.get("focal_mm", 35.0)
    cam = bpy.data.objects.new("PreviewCam", cam_data)
    scene.collection.objects.link(cam)
    cam.rotation_mode = 'QUATERNION'
    scene.camera = cam
    for i, q in enumerate(flow["quaternions_wxyz"], start=1):
        cam.rotation_quaternion = Quaternion(q)
        cam.keyframe_insert(data_path="rotation_quaternion", frame=i)

    bpy.ops.mesh.primitive_uv_sphere_add(radius=60, segments=28, ring_count=14)
    sphere = bpy.context.active_object
    sphere.rotation_euler = (1.5708, 0, 0)  # poles out of the view center
    mod = sphere.modifiers.new("wire", type='WIREFRAME')
    mod.thickness = 0.12
    sphere.data.materials.append(emissive("m_grid", (0.1, 1.0, 0.75)))

    scene.frame_start = 1
    scene.frame_end = flow["num_frames"]
    setup_render(scene, flow["size"][0], flow["size"][1],
                 flow.get("fps", 24), out, width)
    composite_over_clip(scene, clip)
    bpy.ops.render.render(animation=True)
    print(f"[preview] flow preview: {out}")


def main():
    a = get_args()
    if not a["out"]:
        raise ValueError("--out is required")
    if a["flow"]:
        preview_flow(a["flow"], a["footage"], a["out"], a["width"])
    else:
        preview_3d(a["out"], a["width"])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.exit(1)
