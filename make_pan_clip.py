"""Render a tripod-pan test clip with a KNOWN camera rotation.

flow_solve models a rotation-only camera, so a pure pan is its best case. If it
under-recovers the angle here, that is a real defect and not a limitation of
the shot. Ground truth (per-frame quaternion + the exact pan angle) is written
next to the clip.

Run: blender -b -P make_pan_clip.py -- <out.mp4> <truth.json> [pan_degrees] [lens_mm]
"""
import bpy
import json
import math
import random
import sys

argv = sys.argv[sys.argv.index("--") + 1:]
out_path, truth_path = argv[0], argv[1]
PAN_DEG = float(argv[2]) if len(argv) > 2 else 15.0
LENS_MM = float(argv[3]) if len(argv) > 3 else 35.0
SENSOR_MM = 36.0          # flow_solve hardcodes SENSOR_W_MM = 36.0
N_FRAMES = 60

random.seed(42)
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


def confetti(name, size=128):
    img = bpy.data.images.new(name, width=size, height=size)
    px = []
    for _ in range(size * size):
        px.extend((random.random(), random.random(), random.random(), 1.0))
    img.pixels = px
    return img


def textured(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = confetti(name + "_img")
    tex.interpolation = 'Closest'
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def plane(name, loc, rot, size, mat):
    bpy.ops.mesh.primitive_plane_add(size=size, location=loc, rotation=rot)
    o = bpy.context.active_object
    o.name = name
    o.data.materials.append(mat)
    return o


# A room that stays in frame across the whole pan: far wall plus both sides.
plane("floor", (0, 8, -3), (0, 0, 0), 60, textured("m_floor"))
plane("back", (0, 18, 4), (math.radians(90), 0, 0), 44, textured("m_back"))
plane("left", (-14, 6, 4), (0, math.radians(90), 0), 40, textured("m_left"))
plane("right", (14, 6, 4), (0, math.radians(90), 0), 40, textured("m_right"))

# Tripod camera: FIXED position, rotates in place. Yaw about world Z.
bpy.ops.object.camera_add(location=(0, -6, 1))
cam = bpy.context.active_object
cam.data.lens = LENS_MM
cam.data.sensor_width = SENSOR_MM
cam.rotation_mode = 'XYZ'
scene.camera = cam

for f in range(1, N_FRAMES + 1):
    t = (f - 1) / (N_FRAMES - 1)
    yaw = math.radians(PAN_DEG) * t
    # X=90deg points the camera along +Y; Z carries the pan
    cam.rotation_euler = (math.radians(90), 0.0, yaw)
    cam.keyframe_insert(data_path="rotation_euler", frame=f)

# linear pan: no easing, so ground truth is exactly PAN_DEG * t
for fc in cam.animation_data.action.fcurves if hasattr(
        cam.animation_data.action, "fcurves") else []:
    for kp in fc.keyframe_points:
        kp.interpolation = 'LINEAR'

bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
bpy.context.active_object.data.energy = 3.0

scene.render.engine = 'BLENDER_WORKBENCH'
scene.display.shading.color_type = 'TEXTURE'
scene.display.shading.light = 'FLAT'
scene.render.resolution_x = 960
scene.render.resolution_y = 540
scene.render.fps = 24
scene.frame_start, scene.frame_end = 1, N_FRAMES
if hasattr(scene.render.image_settings, "media_type"):
    scene.render.image_settings.media_type = 'VIDEO'
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.ffmpeg.constant_rate_factor = 'PERC_LOSSLESS'
scene.render.filepath = out_path

# ground truth AFTER keys are set, sampled the same way the pipeline would
truth = {"pan_deg_total": PAN_DEG, "lens_mm": LENS_MM,
         "sensor_mm": SENSOR_MM, "num_frames": N_FRAMES,
         "res": [scene.render.resolution_x, scene.render.resolution_y],
         "quats": [], "yaw_deg": []}
for f in range(1, N_FRAMES + 1):
    scene.frame_set(f)
    q = cam.matrix_world.to_quaternion()
    truth["quats"].append([q.w, q.x, q.y, q.z])
    truth["yaw_deg"].append(math.degrees(cam.rotation_euler.z))
with open(truth_path, "w") as fh:
    json.dump(truth, fh)

bpy.ops.render.render(animation=True)
print(f"[pan] {N_FRAMES} frames, pan {PAN_DEG} deg, lens {LENS_MM}mm -> {out_path}")
print(f"[pan] truth -> {truth_path}")
