"""
Render a synthetic test clip for the tracking pipeline.

Scene design matters for tracking: features must sit ON surfaces, not on
depth edges (corners of foreground objects against background drift due to
parallax inside the pattern). So we build a room from flat planes textured
with random color blocks — confetti-like, high-contrast, and every feature
is surface-attached. Camera dollies laterally for solve parallax.

Run: blender -b -P make_test_clip.py -- <output.mp4>
"""
import bpy
import random
import sys

out_path = sys.argv[sys.argv.index("--") + 1]
random.seed(42)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene


def make_confetti_image(name, size=128):
    """Image of random color blocks — ideal tracking features."""
    img = bpy.data.images.new(name, width=size, height=size)
    px = []
    for _ in range(size * size):
        px.extend((random.random(), random.random(), random.random(), 1.0))
    img.pixels = px
    return img


def make_textured_material(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
    tex.image = make_confetti_image(name + "_img")
    tex.interpolation = 'Closest'  # keep blocks crisp
    mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def add_plane(name, location, rotation, size, mat):
    bpy.ops.mesh.primitive_plane_add(size=size, location=location, rotation=rotation)
    obj = bpy.context.active_object
    obj.name = name
    obj.data.materials.append(mat)
    return obj


# Room: floor, back wall, side wall — all textured
add_plane("floor", (0, 8, -2), (0, 0, 0), 30, make_textured_material("m_floor"))
add_plane("back_wall", (0, 16, 3), (1.5708, 0, 0), 24, make_textured_material("m_wall"))
add_plane("side_wall", (-10, 8, 3), (0, 1.5708, 0), 24, make_textured_material("m_side"))

# A few textured boxes for depth variety (big faces, surface features)
box_mat = make_textured_material("m_box")
for i, (x, y, s) in enumerate([(-3, 9, 2.0), (2, 12, 3.0), (5, 8, 1.5)]):
    bpy.ops.mesh.primitive_cube_add(size=s, location=(x, y, -2 + s / 2))
    obj = bpy.context.active_object
    obj.data.materials.append(box_mat)

# Camera dollying sideways, aimed into the room
bpy.ops.object.camera_add(location=(-4, -8, 1))
cam = bpy.context.active_object
scene.camera = cam

target = bpy.data.objects.new("target", None)
scene.collection.objects.link(target)
target.location = (0, 10, 0)
con = cam.constraints.new(type='TRACK_TO')
con.target = target
con.track_axis = 'TRACK_NEGATIVE_Z'
con.up_axis = 'UP_Y'

scene.frame_start = 1
scene.frame_end = 60
cam.location = (-4, -8, 1)
cam.keyframe_insert(data_path="location", frame=1)
cam.location = (4, -7, 1.5)
cam.keyframe_insert(data_path="location", frame=60)

bpy.ops.object.light_add(type='SUN', location=(5, -5, 10))
bpy.context.active_object.data.energy = 3.0

# Render settings: Workbench in TEXTURE shading mode, MP4 out
scene.render.engine = 'BLENDER_WORKBENCH'
scene.display.shading.color_type = 'TEXTURE'
scene.display.shading.light = 'FLAT'
scene.render.resolution_x = 960
scene.render.resolution_y = 540
scene.render.fps = 24
# Blender 5.0: video output is selected via media_type, then FFMPEG format
if hasattr(scene.render.image_settings, "media_type"):
    scene.render.image_settings.media_type = 'VIDEO'
scene.render.image_settings.file_format = 'FFMPEG'
scene.render.ffmpeg.format = 'MPEG4'
scene.render.ffmpeg.codec = 'H264'
scene.render.ffmpeg.constant_rate_factor = 'PERC_LOSSLESS'
scene.render.filepath = out_path

bpy.ops.render.render(animation=True)
print(f"CLIP: rendered {scene.frame_end} frames to {out_path}")
