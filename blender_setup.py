"""
Blender GUI camera-positioning helper for Nimbus Tracker.
=========================================================
Launched (NOT headless) as:
    blender <scene.blend> -P blender_setup.py -- --frame-img <png> --out <json>

Sets up a camera named 'NimbusStartCam' with the shot's first frame as a
background image, drops the viewport into camera view with "lock camera to
view" on (so navigating moves the camera), and adds an N-panel button
"Choose Starting Position". Clicking it writes the camera pose to <json> and
quits Blender — Nimbus reads that pose and carries on.
"""

import bpy
import json
import math
import os
import sys

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []


def opt(flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


FRAME_IMG = opt("--frame-img")
OUT_JSON = opt("--out")


def setup():
    scene = bpy.context.scene

    # dedicated camera the user positions
    cam = bpy.data.objects.get("NimbusStartCam")
    if cam is None:
        cam_data = bpy.data.cameras.new("NimbusStartCam")
        cam = bpy.data.objects.new("NimbusStartCam", cam_data)
        scene.collection.objects.link(cam)
        cam.location = (0, -8, 2)
        cam.rotation_euler = (math.radians(80), 0, 0)
    scene.camera = cam
    cam.data.lens = 35.0

    # first frame as a semi-transparent camera background, over the CG
    if FRAME_IMG and os.path.exists(FRAME_IMG):
        img = bpy.data.images.load(FRAME_IMG)
        cam.data.show_background_images = True
        bg = cam.data.background_images.new()
        bg.image = img
        bg.alpha = 0.55
        bg.display_depth = 'FRONT'
        # match render aspect to the plate so the camera frame lines up
        w, h = img.size
        if w and h:
            scene.render.resolution_x = w
            scene.render.resolution_y = h

    # drop every 3D viewport into camera view with lock-camera-to-view on
    for win in bpy.context.window_manager.windows:
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                sp = area.spaces.active
                sp.region_3d.view_perspective = 'CAMERA'
                sp.lock_camera = True
                sp.show_gizmo = True


def read_render_settings(scene):
    """Snapshot the render settings the user configured in this Blender session
    so Nimbus uses them instead of asking the user to retype anything."""
    r = scene.render
    eng = r.engine  # e.g. CYCLES, BLENDER_EEVEE / BLENDER_EEVEE_NEXT, BLENDER_WORKBENCH
    if eng == "CYCLES":
        name, samples = "cycles", getattr(scene.cycles, "samples", 128)
    elif "WORKBENCH" in eng:
        name, samples = "workbench", 1
    else:  # any Eevee variant
        name = "eevee"
        samples = getattr(getattr(scene, "eevee", None),
                          "taa_render_samples", 64)
    return {
        "engine": name,
        "samples": int(samples),
        "percent": int(r.resolution_percentage),
        "transparent": bool(r.film_transparent),
        "res_x": int(r.resolution_x),
        "res_y": int(r.resolution_y),
    }


class NIMBUS_OT_choose_start(bpy.types.Operator):
    bl_idname = "nimbus.choose_start"
    bl_label = "Choose Starting Position"
    bl_description = ("Save this camera position and return to Nimbus Tracker")

    def execute(self, context):
        scene = context.scene
        cam = bpy.data.objects.get("NimbusStartCam") or scene.camera
        m = cam.matrix_world
        loc = list(m.to_translation())
        eul = m.to_euler('XYZ')
        data = {"loc": loc,
                "rot_deg": [math.degrees(a) for a in eul],
                "focal_mm": cam.data.lens,
                "sensor_mm": cam.data.sensor_width,
                "render": read_render_settings(scene)}
        with open(OUT_JSON, "w") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        self.report({'INFO'}, "Starting position saved — returning to Nimbus")
        # Close Blender immediately with no save dialog. The user's .blend on
        # disk is never modified (our camera lives only in memory; the pose is
        # written to a separate JSON), so a hard exit is safe and clean.
        os._exit(0)


class NIMBUS_PT_panel(bpy.types.Panel):
    bl_label = "Nimbus Tracker"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Nimbus"

    def draw(self, context):
        col = self.layout.column()
        col.label(text="1. Frame the shot (navigate —")
        col.label(text="   the camera follows the view).")
        col.label(text="2. Set your render settings in")
        col.label(text="   the Render + Output tabs")
        col.label(text="   (engine, samples, resolution).")
        col.label(text="3. Then click:")
        col.scale_y = 1.6
        col.operator("nimbus.choose_start", icon='CAMERA_DATA')
        self.layout.label(text="Camera + render settings are")
        self.layout.label(text="saved and sent back to Nimbus.")


def register():
    for cls in (NIMBUS_OT_choose_start, NIMBUS_PT_panel):
        try:
            bpy.utils.register_class(cls)
        except Exception:
            pass


if __name__ == "__main__":
    register()
    setup()
