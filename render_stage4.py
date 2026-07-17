"""
Stage 4: Render — apply render settings to the tracked scene and go.
--------------------------------------------------------------------
Run with:
    blender -b <scene_tracked.blend> -P render_stage4.py -- --out <path> [options]

Options (all optional except --out):
    --out path          Output base path. Extension picks the container:
                        .mp4 renders a video, no extension renders a PNG
                        sequence (path is used as the frame prefix).
    --engine E          cycles | eevee | workbench (or exact Blender enum;
                        default: whatever the .blend already uses)
    --samples N         Cycles/Eevee sample count
    --resolution WxH    e.g. 1920x1080 (default: scene setting)
    --percent P         resolution percentage 1-100 (default: scene setting)
    --fps N             override frame rate
    --frames A-B        frame range (default: scene setting)
    --transparent       render with transparent background (for compositing
                        CG over the footage later)
    --camera NAME       camera to render from (default: scene camera)

Example:
    blender -b user_scene_tracked.blend -P render_stage4.py -- ^
        --out renders/shot03.mp4 --engine CYCLES --samples 64 --percent 50
"""

import bpy
import sys
import os


def get_cli_args():
    argv = sys.argv
    if "--" not in argv:
        raise ValueError("Pass options after '--' (see docstring); --out is required")
    argv = argv[argv.index("--") + 1:]

    def opt(flag, default=None):
        return argv[argv.index(flag) + 1] if flag in argv else default

    out = opt("--out")
    if not out:
        raise ValueError("--out <path> is required")
    return {
        "out": out,
        "engine": opt("--engine"),
        "samples": opt("--samples"),
        "resolution": opt("--resolution"),
        "percent": opt("--percent"),
        "fps": opt("--fps"),
        "frames": opt("--frames"),
        "transparent": "--transparent" in argv,
        "camera": opt("--camera"),
    }


def main():
    a = get_cli_args()
    scene = bpy.context.scene
    r = scene.render

    if a["engine"]:
        wanted = a["engine"].upper()
        if "CYCLES" in wanted:
            # cycles is an addon and can be disabled in user prefs
            try:
                import addon_utils
                addon_utils.enable("cycles", default_set=True)
            except Exception:
                pass
        aliases = {"CYCLES": "CYCLES", "EEVEE": "BLENDER_EEVEE",
                   "WORKBENCH": "BLENDER_WORKBENCH"}
        valid = scene.render.bl_rna.properties["engine"].enum_items.keys()
        # NOTE: the enum list is STALE right after enabling the cycles addon
        # — direct assignment works even when 'CYCLES' isn't listed yet, so
        # try the assignment and only then complain.
        match = aliases.get(wanted) or \
            next((e for e in valid if e == wanted), None) or \
            next((e for e in valid if wanted in e), None) or wanted
        try:
            scene.render.engine = match
        except TypeError:
            raise ValueError(f"Unknown engine {a['engine']}; "
                             f"this Blender has: {list(valid)}")
    if scene.render.engine == 'CYCLES':
        # use the GPU when one exists — CPU Cycles is 5-10x slower
        gpu_type = None
        try:
            prefs = bpy.context.preferences.addons["cycles"].preferences
            for dev_type in ("OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"):
                try:
                    prefs.compute_device_type = dev_type
                except TypeError:
                    continue
                prefs.get_devices()
                gpus = [d for d in prefs.devices if d.type != 'CPU']
                if gpus:
                    for d in prefs.devices:
                        d.use = True
                    scene.cycles.device = 'GPU'
                    gpu_type = dev_type
                    print(f"[stage4] Cycles on GPU ({dev_type}: "
                          f"{gpus[0].name})")
                    break
        except Exception as e:
            print(f"[stage4] GPU setup skipped: {e}")
        # Denoising is the hidden render killer: scenes default to
        # OpenImageDenoise on the CPU, which at 4K takes 20-30 SECONDS per
        # frame — far longer than the render itself at draft samples. Move
        # it to the GPU (OptiX on NVIDIA, or GPU OIDN elsewhere).
        c = scene.cycles
        if getattr(c, "use_denoising", False):
            denoise_dev = "CPU"
            if gpu_type in ("OPTIX", "CUDA"):
                try:
                    c.denoiser = 'OPTIX'
                    denoise_dev = "GPU (OptiX)"
                except TypeError:
                    pass
            if denoise_dev == "CPU" and gpu_type:
                try:
                    c.denoising_use_gpu = True
                    denoise_dev = "GPU (OIDN)"
                except (AttributeError, TypeError):
                    pass
            print(f"[stage4] denoising on {denoise_dev}")
        # keep BVH/textures resident between frames — free speedup for
        # animation renders (the scene doesn't rebuild every frame)
        try:
            scene.render.use_persistent_data = True
        except Exception:
            pass
    if a["samples"]:
        n = int(a["samples"])
        if scene.render.engine == 'CYCLES':
            scene.cycles.samples = n
        elif hasattr(scene, "eevee"):
            scene.eevee.taa_render_samples = n
    if a["resolution"]:
        w, h = a["resolution"].lower().split("x")
        r.resolution_x, r.resolution_y = int(w), int(h)
    if a["percent"]:
        r.resolution_percentage = int(a["percent"])
    if a["fps"]:
        r.fps = int(a["fps"])
    if a["frames"]:
        start, end = a["frames"].split("-")
        scene.frame_start, scene.frame_end = int(start), int(end)
    if a["camera"]:
        cam = bpy.data.objects.get(a["camera"])
        if cam is None:
            raise ValueError(f"No object named {a['camera']} in the scene")
        scene.camera = cam
    if scene.camera is None:
        raise RuntimeError("Scene has no camera — run Stage 3 first")
    r.film_transparent = a["transparent"]

    out = os.path.abspath(a["out"])
    os.makedirs(os.path.dirname(out), exist_ok=True)
    ext = os.path.splitext(out)[1].lower()
    if ext == ".mp4":
        # H.264 requires even pixel dimensions; bake the percentage into an
        # even final resolution
        w = round(r.resolution_x * r.resolution_percentage / 100)
        h = round(r.resolution_y * r.resolution_percentage / 100)
        r.resolution_x, r.resolution_y = w - w % 2, h - h % 2
        r.resolution_percentage = 100
        if hasattr(r.image_settings, "media_type"):
            r.image_settings.media_type = 'VIDEO'
        r.image_settings.file_format = 'FFMPEG'
        r.ffmpeg.format = 'MPEG4'
        r.ffmpeg.codec = 'H264'
        if a["transparent"]:
            print("[stage4] NOTE: mp4 cannot store alpha; --transparent needs "
                  "a PNG sequence output to keep the background transparent.")
    else:
        if hasattr(r.image_settings, "media_type"):
            r.image_settings.media_type = 'IMAGE'
        r.image_settings.file_format = 'PNG'
        r.image_settings.color_mode = 'RGBA' if a["transparent"] else 'RGB'
    r.filepath = out if ext else out + "_"

    print(f"[stage4] Rendering frames {scene.frame_start}-{scene.frame_end} "
          f"with {scene.render.engine} at {r.resolution_x}x{r.resolution_y} "
          f"({r.resolution_percentage}%) -> {r.filepath}")

    if bpy.app.background:
        # headless: block until done, exactly as before
        bpy.ops.render.render(animation=True)
        print(f"[stage4] Done: {r.filepath}")
        return a

    # ---- windowed: LIVE render view -------------------------------------
    # A plain render() call here would run on the UI thread and freeze the
    # window into a white rectangle until the whole animation finishes —
    # you'd see nothing. INVOKE_DEFAULT runs the render as an interactive
    # job (same as pressing Ctrl+F12): Blender opens its render window and
    # draws every frame as it completes, with the progress bar live.
    keep_open = "--keep-open" in sys.argv
    bpy.context.preferences.view.render_display_type = 'WINDOW'

    def _quit(code):
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)  # no save-changes prompt; output is already on disk

    def _on_complete(*_args):
        print(f"[stage4] Done: {r.filepath}")
        sys.stdout.flush()
        if not keep_open:
            # give the movie writer a moment to finalize the container
            bpy.app.timers.register(lambda: _quit(0), first_interval=1.5)

    def _on_cancel(*_args):
        print("[stage4] render cancelled in Blender")
        if not keep_open:
            bpy.app.timers.register(lambda: _quit(1), first_interval=0.5)

    bpy.app.handlers.render_complete.append(_on_complete)
    bpy.app.handlers.render_cancel.append(_on_cancel)

    def _start():
        try:
            bpy.ops.render.render('INVOKE_DEFAULT', animation=True)
        except Exception:
            import traceback
            traceback.print_exc()
            _quit(1)
        return None  # one-shot timer

    # wait until the UI has finished building before invoking the render job
    bpy.app.timers.register(_start, first_interval=0.5)
    return a


if __name__ == "__main__":
    try:
        a = main()
    except Exception:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        if bpy.app.background or "--keep-open" not in sys.argv:
            os._exit(1)   # hard exit: no save-changes dialog, no lingering window
        sys.exit(1)
    sys.stdout.flush()
    # Windowed mode returns immediately after scheduling the render job —
    # Blender's event loop takes over, the render window shows live frames,
    # and the render_complete handler exits the process when finished.
