"""Dump a tracked .blend's solve to JSON (for the QC overlay renderer).

Run: blender -b <tracked.blend> -P dump_solve.py -- <out.json>

Writes everything qc_render.py needs to draw the solve over the footage:
per-frame reconstructed camera matrices, lens/sensor, clip size, the solved
3D bundles, and every track's 2D markers. Runs inside Blender because that is
where the reconstruction lives; the drawing happens outside (Blender has no
cv2, and this keeps the QC renderer testable without Blender).
"""
import json
import sys

import bpy

out_path = sys.argv[sys.argv.index("--") + 1]

clip = bpy.data.movieclips[0] if bpy.data.movieclips else None
if clip is None:
    sys.exit("no movie clip in this .blend")

tr = clip.tracking
rec = tr.reconstruction
cam = tr.camera

data = {
    "clip_size": [clip.size[0], clip.size[1]],
    "frame_start": clip.frame_start,
    "frame_duration": clip.frame_duration,
    "lens_mm": cam.focal_length,
    "sensor_mm": cam.sensor_width,
    # The solve refines radial distortion, and marker positions live in
    # DISTORTED image space — a pinhole reprojection without these misses
    # progressively toward the frame edges and looks like drift.
    "distortion_model": getattr(cam, "distortion_model", "POLYNOMIAL"),
    "k1": getattr(cam, "k1", 0.0),
    "k2": getattr(cam, "k2", 0.0),
    "k3": getattr(cam, "k3", 0.0),
    "division_k1": getattr(cam, "division_k1", 0.0),
    "division_k2": getattr(cam, "division_k2", 0.0),
    "principal_px": (list(cam.principal_point_pixels)
                     if hasattr(cam, "principal_point_pixels") else None),
    "reconstruction_valid": rec.is_valid,
    "average_error": rec.average_error if rec.is_valid else None,
    "cameras": {},   # frame -> 4x4 world matrix of the solved camera
    "tracks": [],
}
if rec.is_valid:
    for c in rec.cameras:
        data["cameras"][str(c.frame)] = [list(row) for row in c.matrix]

for t in tr.tracks:
    entry = {"name": t.name,
             "bundle": list(t.bundle) if t.has_bundle else None,
             "error": t.average_error,
             "markers": {}}
    for m in t.markers:
        if not m.mute:
            entry["markers"][str(m.frame)] = [m.co[0], m.co[1]]
    data["tracks"].append(entry)

with open(out_path, "w") as f:
    json.dump(data, f)
print(f"[dump] {len(data['tracks'])} tracks, {len(data['cameras'])} solved "
      f"cameras -> {out_path}")
