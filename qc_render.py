"""QC overlay: SEE whether a solve is good instead of trusting a number.

Run with the SYSTEM python:
    python qc_render.py <shot_video> <solve.json> <out_dir> [--stride 1]

Draws, on every frame of the shot:
  + green crosses   the tracked 2D markers (what the tracker measured)
  o  dots           the solved 3D bundles reprojected through the solved
                    camera (what the reconstruction predicts) — green when
                    they land on their marker, shading to red as they miss
plus a header with the solve's own numbers. If the dots ride their crosses
through the whole shot, the camera is locked to the footage; if they slide,
the solve drifts — visible in one glance, no Blender needed.

This exists because this pipeline's whole failure history is numbers that
looked fine describing cameras that weren't: reprojection error can't be
eyeballed, but dots-on-crosses can. Output: qc.mp4 + qc_sheet.png (first /
middle / last frame contact sheet).

Tripod solves have no bundles; the overlay then shows the tracked markers
only, labelled as rotation-only.
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("video", help="the shot file the solve was tracked on")
    p.add_argument("solve_json", help="output of dump_solve.py")
    p.add_argument("out_dir")
    p.add_argument("--stride", type=int, default=1,
                   help="draw every Nth frame (default 1 = all)")
    return p.parse_args()


def reproject(bundle, cam_mat, d, W, H):
    """3D world point -> pixel through the solved camera, WITH its distortion.

    Markers live in distorted image space and the solver scored itself there,
    so an undistorted pinhole reprojection misses progressively toward the
    frame edges — the first QC render showed exactly that radial red ring and
    it read as drift. Apply the solve's own model (polynomial k1..k3 or
    division) so dots land where the solve actually put them.
    """
    M = np.array(cam_mat)
    R, t = M[:3, :3], M[:3, 3]
    p = R.T @ (np.array(bundle) - t)       # world -> camera space
    if p[2] >= -1e-6:                       # Blender camera looks down -Z
        return None
    x, y = p[0] / -p[2], p[1] / -p[2]       # camera-plane coords
    r2 = x * x + y * y
    model = d.get("distortion_model", "POLYNOMIAL")
    if model == "DIVISION":
        k1, k2 = d.get("division_k1", 0.0), d.get("division_k2", 0.0)
        s = 1.0 + k1 * r2 + k2 * r2 * r2
        if abs(s) > 1e-9:
            x, y = x / s, y / s
    else:                                   # POLYNOMIAL (libmv default)
        k1, k2, k3 = d.get("k1", 0.0), d.get("k2", 0.0), d.get("k3", 0.0)
        s = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        x, y = x * s, y * s
    f_px = d["lens_mm"] / d["sensor_mm"] * W
    pp = d.get("principal_px") or [W / 2.0, H / 2.0]
    u = pp[0] + f_px * x
    v = (H - pp[1]) - f_px * y              # principal is bottom-left origin
    return (u, v)


def main():
    a = parse_args()
    d = json.load(open(a.solve_json))
    W, H = d["clip_size"]
    fs = d["frame_start"]
    cams = {int(k): v for k, v in d["cameras"].items()}
    solved = [t for t in d["tracks"] if t["bundle"] is not None]
    mode = ("perspective" if solved and cams else
            "tripod / rotation-only" if cams else "no reconstruction")

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f"cannot open {a.video}")
    vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sx, sy = vw / W, vh / H                 # solve coords -> video pixels

    os.makedirs(a.out_dir, exist_ok=True)
    ow = 1280
    oh = round(vh * ow / vw)
    oh -= oh % 2
    writer = cv2.VideoWriter(os.path.join(a.out_dir, "qc.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (ow, oh))
    sheet, n_frames = [], int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keyframes = {1, max(1, n_frames // 2), n_frames}

    f = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        f += 1
        if (f - 1) % a.stride:
            continue
        frame_abs = fs + f - 1
        cam_mat = cams.get(frame_abs)
        for t in d["tracks"]:
            mk = t["markers"].get(str(frame_abs))
            if mk is None:
                continue
            mx, my = mk[0] * vw, (1.0 - mk[1]) * vh
            cv2.drawMarker(img, (int(mx), int(my)), (80, 255, 80),
                           cv2.MARKER_CROSS, 12, 2)
            if t["bundle"] is not None and cam_mat is not None:
                uv = reproject(t["bundle"], cam_mat, d, W, H)
                if uv is None:
                    continue
                px, py = uv[0] * sx, uv[1] * sy
                miss = float(np.hypot(px - mx, py - my))
                # green at 0px miss -> red at >= 8px (scaled to video res)
                k = min(1.0, miss / (8.0 * sx))
                col = (int(60 * (1 - k)), int(255 * (1 - k)), int(255 * k))
                cv2.circle(img, (int(px), int(py)), 7, col, 2)
        hdr = (f"{mode}   solve err "
               f"{d['average_error']:.2f}px" if d["average_error"] is not None
               else f"{mode}")
        hdr += f"   {len(solved)} bundles / {len(d['tracks'])} tracks   f{f}"
        cv2.rectangle(img, (0, 0), (vw, 44), (0, 0, 0), -1)
        cv2.putText(img, hdr, (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2)
        cv2.putText(img, "crosses = tracked   dots = solve reprojected "
                         "(green on target, red = drift)",
                    (12, vh - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (200, 200, 200), 2)
        small = cv2.resize(img, (ow, oh))
        writer.write(small)
        if f in keyframes:
            sheet.append(small.copy())
    cap.release()
    writer.release()
    if sheet:
        cv2.imwrite(os.path.join(a.out_dir, "qc_sheet.png"), np.vstack(sheet))
    print(f"[qc] {f} frames -> {os.path.join(a.out_dir, 'qc.mp4')} "
          f"(+ qc_sheet.png)")


if __name__ == "__main__":
    main()
