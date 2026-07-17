"""Stage 5: composite — CG background behind the actors, the actual deliverable.

Run with the SYSTEM python:
    python comp_stage5.py <footage> <cg_dir> <masks_dir> <out_dir>
        [--frames A-B] [--grow 2] [--feather 5] [--preview]

Everything upstream produces ingredients: stage 1 makes person mattes, stage 4
renders the CG plates. This puts them together:

    comp = plate * alpha + CG * (1 - alpha)

where alpha is the (softened) person matte. The person stays from the original
plate; the CG replaces everything else. Output is a PNG sequence at the CG's
resolution plus an optional small preview .mp4 for review.

Masks come in binary from SAM2, which is right for track exclusion and crunchy
for compositing — so the matte is grown slightly (protect hair/blur that the
segmenter missed) and edge-feathered. Both are tunable; for hair-critical work
a dedicated matting model can replace this (see --alpha-dir).
"""

import argparse
import glob
import json
import os
import re
import sys

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("footage", help="original clip (the shot's frames are read "
                                   "from it at full quality)")
    p.add_argument("cg_dir", help="directory of rendered CG frames "
                                  "(*_0001.png ... shot-relative, 1-based)")
    p.add_argument("masks_dir", help="mask_NNNNNN.png dir (white = person)")
    p.add_argument("out_dir")
    p.add_argument("--frames", help="footage frame range A-B for this shot "
                                    "(1-based inclusive). Default: start at "
                                    "frame 1 for as many frames as there is CG.")
    p.add_argument("--grow", type=int, default=2,
                   help="grow the person matte by N px before feathering "
                        "(default 2) — protects hair and motion blur the "
                        "segmenter clipped")
    p.add_argument("--feather", type=int, default=5,
                   help="edge feather radius in px at CG resolution (default 5)")
    p.add_argument("--alpha-dir",
                   help="optional dir of refined alpha mattes "
                        "(alpha_NNNNNN.png, white=person, gray=soft) — used "
                        "instead of grown/feathered binary masks when present")
    p.add_argument("--solve-json",
                   help="solve dump (dump_solve.py) — when given, the CG is "
                        "warped through the solve's lens-distortion model so "
                        "it sits in the same distorted image space as the "
                        "plate. Blender renders CG through an ideal pinhole; "
                        "the plate has the real lens. Without this the comp "
                        "misaligns progressively toward the frame edges — "
                        "measured at 400px in the corners on one real solve.")
    p.add_argument("--preview", action="store_true",
                   help="also write preview.mp4 (1280 wide) for quick review")
    return p.parse_args()


def build_distort_map(solve_json, cw, ch):
    """Remap grids that warp a pinhole CG render into the plate's distorted
    image space, using the solve's own lens model.

    For every destination pixel (plate space) we need the CG source pixel:
    invert the forward distortion (undistorted -> distorted) per pixel. The
    polynomial model is radial, so it reduces to solving rd = ru * s(ru^2)
    for ru — done with vectorized Newton (the QC overlay already proved the
    forward model puts solve points onto the plate's markers, so this inverse
    puts CG pixels there too).

    Returns (mapx, mapy, max_shift_px) or None when there's no distortion.
    """
    d = json.load(open(solve_json))
    if d.get("distortion_model", "POLYNOMIAL") != "POLYNOMIAL":
        print("[comp] non-polynomial distortion model — CG left unwarped")
        return None
    k1, k2, k3 = d.get("k1", 0.0), d.get("k2", 0.0), d.get("k3", 0.0)
    if abs(k1) < 1e-9 and abs(k2) < 1e-9 and abs(k3) < 1e-9:
        return None
    clip_w, clip_h = d["clip_size"]
    f_px = d["lens_mm"] / d["sensor_mm"] * cw
    pp = d.get("principal_px") or [clip_w / 2.0, clip_h / 2.0]
    ppx, ppy = pp[0] / clip_w * cw, pp[1] / clip_h * ch   # bottom-left origin

    U, V = np.meshgrid(np.arange(cw, dtype=np.float64),
                       np.arange(ch, dtype=np.float64))
    xd = (U - ppx) / f_px
    yd = ((ch - ppy) - V) / f_px
    rd = np.sqrt(xd * xd + yd * yd)

    ru = rd.copy()                       # Newton: solve ru*s(ru^2) = rd
    for _ in range(10):
        r2 = ru * ru
        s = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
        ds = 1.0 + 3 * k1 * r2 + 5 * k2 * r2 ** 2 + 7 * k3 * r2 ** 3
        ru = ru - (ru * s - rd) / np.where(np.abs(ds) < 1e-9, 1e-9, ds)
    scale = np.where(rd > 1e-12, ru / np.where(rd > 1e-12, rd, 1.0), 1.0)
    xu, yu = xd * scale, yd * scale

    # CG was rendered by Blender: centred principal point
    mapx = (cw / 2.0 + f_px * xu).astype(np.float32)
    mapy = (ch / 2.0 - f_px * yu).astype(np.float32)
    shift = float(np.max(np.hypot(mapx - U, mapy - V)))
    # residual of the inversion (roundtrip): should be tiny
    r2 = ru * ru
    resid = float(np.max(np.abs(ru * (1 + k1 * r2 + k2 * r2 ** 2
                                      + k3 * r2 ** 3) - rd))) * f_px
    print(f"[comp] CG distortion-matched to the solve "
          f"(k1={k1:.3f}) — max shift {shift:.1f}px, "
          f"inversion residual {resid:.3f}px")
    return mapx, mapy, shift


def load_cg_frames(cg_dir):
    """Map shot-relative frame number -> CG png path."""
    out = {}
    for p in glob.glob(os.path.join(cg_dir, "*.png")):
        m = re.search(r"_(\d{4})\.png$", os.path.basename(p))
        if m:
            out[int(m.group(1))] = p
    return out


def load_alpha(args, rel_frame, size):
    """Person alpha in [0,1] at `size` (w, h). 1 = keep plate (person)."""
    w, h = size
    if args.alpha_dir:
        p = os.path.join(args.alpha_dir, f"alpha_{rel_frame:06d}.png")
        if os.path.exists(p):
            a = cv2.imread(p, 0)
            if a is not None:
                a = cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR)
                return a.astype(np.float32) / 255.0
    m = None
    if args.masks_dir:
        p = os.path.join(args.masks_dir, f"mask_{rel_frame:06d}.png")
        m = cv2.imread(p, 0) if os.path.exists(p) else None
    if m is None:
        return np.zeros((h, w), np.float32)   # no person this frame -> all CG
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
    a = (m > 127).astype(np.float32)
    if args.grow > 0:
        k = 2 * args.grow + 1
        a = cv2.dilate(a, np.ones((k, k), np.uint8))
    if args.feather > 0:
        k = 2 * args.feather + 1
        a = cv2.GaussianBlur(a, (k, k), 0)
    return a


def main():
    args = parse_args()
    cg = load_cg_frames(args.cg_dir)
    if not cg:
        sys.exit(f"no CG frames (*_NNNN.png) found in {args.cg_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    if args.frames:
        a, b = (int(v) for v in args.frames.split("-"))
    else:
        a, b = 1, max(cg)

    cap = cv2.VideoCapture(args.footage)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.footage}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, a - 1)

    first = cv2.imread(cg[min(cg)])
    ch, cw = first.shape[:2]
    dmap = None
    if args.solve_json and os.path.exists(args.solve_json):
        dmap = build_distort_map(args.solve_json, cw, ch)
    writer = None
    if args.preview:
        pw = 1280
        ph2 = round(ch * pw / cw)
        ph2 -= ph2 % 2
        writer = cv2.VideoWriter(os.path.join(args.out_dir, "preview.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 24.0,
                                 (pw, ph2))

    n_done = 0
    for src_f in range(a, b + 1):
        rel = src_f - a + 1
        ok, plate = cap.read()
        if not ok:
            print(f"[comp] plate ended at source frame {src_f}")
            break
        if rel not in cg:
            continue   # CG frame missing (partial render) — skip, don't fake
        cgf = cv2.imread(cg[rel])
        if cgf is None:
            continue
        if dmap is not None:
            # warp the pinhole CG into the plate's distorted space. Corners
            # can sample slightly beyond the rendered frame (the plate's
            # barrel pulls edges inward); replicate is the least-bad fill —
            # a proper fix is rendering CG with overscan, noted for later.
            cgf = cv2.remap(cgf, dmap[0], dmap[1], cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
        plate_r = cv2.resize(plate, (cw, ch), interpolation=cv2.INTER_AREA)
        alpha = load_alpha(args, rel, (cw, ch))[..., None]
        comp = plate_r.astype(np.float32) * alpha \
            + cgf.astype(np.float32) * (1.0 - alpha)
        out = np.clip(comp, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(args.out_dir, f"comp_{rel:04d}.png"), out)
        if writer is not None:
            writer.write(cv2.resize(out, (pw, ph2)))
        n_done += 1
    cap.release()
    if writer is not None:
        writer.release()
        print(f"[comp] preview: {os.path.join(args.out_dir, 'preview.mp4')}")
    print(f"[comp] {n_done} frames -> {args.out_dir}")
    if n_done == 0:
        sys.exit("no frames composited")


if __name__ == "__main__":
    main()
