"""Stage 5: export comp ELEMENTS for assembly in Resolve (not a baked comp).

Run with the SYSTEM python:
    python comp_stage5.py <footage> <cg_dir> <masks_dir> <out_dir>
        [--frames A-B] [--grow 2] [--feather 5] [--solve-json solve.json]

A pre-flattened composite gives the colorist zero control — you can't grade the
CG and the actor separately, or refine the key, once they're baked into one
image. So instead of ONE comp, this exports the two elements that stack in
Resolve:

    <out>/bg/bg_####.png    CG background, warped to the plate's lens
                            distortion (goes on the lower track, V1)
    <out>/fg/fg_####.png    the actor as RGBA — plate RGB with the person matte
                            in the alpha channel (goes on top, V2)
    <out>/matte/####.png    the alpha on its own, for refining the key/grade

In Resolve: drop fg on V2 over bg on V1 — it composites automatically because
fg carries alpha — then grade each track to taste. A flattened preview
(comp_sheet.png + preview.mp4) is written too, for QC only.

The matte is RVM's soft alpha (see matte_people.py, --alpha-dir); it falls
back to grown+feathered SAM2 masks when no RVM alpha is supplied.
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
    model = d.get("distortion_model", "POLYNOMIAL")
    if model == "DIVISION":
        ks = [d.get("division_k1", 0.0), d.get("division_k2", 0.0)]
    elif model == "POLYNOMIAL":
        ks = [d.get("k1", 0.0), d.get("k2", 0.0), d.get("k3", 0.0)]
    else:
        print(f"[comp] {model} distortion not supported — CG left unwarped")
        return None
    if all(abs(k) < 1e-9 for k in ks):
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

    def fwd(ru):
        """distorted radius as a function of undistorted radius."""
        r2 = ru * ru
        if model == "DIVISION":
            return ru / (1.0 + ks[0] * r2 + ks[1] * r2 ** 2)
        return ru * (1.0 + ks[0] * r2 + ks[1] * r2 ** 2 + ks[2] * r2 ** 3)

    # Invert on a 1D radius table, marching outward from the centre. The
    # model is radial, so this is exact — and it is the only robust way:
    # overfit solves FOLD (fwd stops being monotonic), and 2D Newton then
    # either runs away (a 14-million-px "shift", observed) or converges to
    # the fold's far branch, which passes a residual check while sampling
    # garbage (an 8977px "shift", also observed). Marching from zero keeps
    # every root on the near branch and finds the exact radius where the
    # model stops being invertible; beyond it the map falls back to identity
    # and says so.
    rd_max = float(rd.max())
    n_tab = 4096
    ru_step = rd_max * 2.0 / n_tab
    ru_tab = [0.0]
    rd_tab = [0.0]
    ru_cur, rd_fold = 0.0, None
    while rd_tab[-1] < rd_max:
        ru_nxt = ru_cur + ru_step
        rd_nxt = float(fwd(np.array([ru_nxt]))[0])
        if rd_nxt <= rd_tab[-1] or not np.isfinite(rd_nxt):
            rd_fold = rd_tab[-1]        # fwd turned over: fold reached
            break
        ru_tab.append(ru_nxt)
        rd_tab.append(rd_nxt)
        ru_cur = ru_nxt
        if ru_cur > rd_max * 4.0:       # sampling absurdly far outside CG
            rd_fold = rd_tab[-1]
            break
    # np.interp CLAMPS beyond the table's range: radii past the fold get the
    # fold's ru, not identity. That matters — an identity fallback beyond the
    # fold leaves a hard curved SEAM in the comp where warped meets unwarped
    # (visible as a tear on a detailed background). Clamping instead stretches
    # the CG continuously at the extreme edge: no seam, just mild smear in the
    # last ~1% of frame that an overfit solve can't model anyway.
    ru = np.interp(rd, rd_tab, ru_tab)
    if rd_fold is not None:
        beyond = float((rd > rd_fold).mean())
        print(f"[comp] distortion model folds at r={rd_fold:.3f} "
              f"(frame corner r={rd_max:.3f}) — {100.0 * beyond:.1f}% of "
              "pixels beyond it are edge-clamped (overfit solve)")
    scale = np.where(rd > 1e-12, ru / np.where(rd > 1e-12, rd, 1.0), 1.0)
    xu, yu = xd * scale, yd * scale

    # CG was rendered by Blender: centred principal point
    mapx = (cw / 2.0 + f_px * xu).astype(np.float32)
    mapy = (ch / 2.0 - f_px * yu).astype(np.float32)
    shift = float(np.max(np.hypot(mapx - U, mapy - V)))

    # Refuse a non-physical warp. The distortion refine is unstable run to run
    # (div_k1 seen from 0.17 to 1.25 on the SAME shot) and an overfit solve
    # produces a warp that bends the CG grotesquely: k1=1.25 gave a 4226px
    # shift — over 100% of frame width, which no real lens does (a fisheye is
    # ~15-20% at the corner, a normal lens <2%). Applying it destroys the
    # frame, and an unwarped comp — a few px of edge misalignment — is far
    # better than that. So above a physical ceiling, skip the warp entirely.
    max_frac = 0.12    # 12% of frame width; generously past any real lens
    if shift > max_frac * cw:
        print(f"[comp] distortion warp is {shift:.0f}px "
              f"({100.0 * shift / cw:.0f}% of frame width) — non-physical, "
              "the solve overfit its lens. Skipping the warp (CG left as "
              "rendered; expect a few px of edge misalignment instead).")
        return None
    # residual of the inversion (roundtrip) on the invertible pixels — inside
    # the fold radius, where fwd(ru) should reproduce rd
    inside = rd <= rd_fold if rd_fold is not None else np.ones_like(rd, bool)
    resid = (float(np.max(np.abs(fwd(ru) - rd)[inside])) * f_px
             if inside.any() else 0.0)
    print(f"[comp] CG distortion-matched to the solve "
          f"({model} {[round(k, 4) for k in ks]}) — max shift {shift:.1f}px, "
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
    pw = 1280
    ph2 = round(ch * pw / cw); ph2 -= ph2 % 2
    writer = cv2.VideoWriter(os.path.join(args.out_dir, "preview.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (pw, ph2))

    bg_dir = os.path.join(args.out_dir, "bg")
    fg_dir = os.path.join(args.out_dir, "fg")
    matte_dir = os.path.join(args.out_dir, "matte")
    for dd in (bg_dir, fg_dir, matte_dir):
        os.makedirs(dd, exist_ok=True)

    n_done = 0
    sheet_rows = []
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
            # can sample slightly beyond the rendered frame; replicate fill,
            # and np.interp clamps the map at the fold so there's no seam.
            cgf = cv2.remap(cgf, dmap[0], dmap[1], cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_REPLICATE)
        plate_r = cv2.resize(plate, (cw, ch), interpolation=cv2.INTER_AREA)
        alpha = load_alpha(args, rel, (cw, ch))          # 1 = person
        a8 = np.clip(alpha * 255, 0, 255).astype(np.uint8)

        # BG element: the CG, lens-matched. Straight RGB.
        cv2.imwrite(os.path.join(bg_dir, f"bg_{rel:04d}.png"), cgf)
        # FG element: plate RGB + person alpha, as RGBA. Premultiply is what
        # Resolve/most compositors expect for a clean edge (no dark or bright
        # fringe where alpha is partial); the straight matte rides in A so the
        # colorist can still pull/refine the key.
        af = alpha[..., None]
        fg_rgb = np.clip(plate_r.astype(np.float32) * af, 0, 255).astype(np.uint8)
        fg = cv2.merge([fg_rgb[:, :, 0], fg_rgb[:, :, 1], fg_rgb[:, :, 2], a8])
        cv2.imwrite(os.path.join(fg_dir, f"fg_{rel:04d}.png"), fg)
        # matte on its own — grade/refine handle
        cv2.imwrite(os.path.join(matte_dir, f"matte_{rel:04d}.png"), a8)

        # QC-only flattened preview (over/under composite of the two elements)
        comp = np.clip(plate_r.astype(np.float32) * af
                       + cgf.astype(np.float32) * (1 - af), 0, 255).astype(np.uint8)
        if writer is not None:
            writer.write(cv2.resize(comp, (pw, ph2)))
        if rel in (1, None) or len(sheet_rows) < 3:
            tw = 420
            th = round(ch * tw / cw); th -= th % 2
            def lbl(im, t, c):
                im = cv2.resize(im, (tw, th))
                cv2.putText(im, t, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
                return im
            bg_t = lbl(cgf, f"BG (CG) f{rel}", (0, 200, 255))
            # show FG on checkerboard so its alpha is visible
            chk = np.zeros((ch, cw, 3), np.uint8)
            s = max(16, cw // 40)
            chk[:] = 90
            chk[(np.add.outer(np.arange(ch) // s, np.arange(cw) // s) % 2) == 0] = 150
            fg_over = np.clip(chk.astype(np.float32) * (1 - af)
                              + plate_r.astype(np.float32) * af, 0, 255).astype(np.uint8)
            fg_t = lbl(fg_over, "FG (actor+alpha)", (0, 255, 0))
            comp_t = lbl(comp, "stacked (preview)", (255, 255, 255))
            sheet_rows.append(np.hstack([bg_t, fg_t, comp_t]))
        n_done += 1
    cap.release()
    if writer is not None:
        writer.release()
        print(f"[comp] preview: {os.path.join(args.out_dir, 'preview.mp4')}")
    if sheet_rows:
        cv2.imwrite(os.path.join(args.out_dir, "comp_sheet.png"),
                    np.vstack(sheet_rows[:3]))
        print(f"[comp] contact sheet: "
              f"{os.path.join(args.out_dir, 'comp_sheet.png')}")
    print(f"[comp] {n_done} frames -> bg/ fg/ matte/ in {args.out_dir}")
    print("[comp] Resolve: put fg/ on V2 over bg/ on V1 (fg carries alpha)")
    if n_done == 0:
        sys.exit("no frames composited")


if __name__ == "__main__":
    main()
