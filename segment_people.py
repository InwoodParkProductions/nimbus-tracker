"""
Stage 2a: Person segmentation — clip in, per-frame person masks out.
--------------------------------------------------------------------
Run with the SYSTEM python (not Blender's):
    python segment_people.py <footage_path> <masks_dir> [--engine sam2|yolo]
        [--dilate 32] [--conf 0.2] [--max-dim 512]

Engines:
    sam2 (default): YOLO finds people at chunk boundaries, SAM2 tracks their
        exact silhouettes through the video. Temporally stable — a costumed
        actor detected once stays masked on every frame, no flicker. Falls
        back to 'yolo' automatically if SAM2 can't run.
    yolo: classic per-frame detection (the original engine).

Output:
    <masks_dir>/mask_000001.png ... one 8-bit grayscale PNG per frame,
        white (255) = person (exclude from tracking), black = background.
        Frame numbering is 1-based to match Blender clip frames.
    <masks_dir>/manifest.json   ... frame count, source size, mask size, params.

Masks are dilated at source resolution (so the safety margin is in real
pixels), then downscaled to --max-dim for compact storage. Point-in-mask
tests downstream use normalized coordinates, so mask resolution doesn't
need to match the clip.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
_BUNDLED_MODEL = os.path.join(HERE, "yolo11n-seg.pt")
_BUNDLED_SAM = os.path.join(HERE, "sam2.1_s.pt")

import cv2
import numpy as np


PERSON_CLASS = 0  # COCO class id for 'person'
CHUNK = 48        # SAM2 propagation chunk (re-detect entrants every ~2s)


def parse_args():
    p = argparse.ArgumentParser(description="Generate per-frame person masks for tracking exclusion.")
    p.add_argument("footage", help="Path to input video")
    p.add_argument("masks_dir", help="Directory to write mask PNGs into")
    p.add_argument("--engine", default="sam2", choices=["sam2", "yolo"],
                   help="sam2 (default): detect once, track silhouettes "
                        "through time — stable masks, no flicker. "
                        "yolo: classic per-frame detection.")
    p.add_argument("--dilate", type=int, default=32,
                   help="Dilation radius in source pixels around detected "
                        "people (default 32 — covers costume edges / held "
                        "props that YOLO doesn't segment as 'person')")
    p.add_argument("--conf", type=float, default=0.2,
                   help="YOLO confidence threshold (default 0.2 — a bit "
                        "inclusive so partial/occluded people still get masked)")
    p.add_argument("--max-dim", type=int, default=1024,
                   help="Max dimension of saved masks (default 1024)")
    p.add_argument("--model",
                   default=_BUNDLED_MODEL if os.path.exists(_BUNDLED_MODEL)
                   else "yolo11n-seg.pt",
                   help="Ultralytics segmentation model for detection "
                        "(default: bundled yolo11n-seg.pt, else auto-download)")
    p.add_argument("--sam-model",
                   default=_BUNDLED_SAM if os.path.exists(_BUNDLED_SAM)
                   else "sam2.1_s.pt",
                   help="SAM2 model for silhouette tracking")
    return p.parse_args()


def pick_device():
    """GPU when genuinely usable, else CPU. is_available() alone is not
    enough — it can be True with zero visible devices."""
    try:
        import torch
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return 0, torch.cuda.get_device_name(0)
    except Exception:
        pass
    return "cpu", None


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


_CLOSE_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))


def _drop_specks(mask, min_area):
    """Remove tiny disconnected blobs (detector popcorn) — anything smaller
    than min_area can't be a person."""
    if not mask.any():
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 127).astype(np.uint8))
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def _boxes_from_mask(mask, min_area):
    """Bounding boxes of the people already being tracked (connected
    components of the previous chunk's final mask) — lets SAM2 keep
    following someone even on a frame where YOLO misses them."""
    n, _, stats, _ = cv2.connectedComponentsWithStats((mask > 127).astype(np.uint8))
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area >= min_area:
            out.append([float(x), float(y), float(x + w), float(y + h)])
    return out


def run_sam2(args, cap, src_w, src_h, out_w, out_h, scale, kernel, device):
    """Seamless SAM2 silhouette tracking.

    One continuous SAM2 pass covers the whole shot (its per-object memory
    keeps silhouettes coherent frame to frame — no chunk seams, no popping).
    A separate per-frame YOLO pass is reduced by MAJORITY VOTE (a region only
    counts when detected in >=3 of 5 surrounding frames) and unioned in as a
    safety net for anything SAM2 loses. Late entrants get their own SAM2
    pass, seeded where they first persistently appear."""
    from ultralytics import YOLO
    from ultralytics.models.sam import SAM2VideoPredictor

    det = YOLO(args.model)
    min_area = int(src_w * src_h * 0.0005)
    min_area_out = max(1, int(min_area * scale * scale))

    def to_out(mask_src):
        m = cv2.resize(mask_src, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        return ((m > 127) * 255).astype(np.uint8)

    def poly_mask(result):
        m = np.zeros((src_h, src_w), dtype=np.uint8)
        if result.masks is not None:
            for poly in result.masks.xy:
                if len(poly) >= 3:
                    cv2.fillPoly(m, [poly.astype(np.int32)], 255)
        return m

    # ---- pass A: per-frame YOLO masks (safety net) + motion + first frame ----
    yolo_raw = []
    motion_raw = []   # fast-moving regions: swinging props, whipping limbs
    first_boxes, first_idx = None, 0
    prev_gray = None
    prop_raw = []     # detected non-person objects (held props: swords, staffs)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # one inference, ALL classes: person polygons build the person mask,
        # every other detection is a candidate held prop
        r = det.predict(frame, conf=min(args.conf, 0.15),
                        device=device, verbose=False)[0]
        pm = np.zeros((src_h, src_w), dtype=np.uint8)
        qm = np.zeros((src_h, src_w), dtype=np.uint8)
        person_boxes = []
        if r.masks is not None and r.boxes is not None:
            cls = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            xyxy = r.boxes.xyxy.cpu().numpy()
            for k, poly in enumerate(r.masks.xy):
                if len(poly) < 3:
                    continue
                if cls[k] == PERSON_CLASS and confs[k] >= args.conf:
                    cv2.fillPoly(pm, [poly.astype(np.int32)], 255)
                    person_boxes.append(xyxy[k].tolist())
                elif cls[k] != PERSON_CLASS:
                    cv2.fillPoly(qm, [poly.astype(np.int32)], 255)
        if first_boxes is None and person_boxes:
            first_boxes = person_boxes
            first_idx = len(yolo_raw)
        yolo_raw.append(to_out(pm))
        prop_raw.append(to_out(qm))
        g = cv2.cvtColor(cv2.resize(frame, (out_w, out_h)), cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            mv = (cv2.absdiff(g, prev_gray) > 28).astype(np.uint8) * 255
            mv = cv2.dilate(mv, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (9, 9)))
        else:
            mv = np.zeros((out_h, out_w), np.uint8)
        motion_raw.append(mv)
        prev_gray = g
        if len(yolo_raw) % 50 == 0:
            print(f"[segment] frame {len(yolo_raw)}...")
    n = len(yolo_raw)
    if n == 0:
        return 0, 0

    # Vote kills per-frame detector oscillation. Each frame's mask is dilated
    # BEFORE voting so a fast-moving limb still overlaps itself across
    # neighbouring frames — otherwise the vote erases exactly the raised arms
    # and swinging props we most need masked.
    vote_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
    yolo_fat = [cv2.dilate(m, vote_k) for m in yolo_raw]
    yolo_stable = []
    for i in range(n):
        lo, hi = max(0, i - 2), min(n, i + 3)
        stack = np.stack(yolo_fat[lo:hi]).astype(np.uint8)
        votes = (stack > 127).sum(axis=0)
        stable = ((votes >= 2) * 255).astype(np.uint8)
        # always keep this frame's own detections that CONNECT to the stable
        # mask (an arm attached to a person is that person, even if it moved
        # too fast to win any vote)
        raw = yolo_raw[i]
        if raw.any():
            nlab, labels = cv2.connectedComponents((raw > 127).astype(np.uint8))
            for c in range(1, nlab):
                comp = labels == c
                if (stable[comp] > 127).any():
                    stable[comp] = 255
        yolo_stable.append(stable)

    def sam_pass(start_idx, boxes):
        """One smooth SAM2 propagation from start_idx to the end."""
        out = {}
        if not boxes:
            return out
        src = args.footage
        if start_idx > 0:  # SAM2 prompts land on frame 1 of its source
            src = os.path.join(args.masks_dir, "_trim_tmp.mp4")
            c2 = cv2.VideoCapture(args.footage)
            c2.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
            vw = cv2.VideoWriter(src, cv2.VideoWriter_fourcc(*"mp4v"),
                                 24, (src_w, src_h))
            while True:
                ok, fr = c2.read()
                if not ok:
                    break
                vw.write(fr)
            vw.release(); c2.release()
        overrides = dict(conf=0.25, task="segment", mode="predict",
                         device=device, imgsz=512, model=args.sam_model,
                         save=False, verbose=False)
        pred = SAM2VideoPredictor(overrides=overrides)
        for j, res in enumerate(pred(source=src, bboxes=boxes, stream=True)):
            out[start_idx + j] = to_out(poly_mask(res))
        if src != args.footage:
            try:
                os.remove(src)
            except OSError:
                pass
        return out

    # ---- pass B: one seamless SAM2 track from the first people onward ----
    print("[segment] SAM2: tracking silhouettes through the whole shot...")
    sam_masks = sam_pass(first_idx, first_boxes or [])

    # ---- late entrants: anything YOLO keeps seeing that SAM2 never covers
    covered = lambda i: sam_masks.get(i, np.zeros((out_h, out_w), np.uint8))
    miss_start, run = None, 0
    for i in range(n):
        extra = (yolo_stable[i] > 127) & ~(covered(i) > 127)
        if extra.sum() > 4 * min_area_out:
            run += 1
            if miss_start is None:
                miss_start = i
            if run == 12:  # persistent for half a second -> real person
                c2 = cv2.VideoCapture(args.footage)
                c2.set(cv2.CAP_PROP_POS_FRAMES, miss_start)
                ok, fr = c2.read(); c2.release()
                if ok:
                    r = det.predict(fr, conf=args.conf,
                                    classes=[PERSON_CLASS],
                                    device=device, verbose=False)[0]
                    bx = (r.boxes.xyxy.cpu().numpy().tolist()
                          if r.boxes is not None else [])
                    if bx:
                        print(f"[segment] new person enters ~frame "
                              f"{miss_start + 1} - adding a track for them")
                        extra_masks = sam_pass(miss_start, bx)
                        for k, v in extra_masks.items():
                            np.maximum(covered(k), v, out=v)
                            sam_masks[k] = v
                break  # one rescue pass is enough in practice
        else:
            miss_start, run = None, 0

    # ---- carried gear no detector knows (prop swords, staffs, banners):
    # on keyframes, find strongly-coloured regions hugging a person that
    # nothing has explained, and point SAM2 at them directly. The saturation
    # threshold adapts to the frame, so a saturated backdrop (blue cloth)
    # raises the bar and never gets eaten. ----
    try:
        from ultralytics import SAM
        sam_img = None
        c3 = cv2.VideoCapture(args.footage)
        near_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        kf = 0
        while True:
            ok, fr = c3.read()
            if not ok or kf >= n:
                break
            if kf % 8 == 0:
                person = np.maximum(covered(kf), yolo_stable[kf]) > 127
                if person.any():
                    hsv = cv2.cvtColor(cv2.resize(fr, (out_w, out_h)),
                                       cv2.COLOR_BGR2HSV)
                    sat = hsv[..., 1].astype(np.int32)
                    med = int(np.median(sat))
                    cand = ((sat > med + 55) & ~person &
                            (cv2.dilate(person.astype(np.uint8), near_k) > 0))
                    cand = _drop_specks((cand * 255).astype(np.uint8),
                                        min_area_out)
                    if cand.any():
                        nl, lb, st, cent = cv2.connectedComponentsWithStats(
                            (cand > 127).astype(np.uint8))
                        pts = [cent[c] for c in range(1, min(nl, 5))]
                        if pts:
                            if sam_img is None:
                                sam_img = SAM(args.sam_model)
                            add = np.zeros((out_h, out_w), np.uint8)
                            for cx, cy in pts:
                                sx = cx * src_w / out_w
                                sy = cy * src_h / out_h
                                rr = sam_img(fr, points=[[sx, sy]],
                                             labels=[1], verbose=False)[0]
                                if rr.masks is not None:
                                    for poly in rr.masks.xy:
                                        if len(poly) >= 3:
                                            cv2.fillPoly(
                                                add,
                                                [(poly * [out_w / src_w,
                                                          out_h / src_h]
                                                  ).astype(np.int32)], 255)
                            # only keep additions that stay person-adjacent
                            # and person-sized — never a backdrop grab
                            if add.any() and (add > 127).mean() < 0.25:
                                np.maximum(prop_raw[kf], add,
                                           out=prop_raw[kf])
            kf += 1
        c3.release()
    except Exception as e:
        print(f"[segment] gear pass skipped ({type(e).__name__}: {e})")

    # ---- finish: union, de-speck, solidify, dilate, edge-smooth, write ----
    print("[segment] stabilising masks over time...")
    kernel_out = None
    if args.dilate > 0:
        k = 2 * max(1, int(args.dilate * scale)) + 1
        kernel_out = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    close_out = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * max(1, int(15 * scale)) + 1,) * 2)
    frames_with_people = 0
    finished = []
    for i in range(n):
        m = np.maximum(covered(i), yolo_stable[i])
        # attach fast-moving regions that touch a person: swinging swords,
        # whipping hands, flying capes — person-models don't know props, but
        # props move with their owner. Skipped when most of the frame is in
        # motion (camera whip) so we never eat the background.
        mv = motion_raw[i]
        if m.any() and mv.any() and mv.mean() < 0.35 * 255:
            nlab, labels = cv2.connectedComponents((mv > 127).astype(np.uint8))
            person = m > 127
            cap_area = 0.5 * person.sum() + 4 * min_area_out
            for c in range(1, nlab):
                comp = labels == c
                if comp.sum() <= cap_area and (person & comp).any():
                    m[comp] = 255
        # attach detected OBJECTS touching a person: a held sword/staff is
        # part of its owner even when perfectly still (motion can't see it,
        # and person-models never will)
        pr = prop_raw[i]
        if m.any() and pr.any():
            # small bridge so "in the hand" counts as touching
            prb = cv2.dilate(pr, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (13, 13)))
            nlab, labels = cv2.connectedComponents((prb > 127).astype(np.uint8))
            person = m > 127
            cap_area = 1.5 * person.sum()
            for c in range(1, nlab):
                comp = labels == c
                if comp.sum() <= cap_area and (person & comp).any():
                    m[comp] = 255
        m = _drop_specks(m, min_area_out)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, close_out)
        # fill enclosed pockets inside the silhouette group ("holes in the
        # middle of the people") — background fully surrounded by person
        # pixels is unreachable for the tracker anyway, and leaving it
        # invites markers between bodies
        if m.any():
            # pad with a background ring so the flood reaches every border-
            # connected region from one seed; what stays unfilled is a true
            # enclosed hole
            ff = cv2.copyMakeBorder(m, 1, 1, 1, 1,
                                    cv2.BORDER_CONSTANT, value=0)
            hmask = np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8)
            cv2.floodFill(ff, hmask, (0, 0), 255)
            holes = (ff[1:-1, 1:-1] == 0) & (m == 0)
            if holes.any() and holes.mean() < 0.06:  # only modest pockets
                m[holes] = 255
        if m.any():
            frames_with_people += 1
            if kernel_out is not None:
                m = cv2.dilate(m, kernel_out)
            # gaussian soften + re-threshold rounds the boundary so edges
            # do not sizzle frame to frame
            m = cv2.GaussianBlur(m, (9, 9), 0)
            m = ((m > 100) * 255).astype(np.uint8)
        finished.append(m)
    # Temporal hysteresis: a pixel masks ON instantly (fast limbs are never
    # missed — we even look 1 frame ahead) but only turns OFF after several
    # consecutive clear frames. Anything that blinks becomes solidly on;
    # motion leaves a short safe trail instead of flickering.
    LINGER = 5   # frames a mask persists after its person moves on
    LEAD = 4     # frames a mask appears BEFORE its person arrives — covers
                 # people entering frame before the detector fully commits
    for i in range(n):
        m = finished[i].copy()
        for j in range(i + 1, min(n, i + 1 + LEAD)):
            np.maximum(m, finished[j], out=m)
        for j in range(max(0, i - LINGER), i):
            np.maximum(m, finished[j], out=m)
        cv2.imwrite(os.path.join(args.masks_dir, f"mask_{i + 1:06d}.png"), m)

    return n, frames_with_people


def run_yolo(args, cap, src_w, src_h, out_w, out_h, scale, kernel, device):
    """Classic engine: independent YOLO detection on every frame."""
    from ultralytics import YOLO
    model = YOLO(args.model)

    frame_idx = 0
    frames_with_people = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        try:
            results = model.predict(frame, conf=args.conf,
                                    classes=[PERSON_CLASS],
                                    device=device, verbose=False)
        except Exception:
            if device == "cpu":
                raise  # CPU inference itself failed — a real error
            # GPU rejected the job (weird driver state, out of VRAM, …):
            # drop to CPU and keep going rather than failing the shot.
            print("[segment] GPU inference failed - falling back to CPU")
            device = "cpu"
            results = model.predict(frame, conf=args.conf,
                                    classes=[PERSON_CLASS],
                                    device=device, verbose=False)
        mask = np.zeros((src_h, src_w), dtype=np.uint8)
        r = results[0]
        if r.masks is not None:
            for poly in r.masks.xy:  # polygon per instance, in source pixels
                if len(poly) >= 3:
                    cv2.fillPoly(mask, [poly.astype(np.int32)], 255)
            frames_with_people += 1

        if kernel is not None and mask.any():
            mask = cv2.dilate(mask, kernel)
        if scale < 1.0:
            mask = cv2.resize(mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(os.path.join(args.masks_dir, f"mask_{frame_idx:06d}.png"), mask)
        if frame_idx % 50 == 0:
            print(f"[segment] frame {frame_idx}...")

    return frame_idx, frames_with_people


def main():
    args = parse_args()

    device, gpu_name = pick_device()
    if gpu_name:
        print(f"[segment] running on GPU: {gpu_name}")
    else:
        print("[segment] no usable CUDA GPU - running on CPU")

    cap = cv2.VideoCapture(args.footage)
    if not cap.isOpened():
        sys.exit(f"Cannot open footage: {args.footage}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = min(1.0, args.max_dim / max(src_w, src_h))
    out_w, out_h = round(src_w * scale), round(src_h * scale)

    os.makedirs(args.masks_dir, exist_ok=True)

    kernel = None
    if args.dilate > 0:
        k = 2 * args.dilate + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    engine = args.engine
    if engine == "sam2":
        try:
            print("[segment] engine: SAM2 silhouette tracking "
                  f"(detector: {os.path.basename(args.model)})")
            frame_idx, frames_with_people = run_sam2(
                args, cap, src_w, src_h, out_w, out_h, scale, kernel, device)
        except Exception as e:
            # anything at all goes wrong -> classic engine, shot still masks
            print(f"[segment] SAM2 unavailable ({type(e).__name__}: {e}) "
                  "- falling back to per-frame YOLO")
            engine = "yolo"
            cap.release()
            cap = cv2.VideoCapture(args.footage)
            frame_idx, frames_with_people = run_yolo(
                args, cap, src_w, src_h, out_w, out_h, scale, kernel, device)
    else:
        frame_idx, frames_with_people = run_yolo(
            args, cap, src_w, src_h, out_w, out_h, scale, kernel, device)

    cap.release()

    manifest = {
        "footage": os.path.abspath(args.footage),
        "num_frames": frame_idx,
        "frames_with_people": frames_with_people,
        "source_size": [src_w, src_h],
        "mask_size": [out_w, out_h],
        "dilate_px": args.dilate,
        "conf": args.conf,
        "engine": engine,
        "model": args.model,
        "sam_model": args.sam_model if engine == "sam2" else None,
        "convention": "white=person=exclude, frame numbering 1-based",
    }
    with open(os.path.join(args.masks_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[segment] Done ({engine}). {frame_idx} frames, people found in "
          f"{frames_with_people}. Masks in: {args.masks_dir}")


if __name__ == "__main__":
    main()
