"""Stage 2a: learned point tracking — background tracks on footage KLT can't hold.

Run with the SYSTEM python:
    python cotrack_points.py <shot.mp4> <out.json> [--masks <dir>]
        [--width 512] [--spacing 12] [--max-frames 400] [--device cuda]

Why this exists
---------------
The classic front-end runs goodFeaturesToTrack over the whole frame and then
throws away every feature that landed on a person. On soft, motion-blurred
footage the people ARE the sharpest thing in frame, so that spends the entire
feature budget on pixels it is about to discard. Measured on real shots:

    shot   detected  dropped_on_person  solved   mode
    01           46                 19      13   perspective
    06           66                 52       9   perspective
    09           62                 58       4   2d-flow
    10           53                 53       0   2d-flow     <- all of them
    19           57                 51       5   tripod

The solver was never the problem — fed well it returns 179 tracks at 1.53px.
It was being handed 0-5.

This inverts it: seed a dense grid ON THE BACKGROUND ONLY (the person masks
already tell us where that is), then track those points with CoTracker3, which
follows arbitrary points through blur and low texture instead of requiring
corners. On shot 19 that is 405 usable background tracks vs 5.

Output JSON is resolution-independent (normalized, Blender's bottom-left
origin) so the solve can run at any clip size:
    {"num_frames": T, "seed_frame": f, "tracks": [[[x, y, vis], ...T], ...N]}
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("footage")
    p.add_argument("out_json")
    p.add_argument("--masks", help="mask_NNNNNN.png dir (white=person=don't seed)")
    p.add_argument("--width", type=int, default=512,
                   help="processing width (default 512; the tracker is scale-"
                        "invariant and this keeps VRAM sane)")
    p.add_argument("--spacing", type=int, default=12,
                   help="grid spacing in px at --width (default 12)")
    p.add_argument("--max-frames", type=int, default=400,
                   help="refuse shots longer than this (default 400). The "
                        "offline model holds the whole clip in VRAM, and a "
                        "windowed pass would break tracks across window "
                        "boundaries — which is exactly what the solver needs "
                        "to span its two keyframes. Caller falls back to the "
                        "classic front-end.")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def load_mask(masks_dir, frame_no, w, h):
    """Person mask for a 1-based frame, resized to the processing size.
    True = person = do not seed / not background."""
    if not masks_dir:
        return np.zeros((h, w), bool)
    p = os.path.join(masks_dir, f"mask_{frame_no:06d}.png")
    if not os.path.exists(p):
        return np.zeros((h, w), bool)
    m = cv2.imread(p, 0)
    if m is None:
        return np.zeros((h, w), bool)
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 127


def pick_seed_frame(masks_dir, n_frames, w, h):
    """Seed where the most background is visible. Frame 1 is often the worst
    (person entering, filling frame) — the classic detector has the same
    problem and sweeps candidate frames for the same reason."""
    if not masks_dir:
        return 1
    best, best_bg = 1, -1.0
    for f in np.linspace(1, n_frames, min(8, n_frames)).astype(int):
        bg = 1.0 - load_mask(masks_dir, int(f), w, h).mean()
        if bg > best_bg:
            best, best_bg = int(f), bg
    return best


def main():
    a = parse_args()
    import torch

    cap = cv2.VideoCapture(a.footage)
    if not cap.isOpened():
        sys.exit(f"cannot open {a.footage}")
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        h = round(f.shape[0] * a.width / f.shape[1])
        frames.append(cv2.cvtColor(cv2.resize(f, (a.width, h)), cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        sys.exit("no frames read")
    T = len(frames)
    H, W = frames[0].shape[:2]
    if T > a.max_frames:
        sys.exit(f"shot is {T} frames (> --max-frames {a.max_frames})")

    seed_f = pick_seed_frame(a.masks, T, W, H)
    person = load_mask(a.masks, seed_f, W, H)
    bg = ~person
    m = a.spacing
    ys, xs = np.mgrid[m:H - m:a.spacing, m:W - m:a.spacing]
    pts = [(int(x), int(y)) for x, y in zip(xs.ravel(), ys.ravel()) if bg[y, x]]
    if len(pts) < 8:
        sys.exit(f"only {len(pts)} background points at seed frame {seed_f} — "
                 "the frame is essentially all person")
    print(f"[cotrack] {T} frames at {W}x{H}, seed frame {seed_f}, "
          f"{100*bg.mean():.0f}% background, {len(pts)} points seeded")

    dev = a.device if (a.device != "cuda" or torch.cuda.device_count()) else "cpu"
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline",
                           trust_repo=True).to(dev)
    # queries are (t, x, y) with t the frame the point is seeded on
    q = torch.tensor([[float(seed_f - 1), float(x), float(y)] for x, y in pts],
                     dtype=torch.float32, device=dev)[None]
    vid = torch.tensor(np.stack(frames), device=dev).permute(0, 3, 1, 2)[None].float()
    with torch.no_grad():
        tracks, vis = model(vid, queries=q)
    tr = tracks[0].cpu().numpy()          # (T, N, 2) in processing px
    vv = (vis[0].cpu().numpy() > 0.5)     # (T, N)
    del vid, tracks, vis
    if dev == "cuda":
        torch.cuda.empty_cache()

    # Drop points that wander onto a person: the whole point is a background
    # solve, and a marker riding a moving actor is worse than no marker.
    keep = []
    person_by_frame = {f: load_mask(a.masks, f, W, H) for f in range(1, T + 1)} \
        if a.masks else None
    for i in range(tr.shape[1]):
        live = 0
        for t in range(T):
            if not vv[t, i]:
                continue
            x, y = tr[t, i]
            xi, yi = int(round(x)), int(round(y))
            if not (0 <= xi < W and 0 <= yi < H):
                vv[t, i] = False
                continue
            if person_by_frame is not None and person_by_frame[t + 1][yi, xi]:
                vv[t, i] = False   # on a person this frame: mute, don't kill
                continue
            live += 1
        if live >= max(8, T // 4):     # needs enough life to constrain a solve
            keep.append(i)
    print(f"[cotrack] {len(keep)} tracks survive masking+visibility "
          f"(of {tr.shape[1]} seeded)")
    if not keep:
        sys.exit("no usable background tracks")

    out = {"num_frames": T, "seed_frame": seed_f, "proc_size": [W, H],
           "tracks": []}
    for i in keep:
        # normalized, bottom-left origin — Blender's marker.co convention
        out["tracks"].append(
            [[float(tr[t, i, 0] / W), float(1.0 - tr[t, i, 1] / H),
              bool(vv[t, i])] for t in range(T)])
    with open(a.out_json, "w") as f:
        json.dump(out, f)
    print(f"[cotrack] wrote {len(out['tracks'])} tracks -> {a.out_json}")


if __name__ == "__main__":
    main()
