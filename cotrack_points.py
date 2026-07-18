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
    p.add_argument("--seeds", type=int, default=1,
                   help="frames across the shot to seed points on "
                        "(default 1). Re-seeding SHOULD help long shots — "
                        "seeding only frame 1 means the camera moves off "
                        "everything it started on. It measurably does not: "
                        "on shot 10 (1395 frames) 4 seeds took survivors "
                        "183 -> 947 and keyframe-shared tracks 15 -> 41, "
                        "and the solve went from 5.87px to 295px (no "
                        "solve). More shared tracks, worse answer — the "
                        "extra tracks are seeded at different times and "
                        "appear to poison the bundle. Kept for "
                        "experimentation; do not raise without re-running "
                        "the A/B.")
    p.add_argument("--max-tracks", type=int, default=200,
                   help="cap on tracks handed to the solver (default 200). "
                        "More is not better: every track is 3 more unknowns in "
                        "the bundle, and on a rotation-dominant shot (no "
                        "parallax) those points are unconstrained. 478 tracks "
                        "on a 34-frame pan left Blender's Ceres solver "
                        "thrashing on infinite-cost steps for 8.5 hours. "
                        "~150-200 well-spread tracks solve better AND faster; "
                        "the cap keeps them spread rather than clumped.")
    p.add_argument("--max-frames", type=int, default=4000,
                   help="refuse shots longer than this (default 4000). The "
                        "online model streams in a fixed-size window so VRAM "
                        "does not grow with shot length; this is just a "
                        "sanity bound.")
    p.add_argument("--offline", action="store_true",
                   help="use the offline model (slightly better, but holds "
                        "the WHOLE clip in VRAM — 335 frames already OOMs an "
                        "8GB card, so this is short shots only)")
    p.add_argument("--tracker", default="cotracker",
                   choices=sorted(TRACKERS),
                   help="tracking backend (default cotracker). cotracker is "
                        "CC-BY-NC (non-commercial); the others are Apache-2.0 "
                        "and get wired in for a commercial-clean pipeline. "
                        "Reverting to cotracker is just this flag — no code "
                        "rollback.")
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


def pick_seed_frames(masks_dir, n_frames, w, h, n_seeds):
    """Frames to seed points on, spread across the shot.

    Seeding only frame 1 is fine for a short shot and useless for a long one:
    the camera moves off everything it started on, so tracks die and no two
    keyframes share enough of them to triangulate. Shot 10 (1395 frames) kept
    183 tracks of which only 15 were common to the best keyframe pair — not
    enough for a 3D solve. Re-seeding across the shot keeps live tracks
    everywhere on the timeline.

    Within each segment, prefer the frame with the most visible background —
    frame 1 is often the worst (a person entering, filling frame), which is
    why the classic detector sweeps candidate frames too.
    """
    n_seeds = max(1, min(n_seeds, max(1, n_frames // 8)))
    if n_frames <= 1:
        return [1]
    # segment the shot, then pick the most-background frame inside each
    edges = np.linspace(1, n_frames + 1, n_seeds + 1).astype(int)
    seeds = []
    for i in range(n_seeds):
        lo, hi = int(edges[i]), max(int(edges[i]) + 1, int(edges[i + 1]))
        cands = np.linspace(lo, hi - 1, min(5, hi - lo)).astype(int)
        if not masks_dir:
            seeds.append(int(cands[0]))
            continue
        best, best_bg = int(cands[0]), -1.0
        for f in cands:
            bg = 1.0 - load_mask(masks_dir, int(f), w, h).mean()
            if bg > best_bg:
                best, best_bg = int(f), bg
        seeds.append(best)
    return sorted(set(seeds))


# ---- pluggable tracker backends -----------------------------------------
# The tracker is a swappable backend, selected by --tracker. This exists so
# CoTracker can be replaced with a permissively-licensed tracker (BootsTAPIR /
# TAPNext / LocoTrack, all Apache-2.0) WITHOUT losing the CoTracker path:
# reverting is just `--tracker cotracker` (the default), never a code
# rollback. Each backend takes (vid_cpu[1,T,3,H,W] float on CPU, queries
# [1,N,3] (t,x,y) on dev, T, dev, offline) and returns (tr[T,N,2] px np,
# vv[T,N] bool np, T). A new tracker is one more function registered in
# TRACKERS; the CoTracker code below is never touched by adding one.
#
# LICENSE: cotracker is CC-BY-NC (non-commercial). It stays the default for
# quality, but for commercial use pick a permissive backend once integrated,
# or run the pipeline with --no-cotracker (classic KLT front-end).

def track_cotracker(vid_cpu, q, T, dev, offline):
    """CoTracker3 (Meta). NON-COMMERCIAL license — see the note above."""
    import torch
    if offline:
        model = torch.hub.load("facebookresearch/co-tracker",
                               "cotracker3_offline", trust_repo=True).to(dev)
        with torch.no_grad():
            tracks, vis = model(vid_cpu.to(dev), queries=q)
    else:
        # Online model: fixed-size sliding window, so VRAM is flat in shot
        # length. The offline model holds the entire clip at once and OOMs an
        # 8GB card at 335 frames — measured, on shots 01/06/10 of this project
        # (which is to say: on most real shots).
        model = torch.hub.load("facebookresearch/co-tracker",
                               "cotracker3_online", trust_repo=True).to(dev)
        step = model.step
        with torch.no_grad():
            model(video_chunk=vid_cpu[:, :step * 2].to(dev),
                  is_first_step=True, queries=q)
            tracks = vis = None
            for i in range(0, max(1, T - step), step):
                chunk = vid_cpu[:, i:i + step * 2].to(dev)
                if chunk.shape[1] < 2:
                    break
                out = model(video_chunk=chunk)
                if out[0] is not None:
                    tracks, vis = out
            if tracks is None:
                sys.exit("online tracker returned nothing")
    tr = tracks[0].cpu().numpy()          # (T, N, 2) in processing px
    vv = (vis[0].cpu().numpy() > 0.5)     # (T, N)
    # The online model reports on the frames it has consumed, which can lag
    # the clip by less than one window; clamp so indexing stays safe.
    del tracks, vis
    return tr, vv, min(T, tr.shape[0])


# Permissive backends get registered here as they are integrated + A/B'd
# against cotracker. Until then they raise a clear message rather than
# silently doing nothing.
def _not_integrated(name):
    def _stub(*_a, **_k):
        sys.exit(f"--tracker {name} is not integrated yet; the interface is "
                 "ready but the backend + weights aren't wired in. Use "
                 "--tracker cotracker (default) for now.")
    return _stub


TRACKERS = {
    "cotracker": track_cotracker,       # default; non-commercial
    "tapnext": _not_integrated("tapnext"),      # Apache-2.0 (planned)
    "bootstapir": _not_integrated("bootstapir"),  # Apache-2.0 (planned)
    "locotrack": _not_integrated("locotrack"),    # Apache-2.0 (planned)
}


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

    seed_frames = pick_seed_frames(a.masks, T, W, H, a.seeds)
    m = a.spacing
    ys, xs = np.mgrid[m:H - m:a.spacing, m:W - m:a.spacing]
    pts, seed_of = [], []
    for sf in seed_frames:
        bg = ~load_mask(a.masks, sf, W, H)
        n0 = len(pts)
        for x, y in zip(xs.ravel(), ys.ravel()):
            if bg[y, x]:
                pts.append((int(x), int(y)))
                seed_of.append(sf)
        print(f"[cotrack]   seed frame {sf}: {100*bg.mean():.0f}% background, "
              f"{len(pts)-n0} points")
    if len(pts) < 8:
        sys.exit(f"only {len(pts)} background points across seed frames "
                 f"{seed_frames} — the frame is essentially all person")
    print(f"[cotrack] {T} frames at {W}x{H}, {len(seed_frames)} seed frames, "
          f"{len(pts)} points seeded")

    dev = a.device if (a.device != "cuda" or torch.cuda.device_count()) else "cpu"
    # queries are (t, x, y) with t the frame the point is seeded on
    q = torch.tensor([[float(sf - 1), float(x), float(y)]
                      for (x, y), sf in zip(pts, seed_of)],
                     dtype=torch.float32, device=dev)[None]
    # Video stays on the CPU: at 512x214 a 1395-frame shot is ~1.8GB on its
    # own, before the model allocates anything.
    vid_cpu = torch.tensor(np.stack(frames)).permute(0, 3, 1, 2)[None].float()

    tr, vv, T = TRACKERS[a.tracker](vid_cpu, q, T, dev, a.offline)
    del vid_cpu
    if dev == "cuda":
        torch.cuda.empty_cache()
    print(f"[cotrack] tracked {tr.shape[0]} frames, {tr.shape[1]} points "
          f"(backend: {a.tracker})")

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
        # A point seeded at 3/4 through the shot cannot live T//4 frames.
        # Judge each track against the span still ahead of its seed.
        avail = max(1, T - (seed_of[i] - 1))
        if live >= max(8, avail // 4):
            keep.append(i)
    print(f"[cotrack] {len(keep)} tracks survive masking+visibility "
          f"(of {tr.shape[1]} seeded)")
    if not keep:
        sys.exit("no usable background tracks")

    if len(keep) > a.max_tracks:
        # Thin to max_tracks, keeping them SPREAD across the frame. A solve
        # wants coverage, not count: tracks clumped in one corner constrain
        # the camera far worse than the same number spread wide, and every
        # extra track is 3 more unknowns in an already-degenerate bundle.
        # Stratify by seed cell, then prefer the longest-lived in each.
        cells = {}
        gx = max(1, int(np.ceil(np.sqrt(a.max_tracks * W / max(H, 1)))))
        gy = max(1, int(np.ceil(a.max_tracks / gx)))
        for i in keep:
            x, y = pts[i]
            cell = (min(gx - 1, x * gx // W), min(gy - 1, y * gy // H))
            cells.setdefault(cell, []).append(i)
        for c in cells:
            cells[c].sort(key=lambda i: -int(vv[:, i].sum()))  # longest first
        thinned, r = [], 0
        while len(thinned) < a.max_tracks:
            added = False
            for c in sorted(cells):
                if r < len(cells[c]):
                    thinned.append(cells[c][r])
                    added = True
                    if len(thinned) >= a.max_tracks:
                        break
            if not added:
                break
            r += 1
        print(f"[cotrack] thinned {len(keep)} -> {len(thinned)} tracks across "
              f"{len(cells)} cells (cap {a.max_tracks})")
        keep = thinned

    out = {"num_frames": T, "seed_frames": seed_frames,
           "proc_size": [W, H],
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
