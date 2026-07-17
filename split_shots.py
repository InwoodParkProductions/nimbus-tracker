"""
Stage 0: Shot splitting — edited footage in, single-shot clips out.
-------------------------------------------------------------------
Camera tracking is only meaningful within one continuous shot, so any
edited sequence must be split first.

Run with the SYSTEM python:
    python split_shots.py <footage> <out_dir> [--threshold 27] [--min-len 24] [--list-only]

Output:
    <out_dir>/shot_01.mp4, shot_02.mp4, ...   (re-encoded with mp4v, no ffmpeg needed)
    <out_dir>/shots.json                       (frame ranges, 1-based inclusive)

Use --list-only to just print/write the ranges without writing videos.
"""

import argparse
import glob
import json
import os

import cv2
from scenedetect import detect, AdaptiveDetector

# Proxy codecs. mp4v stays the default — measured, not assumed.
#
# Corner counts say lossless should win: against a clean downscale of real
# (soft, motion-blurred) footage, ffv1 keeps 100% of goodFeaturesToTrack
# corners and mp4v keeps 71%. But corner count is NOT solve quality, and an
# end-to-end A/B of stage 2 says the opposite:
#   shot 19:  mp4v -> tripod solve, 2.58px, 6 tracks
#             ffv1 -> REJECTED, only 3 contributing tracks
#   shot 09:  mp4v -> rejected (4 tracks) | ffv1 -> rejected (4 tracks)
# The likely reason: compression is a denoiser. The corners mp4v discards are
# largely grain that jitters frame to frame and gets thrown out by the track
# error threshold anyway, while lossless faithfully preserves that noise.
# ffv1 also costs ~100x the disk (161 MB vs 1.5 MB for 134 frames at 1080p).
#
# ffv1 is kept as an option for footage where compression is genuinely the
# limit — but don't switch without re-running the A/B on that footage.
# avc1 is deliberately not offered: it measured 120% of clean corners, i.e.
# it invents corners out of ringing, and false features are worse than missing
# ones.
PROXY_CODECS = {"mp4v": ("mp4v", ".mp4"), "ffv1": ("FFV1", ".avi")}

SHOT_EXTS = (".avi", ".mp4", ".mov", ".mkv")


def shot_file_for(shots_dir, shot):
    """Locate a shot's proxy whatever container it was written in.

    split_shots defaults to lossless FFV1 in .avi, but clips split by older
    versions have .mp4 next to them — resolve by search rather than by
    assuming an extension, so both keep working.
    """
    for ext in SHOT_EXTS:
        p = os.path.join(shots_dir, f"shot_{int(shot):02d}{ext}")
        if os.path.exists(p):
            return p
    hits = sorted(glob.glob(os.path.join(shots_dir, f"shot_{int(shot):02d}.*")))
    hits = [h for h in hits if os.path.splitext(h)[1].lower() in SHOT_EXTS]
    return hits[0] if hits else os.path.join(shots_dir,
                                             f"shot_{int(shot):02d}.mp4")


def parse_args():
    p = argparse.ArgumentParser(description="Split an edited video into single-shot clips.")
    p.add_argument("footage")
    p.add_argument("out_dir")
    p.add_argument("--threshold", type=float, default=3.0,
                   help="AdaptiveDetector ratio threshold; cut when a frame changes "
                        "N x more than its neighbors — robust on dark footage (default 3)")
    p.add_argument("--min-len", type=int, default=24,
                   help="Minimum shot length in frames (default 24)")
    p.add_argument("--list-only", action="store_true",
                   help="Only write shots.json, don't write shot videos")
    p.add_argument("--max-height", type=int, default=1080,
                   help="Downscale shot files to this height as tracking "
                        "proxies (default 1080; tracking is resolution-"
                        "independent, the solve applies to the full-res "
                        "plate). 0 = keep source resolution.")
    p.add_argument("--proxy-codec", choices=sorted(PROXY_CODECS), default="mp4v",
                   help="Codec for the shot proxies. 'mp4v' (default) is small "
                        "and, measured end-to-end, solves as well or better "
                        "than lossless — its compression doubles as a "
                        "denoiser. 'ffv1' is lossless and ~100x larger; try it "
                        "only if you suspect compression is limiting a "
                        "specific shot, and compare solve error before "
                        "keeping it. Proxies are disposable — re-split to "
                        "regenerate.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # min_content_val is an absolute floor (default 15) that very dark
    # footage never crosses; drop it so the relative ratio does the work.
    scene_list = detect(args.footage,
                        AdaptiveDetector(adaptive_threshold=args.threshold,
                                         min_scene_len=args.min_len,
                                         min_content_val=3.0))

    cap = cv2.VideoCapture(args.footage)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # proxy size (even dims for codecs)
    if args.max_height and h > args.max_height:
        pw = round(w * args.max_height / h)
        pw -= pw % 2
        ph = args.max_height
    else:
        pw, ph = w, h

    if not scene_list:  # no cuts found: the whole thing is one shot
        ranges = [(1, total)]
    else:
        # scenedetect frame numbers are 0-based, end-exclusive
        ranges = [(s.get_frames() + 1, e.get_frames()) for s, e in scene_list]

    fourcc, ext = PROXY_CODECS[args.proxy_codec]
    shots = []
    for i, (start, end) in enumerate(ranges, 1):
        entry = {"shot": i, "frame_start": start, "frame_end": end,
                 "num_frames": end - start + 1}
        if not args.list_only:
            out_path = os.path.join(args.out_dir, f"shot_{i:02d}{ext}")
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc),
                                     fps, (pw, ph))
            if not writer.isOpened():
                raise RuntimeError(
                    f"cannot encode {args.proxy_codec} ({fourcc}) on this "
                    f"machine — re-run with --proxy-codec mp4v")
            cap.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
            for _ in range(end - start + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                if (pw, ph) != (w, h):
                    frame = cv2.resize(frame, (pw, ph),
                                       interpolation=cv2.INTER_AREA)
                writer.write(frame)
            writer.release()
            entry["file"] = out_path
            print(f"[shots] shot {i:02d}: frames {start}-{end} -> {out_path}")
        else:
            print(f"[shots] shot {i:02d}: frames {start}-{end}")
        shots.append(entry)
    cap.release()

    with open(os.path.join(args.out_dir, "shots.json"), "w") as f:
        json.dump({"footage": os.path.abspath(args.footage), "fps": fps,
                   "size": [w, h], "proxy_size": [pw, ph],
                   "shots": shots}, f, indent=2)
    print(f"[shots] {len(shots)} shots found")


if __name__ == "__main__":
    main()
