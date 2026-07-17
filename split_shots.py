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
import json
import os

import cv2
from scenedetect import detect, AdaptiveDetector


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

    shots = []
    for i, (start, end) in enumerate(ranges, 1):
        entry = {"shot": i, "frame_start": start, "frame_end": end,
                 "num_frames": end - start + 1}
        if not args.list_only:
            out_path = os.path.join(args.out_dir, f"shot_{i:02d}.mp4")
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (pw, ph))
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
