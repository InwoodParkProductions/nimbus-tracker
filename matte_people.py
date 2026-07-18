"""Stage 1b: soft person mattes for compositing (alpha, not silhouettes).

Run with the SYSTEM python:
    python matte_people.py <shot.mp4|mov> <out_dir> [--frames A-B]
        [--downsample 0.25] [--device cuda]

SAM2's binary masks are the right tool for track exclusion and the wrong tool
for the final comp: hard edges clip hair and motion blur, and growing the mask
to protect them drags a halo of the original backdrop into the composite.
This produces per-frame ALPHA mattes (alpha_NNNNNN.png, 0-255 soft) with
RobustVideoMatting, a recurrent model built specifically for people — it
carries temporal state across frames, so edges don't sizzle frame to frame.

Weights download via torch.hub on first use (~15MB).
LICENSE NOTE: RobustVideoMatting is GPL-3.0. Fine for in-house use on your own
footage; talk to a lawyer before bundling it into a commercial build.
"""

import argparse
import os
import sys

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("footage")
    p.add_argument("out_dir")
    p.add_argument("--frames", help="source frame range A-B (1-based "
                                    "inclusive); default = whole file")
    p.add_argument("--model", default="resnet50",
                   choices=["resnet50", "mobilenetv3"],
                   help="RVM backbone (default resnet50 — noticeably "
                        "cleaner edges on hair/fabric than mobilenetv3, "
                        "at ~100MB vs ~15MB and a bit slower)")
    p.add_argument("--downsample", type=float, default=0.25,
                   help="model's internal downsample ratio (default 0.25 — "
                        "the author-recommended value for 4K input)")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    a = parse_args()
    import torch

    dev = a.device if (a.device != "cuda" or torch.cuda.device_count()) else "cpu"
    model = torch.hub.load("PeterL1n/RobustVideoMatting", a.model,
                           trust_repo=True).to(dev).eval()

    cap = cv2.VideoCapture(a.footage)
    if not cap.isOpened():
        sys.exit(f"cannot open {a.footage}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if a.frames:
        f0, f1 = (int(v) for v in a.frames.split("-"))
    else:
        f0, f1 = 1, total
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0 - 1)
    os.makedirs(a.out_dir, exist_ok=True)

    rec = [None] * 4          # the model's recurrent state, carried per frame
    n = 0
    with torch.no_grad():
        for src_f in range(f0, f1 + 1):
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            src = torch.from_numpy(rgb).to(dev).permute(2, 0, 1)[None] \
                .float().div(255.0)
            _, pha, *rec = model(src, *rec, downsample_ratio=a.downsample)
            alpha = (pha[0, 0].clamp(0, 1) * 255).byte().cpu().numpy()
            rel = src_f - f0 + 1
            cv2.imwrite(os.path.join(a.out_dir, f"alpha_{rel:06d}.png"), alpha)
            n += 1
    cap.release()
    print(f"[matte] {n} alpha mattes -> {a.out_dir}")
    if n == 0:
        sys.exit("no frames matted")


if __name__ == "__main__":
    main()
