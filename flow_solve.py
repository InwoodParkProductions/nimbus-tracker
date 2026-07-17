"""
Stage 2c: 2D flow solve — last-resort camera motion from optical flow.
----------------------------------------------------------------------
For shots Blender's 3D solver can't reconstruct (motion-blurred whips,
close-ups against featureless cloth): measures global image motion with
masked LK optical flow, fits a per-frame homography, and converts it to a
camera path — pan/tilt/roll from the rotation, plus a forward/back dolly
from the measured zoom so push-ins/pull-outs are followed. Approximate by
design — right for comping on blurred/handheld shots, wrong for locking CG
to the floor.

Run with the SYSTEM python:
    python flow_solve.py <shot.mp4> <out_json> [--masks <masks_dir>] [--focal-mm 35]

Output JSON: per-frame camera quaternions (Blender convention, camera at
identity looks down -Z), focal length, fps, size, and a quality residual
(median px error of the rotation model vs measured flow, at source scale).
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

SENSOR_W_MM = 36.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("footage")
    p.add_argument("out_json")
    p.add_argument("--masks", help="mask_NNNNNN.png dir (white=person=ignore)")
    p.add_argument("--focal-mm", type=float, default=35.0,
                   help="assumed focal length (full-frame mm, default 35)")
    p.add_argument("--width", type=int, default=768,
                   help="processing width (default 768)")
    return p.parse_args()


def load_mask(masks_dir, frame_no, shape):
    if not masks_dir:
        return None
    path = os.path.join(masks_dir, f"mask_{frame_no:06d}.png")
    if not os.path.exists(path):
        return None
    m = cv2.imread(path, 0)
    m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return m


def rot_from_homography(H, K, Kinv):
    """Nearest rotation matrix to K^-1 H K (rotation-only camera model)."""
    M = Kinv @ H @ K
    M /= np.cbrt(np.linalg.det(M)) if np.linalg.det(M) > 0 else 1.0
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, 1, -1]) @ Vt
    return R


def cv_to_blender(R_cv):
    """OpenCV camera (x right, y down, z forward) -> Blender camera
    (x right, y up, z backward)."""
    C = np.diag([1.0, -1.0, -1.0])
    return C @ R_cv @ C


def mat_to_quat(R):
    """Rotation matrix -> quaternion (w, x, y, z)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                         (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(R[i, i] - R[j, j] - R[k, k] + 1.0) * 2
    q = np.empty(4)
    q[0] = (R[k, j] - R[j, k]) / s
    q[1 + i] = 0.25 * s
    q[1 + j] = (R[j, i] + R[i, j]) / s
    q[1 + k] = (R[k, i] + R[i, k]) / s
    return q


def main():
    args = parse_args()
    cap = cv2.VideoCapture(args.footage)
    if not cap.isOpened():
        sys.exit(f"cannot open {args.footage}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W = args.width
    Hh = round(src_h * W / src_w)

    f_px = args.focal_mm / SENSOR_W_MM * W
    K = np.array([[f_px, 0, W / 2.0], [0, f_px, Hh / 2.0], [0, 0, 1]])
    Kinv = np.linalg.inv(K)

    R_cum = np.eye(3)
    quats = [mat_to_quat(np.eye(3))]
    # Rotation can't represent a dolly/zoom, so we also track how much the
    # image scales frame-to-frame (a push-in makes the world grow) and turn
    # the cumulative scale into a forward/back camera translation downstream.
    S_cum = 1.0
    scales = [1.0]
    residuals = []
    prev = None
    prev_mask = None
    frame_no = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        g = cv2.cvtColor(cv2.resize(frame, (W, Hh)), cv2.COLOR_BGR2GRAY)
        # CLAHE lifts detail out of dark/soft-gradient footage (cloth folds)
        g = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(g)
        mask = load_mask(args.masks, frame_no, g.shape)
        bg = None if mask is None else cv2.bitwise_not(mask)
        if prev is not None:
            pts = cv2.goodFeaturesToTrack(prev, maxCorners=400,
                                          qualityLevel=0.005, minDistance=8,
                                          mask=prev_mask)
            if pts is None or len(pts) < 12:
                # nothing outside the person mask — fall back to whole frame
                pts = cv2.goodFeaturesToTrack(prev, maxCorners=400,
                                              qualityLevel=0.005, minDistance=8)
            H_delta = None
            if pts is not None and len(pts) >= 12:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(
                    prev, g, pts, None, winSize=(31, 31), maxLevel=4)
                good = st.ravel() == 1
                p0, p1 = pts[good], nxt[good]
                if len(p0) >= 12:
                    H_delta, inl = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
                    if H_delta is not None and inl is not None and inl.sum() >= 8:
                        R_d = rot_from_homography(H_delta, K, Kinv)
                        # residual: rotation-model reprojection vs measured flow
                        M = K @ R_d @ Kinv
                        proj = cv2.perspectiveTransform(
                            p0.reshape(-1, 1, 2).astype(np.float64), M)
                        res = np.linalg.norm(
                            proj.reshape(-1, 2) - p1.reshape(-1, 2), axis=1)
                        residuals.append(float(np.median(res)) * src_w / W)
                        R_cum = R_cum @ R_d.T  # camera rotation is inverse of image motion
                        # frame-to-frame zoom = scale of the similarity fit
                        aff, _ = cv2.estimateAffinePartial2D(
                            p0, p1, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                        if aff is not None:
                            s_d = float(np.hypot(aff[0, 0], aff[1, 0]))
                            s_d = min(max(s_d, 0.90), 1.11)  # cap runaway
                            S_cum = min(max(S_cum * s_d, 0.4), 2.5)
                    else:
                        H_delta = None
            if H_delta is None:
                residuals.append(None)  # hold previous orientation and scale
            quats.append(mat_to_quat(cv_to_blender(R_cum)))
            scales.append(S_cum)
        prev, prev_mask = g, bg
    cap.release()

    valid = [r for r in residuals if r is not None]
    out = {
        "solver": "2d-flow-rot-dolly",
        "num_frames": frame_no,
        "fps": fps,
        "size": [src_w, src_h],
        "focal_mm": args.focal_mm,
        "sensor_width_mm": SENSOR_W_MM,
        "quaternions_wxyz": [list(map(float, q)) for q in quats],
        "scale_cum": [float(s) for s in scales],  # per-frame zoom -> dolly
        "median_residual_px": (float(np.median(valid)) if valid else None),
        "frames_without_flow": sum(1 for r in residuals if r is None),
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f)
    print(f"[flow] {frame_no} frames, median residual "
          f"{out['median_residual_px']} px (at source res), "
          f"{out['frames_without_flow']} frames held")


if __name__ == "__main__":
    main()
