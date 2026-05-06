"""
Camera intrinsic calibration from a chessboard video.

Usage:
    python calibrate.py --video calib.mp4 --rows 6 --cols 9 --square_size 0.025 \
                        --output K.npz

Instructions:
    1. Print a chessboard pattern. A common one is 7x10 squares -> 6x9 INNER
       corners (the value to pass as --rows --cols). Measure one square in
       meters and pass it as --square_size (e.g. 0.025 for 25 mm squares).
       If you can't be bothered measuring, leave it at the default; only the
       focal length matters for our use, and that's invariant to square size.
    2. Record a 30-60 second video of the board, holding it in front of the
       camera at varied positions and orientations: tilted left, tilted right,
       tilted up/down, close, far, near each corner of the frame. Keep the
       whole board visible.
    3. Run this script. It samples frames from the video, detects the corners,
       and runs cv2.calibrateCamera. It saves K (3x3 intrinsics) and dist
       (distortion coefficients) to the output .npz.
    4. Repeat for the second camera and save to a different file.

Tips:
    - More variety in board pose = better calibration. Static board = useless.
    - If detection rate is low, increase --sample_every or check lighting.
    - Reprojection error under ~0.5 pixels is good. Above 1.0 means redo it.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


def calibrate(video_path: Path, rows: int, cols: int, square_size: float,
              sample_every: int, max_views: int, output_path: Path,
              show: bool) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pattern_size = (cols, rows)  # OpenCV expects (cols, rows)

    # 3D object points for one board (z=0 plane).
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size

    obj_points: list[np.ndarray] = []
    img_points: list[np.ndarray] = []
    img_size: tuple[int, int] | None = None

    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH +
             cv2.CALIB_CB_NORMALIZE_IMAGE +
             cv2.CALIB_CB_FAST_CHECK)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

    pbar = tqdm(total=n_frames, desc="scanning")
    idx = 0
    accepted = 0
    while accepted < max_views:
        ok, frame = cap.read()
        if not ok:
            break
        pbar.update(1)
        if idx % sample_every != 0:
            idx += 1
            continue
        idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gray.shape[::-1]  # (w, h)

        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags=flags)
        if not found:
            continue

        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp.copy())
        img_points.append(corners)
        accepted += 1

        if show:
            disp = frame.copy()
            cv2.drawChessboardCorners(disp, pattern_size, corners, found)
            cv2.imshow("calib", disp)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    pbar.close()
    cap.release()
    if show:
        cv2.destroyAllWindows()

    if accepted < 10:
        raise RuntimeError(f"Only {accepted} good views found. Need >=10. "
                           "Record a longer / more varied calibration video.")

    print(f"[calib] running calibrateCamera on {accepted} views...")
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None
    )
    print(f"[calib] RMS reprojection error: {rms:.4f} pixels")
    print(f"[calib] image size: {img_size}")
    print(f"[calib] K = \n{K}")
    print(f"[calib] dist = {dist.ravel()}")
    fov_x = 2 * np.degrees(np.arctan2(img_size[0] / 2, K[0, 0]))
    fov_y = 2 * np.degrees(np.arctan2(img_size[1] / 2, K[1, 1]))
    print(f"[calib] horizontal FOV: {fov_x:.1f} deg, vertical FOV: {fov_y:.1f} deg")

    np.savez(output_path, K=K, dist=dist, image_size=np.array(img_size),
             rms=rms)
    print(f"[calib] saved to {output_path}")
    if rms > 1.0:
        print("[warn] RMS > 1.0 px. Calibration is mediocre; consider redoing.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, type=Path)
    p.add_argument("--rows", type=int, default=6,
                   help="Inner corner rows of the chessboard.")
    p.add_argument("--cols", type=int, default=9,
                   help="Inner corner cols of the chessboard.")
    p.add_argument("--square_size", type=float, default=0.025,
                   help="Square edge length in meters (only affects scale, "
                        "not the focal length).")
    p.add_argument("--sample_every", type=int, default=10,
                   help="Take 1 frame every N for detection.")
    p.add_argument("--max_views", type=int, default=40,
                   help="Stop after this many successful detections.")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--show", action="store_true",
                   help="Display the detected corners (slower).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    calibrate(args.video, args.rows, args.cols, args.square_size,
              args.sample_every, args.max_views, args.output, args.show)


if __name__ == "__main__":
    main()
