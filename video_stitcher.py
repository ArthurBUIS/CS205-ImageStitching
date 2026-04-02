"""
video_stitcher.py
=================
Fixed-camera video stitching pipeline.

Key design decisions:
  - Homography is computed ONCE from the first N calibration frames (since cameras are fixed).
  - Homography is saved to disk so it can be reloaded without recomputing.
  - Per-frame processing is limited to warping + multi-band blending only → fast pipeline.
  - Multi-band (Laplacian pyramid) blending for seamless seams.

Usage
-----
  # Full run (calibrate + stitch):
  python video_stitcher.py --left left.mp4 --right right.mp4 --output stitched.mp4

  # Skip calibration, reuse saved homography:
  python video_stitcher.py --left left.mp4 --right right.mp4 --output stitched.mp4 \
                           --homography homography.npy

  # Tune calibration frames:
  python video_stitcher.py --left left.mp4 --right right.mp4 --output stitched.mp4 \
                           --calib-frames 10
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1.  HOMOGRAPHY CALIBRATION
# ---------------------------------------------------------------------------

def detect_and_match(img1_gray: np.ndarray, img2_gray: np.ndarray,
                     max_features: int = 5000,
                     ratio_thresh: float = 0.75):
    """
    Detect SIFT keypoints and match them with Lowe's ratio test.
    Returns matched keypoints (pts1, pts2) as float32 arrays.
    """
    sift = cv2.SIFT_create(nfeatures=max_features)
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return None, None

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = matcher.knnMatch(des1, des2, k=2)

    good = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio_thresh * n.distance:
                good.append(m)

    if len(good) < 4:
        return None, None

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    return pts1, pts2


def compute_homography_from_frames(cap_left: cv2.VideoCapture,
                                   cap_right: cv2.VideoCapture,
                                   n_frames: int = 5,
                                   ransac_thresh: float = 4.0):
    """
    Read the first n_frames from both videos, compute a homography per frame
    using SIFT + RANSAC, then return the median homography (element-wise).

    The median across frames makes the estimate robust to transient occlusions
    or motion in the first few frames.

    Returns: H (3×3 float64 numpy array) that maps right-frame pixels → left-frame plane.
    """
    print(f"[Calibration] Computing homography from first {n_frames} frames …")
    homographies = []

    for i in range(n_frames):
        ok1, frame1 = cap_left.read()
        ok2, frame2 = cap_right.read()
        if not ok1 or not ok2:
            print(f"  [!] Could not read frame {i} from one of the videos — stopping early.")
            break

        g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        pts1, pts2 = detect_and_match(g1, g2)
        if pts1 is None:
            print(f"  [!] Not enough matches on frame {i}, skipping.")
            continue

        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, ransac_thresh)
        if H is None:
            print(f"  [!] RANSAC failed on frame {i}, skipping.")
            continue

        inliers = int(mask.sum()) if mask is not None else 0
        print(f"  Frame {i}: {inliers}/{len(pts1)} inliers  ✓")
        homographies.append(H)

    if not homographies:
        raise RuntimeError("Could not compute any valid homography from calibration frames.")

    # Element-wise median across all computed homographies
    H_final = np.median(np.stack(homographies, axis=0), axis=0)
    print(f"[Calibration] Done. Used {len(homographies)}/{n_frames} frames.")
    return H_final


# ---------------------------------------------------------------------------
# 2.  CANVAS + WARPING UTILITIES
# ---------------------------------------------------------------------------

def compute_canvas_size(H: np.ndarray, left_shape, right_shape):
    """
    Compute the bounding box of the stitched image and the translation offset
    needed so that no pixel falls at a negative coordinate.

    Returns: (canvas_w, canvas_h, tx, ty)
      - tx, ty: translation to apply to both images so they sit on a positive canvas.
    """
    h1, w1 = left_shape[:2]
    h2, w2 = right_shape[:2]

    # Corners of the right frame, warped into left-frame coordinates
    corners_right = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    warped_corners = cv2.perspectiveTransform(corners_right, H)

    # All corners in left-frame coordinates (right image already in left plane,
    # left image corners are trivially its own rectangle)
    all_corners = np.concatenate([
        np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]]).reshape(-1, 1, 2),
        warped_corners
    ], axis=0)

    x_min, y_min = np.floor(all_corners[:, 0, :].min(axis=0)).astype(int)
    x_max, y_max = np.ceil(all_corners[:, 0, :].max(axis=0)).astype(int)

    tx = int(-x_min) if x_min < 0 else 0
    ty = int(-y_min) if y_min < 0 else 0

    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    return canvas_w, canvas_h, tx, ty


def warp_images(left: np.ndarray, right: np.ndarray,
                H: np.ndarray, canvas_w: int, canvas_h: int,
                tx: int, ty: int):
    """
    Place both images on the shared canvas.

    - `left`  is translated by (tx, ty).
    - `right` is perspective-warped with H then translated by (tx, ty).

    Returns (warped_left, warped_right) — both BGR, same canvas size.
    """
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0,  1]], dtype=np.float64)

    # Left image: simple translation
    warped_left = cv2.warpPerspective(left, T, (canvas_w, canvas_h))

    # Right image: H then translation
    H_translated = T @ H
    warped_right = cv2.warpPerspective(right, H_translated, (canvas_w, canvas_h))

    return warped_left, warped_right


# ---------------------------------------------------------------------------
# 3.  MULTI-BAND BLENDING
# ---------------------------------------------------------------------------

def build_gaussian_pyramid(img: np.ndarray, levels: int):
    gp = [img.astype(np.float32)]
    for _ in range(levels):
        gp.append(cv2.pyrDown(gp[-1]))
    return gp


def build_laplacian_pyramid(img: np.ndarray, levels: int):
    gp = build_gaussian_pyramid(img, levels)
    lp = []
    for i in range(levels):
        up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
        lp.append(gp[i] - up)
    lp.append(gp[levels])  # coarsest level is kept as-is
    return lp


def blend_laplacian_pyramids(lp1, lp2, mask_gp):
    """Blend two Laplacian pyramids using a Gaussian mask pyramid."""
    blended = []
    for l1, l2, gm in zip(lp1, lp2, mask_gp):
        # Ensure mask has 3 channels if images do
        if l1.ndim == 3 and gm.ndim == 2:
            gm = gm[:, :, np.newaxis]
        blended.append(l1 * gm + l2 * (1.0 - gm))
    return blended


def reconstruct_from_laplacian(lp):
    img = lp[-1]
    for level in reversed(lp[:-1]):
        img = cv2.pyrUp(img, dstsize=(level.shape[1], level.shape[0]))
        img = img + level
    return np.clip(img, 0, 255).astype(np.uint8)


def multiband_blend(left_warped: np.ndarray, right_warped: np.ndarray,
                    mask_left: np.ndarray, levels: int = 6):
    """
    Multi-band (Laplacian pyramid) blending.

    mask_left : float32 mask in [0,1], same H×W as the canvas.
                1 = take from left, 0 = take from right, gradient in between.
    """
    # Clamp pyramid levels to what the image size can support
    min_dim = min(left_warped.shape[:2])
    max_levels = int(np.log2(min_dim)) - 1
    levels = min(levels, max_levels)

    lp_left  = build_laplacian_pyramid(left_warped.astype(np.float32),  levels)
    lp_right = build_laplacian_pyramid(right_warped.astype(np.float32), levels)
    gp_mask  = build_gaussian_pyramid(mask_left.astype(np.float32),     levels)
    # add coarsest level to mask gp
    gp_mask.append(cv2.pyrDown(gp_mask[-1]) if len(gp_mask) > 0 else mask_left)

    # Align list lengths
    n = min(len(lp_left), len(lp_right), len(gp_mask))
    blended_lp = blend_laplacian_pyramids(lp_left[:n], lp_right[:n], gp_mask[:n])
    return reconstruct_from_laplacian(blended_lp)


def compute_blend_mask(left_warped: np.ndarray, right_warped: np.ndarray,
                       blend_width: int = 80):
    """
    Build a soft alpha mask for multi-band blending.

    Strategy: find the horizontal seam (vertical line where both images overlap)
    and create a smooth gradient of width `blend_width` pixels around it.
    Pixels only in the left  → mask = 1
    Pixels only in the right → mask = 0
    Overlap region           → smooth gradient 1→0 around the seam
    """
    h, w = left_warped.shape[:2]

    # Binary valid-pixel masks
    left_valid  = (left_warped.sum(axis=2)  > 0).astype(np.float32)
    right_valid = (right_warped.sum(axis=2) > 0).astype(np.float32)
    overlap     = (left_valid * right_valid)

    mask = left_valid.copy()

    # For each row, find the horizontal centre of the overlap band
    for row in range(h):
        cols = np.where(overlap[row] > 0)[0]
        if len(cols) == 0:
            continue
        seam_x = int(cols.mean())
        x0 = max(0, seam_x - blend_width // 2)
        x1 = min(w, seam_x + blend_width // 2)
        ramp = np.linspace(1.0, 0.0, x1 - x0)
        mask[row, x0:x1] = ramp
        mask[row, x1:]   = 0.0  # right side of seam → right image

    # Pixels only in right image: mask = 0 (already)
    # Pixels only in left image: keep mask = 1
    mask[right_valid == 0] = 1.0
    mask[left_valid  == 0] = 0.0

    return mask


# ---------------------------------------------------------------------------
# 4.  MAIN PIPELINE
# ---------------------------------------------------------------------------

def stitch_videos(left_path: str, right_path: str, output_path: str,
                  calib_frames: int = 5,
                  homography_path: str = None,
                  save_homography: str = "homography.npy",
                  blend_levels: int = 6,
                  blend_width: int = 80):

    cap_left  = cv2.VideoCapture(left_path)
    cap_right = cv2.VideoCapture(right_path)

    if not cap_left.isOpened():
        raise IOError(f"Cannot open left video: {left_path}")
    if not cap_right.isOpened():
        raise IOError(f"Cannot open right video: {right_path}")

    fps    = cap_left.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(min(cap_left.get(cv2.CAP_PROP_FRAME_COUNT),
                     cap_right.get(cv2.CAP_PROP_FRAME_COUNT)))

    # ---- Read one frame to know dimensions --------------------------------
    ok1, sample_left  = cap_left.read()
    ok2, sample_right = cap_right.read()
    if not ok1 or not ok2:
        raise IOError("Could not read the first frame from one of the videos.")

    cap_left.set(cv2.CAP_PROP_POS_FRAMES,  0)
    cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ---- Homography -------------------------------------------------------
    if homography_path and Path(homography_path).exists():
        H = np.load(homography_path)
        print(f"[Homography] Loaded from {homography_path}")
    else:
        H = compute_homography_from_frames(cap_left, cap_right, n_frames=calib_frames)
        np.save(save_homography, H)
        print(f"[Homography] Saved to {save_homography}")
        # Reset to start for the stitching pass
        cap_left.set(cv2.CAP_PROP_POS_FRAMES,  0)
        cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"[Homography]\n{H}")

    # ---- Canvas geometry (computed ONCE) ----------------------------------
    canvas_w, canvas_h, tx, ty = compute_canvas_size(H, sample_left.shape, sample_right.shape)
    print(f"[Canvas] {canvas_w}×{canvas_h}  offset=({tx},{ty})")

    # ---- VideoWriter -------------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (canvas_w, canvas_h))
    if not writer.isOpened():
        raise IOError(f"Cannot open VideoWriter for: {output_path}")

    # ---- Blend mask (computed ONCE on first frame) ------------------------
    print("[Blend mask] Computing from first frame …")
    left0_w, right0_w = warp_images(sample_left, sample_right, H, canvas_w, canvas_h, tx, ty)
    blend_mask = compute_blend_mask(left0_w, right0_w, blend_width=blend_width)
    print("[Blend mask] Done.")

    # ---- Frame loop --------------------------------------------------------
    print(f"[Stitching] Processing {total} frames …")
    t0 = time.time()
    frame_idx = 0

    while True:
        ok1, frame_left  = cap_left.read()
        ok2, frame_right = cap_right.read()
        if not ok1 or not ok2:
            break

        # Warp both frames onto the shared canvas
        wl, wr = warp_images(frame_left, frame_right, H, canvas_w, canvas_h, tx, ty)

        # Multi-band blend
        stitched = multiband_blend(wl, wr, blend_mask, levels=blend_levels)

        writer.write(stitched)
        frame_idx += 1

        if frame_idx % 30 == 0:
            elapsed = time.time() - t0
            fps_actual = frame_idx / elapsed
            print(f"  {frame_idx}/{total}  ({fps_actual:.1f} fps)", end="\r")

    elapsed = time.time() - t0
    print(f"\n[Done] {frame_idx} frames in {elapsed:.1f}s  "
          f"({frame_idx/elapsed:.1f} fps avg)")

    cap_left.release()
    cap_right.release()
    writer.release()
    print(f"[Output] Saved to {output_path}")


# ---------------------------------------------------------------------------
# 5.  CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Fixed-camera video stitcher with one-time homography calibration.")
    p.add_argument("--left",         required=True,  help="Path to left video")
    p.add_argument("--right",        required=True,  help="Path to right video")
    p.add_argument("--output",       required=True,  help="Path to output video (.mp4)")
    p.add_argument("--homography",   default=None,
                   help="Path to a pre-saved homography .npy file (skips calibration)")
    p.add_argument("--save-homography", default="homography.npy",
                   help="Where to save the computed homography (default: homography.npy)")
    p.add_argument("--calib-frames", type=int, default=5,
                   help="Number of frames used for homography calibration (default: 5)")
    p.add_argument("--blend-levels", type=int, default=6,
                   help="Laplacian pyramid levels for multi-band blending (default: 6)")
    p.add_argument("--blend-width",  type=int, default=80,
                   help="Width in pixels of the blending gradient zone (default: 80)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    stitch_videos(
        left_path        = args.left,
        right_path       = args.right,
        output_path      = args.output,
        calib_frames     = args.calib_frames,
        homography_path  = args.homography,
        save_homography  = args.save_homography,
        blend_levels     = args.blend_levels,
        blend_width      = args.blend_width,
    )