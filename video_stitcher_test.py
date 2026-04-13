"""
video_stitcher.py
=================
Fixed-camera video stitching pipeline.

Key design decisions:
  - Homography is computed ONCE from the first N calibration frames (since cameras are fixed).
  - Homography estimation uses MAGSAC++ (cv2.USAC_MAGSAC) instead of vanilla RANSAC.
    MAGSAC++ uses a marginalised score over a noise distribution rather than a hard inlier
    threshold, producing more accurate homographies — especially with noisy or low-overlap
    matches.  The `sigma_max` parameter is an upper bound on the noise scale (in pixels),
    not a hard cutoff as in RANSAC.
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

  # Tune calibration frames and MAGSAC++ sigma:
  python video_stitcher.py --left left.mp4 --right right.mp4 --output stitched.mp4 \
                           --calib-frames 10 --sigma-max 2.0
"""

import argparse
import queue
import sys
import time
import threading
from pathlib import Path

import cv2
import numpy as np

# Import optimization class
sys.path.insert(0, str(Path(__file__).parent))
from optimization import ConvexPolygonMaxRectangle


# ---------------------------------------------------------------------------
# 1.  HOMOGRAPHY CALIBRATION
# ---------------------------------------------------------------------------

_XFEAT_MODEL = None
_XFEAT_DEVICE = None


def _load_xfeat_model(device: str = "cpu"):
    """Return a lazily created XFeat model from Torch Hub."""
    global _XFEAT_MODEL, _XFEAT_DEVICE

    if _XFEAT_MODEL is not None and _XFEAT_DEVICE == device:
        return _XFEAT_MODEL

    import torch

    model = torch.hub.load(
        "verlab/accelerated_features",
        "XFeat",
        pretrained=True,
        top_k=5000,
    ).to(torch.device(device)).eval()
    _XFEAT_MODEL = model
    _XFEAT_DEVICE = device
    return model

def detect_and_match(img1_gray: np.ndarray, img2_gray: np.ndarray,
                     max_features: int = 5000,
                     ratio_thresh: float = 0.75,
                     device: str = "cpu"):
    """
    Detect XFeat keypoints and match them directly.
    Returns matched keypoints (pts1, pts2) as float32 arrays.
    """
    import torch

    def to_tensor(img):
        rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(torch.device(device))

    model = _load_xfeat_model(device=device)

    with torch.no_grad():
        out1 = model.detectAndCompute(to_tensor(img1_gray), top_k=max_features)[0]
        out2 = model.detectAndCompute(to_tensor(img2_gray), top_k=max_features)[0]
        idxs0, idxs1 = model.match(out1["descriptors"], out2["descriptors"], min_cossim=-1)

    if idxs0 is None or idxs1 is None or len(idxs0) < 4:
        return None, None

    pts1 = out1["keypoints"][idxs0].cpu().numpy().astype(np.float32)
    pts2 = out2["keypoints"][idxs1].cpu().numpy().astype(np.float32)
    return pts1, pts2


def compute_homography_from_frames(cap_left: cv2.VideoCapture,
                                   cap_right: cv2.VideoCapture,
                                   n_frames: int = 5,
                                   sigma_max: float = 4.0):
    """
    Read the first n_frames from both videos, compute a homography per frame
    using XFeat + MAGSAC++, then return the median homography (element-wise).

    MAGSAC++ (cv2.USAC_MAGSAC) differences vs RANSAC
    --------------------------------------------------
    - Instead of a hard inlier/outlier threshold, it marginalises the fitting
      score over a distribution of noise scales up to `sigma_max` (pixels).
        - This makes it significantly more accurate when matches have heterogeneous
            noise and more robust in
      low-overlap or low-inlier-ratio situations — both common during video
      stitching calibration.
    - `sigma_max` is an upper bound on the expected noise standard deviation,
      NOT a binary threshold. A value of 1–4 px is appropriate for 1080p footage
      from fixed, reasonably calibrated cameras. Increase to 6–8 px for wider
      lenses or more lens distortion.

    The median across frames makes the estimate robust to transient occlusions
    or motion in the first few frames.

    Returns: H (3×3 float64 numpy array) that maps right-frame pixels → left-frame plane.
    """
    print(f"[Calibration] Computing homography from first {n_frames} frames (MAGSAC++) …")
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

        # MAGSAC++: marginalised score, sigma_max is noise upper bound (not a hard threshold)
        H, mask = cv2.findHomography(pts2, pts1, cv2.USAC_MAGSAC, sigma_max)
        if H is None:
            print(f"  [!] MAGSAC++ failed on frame {i}, skipping.")
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


def prepare_remap_grids(H: np.ndarray, canvas_w: int, canvas_h: int, tx: int, ty: int):
    """Precompute dense source-coordinate maps for cv2.remap()."""
    grid_x, grid_y = np.meshgrid(
        np.arange(canvas_w, dtype=np.float32),
        np.arange(canvas_h, dtype=np.float32),
    )

    # Left image is a pure translation onto the canvas.
    left_map_x = grid_x - float(tx)
    left_map_y = grid_y - float(ty)

    # Right image uses the inverse of the translated homography.
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0,  1]], dtype=np.float64)
    H_translated = T @ H
    H_inv = np.linalg.inv(H_translated)

    canvas_points = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1).reshape(-1, 1, 2)
    right_src = cv2.perspectiveTransform(canvas_points, H_inv).reshape(canvas_h, canvas_w, 2)
    right_map_x = right_src[:, :, 0].astype(np.float32)
    right_map_y = right_src[:, :, 1].astype(np.float32)

    return left_map_x, left_map_y, right_map_x, right_map_y


def remap_images(left: np.ndarray, right: np.ndarray,
                 left_map_x: np.ndarray, left_map_y: np.ndarray,
                 right_map_x: np.ndarray, right_map_y: np.ndarray):
    """Warp both frames using precomputed remap grids."""
    warped_left = cv2.remap(
        left, left_map_x, left_map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    warped_right = cv2.remap(
        right, right_map_x, right_map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
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


def line_segment_intersection_with_vertical(p1, p2, x):
    """
    Find intersection of line segment (p1, p2) with vertical line at x=const.
    Returns the intersection point or None if no intersection.
    """
    x1, y1 = p1
    x2, y2 = p2

    # Check if segment crosses or touches the vertical line
    if (x1 - x) * (x2 - x) > 1e-9:  # Both strictly on same side
        return None

    # Segment is vertical at a different x
    if abs(x2 - x1) < 1e-9:
        return None

    # Linear interpolation to find y
    t = (x - x1) / (x2 - x1)
    y = y1 + t * (y2 - y1)
    return np.array([x, y])


def line_segment_intersection_with_horizontal(p1, p2, y):
    """
    Find intersection of line segment (p1, p2) with horizontal line at y=const.
    Returns the intersection point or None if no intersection.
    """
    x1, y1 = p1
    x2, y2 = p2

    # Check if segment crosses or touches the horizontal line
    if (y1 - y) * (y2 - y) > 1e-9:  # Both strictly on same side
        return None

    # Segment is horizontal at a different y
    if abs(y2 - y1) < 1e-9:
        return None

    # Linear interpolation to find x
    t = (y - y1) / (y2 - y1)
    x = x1 + t * (x2 - x1)
    return np.array([x, y])


def find_crop_rectangle_handcrafted(H: np.ndarray, left_shape, right_shape,
                                     canvas_w: int, canvas_h: int, tx: int, ty: int):
    """
    Simple handcrafted approach to find crop rectangle.

    Algorithm:
    1. Get left image corners on canvas (defines y_min, y_max)
    2. Draw horizontal lines at y_min and y_max
    3. Find where these lines intersect the convex hull edges
    4. Find the intersection with smallest x, constrained to x > x_left (right edge of left image)
    5. Use that x as x_max, crop region is (0, y_min, x_max, y_max - y_min)

    Returns: (x, y, width, height) in canvas coordinates
    """
    h1, w1 = left_shape[:2]
    h2, w2 = right_shape[:2]

    # Get left image corners in canvas coordinates
    left_corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
    left_canvas = left_corners + np.array([tx, ty])

    # Get y bounds from left image
    y_min = int(left_canvas[:, 1].min())
    y_max = int(left_canvas[:, 1].max())

    # Get x bound: right edge of left image on canvas
    x_left = int(left_canvas[:, 0].max())

    # Get right image corners transformed
    right_corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    right_warped = cv2.perspectiveTransform(right_corners, H).reshape(-1, 2)
    right_canvas = right_warped + np.array([tx, ty])

    # Compute convex hull of all corners
    all_corners = np.vstack([left_canvas, right_canvas]).astype(np.float32)
    hull = cv2.convexHull(all_corners)
    hull_points = hull.reshape(-1, 2).astype(np.float64)

    # Find intersections of horizontal lines (y_min and y_max) with hull edges
    intersections = []

    for y_line in [y_min, y_max]:
        for i in range(len(hull_points)):
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % len(hull_points)]

            # Find intersection with horizontal line at y_line
            intersection = line_segment_intersection_with_horizontal(p1, p2, y_line)
            if intersection is not None:
                intersections.append(intersection)

    if len(intersections) == 0:
        # Fallback: use canvas dimensions
        return 0, y_min, canvas_w, y_max - y_min

    # Find the intersection with smallest x, constrained to x > x_left
    intersections = np.array(intersections)
    valid_intersections = intersections[intersections[:, 0] > x_left]

    if len(valid_intersections) == 0:
        # No valid intersections beyond left image, fallback to right edge of canvas
        return 0, y_min, canvas_w, y_max - y_min

    x_max = int(valid_intersections[:, 0].min())  # Smallest valid x

    # Ensure x_max is within bounds
    x_max = max(x_left + 1, min(x_max, canvas_w))

    return 0, y_min, x_max, y_max - y_min


def find_crop_rectangle_optimization(H: np.ndarray, left_shape, right_shape,
                                      canvas_w: int, canvas_h: int, tx: int, ty: int):
    """
    Optimization-based approach: find the largest axis-aligned rectangle that fits
    entirely within the convex hull of the image corners.

    Uses ConvexPolygonMaxRectangle to search over all possible x-ranges and find
    the maximum area rectangle.

    Returns: (x, y, width, height) in canvas coordinates
    """
    h1, w1 = left_shape[:2]
    h2, w2 = right_shape[:2]

    # Get left image corners in canvas coordinates
    left_corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
    left_canvas = left_corners + np.array([tx, ty])

    # Get right image corners transformed
    right_corners = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]]).reshape(-1, 1, 2)
    right_warped = cv2.perspectiveTransform(right_corners, H).reshape(-1, 2)
    right_canvas = right_warped + np.array([tx, ty])

    # Compute convex hull of all corners
    all_corners = np.vstack([left_canvas, right_canvas]).astype(np.float32)
    hull = cv2.convexHull(all_corners)
    hull_points = hull.reshape(-1, 2)

    # Convert to list of tuples for the optimizer
    vertices = [tuple(pt) for pt in hull_points]

    # Run optimization
    optimizer = ConvexPolygonMaxRectangle(vertices)

    # Extended algorithm to track coordinates
    xs = optimizer.xs
    upper = optimizer.upper
    lower = optimizer.lower
    upper_x = [p[0] for p in upper]
    lower_x = [p[0] for p in lower]

    best_area = 0.0
    best_rect = (0, 0, 1, 1)

    for i in range(len(xs)):
        x1 = xs[i]

        top_min = float('inf')
        bot_max = -float('inf')
        y_top_at_x1 = None
        y_bot_at_x1 = None

        # Reset pointers for chains
        up_ptr = 0
        low_ptr = 0

        # Advance pointers to x1
        while up_ptr + 1 < len(upper) and upper_x[up_ptr + 1] < x1:
            up_ptr += 1
        while low_ptr + 1 < len(lower) and lower_x[low_ptr + 1] < x1:
            low_ptr += 1

        for j in range(i + 1, len(xs)):
            x2 = xs[j]

            # Evaluate upper chain at x2
            while up_ptr + 1 < len(upper) and upper_x[up_ptr + 1] < x2:
                up_ptr += 1
            y_top_x2 = optimizer._interp(upper[up_ptr], upper[min(up_ptr + 1, len(upper) - 1)], x2)

            # Evaluate lower chain at x2
            while low_ptr + 1 < len(lower) and lower_x[low_ptr + 1] < x2:
                low_ptr += 1
            y_bot_x2 = optimizer._interp(lower[low_ptr], lower[min(low_ptr + 1, len(lower) - 1)], x2)

            # Update constraints
            top_min = min(top_min, y_top_x2)
            bot_max = max(bot_max, y_bot_x2)

            height = top_min - bot_max
            if height > 0:
                area = height * (x2 - x1)
                if area > best_area:
                    best_area = area
                    best_rect = (int(x1), int(bot_max), int(x2 - x1), int(height))

    if best_area == 0:
        # Fallback: use handcrafted approach
        return find_crop_rectangle_handcrafted(H, left_shape, right_shape, canvas_w, canvas_h, tx, ty)

    return best_rect


def find_crop_rectangle_from_corners(H: np.ndarray, left_shape, right_shape,
                                      canvas_w: int, canvas_h: int, tx: int, ty: int,
                                      method: str = "handcrafted"):
    """
    Find crop rectangle using selected method.

    method: "handcrafted" (simple) or "optimization" (placeholder for future)
    """
    if method == "handcrafted":
        return find_crop_rectangle_handcrafted(H, left_shape, right_shape, canvas_w, canvas_h, tx, ty)
    elif method == "optimization":
        return find_crop_rectangle_optimization(H, left_shape, right_shape, canvas_w, canvas_h, tx, ty)
    else:
        raise ValueError(f"Unknown auto-crop method: {method}")




def compute_zone_boundaries(left_warped: np.ndarray,
                             right_warped: np.ndarray) -> tuple[int, int]:
    """
    Compute the two seam-line x-positions that define the three zones.

    Zone 1  [0,          seam_left)  : left image only
    Zone 2  [seam_left,  seam_right) : overlap — both images present
    Zone 3  [seam_right, canvas_w)   : right image only

    seam_left  = first canvas column where the right image appears
                 (left frontier of the overlap = boundary between zone 1/2)
    seam_right = last  canvas column where the left  image appears + 1
                 (right frontier of the overlap = boundary between zone 2/3)

    Returns (seam_left, seam_right) as integer column indices.
    """
    # Per-row first/last valid column for each image
    left_valid  = left_warped.sum(axis=2)  > 0   # (H, W) bool
    right_valid = right_warped.sum(axis=2) > 0

    # Collapse to per-column presence
    left_cols  = np.where(left_valid.any(axis=0))[0]
    right_cols = np.where(right_valid.any(axis=0))[0]

    seam_left  = int(right_cols[0])               # right image starts here
    seam_right = int(left_cols[-1]) + 1           # left image ends here (+1 for slice)
    return seam_left, seam_right


def build_blend_mask(seam_x: int,
                     left_warped: np.ndarray,
                     right_warped: np.ndarray,
                     blend_width: int = 80) -> np.ndarray:
    """
    Build a float32 blend mask (H, W) with values in [0, 1].
    1.0 = take from left image, 0.0 = take from right image.

    The seam is placed at `seam_x`.  A gradient ramp of `blend_width`
    pixels is applied symmetrically around the seam so the cut is smooth.
    Outside the overlap zone the mask is clamped hard to 1 or 0.
    """
    h, w = left_warped.shape[:2]
    left_valid  = (left_warped.sum(axis=2)  > 0).astype(np.float32)
    right_valid = (right_warped.sum(axis=2) > 0).astype(np.float32)

    mask = np.zeros((h, w), dtype=np.float32)

    half = blend_width // 2
    x0 = max(0,     seam_x - half)
    x1 = min(w - 1, seam_x + half)
    n  = x1 - x0

    mask[:, :x0]    = 1.0
    if n > 0:
        ramp = np.linspace(1.0, 0.0, n, dtype=np.float32)
        mask[:, x0:x1] = ramp[np.newaxis, :]   # broadcast over rows
    # mask[:, x1:] stays 0.0

    # Hard constraints outside the overlap zone
    mask[right_valid == 0] = 1.0
    mask[left_valid  == 0] = 0.0

    return mask


# ---------------------------------------------------------------------------
# 3b.  MOTION DETECTION
# ---------------------------------------------------------------------------

def update_background(bg_mean: np.ndarray,
                       gray: np.ndarray,
                       alpha: float = 0.05) -> np.ndarray:
    """Exponential moving average background update. Returns new bg_mean."""
    return bg_mean * (1.0 - alpha) + gray * alpha


def detect_motion(gray: np.ndarray,
                  bg_mean: np.ndarray,
                  threshold: int = 25) -> np.ndarray:
    """
    Return a boolean mask (H, W) — True where motion is detected.
    Uses absolute difference from the background model + morphological
    cleanup to remove noise.
    """
    diff = np.abs(gray.astype(np.float32) - bg_mean)
    fg   = (diff > threshold).astype(np.uint8)

    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,   kernel)
    fg = cv2.morphologyEx(fg, cv2.MORPH_DILATE, kernel)
    return fg.astype(bool)


def choose_seam(motion_mask: np.ndarray,
                prev_gray:   np.ndarray,
                curr_gray:   np.ndarray,
                seam_left:   int,
                seam_right:  int,
                canvas_w:    int) -> int:
    """
    Given a motion mask and the two fixed seam positions, decide which
    seam x-coordinate to use this frame.

    Rules (from the paper / user spec):
      - No motion detected          -> keep current seam unchanged (caller handles)
      - Object in zone 1 (left-only)-> use seam_right  (push seam as far right as possible
                                        so the object stays entirely in the left image)
      - Object in zone 3 (right-only)-> use seam_left  (push seam as far left as possible
                                        so the object stays entirely in the right image)
      - Object in zone 2 (overlap)  -> determine direction of motion:
            moving left  -> use seam_right
            moving right -> use seam_left

    Returns the chosen seam x-position (seam_left or seam_right).
    """
    if not motion_mask.any():
        return None   # signal: no change needed

    # Find bounding-box centroid of the motion region
    ys, xs = np.where(motion_mask)
    cx = int(xs.mean())

    in_zone1 = cx < seam_left
    in_zone3 = cx >= seam_right

    if in_zone1:
        return seam_right

    if in_zone3:
        return seam_left

    # Zone 2 — determine horizontal motion direction by comparing
    # the centroid of the motion blob between the previous and current frame.
    # We re-detect on the previous frame to get its centroid.
    diff_prev = np.abs(prev_gray.astype(np.float32) - curr_gray.astype(np.float32))
    fg_prev   = (diff_prev > 10).astype(np.uint8)
    kernel    = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_prev   = cv2.morphologyEx(fg_prev, cv2.MORPH_OPEN, kernel)

    ys_p, xs_p = np.where(fg_prev.astype(bool))
    if len(xs_p) > 0:
        cx_prev = int(xs_p.mean())
        moving_left = (cx - cx_prev) < 0
    else:
        # Fallback: can't determine direction, default to seam_right
        moving_left = True

    return seam_right if moving_left else seam_left


# ---------------------------------------------------------------------------
# 4.  MAIN PIPELINE
# ---------------------------------------------------------------------------

def stitch_videos(left_path: str, right_path: str, output_path: str,
                  calib_frames: int = 5,
                  homography_path: str = None,
                  save_homography: str = "homography.npy",
                  blend_levels: int = 6,
                  blend_width: int = 80,
                  sigma_max: float = 4.0,
                  auto_crop: bool = False,
                  auto_crop_method: str = "handcrafted",
                  motion_threshold: int = 25):

    frame_sentinel = object()

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
        H = compute_homography_from_frames(cap_left, cap_right,
                                           n_frames=calib_frames,
                                           sigma_max=sigma_max)
        np.save(save_homography, H)
        print(f"[Homography] Saved to {save_homography}")
        cap_left.set(cv2.CAP_PROP_POS_FRAMES,  0)
        cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print(f"[Homography]\n{H}")

    # ---- Canvas geometry (computed ONCE) ----------------------------------
    canvas_w, canvas_h, tx, ty = compute_canvas_size(H, sample_left.shape, sample_right.shape)
    print(f"[Canvas] {canvas_w}x{canvas_h}  offset=({tx},{ty})")

    # ---- Warp grids (computed ONCE) ---------------------------------------
    left_map_x, left_map_y, right_map_x, right_map_y = prepare_remap_grids(
        H, canvas_w, canvas_h, tx, ty)

    # ---- Auto-crop -------------------------------------------------------
    crop_x, crop_y, crop_w, crop_h = 0, 0, canvas_w, canvas_h
    if auto_crop:
        print("[Auto-crop] Computing crop rectangle from image corners...")
        crop_x, crop_y, crop_w, crop_h = find_crop_rectangle_from_corners(
            H, sample_left.shape, sample_right.shape, canvas_w, canvas_h, tx, ty,
            method=auto_crop_method)
        print(f"[Auto-crop] Crop region: x={crop_x}, y={crop_y}, "
              f"size={crop_w}x{crop_h} (original canvas: {canvas_w}x{canvas_h})")

    # ---- VideoWriter -----------------------------------------------------
    output_w = crop_w if auto_crop else canvas_w
    output_h = crop_h if auto_crop else canvas_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (output_w, output_h))
    if not writer.isOpened():
        raise IOError(f"Cannot open VideoWriter for: {output_path}")

    # ---- Warp sample frame -----------------------------------------------
    left0_w, right0_w = remap_images(
        sample_left, sample_right,
        left_map_x, left_map_y,
        right_map_x, right_map_y,
    )

    # ---- Zone boundaries (fixed for fixed cameras) -----------------------
    seam_left, seam_right = compute_zone_boundaries(left0_w, right0_w)
    print(f"[Zones] seam_left={seam_left}  seam_right={seam_right}  "
          f"overlap_width={seam_right - seam_left}px")

    # ---- Initial seam: use left frontier by default ----------------------
    current_seam = seam_left
    blend_mask   = build_blend_mask(current_seam, left0_w, right0_w, blend_width)
    print(f"[Seam] Initial seam_x={current_seam} (left frontier)")

    # ---- Background model (initialised from sample frame) ----------------
    bg_gray = cv2.cvtColor(left0_w, cv2.COLOR_BGR2GRAY).astype(np.float32)
    prev_gray = bg_gray.copy()

    # Reset to start
    cap_left.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ---- Async I/O queues ------------------------------------------------
    read_queue:  queue.Queue = queue.Queue(maxsize=8)
    write_queue: queue.Queue = queue.Queue(maxsize=8)

    def reader_worker():
        try:
            while True:
                ok1, frame_left  = cap_left.read()
                ok2, frame_right = cap_right.read()
                if not ok1 or not ok2:
                    break
                read_queue.put((frame_left, frame_right))
        finally:
            read_queue.put(frame_sentinel)

    def writer_worker():
        try:
            while True:
                item = write_queue.get()
                if item is frame_sentinel:
                    break
                writer.write(item)
        finally:
            pass

    reader_thread = threading.Thread(target=reader_worker,  daemon=True)
    writer_thread = threading.Thread(target=writer_worker, daemon=True)
    reader_thread.start()
    writer_thread.start()

    # ---- Frame loop ------------------------------------------------------
    print(f"[Stitching] Processing {total} frames ...")
    t0 = time.time()
    frame_idx    = 0
    seam_changes = 0

    while True:
        item = read_queue.get()
        if item is frame_sentinel:
            break
        frame_left, frame_right = item

        # Warp both frames onto the shared canvas
        wl, wr = remap_images(
            frame_left, frame_right,
            left_map_x, left_map_y,
            right_map_x, right_map_y,
        )

        # --- Motion-aware seam selection ----------------------------------
        curr_gray    = cv2.cvtColor(wl, cv2.COLOR_BGR2GRAY).astype(np.float32)
        motion_mask  = detect_motion(curr_gray, bg_gray, threshold=motion_threshold)
        bg_gray      = update_background(bg_gray, curr_gray)

        new_seam = choose_seam(motion_mask, prev_gray, curr_gray,
                               seam_left, seam_right, canvas_w)

        if new_seam is not None and new_seam != current_seam:
            current_seam = new_seam
            blend_mask   = build_blend_mask(current_seam, wl, wr, blend_width)
            seam_changes += 1

        prev_gray = curr_gray

        # Multi-band blend using the current seam mask
        stitched_full = multiband_blend(wl, wr, blend_mask, levels=blend_levels)

        if auto_crop:
            stitched = stitched_full[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        else:
            stitched = stitched_full

        write_queue.put(stitched)
        frame_idx += 1

        if frame_idx % 30 == 0:
            elapsed    = time.time() - t0
            fps_actual = frame_idx / elapsed
            print(f"  {frame_idx}/{total}  ({fps_actual:.1f} fps)  "
                  f"seam_x={current_seam}  changes={seam_changes}", end="\r")

    write_queue.put(frame_sentinel)
    writer_thread.join()
    reader_thread.join()

    elapsed = time.time() - t0
    print(f"\n[Done] {frame_idx} frames in {elapsed:.1f}s  "
          f"({frame_idx/elapsed:.1f} fps avg)  seam changes={seam_changes}")

    cap_left.release()
    cap_right.release()
    writer.release()
    print(f"[Output] Saved to {output_path}")


# ---------------------------------------------------------------------------
# 5.  CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Fixed-camera video stitcher — MAGSAC++ homography calibration.")
    p.add_argument("--left",         required=True,  help="Path to left video")
    p.add_argument("--right",        required=True,  help="Path to right video")
    p.add_argument("--output",       required=True,  help="Path to output video (.mp4)")
    p.add_argument("--homography",   default=None,
                   help="Path to a pre-saved homography .npy file (skips calibration)")
    p.add_argument("--save-homography", default="homography.npy",
                   help="Where to save the computed homography (default: homography.npy)")
    p.add_argument("--calib-frames", type=int, default=5,
                   help="Number of frames used for homography calibration (default: 5)")
    p.add_argument("--sigma-max",    type=float, default=4.0,
                   help="MAGSAC++ upper bound on noise std-dev in pixels (default: 4.0). "
                        "Lower (1-2) = stricter, better for clean footage. "
                        "Higher (6-8) = more tolerant of distortion or blur.")
    p.add_argument("--blend-levels", type=int, default=6,
                   help="Laplacian pyramid levels for multi-band blending (default: 6)")
    p.add_argument("--blend-width",  type=int, default=80,
                   help="Width in pixels of the blending gradient zone (default: 80)")
    p.add_argument("--auto-crop",    action="store_true",
                   help="Automatically crop to largest rectangle using homography corners "
                        "(removes black/empty zones at image edges)")
    p.add_argument("--auto-crop-method", type=str, default="handcrafted",
                   choices=["handcrafted", "optimization"],
                   help="Auto-crop method: 'handcrafted' (simple, default) or 'optimization' (future)")
    p.add_argument("--motion-threshold", type=int, default=25,
                   help="Pixel brightness-change threshold for motion detection (default: 25). "
                        "Lower = more sensitive, higher = less sensitive.")
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
        sigma_max        = args.sigma_max,
        auto_crop        = args.auto_crop,
        auto_crop_method = args.auto_crop_method,
        motion_threshold = args.motion_threshold,
    )