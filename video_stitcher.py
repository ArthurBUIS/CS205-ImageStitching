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




def compute_blend_mask(left_warped: np.ndarray, right_warped: np.ndarray,
                       blend_width: int = 80):
    """
    Fallback: simple gradient blend mask (used when graph-cut is disabled).

    Strategy: find the horizontal seam (vertical line where both images overlap)
    and create a smooth gradient of width `blend_width` pixels around it.
    Pixels only in the left  → mask = 1
    Pixels only in the right → mask = 0
    Overlap region           → smooth gradient 1→0 around the seam
    """
    h, w = left_warped.shape[:2]

    left_valid  = (left_warped.sum(axis=2)  > 0).astype(np.float32)
    right_valid = (right_warped.sum(axis=2) > 0).astype(np.float32)
    overlap     = (left_valid * right_valid)

    mask = left_valid.copy()

    for row in range(h):
        cols = np.where(overlap[row] > 0)[0]
        if len(cols) == 0:
            continue
        seam_x = int(cols.mean())
        x0 = max(0, seam_x - blend_width // 2)
        x1 = min(w, seam_x + blend_width // 2)
        ramp = np.linspace(1.0, 0.0, x1 - x0)
        mask[row, x0:x1] = ramp
        mask[row, x1:]   = 0.0

    mask[right_valid == 0] = 1.0
    mask[left_valid  == 0] = 0.0

    return mask


# ---------------------------------------------------------------------------
# 3b. MOTION-AWARE GRAPH-CUT FUSION  (Step B from the paper)
# ---------------------------------------------------------------------------
#
# Reference: Section 3.3 of
#   "Real-Time Panoramic Surveillance Video Stitching Method for
#    Complex Industrial Environments", Zhu et al., Sensors 2026.
#
# The fusion works in two modes:
#   • First frame  → run full graph-cut over the whole overlap region Ω
#                    (Eq. 6 from the paper) and cache the seam label map.
#   • Subsequent frames → detect moving objects in Ω via background
#                    subtraction; if any motion pixel touches the cached
#                    seam, recompute the graph-cut restricted to Ω \\ M_Ω
#                    (Eq. 7); otherwise reuse the cached seam.
#
# Energy function (Eqs. 6-15 in the paper)
#   E(l) = α · Σ Es(p,q,lp,lq)   [smoothness — colour difference]
#          + β · Σ Eg(p,q,lp,lq)  [gradient   — Sobel edge weight]
#   where both α and β are set to 1 (equal weighting, as in the paper).
#
# Moving-object constraint (Eq. 16)
#   Pixels in M_Ω are removed from the graph → seam is forced to go
#   around detected motion regions.
# ---------------------------------------------------------------------------

def _compute_overlap_mask(left_warped: np.ndarray,
                          right_warped: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask that is True wherever BOTH warped images have
    valid (non-zero) pixels — i.e., the overlap region Ω.
    """
    left_valid  = left_warped.sum(axis=2)  > 0
    right_valid = right_warped.sum(axis=2) > 0
    return left_valid & right_valid


def _compute_color_diff(left_warped: np.ndarray,
                        right_warped: np.ndarray) -> np.ndarray:
    """
    Per-pixel squared L2 colour difference Id(p) = ||I0(p) - I1(p)||²  (Eq. 10).
    Returns a float32 (H,W) array.
    """
    l = left_warped.astype(np.float32)
    r = right_warped.astype(np.float32)
    diff = l - r
    return (diff * diff).sum(axis=2)          # shape (H, W)


def _compute_gradient_weight(left_warped: np.ndarray,
                              right_warped: np.ndarray) -> np.ndarray:
    """
    Edge-weight map W(p) for the gradient term  (Eqs. 11-15).

    For each pixel p the weight is  σ(Wx(p) + Wy(p))  where:
      Wx(p) = [Sx * (I_{lp} - I_{lq})](p)²
      Wy(p) = [Sy * (I_{lp} - I_{lq})](p)²
    We approximate I_{lp}=I0, I_{lq}=I1 (left, right) and use the
    grayscale difference as the input to the Sobel filters — this is
    the standard graph-cut seam formulation used in the paper.

    The sigmoid maps the result to (0,1) so it acts as a soft barrier
    that raises the seam cost near strong edges.

    Returns a float32 (H, W) array in (0, 1).
    """
    diff_gray = cv2.cvtColor(
        np.clip(left_warped.astype(np.float32) - right_warped.astype(np.float32)
                + 128, 0, 255).astype(np.uint8),
        cv2.COLOR_BGR2GRAY
    ).astype(np.float32)

    # Sobel operators (Eq. 15 uses a non-standard 3×3 kernel — we use the
    # OpenCV default which is equivalent in capturing horizontal/vertical
    # gradient structure)
    sx = cv2.Sobel(diff_gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(diff_gray, cv2.CV_32F, 0, 1, ksize=3)
    wx = sx * sx
    wy = sy * sy
    raw = wx + wy                              # Wx + Wy

    # Sigmoid  σ(x) = 1 / (1 + exp(-x/scale))
    # We normalise first so values span a reasonable range for sigmoid.
    scale = float(raw.max()) + 1e-6
    return 1.0 / (1.0 + np.exp(-(raw / scale - 0.5) * 6))  # (H, W), float32


def _detect_motion_region(left_warped: np.ndarray,
                           bg_model,
                           overlap_mask: np.ndarray,
                           motion_threshold: int = 30) -> np.ndarray:
    """
    Detect pixels in Ω that belong to a moving object: M_Ω.

    We use a per-pixel background model (running average of the left
    warped frames) to detect sudden appearance changes consistent with
    a moving object crossing the overlap zone.  A morphological opening
    removes small noise blobs, and dilation ensures the seam path stays
    safely away from object boundaries.

    Parameters
    ----------
    left_warped       : current left frame on the canvas (BGR uint8)
    bg_model          : dict holding the accumulated background estimate
    overlap_mask      : boolean (H,W) — only Ω pixels are considered
    motion_threshold  : pixel-wise absolute difference threshold (0-255)

    Returns
    -------
    motion_mask : boolean (H,W) — True where a moving object is detected
                  inside Ω (i.e., M_Ω in the paper's notation).
    """
    gray = cv2.cvtColor(left_warped, cv2.COLOR_BGR2GRAY).astype(np.float32)

    if bg_model["mean"] is None:
        bg_model["mean"] = gray.copy()
        return np.zeros(gray.shape, dtype=bool)

    # Pixel-wise absolute difference from the background model
    diff = np.abs(gray - bg_model["mean"])
    fg = (diff > motion_threshold) & overlap_mask

    # Morphological cleanup: remove small noise, dilate to create a safety margin
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_clean = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_OPEN,  kernel)
    fg_clean = cv2.morphologyEx(fg_clean,             cv2.MORPH_DILATE, kernel)

    # Update background with slow exponential moving average (α=0.05)
    # Only update from static-background pixels so moving objects don't
    # "absorb" into the model too quickly.
    static = (fg_clean == 0).astype(np.float32)
    bg_model["mean"] = bg_model["mean"] * (1.0 - 0.05 * static) + gray * (0.05 * static)

    return fg_clean.astype(bool)


def _run_graph_cut(overlap_mask: np.ndarray,
                   color_diff: np.ndarray,
                   gradient_weight: np.ndarray,
                   motion_mask: np.ndarray,
                   left_valid: np.ndarray,
                   alpha: float = 1.0,
                   beta: float = 1.0) -> np.ndarray:
    """
    Solve the binary graph-cut problem described in Section 3.3 of the paper
    and return a label map  l  of shape (H, W) where:
        l = 1  ->  pixel is taken from the LEFT  image
        l = 0  ->  pixel is taken from the RIGHT image

    Implementation details
    ----------------------
    The graph has one node per pixel in the *feasible domain*
    Ω_feasible = Ω \\ M_Ω (Eq. 7).  We add:
      • n-links (neighbourhood edges): weight = α·Es + β·Eg  (Eqs. 9, 11)
      • t-links (source/sink):
          – source S represents label=1 (left image)
          – sink   T represents label=0 (right image)
          – pixels in "left-only" zone connect to S with ∞ weight
          – pixels in "right-only" zone connect to T with ∞ weight
          – pixels in Ω_feasible connect to both S and T with small
            equal weight so they are "free" and the seam is found purely
            by the n-link energies (standard formulation).

    We use scipy.sparse.csgraph.maximum_flow as the underlying solver.
    The graph is scaled to integers (×1000, capped at 10^9) because the
    scipy solver requires integer capacities.
    """
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow

    h, w = overlap_mask.shape

    # --- Build feasible domain ------------------------------------------
    # Exclude motion pixels from the graph so the seam avoids them.
    feasible = overlap_mask & ~motion_mask

    left_valid_b  = left_valid.astype(bool)
    right_valid_b = (~left_valid_b) | overlap_mask   # right has valid pixels
    # Re-derive right_valid from complement of left-only zone
    # (both valid in overlap; right-only = neither left_valid nor overlap)
    # We already have overlap_mask; compute proper right_valid separately.
    # NOTE: these are needed for t-link terminals only outside overlap.

    # --- Node numbering ---------------------------------------------------
    # Nodes 0..H*W-1 → pixels, node H*W → source S, node H*W+1 → sink T
    N   = h * w
    S   = N        # source (label = left)
    T   = N + 1    # sink   (label = right)
    num_nodes = N + 2

    INF = int(1e9)
    SCALE = 1000  # float → int scaling

    rows_list, cols_list, caps_list = [], [], []

    def add_edge(u, v, cap):
        rows_list.append(u)
        cols_list.append(v)
        caps_list.append(int(min(cap, INF)))

    # --- T-links ----------------------------------------------------------
    # Outside overlap: force pixels to their respective image side.
    flat_idx = np.arange(N).reshape(h, w)

    # left-only pixels → always label 1 (left) → connect to source with ∞
    left_only = left_valid_b & ~overlap_mask
    for px in flat_idx[left_only].ravel():
        add_edge(S, px, INF)
        add_edge(px, T, 0)

    # right-only pixels → always label 0 (right) → connect to sink with ∞
    right_only_mask = (~left_valid_b) & ~overlap_mask
    for px in flat_idx[right_only_mask].ravel():
        add_edge(S, px, 0)
        add_edge(px, T, INF)

    # feasible overlap pixels → free choice; small equal t-link weights
    for px in flat_idx[feasible].ravel():
        add_edge(S, px, 1)
        add_edge(px, T, 1)

    # motion pixels inside overlap → blocked (force to right / skip)
    motion_in_overlap = motion_mask & overlap_mask
    for px in flat_idx[motion_in_overlap].ravel():
        add_edge(S, px, 0)
        add_edge(px, T, INF)

    # --- N-links (neighbourhood edges) ----------------------------------
    # We iterate over horizontal and vertical 4-connected pairs inside
    # the feasible domain.  Weight = α·(Id(p)+Id(q)) + β·(W(p)+W(q))
    # (combining Eqs. 9 and 11 — the paper uses them summed with equal
    # weighting α=β=1).

    ys, xs = np.where(feasible)
    for y, x in zip(ys, xs):
        p = flat_idx[y, x]
        cd_p = color_diff[y, x]
        gw_p = gradient_weight[y, x]

        for dy, dx in ((0, 1), (1, 0)):   # right and down neighbours
            ny, nx = y + dy, x + dx
            if ny >= h or nx >= w:
                continue
            if not feasible[ny, nx]:
                continue
            q = flat_idx[ny, nx]
            cd_q = color_diff[ny, nx]
            gw_q = gradient_weight[ny, nx]

            # Smoothness  Es = |lp-lq| * (Id(p) + Id(q))  (Eq. 9)
            es = alpha * (cd_p + cd_q)
            # Gradient    Eg = |lp-lq| * (W(p)  + W(q))   (Eq. 11)
            eg = beta  * (gw_p + gw_q)

            weight = int(min((es + eg) * SCALE, INF))
            if weight <= 0:
                weight = 1  # avoid zero-weight edges

            # Undirected edge → two directed edges
            add_edge(p, q, weight)
            add_edge(q, p, weight)

    # --- Solve max-flow / min-cut ----------------------------------------
    if not rows_list:
        # Degenerate case: no feasible pixels — return left image everywhere
        label = left_valid_b.astype(np.int32)
        return label

    cap_matrix = csr_matrix(
        (caps_list, (rows_list, cols_list)),
        shape=(num_nodes, num_nodes),
        dtype=np.int32,
    )

    result   = maximum_flow(cap_matrix, S, T)
    flow_mat = result.flow   # CSR matrix: flow[i,j] = flow on edge (i,j)

    # Min-cut: BFS from S through edges with remaining capacity > 0.
    # Remaining capacity on (i,j) = cap[i,j] - flow[i,j].
    # We need both the original capacity matrix and the flow matrix.
    residual_cap = cap_matrix - flow_mat   # element-wise; can be negative for back-edges

    visited = np.zeros(num_nodes, dtype=bool)
    queue_bfs = [S]
    visited[S] = True
    while queue_bfs:
        node = queue_bfs.pop()
        # Get all outgoing edges from this node
        start = residual_cap.indptr[node]
        end   = residual_cap.indptr[node + 1]
        for idx in range(start, end):
            nbr = residual_cap.indices[idx]
            if not visited[nbr] and residual_cap.data[idx] > 0:
                visited[nbr] = True
                queue_bfs.append(int(nbr))

    pixel_visited = visited[:N].reshape(h, w)

    # Build final label map:
    #   reachable from S → label=1 (left)
    #   otherwise        → label=0 (right)
    #   left-only zone   → always 1; right-only zone → always 0
    label = np.zeros((h, w), dtype=np.int32)
    label[pixel_visited]   = 1
    label[left_only]       = 1
    label[right_only_mask] = 0

    return label


def seam_label_to_blend_mask(label: np.ndarray,
                              left_valid: np.ndarray,
                              right_valid: np.ndarray,
                              feather_px: int = 3) -> np.ndarray:
    """
    Convert the integer label map (0/1) produced by graph-cut into a
    float32 blend mask suitable for multiband_blend().

    A thin feathering zone of `feather_px` pixels is applied along the
    seam to avoid a hard step discontinuity in the final blended output.

    mask = 1.0 → take from left image
    mask = 0.0 → take from right image
    """
    mask = label.astype(np.float32)   # 1.0 = left, 0.0 = right

    # Feather: dilate/erode the binary mask by feather_px and average
    if feather_px > 0:
        k = 2 * feather_px + 1
        kernel = np.ones((k, k), np.float32) / (k * k)
        mask = cv2.filter2D(mask, -1, kernel)

    # Enforce hard constraints outside the overlap zone
    lv = (left_valid.sum(axis=2) > 0) if left_valid.ndim == 3 else left_valid.astype(bool)
    rv = (right_valid.sum(axis=2) > 0) if right_valid.ndim == 3 else right_valid.astype(bool)

    mask[~lv] = 0.0
    mask[~rv] = 1.0
    mask[lv & ~rv] = 1.0   # left-only zone
    mask[rv & ~lv] = 0.0   # right-only zone

    return np.clip(mask, 0.0, 1.0)


def compute_seam_mask_graphcut(
        left_warped: np.ndarray,
        right_warped: np.ndarray,
        motion_mask: np.ndarray | None = None,
        alpha: float = 1.0,
        beta: float = 1.0,
        feather_px: int = 3,
) -> np.ndarray:
    """
    High-level wrapper: compute the optimal seam via graph-cut and return
    a float32 blend mask.

    Parameters
    ----------
    left_warped   : warped left frame on canvas  (BGR uint8)
    right_warped  : warped right frame on canvas (BGR uint8)
    motion_mask   : optional boolean (H,W) — True for M_Ω pixels to avoid
    alpha         : weight for smoothness term  (default 1, as in paper)
    beta          : weight for gradient term    (default 1, as in paper)
    feather_px    : half-width of seam feathering in pixels

    Returns
    -------
    blend_mask : float32 (H,W), 1=left, 0=right
    """
    overlap_mask = _compute_overlap_mask(left_warped, right_warped)

    if not overlap_mask.any():
        # No overlap at all → trivial split
        lv = (left_warped.sum(axis=2) > 0).astype(np.float32)
        return lv

    color_diff      = _compute_color_diff(left_warped, right_warped)
    gradient_weight = _compute_gradient_weight(left_warped, right_warped)

    if motion_mask is None:
        motion_mask = np.zeros(overlap_mask.shape, dtype=bool)

    left_valid = (left_warped.sum(axis=2) > 0)

    label = _run_graph_cut(
        overlap_mask, color_diff, gradient_weight,
        motion_mask, left_valid, alpha=alpha, beta=beta,
    )

    right_valid = (right_warped.sum(axis=2) > 0)
    blend_mask = seam_label_to_blend_mask(label, left_valid, right_valid,
                                          feather_px=feather_px)
    return blend_mask


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
                  use_graphcut: bool = True,
                  motion_threshold: int = 30,
                  seam_feather: int = 3):

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
    print(f"[Canvas] {canvas_w}×{canvas_h}  offset=({tx},{ty})")

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
              f"size={crop_w}×{crop_h} (original canvas: {canvas_w}×{canvas_h})")

    # ---- VideoWriter -----------------------------------------------------
    output_w = crop_w if auto_crop else canvas_w
    output_h = crop_h if auto_crop else canvas_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (output_w, output_h))
    if not writer.isOpened():
        raise IOError(f"Cannot open VideoWriter for: {output_path}")

    # ---- Warp sample frame (used for initial seam) -----------------------
    left0_w, right0_w = remap_images(
        sample_left, sample_right,
        left_map_x, left_map_y,
        right_map_x, right_map_y,
    )

    # ---- Seam / blend mask (computed from FIRST frame) -------------------
    # -----------------------------------------------------------------------
    # Paper workflow (Section 3.1):
    #   • First frame  → run graph-cut (or fallback) to find optimal seam.
    #                    Save the seam label map as a template.
    #   • Later frames → check if motion crosses the cached seam;
    #                    update only when needed.
    # -----------------------------------------------------------------------
    if use_graphcut:
        print("[Seam] Computing optimal seam via graph-cut on first frame …")
        blend_mask = compute_seam_mask_graphcut(
            left0_w, right0_w,
            motion_mask=None,
            feather_px=seam_feather,
        )
        print("[Seam] Done.")
    else:
        print("[Blend mask] Computing gradient blend mask from first frame …")
        blend_mask = compute_blend_mask(left0_w, right0_w, blend_width=blend_width)
        print("[Blend mask] Done.")

    # Cache the seam label map (binary, pre-feathering) for motion checking.
    # We derive it from blend_mask: pixels > 0.5 → label 1 (left).
    cached_seam_label: np.ndarray = (blend_mask >= 0.5).astype(np.int32)

    # Background model for motion detection (running mean of left channel).
    bg_model: dict = {"mean": None}

    # Pre-compute overlap mask (static for fixed cameras).
    overlap_mask = _compute_overlap_mask(left0_w, right0_w)

    # Reset to start for the main processing pass.
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
    print(f"[Stitching] Processing {total} frames …  "
          f"(fusion={'graph-cut+motion' if use_graphcut else 'gradient-blend'})")
    t0 = time.time()
    frame_idx   = 0
    seam_updates = 0

    while True:
        item = read_queue.get()
        if item is frame_sentinel:
            break
        frame_left, frame_right = item

        # Warp both frames onto the shared canvas.
        wl, wr = remap_images(
            frame_left, frame_right,
            left_map_x, left_map_y,
            right_map_x, right_map_y,
        )

        # ------------------------------------------------------------------
        # Motion-aware seam update  (paper Section 3.1, Step B)
        # ------------------------------------------------------------------
        if use_graphcut:
            # Detect moving objects in the overlap region.
            motion_mask = _detect_motion_region(
                wl, bg_model, overlap_mask,
                motion_threshold=motion_threshold,
            )

            # Check whether any motion pixel overlaps the current seam
            # boundary (pixels where the seam transitions, i.e. near 0.5).
            seam_boundary = np.abs(blend_mask - 0.5) < 0.25
            motion_hits_seam = bool((motion_mask & seam_boundary).any())

            if motion_hits_seam:
                # Recompute seam restricted to Ω \\ M_Ω  (Eq. 7)
                blend_mask = compute_seam_mask_graphcut(
                    wl, wr,
                    motion_mask=motion_mask,
                    feather_px=seam_feather,
                )
                cached_seam_label = (blend_mask >= 0.5).astype(np.int32)
                seam_updates += 1

        # Multi-band blend using the current seam mask.
        stitched_full = multiband_blend(wl, wr, blend_mask, levels=blend_levels)

        if auto_crop:
            stitched = stitched_full[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
        else:
            stitched = stitched_full

        write_queue.put(stitched)
        frame_idx += 1

        if frame_idx % 30 == 0:
            elapsed     = time.time() - t0
            fps_actual  = frame_idx / elapsed
            print(f"  {frame_idx}/{total}  ({fps_actual:.1f} fps)  "
                  f"seam updates: {seam_updates}", end="\r")

    write_queue.put(frame_sentinel)
    writer_thread.join()
    reader_thread.join()

    elapsed = time.time() - t0
    print(f"\n[Done] {frame_idx} frames in {elapsed:.1f}s  "
          f"({frame_idx/elapsed:.1f} fps avg)  "
          f"seam recomputations: {seam_updates}")

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

    # ---- Graph-cut / motion-aware seam options --------------------------
    p.add_argument("--no-graphcut", action="store_true",
                   help="Disable graph-cut seam search and fall back to the "
                        "original gradient blend mask.")
    p.add_argument("--motion-threshold", type=int, default=30,
                   help="Pixel-wise absolute difference threshold (0-255) used "
                        "to detect moving objects in the overlap region. "
                        "Lower = more sensitive (default: 30).")
    p.add_argument("--seam-feather", type=int, default=3,
                   help="Half-width in pixels of the feathering zone applied "
                        "along the graph-cut seam (default: 3).")
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
        use_graphcut     = not args.no_graphcut,
        motion_threshold = args.motion_threshold,
        seam_feather     = args.seam_feather,
    )