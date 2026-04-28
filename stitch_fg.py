"""
Real-time video stitching from two fixed cameras — with static-foreground avoidance.

Extends stitch_motion.py (YOLO + MOG2 seam-avoidance) with a THIRD source
of avoid-mask information: a static foreground detector based on stereo
depth from the overlap region.

Rationale
---------
YOLO catches people (regardless of motion). MOG2 catches moving non-people.
Neither catches static non-people close to the camera — like a desk, a
sofa arm, a chair back. When the seam crosses such an object, parallax
causes a visible "teleportation" artifact as the seam moves.

The solution: at startup, use the two cameras as a stereo pair. In the
overlap region we have two views of the same content from slightly
different positions. Close objects have larger disparity; far objects
have small disparity. Thresholding the disparity map gives us a binary
mask of "parallax-risky" pixels — the static foreground.

This mask is added to the cost map with the same penalty as motion
(default 5e7). By default it's computed once at startup. Optionally,
--fg_recompute_seconds N re-runs the detector every N seconds to
handle furniture that gets rearranged (chairs pushed, etc).

Stereo math notes
-----------------
Our two warped images in the overlap bbox are aligned via the homography.
For approximately-coplanar cameras looking at an approximately-planar
scene, this is close to rectified — horizontal disparities dominate and
cv2.StereoSGBM works well. For more complex geometry the homography
warp is an approximation of rectification; we assume it's good enough
and verify empirically via the preview mode.

Disparity is inversely proportional to depth:
    depth ∝ 1 / disparity
A pixel with disparity 20 is roughly twice as close as a pixel with
disparity 10. Choosing a disparity threshold = choosing a "closeness
cutoff" above which pixels are treated as foreground.

Standalone preview mode
-----------------------
Run with --preview_fg_only to skip the full stitching pipeline and just
compute + display the static foreground mask. Useful for tuning the
SGBM parameters and the disparity threshold before enabling the full
pipeline.

    python stitch_fg.py --video_a A.mp4 --video_b B.mp4 --output /dev/null \
                        --preview_fg_only

This writes a 3-panel diagnostic image to fg_preview.png:
  1. The first warped A frame
  2. The raw disparity map (heatmap)
  3. The final binary FG mask overlaid on the warped frame

New flags vs stitch_motion.py:
    --no_fg                        Disable static FG detection entirely.
    --fg_disparity_threshold N     Disparity above this = foreground
                                   (default 16). Lower = more permissive.
    --fg_dilate PX                 Dilation radius for the FG mask (default 10).
    --fg_penalty F                 Cost penalty for FG pixels (default 5e7).
    --fg_recompute_seconds F       Seconds between FG mask recomputations
                                   (default 0 = compute once at startup).
    --fg_sgbm_block_size N         SGBM matching block size (default 9).
                                   Odd, 5-11 typical.
    --fg_sgbm_num_disparities N    Maximum disparity to search
                                   (default 128). Must be multiple of 16.
    --fg_min_blob_area PX          Minimum blob area in pixels; smaller
                                   blobs are dropped as SGBM noise
                                   (default 500).
    --preview_fg_only              Write an fg_preview.png and exit.
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ESTIMATE_HOMOGRAPHY_FROM_FIRST_FRAME = True
HOMOGRAPHY_PATH = "homography.npy"
PERSON_CLASS_ID = 0
PERSON_PENALTY = 1e8
EDGE_PENALTY = 1e6


# ---------------------------------------------------------------------------
# Homography + canvas + remap + static geometry (unchanged)
# ---------------------------------------------------------------------------

def estimate_homography(img_a, img_b):
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=4000)
    kpts_a, desc_a = orb.detectAndCompute(gray_a, None)
    kpts_b, desc_b = orb.detectAndCompute(gray_b, None)
    if desc_a is None or desc_b is None:
        raise RuntimeError("Could not detect ORB features.")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc_b, desc_a)
    matches = sorted(matches, key=lambda m: m.distance)
    matches = matches[: max(50, len(matches) // 2)]
    pts_b = np.float32([kpts_b[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts_a = np.float32([kpts_a[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(pts_b, pts_a, cv2.RANSAC, 5.0)
    if H is None:
        raise RuntimeError("Homography estimation failed.")
    return H


def compute_canvas(shape_a, shape_b, H_b_to_a):
    h_a, w_a = shape_a[:2]
    h_b, w_b = shape_b[:2]
    corners_b = np.float32([[0, 0], [w_b, 0], [w_b, h_b], [0, h_b]]).reshape(-1, 1, 2)
    corners_b_in_a = cv2.perspectiveTransform(corners_b, H_b_to_a)
    corners_a = np.float32([[0, 0], [w_a, 0], [w_a, h_a], [0, h_a]]).reshape(-1, 1, 2)
    all_corners = np.concatenate([corners_a, corners_b_in_a], axis=0)
    x_min, y_min = np.floor(all_corners.min(axis=0).ravel()).astype(int)
    x_max, y_max = np.ceil(all_corners.max(axis=0).ravel()).astype(int)
    tx, ty = -x_min, -y_min
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)
    canvas_w = x_max - x_min
    canvas_h = y_max - y_min
    H_b_to_canvas = T @ H_b_to_a
    H_a_to_canvas = T.copy()
    return (canvas_w, canvas_h), T, H_b_to_canvas, H_a_to_canvas


def build_remap(H, canvas_size):
    W, H_canvas = canvas_size
    H_inv = np.linalg.inv(H)
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H_canvas, dtype=np.float32))
    ones = np.ones_like(xs)
    canvas_coords = np.stack([xs.ravel(), ys.ravel(), ones.ravel()], axis=0)
    src = H_inv @ canvas_coords
    src /= src[2:3, :]
    map_x = src[0].reshape(H_canvas, W).astype(np.float32)
    map_y = src[1].reshape(H_canvas, W).astype(np.float32)
    return map_x, map_y


def build_static_geometry(src_shape_a, src_shape_b, map_ax, map_ay,
                          map_bx, map_by, canvas_size):
    W, H = canvas_size
    src_white_a = np.full(src_shape_a[:2], 255, dtype=np.uint8)
    src_white_b = np.full(src_shape_b[:2], 255, dtype=np.uint8)
    mask_a = cv2.remap(src_white_a, map_ax, map_ay, cv2.INTER_NEAREST)
    mask_b = cv2.remap(src_white_b, map_bx, map_by, cv2.INTER_NEAREST)

    overlap_bool = (mask_a > 0) & (mask_b > 0)
    only_a_bool = (mask_a > 0) & ~overlap_bool
    only_b_bool = (mask_b > 0) & ~overlap_bool
    if not overlap_bool.any():
        raise RuntimeError("No overlap between cameras.")

    rows = np.where(overlap_bool.any(axis=1))[0]
    cols = np.where(overlap_bool.any(axis=0))[0]
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1

    overlap_in_bbox = (overlap_bool[y0:y1, x0:x1].astype(np.uint8)) * 255
    mask_a_in_bbox = mask_a[y0:y1, x0:x1].copy()
    mask_b_in_bbox = mask_b[y0:y1, x0:x1].copy()
    only_a_in_bbox = ((only_a_bool[y0:y1, x0:x1]).astype(np.uint8)) * 255
    only_b_in_bbox = ((only_b_bool[y0:y1, x0:x1]).astype(np.uint8)) * 255

    return {
        "only_a_u8": (only_a_bool.astype(np.uint8)) * 255,
        "only_b_u8": (only_b_bool.astype(np.uint8)) * 255,
        "overlap_bbox": (x0, y0, x1, y1),
        "overlap_in_bbox": overlap_in_bbox,
        "mask_a_in_bbox": mask_a_in_bbox,
        "mask_b_in_bbox": mask_b_in_bbox,
        "only_a_in_bbox": only_a_in_bbox,
        "only_b_in_bbox": only_b_in_bbox,
    }


# ---------------------------------------------------------------------------
# Gain + cost + seam + multi-band blending (unchanged from stitch_motion.py)
# ---------------------------------------------------------------------------

def compute_gain_compensation(warped_a, warped_b, overlap_bbox, overlap_in_bbox):
    x0, y0, x1, y1 = overlap_bbox
    wa = warped_a[y0:y1, x0:x1]
    wb = warped_b[y0:y1, x0:x1]
    mean_a = np.array(cv2.mean(wa, mask=overlap_in_bbox)[:3], dtype=np.float32)
    mean_b = np.array(cv2.mean(wb, mask=overlap_in_bbox)[:3], dtype=np.float32)
    mean_a = np.clip(mean_a, 1.0, None)
    mean_b = np.clip(mean_b, 1.0, None)
    target = np.sqrt(mean_a * mean_b)
    g_a = (target / mean_a).astype(np.float32)
    g_b = (target / mean_b).astype(np.float32)
    return g_a, g_b


def build_gain_lut(gains_bgr):
    x = np.arange(256, dtype=np.float32)
    scaled = x[:, None] * gains_bgr[None, :]
    scaled = np.clip(scaled, 0, 255).astype(np.uint8)
    return scaled.reshape(1, 256, 3)


def apply_gain_lut(img_uint8, lut):
    return cv2.LUT(img_uint8, lut)


def compute_cost_fast(wa_bb, wb_bb, overlap_in_bbox, cost_scratch):
    diff = cv2.absdiff(wa_bb, wb_bb)
    diff_f = diff.astype(np.float32, copy=False)
    np.multiply(diff_f, diff_f, out=cost_scratch)
    cost = cost_scratch.sum(axis=2)
    cost[overlap_in_bbox == 0] = 1e9
    return cost


def find_dp_seam(cost):
    H, W = cost.shape
    dp = cost.copy()
    for y in range(1, H):
        prev = dp[y - 1]
        left  = np.concatenate(([np.inf], prev[:-1]))
        right = np.concatenate((prev[1:], [np.inf]))
        dp[y] += np.minimum(np.minimum(prev, left), right)
    seam_x = np.empty(H, dtype=np.int32)
    seam_x[-1] = int(np.argmin(dp[-1]))
    for y in range(H - 2, -1, -1):
        x = seam_x[y + 1]
        x0 = max(x - 1, 0)
        x1 = min(x + 2, W)
        local = dp[y, x0:x1]
        seam_x[y] = x0 + int(np.argmin(local))
    return seam_x


def upscale_seam(seam_x_small, bbox_shape, downscale):
    H_bb, W_bb = bbox_shape
    H_small = seam_x_small.shape[0]
    ys_small = np.arange(H_small, dtype=np.float32)
    ys_full  = np.linspace(0, H_small - 1, H_bb, dtype=np.float32)
    seam_x_full = np.interp(ys_full, ys_small, seam_x_small.astype(np.float32))
    seam_x_full = (seam_x_full * downscale).astype(np.int32)
    return np.clip(seam_x_full, 0, W_bb - 1)


def add_edge_margin_penalty(cost, margin, edge_penalty=EDGE_PENALTY):
    if margin <= 0:
        return
    margin = min(margin, cost.shape[1] // 2)
    cost[:,  :margin]  += edge_penalty
    cost[:, -margin:] += edge_penalty


def add_seam_regularizer(cost, seam_prev_small, lam):
    if seam_prev_small is None or lam <= 0:
        return
    H, W = cost.shape
    col_idx = np.arange(W, dtype=np.float32)[None, :]
    seam_prev_col = seam_prev_small.astype(np.float32)[:, None]
    dx = col_idx - seam_prev_col
    penalty = (dx * dx) * float(lam)
    cost += penalty


def build_soft_mask_fast(seam_x_full, bbox_shape, static, blend_width):
    H_bb, W_bb = bbox_shape
    col_idx = np.arange(W_bb, dtype=np.int32)[None, :]
    seam_col = seam_x_full[:, None]
    hard = (col_idx < seam_col).astype(np.float32)

    target_sigma = max(1.0, blend_width / 3.0)
    depth = int(np.floor(np.log2(target_sigma / 3.0)))
    depth = max(0, min(depth, 4))

    cur = hard
    for _ in range(depth):
        cur = cv2.pyrDown(cur)
    coarse_sigma = target_sigma / (2 ** depth)
    ks = max(3, int(6 * coarse_sigma) | 1)
    cur = cv2.GaussianBlur(cur, (ks, ks), sigmaX=coarse_sigma,
                           sigmaY=coarse_sigma)

    shapes = [(H_bb, W_bb)]
    for _ in range(depth):
        ph, pw = shapes[-1]
        shapes.append(((ph + 1) // 2, (pw + 1) // 2))
    for i in range(depth, 0, -1):
        th, tw = shapes[i - 1]
        cur = cv2.pyrUp(cur, dstsize=(tw, th))

    soft = cur
    only_a = static["only_a_in_bbox"]
    only_b = static["only_b_in_bbox"]
    soft[only_a > 0] = 1.0
    soft[only_b > 0] = 0.0
    return soft


def fill_invalid_with_other_fast(a_bb_u8, b_bb_u8, static):
    only_a = static["only_a_in_bbox"]
    only_b = static["only_b_in_bbox"]
    a_out = a_bb_u8.copy()
    b_out = b_bb_u8.copy()
    cv2.copyTo(b_bb_u8, only_b, a_out)
    cv2.copyTo(a_bb_u8, only_a, b_out)
    return a_out, b_out


def build_gaussian_pyramid(img_f32, levels):
    gp = [img_f32]
    for _ in range(levels):
        gp.append(cv2.pyrDown(gp[-1]))
    return gp


def build_laplacian_pyramid(img_f32, levels):
    gp = build_gaussian_pyramid(img_f32, levels)
    lp = []
    for i in range(levels):
        up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
        lp.append(gp[i] - up)
    lp.append(gp[levels])
    return lp


def reconstruct_from_laplacian(lp):
    img = lp[-1]
    for level in reversed(lp[:-1]):
        img = cv2.pyrUp(img, dstsize=(level.shape[1], level.shape[0]))
        img = img + level
    return img


def blend_pyramids_fast(lp_a, lp_b, gp_m):
    blended = []
    for la, lb, gm in zip(lp_a, lp_b, gp_m):
        if la.ndim == 3:
            if gm.ndim == 2:
                gm3 = cv2.merge([gm] * la.shape[2])
            elif gm.ndim == 3 and gm.shape[2] == 1:
                gm3 = cv2.merge([gm[:, :, 0]] * la.shape[2])
            else:
                gm3 = gm
        else:
            gm3 = gm
        la_gm = cv2.multiply(la, gm3)
        one_minus = np.empty_like(gm3)
        np.subtract(1.0, gm3, out=one_minus)
        lb_gm = cv2.multiply(lb, one_minus)
        out = cv2.add(la_gm, lb_gm)
        blended.append(out)
    return blended


def composite_multiband_fast(warped_a, warped_b, static, seam_x_full,
                             blend_width, blend_levels, out_buf):
    x0, y0, x1, y1 = static["overlap_bbox"]
    out_buf.fill(0)
    cv2.copyTo(warped_a, static["only_a_u8"], out_buf)
    cv2.copyTo(warped_b, static["only_b_u8"], out_buf)

    H_bb = y1 - y0
    W_bb = x1 - x0
    bbox_shape = (H_bb, W_bb)

    a_bb = warped_a[y0:y1, x0:x1]
    b_bb = warped_b[y0:y1, x0:x1]

    a_bb_pad, b_bb_pad = fill_invalid_with_other_fast(a_bb, b_bb, static)
    mask_f32 = build_soft_mask_fast(
        seam_x_full, bbox_shape, static, blend_width,
    )

    min_dim = min(a_bb_pad.shape[:2])
    max_levels = max(1, int(np.log2(min_dim)) - 2)
    levels = min(blend_levels, max_levels)

    a_f = a_bb_pad.astype(np.float32)
    b_f = b_bb_pad.astype(np.float32)

    lp_a = build_laplacian_pyramid(a_f, levels)
    lp_b = build_laplacian_pyramid(b_f, levels)
    gp_m = build_gaussian_pyramid(mask_f32, levels)

    blended_lp = blend_pyramids_fast(lp_a, lp_b, gp_m)

    recon = reconstruct_from_laplacian(blended_lp)
    np.clip(recon, 0, 255, out=recon)
    blended_bb = recon.astype(np.uint8)

    valid_in_bbox = cv2.bitwise_or(static["mask_a_in_bbox"],
                                   static["mask_b_in_bbox"])
    cv2.copyTo(blended_bb, valid_in_bbox, out_buf[y0:y1, x0:x1])

    return out_buf


def composite_feathered(warped_a, warped_b, take_from_a, take_from_b,
                        seam_x_full, overlap_bbox, overlap_in_bbox,
                        feather_px, out_buf):
    out_buf.fill(0)
    cv2.copyTo(warped_a, take_from_a, out_buf)
    cv2.copyTo(warped_b, take_from_b, out_buf)
    if feather_px <= 0:
        return out_buf
    x0, y0, x1, y1 = overlap_bbox
    H_bb = y1 - y0
    W_bb = x1 - x0
    fp = int(feather_px)
    strip_w = 2 * fp + 1
    alpha_1d = np.linspace(1.0, 0.0, strip_w, dtype=np.float32)
    for yi in range(H_bb):
        seam_c_bb = int(seam_x_full[yi])
        xl_bb = max(seam_c_bb - fp, 0)
        xr_bb = min(seam_c_bb + fp, W_bb - 1)
        if xr_bb < xl_bb:
            continue
        a_start = fp - (seam_c_bb - xl_bb)
        a_end   = fp + (xr_bb - seam_c_bb) + 1
        if a_end - a_start != (xr_bb - xl_bb + 1):
            continue
        row_valid = overlap_in_bbox[yi, xl_bb:xr_bb + 1]
        if not row_valid.any():
            continue
        y_c = yi + y0
        xl_c = xl_bb + x0
        xr_c = xr_bb + x0
        alpha = alpha_1d[a_start:a_end].reshape(-1, 1)
        a_pix = warped_a[y_c, xl_c:xr_c + 1].astype(np.float32)
        b_pix = warped_b[y_c, xl_c:xr_c + 1].astype(np.float32)
        blended = alpha * a_pix + (1.0 - alpha) * b_pix
        valid_idx = np.where(row_valid > 0)[0]
        if len(valid_idx) == 0:
            continue
        dst = out_buf[y_c, xl_c:xr_c + 1]
        dst[valid_idx] = blended[valid_idx].astype(np.uint8)
    return out_buf


def seam_to_hardcut_masks(seam_x_full, static, bbox_shape,
                          out_take_a, out_take_b):
    H_bb, W_bb = bbox_shape
    col_idx = np.arange(W_bb, dtype=np.int32)[None, :]
    seam_col = seam_x_full[:, None]
    take_a_bool = col_idx < seam_col
    take_a = (take_a_bool.astype(np.uint8)) * 255
    take_b = ((~take_a_bool).astype(np.uint8)) * 255

    x0, y0, x1, y1 = static["overlap_bbox"]
    np.copyto(out_take_a, static["only_a_u8"])
    np.copyto(out_take_b, static["only_b_u8"])
    overlap_bb = static["overlap_in_bbox"]
    take_a_in = cv2.bitwise_and(take_a, overlap_bb)
    take_b_in = cv2.bitwise_and(take_b, overlap_bb)
    out_take_a[y0:y1, x0:x1] = cv2.bitwise_or(out_take_a[y0:y1, x0:x1], take_a_in)
    out_take_b[y0:y1, x0:x1] = cv2.bitwise_or(out_take_b[y0:y1, x0:x1], take_b_in)


# ---------------------------------------------------------------------------
# YOLO + MOG2 (unchanged from stitch_motion.py)
# ---------------------------------------------------------------------------

class PersonSegmenter:
    def __init__(self, weights_path: str):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError("pip install ultralytics") from e
        self.model = YOLO(weights_path)

    def predict_mask(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        results = self.model.predict(
            frame_bgr, classes=[PERSON_CLASS_ID],
            verbose=False, retina_masks=False,
        )
        mask = np.zeros((H, W), dtype=np.uint8)
        if not results:
            return mask
        r = results[0]
        if r.masks is None or r.masks.data is None or len(r.masks.data) == 0:
            return mask
        mdata = r.masks.data.cpu().numpy()
        merged_small = (mdata > 0.5).any(axis=0).astype(np.uint8) * 255
        mask = cv2.resize(merged_small, (W, H), interpolation=cv2.INTER_NEAREST)
        return mask


class MotionDetector:
    def __init__(self, history_frames, var_threshold=25.0, detect_shadows=False):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=int(history_frames),
            varThreshold=float(var_threshold),
            detectShadows=bool(detect_shadows),
        )
        self.history_frames = history_frames
        self.var_threshold = var_threshold

    def apply(self, frame_bgr):
        fg = self.bg.apply(frame_bgr)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        return fg


# ---------------------------------------------------------------------------
# *** NEW: Static foreground detection via stereo in the overlap ***
# ---------------------------------------------------------------------------

def compute_static_fg_mask(warped_a_bb, warped_b_bb, overlap_in_bbox,
                           disparity_threshold, block_size, num_disparities,
                           min_blob_area, return_disparity=False):
    """
    Detect close-to-camera (high-parallax) static objects in the overlap.

    Inputs are the bbox-cropped warped images (shape: H_bb, W_bb, 3 BGR).
    Since the homography aligns far-away points, residual horizontal
    shifts between the two warped images correspond to parallax (close
    objects). Stereo SGBM measures these shifts as "disparity."

    Steps:
      1. Convert to grayscale.
      2. Run cv2.StereoSGBM to get a per-pixel disparity map. SGBM returns
         disparity * 16 as int16; we divide by 16 to get pixels.
      3. Threshold: pixels with disparity > `disparity_threshold` are
         candidates.
      4. Restrict to pixels inside the true overlap shape.
      5. Clean up: morphological opening to remove specks, then drop
         connected components smaller than `min_blob_area`.
      6. Morphological closing to fill small holes inside detected blobs.

    Returns
    -------
    fg_mask_bb : uint8 (H_bb, W_bb), 0 or 255
        Binary foreground mask in bbox coordinates.
    disparity : int16 (H_bb, W_bb) OR None
        Raw disparity map, returned only if return_disparity is True
        (used by preview mode).
    """
    gray_a = cv2.cvtColor(warped_a_bb, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(warped_b_bb, cv2.COLOR_BGR2GRAY)

    # StereoSGBM parameters — tuned for "pretty good" defaults, tunable via CLI.
    # For background on these, see OpenCV docs / Hirschmuller 2005.
    P1 = 8  * 3 * block_size * block_size
    P2 = 32 * 3 * block_size * block_size
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=int(num_disparities),
        blockSize=int(block_size),
        P1=P1,
        P2=P2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )

    # SGBM returns int16 disparity * 16. We divide by 16 to get pixels.
    disp_raw = matcher.compute(gray_a, gray_b)   # int16, (H_bb, W_bb)
    disparity = disp_raw.astype(np.float32) / 16.0

    # Threshold.
    fg = (disparity > disparity_threshold).astype(np.uint8) * 255

    # Only valid inside the true overlap shape — outside is garbage.
    fg = cv2.bitwise_and(fg, overlap_in_bbox)

    # Morphological opening to remove specks.
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, open_kernel)

    # Remove connected components smaller than min_blob_area.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, 8)
    cleaned = np.zeros_like(fg)
    for i in range(1, num_labels):  # skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_blob_area:
            cleaned[labels == i] = 255
    fg = cleaned

    # Morphological closing to fill small holes inside kept blobs.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, close_kernel)

    if return_disparity:
        return fg, disp_raw
    return fg, None


def save_fg_preview(warped_a_bb, disparity_raw, fg_mask_bb, output_path):
    """
    Write a 3-panel diagnostic image:
      - panel 1: the warped A frame (for visual reference)
      - panel 2: disparity heatmap
      - panel 3: fg_mask overlaid in red on the warped A frame

    Useful for tuning the FG detector before enabling the full pipeline.
    """
    H_bb, W_bb = fg_mask_bb.shape

    # Panel 2: disparity heatmap.
    disp_f = disparity_raw.astype(np.float32) / 16.0
    disp_f = np.clip(disp_f, 0, None)
    if disp_f.max() > 0:
        disp_norm = (disp_f / disp_f.max() * 255).astype(np.uint8)
    else:
        disp_norm = disp_f.astype(np.uint8)
    disp_vis = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)

    # Panel 3: overlay.
    overlay = warped_a_bb.copy()
    red = np.zeros_like(overlay)
    red[:, :, 2] = 255  # BGR: red
    alpha = 0.5
    mask3 = cv2.merge([fg_mask_bb] * 3) > 0
    blended = np.where(mask3, (alpha * red + (1 - alpha) * overlay).astype(np.uint8), overlay)

    # Stack horizontally.
    panels = np.hstack([warped_a_bb, disp_vis, blended])
    cv2.imwrite(output_path, panels)


# ---------------------------------------------------------------------------
# Debug overlays
# ---------------------------------------------------------------------------

def draw_seam_overlay(canvas, seam_x_full, bbox):
    x0, y0, x1, y1 = bbox
    H_bb = y1 - y0
    ys = np.arange(H_bb) + y0
    xs = seam_x_full + x0
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    for i in range(len(pts) - 1):
        cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]), (0, 0, 255), 2)


def draw_mask_overlay(canvas, mask_bbox, bbox, color=(0, 0, 255), alpha=0.35):
    x0, y0, x1, y1 = bbox
    region = canvas[y0:y1, x0:x1]
    overlay = region.copy()
    overlay[mask_bbox > 0] = color
    cv2.addWeighted(overlay, alpha, region, 1 - alpha, 0, dst=region)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_a", required=True)
    parser.add_argument("--video_b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--debug_seam", action="store_true")
    parser.add_argument("--debug_mask", action="store_true")
    parser.add_argument("--yolo_every", type=int, default=3)
    parser.add_argument("--mask_dilate", type=int, default=15)
    parser.add_argument("--seam_downscale", type=int, default=4)
    parser.add_argument("--yolo_weights", default="yolov8n-seg.pt")
    parser.add_argument("--no_gain_comp", action="store_true")
    parser.add_argument("--cost_ema", type=float, default=0.4)
    parser.add_argument("--no_cost_ema", action="store_true")
    parser.add_argument("--blend_width", type=int, default=80)
    parser.add_argument("--blend_levels", type=int, default=5)
    parser.add_argument("--no_multiband", action="store_true")
    parser.add_argument("--feather_px", type=int, default=8)
    parser.add_argument("--seam_lambda", type=float, default=5.0)
    parser.add_argument("--seam_edge_margin", type=int, default=50)
    parser.add_argument("--no_yolo", action="store_true")
    parser.add_argument("--no_motion", action="store_true")
    parser.add_argument("--motion_history_seconds", type=float, default=15.0)
    parser.add_argument("--motion_var_threshold", type=float, default=25.0)
    parser.add_argument("--motion_penalty", type=float, default=5e7)
    parser.add_argument("--motion_dilate", type=int, default=15)
    # *** Step C flags ***
    parser.add_argument("--no_fg", action="store_true",
                        help="Disable static foreground detection.")
    parser.add_argument("--fg_disparity_threshold", type=float, default=16.0,
                        help="Disparity above this = foreground (default 16).")
    parser.add_argument("--fg_dilate", type=int, default=10,
                        help="FG mask dilation radius in px (default 10).")
    parser.add_argument("--fg_penalty", type=float, default=5e7,
                        help="Cost penalty for FG pixels (default 5e7).")
    parser.add_argument("--fg_recompute_seconds", type=float, default=0.0,
                        help="Seconds between FG recomputations "
                             "(0 = compute once at startup).")
    parser.add_argument("--fg_sgbm_block_size", type=int, default=9)
    parser.add_argument("--fg_sgbm_num_disparities", type=int, default=128)
    parser.add_argument("--fg_min_blob_area", type=int, default=500)
    parser.add_argument("--preview_fg_only", action="store_true",
                        help="Compute + save fg_preview.png and exit.")
    args = parser.parse_args()

    use_yolo = not args.no_yolo
    use_motion = not args.no_motion
    use_fg = not args.no_fg
    if not use_yolo and not use_motion and not use_fg and not args.preview_fg_only:
        raise RuntimeError("All three avoid-mask sources disabled. "
                           "Enable at least one.")

    print(f"[info] OpenCV: {cv2.getNumberOfCPUs()} CPUs, "
          f"using {cv2.getNumThreads()} threads.")
    try:
        import torch as _torch
        print(f"[info] torch num_threads = {_torch.get_num_threads()}")
    except Exception:
        pass
    ema_eff = 1.0 if args.no_cost_ema else float(args.cost_ema)
    use_multiband = not args.no_multiband
    print(f"[info] yolo={use_yolo} (every {args.yolo_every} fr)  "
          f"motion={use_motion}  static_fg={use_fg}")
    print(f"[info] gain_comp={not args.no_gain_comp}  "
          f"cost_ema={ema_eff}  multiband={use_multiband}")
    print(f"[info] person_penalty={PERSON_PENALTY:.1e}  "
          f"motion_penalty={args.motion_penalty:.1e}  "
          f"fg_penalty={args.fg_penalty:.1e}")

    cap_a = cv2.VideoCapture(args.video_a)
    cap_b = cv2.VideoCapture(args.video_b)
    if not cap_a.isOpened() or not cap_b.isOpened():
        raise RuntimeError("Could not open one of the input videos.")
    fps = cap_a.get(cv2.CAP_PROP_FPS) or 25.0
    motion_history_frames = max(2, int(round(args.motion_history_seconds * fps)))
    fg_recompute_frames = (int(round(args.fg_recompute_seconds * fps))
                           if args.fg_recompute_seconds > 0 else 0)
    print(f"[info] FPS={fps:.2f}  motion_history={motion_history_frames} fr  "
          f"fg_recompute_every={fg_recompute_frames or 'never (startup only)'} fr")

    ok_a, frame_a = cap_a.read()
    ok_b, frame_b = cap_b.read()
    if not (ok_a and ok_b):
        raise RuntimeError("Could not read first frame.")

    if ESTIMATE_HOMOGRAPHY_FROM_FIRST_FRAME:
        print("[info] Estimating homography from first frame pair...")
        H_b_to_a = estimate_homography(frame_a, frame_b)
        np.save(HOMOGRAPHY_PATH, H_b_to_a)
    else:
        if not Path(HOMOGRAPHY_PATH).exists():
            raise RuntimeError(f"{HOMOGRAPHY_PATH} not found.")
        H_b_to_a = np.load(HOMOGRAPHY_PATH)

    canvas_size, T, H_b_to_canvas, H_a_to_canvas = compute_canvas(
        frame_a.shape, frame_b.shape, H_b_to_a
    )
    print(f"[info] Canvas size: {canvas_size[0]} x {canvas_size[1]}")

    print("[info] Precomputing remap maps + static geometry...")
    map_ax, map_ay = build_remap(H_a_to_canvas, canvas_size)
    map_bx, map_by = build_remap(H_b_to_canvas, canvas_size)
    static = build_static_geometry(
        frame_a.shape, frame_b.shape,
        map_ax, map_ay, map_bx, map_by,
        canvas_size,
    )
    x0, y0, x1, y1 = static["overlap_bbox"]
    bbox_shape = (y1 - y0, x1 - x0)
    print(f"[info] Overlap bbox: x=[{x0},{x1}) y=[{y0},{y1}) "
          f"size={bbox_shape[1]}x{bbox_shape[0]}")

    lut_a = None
    lut_b = None
    if not args.no_gain_comp:
        wa0 = cv2.remap(frame_a, map_ax, map_ay, cv2.INTER_LINEAR)
        wb0 = cv2.remap(frame_b, map_bx, map_by, cv2.INTER_LINEAR)
        gains_a, gains_b = compute_gain_compensation(
            wa0, wb0, static["overlap_bbox"], static["overlap_in_bbox"]
        )
        print(f"[info] gains_a = [{gains_a[0]:.3f}, {gains_a[1]:.3f}, {gains_a[2]:.3f}]")
        print(f"[info] gains_b = [{gains_b[0]:.3f}, {gains_b[1]:.3f}, {gains_b[2]:.3f}]")
        lut_a = build_gain_lut(gains_a)
        lut_b = build_gain_lut(gains_b)

    # --- Static FG mask (compute once at startup) -------------------------
    fg_mask_bbox = np.zeros(bbox_shape, dtype=np.uint8)
    fg_dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * args.fg_dilate + 1, 2 * args.fg_dilate + 1),
    )

    def recompute_fg_mask(frame_a_src, frame_b_src, save_preview=False):
        """Recompute the static FG mask from a fresh frame pair."""
        wa = cv2.remap(frame_a_src, map_ax, map_ay, cv2.INTER_LINEAR)
        wb = cv2.remap(frame_b_src, map_bx, map_by, cv2.INTER_LINEAR)
        if lut_a is not None:
            wa = apply_gain_lut(wa, lut_a)
            wb = apply_gain_lut(wb, lut_b)
        wa_bb = wa[y0:y1, x0:x1]
        wb_bb = wb[y0:y1, x0:x1]
        fg_raw, disp_raw = compute_static_fg_mask(
            wa_bb, wb_bb, static["overlap_in_bbox"],
            disparity_threshold=args.fg_disparity_threshold,
            block_size=args.fg_sgbm_block_size,
            num_disparities=args.fg_sgbm_num_disparities,
            min_blob_area=args.fg_min_blob_area,
            return_disparity=save_preview,
        )
        fg_dilated = cv2.dilate(fg_raw, fg_dilate_kernel)
        if save_preview:
            save_fg_preview(wa_bb, disp_raw, fg_dilated, "fg_preview.png")
            print(f"[info] Wrote fg_preview.png (three panels: frame, "
                  f"disparity, overlay). Inspect to tune "
                  f"--fg_disparity_threshold and related flags.")
        return fg_dilated

    if use_fg or args.preview_fg_only:
        print("[info] Computing static foreground mask via stereo SGBM...")
        t0 = time.time()
        fg_mask_bbox = recompute_fg_mask(
            frame_a, frame_b, save_preview=args.preview_fg_only,
        )
        print(f"[info] FG mask computed in {(time.time()-t0)*1000:.1f} ms  "
              f"({100*int(fg_mask_bbox.sum()/255)/fg_mask_bbox.size:.1f}% "
              f"of bbox flagged as foreground)")

    if args.preview_fg_only:
        print("[info] --preview_fg_only requested; exiting before stitching.")
        cap_a.release()
        cap_b.release()
        return

    # --- YOLO + MOG2 setup ------------------------------------------------
    segmenter = None
    person_dilate_kernel = None
    if use_yolo:
        print(f"[info] Loading YOLO weights: {args.yolo_weights}")
        segmenter = PersonSegmenter(args.yolo_weights)
        person_dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * args.mask_dilate + 1, 2 * args.mask_dilate + 1),
        )

    motion_a = None
    motion_b = None
    motion_dilate_kernel = None
    if use_motion:
        motion_a = MotionDetector(motion_history_frames, args.motion_var_threshold)
        motion_b = MotionDetector(motion_history_frames, args.motion_var_threshold)
        motion_dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * args.motion_dilate + 1, 2 * args.motion_dilate + 1),
        )

    W, H = canvas_size
    out_buf      = np.zeros((H, W, 3), dtype=np.uint8)
    take_from_a  = np.zeros((H, W), dtype=np.uint8)
    take_from_b  = np.zeros((H, W), dtype=np.uint8)
    cost_scratch = np.empty((bbox_shape[0], bbox_shape[1], 3), dtype=np.float32)
    person_mask_bbox = np.zeros(bbox_shape, dtype=np.uint8)
    motion_mask_bbox = np.zeros(bbox_shape, dtype=np.uint8)
    cost_ema = None
    seam_prev_small = None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, canvas_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output writer for {args.output}")

    t_read = t_warp = t_gain = t_yolo = t_motion = t_fg = t_maskwarp = 0.0
    t_cost = t_seam = t_composite = t_write = 0.0

    frame_idx = 0
    t_start = time.time()

    while True:
        tt = time.perf_counter()
        if frame_idx > 0:
            ok_a, frame_a = cap_a.read()
            ok_b, frame_b = cap_b.read()
            if not (ok_a and ok_b):
                break
        t1 = time.perf_counter()

        warped_a = cv2.remap(frame_a, map_ax, map_ay, cv2.INTER_LINEAR)
        warped_b = cv2.remap(frame_b, map_bx, map_by, cv2.INTER_LINEAR)
        t2 = time.perf_counter()

        if lut_a is not None:
            warped_a = apply_gain_lut(warped_a, lut_a)
            warped_b = apply_gain_lut(warped_b, lut_b)
        t3 = time.perf_counter()
        t_gain += t3 - t2

        # --- Periodic FG recompute ----------------------------------------
        if (use_fg and fg_recompute_frames > 0 and frame_idx > 0
                and frame_idx % fg_recompute_frames == 0):
            t0 = time.perf_counter()
            fg_mask_bbox = recompute_fg_mask(frame_a, frame_b, save_preview=False)
            t_fg += time.perf_counter() - t0

        t_after_gain = t3

        if use_yolo and frame_idx % args.yolo_every == 0:
            mask_a_src = segmenter.predict_mask(frame_a)
            mask_b_src = segmenter.predict_mask(frame_b)
            t_after_yolo = time.perf_counter()
            mask_a_canvas = cv2.remap(mask_a_src, map_ax, map_ay, cv2.INTER_NEAREST)
            mask_b_canvas = cv2.remap(mask_b_src, map_bx, map_by, cv2.INTER_NEAREST)
            union = cv2.bitwise_or(mask_a_canvas, mask_b_canvas)
            union = cv2.dilate(union, person_dilate_kernel)
            person_mask_bbox = union[y0:y1, x0:x1].copy()
            t_yolo += t_after_yolo - t_after_gain
            t_maskwarp += time.perf_counter() - t_after_yolo
            t_after_perception = time.perf_counter()
        else:
            t_after_perception = t_after_gain

        if use_motion:
            fg_a = motion_a.apply(frame_a)
            fg_b = motion_b.apply(frame_b)
            t_after_mog = time.perf_counter()
            fg_a_canvas = cv2.remap(fg_a, map_ax, map_ay, cv2.INTER_NEAREST)
            fg_b_canvas = cv2.remap(fg_b, map_bx, map_by, cv2.INTER_NEAREST)
            motion_union = cv2.bitwise_or(fg_a_canvas, fg_b_canvas)
            motion_union = cv2.dilate(motion_union, motion_dilate_kernel)
            motion_mask_bbox = motion_union[y0:y1, x0:x1].copy()
            t_motion += t_after_mog - t_after_perception
            t_maskwarp += time.perf_counter() - t_after_mog
            t_after_perception = time.perf_counter()

        wa_bb = warped_a[y0:y1, x0:x1]
        wb_bb = warped_b[y0:y1, x0:x1]
        photo_cost = compute_cost_fast(
            wa_bb, wb_bb, static["overlap_in_bbox"], cost_scratch,
        )

        if ema_eff >= 1.0 or cost_ema is None:
            if cost_ema is None or cost_ema.shape != photo_cost.shape:
                cost_ema = photo_cost.copy()
            else:
                np.copyto(cost_ema, photo_cost)
        else:
            cv2.addWeighted(photo_cost, ema_eff,
                            cost_ema, 1.0 - ema_eff,
                            0, dst=cost_ema)

        cost_for_dp = cost_ema.copy()
        add_edge_margin_penalty(cost_for_dp, args.seam_edge_margin)

        # --- Apply penalties (priority: person > motion == fg) ----------
        # A pixel flagged by any of {motion, fg} but NOT person gets the
        # higher of the two penalties applied once. We model this as:
        # set the medium-priority penalty on (motion OR fg) pixels minus
        # person pixels, then the person penalty on person pixels.
        medium_mask = None
        if use_motion and motion_mask_bbox.any():
            medium_mask = motion_mask_bbox.copy()
        if use_fg and fg_mask_bbox.any():
            medium_mask = (fg_mask_bbox if medium_mask is None
                           else cv2.bitwise_or(medium_mask, fg_mask_bbox))
        if medium_mask is not None:
            if use_yolo:
                medium_only = cv2.bitwise_and(
                    medium_mask, cv2.bitwise_not(person_mask_bbox))
            else:
                medium_only = medium_mask
            # motion_penalty and fg_penalty may differ; we use the max of
            # the two to be safe. In the default config they're equal (5e7).
            medium_penalty = max(args.motion_penalty if use_motion else 0.0,
                                 args.fg_penalty if use_fg else 0.0)
            cost_for_dp[medium_only > 0] += medium_penalty
        if use_yolo and person_mask_bbox.any():
            cost_for_dp[person_mask_bbox > 0] += PERSON_PENALTY

        t5 = time.perf_counter()
        t_cost += t5 - t_after_perception

        ds = max(1, args.seam_downscale)
        if ds > 1:
            cost_small = cv2.resize(
                cost_for_dp,
                (cost_for_dp.shape[1] // ds, cost_for_dp.shape[0] // ds),
                interpolation=cv2.INTER_AREA,
            )
        else:
            cost_small = cost_for_dp.copy()

        add_seam_regularizer(cost_small, seam_prev_small, args.seam_lambda)

        seam_x_small = find_dp_seam(cost_small)
        seam_prev_small = seam_x_small.copy()
        seam_x_full = upscale_seam(seam_x_small, bbox_shape, ds)
        t6 = time.perf_counter()
        t_seam += t6 - t5

        if use_multiband:
            stitched = composite_multiband_fast(
                warped_a, warped_b, static, seam_x_full,
                args.blend_width, args.blend_levels, out_buf,
            )
        else:
            seam_to_hardcut_masks(
                seam_x_full, static, bbox_shape,
                take_from_a, take_from_b,
            )
            stitched = composite_feathered(
                warped_a, warped_b, take_from_a, take_from_b,
                seam_x_full, static["overlap_bbox"],
                static["overlap_in_bbox"], args.feather_px, out_buf,
            )
        if args.debug_mask:
            # Layer: fg (yellow) < motion (orange) < person (red).
            if use_fg:
                draw_mask_overlay(stitched, fg_mask_bbox,
                                  static["overlap_bbox"],
                                  color=(0, 255, 255), alpha=0.25)
            if use_motion:
                draw_mask_overlay(stitched, motion_mask_bbox,
                                  static["overlap_bbox"],
                                  color=(0, 165, 255), alpha=0.30)
            if use_yolo:
                draw_mask_overlay(stitched, person_mask_bbox,
                                  static["overlap_bbox"],
                                  color=(0, 0, 255), alpha=0.35)
        if args.debug_seam:
            draw_seam_overlay(stitched, seam_x_full, static["overlap_bbox"])
        t7 = time.perf_counter()
        t_composite += t7 - t6

        writer.write(stitched)
        t8 = time.perf_counter()
        t_write += t8 - t7

        t_read += t1 - tt
        t_warp += t2 - t1

        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break

    elapsed = time.time() - t_start
    n = max(frame_idx, 1)
    stages = [
        ("read",          t_read),
        ("warp",          t_warp),
        ("gain",          t_gain),
        ("yolo",          t_yolo),
        ("motion (MOG2)", t_motion),
        ("static FG",     t_fg),
        ("mask warp+dil", t_maskwarp),
        ("cost + ema",    t_cost),
        ("dp seam",       t_seam),
        ("composite",     t_composite),
        ("write",         t_write),
    ]
    total = max(sum(t for _, t in stages), 1e-9)
    print()
    for name, t in stages:
        print(f"[timing] {name:<14s} {t*1000/n:7.2f} ms  "
              f"({100*t/total:5.1f}%)")
    print(f"[info] Processed {frame_idx} frames in {elapsed:.2f}s "
          f"({frame_idx / max(elapsed, 1e-6):.2f} fps)")
    print(f"[info] Output written to {args.output}")

    cap_a.release()
    cap_b.release()
    writer.release()


if __name__ == "__main__":
    main()
