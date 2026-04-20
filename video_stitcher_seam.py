"""
Real-time video stitching from two fixed cameras.
 
Stitches two synchronized video streams into a single panorama video, with
special care taken so that people moving through the seam region are neither
ghosted, doubled, nor visibly cut.
 
Pipeline overview
-----------------
The pipeline has two phases.
 
Startup (runs once):
    1. Estimate a single homography from the first frame pair.
    2. Compute the panorama canvas size and precompute pixel-level remap
       tables so per-frame warping is a single SIMD pass.
    3. Compute static geometry: validity masks of each warped camera, the
       overlap region, and its bounding box. All per-frame work will be
       restricted to this bbox.
    4. (Optional) Compute per-camera BGR gain scalars for exposure matching.
    5. Load the YOLO person-segmentation model.
 
Per frame:
    1. Warp both frames onto the canvas via remap.
    2. Apply the precomputed BGR gains.
    3. Run YOLO on both original frames (every N frames); union the two
       masks on the canvas and dilate — this is the "person mask".
    4. Compute a photometric cost map (squared BGR difference) over the
       overlap bbox.
    5. Smooth the photometric cost across frames via an exponential moving
       average.
    6. Inject hard penalties into a fresh copy of the cost: forbid image
       edges, forbid person pixels.
    7. Downsample the cost and add a quadratic attractor toward the
       previous frame's seam (stability regularizer).
    8. Find the minimum-cost top-to-bottom seam by dynamic programming.
    9. Build a soft mask from the seam and run multi-band (Laplacian
       pyramid) blending on the overlap bbox. Hard-copy the rest of the
       canvas directly from whichever camera is valid.
    10. Write the resulting frame to the output video.
 
The seam is what keeps people from being cut; the multi-band blend is what
makes the seam invisible on the background.
 
Command-line arguments
----------------------
Required:
    --video_a PATH              Input video from camera A (left).
    --video_b PATH              Input video from camera B (right).
    --output PATH               Output stitched video (.mp4).
 
General:
    --max_frames N              Process only the first N frames (0 = all).
                                Useful for quick iteration. Default: 0.
    --debug_seam                Overlay the DP seam as a red line.
    --debug_mask                Overlay the person mask as translucent red.
 
Person segmentation (YOLOv8n-seg):
    --yolo_weights PATH         Path/name of YOLO weights file.
                                Default: yolov8n-seg.pt (auto-downloaded on
                                first run, ~7 MB).
    --yolo_every N              Run YOLO once every N frames; reuse the
                                cached mask in between. Lower = fresher
                                mask but slower. Higher = staler mask but
                                faster. Default: 3.
    --mask_dilate PX            Dilation radius applied to the unioned
                                person mask, in pixels. Absorbs the
                                parallax offset between how camera A and
                                camera B see the same person. Increase if
                                the seam sometimes grazes a person's
                                outline; decrease if the mask engulfs too
                                much background. Default: 15.
 
Cost-map behavior:
    --cost_ema A                EMA factor in [0, 1] for the photometric        # EMA stands for Exponential Moving Average
                                cost. Lower = smoother but slower to react
                                to genuine scene changes. Higher = more
                                reactive but jitterier. Default: 0.4.
    --no_cost_ema               Disable EMA entirely (equivalent to
                                cost_ema = 1.0). The seam will jitter
                                frame-to-frame on flat backgrounds.
    --seam_lambda F             Strength of the quadratic "stay near the
                                previous seam" attractor. 0 disables it
                                (first-frame-only behavior). Higher values
                                pin the seam harder to its previous
                                position; too high and the seam reacts
                                sluggishly when a person approaches.
                                Default: 5.0.
    --seam_edge_margin N        Width in pixels of the forbidden band at
                                the left and right edges of the overlap
                                bbox. Must be at least blend_width / 2,
                                otherwise the multi-band blur reaches into
                                padded pixels and produces artifacts.
                                0 disables. Default: 50.
 
Seam computation:
    --seam_downscale N          Factor by which the cost map is downscaled
                                before DP. Higher = much faster DP, but
                                coarser seam (1 px at small resolution =
                                N px at full resolution). Default: 4.
 
Gain compensation:
    --no_gain_comp              Disable global per-channel gain
                                compensation. With the multi-band blend,
                                gain comp is partially redundant — its
                                coarse-level pyramid band already handles
                                low-frequency exposure matching. Disabling
                                here saves one pass per frame but may
                                leave a faint low-frequency color step
                                depending on the cameras.
 
Composite:
    --blend_width PX            Width in pixels of the soft mask ramp
                                around the DP seam. Feeds the Gaussian
                                pyramid. Must satisfy
                                seam_edge_margin > blend_width / 2.
                                Default: 80.
    --blend_levels N            Laplacian pyramid depth used for the
                                multi-band blend. Higher = wider
                                low-frequency blending (better exposure
                                hiding) at the cost of more pyrDown/pyrUp
                                per frame. Default: 5.
    --no_multiband              Fall back to a narrow (2 * feather_px)
                                linear alpha feather. Much faster; much
                                more visible seam. For A/B comparison.
    --feather_px PX             Half-width of the fallback narrow feather
                                when --no_multiband is set. Default: 8.
 
Usage
-----
    python video_stitcher_seam.py --video_a camA.mp4 --video_b camB.mp4 \
                                 --output stitched.mp4
 
For debugging, add --debug_seam --debug_mask to overlay the seam and
person mask on the output. Recommended for any first run on a new
scene.
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
# Homography + canvas + remap (unchanged)
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
# Gain — LUT (from step 5a)
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


# ---------------------------------------------------------------------------
# Cost map + DP seam (unchanged)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# FIX 1 — soft mask via pyramid blur instead of a 163x163 Gaussian
# ---------------------------------------------------------------------------

def build_soft_mask_fast(seam_x_full, bbox_shape, static, blend_width):
    """
    Produce the same soft mask as build_soft_mask_from_seam_v2, much faster.

    Strategy: the v3 version needed sigma ≈ blend_width/3 ≈ 27 px, which
    forces a 163x163 kernel even with separable convolution (~52 ms). But
    a large-sigma Gaussian can be equivalently obtained by:
        (a) downsampling the hard mask a few times (each pyrDown is
            itself a small Gaussian blur followed by 2x decimation),
        (b) applying a SMALL Gaussian blur at the coarse level, and
        (c) upsampling back.

    The hard 0/1 step only contains one edge per row, so the downsampled
    mask still has a crisp transition we can blur with a tiny kernel.
    After pyrUp back to full resolution the result is a wide, smooth ramp
    that matches v3's output visually.

    Pixels outside the true overlap are then clamped to the correct hard
    values, same as v3.
    """
    H_bb, W_bb = bbox_shape
    col_idx = np.arange(W_bb, dtype=np.int32)[None, :]
    seam_col = seam_x_full[:, None]
    hard = (col_idx < seam_col).astype(np.float32)

    # Pick pyramid depth so that at the coarsest level the desired blur
    # is achievable with sigma ~= 3 (kernel size ~17). Each pyrDown halves
    # spatial resolution, which corresponds to doubling the effective
    # sigma back at full resolution.
    # target_sigma = blend_width / 3
    # coarse_sigma = target_sigma / (2 ** depth)  ~ 3
    target_sigma = max(1.0, blend_width / 3.0)
    depth = int(np.floor(np.log2(target_sigma / 3.0)))
    depth = max(0, min(depth, 4))   # 0-4 levels of downsample

    # Downsample hard mask.
    cur = hard
    for _ in range(depth):
        cur = cv2.pyrDown(cur)

    # Small Gaussian at the coarse level. This sigma brings the total
    # effective sigma at full resolution back to ~target_sigma.
    coarse_sigma = target_sigma / (2 ** depth)
    ks = max(3, int(6 * coarse_sigma) | 1)
    cur = cv2.GaussianBlur(cur, (ks, ks), sigmaX=coarse_sigma,
                           sigmaY=coarse_sigma)

    # Upsample back.
    # cv2.pyrUp doubles size; we need to match our starting bbox exactly,
    # so specify dstsize at each step using the shapes recorded on the way
    # down. We didn't record them — simplest: recompute them here.
    shapes = [(H_bb, W_bb)]
    for _ in range(depth):
        ph, pw = shapes[-1]
        shapes.append(((ph + 1) // 2, (pw + 1) // 2))
    # Walk back up.
    for i in range(depth, 0, -1):
        th, tw = shapes[i - 1]
        cur = cv2.pyrUp(cur, dstsize=(tw, th))

    # Clamp outside overlap.
    soft = cur
    only_a = static["only_a_in_bbox"]
    only_b = static["only_b_in_bbox"]
    soft[only_a > 0] = 1.0
    soft[only_b > 0] = 0.0
    return soft


# ---------------------------------------------------------------------------
# FIX 3 — fill invalid pixels via cv2.copyTo instead of boolean indexing
# ---------------------------------------------------------------------------

def fill_invalid_with_other_fast(a_bb_u8, b_bb_u8, static):
    """
    Same semantics as fill_invalid_with_other:
      - a_out: inside bbox but outside overlap on B-only regions, replace
        with B's content; elsewhere keep A.
      - b_out: symmetric.

    cv2.copyTo(src, mask, dst) writes src into dst where mask is non-zero,
    all in a single SIMD pass and in-place. Much faster than
    `a_out[only_b > 0] = b_bb[only_b > 0]` which builds a boolean index
    and does a gather/scatter in Python.
    """
    only_a = static["only_a_in_bbox"]
    only_b = static["only_b_in_bbox"]

    a_out = a_bb_u8.copy()     # unavoidable: we need a writeable buffer
    b_out = b_bb_u8.copy()
    # Copy B into A where only B is valid.
    cv2.copyTo(b_bb_u8, only_b, a_out)
    # Copy A into B where only A is valid.
    cv2.copyTo(a_bb_u8, only_a, b_out)
    return a_out, b_out


# ---------------------------------------------------------------------------
# Pyramids (unchanged math, unchanged speed)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# FIX 2 — per-level blend via cv2.addWeighted with scratch buffers
# ---------------------------------------------------------------------------

def blend_pyramids_fast(lp_a, lp_b, gp_m):
    """
    Per-level blend: out[i] = la[i] * gm[i] + lb[i] * (1 - gm[i])

    v3 implementation: `la * gm3 + lb * (1 - gm3)` — numpy expression that
    allocates ~5 temporaries per level (gm3 broadcast, la * gm3, 1 - gm3,
    lb * (1 - gm3), final sum). For 6 levels that's ~30 allocations per
    frame of float32 arrays sized 6-MB down to 100 KB.

    Fast implementation: we can't use cv2.addWeighted directly because
    the weights are SPATIALLY VARYING (per-pixel alpha from the mask).
    But we can do the math in-place with cv2.multiply + cv2.add, which
    write into a preallocated output buffer instead of allocating a
    fresh one each operation.

    Per level:
        tmp_a = la * gm3     (cv2.multiply with dst=tmp_a)
        tmp_b = lb * (1-gm3) (cv2.multiply with dst=tmp_b)
        out   = tmp_a + tmp_b (cv2.add with dst=out)

    Still 3 ops per level but with reusable scratch buffers, the only
    per-frame allocation is the level-0 result.
    """
    blended = []
    for la, lb, gm in zip(lp_a, lp_b, gp_m):
        if la.ndim == 3:
            # OpenCV arithmetics do not broadcast 1-channel masks to 3 channels,
            # so explicitly expand the mask to match the image channel count.
            if gm.ndim == 2:
                gm3 = cv2.merge([gm] * la.shape[2])
            elif gm.ndim == 3 and gm.shape[2] == 1:
                gm3 = cv2.merge([gm[:, :, 0]] * la.shape[2])
            else:
                gm3 = gm
        else:
            gm3 = gm
        # la * gm3.
        la_gm = cv2.multiply(la, gm3)
        # 1 - gm3.  We have to materialize this; no in-place subtract in
        # cv2 that handles broadcasting cleanly.
        one_minus = np.empty_like(gm3)
        np.subtract(1.0, gm3, out=one_minus)
        lb_gm = cv2.multiply(lb, one_minus)
        out = cv2.add(la_gm, lb_gm)
        blended.append(out)
    return blended


# ---------------------------------------------------------------------------
# Instrumented composite — same sub-stage keys as the diag build
# ---------------------------------------------------------------------------

def composite_multiband_fast(warped_a, warped_b, static, seam_x_full,
                             blend_width, blend_levels, out_buf):
    timings = {}
    x0, y0, x1, y1 = static["overlap_bbox"]

    # -- writeback preamble -----------------------------------------------
    t0 = time.perf_counter()
    out_buf.fill(0)
    cv2.copyTo(warped_a, static["only_a_u8"], out_buf)
    cv2.copyTo(warped_b, static["only_b_u8"], out_buf)
    t_prewrite = time.perf_counter() - t0

    H_bb = y1 - y0
    W_bb = x1 - x0
    bbox_shape = (H_bb, W_bb)

    a_bb = warped_a[y0:y1, x0:x1]
    b_bb = warped_b[y0:y1, x0:x1]

    # -- FIX 3: fill invalid ---------------------------------------------
    t0 = time.perf_counter()
    a_bb_pad, b_bb_pad = fill_invalid_with_other_fast(a_bb, b_bb, static)
    timings["fill"] = time.perf_counter() - t0

    # -- FIX 1: soft mask via pyramid blur -------------------------------
    t0 = time.perf_counter()
    mask_f32 = build_soft_mask_fast(
        seam_x_full, bbox_shape, static, blend_width,
    )
    timings["softmask"] = time.perf_counter() - t0

    min_dim = min(a_bb_pad.shape[:2])
    max_levels = max(1, int(np.log2(min_dim)) - 2)
    levels = min(blend_levels, max_levels)

    # -- cast to float32 --------------------------------------------------
    t0 = time.perf_counter()
    a_f = a_bb_pad.astype(np.float32)
    b_f = b_bb_pad.astype(np.float32)
    timings["cast"] = time.perf_counter() - t0

    # -- pyramids ---------------------------------------------------------
    t0 = time.perf_counter()
    lp_a = build_laplacian_pyramid(a_f, levels)
    timings["pyr_a"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    lp_b = build_laplacian_pyramid(b_f, levels)
    timings["pyr_b"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    gp_m = build_gaussian_pyramid(mask_f32, levels)
    timings["pyr_mask"] = time.perf_counter() - t0

    # -- FIX 2: per-level blend ------------------------------------------
    t0 = time.perf_counter()
    blended_lp = blend_pyramids_fast(lp_a, lp_b, gp_m)
    timings["blend_lvl"] = time.perf_counter() - t0

    # -- reconstruct ------------------------------------------------------
    t0 = time.perf_counter()
    recon = reconstruct_from_laplacian(blended_lp)
    np.clip(recon, 0, 255, out=recon)
    blended_bb = recon.astype(np.uint8)
    timings["recon"] = time.perf_counter() - t0

    # -- writeback --------------------------------------------------------
    t0 = time.perf_counter()
    valid_in_bbox = cv2.bitwise_or(static["mask_a_in_bbox"],
                                   static["mask_b_in_bbox"])
    cv2.copyTo(blended_bb, valid_in_bbox, out_buf[y0:y1, x0:x1])
    timings["writeback"] = (time.perf_counter() - t0) + t_prewrite

    return out_buf, timings


# ---------------------------------------------------------------------------
# Fallback: narrow feather (unchanged)
# ---------------------------------------------------------------------------

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
# YOLO
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
    args = parser.parse_args()

    print(f"[info] OpenCV: {cv2.getNumberOfCPUs()} CPUs, "
          f"using {cv2.getNumThreads()} threads.")
    try:
        import torch as _torch
        print(f"[info] torch num_threads = {_torch.get_num_threads()}")
    except Exception:
        pass
    ema_eff = 1.0 if args.no_cost_ema else float(args.cost_ema)
    use_multiband = not args.no_multiband
    print(f"[info] yolo_every={args.yolo_every}  "
          f"mask_dilate={args.mask_dilate}  "
          f"DP_downscale={args.seam_downscale}  "
          f"gain_comp={not args.no_gain_comp} (LUT)  "
          f"cost_ema={ema_eff}  "
          f"multiband={use_multiband}  "
          f"blend_width={args.blend_width}  "
          f"blend_levels={args.blend_levels}")
    print(f"[info] seam_lambda={args.seam_lambda}  "
          f"seam_edge_margin={args.seam_edge_margin}")

    cap_a = cv2.VideoCapture(args.video_a)
    cap_b = cv2.VideoCapture(args.video_b)
    if not cap_a.isOpened() or not cap_b.isOpened():
        raise RuntimeError("Could not open one of the input videos.")
    fps = cap_a.get(cv2.CAP_PROP_FPS) or 25.0

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
        print("[info] Computing gain compensation from first frame pair...")
        wa0 = cv2.remap(frame_a, map_ax, map_ay, cv2.INTER_LINEAR)
        wb0 = cv2.remap(frame_b, map_bx, map_by, cv2.INTER_LINEAR)
        gains_a, gains_b = compute_gain_compensation(
            wa0, wb0, static["overlap_bbox"], static["overlap_in_bbox"]
        )
        print(f"[info] gains_a = [{gains_a[0]:.3f}, {gains_a[1]:.3f}, {gains_a[2]:.3f}]")
        print(f"[info] gains_b = [{gains_b[0]:.3f}, {gains_b[1]:.3f}, {gains_b[2]:.3f}]")
        lut_a = build_gain_lut(gains_a)
        lut_b = build_gain_lut(gains_b)

    print(f"[info] Loading YOLO weights: {args.yolo_weights}")
    segmenter = PersonSegmenter(args.yolo_weights)
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * args.mask_dilate + 1, 2 * args.mask_dilate + 1),
    )

    W, H = canvas_size
    out_buf      = np.zeros((H, W, 3), dtype=np.uint8)
    take_from_a  = np.zeros((H, W), dtype=np.uint8)
    take_from_b  = np.zeros((H, W), dtype=np.uint8)
    cost_scratch = np.empty((bbox_shape[0], bbox_shape[1], 3), dtype=np.float32)
    person_mask_bbox = np.zeros(bbox_shape, dtype=np.uint8)
    cost_ema = None
    seam_prev_small = None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, canvas_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open output writer for {args.output}")

    t_read = t_warp = t_gain = t_yolo = t_maskwarp = 0.0
    t_cost = t_seam = t_composite = t_write = 0.0

    sub_keys = ["fill", "softmask", "cast", "pyr_a", "pyr_b",
                "pyr_mask", "blend_lvl", "recon", "writeback"]
    sub_totals = {k: 0.0 for k in sub_keys}

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

        if frame_idx % args.yolo_every == 0:
            mask_a_src = segmenter.predict_mask(frame_a)
            mask_b_src = segmenter.predict_mask(frame_b)
            t_after_yolo = time.perf_counter()
            mask_a_canvas = cv2.remap(mask_a_src, map_ax, map_ay, cv2.INTER_NEAREST)
            mask_b_canvas = cv2.remap(mask_b_src, map_bx, map_by, cv2.INTER_NEAREST)
            union = cv2.bitwise_or(mask_a_canvas, mask_b_canvas)
            union = cv2.dilate(union, dilate_kernel)
            person_mask_bbox = union[y0:y1, x0:x1].copy()
            t_after_mask = time.perf_counter()
            t_yolo += t_after_yolo - t3
            t_maskwarp += t_after_mask - t_after_yolo
        else:
            t_after_mask = t3

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
        if person_mask_bbox.any():
            cost_for_dp[person_mask_bbox > 0] += PERSON_PENALTY

        t5 = time.perf_counter()
        t_cost += t5 - t_after_mask

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
            stitched, sub_t = composite_multiband_fast(
                warped_a, warped_b, static, seam_x_full,
                args.blend_width, args.blend_levels, out_buf,
            )
            for k in sub_keys:
                sub_totals[k] += sub_t[k]
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
            draw_mask_overlay(stitched, person_mask_bbox, static["overlap_bbox"])
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

    if use_multiband:
        print()
        print("[composite breakdown]")
        comp_total = max(sum(sub_totals.values()), 1e-9)
        for k in sub_keys:
            t = sub_totals[k]
            print(f"[timing]   {k:<12s} {t*1000/n:7.2f} ms  "
                  f"({100*t/comp_total:5.1f}% of composite)")
        print(f"[info] Sum of composite sub-stages: {comp_total*1000/n:.2f} ms  "
              f"vs reported composite total: {t_composite*1000/n:.2f} ms")

    print(f"[info] Processed {frame_idx} frames in {elapsed:.2f}s "
          f"({frame_idx / max(elapsed, 1e-6):.2f} fps)")
    print(f"[info] Output written to {args.output}")

    cap_a.release()
    cap_b.release()
    writer.release()


if __name__ == "__main__":
    main()
