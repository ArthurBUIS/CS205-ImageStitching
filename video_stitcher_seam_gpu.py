"""
Real-time video stitching from two fixed cameras — v5: FPS desync handling.

Same as v4, with one structural change: the two input videos are no
longer assumed to have identical frame rates. A new FrameSyncReader
pairs each output frame with the temporally-nearest frame from each
camera, dropping frames from the faster stream as needed.

Why this matters
----------------
If camera A records at 26.43 fps and camera B at 25.37 fps, then after
N frames their timestamps diverge by N * (1/25.37 - 1/26.43) seconds.
At N=600 (~24 seconds at the slower rate), the divergence is roughly
1 second. cv2.VideoCapture.read() doesn't know about timestamps, so
naive lockstep reading produces frame pairs that represent different
real-world moments — visible as e.g. two YOLO masks for the same person
appearing at different positions in the stitched output.

How FrameSyncReader works
-------------------------
At startup it reads each video's nominal FPS via CAP_PROP_FPS. The
slower stream is designated the "driver" — its frames are output
verbatim, one per pipeline iteration. The faster stream is the
"follower"; for each driver frame at timestamp t, the reader
advances through the follower until it lands on the frame whose
timestamp is closest to t.

If the FPS values match within DESYNC_TOLERANCE (default 0.5%), the
reader degrades to identical-lockstep behavior — no frames dropped,
no overhead.

Output FPS = the slower input FPS. We never duplicate frames; we
only drop them. That's the cleaner direction (no stutter) and
matches what the user asked for.

Limits of this approach
-----------------------
This corrects nominal-rate mismatch (the dominant cause of multi-cam
desync). It cannot correct intra-stream jitter (frames arriving
non-uniformly within a stream) or wall-clock drift unrelated to FPS
declarations. For those, audio cross-correlation or hardware sync
would be needed. For typical webcam-class recordings with declared
but mismatched FPS, this is sufficient.
"""

import argparse
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import os

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HOMOGRAPHY_PATH = "homography.npy"
PERSON_CLASS_ID = 0
PERSON_PENALTY = 1e8
EDGE_PENALTY = 1e6

# If two FPS values are within this fractional tolerance, treat them as equal
# and skip the desync logic entirely.
DESYNC_TOLERANCE = 0.005   # 0.5%


# ---------------------------------------------------------------------------
# *** NEW: FrameSyncReader — handles FPS mismatch ***
# ---------------------------------------------------------------------------

class FrameSyncReader:
    """
    Reads paired frames from two cv2.VideoCapture objects whose nominal
    FPS may differ. The slower stream is the "driver" (one frame per
    output tick); the faster stream is the "follower" (advance to the
    closest-in-time frame, drop intermediates).

    If the two FPS values match within DESYNC_TOLERANCE, falls through
    to a plain lockstep read with zero overhead.

    Usage:
        reader = FrameSyncReader(cap_a, cap_b, fps_a, fps_b)
        print(reader.summary())
        while True:
            ok, frame_a, frame_b = reader.read()
            if not ok:
                break
            ...
        print(reader.summary_post())   # optional: drop counts

    Properties exposed:
        output_fps : the FPS to use for the output video writer.
    """

    def __init__(self, cap_a, cap_b, fps_a, fps_b):
        self.cap_a = cap_a
        self.cap_b = cap_b
        self.fps_a = float(fps_a)
        self.fps_b = float(fps_b)

        # Detect mismatch.
        if self.fps_a <= 0 or self.fps_b <= 0:
            raise RuntimeError(f"Invalid FPS values: A={self.fps_a}, B={self.fps_b}")

        rel_diff = abs(self.fps_a - self.fps_b) / max(self.fps_a, self.fps_b)
        self.desync_active = rel_diff > DESYNC_TOLERANCE

        if self.desync_active:
            # Driver = slower (we read 1:1 from it). Follower = faster.
            if self.fps_a < self.fps_b:
                self._driver_label = "A"
                self._follower_label = "B"
                self._fps_driver = self.fps_a
                self._fps_follower = self.fps_b
            else:
                self._driver_label = "B"
                self._follower_label = "A"
                self._fps_driver = self.fps_b
                self._fps_follower = self.fps_a
        else:
            self._driver_label = None
            self._follower_label = None
            self._fps_driver = min(self.fps_a, self.fps_b)
            self._fps_follower = max(self.fps_a, self.fps_b)

        self.output_fps = self._fps_driver

        # Counters (used for stats and for the matching arithmetic).
        self._driver_idx = 0          # next driver index to read
        self._follower_idx = 0        # next follower index to read
        self._dropped_count = 0       # number of follower frames discarded

    def summary(self):
        if self.desync_active:
            return (f"[sync] FPS mismatch detected: A={self.fps_a:.3f}, "
                    f"B={self.fps_b:.3f} (diff={100*abs(self.fps_a-self.fps_b)/max(self.fps_a, self.fps_b):.2f}%). "
                    f"Driver={self._driver_label} ({self._fps_driver:.3f} fps), "
                    f"Follower={self._follower_label} ({self._fps_follower:.3f} fps). "
                    f"Output FPS = {self.output_fps:.3f}.")
        else:
            return (f"[sync] FPS match (A={self.fps_a:.3f}, B={self.fps_b:.3f}). "
                    f"Lockstep read. Output FPS = {self.output_fps:.3f}.")

    def summary_post(self):
        if self.desync_active:
            return (f"[sync] Read {self._driver_idx} driver frames from "
                    f"{self._driver_label}, {self._follower_idx} follower frames "
                    f"from {self._follower_label}, dropped {self._dropped_count} "
                    f"follower frames to maintain temporal alignment.")
        else:
            return (f"[sync] Read {self._driver_idx} pairs in lockstep "
                    f"(no frames dropped).")

    def read(self):
        """
        Returns (ok, frame_a, frame_b).
        ok is False when either stream runs out.
        """
        if not self.desync_active:
            ok_a, fa = self.cap_a.read()
            ok_b, fb = self.cap_b.read()
            if not (ok_a and ok_b):
                return False, None, None
            self._driver_idx += 1
            self._follower_idx += 1
            return True, fa, fb

        # Desync path.
        # 1. Read the next driver frame.
        cap_driver = self.cap_a if self._driver_label == "A" else self.cap_b
        cap_follower = self.cap_b if self._driver_label == "A" else self.cap_a
        ok_d, frame_driver = cap_driver.read()
        if not ok_d:
            return False, None, None

        # The driver frame's timestamp is (driver_idx) / fps_driver.
        # Note: we use driver_idx BEFORE incrementing, because the frame
        # we just read corresponds to that index.
        driver_t = self._driver_idx / self._fps_driver
        self._driver_idx += 1

        # 2. Advance the follower to the frame whose timestamp is closest
        # to driver_t. The follower's frame i has timestamp i / fps_follower.
        # We want the integer i minimizing |i/fps_follower - driver_t|, i.e.
        # the round() of (driver_t * fps_follower).
        target_follower_idx = int(round(driver_t * self._fps_follower))
        # We must read at least once (can't skip the current frame and then
        # not consume any), and we must consume exactly enough to reach
        # target_follower_idx + 1 (i.e., next position is target+1, last
        # consumed is target).
        # _follower_idx is the index of the NEXT frame to read.
        # We need to read frames until we've consumed index target_follower_idx.
        while self._follower_idx < target_follower_idx:
            ok_f, _ = cap_follower.read()
            if not ok_f:
                return False, None, None
            self._follower_idx += 1
            self._dropped_count += 1
        # Now read the target frame itself.
        ok_f, frame_follower = cap_follower.read()
        if not ok_f:
            return False, None, None
        self._follower_idx += 1

        # 3. Map back to (frame_a, frame_b) regardless of which is driver.
        if self._driver_label == "A":
            return True, frame_driver, frame_follower
        else:
            return True, frame_follower, frame_driver


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------

def detect_device():
    info = {
        "cuda_available": False,
        "device": "cpu",
        "gpu_name": None,
        "gpu_mem_gb": None,
        "yolo_device": "cpu",
        "composite_device": "cpu",
        "warp_device": "cpu",
        "cost_device": "cpu",
        "mask_device": "cpu",
        "gain_device": "cpu",
    }
    if not torch.cuda.is_available():
        return info
    try:
        probe = torch.zeros(1, device="cuda")
        _ = probe + 1
        del probe
        torch.cuda.synchronize()
    except Exception as e:
        print(f"[device] CUDA reported available but probe failed: {e}")
        return info

    info["cuda_available"] = True
    info["device"] = "cuda"
    info["yolo_device"] = "cuda"
    info["composite_device"] = "cuda"
    info["warp_device"] = "cuda"
    info["cost_device"] = "cuda"
    info["mask_device"] = "cuda"
    info["gain_device"] = "cuda"
    try:
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_mem_gb"] = props.total_memory / (1024 ** 3)
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Homography + canvas + static geometry (unchanged)
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


def find_autocrop_rect(H_b_to_a, shape_a, shape_b, canvas_size, T):
    """
    Build a rectangle from B's two right corners (warped to canvas) and A's
    left edge. The right side of the rectangle is the more-conservative
    (smaller-x) of B's two right corners; the rectangle spans the y-range
    formed by both B right corners; the left side is A's left edge on canvas.

    Returns (x, y, w, h) in canvas coords.
    """
    canvas_w, canvas_h = canvas_size
    h_a, w_a = shape_a[:2]
    h_b, w_b = shape_b[:2]
    tx, ty = float(T[0, 2]), float(T[1, 2])

    # B's right corners after warp to canvas (top_right = warped (w_b, 0),
    # bottom_right = warped (w_b, h_b)).
    b_right_src = np.float32([[w_b, 0], [w_b, h_b]]).reshape(-1, 1, 2)
    b_right_canvas = cv2.perspectiveTransform(b_right_src, H_b_to_a).reshape(-1, 2)
    b_right_canvas += np.array([tx, ty], dtype=np.float32)
    top_right, bottom_right = b_right_canvas[0], b_right_canvas[1]

    # Pick the one with the lower x as (x_1, y_1); the other as (x_2, y_2).
    if top_right[0] <= bottom_right[0]:
        (x_1, y_1), (x_2, y_2) = top_right, bottom_right
    else:
        (x_1, y_1), (x_2, y_2) = bottom_right, top_right

    # Left edge x = A's left edge on canvas (0 in A's local coords + tx).
    x_left = tx

    final_corners = [
        (x_1,    y_1),
        (x_1,    y_2),
        (x_left, y_1),
        (x_left, y_2),
    ]

    xs = [c[0] for c in final_corners]
    ys = [c[1] for c in final_corners]
    x0 = max(0,        int(round(min(xs))))
    y0 = max(0,        int(round(min(ys))))
    x1 = min(canvas_w, int(round(max(xs))))
    y1 = min(canvas_h, int(round(max(ys))))
    return x0, y0, x1 - x0, y1 - y0


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
# Gain compensation
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


def build_gain_tensor(gains_bgr, device):
    t = torch.from_numpy(gains_bgr).to(device).view(1, 3, 1, 1)
    return t


# ---------------------------------------------------------------------------
# GPU warp
# ---------------------------------------------------------------------------

def build_grid_sample_tensor(map_x, map_y, src_shape, device):
    H_src, W_src = src_shape[:2]
    grid_x = 2.0 * map_x / max(W_src - 1, 1) - 1.0
    grid_y = 2.0 * map_y / max(H_src - 1, 1) - 1.0
    grid_np = np.stack([grid_x, grid_y], axis=-1).astype(np.float32)
    grid_t = torch.from_numpy(grid_np).unsqueeze(0).to(device)
    return grid_t


def warp_gpu(frame_bgr_cpu, grid_t, device, gain_t=None, non_blocking=True):
    t = torch.from_numpy(frame_bgr_cpu).to(device, non_blocking=non_blocking)
    t = t.permute(2, 0, 1).unsqueeze(0).float()
    if gain_t is not None:
        t = t * gain_t
        t = t.clamp(0, 255)
    warped = F.grid_sample(
        t, grid_t,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    warped = warped.clamp(0, 255).to(torch.uint8)
    return warped


# ---------------------------------------------------------------------------
# GPU mask warp + dilation
# ---------------------------------------------------------------------------

def warp_mask_gpu(mask_t, grid_t):
    m = mask_t.float().unsqueeze(0).unsqueeze(0)
    if m.max() > 1.5:
        m = m / 255.0
    warped = F.grid_sample(
        m, grid_t,
        mode="nearest",
        padding_mode="zeros",
        align_corners=True,
    )
    out = (warped[0, 0] * 255).clamp(0, 255).to(torch.uint8)
    return out


def dilate_gpu(mask_u8_t, radius):
    if radius <= 0:
        return mask_u8_t
    k = 2 * radius + 1
    m = mask_u8_t.float().unsqueeze(0).unsqueeze(0)
    dilated = F.max_pool2d(m, kernel_size=k, stride=1, padding=radius)
    return dilated[0, 0].to(torch.uint8)


# ---------------------------------------------------------------------------
# GPU cost + EMA
# ---------------------------------------------------------------------------

def compute_cost_and_ema_gpu(warped_a_t, warped_b_t, overlap_in_bbox_t,
                             cost_ema_t, ema_alpha, person_mask_bbox_t,
                             fg_mask_bbox_t, fg_penalty,
                             overlap_bbox):
    """
    GPU cost + EMA + penalty injection.

    Penalty hierarchy (additive on cost_ema):
        photometric (0 - 1e5)
        + fg_penalty (default 5e7) where fg_mask AND NOT person_mask
        + PERSON_PENALTY (1e8) where person_mask
    """
    x0, y0, x1, y1 = overlap_bbox
    wa_bb = warped_a_t[0, :, y0:y1, x0:x1].float()
    wb_bb = warped_b_t[0, :, y0:y1, x0:x1].float()

    diff = wa_bb - wb_bb
    photo_cost = (diff * diff).sum(dim=0)

    overlap_mask = overlap_in_bbox_t > 0
    photo_cost = torch.where(overlap_mask, photo_cost,
                             torch.tensor(1e9, device=photo_cost.device,
                                          dtype=photo_cost.dtype))

    if cost_ema_t is None or cost_ema_t.shape != photo_cost.shape:
        cost_ema_t = photo_cost.clone()
    else:
        if ema_alpha >= 1.0:
            cost_ema_t = photo_cost.clone()
        else:
            cost_ema_t = ema_alpha * photo_cost + (1.0 - ema_alpha) * cost_ema_t

    cost_for_dp = cost_ema_t.clone()

    # FG penalty first (lower priority), person second (higher).
    # A pixel that's both gets the sum, but person (1e8) >> fg (5e7) so
    # the effect is the same as taking the max.
    if fg_mask_bbox_t is not None:
        if person_mask_bbox_t is not None:
            fg_only = fg_mask_bbox_t & (~person_mask_bbox_t)
        else:
            fg_only = fg_mask_bbox_t
        cost_for_dp = torch.where(fg_only > 0,
                                  cost_for_dp + fg_penalty,
                                  cost_for_dp)
    if person_mask_bbox_t is not None:
        cost_for_dp = torch.where(person_mask_bbox_t > 0,
                                  cost_for_dp + PERSON_PENALTY,
                                  cost_for_dp)

    cost_for_dp_cpu = cost_for_dp.cpu().numpy()
    return cost_ema_t, cost_for_dp_cpu


# ---------------------------------------------------------------------------
# DP seam + utilities
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# GPU composite
# ---------------------------------------------------------------------------

_PYR_KERNEL_1D = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32) / 16.0


def _get_pyr_kernel_2d(device):
    k1 = _PYR_KERNEL_1D.to(device)
    k2 = k1[:, None] * k1[None, :]
    return k2[None, None, :, :]


def _pyr_down_torch(x, kernel2d):
    C, H, W = x.shape
    x_pad = F.pad(x[None, :, :, :], (2, 2, 2, 2), mode="replicate")
    kern = kernel2d.expand(C, 1, 5, 5)
    blurred = F.conv2d(x_pad, kern, groups=C)
    down = blurred[:, :, ::2, ::2]
    return down[0]


def _pyr_up_torch(x, target_hw, kernel2d):
    C, Hs, Ws = x.shape
    up = x.new_zeros((C, Hs * 2, Ws * 2))
    up[:, ::2, ::2] = x
    up_pad = F.pad(up[None, :, :, :], (2, 2, 2, 2), mode="replicate")
    kern = (kernel2d * 4.0).expand(C, 1, 5, 5)
    blurred = F.conv2d(up_pad, kern, groups=C)
    Th, Tw = target_hw
    out = blurred[0, :, :Th, :Tw]
    return out


def _build_gaussian_pyramid_torch(x, levels, kernel2d):
    gp = [x]
    for _ in range(levels):
        gp.append(_pyr_down_torch(gp[-1], kernel2d))
    return gp


def _build_laplacian_pyramid_torch(x, levels, kernel2d):
    gp = _build_gaussian_pyramid_torch(x, levels, kernel2d)
    lp = []
    for i in range(levels):
        up = _pyr_up_torch(gp[i + 1], gp[i].shape[1:], kernel2d)
        lp.append(gp[i] - up)
    lp.append(gp[levels])
    return lp


def _reconstruct_from_laplacian_torch(lp, kernel2d):
    img = lp[-1]
    for level in reversed(lp[:-1]):
        img = _pyr_up_torch(img, level.shape[1:], kernel2d)
        img = img + level
    return img


def composite_multiband_gpu_resident(warped_a_t, warped_b_t, static, seam_x_full,
                                     blend_width, blend_levels, out_buf,
                                     gpu_ctx):
    device = gpu_ctx["device"]
    kernel2d = gpu_ctx["kernel2d"]
    x0, y0, x1, y1 = static["overlap_bbox"]

    a_full_t = warped_a_t[0].permute(1, 2, 0).contiguous()
    b_full_t = warped_b_t[0].permute(1, 2, 0).contiguous()

    H_canvas, W_canvas = a_full_t.shape[:2]
    out_t = torch.zeros((H_canvas, W_canvas, 3), dtype=torch.uint8,
                        device=device)
    only_a_m = gpu_ctx["only_a_u8_t"].unsqueeze(-1) > 0
    only_b_m = gpu_ctx["only_b_u8_t"].unsqueeze(-1) > 0
    out_t = torch.where(only_a_m, a_full_t, out_t)
    out_t = torch.where(only_b_m, b_full_t, out_t)

    a_bb_t = a_full_t[y0:y1, x0:x1].contiguous()
    b_bb_t = b_full_t[y0:y1, x0:x1].contiguous()
    only_a_bb = gpu_ctx["only_a_in_bbox_t"].unsqueeze(-1) > 0
    only_b_bb = gpu_ctx["only_b_in_bbox_t"].unsqueeze(-1) > 0
    a_bb_filled = torch.where(only_b_bb, b_bb_t, a_bb_t)
    b_bb_filled = torch.where(only_a_bb, a_bb_t, b_bb_t)

    H_bb = y1 - y0
    W_bb = x1 - x0
    bbox_shape = (H_bb, W_bb)
    mask_f32_np = build_soft_mask_fast(
        seam_x_full, bbox_shape, static, blend_width,
    )
    mask_t = torch.from_numpy(mask_f32_np).to(device, non_blocking=True)

    a_f = a_bb_filled.permute(2, 0, 1).float()
    b_f = b_bb_filled.permute(2, 0, 1).float()
    m_f = mask_t.unsqueeze(0)

    min_dim = min(H_bb, W_bb)
    max_levels = max(1, int(np.log2(min_dim)) - 2)
    levels = min(blend_levels, max_levels)

    lp_a = _build_laplacian_pyramid_torch(a_f, levels, kernel2d)
    lp_b = _build_laplacian_pyramid_torch(b_f, levels, kernel2d)
    gp_m = _build_gaussian_pyramid_torch(m_f, levels, kernel2d)

    blended_lp = []
    for la, lb, gm in zip(lp_a, lp_b, gp_m):
        blended_lp.append(la * gm + lb * (1.0 - gm))

    recon = _reconstruct_from_laplacian_torch(blended_lp, kernel2d)
    recon = recon.clamp(0, 255)
    blended_bb_t = recon.permute(1, 2, 0).to(torch.uint8).contiguous()

    valid_bb = gpu_ctx["valid_in_bbox_t"].unsqueeze(-1) > 0
    out_bbox_slice = out_t[y0:y1, x0:x1]
    out_bbox_slice = torch.where(valid_bb, blended_bb_t, out_bbox_slice)
    out_t[y0:y1, x0:x1] = out_bbox_slice

    result_np = out_t.cpu().numpy()
    np.copyto(out_buf, result_np)
    return out_buf


# ---------------------------------------------------------------------------
# CPU fallback composite + cost
# ---------------------------------------------------------------------------

def compute_cost_fast_cpu(wa_bb, wb_bb, overlap_in_bbox, cost_scratch):
    diff = cv2.absdiff(wa_bb, wb_bb)
    diff_f = diff.astype(np.float32, copy=False)
    np.multiply(diff_f, diff_f, out=cost_scratch)
    cost = cost_scratch.sum(axis=2)
    cost[overlap_in_bbox == 0] = 1e9
    return cost


def fill_invalid_with_other_cpu(a_bb_u8, b_bb_u8, static):
    only_a = static["only_a_in_bbox"]
    only_b = static["only_b_in_bbox"]
    a_out = a_bb_u8.copy()
    b_out = b_bb_u8.copy()
    cv2.copyTo(b_bb_u8, only_b, a_out)
    cv2.copyTo(a_bb_u8, only_a, b_out)
    return a_out, b_out


def build_gaussian_pyramid_cpu(img_f32, levels):
    gp = [img_f32]
    for _ in range(levels):
        gp.append(cv2.pyrDown(gp[-1]))
    return gp


def build_laplacian_pyramid_cpu(img_f32, levels):
    gp = build_gaussian_pyramid_cpu(img_f32, levels)
    lp = []
    for i in range(levels):
        up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
        lp.append(gp[i] - up)
    lp.append(gp[levels])
    return lp


def reconstruct_from_laplacian_cpu(lp):
    img = lp[-1]
    for level in reversed(lp[:-1]):
        img = cv2.pyrUp(img, dstsize=(level.shape[1], level.shape[0]))
        img = img + level
    return img


def blend_pyramids_fast_cpu(lp_a, lp_b, gp_m):
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


def composite_multiband_cpu(warped_a, warped_b, static, seam_x_full,
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
    a_bb_pad, b_bb_pad = fill_invalid_with_other_cpu(a_bb, b_bb, static)
    mask_f32 = build_soft_mask_fast(seam_x_full, bbox_shape, static, blend_width)
    min_dim = min(a_bb_pad.shape[:2])
    max_levels = max(1, int(np.log2(min_dim)) - 2)
    levels = min(blend_levels, max_levels)
    a_f = a_bb_pad.astype(np.float32)
    b_f = b_bb_pad.astype(np.float32)
    lp_a = build_laplacian_pyramid_cpu(a_f, levels)
    lp_b = build_laplacian_pyramid_cpu(b_f, levels)
    gp_m = build_gaussian_pyramid_cpu(mask_f32, levels)
    blended_lp = blend_pyramids_fast_cpu(lp_a, lp_b, gp_m)
    recon = reconstruct_from_laplacian_cpu(blended_lp)
    np.clip(recon, 0, 255, out=recon)
    blended_bb = recon.astype(np.uint8)
    valid_in_bbox = cv2.bitwise_or(static["mask_a_in_bbox"],
                                   static["mask_b_in_bbox"])
    cv2.copyTo(blended_bb, valid_in_bbox, out_buf[y0:y1, x0:x1])
    return out_buf


# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------

# Default COCO classes for static foreground: furniture and large electronics.
# 56=chair, 57=couch, 59=bed, 60=dining table, 62=tv, 63=laptop, 73=book.
DEFAULT_FG_CLASS_IDS = [56, 57, 59, 60, 62, 63, 73]


def compute_fg_mask_seg_gpu(segmenter, frame_a, frame_b, class_ids,
                             grid_a_t, grid_b_t, dilate_radius,
                             overlap_bbox, overlap_in_bbox_t):
    """
    Static foreground mask via instance segmentation (YOLO).
    Runs the segmenter on each ORIGINAL frame asking for `class_ids`,
    warps each mask to the canvas via grid_sample, unions, dilates,
    and crops to the overlap bbox.

    Returns a (H_bb, W_bb) uint8 tensor on GPU (0 or 255).
    """
    H_a, W_a = frame_a.shape[:2]
    H_b, W_b = frame_b.shape[:2]

    mask_a_src_t = segmenter.predict_classes_mask_gpu(frame_a, (H_a, W_a), class_ids)
    mask_b_src_t = segmenter.predict_classes_mask_gpu(frame_b, (H_b, W_b), class_ids)

    mask_a_canvas_t = warp_mask_gpu(mask_a_src_t, grid_a_t)
    mask_b_canvas_t = warp_mask_gpu(mask_b_src_t, grid_b_t)

    union_t = torch.bitwise_or(mask_a_canvas_t, mask_b_canvas_t)
    union_t = dilate_gpu(union_t, dilate_radius)

    x0, y0, x1, y1 = overlap_bbox
    fg_mask_bbox_t = union_t[y0:y1, x0:x1].contiguous()
    # AND with overlap shape inside bbox (from passed overlap_in_bbox_t,
    # which is the bbox-sized version).
    fg_mask_bbox_t = torch.where(overlap_in_bbox_t > 0,
                                 fg_mask_bbox_t,
                                 torch.zeros_like(fg_mask_bbox_t))
    return fg_mask_bbox_t

class PersonSegmenter:
    def __init__(self, weights_path: str, device: str = "cpu"):
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError("pip install ultralytics") from e
        self.model = YOLO(weights_path)
        self.device = device
        try:
            self.model.to(device)
        except Exception as e:
            print(f"[yolo] Could not move model to {device}: {e}.")

    def predict_mask(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        results = self.model.predict(
            frame_bgr, classes=[PERSON_CLASS_ID],
            verbose=False, retina_masks=False,
            device=self.device,
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

    def predict_classes_mask_gpu(self, frame_bgr, target_hw,
                                 class_ids=(PERSON_CLASS_ID,)):
        """
        Returns a (H, W) uint8 tensor on GPU with the union of all detected
        instances of any class in `class_ids`. Default = persons only.
        """
        H_tgt, W_tgt = target_hw
        results = self.model.predict(
            frame_bgr, classes=list(class_ids),
            verbose=False, retina_masks=False,
            device=self.device,
        )
        if not results:
            return torch.zeros((H_tgt, W_tgt), dtype=torch.uint8,
                               device=self.device)
        r = results[0]
        if r.masks is None or r.masks.data is None or len(r.masks.data) == 0:
            return torch.zeros((H_tgt, W_tgt), dtype=torch.uint8,
                               device=self.device)
        mdata = r.masks.data
        merged = (mdata > 0.5).any(dim=0).float()
        m = merged.unsqueeze(0).unsqueeze(0)
        m = F.interpolate(m, size=(H_tgt, W_tgt), mode="nearest")
        mask = (m[0, 0] * 255).to(torch.uint8)
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
# Threaded video writer
# ---------------------------------------------------------------------------

class ThreadedVideoWriter:
    _SENTINEL = object()

    def __init__(self, writer, queue_depth=4):
        self.writer = writer
        self.q = queue.Queue(maxsize=queue_depth)
        self.exception = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while True:
                item = self.q.get()
                if item is self._SENTINEL:
                    break
                self.writer.write(item)
        except Exception as e:
            self.exception = e

    def write(self, frame_bgr):
        self.q.put(frame_bgr.copy())
        if self.exception is not None:
            raise self.exception

    def close(self):
        self.q.put(self._SENTINEL)
        self.thread.join(timeout=30)
        if self.exception is not None:
            raise self.exception
        self.writer.release()


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
    parser.add_argument("--autocrop", action="store_true",
                        help="Crop output to the largest axis-aligned "
                             "rectangle inside the stitched canvas.")
    parser.add_argument("--yolo_every", type=int, default=3)
    parser.add_argument("--mask_dilate", type=int, default=15)
    parser.add_argument("--seam_downscale", type=int, default=4)
    parser.add_argument("--yolo_weights", default="yolov8n-seg.pt")
    parser.add_argument("--no_gain_comp", action="store_true")
    parser.add_argument("--cost_ema", type=float, default=0.4)
    parser.add_argument("--no_cost_ema", action="store_true")
    parser.add_argument("--blend_width", type=int, default=80)
    parser.add_argument("--blend_levels", type=int, default=5)
    parser.add_argument("--seam_lambda", type=float, default=5.0)
    parser.add_argument("--seam_edge_margin", type=int, default=50)
    # Static foreground (segmentation-based) flags.
    parser.add_argument("--no_fg", action="store_true",
                        help="Disable static foreground detection.")
    parser.add_argument("--fg_classes", type=int, nargs="+",
                        default=DEFAULT_FG_CLASS_IDS,
                        help="COCO class IDs for static foreground "
                             "(default: chair, couch, bed, table, tv, "
                             "laptop, book).")
    parser.add_argument("--fg_dilate", type=int, default=10,
                        help="FG mask dilation radius in px (default 10).")
    parser.add_argument("--fg_penalty", type=float, default=5e7,
                        help="Cost penalty for FG pixels (default 5e7).")
    parser.add_argument("--fg_recompute_seconds", type=float, default=0.0,
                        help="Seconds between FG recomputations "
                             "(0 = startup only).")
    args = parser.parse_args()

    dev = detect_device()
    print("=" * 60)
    print("[device] Device detection")
    print("=" * 60)
    if dev["cuda_available"]:
        print(f"[device] CUDA: AVAILABLE")
        print(f"[device] GPU: {dev['gpu_name']}  ({dev['gpu_mem_gb']:.1f} GB)")
        print(f"[device] YOLO          -> GPU")
        print(f"[device] gain          -> GPU (fused with warp upload)")
        print(f"[device] warp          -> GPU (PyTorch grid_sample)")
        print(f"[device] mask warp+dil -> GPU (grid_sample + max_pool2d)")
        print(f"[device] cost+ema      -> GPU (PyTorch)")
        print(f"[device] composite     -> GPU (PyTorch, resident)")
        print(f"[device] DP seam       -> CPU (tiny)")
        print(f"[device] decode        -> CPU")
        print(f"[device] write         -> threaded CPU")
    else:
        print(f"[device] CUDA: NOT AVAILABLE — running entirely on CPU")
    print("=" * 60)

    if not dev["cuda_available"]:
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

    print(f"[info] OpenCV: {cv2.getNumberOfCPUs()} CPUs, "
          f"using {cv2.getNumThreads()} threads.")
    print(f"[info] torch num_threads = {torch.get_num_threads()}")

    ema_eff = 1.0 if args.no_cost_ema else float(args.cost_ema)
    print(f"[info] yolo_every={args.yolo_every}  "
          f"mask_dilate={args.mask_dilate}  "
          f"DP_downscale={args.seam_downscale}  "
          f"gain_comp={not args.no_gain_comp}  "
          f"cost_ema={ema_eff}  "
          f"blend_width={args.blend_width}  "
          f"blend_levels={args.blend_levels}")
    print(f"[info] seam_lambda={args.seam_lambda}  "
          f"seam_edge_margin={args.seam_edge_margin}")

    # Resolve relative video paths by searching upward from this script's
    # directory so invoking the script from a subdirectory still finds
    # project-local `videos/...` paths.
    def _resolve_relpath(p):
        if os.path.isabs(p):
            return p
        if os.path.exists(p):
            return p
        cur = os.path.dirname(__file__)
        for _ in range(6):
            candidate = os.path.join(cur, p)
            if os.path.exists(candidate):
                return candidate
            cur = os.path.dirname(cur)
        return p

    args.video_a = _resolve_relpath(args.video_a)
    args.video_b = _resolve_relpath(args.video_b)

    cap_a = cv2.VideoCapture(args.video_a)
    cap_b = cv2.VideoCapture(args.video_b)
    if not cap_a.isOpened() or not cap_b.isOpened():
        raise RuntimeError("Could not open one of the input videos.")
    fps_a = cap_a.get(cv2.CAP_PROP_FPS) or 25.0
    fps_b = cap_b.get(cv2.CAP_PROP_FPS) or 25.0

    # *** NEW: FPS-aware frame reader ***
    sync_reader = FrameSyncReader(cap_a, cap_b, fps_a, fps_b)
    print(sync_reader.summary())

    # Read the first paired frame (used for homography + gain seed).
    ok, frame_a, frame_b = sync_reader.read()
    if not ok:
        raise RuntimeError("Could not read first frame pair.")

    print("[info] Estimating homography from first frame pair...")
    H_b_to_a = estimate_homography(frame_a, frame_b)
    np.save(HOMOGRAPHY_PATH, H_b_to_a)

    canvas_size, T, H_b_to_canvas, H_a_to_canvas = compute_canvas(
        frame_a.shape, frame_b.shape, H_b_to_a
    )
    print(f"[info] Canvas size: {canvas_size[0]} x {canvas_size[1]}")

    crop_rect = None
    if args.autocrop:
        crop_rect = find_autocrop_rect(
            H_b_to_a, frame_a.shape, frame_b.shape, canvas_size, T,
        )
        cx, cy, cw, ch = crop_rect
        print(f"[info] Autocrop: x={cx} y={cy} size={cw}x{ch} "
              f"(from full canvas {canvas_size[0]}x{canvas_size[1]})")

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
    gain_a_t = None
    gain_b_t = None
    if not args.no_gain_comp:
        print("[info] Computing gain compensation from first frame pair...")
        wa0 = cv2.remap(frame_a, map_ax, map_ay, cv2.INTER_LINEAR)
        wb0 = cv2.remap(frame_b, map_bx, map_by, cv2.INTER_LINEAR)
        gains_a, gains_b = compute_gain_compensation(
            wa0, wb0, static["overlap_bbox"], static["overlap_in_bbox"]
        )
        print(f"[info] gains_a = [{gains_a[0]:.3f}, {gains_a[1]:.3f}, {gains_a[2]:.3f}]")
        print(f"[info] gains_b = [{gains_b[0]:.3f}, {gains_b[1]:.3f}, {gains_b[2]:.3f}]")
        if dev["cuda_available"]:
            gain_a_t = build_gain_tensor(gains_a, torch.device("cuda"))
            gain_b_t = build_gain_tensor(gains_b, torch.device("cuda"))
        else:
            lut_a = build_gain_lut(gains_a)
            lut_b = build_gain_lut(gains_b)

    print(f"[info] Loading YOLO weights: {args.yolo_weights}")
    segmenter = PersonSegmenter(args.yolo_weights, device=dev["yolo_device"])
    dilate_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * args.mask_dilate + 1, 2 * args.mask_dilate + 1),
    )

    gpu_ctx = None
    grid_a_t = None
    grid_b_t = None
    overlap_in_bbox_t = None
    cost_ema_t = None
    # Defaults for foreground (FG) variables when running on CPU
    use_fg = False
    fg_mask_bbox_t = None
    fg_recompute_frames = 0
    if dev["cuda_available"]:
        torch_device = torch.device("cuda")
        valid_in_bbox_np = cv2.bitwise_or(static["mask_a_in_bbox"],
                                          static["mask_b_in_bbox"])
        gpu_ctx = {
            "device": torch_device,
            "kernel2d": _get_pyr_kernel_2d(torch_device),
            "only_a_u8_t": torch.from_numpy(static["only_a_u8"]).to(torch_device),
            "only_b_u8_t": torch.from_numpy(static["only_b_u8"]).to(torch_device),
            "only_a_in_bbox_t": torch.from_numpy(static["only_a_in_bbox"]).to(torch_device),
            "only_b_in_bbox_t": torch.from_numpy(static["only_b_in_bbox"]).to(torch_device),
            "valid_in_bbox_t": torch.from_numpy(valid_in_bbox_np).to(torch_device),
        }
        grid_a_t = build_grid_sample_tensor(map_ax, map_ay, frame_a.shape, torch_device)
        grid_b_t = build_grid_sample_tensor(map_bx, map_by, frame_b.shape, torch_device)
        overlap_in_bbox_t = torch.from_numpy(static["overlap_in_bbox"]).to(torch_device)
        print("[device] GPU contexts (gain + warp + mask + cost + composite) initialized.")
        # --- Static foreground mask (segmentation-based) -------------------
        use_fg = not args.no_fg and dev["cuda_available"]
        fg_mask_bbox_t = None
        if use_fg:
            print(f"[info] Computing static FG mask via YOLO segmentation. "
                f"Classes: {args.fg_classes}")
            t0 = time.time()
            fg_mask_bbox_t = compute_fg_mask_seg_gpu(
                segmenter, frame_a, frame_b, args.fg_classes,
                grid_a_t, grid_b_t, args.fg_dilate,
                static["overlap_bbox"], overlap_in_bbox_t,
            )
            coverage = (fg_mask_bbox_t > 0).float().mean().item() * 100
            print(f"[info] FG mask computed in {(time.time()-t0)*1000:.1f} ms  "
                f"({coverage:.1f}% of bbox flagged).")
        elif not dev["cuda_available"]:
            print("[info] FG detection disabled (CPU mode not implemented for seg).")

        fg_recompute_frames = (
            int(round(args.fg_recompute_seconds * sync_reader.output_fps))
            if args.fg_recompute_seconds > 0 else 0
        )
        if use_fg and fg_recompute_frames > 0:
            print(f"[info] FG recompute every {fg_recompute_frames} frames "
                f"(~{args.fg_recompute_seconds}s).")
            
    W, H = canvas_size
    out_buf      = np.zeros((H, W, 3), dtype=np.uint8)
    cost_scratch = np.empty((bbox_shape[0], bbox_shape[1], 3), dtype=np.float32)
    person_mask_bbox = np.zeros(bbox_shape, dtype=np.uint8)
    person_mask_bbox_t = None
    if dev["cuda_available"]:
        person_mask_bbox_t = torch.zeros(bbox_shape, dtype=torch.uint8,
                                         device=gpu_ctx["device"])
    cost_ema = None
    seam_prev_small = None

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # *** Output FPS now comes from the FrameSyncReader ***
    output_size = (crop_rect[2], crop_rect[3]) if crop_rect else canvas_size
    raw_writer = cv2.VideoWriter(args.output, fourcc, sync_reader.output_fps, output_size)
    if not raw_writer.isOpened():
        raise RuntimeError(f"Could not open output writer for {args.output}")
    writer = ThreadedVideoWriter(raw_writer, queue_depth=4)

    t_read = t_warp = t_gain = t_yolo = t_maskwarp = 0.0
    t_cost = t_seam = t_composite = t_write = 0.0
    frame_idx = 0
    t_start = time.time()

    # Reuse the homography frame as the first iteration; subsequent iterations
    # pull from the sync reader.
    pending_first_pair = (frame_a, frame_b)

    try:
        while True:
            tt = time.perf_counter()
            if pending_first_pair is not None:
                frame_a, frame_b = pending_first_pair
                pending_first_pair = None
            else:
                ok, frame_a, frame_b = sync_reader.read()
                if not ok:
                    break
            t1 = time.perf_counter()
            # Periodic FG recompute.
            if (use_fg and fg_recompute_frames > 0 and frame_idx > 0
                    and frame_idx % fg_recompute_frames == 0):
                fg_mask_bbox_t = compute_fg_mask_seg_gpu(
                    segmenter, frame_a, frame_b, args.fg_classes,
                    grid_a_t, grid_b_t, args.fg_dilate,
                    static["overlap_bbox"], overlap_in_bbox_t,
                )
            if dev["cuda_available"]:
                warped_a_t = warp_gpu(frame_a, grid_a_t, gpu_ctx["device"],
                                      gain_t=gain_a_t)
                warped_b_t = warp_gpu(frame_b, grid_b_t, gpu_ctx["device"],
                                      gain_t=gain_b_t)
            else:
                if lut_a is not None:
                    frame_a_g = apply_gain_lut(frame_a, lut_a)
                    frame_b_g = apply_gain_lut(frame_b, lut_b)
                else:
                    frame_a_g = frame_a
                    frame_b_g = frame_b
                t_gain_end = time.perf_counter()
                t_gain += t_gain_end - t1
                warped_a = cv2.remap(frame_a_g, map_ax, map_ay, cv2.INTER_LINEAR)
                warped_b = cv2.remap(frame_b_g, map_bx, map_by, cv2.INTER_LINEAR)
            t3 = time.perf_counter()
            if dev["cuda_available"]:
                t_warp += t3 - t1
            else:
                t_warp += t3 - t_gain_end

            if frame_idx % args.yolo_every == 0:
                if dev["cuda_available"]:
                    mask_a_src_t = segmenter.predict_classes_mask_gpu(
                        frame_a, frame_a.shape[:2],
                    )
                    mask_b_src_t = segmenter.predict_classes_mask_gpu(
                        frame_b, frame_b.shape[:2],
                    )
                    t_after_yolo = time.perf_counter()
                    mask_a_canvas_t = warp_mask_gpu(mask_a_src_t, grid_a_t)
                    mask_b_canvas_t = warp_mask_gpu(mask_b_src_t, grid_b_t)
                    union_t = torch.bitwise_or(mask_a_canvas_t, mask_b_canvas_t)
                    union_t = dilate_gpu(union_t, args.mask_dilate)
                    person_mask_bbox_t = union_t[y0:y1, x0:x1].contiguous()
                    if args.debug_mask:
                        person_mask_bbox = person_mask_bbox_t.cpu().numpy()
                    t_after_mask = time.perf_counter()
                else:
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

            if dev["cuda_available"]:
                has_person = (person_mask_bbox_t.any().item()
                              if person_mask_bbox_t is not None else False)
                cost_ema_t, cost_for_dp = compute_cost_and_ema_gpu(
                    warped_a_t, warped_b_t, overlap_in_bbox_t,
                    cost_ema_t, ema_eff,
                    person_mask_bbox_t if has_person else None,
                    fg_mask_bbox_t if use_fg else None,
                    args.fg_penalty,
                    static["overlap_bbox"],
                )
            else:
                wa_bb = warped_a[y0:y1, x0:x1]
                wb_bb = warped_b[y0:y1, x0:x1]
                photo_cost = compute_cost_fast_cpu(
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
                if person_mask_bbox.any():
                    cost_for_dp[person_mask_bbox > 0] += PERSON_PENALTY

            add_edge_margin_penalty(cost_for_dp, args.seam_edge_margin)

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

            if gpu_ctx is not None:
                stitched = composite_multiband_gpu_resident(
                    warped_a_t, warped_b_t, static, seam_x_full,
                    args.blend_width, args.blend_levels, out_buf,
                    gpu_ctx,
                )
            else:
                stitched = composite_multiband_cpu(
                    warped_a, warped_b, static, seam_x_full,
                    args.blend_width, args.blend_levels, out_buf,
                )
            if args.debug_mask:
                # Layer FG (yellow, bottom) under person (red, top).
                if use_fg and fg_mask_bbox_t is not None:
                    fg_cpu = fg_mask_bbox_t.cpu().numpy()
                    draw_mask_overlay(stitched, fg_cpu, static["overlap_bbox"],
                                      color=(0, 255, 255), alpha=0.25)
                draw_mask_overlay(stitched, person_mask_bbox, static["overlap_bbox"])
            if args.debug_seam:
                draw_seam_overlay(stitched, seam_x_full, static["overlap_bbox"])
            t7 = time.perf_counter()
            t_composite += t7 - t6

            if crop_rect is not None:
                cx, cy, cw, ch = crop_rect
                stitched = stitched[cy:cy + ch, cx:cx + cw]
            writer.write(stitched)
            t8 = time.perf_counter()
            t_write += t8 - t7

            t_read += t1 - tt

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        writer.close()

    elapsed = time.time() - t_start
    n = max(frame_idx, 1)
    stages = [
        ("read",          t_read),
        ("gain",          t_gain),
        ("warp",          t_warp),
        ("yolo",          t_yolo),
        ("mask warp+dil", t_maskwarp),
        ("cost + ema",    t_cost),
        ("dp seam",       t_seam),
        ("composite",     t_composite),
        ("write (enq)",   t_write),
    ]
    total = max(sum(t for _, t in stages), 1e-9)
    print()
    for name, t in stages:
        print(f"[timing] {name:<14s} {t*1000/n:7.2f} ms  "
              f"({100*t/total:5.1f}%)")
    if dev["cuda_available"]:
        print("[info] On GPU path, gain is folded into warp.")
    print(f"[info] Processed {frame_idx} frames in {elapsed:.2f}s "
          f"({frame_idx / max(elapsed, 1e-6):.2f} fps)")
    print(sync_reader.summary_post())
    print(f"[info] Output written to {args.output}")

    cap_a.release()
    cap_b.release()


if __name__ == "__main__":
    main()
