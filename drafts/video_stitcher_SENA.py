# Not working for the moment

"""
video_stitcher_SENA.py
======================
Fixed-camera video stitching pipeline based on SENA:

  "Seamlessly Natural: Image Stitching with Natural Appearance Preservation"
  Tchana, Fotso, Hendricks, Bobda — Technologies 2026, 14(3), 186.
  https://doi.org/10.3390/technologies14030186

How it differs from video_stitcher.py (homography-based)
---------------------------------------------------------
| Aspect              | v1 / v2 (homography)      | SENA                        |
|---------------------|---------------------------|-----------------------------|
| Alignment model     | Global homography (8 DOF) | Global affine + 2×2 local   |
|                     |                           | affine grid + FFD field     |
| Distortion          | Stretching / bulging on   | Reduced: affine preserves   |
|                     | non-planar scenes         | parallelism & aspect ratio  |
| Seam strategy       | Multi-band pyramid blend  | Anchor-based vertical slice |
|                     | over overlap strip        | + linear alpha + Gaussian   |
| Parallax handling   | None                      | Disparity-clustering zone   |
|                     |                           | detection (Algorithm 1)     |
| Feature extractor   | SIFT                      | XFeat (CVPR 2024) w/ SIFT  |
|                     |                           | fallback                    |

Fixed-camera optimisation
-------------------------
As in v1/v2, the calibration (feature matching + warping geometry) is
computed ONCE from the first N frames and reused for every subsequent frame.
Per-frame processing is limited to:
  1. Warp source image using the pre-computed remap maps (cv2.remap)
  2. Place target image on canvas
  3. Reconstruct via the pre-computed anchor chain + zone boundaries

Installation
------------
  pip install opencv-contrib-python numpy

  # For XFeat (recommended, as in the paper):
  pip install torch accelerated-features

Usage
-----
  # Full run — calibrate on first 5 frames, then stitch:
  python video_stitcher_SENA.py --left left.mp4 --right right.mp4 --output out.mp4

  # Skip calibration, reuse saved state:
  python video_stitcher_SENA.py --left left.mp4 --right right.mp4 --output out.mp4 \\
                                --state sena_state.npz

  # Force SIFT even if XFeat is available:
  python video_stitcher_SENA.py --left left.mp4 --right right.mp4 --output out.mp4 \\
                                --no-xfeat

  # Live streams (RTSP URL or webcam index):
  python video_stitcher_SENA.py --left 0 --right 1 --output live.mp4 --live
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from sena_core import (
    AnchorPartitioner, AdequateZoneDetector, FeatureMatcher,
    LocalAffineWarper, PartitionConfig, WarpConfig, ZoneConfig,
)


# ---------------------------------------------------------------------------
# Calibration state  (everything computed once from N frames)
# ---------------------------------------------------------------------------

class SENAState:
    """
    Holds all artefacts computed during calibration that are reused per frame.

    Attributes
    ----------
    map_x, map_y     : full-resolution cv2.remap maps for the source image
    canvas_h/w       : output canvas dimensions
    ox, oy           : offset of target image origin on canvas
    zone_x0/x1       : adequate zone x-column range (canvas coords)
    chain_s, chain_t : refined anchor keypoint arrays (canvas coords)
    overlap_mask     : binary uint8 canvas mask of overlap region
    """

    def __init__(self):
        self.map_x:        np.ndarray = None
        self.map_y:        np.ndarray = None
        self.canvas_h:     int = 0
        self.canvas_w:     int = 0
        self.ox:           int = 0
        self.oy:           int = 0
        self.zone_x0:      int = 0
        self.zone_x1:      int = 0
        self.chain_s:      np.ndarray = None
        self.chain_t:      np.ndarray = None
        self.overlap_mask: np.ndarray = None

    def save(self, path: str):
        np.savez_compressed(
            path,
            map_x=self.map_x, map_y=self.map_y,
            canvas_h=np.array(self.canvas_h),
            canvas_w=np.array(self.canvas_w),
            ox=np.array(self.ox), oy=np.array(self.oy),
            zone_x0=np.array(self.zone_x0),
            zone_x1=np.array(self.zone_x1),
            chain_s=self.chain_s, chain_t=self.chain_t,
            overlap_mask=self.overlap_mask,
        )
        print(f"[State] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "SENAState":
        data = np.load(path)
        state = cls()
        state.map_x        = data["map_x"]
        state.map_y        = data["map_y"]
        state.canvas_h     = int(data["canvas_h"])
        state.canvas_w     = int(data["canvas_w"])
        state.ox           = int(data["ox"])
        state.oy           = int(data["oy"])
        state.zone_x0      = int(data["zone_x0"])
        state.zone_x1      = int(data["zone_x1"])
        state.chain_s      = data["chain_s"]
        state.chain_t      = data["chain_t"]
        state.overlap_mask = data["overlap_mask"]
        print(f"[State] Loaded from {path}")
        return state


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def calibrate(
    cap_left:  cv2.VideoCapture,
    cap_right: cv2.VideoCapture,
    n_frames:  int,
    matcher:   FeatureMatcher,
    warp_cfg:  WarpConfig,
    zone_cfg:  ZoneConfig,
    part_cfg:  PartitionConfig,
) -> SENAState:
    """
    Read the first n_frames from both captures, compute SENA state.
    Uses element-wise median across per-frame homographies / chain candidates
    to be robust against transient occlusions or motion.
    """
    print(f"[Calibration] SENA — processing {n_frames} calibration frame(s) …")

    warper     = LocalAffineWarper(warp_cfg)
    detector   = AdequateZoneDetector(zone_cfg)
    partitioner = AnchorPartitioner(part_cfg)

    # Collect per-frame artefacts
    A_globs:   list = []
    pts_srcs:  list = []
    pts_tgts:  list = []
    shapes_l:  list = []
    shapes_r:  list = []

    for i in range(n_frames):
        ok1, f1 = cap_left.read()
        ok2, f2 = cap_right.read()
        if not ok1 or not ok2:
            print(f"  [!] Frame {i}: read failed — stopping early.")
            break

        p1, p2 = matcher.match(f1, f2)
        if p1 is None or len(p1) < 4:
            print(f"  Frame {i}: not enough matches — skipped.")
            continue

        A = matcher.estimate_global_affine(p1, p2)
        if A is None:
            print(f"  Frame {i}: affine estimation failed — skipped.")
            continue

        print(f"  Frame {i}: {len(p1)} inlier matches ✓")
        A_globs.append(A)
        pts_srcs.append(p1)
        pts_tgts.append(p2)
        shapes_l.append(f1.shape)
        shapes_r.append(f2.shape)

    if not A_globs:
        raise RuntimeError("[Calibration] No valid frames. "
                           "Check that the videos overlap sufficiently.")

    # Median global affine across frames
    A_glob = np.median(np.stack(A_globs, 0), axis=0)
    # Use all inliers pooled for richer zone / chain estimation
    all_pts_s = np.vstack(pts_srcs)
    all_pts_t = np.vstack(pts_tgts)

    # Use first successful frame's images for geometry + mask computation
    cap_left.set(cv2.CAP_PROP_POS_FRAMES,  0)
    cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)
    _, f1 = cap_left.read()
    _, f2 = cap_right.read()

    print("[Calibration] Building warp maps …")
    warped_src, canvas_tgt, overlap_mask, (ox, oy) = warper.warp(
        f1, f2, all_pts_s, all_pts_t, A_glob)

    canvas_h, canvas_w = canvas_tgt.shape[:2]

    # Keypoints in canvas coordinates
    pts_s_canvas = all_pts_s + np.array([ox, oy], np.float32)
    pts_t_canvas = all_pts_t + np.array([ox, oy], np.float32)

    print("[Calibration] Detecting adequate zone …")
    zone_x0, zone_x1 = detector.detect(pts_s_canvas, pts_t_canvas, canvas_w)
    print(f"  Adequate zone: x=[{zone_x0}, {zone_x1}] ({zone_x1-zone_x0}px wide)")

    print("[Calibration] Refining keypoint chain …")
    chain_s, chain_t = partitioner._refine_chain(
        pts_s_canvas[
            (pts_s_canvas[:, 0] >= zone_x0) & (pts_s_canvas[:, 0] < zone_x1)],
        pts_t_canvas[
            (pts_s_canvas[:, 0] >= zone_x0) & (pts_s_canvas[:, 0] < zone_x1)],
        warped_src, canvas_tgt, canvas_w)
    print(f"  Anchor chain: {len(chain_s)} keypoints")

    # Extract remap maps from warper (reuse the ones just computed)
    # Recompute via warper's internal helper so they're stored cleanly
    dx_gated, dy_gated = warper._build_ffd_and_gate(
        [],  # no local transforms needed — use global-only remap for speed
        A_glob, canvas_h, canvas_w, (ox, oy),
        overlap_mask, all_pts_t)
    # For the per-frame remap, we only need global affine + gated FFD
    # Re-derive from the full warp result via internal remap builder
    map_x, map_y = LocalAffineWarper._build_remap(
        A_glob, dx_gated, dy_gated, canvas_h, canvas_w, ox, oy)

    # Package state
    state          = SENAState()
    state.map_x    = map_x
    state.map_y    = map_y
    state.canvas_h = canvas_h
    state.canvas_w = canvas_w
    state.ox       = ox
    state.oy       = oy
    state.zone_x0  = zone_x0
    state.zone_x1  = zone_x1
    state.chain_s  = chain_s
    state.chain_t  = chain_t
    state.overlap_mask = overlap_mask

    print("[Calibration] Done.")
    return state


# ---------------------------------------------------------------------------
# Per-frame stitching  (fast path — no feature matching)
# ---------------------------------------------------------------------------

def stitch_frame(
    frame_left:  np.ndarray,
    frame_right: np.ndarray,
    state:       SENAState,
    partitioner: AnchorPartitioner,
) -> np.ndarray:
    """
    Stitch one frame pair using the pre-computed SENA state.
    Steps: remap source → place target → partition + reconstruct.
    """
    canvas_h, canvas_w = state.canvas_h, state.canvas_w
    ox, oy = state.ox, state.oy

    # Warp source onto canvas
    warped_src = cv2.remap(
        frame_left,
        state.map_x, state.map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Place target on canvas
    canvas_tgt = np.zeros((canvas_h, canvas_w, 3), np.uint8)
    ht, wt = frame_right.shape[:2]
    canvas_tgt[oy:oy+ht, ox:ox+wt] = frame_right

    # Anchor-based partition + reconstruct (Stage 3)
    if state.chain_s is not None and len(state.chain_s) >= 2:
        result = partitioner.partition_and_reconstruct(
            warped_src, canvas_tgt,
            state.chain_s, state.chain_t,
            state.zone_x0, state.zone_x1)
    else:
        result = AnchorPartitioner._alpha_blend_full(warped_src, canvas_tgt)

    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def open_source(path: str) -> cv2.VideoCapture:
    try:
        src = int(path)
    except ValueError:
        src = path
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise IOError(f"Cannot open video source: {path!r}")
    return cap


def run(
    left_path:    str,
    right_path:   str,
    output_path:  str,
    calib_frames: int  = 5,
    state_path:   str  = None,
    save_state:   str  = "sena_state.npz",
    use_xfeat:    bool = True,
    sigma_max:    float = 4.0,
    live:         bool = False,
    # SENA hyperparameters (paper defaults)
    warp_cfg:    WarpConfig    = None,
    zone_cfg:    ZoneConfig    = None,
    part_cfg:    PartitionConfig = None,
):
    warp_cfg = warp_cfg or WarpConfig()
    zone_cfg = zone_cfg or ZoneConfig()
    part_cfg = part_cfg or PartitionConfig()

    cap_left  = open_source(left_path)
    cap_right = open_source(right_path)

    fps   = cap_left.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(min(cap_left.get(cv2.CAP_PROP_FRAME_COUNT),
                    cap_right.get(cv2.CAP_PROP_FRAME_COUNT)))

    # ── Calibration / load state ─────────────────────────────────────────
    if state_path and Path(state_path).exists():
        state = SENAState.load(state_path)
    else:
        matcher = FeatureMatcher(
            use_xfeat=use_xfeat,
            sigma_max=sigma_max)
        state = calibrate(
            cap_left, cap_right,
            n_frames=calib_frames,
            matcher=matcher,
            warp_cfg=warp_cfg,
            zone_cfg=zone_cfg,
            part_cfg=part_cfg)
        state.save(save_state)
        # Reset to start for stitching pass
        cap_left.set(cv2.CAP_PROP_POS_FRAMES,  0)
        cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # ── VideoWriter ──────────────────────────────────────────────────────
    cw, ch = state.canvas_w, state.canvas_h
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (cw, ch))
    if not writer.isOpened():
        raise IOError(f"Cannot open VideoWriter: {output_path}")

    partitioner = AnchorPartitioner(part_cfg)

    print(f"[Stitching] SENA — {total if not live else '∞'} frames  "
          f"| canvas {cw}×{ch}")

    t_start  = time.perf_counter()
    n_frames = 0
    fps_buf  = []

    while True:
        t0 = time.perf_counter()
        ok1, fl = cap_left.read()
        ok2, fr = cap_right.read()
        if not ok1 or not ok2:
            break

        stitched = stitch_frame(fl, fr, state, partitioner)
        writer.write(stitched)
        n_frames += 1

        elapsed = time.perf_counter() - t0
        fps_buf.append(1.0 / max(elapsed, 1e-6))
        if len(fps_buf) > 30:
            fps_buf.pop(0)

        if n_frames % 15 == 0:
            avg = sum(fps_buf) / len(fps_buf)
            tot = time.perf_counter() - t_start
            print(f"  [{n_frames:5d}/{total if not live else '?':>5}]  "
                  f"{avg:5.1f} fps  ({tot:.0f}s)", end="\r")

    total_time = time.perf_counter() - t_start
    avg_fps    = n_frames / total_time if total_time > 0 else 0
    print(f"\n[Done] {n_frames} frames in {total_time:.1f}s  ({avg_fps:.1f} fps avg)")

    cap_left.release()
    cap_right.release()
    writer.release()
    print(f"[Output] Saved → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="SENA video stitcher — affine + FFD + anchor-based reconstruction.")
    p.add_argument("--left",          required=True)
    p.add_argument("--right",         required=True)
    p.add_argument("--output",        required=True)
    p.add_argument("--state",         default=None,
                   help="Pre-saved .npz state file (skips calibration)")
    p.add_argument("--save-state",    default="sena_state.npz",
                   help="Where to save the SENA state (default: sena_state.npz)")
    p.add_argument("--calib-frames",  type=int,   default=5)
    p.add_argument("--sigma-max",     type=float, default=4.0,
                   help="MAGSAC++ noise upper bound in px (default: 4.0)")
    p.add_argument("--no-xfeat",      action="store_true",
                   help="Force SIFT even if XFeat is available")
    p.add_argument("--live",          action="store_true",
                   help="Live stream mode (no frame count limit)")
    # SENA hyperparameters (paper defaults exposed for experimentation)
    p.add_argument("--grid-x",        type=int,   default=2)
    p.add_argument("--grid-y",        type=int,   default=2)
    p.add_argument("--lattice-y",     type=int,   default=64)
    p.add_argument("--lattice-x",     type=int,   default=64)
    p.add_argument("--lambda1",       type=float, default=2.2)
    p.add_argument("--lambda2",       type=float, default=2.8)
    p.add_argument("--zone-v",        type=float, default=5.0,
                   help="Disparity clustering threshold v (default: 5.0 px)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    warp_cfg          = WarpConfig()
    warp_cfg.grid_x   = args.grid_x
    warp_cfg.grid_y   = args.grid_y
    warp_cfg.lattice_y = args.lattice_y
    warp_cfg.lattice_x = args.lattice_x
    warp_cfg.lambda1  = args.lambda1
    warp_cfg.lambda2  = args.lambda2

    zone_cfg = ZoneConfig(v=args.zone_v)

    run(
        left_path    = args.left,
        right_path   = args.right,
        output_path  = args.output,
        calib_frames = args.calib_frames,
        state_path   = args.state,
        save_state   = args.save_state,
        use_xfeat    = not args.no_xfeat,
        sigma_max    = args.sigma_max,
        live         = args.live,
        warp_cfg     = warp_cfg,
        zone_cfg     = zone_cfg,
    )
