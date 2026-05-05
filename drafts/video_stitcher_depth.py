"""
Depth-aware video stitching from two fixed cameras — draft.

Motivation
----------
The v5/v6 pipeline (video_stitcher_seam_gpu*.py) aligns the two views with a
single homography estimated once on frame 0. A homography is exact only for a
single plane in the scene; with a ~1 m baseline and an indoor scene that
contains depth from ~1 m to ~10 m, no single plane works for everything. The
stitcher then picks one (implicitly: whichever the ORB features cluster on)
and tolerates ghosting / tearing elsewhere — papered over by seam-finding
with YOLO penalties.

This draft replaces the 2D homographic warp with a 3D reprojection:

    For each camera, estimate per-pixel depth (Depth Anything V2 small).
    Lift each pixel into 3D (camera frame) using assumed intrinsics K.
    Reproject every 3D point into camera A's image plane.
    Composite by z-buffer + depth-consistent blending.

Calibration once, on frame 0:

    1. ORB matches between A and B.
    2. Default intrinsics: f = max(W, H), principal point = image center
       (we have no calibrated K).
    3. Essential matrix → recoverPose → R_a→b, t_a→b (||t|| = 1).
       Translation scale is therefore "1 baseline ≈ 1 m" in our setup.
    4. Triangulate the inlier matches → metric 3D points (in A's frame).
    5. Run Depth Anything V2 on frame_a and frame_b. Its output is
       disparity-like (larger = closer); fit an affine map per camera so
       that  disp_metric ≈ scale · disp_mono + shift,  using the
       triangulated points as anchors. Both cameras end up consistent with
       the same triangulated 3D world.
    6. Compute output canvas size: project a coarse grid of B's pixels into
       A's image plane through 3D, take bbox with A's frame.

Per frame:

    1. Run Depth Anything V2 on each frame, convert to metric depth via the
       fitted (scale, shift).
    2. Forward-warp each camera into the canvas with a z-buffer:
       sort source pixels by depth descending and scatter so that nearer
       pixels overwrite farther ones.
    3. Blend the two canvases:
         - both contribute and z agrees within `agree_tau` → 50/50 average
         - both contribute and z disagrees → take the closer one
         - only one contributes → take it
       Pixels nobody hits stay black.

Limitations of this draft (deliberately simple, fix later as needed)
-------------------------------------------------------------------
  - Default intrinsics are a guess. With real calibration (checkerboard or
    SfM bundle adjustment) the reprojection becomes much sharper.
  - Forward warp via scatter leaves single-pixel cracks in regions where
    the surface is angled away from the camera. A morphological close on
    each cam's canvas before blending would help.
  - Depth is recomputed per frame with no temporal smoothing → flicker.
  - Output stays in A's plane (your pick). Switching to a midpoint virtual
    camera is a one-line change to R, t in the warp call.
  - Depth Anything's affine fit is plain lstsq with no outlier rejection
    beyond ORB's RANSAC. If a scene has a few mismatched points the fit
    can drift; add a Huber re-fit if it bites.
  - No FPS desync handling. Borrow FrameSyncReader from v5 if input rates
    differ.

Dependency
----------
    pip install transformers  (for the Depth Anything V2 small model)

Usage
-----
    from video_stitcher_depth import DepthAwareStitcher

    stitcher = DepthAwareStitcher(device="cuda")
    bundle = stitcher.calibrate(frame_a_0, frame_b_0)
    for frame_a, frame_b in pairs:
        panorama = stitcher.process_frame(frame_a, frame_b, bundle)

CLI demo:
    python video_stitcher_depth.py --video_a A.mp4 --video_b B.mp4 \
        --output stitched.mp4
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DepthStitchBundle: persistent calibration state (mirrors SENA's LUTBundle)
# ---------------------------------------------------------------------------

@dataclass
class DepthStitchBundle:
    K: np.ndarray              # 3x3, intrinsics shared by both cams
    R_b_to_a: np.ndarray       # 3x3, rotates B-frame coords into A-frame
    t_b_to_a: np.ndarray       # 3,  B's origin expressed in A's frame
    R_a_to_b: np.ndarray       # cached for completeness
    t_a_to_b: np.ndarray
    disp_scale_a: float        # disp_metric ≈ scale * disp_mono + shift
    disp_shift_a: float
    disp_scale_b: float
    disp_shift_b: float
    canvas_w: int
    canvas_h: int
    ox: int                    # canvas origin offset (add to A pixel coords)
    oy: int
    img_a_shape: tuple
    img_b_shape: tuple
    n_inliers: int             # number of ORB inliers used for pose
    fit_residual_a: float      # rms of disparity affine fit, cam A
    fit_residual_b: float      # rms of disparity affine fit, cam B


# ---------------------------------------------------------------------------
# Depth Anything V2 small wrapper
# ---------------------------------------------------------------------------

class DepthAnythingV2Small:
    """
    Thin HuggingFace wrapper around depth-anything/Depth-Anything-V2-Small-hf.

    The model is trained with a scale-shift-invariant loss on disparity
    (inverse depth), so we treat the output as a disparity-like quantity
    where larger values mean closer surfaces. Absolute scale is recovered
    later via affine fitting against triangulated points.
    """

    MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

    def __init__(self, device="cuda"):
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        except ImportError as e:
            raise RuntimeError(
                "Depth Anything V2 requires `pip install transformers`."
            ) from e
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(self.MODEL_ID)
            .to(device).eval()
        )
        self.device = device

    @torch.no_grad()
    def predict_disparity(self, frame_bgr):
        """
        Returns a (H, W) float32 tensor on self.device.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        pred = outputs.predicted_depth  # (1, h, w), smaller than input
        pred = F.interpolate(
            pred.unsqueeze(1), size=(H, W),
            mode="bilinear", align_corners=False,
        )
        return pred[0, 0].float()


# ---------------------------------------------------------------------------
# Calibration helpers
# ---------------------------------------------------------------------------

def default_intrinsics(W, H):
    """
    Reasonable guess when no calibration is available. f = max(W, H) is the
    convention used by COLMAP / many SfM pipelines for "unknown camera" runs.
    """
    f = float(max(W, H))
    return np.array(
        [[f, 0.0, W * 0.5],
         [0.0, f, H * 0.5],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def orb_correspondences(frame_a, frame_b, n_features=4000):
    """ORB + brute-force matcher with cross-check. Returns matched (xy_a, xy_b)."""
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=n_features)
    ka, da = orb.detectAndCompute(gray_a, None)
    kb, db = orb.detectAndCompute(gray_b, None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        raise RuntimeError("ORB found too few features for pose estimation.")
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(da, db)
    if len(matches) < 8:
        raise RuntimeError("Too few ORB matches for pose estimation.")
    matches = sorted(matches, key=lambda m: m.distance)
    matches = matches[: max(50, len(matches) // 2)]
    pts_a = np.float64([ka[m.queryIdx].pt for m in matches])
    pts_b = np.float64([kb[m.trainIdx].pt for m in matches])
    return pts_a, pts_b


def estimate_relative_pose(pts_a, pts_b, K):
    """
    Recover the rigid transform between cameras from 2D correspondences.

    OpenCV's recoverPose returns (R, t) such that  x_b ≈ R x_a + t  i.e. R, t
    take a point expressed in A's frame and produce its coordinates in B's
    frame. ||t|| = 1 (scale is unobservable from monocular cues).
    """
    E, mask = cv2.findEssentialMat(
        pts_a, pts_b, K, method=cv2.RANSAC, prob=0.999, threshold=1.0,
    )
    if E is None:
        raise RuntimeError("findEssentialMat returned None.")
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < 8:
        raise RuntimeError(
            f"Too few essential-matrix inliers ({int(inliers.sum())})."
        )
    pts_a_in = pts_a[inliers]
    pts_b_in = pts_b[inliers]
    n_in, R_a_to_b, t_a_to_b, mask2 = cv2.recoverPose(
        E, pts_a_in, pts_b_in, K,
    )
    cheir = mask2.ravel().astype(bool)
    pts_a_in = pts_a_in[cheir]
    pts_b_in = pts_b_in[cheir]
    t_a_to_b = t_a_to_b.ravel()
    R_b_to_a = R_a_to_b.T
    t_b_to_a = -R_a_to_b.T @ t_a_to_b
    return (R_a_to_b, t_a_to_b, R_b_to_a, t_b_to_a, pts_a_in, pts_b_in)


def triangulate_in_a(pts_a, pts_b, K, R_a_to_b, t_a_to_b):
    """Linear triangulation. Returns (N, 3) points in A's frame."""
    P_a = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P_b = K @ np.hstack([R_a_to_b, t_a_to_b.reshape(3, 1)])
    pts4 = cv2.triangulatePoints(P_a, P_b, pts_a.T, pts_b.T)
    w = pts4[3:4]
    w[np.abs(w) < 1e-12] = 1e-12
    pts3 = (pts4[:3] / w).T
    return pts3


def fit_disparity_affine(disp_map_t, sparse_pixels, sparse_z, eps=1e-6):
    """
    Solve  disp_metric ≈ s * disp_mono + b  where  disp_metric = 1 / z_metric
    using sparse points anchored to triangulated z. Returns (s, b, rms).
    """
    valid = sparse_z > eps
    pix = sparse_pixels[valid]
    z = sparse_z[valid]
    if pix.shape[0] < 4:
        # Degenerate: fall back to identity. Output will be clearly wrong
        # and visible — the caller can decide what to do.
        return 1.0, 0.0, float("inf")
    H, W = disp_map_t.shape
    xs = np.clip(pix[:, 0].round().astype(int), 0, W - 1)
    ys = np.clip(pix[:, 1].round().astype(int), 0, H - 1)
    disp_at = disp_map_t[ys, xs].detach().cpu().numpy().astype(np.float64)
    disp_metric = 1.0 / z
    A_mat = np.stack([disp_at, np.ones_like(disp_at)], axis=1)
    sol, *_ = np.linalg.lstsq(A_mat, disp_metric, rcond=None)
    s, b = float(sol[0]), float(sol[1])
    pred = s * disp_at + b
    rms = float(np.sqrt(np.mean((pred - disp_metric) ** 2)))
    return s, b, rms


# ---------------------------------------------------------------------------
# Canvas estimation: where do B's pixels land in A's image plane?
# ---------------------------------------------------------------------------

def estimate_canvas(K, R_b_to_a, t_b_to_a, disp_b_t, disp_scale_b, disp_shift_b,
                    img_a_shape, img_b_shape, grid_n=12, eps=1e-6):
    """
    Sample a coarse grid over B's image, push each sample through 3D into
    A's image plane, and take the bbox with A's frame corners.
    """
    H_a, W_a = img_a_shape[:2]
    H_b, W_b = img_b_shape[:2]
    xs = np.linspace(0, W_b - 1, grid_n)
    ys = np.linspace(0, H_b - 1, grid_n)
    gx, gy = np.meshgrid(xs, ys)
    pix_b = np.stack([gx.ravel(), gy.ravel()], axis=1)

    disp_np = disp_b_t.detach().cpu().numpy().astype(np.float64)
    pix_int_x = np.clip(pix_b[:, 0].round().astype(int), 0, W_b - 1)
    pix_int_y = np.clip(pix_b[:, 1].round().astype(int), 0, H_b - 1)
    disp_at = disp_np[pix_int_y, pix_int_x]
    disp_metric = disp_scale_b * disp_at + disp_shift_b
    z_b = 1.0 / np.clip(disp_metric, eps, None)

    K_inv = np.linalg.inv(K)
    pix_h = np.hstack([pix_b, np.ones((pix_b.shape[0], 1))])
    rays_b = (K_inv @ pix_h.T).T
    pts_b3 = rays_b * z_b.reshape(-1, 1)

    pts_a3 = (R_b_to_a @ pts_b3.T).T + t_b_to_a.reshape(1, 3)
    z_a = pts_a3[:, 2]
    keep = z_a > eps
    pts_a3 = pts_a3[keep]
    if pts_a3.shape[0] == 0:
        raise RuntimeError(
            "No B-pixels project in front of camera A. "
            "Pose or depth fit is likely degenerate."
        )
    proj = (K @ pts_a3.T).T
    proj_xy = proj[:, :2] / proj[:, 2:3]

    a_corners = np.array(
        [[0, 0], [W_a - 1, 0], [W_a - 1, H_a - 1], [0, H_a - 1]],
        dtype=np.float64,
    )
    all_xy = np.vstack([a_corners, proj_xy])
    x_min = int(np.floor(all_xy[:, 0].min()))
    x_max = int(np.ceil(all_xy[:, 0].max()))
    y_min = int(np.floor(all_xy[:, 1].min()))
    y_max = int(np.ceil(all_xy[:, 1].max()))
    canvas_w = x_max - x_min + 1
    canvas_h = y_max - y_min + 1
    ox = -x_min
    oy = -y_min
    return canvas_w, canvas_h, ox, oy


# ---------------------------------------------------------------------------
# Forward warp with z-buffer
# ---------------------------------------------------------------------------

def forward_warp_zbuffer(img_bgr_t, depth_t, K_src, K_dst, R, t,
                          canvas_size, offset, eps=1e-3):
    """
    Splat src image onto a canvas in dst's image plane using per-pixel depth.

    P_dst = R @ P_src + t,  where R, t come in as (3, 3) and (3,) numpy.
    Z-buffer: when many src pixels collide on the same canvas pixel, the
    nearer one wins (we sort by depth descending and scatter, so the
    nearest write happens last).

    Returns (rgb_canvas u8, z_canvas f32, valid_canvas bool).
    """
    device = depth_t.device
    H, W = depth_t.shape
    Cw, Ch = canvas_size
    ox, oy = offset

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    pix_h = torch.stack(
        [xx.reshape(-1), yy.reshape(-1),
         torch.ones(H * W, device=device, dtype=torch.float32)],
        dim=0,
    )
    K_src_inv = torch.from_numpy(np.linalg.inv(K_src)).float().to(device)
    rays = K_src_inv @ pix_h
    z_src = depth_t.reshape(-1).clamp(min=eps)
    pts_src = rays * z_src.unsqueeze(0)

    R_t = torch.from_numpy(np.ascontiguousarray(R, dtype=np.float64)).float().to(device)
    t_t = torch.from_numpy(np.ascontiguousarray(t.reshape(3, 1), dtype=np.float64)).float().to(device)
    pts_dst = R_t @ pts_src + t_t

    K_dst_t = torch.from_numpy(K_dst).float().to(device)
    proj = K_dst_t @ pts_dst
    z_dst = proj[2]
    in_front = z_dst > eps
    z_safe = z_dst.clamp(min=eps)
    x_dst = (proj[0] / z_safe + ox).round().long()
    y_dst = (proj[1] / z_safe + oy).round().long()

    in_bounds = (x_dst >= 0) & (x_dst < Cw) & (y_dst >= 0) & (y_dst < Ch)
    valid = in_front & in_bounds

    rgb_flat = img_bgr_t.reshape(-1, 3)
    z_v = z_dst[valid]
    x_v = x_dst[valid]
    y_v = y_dst[valid]
    rgb_v = rgb_flat[valid]

    rgb_canvas = torch.zeros((Ch, Cw, 3), dtype=torch.uint8, device=device)
    z_canvas = torch.full((Ch, Cw), float("inf"),
                           dtype=torch.float32, device=device)
    valid_canvas = torch.zeros((Ch, Cw), dtype=torch.bool, device=device)

    if z_v.numel() > 0:
        # Far pixels first, near pixels last → near wins.
        order = torch.argsort(z_v, descending=True)
        rgb_canvas[y_v[order], x_v[order]] = rgb_v[order]
        z_canvas[y_v[order], x_v[order]] = z_v[order]
        valid_canvas[y_v[order], x_v[order]] = True

    return rgb_canvas, z_canvas, valid_canvas


# ---------------------------------------------------------------------------
# Depth-aware blend
# ---------------------------------------------------------------------------

def blend_depth_aware(rgb_a, z_a, valid_a, rgb_b, z_b, valid_b, agree_tau=0.10):
    """
    For each canvas pixel:
      - both cams contributed and z agrees within `agree_tau` → 50/50 blend
      - both contributed but z disagrees → take the closer one
        (the farther write is presumed to be a surface that the closer cam
         correctly occludes)
      - only one contributed → take that one
      - neither contributed → black
    """
    only_a = valid_a & (~valid_b)
    only_b = valid_b & (~valid_a)
    both = valid_a & valid_b
    z_min = torch.minimum(z_a, z_b).clamp(min=1e-6)
    rel_diff = torch.abs(z_a - z_b) / z_min
    agree = both & (rel_diff < agree_tau)
    a_closer = both & (~agree) & (z_a <= z_b)
    b_closer = both & (~agree) & (z_b < z_a)

    blend = ((rgb_a.float() + rgb_b.float()) * 0.5).clamp(0, 255).to(torch.uint8)
    out = torch.zeros_like(rgb_a)
    out = torch.where(agree.unsqueeze(-1), blend, out)
    out = torch.where((a_closer | only_a).unsqueeze(-1), rgb_a, out)
    out = torch.where((b_closer | only_b).unsqueeze(-1), rgb_b, out)
    return out


# ---------------------------------------------------------------------------
# DepthAwareStitcher: two-phase API mirroring SENAPipeline
# ---------------------------------------------------------------------------

class DepthAwareStitcher:
    """
    Two-phase API:
        bundle = stitcher.calibrate(frame_a_0, frame_b_0)   # once
        out    = stitcher.process_frame(frame_a, frame_b, bundle)   # per-frame

    The bundle contains everything the per-frame path needs (intrinsics,
    extrinsics, disparity-affine fits, canvas geometry) — no hidden state on
    the stitcher. This makes process_frame stateless and the bundle
    serializable, same as SENA's LUTBundle.
    """

    def __init__(self, device=None, agree_tau=0.10):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.agree_tau = float(agree_tau)
        self.depth_model = DepthAnythingV2Small(device=device)

    def calibrate(self, frame_a, frame_b):
        H_a, W_a = frame_a.shape[:2]
        K = default_intrinsics(W_a, H_a)

        pts_a, pts_b = orb_correspondences(frame_a, frame_b)
        (R_a_to_b, t_a_to_b, R_b_to_a, t_b_to_a,
         pts_a_in, pts_b_in) = estimate_relative_pose(pts_a, pts_b, K)
        n_in = pts_a_in.shape[0]

        pts3_in_a = triangulate_in_a(pts_a_in, pts_b_in, K, R_a_to_b, t_a_to_b)

        disp_a_t = self.depth_model.predict_disparity(frame_a)
        disp_b_t = self.depth_model.predict_disparity(frame_b)

        z_in_a = pts3_in_a[:, 2]
        s_a, b_a, rms_a = fit_disparity_affine(disp_a_t, pts_a_in, z_in_a)

        pts3_in_b = (R_a_to_b @ pts3_in_a.T).T + t_a_to_b.reshape(1, 3)
        z_in_b = pts3_in_b[:, 2]
        s_b, b_b, rms_b = fit_disparity_affine(disp_b_t, pts_b_in, z_in_b)

        Cw, Ch, ox, oy = estimate_canvas(
            K, R_b_to_a, t_b_to_a, disp_b_t, s_b, b_b,
            frame_a.shape, frame_b.shape,
        )

        return DepthStitchBundle(
            K=K,
            R_b_to_a=R_b_to_a, t_b_to_a=t_b_to_a,
            R_a_to_b=R_a_to_b, t_a_to_b=t_a_to_b,
            disp_scale_a=s_a, disp_shift_a=b_a,
            disp_scale_b=s_b, disp_shift_b=b_b,
            canvas_w=int(Cw), canvas_h=int(Ch),
            ox=int(ox), oy=int(oy),
            img_a_shape=tuple(frame_a.shape),
            img_b_shape=tuple(frame_b.shape),
            n_inliers=int(n_in),
            fit_residual_a=float(rms_a),
            fit_residual_b=float(rms_b),
        )

    def _disparity_to_depth(self, disp_t, scale, shift, eps=1e-6):
        disp_metric = scale * disp_t + shift
        return 1.0 / disp_metric.clamp(min=eps)

    @torch.no_grad()
    def process_frame(self, frame_a, frame_b, bundle):
        device = self.device

        disp_a_t = self.depth_model.predict_disparity(frame_a)
        disp_b_t = self.depth_model.predict_disparity(frame_b)
        depth_a_t = self._disparity_to_depth(
            disp_a_t, bundle.disp_scale_a, bundle.disp_shift_a,
        )
        depth_b_t = self._disparity_to_depth(
            disp_b_t, bundle.disp_scale_b, bundle.disp_shift_b,
        )

        frame_a_t = torch.from_numpy(frame_a).to(device)
        frame_b_t = torch.from_numpy(frame_b).to(device)

        canvas_size = (bundle.canvas_w, bundle.canvas_h)
        offset = (bundle.ox, bundle.oy)

        rgb_a_canvas, z_a_canvas, valid_a = forward_warp_zbuffer(
            frame_a_t, depth_a_t,
            bundle.K, bundle.K,
            np.eye(3), np.zeros(3),
            canvas_size, offset,
        )
        rgb_b_canvas, z_b_canvas, valid_b = forward_warp_zbuffer(
            frame_b_t, depth_b_t,
            bundle.K, bundle.K,
            bundle.R_b_to_a, bundle.t_b_to_a,
            canvas_size, offset,
        )

        out_t = blend_depth_aware(
            rgb_a_canvas, z_a_canvas, valid_a,
            rgb_b_canvas, z_b_canvas, valid_b,
            agree_tau=self.agree_tau,
        )
        return out_t.cpu().numpy()


# ---------------------------------------------------------------------------
# Demo CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Depth-aware video stitching (draft)."
    )
    parser.add_argument("--video_a", required=True)
    parser.add_argument("--video_b", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_frames", type=int, default=0,
                        help="Stop after this many frames (0 = all).")
    parser.add_argument("--agree_tau", type=float, default=0.10,
                        help="Relative depth agreement threshold for "
                             "blending (default 0.10 = 10%%).")
    parser.add_argument("--device", default=None,
                        help="cuda or cpu (default: auto).")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    if device == "cuda":
        try:
            props = torch.cuda.get_device_properties(0)
            print(f"[device] {props.name} ({props.total_memory / (1024**3):.1f} GB)")
        except Exception:
            pass

    cap_a = cv2.VideoCapture(args.video_a)
    cap_b = cv2.VideoCapture(args.video_b)
    if not cap_a.isOpened() or not cap_b.isOpened():
        raise RuntimeError("Could not open one of the input videos.")
    fps_a = cap_a.get(cv2.CAP_PROP_FPS) or 25.0
    fps_b = cap_b.get(cv2.CAP_PROP_FPS) or 25.0
    fps_out = min(fps_a, fps_b)
    print(f"[info] Input FPS: A={fps_a:.3f}  B={fps_b:.3f}  out={fps_out:.3f}")

    ok_a, frame_a = cap_a.read()
    ok_b, frame_b = cap_b.read()
    if not (ok_a and ok_b):
        raise RuntimeError("Could not read first frame from both videos.")

    print("[info] Calibrating from first frame pair...")
    t0 = time.time()
    stitcher = DepthAwareStitcher(device=device, agree_tau=args.agree_tau)
    bundle = stitcher.calibrate(frame_a, frame_b)
    print(f"[info] Calibration: {time.time() - t0:.2f}s")
    print(f"[info]   ORB inliers: {bundle.n_inliers}")
    print(f"[info]   Canvas: {bundle.canvas_w} x {bundle.canvas_h}")
    print(f"[info]   t_a_to_b = {bundle.t_a_to_b}  (||t||=1, scale ≈ baseline)")
    print(f"[info]   disparity fit A: scale={bundle.disp_scale_a:.4g}  "
          f"shift={bundle.disp_shift_a:.4g}  rms={bundle.fit_residual_a:.4g}")
    print(f"[info]   disparity fit B: scale={bundle.disp_scale_b:.4g}  "
          f"shift={bundle.disp_shift_b:.4g}  rms={bundle.fit_residual_b:.4g}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        args.output, fourcc, fps_out,
        (bundle.canvas_w, bundle.canvas_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open writer for {args.output}")

    frame_idx = 0
    t_start = time.time()
    try:
        while True:
            if frame_idx > 0:
                ok_a, frame_a = cap_a.read()
                ok_b, frame_b = cap_b.read()
                if not (ok_a and ok_b):
                    break
            out = stitcher.process_frame(frame_a, frame_b, bundle)
            writer.write(out)
            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
            if frame_idx % 10 == 0:
                elapsed = time.time() - t_start
                fps = frame_idx / max(elapsed, 1e-6)
                print(f"[info] frame {frame_idx}  ({fps:.2f} fps avg)")
    finally:
        writer.release()
        cap_a.release()
        cap_b.release()

    elapsed = time.time() - t_start
    fps = frame_idx / max(elapsed, 1e-6)
    print(f"[info] Done. {frame_idx} frames in {elapsed:.2f}s "
          f"({fps:.2f} fps).")
    print(f"[info] Output written to {args.output}")


if __name__ == "__main__":
    main()
