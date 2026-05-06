"""
Depth-aware stereo video stitching (proof of concept, v3).

Changes vs. v2:
  * --focal lets you override the guessed focal length.
  * Automatic focal sweep: if no calibration is provided, try several
    focal lengths and pick the one whose recovered translation is most
    consistent with cameras-side-by-side (X-dominated).
  * --force_translation lets you bypass the recovered translation entirely
    and inject the known physical translation (e.g. "0.8,0,0" for cameras
    on a wall, B to the right of A).
  * Print epipolar residuals for the recovered pose so we can tell whether
    the geometry is self-consistent independently of the visual output.

Usage:
    # Quickest test, with translation injected from your physical setup:
    python depth_stitch.py --video_a A.mp4 --video_b B.mp4 \
                           --force_translation 0.8,0,0 \
                           --output C.mp4 --max_frames 100

    # With chessboard calibration (best):
    python depth_stitch.py --video_a A.mp4 --video_b B.mp4 \
                           --calib_a K_a.npz --calib_b K_b.npz \
                           --output C.mp4 --max_frames 300
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


# ---------------------------------------------------------------------------
# Depth estimation
# ---------------------------------------------------------------------------

class DepthEstimator:
    def __init__(self, model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
                 device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[depth] loading {model_name} on {self.device}")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def __call__(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        pred = outputs.predicted_depth
        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1),
            size=bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy().astype(np.float32)
        depth = 1.0 / np.clip(pred, 1e-3, None)
        return depth


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def load_calib(path: Path | None, image_shape: tuple[int, int],
               focal_override: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_shape
    if path is None:
        f = focal_override if focal_override is not None else float(w)
        K = np.array([[f, 0, w / 2],
                      [0, f, h / 2],
                      [0, 0, 1]], dtype=np.float64)
        dist = np.zeros(5)
        if focal_override is not None:
            print(f"[calib] no calibration file -- using user-provided focal = {f:.1f}px")
        else:
            print(f"[calib] no calibration file -- guessing focal = {f:.1f}px")
        return K, dist
    data = np.load(path)
    K = data["K"].astype(np.float64)
    dist = data["dist"].astype(np.float64).ravel()
    print(f"[calib] loaded {path}")
    print(f"[calib] K =\n{K}")
    print(f"[calib] dist = {dist}")
    return K, dist


def undistort(img: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    if np.allclose(dist, 0):
        return img
    return cv2.undistort(img, K, dist)


# ---------------------------------------------------------------------------
# One-shot pose estimation
# ---------------------------------------------------------------------------

def collect_matches(cap_a: cv2.VideoCapture, cap_b: cv2.VideoCapture,
                    K_a: np.ndarray, dist_a: np.ndarray,
                    K_b: np.ndarray, dist_b: np.ndarray,
                    n_pairs: int = 8) -> tuple[np.ndarray, np.ndarray]:
    n_frames = min(int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT)),
                   int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT)))
    indices = np.linspace(0, n_frames - 1, n_pairs).astype(int)

    sift = cv2.SIFT_create(nfeatures=4000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    all_a, all_b = [], []
    for idx in indices:
        cap_a.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        cap_b.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok_a, fa = cap_a.read()
        ok_b, fb = cap_b.read()
        if not (ok_a and ok_b):
            continue
        fa = undistort(fa, K_a, dist_a)
        fb = undistort(fb, K_b, dist_b)
        kp_a, des_a = sift.detectAndCompute(cv2.cvtColor(fa, cv2.COLOR_BGR2GRAY), None)
        kp_b, des_b = sift.detectAndCompute(cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY), None)
        if des_a is None or des_b is None:
            continue
        knn = matcher.knnMatch(des_a, des_b, k=2)
        good = [m for m, n in knn if m.distance < 0.75 * n.distance]
        if len(good) < 20:
            continue
        pts_a = np.float32([kp_a[m.queryIdx].pt for m in good])
        pts_b = np.float32([kp_b[m.trainIdx].pt for m in good])
        all_a.append(pts_a)
        all_b.append(pts_b)
        print(f"[pose] frame {idx}: {len(good)} matches")
    if not all_a:
        raise RuntimeError("No usable frame pairs for pose estimation.")
    return np.vstack(all_a), np.vstack(all_b)


def estimate_pose_once(pts_a: np.ndarray, pts_b: np.ndarray,
                       K: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    E, mask = cv2.findEssentialMat(pts_a, pts_b, K, method=cv2.RANSAC,
                                   prob=0.999, threshold=1.0)
    if E is None:
        raise RuntimeError("Essential matrix failed.")
    _, R, t, mask_pose = cv2.recoverPose(E, pts_a, pts_b, K, mask=mask)
    return R, t.reshape(3), mask_pose.ravel().astype(bool)


def epipolar_residual(pts_a: np.ndarray, pts_b: np.ndarray,
                      K: np.ndarray, R: np.ndarray, t: np.ndarray) -> float:
    """Median Sampson distance for the (R, t) hypothesis. Lower = better.

    Useful as a self-consistency score that does not depend on visual output.
    """
    # E = [t]_x R
    tx = np.array([[0, -t[2], t[1]],
                   [t[2], 0, -t[0]],
                   [-t[1], t[0], 0]], dtype=np.float64)
    E = tx @ R
    F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    a = np.hstack([pts_a, np.ones((len(pts_a), 1))])
    b = np.hstack([pts_b, np.ones((len(pts_b), 1))])
    Fa = (F @ a.T).T
    Ftb = (F.T @ b.T).T
    bFa = np.einsum("ij,ij->i", b, (F @ a.T).T)
    denom = Fa[:, 0] ** 2 + Fa[:, 1] ** 2 + Ftb[:, 0] ** 2 + Ftb[:, 1] ** 2
    sampson = (bFa ** 2) / np.clip(denom, 1e-12, None)
    return float(np.median(np.sqrt(sampson)))


def focal_sweep(pts_a: np.ndarray, pts_b: np.ndarray,
                image_shape: tuple[int, int],
                f_min: float, f_max: float, n: int = 17) -> float:
    """Try a range of focal lengths; pick the one whose recovered t is most
    X-aligned (best for side-by-side cameras) while keeping epipolar
    residual low.

    Returns the chosen focal length in pixels.
    """
    h, w = image_shape
    focals = np.linspace(f_min, f_max, n)
    best_f, best_score = None, -np.inf
    print(f"[sweep] trying focals from {f_min:.0f} to {f_max:.0f}px ({n} steps)")
    for f in focals:
        K = np.array([[f, 0, w / 2],
                      [0, f, h / 2],
                      [0, 0, 1]], dtype=np.float64)
        try:
            R, t, mask = estimate_pose_once(pts_a, pts_b, K)
        except RuntimeError:
            continue
        # Score: prefer X-dominated translation, penalize epipolar residual.
        x_dom = abs(t[0]) / (np.linalg.norm(t) + 1e-9)
        resid = epipolar_residual(pts_a[mask], pts_b[mask], K, R, t)
        # Combine: high x_dom, low resid. Normalize residual loosely.
        score = x_dom - 0.05 * resid
        print(f"[sweep]   f={f:7.1f}px  |tx|/|t|={x_dom:.3f}  resid={resid:.3f}px  score={score:.3f}")
        if score > best_score:
            best_score = score
            best_f = float(f)
    if best_f is None:
        raise RuntimeError("Focal sweep failed for all candidates.")
    print(f"[sweep] selected focal = {best_f:.1f}px")
    return best_f


def describe_pose(R: np.ndarray, t: np.ndarray) -> None:
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    if abs(angle) < 1e-6:
        axis = np.array([0, 0, 1.0])
    else:
        axis = np.array([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]])
        axis = axis / (np.linalg.norm(axis) + 1e-12)
    print(f"[pose] rotation angle: {angle:.2f} deg")
    print(f"[pose] rotation axis : [{axis[0]:+.3f}, {axis[1]:+.3f}, {axis[2]:+.3f}]")
    print(f"[pose] translation   : [{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}] (unit norm)")
    if angle > 45:
        print(f"[warn] rotation > 45 deg is unusual for two cameras on the "
              f"same wall facing the same direction.")
    if abs(t[0]) < max(abs(t[1]), abs(t[2])):
        print(f"[warn] translation is not dominated by the X axis. If your "
              f"cameras are side-by-side, |tx| should be the largest. "
              f"Got |tx|={abs(t[0]):.2f}, |ty|={abs(t[1]):.2f}, "
              f"|tz|={abs(t[2]):.2f}.")


# ---------------------------------------------------------------------------
# Mono depth alignment
# ---------------------------------------------------------------------------

def align_mono_depth(depth_mono: np.ndarray, pts2d: np.ndarray,
                     z_ref: np.ndarray) -> np.ndarray:
    h, w = depth_mono.shape
    xs = np.clip(pts2d[:, 0].astype(int), 0, w - 1)
    ys = np.clip(pts2d[:, 1].astype(int), 0, h - 1)
    d_mono = depth_mono[ys, xs]
    valid = (z_ref > 0) & np.isfinite(z_ref) & np.isfinite(d_mono) & (d_mono > 0)
    if valid.sum() < 10:
        return depth_mono.copy()
    d_mono, z_ref = d_mono[valid], z_ref[valid]

    rng = np.random.default_rng(0)
    best, best_err = None, np.inf
    n = len(d_mono)
    for _ in range(300):
        i, j = rng.choice(n, size=2, replace=False)
        if abs(d_mono[i] - d_mono[j]) < 1e-6:
            continue
        s = (z_ref[i] - z_ref[j]) / (d_mono[i] - d_mono[j])
        b = z_ref[i] - s * d_mono[i]
        if s <= 0:
            continue
        err = np.median(np.abs(s * d_mono + b - z_ref))
        if err < best_err:
            best_err, best = err, (s, b)
    if best is None:
        return depth_mono.copy()
    s, b = best
    return np.clip(s * depth_mono + b, 0.05, 100.0)


# ---------------------------------------------------------------------------
# Edge-aware unproject + rendering
# ---------------------------------------------------------------------------

def unproject_edge_aware(depth: np.ndarray, K: np.ndarray, stride: int = 2,
                         edge_thresh: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.ascontiguousarray(depth, dtype=np.float32)
    h, w = depth.shape
    dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(dx * dx + dy * dy)
    rel = grad / np.clip(depth, 1e-3, None)
    edge_mask = rel > edge_thresh

    ys, xs = np.mgrid[0:h:stride, 0:w:stride]
    xs = xs.ravel()
    ys = ys.ravel()
    z = depth[ys, xs]
    on_edge = edge_mask[ys, xs]
    valid = (z > 0) & np.isfinite(z) & ~on_edge
    xs, ys, z = xs[valid], ys[valid], z[valid]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    X = (xs - cx) * z / fx
    Y = (ys - cy) * z / fy
    return np.stack([X, Y, z], axis=1), xs, ys


def render_points(pts_world: np.ndarray, colors: np.ndarray,
                  K_virt: np.ndarray, R_w2v: np.ndarray, t_w2v: np.ndarray,
                  out_h: int, out_w: int,
                  splat_radius: int = 1) -> tuple[np.ndarray, np.ndarray]:
    pts_cam = (R_w2v @ pts_world.T).T + t_w2v
    z = pts_cam[:, 2]
    front = z > 1e-3
    pts_cam, cols, z = pts_cam[front], colors[front], z[front]

    fx, fy = K_virt[0, 0], K_virt[1, 1]
    cx, cy = K_virt[0, 2], K_virt[1, 2]
    u = (pts_cam[:, 0] * fx / z + cx).astype(np.int32)
    v = (pts_cam[:, 1] * fy / z + cy).astype(np.int32)

    image = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    zbuf = np.full((out_h, out_w), np.inf, dtype=np.float32)
    r = splat_radius
    for du in range(-r, r + 1):
        for dv in range(-r, r + 1):
            uu = u + du
            vv = v + dv
            inside = (uu >= 0) & (uu < out_w) & (vv >= 0) & (vv < out_h)
            uu, vv, zz, cc = uu[inside], vv[inside], z[inside], cols[inside]
            current = zbuf[vv, uu]
            closer = zz < current
            uu, vv, zz, cc = uu[closer], vv[closer], zz[closer], cc[closer]
            zbuf[vv, uu] = zz
            image[vv, uu] = cc
    return image, zbuf


def fill_small_holes(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        return img
    return cv2.inpaint(img, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def stitch_videos(path_a: Path, path_b: Path, out_path: Path,
                  max_frames: int,
                  calib_a: Path | None, calib_b: Path | None,
                  baseline_meters: float,
                  focal_override: float | None,
                  do_focal_sweep: bool,
                  force_translation: np.ndarray | None) -> None:
    cap_a = cv2.VideoCapture(str(path_a))
    cap_b = cv2.VideoCapture(str(path_b))
    if not cap_a.isOpened() or not cap_b.isOpened():
        raise RuntimeError("Failed to open one or both input videos.")

    fps = cap_a.get(cv2.CAP_PROP_FPS) or 30.0
    n_a = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
    n_b = int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT))
    n_total = min(n_a, n_b, max_frames)

    ok_a, frame_a = cap_a.read()
    ok_b, frame_b = cap_b.read()
    if not (ok_a and ok_b):
        raise RuntimeError("Could not read first frame.")
    h, w = frame_a.shape[:2]

    K_a, dist_a = load_calib(calib_a, (h, w), focal_override)
    K_b, dist_b = load_calib(calib_b, (h, w), focal_override)

    print("\n[pose] sampling frames for one-shot pose estimation...")
    pts_a_all, pts_b_all = collect_matches(cap_a, cap_b, K_a, dist_a, K_b, dist_b,
                                           n_pairs=8)
    print(f"[pose] total matches across samples: {len(pts_a_all)}")

    # Optional: sweep focal length to find a sensible value (only if no
    # calibration file was supplied and the user asked for it).
    if do_focal_sweep and calib_a is None and calib_b is None:
        f_chosen = focal_sweep(pts_a_all, pts_b_all, (h, w),
                               f_min=max(400.0, w * 0.5),
                               f_max=w * 1.4, n=17)
        K_a, dist_a = load_calib(None, (h, w), focal_override=f_chosen)
        K_b = K_a.copy()
        dist_b = np.zeros(5)

    R_ba, t_ba_unit, mask_pose = estimate_pose_once(pts_a_all, pts_b_all, K_a)
    pts_a_inl = pts_a_all[mask_pose]
    pts_b_inl = pts_b_all[mask_pose]
    print(f"[pose] inliers after recoverPose: {len(pts_a_inl)}")
    describe_pose(R_ba, t_ba_unit)
    resid = epipolar_residual(pts_a_inl, pts_b_inl, K_a, R_ba, t_ba_unit)
    print(f"[pose] median epipolar residual: {resid:.3f} px")

    if force_translation is not None:
        norm = np.linalg.norm(force_translation)
        if norm < 1e-6:
            raise ValueError("--force_translation cannot be zero.")
        t_ba_unit = force_translation / norm
        print(f"[pose] OVERRIDE: using forced translation direction "
              f"{t_ba_unit} (was [{t_ba_unit[0]:+.3f},...])")
        # Sanity-recheck with the override.
        resid_forced = epipolar_residual(pts_a_inl, pts_b_inl, K_a, R_ba, t_ba_unit)
        print(f"[pose] epipolar residual with forced t: {resid_forced:.3f} px "
              f"(if much larger than original, the rotation may also be off)")

    t_ba = t_ba_unit * baseline_meters

    P1 = K_a @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K_a @ np.hstack([R_ba, t_ba.reshape(3, 1)])
    pts4d = cv2.triangulatePoints(P1, P2, pts_a_inl.T, pts_b_inl.T)
    pts3d_a_frame = (pts4d[:3] / pts4d[3]).T
    z_med = np.median(pts3d_a_frame[:, 2])
    print(f"[pose] median triangulated Z in A frame: {z_med:.3f} m")
    if not (0.2 < z_med < 50):
        print("[warn] median Z is outside a plausible indoor range. Pose is "
              "probably bad -- check calibration / input videos.")

    t_virt_world = t_ba / 2.0
    R_virt_world = np.eye(3)
    R_w2v = R_virt_world.T
    t_w2v = -R_w2v @ t_virt_world

    out_w = int(w * 1.5)
    out_h = h
    K_virt = K_a.copy()
    K_virt[0, 2] = out_w / 2
    K_virt[1, 2] = out_h / 2

    depth_net = DepthEstimator()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))

    cap_a.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap_b.set(cv2.CAP_PROP_POS_FRAMES, 0)

    sift = cv2.SIFT_create(nfeatures=2000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    pbar = tqdm(total=n_total, desc="stitching")
    frame_idx = 0
    while frame_idx < n_total:
        ok_a, fa = cap_a.read()
        ok_b, fb = cap_b.read()
        if not (ok_a and ok_b):
            break
        fa = undistort(fa, K_a, dist_a)
        fb = undistort(fb, K_b, dist_b)

        d_a = depth_net(fa)
        d_b = depth_net(fb)

        try:
            kp_a, des_a = sift.detectAndCompute(cv2.cvtColor(fa, cv2.COLOR_BGR2GRAY), None)
            kp_b, des_b = sift.detectAndCompute(cv2.cvtColor(fb, cv2.COLOR_BGR2GRAY), None)
            knn = matcher.knnMatch(des_a, des_b, k=2)
            good = [m for m, n in knn if m.distance < 0.75 * n.distance]
            if len(good) >= 20:
                pa = np.float32([kp_a[m.queryIdx].pt for m in good])
                pb = np.float32([kp_b[m.trainIdx].pt for m in good])
                pts4d = cv2.triangulatePoints(P1, P2, pa.T, pb.T)
                pts3d = (pts4d[:3] / pts4d[3]).T
                d_a = align_mono_depth(d_a, pa, pts3d[:, 2])
                pts3d_b = (R_ba @ pts3d.T).T + t_ba
                d_b = align_mono_depth(d_b, pb, pts3d_b[:, 2])
        except Exception:
            pass

        pa_cam, xa, ya = unproject_edge_aware(d_a, K_a, stride=2)
        col_a = fa[ya, xa]
        pb_cam, xb, yb = unproject_edge_aware(d_b, K_b, stride=2)
        col_b = fb[yb, xb]

        pa_w = pa_cam
        pb_w = (R_ba @ pb_cam.T).T + t_ba

        img_av, z_av = render_points(pa_w, col_a, K_virt, R_w2v, t_w2v,
                                     out_h, out_w, splat_radius=1)
        img_bv, z_bv = render_points(pb_w, col_b, K_virt, R_w2v, t_w2v,
                                     out_h, out_w, splat_radius=1)

        has_a = np.isfinite(z_av)
        has_b = np.isfinite(z_bv)
        only_a = has_a & ~has_b
        only_b = has_b & ~has_a
        both = has_a & has_b

        x_grad = np.linspace(0, 1, out_w, dtype=np.float32)
        wB = np.broadcast_to(x_grad, (out_h, out_w))
        wA = 1.0 - wB
        out = np.zeros_like(img_av)
        out[only_a] = img_av[only_a]
        out[only_b] = img_bv[only_b]
        if both.any():
            blended = (img_av.astype(np.float32) * wA[..., None] +
                       img_bv.astype(np.float32) * wB[..., None])
            out[both] = blended[both].astype(np.uint8)

        empty = (~has_a) & (~has_b)
        out = fill_small_holes(out, empty.astype(np.uint8) * 255)

        writer.write(out)
        pbar.update(1)
        frame_idx += 1

    pbar.close()
    cap_a.release()
    cap_b.release()
    writer.release()
    print(f"[done] wrote {frame_idx} frames to {out_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video_a", required=True, type=Path)
    p.add_argument("--video_b", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--max_frames", type=int, default=300)
    p.add_argument("--calib_a", type=Path, default=None,
                   help="Optional .npz from calibrate.py (camera A).")
    p.add_argument("--calib_b", type=Path, default=None,
                   help="Optional .npz from calibrate.py (camera B).")
    p.add_argument("--baseline", type=float, default=0.8,
                   help="Physical baseline in meters (default 0.8).")
    p.add_argument("--focal", type=float, default=None,
                   help="Manual focal length override in pixels. "
                        "Ignored if --calib_a / --calib_b are provided.")
    p.add_argument("--focal_sweep", action="store_true",
                   help="Sweep focal lengths and pick the one that yields "
                        "the most X-aligned translation. Only meaningful "
                        "without calibration files.")
    p.add_argument("--force_translation", type=str, default=None,
                   help="Inject a known translation direction as 'X,Y,Z' "
                        "(e.g. '0.8,0,0' for B to the right of A). The "
                        "direction is normalized; only the direction is used "
                        "and then scaled by --baseline.")
    return p.parse_args()


def parse_translation(s: str | None) -> np.ndarray | None:
    if s is None:
        return None
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"--force_translation must be 'X,Y,Z', got '{s}'")
    return np.array(parts, dtype=np.float64)


def main() -> None:
    args = parse_args()
    forced = parse_translation(args.force_translation)
    stitch_videos(args.video_a, args.video_b, args.output, args.max_frames,
                  args.calib_a, args.calib_b, args.baseline,
                  args.focal, args.focal_sweep, forced)


if __name__ == "__main__":
    main()
