"""
Compute relative pose from manually-picked correspondences.

This is more reliable than auto-SIFT pose estimation for our scenario
because (a) the user's clicks exclude moving people and screens, and
(b) we can sweep the focal length and pick the value that minimizes the
epipolar residual on a small clean point set.

Usage:
    python pose_from_clicks.py --clicks clicks.json --baseline 0.8 \
                               --output pose.npz

Output:
    pose.npz containing:
        K              -- 3x3 intrinsics (shared between cameras)
        R_ba, t_ba     -- pose of camera B in A's frame, t scaled to baseline
        focal          -- chosen focal length in pixels
        residual       -- median epipolar residual at the chosen focal (px)
        points_a, points_b -- the inlier 2D matches
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def epipolar_residual(pts_a: np.ndarray, pts_b: np.ndarray,
                      K: np.ndarray, R: np.ndarray, t: np.ndarray) -> float:
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


def estimate_pose_for_focal(pts_a: np.ndarray, pts_b: np.ndarray,
                            f: float, image_size: tuple[int, int]
                            ) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Return (R, t_unit, residual_px, n_inliers) for a given focal length."""
    w, h = image_size
    K = np.array([[f, 0, w / 2],
                  [0, f, h / 2],
                  [0, 0, 1]], dtype=np.float64)
    # With manually-picked clean correspondences, RANSAC threshold can be tight.
    E, mask = cv2.findEssentialMat(pts_a, pts_b, K, method=cv2.RANSAC,
                                   prob=0.999, threshold=1.0)
    if E is None:
        return None, None, np.inf, 0  # type: ignore
    _, R, t, mask_pose = cv2.recoverPose(E, pts_a, pts_b, K, mask=mask)
    inl = mask_pose.ravel().astype(bool)
    if inl.sum() < 6:
        return R, t.reshape(3), np.inf, int(inl.sum())
    resid = epipolar_residual(pts_a[inl], pts_b[inl], K, R, t.reshape(3))
    return R, t.reshape(3), resid, int(inl.sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clicks", required=True, type=Path,
                        help="JSON file from pick_correspondences.py")
    parser.add_argument("--baseline", type=float, default=0.8,
                        help="Physical baseline in meters.")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output .npz path.")
    parser.add_argument("--f_min", type=float, default=None,
                        help="Min focal to search. Default: 0.4 * image_width.")
    parser.add_argument("--f_max", type=float, default=None,
                        help="Max focal to search. Default: 2.5 * image_width.")
    parser.add_argument("--n_steps", type=int, default=41)
    args = parser.parse_args()

    data = json.loads(args.clicks.read_text())
    pts_a = np.array(data["points_a"], dtype=np.float64)
    pts_b = np.array(data["points_b"], dtype=np.float64)
    w, h = data["image_size_a"]
    if data["image_size_a"] != data["image_size_b"]:
        print("[warn] different image sizes between A and B; using A's.")
    print(f"[pose] {len(pts_a)} correspondences, image {w}x{h}")

    f_min = args.f_min if args.f_min is not None else 0.4 * w
    f_max = args.f_max if args.f_max is not None else 2.5 * w
    focals = np.linspace(f_min, f_max, args.n_steps)

    print(f"[pose] sweeping focal from {f_min:.0f} to {f_max:.0f}px "
          f"({args.n_steps} steps)")
    best = None  # (focal, R, t, resid, n_inl)
    for f in focals:
        R, t, resid, n_inl = estimate_pose_for_focal(pts_a, pts_b, float(f), (w, h))
        if R is None:
            continue
        if best is None or resid < best[3]:
            best = (float(f), R, t, resid, n_inl)

    if best is None:
        raise RuntimeError("Pose estimation failed for all focal candidates.")
    f_chosen, R, t_unit, resid, n_inl = best

    K = np.array([[f_chosen, 0, w / 2],
                  [0, f_chosen, h / 2],
                  [0, 0, 1]], dtype=np.float64)
    fov_h = 2 * np.degrees(np.arctan2(w / 2, f_chosen))
    fov_v = 2 * np.degrees(np.arctan2(h / 2, f_chosen))

    # Pretty-print rotation as axis-angle.
    angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
    if abs(angle) > 1e-6:
        axis = np.array([R[2, 1] - R[1, 2],
                         R[0, 2] - R[2, 0],
                         R[1, 0] - R[0, 1]])
        axis = axis / (np.linalg.norm(axis) + 1e-12)
    else:
        axis = np.array([0, 0, 1.0])

    print("\n--- chosen pose ---")
    print(f"focal       : {f_chosen:.1f} px  (HFOV {fov_h:.1f} deg, "
          f"VFOV {fov_v:.1f} deg)")
    print(f"residual    : {resid:.3f} px")
    print(f"inliers     : {n_inl} / {len(pts_a)}")
    print(f"rotation    : {angle:.2f} deg about axis "
          f"[{axis[0]:+.3f}, {axis[1]:+.3f}, {axis[2]:+.3f}]")
    print(f"translation : [{t_unit[0]:+.3f}, {t_unit[1]:+.3f}, "
          f"{t_unit[2]:+.3f}] (unit norm)")
    print(f"K =\n{K}")

    # Sanity warnings (not errors).
    if angle > 110:
        print("[warn] rotation > 110 deg looks like a sign-flip in the "
              "essential matrix decomposition. The result may still be valid "
              "but verify with visualize_setup.py.")
    if resid > 3.0:
        print("[warn] residual > 3 px suggests inconsistent clicks. Consider "
              "redoing the click step with more careful, distributed points.")

    t_ba = t_unit * args.baseline
    np.savez(args.output, K=K, R_ba=R, t_ba=t_ba, focal=f_chosen,
             residual=resid, points_a=pts_a, points_b=pts_b)
    print(f"\n[pose] saved to {args.output}")


if __name__ == "__main__":
    main()
