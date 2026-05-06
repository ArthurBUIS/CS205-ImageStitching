"""
Visualize the recovered camera geometry to sanity-check pose before running
the full pipeline.

Usage:
    python visualize_setup.py --pose pose.npz [--save fig.png]

Shows:
    - Camera A frustum (world origin)
    - Camera B frustum (placed by R_ba, t_ba)
    - Triangulated world points from the click correspondences
    - A suggested "director's view" virtual camera position

If you cannot run a GUI (e.g. headless server), pass --save to dump the
figure to PNG and inspect it later.
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401, registers 3d projection
import cv2


def frustum_lines(K: np.ndarray, R_w2c: np.ndarray, t_w2c: np.ndarray,
                  image_size: tuple[int, int], depth: float = 1.0
                  ) -> np.ndarray:
    """Return line segments (Mx2x3) drawing a camera frustum in world coords.

    R_w2c, t_w2c is the world->camera transform. Equivalently the camera
    center in world is C = -R_w2c.T @ t_w2c, and a pixel (u,v) at depth d
    corresponds to world point C + d * R_w2c.T @ K^-1 @ [u,v,1].
    """
    w, h = image_size
    pixels = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float64)
    homo = np.hstack([pixels, np.ones((4, 1))])
    rays_cam = (np.linalg.inv(K) @ homo.T).T
    rays_world = (R_w2c.T @ rays_cam.T).T  # rotate into world
    C = -R_w2c.T @ t_w2c
    far = C + depth * rays_world
    segments = []
    # Lines from center to four corners.
    for f in far:
        segments.append([C, f])
    # Image plane rectangle.
    for i in range(4):
        segments.append([far[i], far[(i + 1) % 4]])
    return np.array(segments)  # (M, 2, 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose", required=True, type=Path)
    parser.add_argument("--image_size", type=int, nargs=2, default=None,
                        help="WxH; if omitted, derived from K's principal point.")
    parser.add_argument("--frustum_depth", type=float, default=1.0)
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    data = np.load(args.pose)
    K = data["K"]
    R_ba = data["R_ba"]
    t_ba = data["t_ba"]
    pts_a = data["points_a"]
    pts_b = data["points_b"]

    if args.image_size is None:
        # K's cx, cy are at image_w/2, image_h/2 if we built K that way.
        w = int(round(2 * K[0, 2]))
        h = int(round(2 * K[1, 2]))
    else:
        w, h = args.image_size

    # Triangulate the click points.
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R_ba, t_ba.reshape(3, 1)])
    pts4d = cv2.triangulatePoints(P1, P2, pts_a.T, pts_b.T)
    pts3d = (pts4d[:3] / pts4d[3]).T  # in A frame

    # Camera A: world->A is identity. Camera B: cam_B = R_ba @ world + t_ba.
    R_a = np.eye(3)
    t_a = np.zeros(3)
    R_b = R_ba
    t_b = t_ba

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    seg_a = frustum_lines(K, R_a, t_a, (w, h), depth=args.frustum_depth)
    seg_b = frustum_lines(K, R_b, t_b, (w, h), depth=args.frustum_depth)

    for seg in seg_a:
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="C0", linewidth=1.5)
    for seg in seg_b:
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color="C3", linewidth=1.5)

    ax.scatter([0], [0], [0], color="C0", s=60, label="Camera A")
    C_b = -R_b.T @ t_b
    ax.scatter([C_b[0]], [C_b[1]], [C_b[2]], color="C3", s=60, label="Camera B")

    # World points.
    ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2], c="black", s=20,
               label="triangulated clicks")

    # Suggested director's view: midpoint, no rotation, slightly above.
    director = (np.zeros(3) + C_b) / 2.0
    director[1] -= 0.3  # a little higher (Y down in image convention)
    ax.scatter([director[0]], [director[1]], [director[2]],
               color="C2", s=80, marker="*", label="director's view (suggested)")

    ax.set_xlabel("X (m, right in A)")
    ax.set_ylabel("Y (m, down in A)")
    ax.set_zlabel("Z (m, forward in A)")
    ax.legend()
    ax.set_title(f"Recovered geometry (baseline = {np.linalg.norm(t_b):.2f} m)")

    # Make axes roughly equal.
    all_xyz = np.vstack([np.array([[0, 0, 0]]),
                         C_b.reshape(1, 3),
                         pts3d])
    rng = np.ptp(all_xyz, axis=0).max() * 0.6 + 0.5
    mid = all_xyz.mean(axis=0)
    ax.set_xlim(mid[0] - rng, mid[0] + rng)
    ax.set_ylim(mid[1] - rng, mid[1] + rng)
    ax.set_zlim(mid[2] - rng, mid[2] + rng)

    if args.save:
        fig.savefig(args.save, dpi=120, bbox_inches="tight")
        print(f"[viz] saved figure to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
