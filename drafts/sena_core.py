"""
sena_core.py
============
Core implementation of the SENA algorithm:

  "Seamlessly Natural: Image Stitching with Natural Appearance Preservation"
  Tchana et al., Technologies 2026, 14(3), 186.
  https://doi.org/10.3390/technologies14030186

Architecture
------------
Three independent, importable stages:

  Stage 1 · LocalAffineWarper
    - Global affine estimation (RANSAC on XFeat matches)
    - Sutherland-Hodgman overlap polygon
    - 2×2 local affine grid with ridge regression (λ1, λ2)
    - Confidence scoring (Eq. 1) + composite instability score (Eq. 2)
    - Confidence-weighted FFD field on a 64×64 lattice (Eq. 3)
    - Dual-channel seam gate: geometric ramp × match density (Eq. 4)
    - Final bilinear warp of source image onto canvas

  Stage 2 · AdequateZoneDetector
    - 20-class disparity binning
    - Threshold-based clustering (Algorithm 1)
    - Cluster scoring (Eq. 5) → optimal parallax-minimised zone

  Stage 3 · AnchorPartitioner
    - Keypoint chain refinement (brightness filter + greedy unique-x chain)
    - Directional segment validation
    - Vertical slice extraction, linear alpha blending, Gaussian seam smoothing

Feature extraction
------------------
Uses XFeat (CVPR 2024) as specified by the paper.
Install:
    pip install accelerated-features
    # or: pip install git+https://github.com/verlab/accelerated_features

If XFeat is unavailable, the module falls back to SIFT automatically and
prints a warning. The rest of the algorithm is identical.

All paper hyperparameters are exposed as constructor arguments with the
exact values reported in Section 4.1.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# XFeat import with SIFT fallback
# ---------------------------------------------------------------------------

_XFEAT_AVAILABLE = False
try:
    import torch
    from accelerated_features import XFeat as _XFeatModule
    _XFEAT_AVAILABLE = True
except ImportError:
    pass


def _load_xfeat():
    """Return an XFeat model on CPU, or None if unavailable."""
    if not _XFEAT_AVAILABLE:
        return None
    return _XFeatModule()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _smootherstep(t: np.ndarray) -> np.ndarray:
    """C2-smooth 0→1 ramp (used in seam gate)."""
    t = np.clip(t, 0.0, 1.0)
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def _inside_half_plane(p, a, b):
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0


def _intersect_lines(s, e, a, b):
    dc = [a[0] - b[0], a[1] - b[1]]
    dp = [s[0] - e[0], s[1] - e[1]]
    n1 = a[0] * b[1] - a[1] * b[0]
    n2 = s[0] * e[1] - s[1] * e[0]
    denom = dc[0] * dp[1] - dc[1] * dp[0]
    if abs(denom) < 1e-12:
        return list(s)
    n3 = 1.0 / denom
    return [(n1 * dp[0] - n2 * dc[0]) * n3,
            (n1 * dp[1] - n2 * dc[1]) * n3]


def sutherland_hodgman(subject: list, clip_w: float, clip_h: float) -> list:
    """
    Clip a convex polygon `subject` (list of [x,y]) against the
    axis-aligned rectangle [0, clip_w] × [0, clip_h].
    Returns list of [x,y] vertices (may be empty).
    """
    clip_poly = [[0, 0], [clip_w, 0], [clip_w, clip_h], [0, clip_h]]
    output = [list(p) for p in subject]
    for i in range(len(clip_poly)):
        if not output:
            break
        inp = output
        output = []
        a = clip_poly[i]
        b = clip_poly[(i + 1) % len(clip_poly)]
        for j in range(len(inp)):
            cur  = inp[j]
            prev = inp[(j - 1) % len(inp)]
            if _inside_half_plane(cur, a, b):
                if not _inside_half_plane(prev, a, b):
                    output.append(_intersect_lines(prev, cur, a, b))
                output.append(cur)
            elif _inside_half_plane(prev, a, b):
                output.append(_intersect_lines(prev, cur, a, b))
    return output


def _poly_to_mask(poly: list, h: int, w: int) -> np.ndarray:
    """Rasterise a polygon into a binary uint8 mask."""
    if not poly:
        return np.zeros((h, w), np.uint8)
    pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


# ---------------------------------------------------------------------------
# Stage 1 — Locally Adaptive Affine Warper
# ---------------------------------------------------------------------------

@dataclass
class WarpConfig:
    # Grid
    grid_x: int = 2
    grid_y: int = 2
    lattice_y: int = 64
    lattice_x: int = 64
    # Ridge regularisation
    lambda1: float = 2.2
    lambda2: float = 2.8
    # Confidence
    kappa_min: float = 0.2
    kappa_max: float = 1.5
    alpha_sigma: float = 0.5   # cell bounding-box diagonal scale for Gaussian weight
    beta_norm: float = 1.0
    # FFD
    alpha_f: float = 0.5       # spatial decay factor for blending
    d_max: float = 50.0        # displacement clipping (px)
    sigma_l: float = 1.0       # lattice Gaussian smooth σ
    # Instability score weights (Eq. 2)
    w_cond: float = 0.01
    w_det: float = 1.0
    w_delta: float = 0.5
    tau_det: float = 0.1
    Ng: int = 4                # evaluation grid for delta_mean
    # Seam gate (Eq. 4)
    rho: float = 0.10
    gamma_p: float = 1.2
    sigma_d: float = 24.0
    gamma_min: float = 0.30
    sigma_g: float = 1.1


class LocalAffineWarper:
    """
    Implements Section 3.2 of SENA: locally adaptive affine warping.
    Takes a source image and a target image, aligns source → target canvas.
    """

    def __init__(self, cfg: WarpConfig = None):
        self.cfg = cfg or WarpConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def warp(
        self,
        img_src: np.ndarray,
        img_tgt: np.ndarray,
        pts_src: np.ndarray,   # Nx2 float32 inlier matches in source image
        pts_tgt: np.ndarray,   # Nx2 float32 inlier matches in target image
        A_glob: np.ndarray,    # 3×3 global affine (src→tgt)
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Tuple[int, int]]:
        """
        Warp img_src into target canvas frame.

        Returns
        -------
        warped_src  : canvas-sized BGR image (source warped)
        canvas_tgt  : canvas-sized BGR image (target placed)
        overlap_mask: canvas-sized uint8 binary mask of overlap region
        offset_xy   : (ox, oy) — where target origin sits on canvas
        """
        cfg = self.cfg
        hs, ws = img_src.shape[:2]
        ht, wt = img_tgt.shape[:2]

        # ── Step 2: compute canvas size and overlap mask ──────────────────
        # Project source corners into target space
        src_corners = np.array([[0,0],[ws,0],[ws,hs],[0,hs]], np.float64)
        src_h = np.hstack([src_corners, np.ones((4,1))])
        tgt_corners_raw = (A_glob @ src_h.T).T[:, :2]

        # All corners (target image + warped source)
        tgt_img_corners = np.array([[0,0],[wt,0],[wt,ht],[0,ht]], np.float64)
        all_corners = np.vstack([tgt_img_corners, tgt_corners_raw])
        x_min = np.floor(all_corners[:, 0].min()).astype(int)
        y_min = np.floor(all_corners[:, 1].min()).astype(int)
        x_max = np.ceil(all_corners[:, 0].max()).astype(int)
        y_max = np.ceil(all_corners[:, 1].max()).astype(int)

        ox = int(-x_min) if x_min < 0 else 0
        oy = int(-y_min) if y_min < 0 else 0
        canvas_w = int(x_max - x_min)
        canvas_h = int(y_max - y_min)

        # Overlap polygon via Sutherland-Hodgman (source quad clipped to target)
        src_poly_tgt_space = tgt_corners_raw.tolist()
        overlap_poly = sutherland_hodgman(src_poly_tgt_space, wt, ht)
        # Shift to canvas coordinates
        overlap_poly_canvas = [[p[0] + ox, p[1] + oy] for p in overlap_poly]
        overlap_mask = _poly_to_mask(overlap_poly_canvas, canvas_h, canvas_w)

        # ── Step 3: local affine refinement on 2×2 grid ──────────────────
        # Overlap bounding box in target coordinates
        if overlap_poly:
            op = np.array(overlap_poly)
            ov_x0, ov_y0 = op[:, 0].min(), op[:, 1].min()
            ov_x1, ov_y1 = op[:, 0].max(), op[:, 1].max()
        else:
            ov_x0, ov_y0, ov_x1, ov_y1 = 0, 0, wt, ht

        cell_w = (ov_x1 - ov_x0) / cfg.grid_x
        cell_h = (ov_y1 - ov_y0) / cfg.grid_y

        local_transforms = []
        for gy in range(cfg.grid_y):
            for gx in range(cfg.grid_x):
                cx0 = ov_x0 + gx * cell_w
                cy0 = ov_y0 + gy * cell_h
                cx1 = cx0 + cell_w
                cy1 = cy0 + cell_h
                centroid_x = (cx0 + cx1) / 2.0
                centroid_y = (cy0 + cy1) / 2.0
                diag = np.sqrt(cell_w**2 + cell_h**2)
                sigma_j = cfg.alpha_sigma * diag

                # Collect inlier points in cell + neighbours (1-cell margin)
                margin = max(cell_w, cell_h)
                mask_cell = (
                    (pts_tgt[:, 0] >= cx0 - margin) & (pts_tgt[:, 0] < cx1 + margin) &
                    (pts_tgt[:, 1] >= cy0 - margin) & (pts_tgt[:, 1] < cy1 + margin)
                )
                ps_cell = pts_src[mask_cell]
                pt_cell = pts_tgt[mask_cell]

                # Confidence score (Eq. 1)
                if len(ps_cell) >= 3:
                    dist2 = ((pt_cell[:, 0] - centroid_x)**2 +
                             (pt_cell[:, 1] - centroid_y)**2)
                    w_i = np.exp(-dist2 / (2 * sigma_j**2 + 1e-12))
                    w_sum = w_i.sum()
                    w_max = w_i.max() if w_i.max() > 0 else 1.0
                    conf_j = float(np.clip(
                        w_sum / (cfg.beta_norm * w_max + 1e-9),
                        cfg.kappa_min, cfg.kappa_max))
                else:
                    conf_j = cfg.kappa_min

                # Fit local affine (λ1)
                if len(ps_cell) >= 3:
                    T_j = self._fit_affine_ridge(ps_cell, pt_cell, cfg.lambda1, A_glob)
                    sc1 = self._instability_score(T_j, ps_cell, pt_cell, A_glob,
                                                  cx0, cy0, cx1, cy1)
                    # Retry with λ2 if unstable
                    T_j2 = self._fit_affine_ridge(ps_cell, pt_cell, cfg.lambda2, A_glob)
                    sc2  = self._instability_score(T_j2, ps_cell, pt_cell, A_glob,
                                                   cx0, cy0, cx1, cy1)
                    T_j = T_j if sc1 <= sc2 else T_j2
                else:
                    T_j = A_glob.copy()

                local_transforms.append((T_j, conf_j, (centroid_x, centroid_y), diag))

        # ── Steps 4-5: FFD field + seam gate ─────────────────────────────
        dx_gated, dy_gated = self._build_ffd_and_gate(
            local_transforms, A_glob, canvas_h, canvas_w,
            (ox, oy), overlap_mask, pts_tgt)

        # ── Step 6: final warp ────────────────────────────────────────────
        # Build full remap: source_xy = A_glob_inv(canvas_xy - offset) + FFD
        map_x, map_y = self._build_remap(
            A_glob, dx_gated, dy_gated, canvas_h, canvas_w, ox, oy)

        warped_src = cv2.remap(img_src, map_x, map_y,
                               interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        # Place target image on canvas
        canvas_tgt = np.zeros((canvas_h, canvas_w, 3), np.uint8)
        canvas_tgt[oy:oy+ht, ox:ox+wt] = img_tgt

        return warped_src, canvas_tgt, overlap_mask, (ox, oy)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_affine_ridge(pts_s: np.ndarray, pts_t: np.ndarray,
                          lam: float, A_glob: np.ndarray) -> np.ndarray:
        """Ridge-regularised affine fit, biased toward A_glob."""
        n = len(pts_s)
        X = np.hstack([pts_s.astype(np.float64),
                       np.ones((n, 1))])              # Nx3
        Y = pts_t.astype(np.float64)                  # Nx2
        W_prior = A_glob[:2, :].T                     # 3x2
        A_mat = X.T @ X + lam * np.eye(3)
        b_mat = X.T @ Y + lam * W_prior
        W = np.linalg.solve(A_mat, b_mat)
        T = np.eye(3, dtype=np.float64)
        T[:2, :] = W.T
        return T

    def _instability_score(self, T: np.ndarray,
                           pts_s: np.ndarray, pts_t: np.ndarray,
                           A_glob: np.ndarray,
                           x0: float, y0: float,
                           x1: float, y1: float) -> float:
        """Composite instability score (Eq. 2)."""
        cfg = self.cfg
        n = len(pts_s)
        if n == 0:
            return 1e9
        X = np.hstack([pts_s.astype(np.float64), np.ones((n, 1))])
        pred = (T @ X.T).T[:, :2]
        rmse = float(np.sqrt(np.mean((pred - pts_t.astype(np.float64))**2)))
        det  = float(abs(np.linalg.det(T[:2, :2])))
        cond = float(np.linalg.cond(T[:2, :2]))
        # delta_mean over Ng×Ng grid inside cell
        xs = np.linspace(x0, x1, cfg.Ng)
        ys = np.linspace(y0, y1, cfg.Ng)
        gx, gy = np.meshgrid(xs, ys)
        grid = np.stack([gx.ravel(), gy.ravel(), np.ones(cfg.Ng**2)], 1)
        delta = np.mean(np.linalg.norm(
            (T @ grid.T).T[:, :2] - (A_glob @ grid.T).T[:, :2], axis=1))
        score = (rmse
                 + cfg.w_cond * cond
                 + cfg.w_det * max(0.0, cfg.tau_det - det)
                 + cfg.w_delta * delta)
        return score

    def _build_ffd_and_gate(
        self,
        local_transforms: list,
        A_glob: np.ndarray,
        canvas_h: int, canvas_w: int,
        offset_xy: Tuple[int, int],
        overlap_mask: np.ndarray,
        pts_tgt: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Steps 4–5: build FFD displacement field and apply seam gate."""
        cfg = self.cfg
        Ny, Nx = cfg.lattice_y, cfg.lattice_x
        ox, oy = offset_xy

        # Lattice coordinates
        u = np.linspace(0, canvas_w - 1, Nx)
        v = np.linspace(0, canvas_h - 1, Ny)
        uu, vv = np.meshgrid(u, v)
        pt_x = (uu - ox).ravel()
        pt_y = (vv - oy).ravel()
        ones  = np.ones_like(pt_x)
        pt_h  = np.stack([pt_x, pt_y, ones], 0)   # 3×(Ny*Nx)

        A_inv = np.linalg.inv(A_glob)
        p_base = (A_inv @ pt_h)[:2, :].T           # (Ny*Nx)×2

        # sigma_f from mean cell diagonal
        mean_diag = float(np.mean([d for *_, d in local_transforms])) \
                    if local_transforms else float(np.sqrt(canvas_w**2 + canvas_h**2))
        sigma_f = cfg.alpha_f * mean_diag

        disp_accum   = np.zeros((Ny * Nx, 2), np.float64)
        weight_accum = np.zeros(Ny * Nx, np.float64)

        for T_j, conf_j, (cx, cy), _ in local_transforms:
            T_inv   = np.linalg.inv(T_j)
            p_local = (T_inv @ pt_h)[:2, :].T
            delta_p = p_local - p_base

            dist2  = (pt_x - cx)**2 + (pt_y - cy)**2
            w_j    = conf_j * np.exp(-dist2 / (2 * sigma_f**2 + 1e-12))
            disp_accum   += w_j[:, None] * delta_p
            weight_accum += w_j

        weight_accum = np.maximum(weight_accum, 1e-9)
        delta_p_field = disp_accum / weight_accum[:, None]
        delta_p_field = np.clip(delta_p_field, -cfg.d_max, cfg.d_max)

        # Smooth on lattice
        for ch in range(2):
            layer = delta_p_field[:, ch].reshape(Ny, Nx).astype(np.float32)
            layer = cv2.GaussianBlur(layer, (0, 0), cfg.sigma_l)
            delta_p_field[:, ch] = layer.ravel()

        # Upsample to canvas
        dx_full = cv2.resize(
            delta_p_field[:, 0].reshape(Ny, Nx).astype(np.float32),
            (canvas_w, canvas_h), interpolation=cv2.INTER_CUBIC)
        dy_full = cv2.resize(
            delta_p_field[:, 1].reshape(Ny, Nx).astype(np.float32),
            (canvas_w, canvas_h), interpolation=cv2.INTER_CUBIC)

        # ── Seam gate (Eq. 4) ─────────────────────────────────────────────
        diag_img = float(np.sqrt(canvas_w**2 + canvas_h**2))
        bw = cfg.rho * diag_img
        # Geometric ramp
        dist_raw  = cv2.distanceTransform(overlap_mask, cv2.DIST_L2, 5)
        dist_norm = np.clip(dist_raw / (bw + 1e-9), 0, 1).astype(np.float32)
        R_canvas  = _smootherstep(dist_norm) ** cfg.gamma_p

        # Match density map
        D_canvas = np.zeros((canvas_h, canvas_w), np.float32)
        if len(pts_tgt) > 0:
            for px, py in pts_tgt:
                ix = int(np.clip(px + ox, 0, canvas_w - 1))
                iy = int(np.clip(py + oy, 0, canvas_h - 1))
                D_canvas[iy, ix] = 1.0
            D_canvas = cv2.GaussianBlur(D_canvas, (0, 0), cfg.sigma_d)
            dmax = D_canvas.max()
            if dmax > 0:
                D_canvas /= dmax

        G = R_canvas * (cfg.gamma_min + (1 - cfg.gamma_min) * D_canvas)

        dx_gated = cv2.GaussianBlur(dx_full * G, (0, 0), cfg.sigma_g)
        dy_gated = cv2.GaussianBlur(dy_full * G, (0, 0), cfg.sigma_g)

        return dx_gated, dy_gated

    @staticmethod
    def _build_remap(A_glob: np.ndarray,
                     dx: np.ndarray, dy: np.ndarray,
                     canvas_h: int, canvas_w: int,
                     ox: int, oy: int
                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build cv2.remap maps: for each canvas pixel (u,v),
        compute the source pixel via A_glob_inv + FFD.
        """
        u_range = np.arange(canvas_w, dtype=np.float32)
        v_range = np.arange(canvas_h, dtype=np.float32)
        uu, vv = np.meshgrid(u_range, v_range)

        # Target-space coordinates
        pt_x = (uu - ox).ravel()
        pt_y = (vv - oy).ravel()
        ones  = np.ones_like(pt_x)
        pt_h  = np.stack([pt_x, pt_y, ones], 0)

        A_inv   = np.linalg.inv(A_glob)
        p_base  = (A_inv @ pt_h.astype(np.float64))[:2, :].T   # Nx2

        src_x = (p_base[:, 0] + dx.ravel().astype(np.float64)).reshape(canvas_h, canvas_w)
        src_y = (p_base[:, 1] + dy.ravel().astype(np.float64)).reshape(canvas_h, canvas_w)

        return src_x.astype(np.float32), src_y.astype(np.float32)


# ---------------------------------------------------------------------------
# Stage 2 — Adequate Zone Detector
# ---------------------------------------------------------------------------

@dataclass
class ZoneConfig:
    n_classes: int = 20         # disparity bins (width / n_classes)
    v: float = 5.0              # clustering threshold (px)
    lam: float = 1.0            # lambda weight in Eq. 5
    eps: float = 1e-6           # division guard


class AdequateZoneDetector:
    """
    Section 3.3: identifies the parallax-minimised adequate zone.
    Returns the x-column range [x_start, x_end] within the overlap.
    """

    def __init__(self, cfg: ZoneConfig = None):
        self.cfg = cfg or ZoneConfig()

    def detect(self,
               pts_src: np.ndarray,   # Nx2 inlier matches in source image
               pts_tgt: np.ndarray,   # Nx2 inlier matches in target image
               image_width: int
               ) -> Tuple[int, int]:
        """
        Returns (x_start, x_end) of the adequate parallax-minimised zone
        in target image coordinates.
        Falls back to full width if detection fails.
        """
        cfg = self.cfg
        if len(pts_src) < 4:
            return 0, image_width

        R = image_width / cfg.n_classes

        class_disparities:   List[float] = []
        class_cardinalities: List[int]   = []
        class_x_ranges:      List[Tuple] = []

        for i in range(cfg.n_classes):
            x_lo = i * R
            x_hi = (i + 1) * R
            mask = (pts_src[:, 0] >= x_lo) & (pts_src[:, 0] < x_hi)
            d = pts_src[mask, 0] - pts_tgt[mask, 0]
            if len(d) == 0:
                prev = class_disparities[-1] if class_disparities else 0.0
                class_disparities.append(prev)
                class_cardinalities.append(0)
            else:
                class_disparities.append(float(np.mean(d)))
                class_cardinalities.append(int(len(d)))
            class_x_ranges.append((x_lo, x_hi))

        disp_arr = np.array(class_disparities)
        global_mean = float(disp_arr.mean())

        clusters = self._cluster(disp_arr, cfg.v)
        if not clusters:
            return 0, image_width

        best = max(clusters, key=lambda cl: self._score(
            cl, disp_arr, class_cardinalities, global_mean))

        x_start = int(class_x_ranges[best[0]][0])
        x_end   = int(class_x_ranges[best[-1]][1])
        return x_start, x_end

    # ------------------------------------------------------------------

    @staticmethod
    def _cluster(mean_disparities: np.ndarray, v: float) -> List[List[int]]:
        """Algorithm 1: threshold-based disparity clustering."""
        clusters: List[List[int]] = []
        current = [0]
        for i in range(1, len(mean_disparities)):
            if abs(mean_disparities[i] - mean_disparities[i - 1]) <= v:
                current.append(i)
            else:
                if len(current) >= 2:
                    clusters.append(current)
                current = [i]
        if len(current) >= 2:
            clusters.append(current)
        return clusters

    def _score(self, cluster_indices: List[int],
               class_disparities: np.ndarray,
               class_cardinalities: List[int],
               global_mean: float) -> float:
        """Equation 5: cluster quality score."""
        cfg = self.cfg
        d   = [class_disparities[i]   for i in cluster_indices]
        c   = [class_cardinalities[i] for i in cluster_indices]
        sigma    = float(np.std(d)) if len(d) > 1 else 0.0
        C        = float(sum(c))
        mu_c     = float(np.mean(d))
        delta_mu = abs(mu_c - global_mean)
        return C / (sigma + cfg.lam * delta_mu + cfg.eps)


# ---------------------------------------------------------------------------
# Stage 3 — Anchor-Based Partitioner
# ---------------------------------------------------------------------------

@dataclass
class PartitionConfig:
    blend_sigma: float = 2.0    # Gaussian σ for seam smoothing (px)
    brightness_tol: float = 30.0  # max intensity difference for keypoint keep


class AnchorPartitioner:
    """
    Section 3.3–3.4: keypoint chain refinement, directional segment
    validation, slice extraction, alpha blend + reconstruction.
    """

    def __init__(self, cfg: PartitionConfig = None):
        self.cfg = cfg or PartitionConfig()

    def partition_and_reconstruct(
        self,
        warped_src: np.ndarray,   # canvas-sized warped source (BGR)
        canvas_tgt: np.ndarray,   # canvas-sized target placement (BGR)
        pts_src_canvas: np.ndarray,  # Nx2 inlier keypoints in canvas coords (src side)
        pts_tgt_canvas: np.ndarray,  # Nx2 inlier keypoints in canvas coords (tgt side)
        zone_x0: int,
        zone_x1: int,
    ) -> np.ndarray:
        """
        Returns final stitched image (same size as inputs, uint8 BGR).
        """
        canvas_h, canvas_w = warped_src.shape[:2]

        # Filter keypoints to the adequate zone
        in_zone = (pts_src_canvas[:, 0] >= zone_x0) & \
                  (pts_src_canvas[:, 0] < zone_x1)
        pts_s_z = pts_src_canvas[in_zone]
        pts_t_z = pts_tgt_canvas[in_zone]

        # Refine keypoint chain
        chain_s, chain_t = self._refine_chain(
            pts_s_z, pts_t_z, warped_src, canvas_tgt, canvas_w)

        if len(chain_s) < 2:
            # Fallback: simple 50/50 alpha blend
            return self._alpha_blend_full(warped_src, canvas_tgt)

        return self._reconstruct(
            warped_src, canvas_tgt, chain_s, chain_t,
            canvas_h, canvas_w)

    # ------------------------------------------------------------------

    def _refine_chain(self,
                      pts_s: np.ndarray,
                      pts_t: np.ndarray,
                      img_s: np.ndarray,
                      img_t: np.ndarray,
                      canvas_w: int
                      ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build an ordered, unique-x keypoint chain.
        Steps: brightness consistency filter → greedy nearest-neighbour → sort by x.
        """
        cfg = self.cfg
        if len(pts_s) == 0:
            return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

        h, w = img_s.shape[:2]

        # Brightness filter
        keep = []
        for i, (ps, pt) in enumerate(zip(pts_s, pts_t)):
            xs, ys = int(np.clip(ps[0], 0, w-1)), int(np.clip(ps[1], 0, h-1))
            xt, yt = int(np.clip(pt[0], 0, w-1)), int(np.clip(pt[1], 0, h-1))
            gs = float(np.mean(img_s[ys, xs]))
            gt = float(np.mean(img_t[yt, xt]))
            if abs(gs - gt) <= cfg.brightness_tol:
                keep.append(i)
        if len(keep) < 2:
            keep = list(range(len(pts_s)))   # skip filter if too aggressive

        pts_s = pts_s[keep]
        pts_t = pts_t[keep]

        # Anchor at minimum x
        anchor = int(np.argmin(pts_s[:, 0]))
        used      = {anchor}
        used_x    = {int(pts_s[anchor, 0])}
        chain_s   = [pts_s[anchor]]
        chain_t   = [pts_t[anchor]]
        current   = anchor

        for _ in range(len(pts_s) - 1):
            free = [i for i in range(len(pts_s)) if i not in used]
            if not free:
                break
            dists = np.linalg.norm(pts_t[free] - pts_t[current], axis=1)
            nn    = free[int(np.argmin(dists))]
            xval  = int(pts_s[nn, 0])
            if xval in used_x:
                used.add(nn)   # mark as used but skip
                continue
            used.add(nn)
            used_x.add(xval)
            chain_s.append(pts_s[nn])
            chain_t.append(pts_t[nn])
            current = nn

        cs = np.array(chain_s, np.float32)
        ct = np.array(chain_t, np.float32)

        # Sort by x for clean slicing
        order = np.argsort(cs[:, 0])
        return cs[order], ct[order]

    @staticmethod
    def _validate_segment(ax: float, bx: float,
                           apx: float, bpx: float) -> bool:
        """True if both pairs move in the same horizontal direction."""
        return (ax > bx) == (apx > bpx)

    def _reconstruct(self,
                     img_s: np.ndarray,
                     img_t: np.ndarray,
                     chain_s: np.ndarray,
                     chain_t: np.ndarray,
                     canvas_h: int, canvas_w: int
                     ) -> np.ndarray:
        """
        Section 3.4.2: partition → validated slices → linear alpha blend
        → light Gaussian smoothing → horizontal + vertical stacking.
        """
        cfg = self.cfg
        # Anchor x-lists with sentinels
        ax_s = [0] + [int(round(p[0])) for p in chain_s] + [canvas_w]
        ax_t = [0] + [int(round(p[0])) for p in chain_t] + [canvas_w]
        n_slices = len(ax_s) - 1

        output    = np.zeros((canvas_h, canvas_w, 3), np.uint8)
        seam_cols = []

        for k in range(n_slices):
            xs0, xs1 = ax_s[k], ax_s[k + 1]
            xt0, xt1 = ax_t[k], ax_t[k + 1]

            # Directional validation for interior segments
            if 0 < k < n_slices - 1:
                if not self._validate_segment(ax_s[k], ax_s[k+1],
                                              ax_t[k], ax_t[k+1]):
                    continue

            sw = min(max(xs1 - xs0, 1), canvas_w - xs0)
            if sw <= 0:
                continue

            sl_s = img_s[:, xs0:xs0 + sw]
            # Resize target slice to same width
            sl_t_raw = img_t[:, xt0:min(xt1, canvas_w)]
            if sl_t_raw.shape[1] == 0:
                sl_t = np.zeros_like(sl_s)
            elif sl_t_raw.shape[1] != sw:
                sl_t = cv2.resize(sl_t_raw, (sw, canvas_h))
            else:
                sl_t = sl_t_raw

            # Linear alpha blend (Eq. in Section 3.4.2)
            alpha = np.linspace(1.0, 0.0, sw, dtype=np.float32)[None, :, None]
            blended = (sl_s.astype(np.float32) * alpha +
                       sl_t.astype(np.float32) * (1.0 - alpha)
                       ).astype(np.uint8)

            out_end = min(xs0 + sw, canvas_w)
            output[:, xs0:out_end] = blended[:, :out_end - xs0]
            seam_cols.append(out_end)

        # Light Gaussian at seam boundaries
        for sc in seam_cols:
            x0 = max(0, sc - 3)
            x1 = min(canvas_w, sc + 3)
            if x1 > x0:
                region = output[:, x0:x1].astype(np.float32)
                region = cv2.GaussianBlur(region, (0, 0), cfg.blend_sigma)
                output[:, x0:x1] = np.clip(region, 0, 255).astype(np.uint8)

        return output

    @staticmethod
    def _alpha_blend_full(img_s: np.ndarray, img_t: np.ndarray) -> np.ndarray:
        """Fallback: simple 50/50 blend where both images have pixels."""
        h, w = img_s.shape[:2]
        has_s = (img_s.sum(axis=2) > 0).astype(np.float32)
        has_t = (img_t.sum(axis=2) > 0).astype(np.float32)
        both  = has_s * has_t
        alpha = 0.5 * both + has_s * (1 - both)
        out   = (img_s.astype(np.float32) * alpha[:, :, None] +
                 img_t.astype(np.float32) * (1 - alpha[:, :, None]))
        return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Feature matching interface (XFeat or SIFT)
# ---------------------------------------------------------------------------

class FeatureMatcher:
    """
    Thin wrapper that exposes a single `match(img1_gray, img2_gray)` method
    returning (pts1, pts2) as float32 Nx2 arrays of RANSAC inliers.

    Uses XFeat if available, falls back to SIFT + BFMatcher + Lowe's ratio test.
    """

    def __init__(self,
                 use_xfeat: bool = True,
                 max_features: int = 5000,
                 ratio_thresh: float = 0.75,
                 sigma_max: float = 4.0,
                 device: str = "cpu"):
        self._sigma_max     = sigma_max
        self._max_features  = max_features
        self._ratio_thresh  = ratio_thresh
        self._xfeat_model   = None

        if use_xfeat:
            if _XFEAT_AVAILABLE:
                import torch
                self._device      = torch.device(device)
                self._xfeat_model = _load_xfeat()
                if self._xfeat_model is not None:
                    self._xfeat_model = self._xfeat_model.to(self._device)
                    print("[FeatureMatcher] Using XFeat.")
                else:
                    warnings.warn("[FeatureMatcher] XFeat model failed to load; "
                                  "falling back to SIFT.")
            else:
                warnings.warn(
                    "[FeatureMatcher] accelerated-features / torch not installed; "
                    "falling back to SIFT. Install with:\n"
                    "  pip install torch accelerated-features")

    def match(self,
              img1: np.ndarray,
              img2: np.ndarray
              ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Match img1 and img2 (BGR uint8).
        Returns (pts1, pts2) float32 Nx2 inlier arrays, or (None, None).
        """
        if self._xfeat_model is not None:
            return self._match_xfeat(img1, img2)
        return self._match_sift(img1, img2)

    def _match_xfeat(self, img1, img2):
        import torch
        model = self._xfeat_model

        def to_tensor(img):
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t   = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
            return t.to(self._device)

        with torch.no_grad():
            out1 = model.detectAndCompute(to_tensor(img1), top_k=self._max_features)
            out2 = model.detectAndCompute(to_tensor(img2), top_k=self._max_features)
            matches = model.match(out1, out2)

        if matches is None or len(matches) < 4:
            return None, None

        kp1 = out1['keypoints'][0].cpu().numpy()
        kp2 = out2['keypoints'][0].cpu().numpy()
        m   = matches.cpu().numpy()

        pts1 = kp1[m[:, 0]].astype(np.float32)
        pts2 = kp2[m[:, 1]].astype(np.float32)

        # RANSAC filtering (global affine)
        _, mask = cv2.estimateAffine2D(
            pts1, pts2,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=self._sigma_max)
        if mask is None or mask.sum() < 4:
            return None, None

        inliers = mask.ravel().astype(bool)
        return pts1[inliers], pts2[inliers]

    def _match_sift(self, img1, img2):
        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create(nfeatures=self._max_features)
        kp1, d1 = sift.detectAndCompute(g1, None)
        kp2, d2 = sift.detectAndCompute(g2, None)
        if d1 is None or d2 is None or len(kp1) < 4 or len(kp2) < 4:
            return None, None
        matches = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2)
        good = [m for m, n in matches
                if len([m, n]) == 2 and m.distance < self._ratio_thresh * n.distance]
        if len(good) < 4:
            return None, None
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        _, mask = cv2.findHomography(pts1, pts2, cv2.USAC_MAGSAC, self._sigma_max)
        if mask is None or mask.sum() < 4:
            return None, None
        inliers = mask.ravel().astype(bool)
        return pts1[inliers], pts2[inliers]

    def estimate_global_affine(self,
                               pts_src: np.ndarray,
                               pts_tgt: np.ndarray
                               ) -> Optional[np.ndarray]:
        """
        Estimate the global affine A_glob (3×3) mapping src → tgt,
        using MAGSAC++ for robustness.
        Returns None if estimation fails.
        """
        if len(pts_src) < 3:
            return None
        A, mask = cv2.estimateAffine2D(
            pts_src, pts_tgt,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=self._sigma_max,
            maxIters=2000,
            confidence=0.999)
        if A is None:
            return None
        A3 = np.eye(3, dtype=np.float64)
        A3[:2, :] = A
        return A3
