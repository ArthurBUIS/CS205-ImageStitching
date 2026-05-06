"""
Interactive tool to manually pick matching points between frame 0 of two videos.

Usage:
    python pick_correspondences.py --video_a A.mp4 --video_b B.mp4 \
                                   --output clicks.json [--frame 0]

Controls:
    Left click on the LEFT panel  to start a new correspondence (point in A).
    Left click on the RIGHT panel to complete it (matching point in B).
    The left/right alternation is enforced: you must click left before right.

    'u'   undo last completed pair (or undo a half-completed click)
    'r'   reset all
    's'   save and quit
    'q'   quit without saving
    'h'   show help in console

Tips for picking good correspondences:
    - Click on STATIC scene features only -- corners of furniture, edges of
      doors, posters, light fixtures, electrical outlets. Avoid the moving
      person and avoid screens that show other camera feeds.
    - Spread the points across the image: not all clustered in one region.
    - Distribute in depth: pick some points on near objects and some on far
      objects. Co-planar points cause pose ambiguity.
    - 8 points minimum. 12-15 is comfortable. More is fine.
    - Aim for sub-pixel precision when possible; zoom in mentally on a corner.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


HELP_TEXT = """
Pick matching points between video A (left) and video B (right).
- Click left panel first, then right panel, to complete one correspondence.
- 'u' undo, 'r' reset, 's' save+quit, 'q' quit, 'h' help.
"""


class Picker:
    def __init__(self, frame_a: np.ndarray, frame_b: np.ndarray,
                 max_panel_h: int = 720):
        # Resize each panel to fit on screen while remembering the scale so we
        # can store clicks in the original image coordinates.
        h_a, w_a = frame_a.shape[:2]
        h_b, w_b = frame_b.shape[:2]
        scale = min(1.0, max_panel_h / max(h_a, h_b))
        self.scale = scale
        self.frame_a = frame_a
        self.frame_b = frame_b
        self.disp_a = cv2.resize(frame_a, (int(w_a * scale), int(h_a * scale)))
        self.disp_b = cv2.resize(frame_b, (int(w_b * scale), int(h_b * scale)))
        self.panel_w_a = self.disp_a.shape[1]
        self.panel_h = max(self.disp_a.shape[0], self.disp_b.shape[0])
        self.disp_a = self._pad_to_panel_height(self.disp_a)
        self.disp_b = self._pad_to_panel_height(self.disp_b)

        # State
        self.points_a: list[tuple[float, float]] = []
        self.points_b: list[tuple[float, float]] = []
        self.pending_a: tuple[float, float] | None = None
        self.window = "pick_correspondences"
        cv2.namedWindow(self.window, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window, self._on_mouse)

    def _pad_to_panel_height(self, img: np.ndarray) -> np.ndarray:
        pad_h = self.panel_h - img.shape[0]
        if pad_h <= 0:
            return img
        return cv2.copyMakeBorder(img, 0, pad_h, 0, 0,
                                  borderType=cv2.BORDER_CONSTANT,
                                  value=0)

    # ----- drawing -----------------------------------------------------------
    def _draw(self) -> np.ndarray:
        canvas = np.hstack([self.disp_a.copy(), self.disp_b.copy()])
        # Draw all completed pairs in the same color, indexed.
        rng = np.random.default_rng(42)
        for idx, (pa, pb) in enumerate(zip(self.points_a, self.points_b)):
            color = tuple(int(c) for c in rng.integers(64, 255, size=3))
            ax = int(pa[0] * self.scale)
            ay = int(pa[1] * self.scale)
            bx = int(pb[0] * self.scale) + self.panel_w_a
            by = int(pb[1] * self.scale)
            cv2.circle(canvas, (ax, ay), 6, color, 2)
            cv2.circle(canvas, (bx, by), 6, color, 2)
            cv2.putText(canvas, str(idx), (ax + 8, ay - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(canvas, str(idx), (bx + 8, by - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.line(canvas, (ax, ay), (bx, by), color, 1)
        # Draw a half-complete click if there is one.
        if self.pending_a is not None:
            ax = int(self.pending_a[0] * self.scale)
            ay = int(self.pending_a[1] * self.scale)
            cv2.circle(canvas, (ax, ay), 8, (0, 255, 255), 2)
            cv2.putText(canvas, "click match in B ->",
                        (ax + 10, ay), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1)
        # Status bar at the bottom.
        status = (f"pairs: {len(self.points_a)}  "
                  f"pending: {'yes' if self.pending_a else 'no'}  "
                  "[u]ndo  [r]eset  [s]ave+quit  [q]uit  [h]elp")
        bar = np.zeros((28, canvas.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, status, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        return np.vstack([canvas, bar])

    # ----- mouse callback ----------------------------------------------------
    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        # Determine which panel.
        if x < self.panel_w_a:
            # Click in A.
            if self.pending_a is not None:
                # User clicked A again before finishing -> replace pending.
                print("[pick] replacing pending point in A")
            real_x = x / self.scale
            real_y = y / self.scale
            self.pending_a = (real_x, real_y)
        else:
            # Click in B.
            if self.pending_a is None:
                print("[pick] click on the LEFT panel first to start a pair")
                return
            real_x = (x - self.panel_w_a) / self.scale
            real_y = y / self.scale
            self.points_a.append(self.pending_a)
            self.points_b.append((real_x, real_y))
            print(f"[pick] pair {len(self.points_a) - 1}: "
                  f"A=({self.pending_a[0]:.1f},{self.pending_a[1]:.1f})  "
                  f"B=({real_x:.1f},{real_y:.1f})")
            self.pending_a = None

    # ----- main loop ---------------------------------------------------------
    def run(self) -> tuple[list, list]:
        print(HELP_TEXT)
        while True:
            cv2.imshow(self.window, self._draw())
            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return [], []
            if key == ord('s'):
                cv2.destroyAllWindows()
                return self.points_a, self.points_b
            if key == ord('u'):
                if self.pending_a is not None:
                    self.pending_a = None
                    print("[pick] undid pending click")
                elif self.points_a:
                    self.points_a.pop()
                    self.points_b.pop()
                    print(f"[pick] undid pair, now {len(self.points_a)} pairs")
            if key == ord('r'):
                self.points_a.clear()
                self.points_b.clear()
                self.pending_a = None
                print("[pick] reset all")
            if key == ord('h'):
                print(HELP_TEXT)


def grab_frame(path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from {path}")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_a", required=True, type=Path)
    parser.add_argument("--video_b", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frame", type=int, default=0,
                        help="Which frame index to use from each video.")
    parser.add_argument("--max_panel_h", type=int, default=720,
                        help="Max display height per panel in pixels.")
    args = parser.parse_args()

    fa = grab_frame(args.video_a, args.frame)
    fb = grab_frame(args.video_b, args.frame)
    print(f"[pick] video A frame size: {fa.shape[1]}x{fa.shape[0]}")
    print(f"[pick] video B frame size: {fb.shape[1]}x{fb.shape[0]}")
    if fa.shape[:2] != fb.shape[:2]:
        print("[warn] frames have different sizes; this is unusual for a "
              "stereo rig but won't break anything here.")

    picker = Picker(fa, fb, max_panel_h=args.max_panel_h)
    pts_a, pts_b = picker.run()

    if not pts_a:
        print("[pick] no points to save (or quit without save). Exiting.")
        return
    if len(pts_a) < 8:
        print(f"[warn] only {len(pts_a)} pairs picked. 8+ is recommended for "
              f"reliable pose estimation. Saving anyway.")

    payload = {
        "video_a": str(args.video_a),
        "video_b": str(args.video_b),
        "frame_index": args.frame,
        "image_size_a": [fa.shape[1], fa.shape[0]],
        "image_size_b": [fb.shape[1], fb.shape[0]],
        "points_a": pts_a,
        "points_b": pts_b,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    print(f"[pick] saved {len(pts_a)} pairs to {args.output}")


if __name__ == "__main__":
    main()
