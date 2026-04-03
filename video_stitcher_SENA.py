import cv2
import numpy as np
import argparse
import sys

# Made with gemini, I can tell he's a terrible coder

class SENAStitcherLite:
    def __init__(self):
        # Using ORB: The SOTA review recommends this for real-time speed
        self.detector = cv2.ORB_create(nfeatures=3000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    def stitch_frames(self, img_l, img_r):
        # 1. Feature Extraction (SENA Global Alignment step)
        kp1, des1 = self.detector.detectAndCompute(img_l, None)
        kp2, des2 = self.detector.detectAndCompute(img_r, None)

        if des1 is None or des2 is None:
            return img_r

        # 2. Matching
        matches = self.matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        good = matches[:100] # Use top matches as "Anchors"

        if len(good) < 10:
            return img_r

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # 3. Global Alignment with USAC_MAGSAC (2026 SOTA RANSAC)
        # This replaces traditional RANSAC to minimize geometric distortion
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 5.0)

        # Check if homography was found
        if M is None:
            return img_r

        # 4. Overlap & Canvas Setup
        h1, w1 = img_l.shape[:2]
        h2, w2 = img_r.shape[:2]

        # Calculate dimensions for the stitched canvas
        pts = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        dst_transformed = cv2.perspectiveTransform(pts, M)

        [xmin, ymin] = np.int32(dst_transformed.min(axis=0).ravel() - 0.5)
        [xmax, ymax] = np.int32(dst_transformed.max(axis=0).ravel() + 0.5)

        # Ensure canvas accommodates both images
        xmin = min(xmin, 0)
        ymin = min(ymin, 0)
        xmax = max(xmax, w2)
        ymax = max(ymax, h2)

        # Translation to keep everything in frame
        t = [-xmin, -ymin]
        Ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]], dtype=np.float32)

        # 5. Warping & Anchor-Based Reconstruction
        # We warp the left image onto the expanded coordinate system
        canvas_w = xmax - xmin
        canvas_h = ymax - ymin
        result = cv2.warpPerspective(img_l, Ht @ M, (canvas_w, canvas_h))

        # Blend the right image (the reference "Anchor" frame) with overlap handling
        right_y_start = t[1]
        right_x_start = t[0]
        right_y_end = min(right_y_start + h2, canvas_h)
        right_x_end = min(right_x_start + w2, canvas_w)

        # Calculate overlap region for blending
        overlap_y_start = max(0, right_y_start)
        overlap_x_start = max(0, right_x_start)
        overlap_y_end = min(right_y_end, canvas_h)
        overlap_x_end = min(right_x_end, canvas_w)

        # Simple alpha blending in overlap region to reduce seams
        if (overlap_y_end > overlap_y_start) and (overlap_x_end > overlap_x_start):
            alpha = 0.5
            result[overlap_y_start:overlap_y_end, overlap_x_start:overlap_x_end] = cv2.addWeighted(
                result[overlap_y_start:overlap_y_end, overlap_x_start:overlap_x_end], alpha,
                img_r[overlap_y_start-right_y_start:overlap_y_end-right_y_start,
                      overlap_x_start-right_x_start:overlap_x_end-right_x_start], 1-alpha, 0
            )

        # Place non-overlapping parts of right image
        result[right_y_start:right_y_end, right_x_start:right_x_end] = img_r[
            :right_y_end-right_y_start, :right_x_end-right_x_start
        ]

        return result

def main():
    parser = argparse.ArgumentParser(description="SENA Video Stitcher (Torch-Free)")
    parser.add_argument("--left", required=True, help="Path to left video")
    parser.add_argument("--right", required=True, help="Path to right video")
    parser.add_argument("--output", required=True, help="Output video path")
    args = parser.parse_args()

    cap_l = cv2.VideoCapture(args.left)
    cap_r = cv2.VideoCapture(args.right)

    if not cap_l.isOpened() or not cap_r.isOpened():
        print("Error: Could not open videos.")
        cap_l.release()
        cap_r.release()
        return

    fps = cap_l.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # Default FPS if not available

    stitcher = SENAStitcherLite()
    writer = None
    frame_count = 0

    print("Running SENA-inspired Real-Time Pipeline...")

    try:
        while True:
            ret1, frame_l = cap_l.read()
            ret2, frame_r = cap_r.read()

            if not ret1 or not ret2:
                break

            # Process frame
            stitched = stitcher.stitch_frames(frame_l, frame_r)

            if writer is None:
                h, w = stitched.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
                if not writer.isOpened():
                    print("Error: Could not open output video writer.")
                    break

            if writer is not None:
                writer.write(stitched)
                frame_count += 1

            cv2.imshow('SENA Stitching (No-Torch)', stitched)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Interrupted by user.")
                break

    finally:
        cap_l.release()
        cap_r.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print(f"Done! Processed {frame_count} frames. Video saved to {args.output}")

if __name__ == "__main__":
    main()