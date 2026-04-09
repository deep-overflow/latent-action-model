"""Verify iterative flow composition vs naive sum.

For a given video and frame_skip K, compares two approaches to multi-frame flow:
  1. Naive sum: simply add K consecutive flows (current buggy code)
  2. Iterative warp: properly compose K flows via warping (proposed fix)

Validation: backward-warp frame_{t+K} using each flow. The result should match
frame_t. The warped reconstruction tells us how accurate each flow is.

Output layout (per row):
  frame_t | frame_{t+K} | naive flow | warped flow | naive recon | warped recon

Usage:
    python scripts/verify_flow_warping.py \
        --video /media/data1/chan/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot/videos/chunk-000/observation.images.ego_view_freq20/episode_000000.mp4 \
        --flow-dir /media/data1/chan/GR1_optical_flow \
        --video-root /media/data1/chan/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot \
        --output verify_flow.png
"""

import argparse
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lam.flow_utils import compose_flow_and_mask

TARGET_H, TARGET_W = 240, 320


def load_video_frames(video_path: str) -> np.ndarray:
    """Load all frames from video, resize to (TARGET_H, TARGET_W). Returns (N, H, W, 3) uint8 RGB."""
    cap = cv.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.resize(frame, (TARGET_W, TARGET_H), interpolation=cv.INTER_LINEAR)
        frames.append(frame)
    cap.release()
    return np.stack(frames)


def flow_to_color(flow: np.ndarray, max_mag: float = None) -> np.ndarray:
    """Convert (H, W, 2) flow to (H, W, 3) RGB visualization (HSV magnitude/angle)."""
    h, w, _ = flow.shape
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    ang = np.arctan2(flow[..., 1], flow[..., 0])
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = (ang * 180 / np.pi / 2 + 180) % 180
    hsv[..., 1] = 255
    if max_mag is None:
        max_mag = mag.max() + 1e-6
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    return cv.cvtColor(hsv, cv.COLOR_HSV2RGB)


def backward_warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Backward-warp img using forward flow.

    For each output pixel (x, y), look up img at (x + flow.u, y + flow.v).
    If flow is the correct frame_t -> frame_{t+K} flow, then
        backward_warp(frame_{t+K}, flow) ≈ frame_t.
    """
    H, W = img.shape[:2]
    grid_y, grid_x = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )
    sample_x = grid_x + flow[..., 0]
    sample_y = grid_y + flow[..., 1]
    return cv.remap(
        img,
        sample_x,
        sample_y,
        interpolation=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0,
    )


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    """Add a small text label on top of an image."""
    out = img.copy()
    cv.putText(out, text, (5, 18), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(out, text, (5, 18), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--flow-dir", type=str, required=True)
    parser.add_argument("--video-root", type=str, required=True)
    parser.add_argument("--output", type=str, default="verify_flow.png")
    parser.add_argument("--start-frame", type=int, default=50)
    parser.add_argument("--skips", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()

    video_path = Path(args.video)
    rel = video_path.relative_to(args.video_root).with_suffix("")
    flow_dir = Path(args.flow_dir) / rel

    print(f"Loading video: {video_path}")
    frames = load_video_frames(str(video_path))
    print(f"  → {len(frames)} frames")

    print(f"Loading flow: {flow_dir}")
    all_flow = np.load(flow_dir / "flow.npy", mmap_mode="r")  # (N-1, H, W, 2) float16
    all_mask = np.load(flow_dir / "mask.npy", mmap_mode="r")  # (N-1, H, W) bool
    print(f"  → flow {all_flow.shape}, mask {all_mask.shape}")

    rows = []
    for skip in args.skips:
        start = args.start_frame
        end = start + skip
        if end >= len(frames):
            print(f"Skip {skip}: not enough frames, skipping")
            continue

        # Load consecutive flows + masks for this skip
        flows = np.array(all_flow[start:end]).astype(np.float32)  # (K, H, W, 2)
        masks = np.array(all_mask[start:end])  # (K, H, W)

        # Naive sum (current buggy method)
        naive_flow = flows.sum(axis=0)  # (H, W, 2)

        # Iterative warping (correct method)
        warped_flow, warped_mask = compose_flow_and_mask(flows, masks)

        # Reconstruction: backward-warp frame_{t+K} with each flow → should match frame_t
        frame_t = frames[start]
        frame_tk = frames[end]
        recon_naive = backward_warp(frame_tk, naive_flow)
        recon_warped = backward_warp(frame_tk, warped_flow)

        # Per-pixel reconstruction error (only on warped mask region for fairness)
        err_naive = np.abs(recon_naive.astype(np.float32) - frame_t.astype(np.float32)).mean()
        err_warped = np.abs(recon_warped.astype(np.float32) - frame_t.astype(np.float32)).mean()

        # Use the same color scale for both flows for fair comparison
        max_mag = max(
            np.linalg.norm(naive_flow, axis=-1).max(),
            np.linalg.norm(warped_flow, axis=-1).max(),
        )
        naive_vis = flow_to_color(naive_flow, max_mag=max_mag)
        warped_vis = flow_to_color(warped_flow, max_mag=max_mag)

        # Diff between naive and warped flows (per-pixel L2)
        diff = np.linalg.norm(naive_flow - warped_flow, axis=-1)  # (H, W)
        diff_max = diff.max() + 1e-6
        diff_vis = (np.clip(diff / diff_max, 0, 1) * 255).astype(np.uint8)
        diff_vis = cv.applyColorMap(diff_vis, cv.COLORMAP_HOT)
        diff_vis = cv.cvtColor(diff_vis, cv.COLOR_BGR2RGB)

        # Build a row: 7 panels
        row = np.concatenate(
            [
                add_label(frame_t, f"frame_t (skip={skip})"),
                add_label(frame_tk, f"frame_t+{skip}"),
                add_label(naive_vis, "naive sum"),
                add_label(warped_vis, "warped"),
                add_label(diff_vis, f"|naive-warped| max={diff_max:.1f}"),
                add_label(recon_naive, f"recon naive (L1={err_naive:.1f})"),
                add_label(recon_warped, f"recon warped (L1={err_warped:.1f})"),
            ],
            axis=1,
        )
        rows.append(row)

        print(
            f"skip={skip}: L1 recon naive={err_naive:6.2f}  warped={err_warped:6.2f}"
            f"  |  flow diff: mean={diff.mean():.3f} max={diff.max():.3f}"
        )

    image = np.concatenate(rows, axis=0)
    cv.imwrite(args.output, cv.cvtColor(image, cv.COLOR_RGB2BGR))
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
