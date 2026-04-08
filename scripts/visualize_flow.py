"""
Visualize pre-computed optical flow as color images.

Usage:
    # Visualize a single .npy flow file (all frames)
    python scripts/visualize_flow.py \
        --flow-path /media/data1/chan/GR1_optical_flow/videos/chunk-000/observation.images.ego_view_freq20/episode_000502.npy \
        --output-dir outputs/flow_vis

    # Visualize with corresponding video side-by-side
    python scripts/visualize_flow.py \
        --flow-path /media/data1/chan/GR1_optical_flow/videos/chunk-000/observation.images.ego_view_freq20/episode_000502.npy \
        --video-path /media/data1/chan/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot/videos/chunk-000/observation.images.ego_view_freq20/episode_000502.mp4 \
        --output-dir outputs/flow_vis

    # Visualize specific frame range
    python scripts/visualize_flow.py \
        --flow-path /path/to/flow.npy \
        --output-dir outputs/flow_vis \
        --start 0 --end 10
"""

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np


def flow_to_color(flow: np.ndarray) -> np.ndarray:
    """Convert optical flow (H, W, 2) to RGB color visualization.

    Uses HSV color wheel: hue = flow direction, value = flow magnitude.
    """
    h, w, _ = flow.shape
    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    ang = np.arctan2(flow[..., 1], flow[..., 0])

    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 0] = ((ang * 180 / np.pi / 2) + 180) % 180
    hsv[..., 1] = 255
    max_mag = mag.max() + 1e-6
    hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)

    return cv.cvtColor(hsv, cv.COLOR_HSV2RGB)


def load_video_frames(video_path: str, target_h: int = 240, target_w: int = 320) -> np.ndarray:
    """Load all frames from video, resize to target resolution."""
    cap = cv.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = cv.resize(frame, (target_w, target_h))
        frames.append(frame)
    cap.release()
    return np.stack(frames) if frames else np.array([])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-path", type=str, required=True)
    parser.add_argument("--video-path", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="outputs/flow_vis")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    parser.add_argument("--save-video", action="store_true", help="Save as mp4 video")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load flow
    flows = np.load(args.flow_path)  # (N-1, H, W, 2)
    end = args.end if args.end > 0 else len(flows)
    flows = flows[args.start:end]
    print(f"Flow shape: {flows.shape}, range: [{flows.min():.2f}, {flows.max():.2f}]")

    # Load video if provided
    video_frames = None
    if args.video_path:
        video_frames = load_video_frames(args.video_path)
        print(f"Video frames: {video_frames.shape}")

    # Visualize
    stem = Path(args.flow_path).stem
    vis_frames = []

    for i, flow in enumerate(flows):
        flow_color = flow_to_color(flow.astype(np.float32))  # (H, W, 3)

        if video_frames is not None:
            frame_idx = args.start + i
            if frame_idx + 1 < len(video_frames):
                frame0 = video_frames[frame_idx]
                frame1 = video_frames[frame_idx + 1]
                # Side-by-side: frame0 | frame1 | flow
                vis = np.concatenate([frame0, frame1, flow_color], axis=1)
            else:
                vis = flow_color
        else:
            vis = flow_color

        vis_frames.append(vis)

        if not args.save_video:
            out_path = output_dir / f"{stem}_frame{args.start + i:04d}.png"
            cv.imwrite(str(out_path), cv.cvtColor(vis, cv.COLOR_RGB2BGR))

    if args.save_video:
        h, w = vis_frames[0].shape[:2]
        video_out = output_dir / f"{stem}.mp4"
        writer = cv.VideoWriter(str(video_out), cv.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
        for frame in vis_frames:
            writer.write(cv.cvtColor(frame, cv.COLOR_RGB2BGR))
        writer.release()
        print(f"Saved video: {video_out}")
    else:
        print(f"Saved {len(vis_frames)} images to {output_dir}/")


if __name__ == "__main__":
    main()
