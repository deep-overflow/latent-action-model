"""
Pre-compute optical flow for GR1 robot videos using RAFT.

Computes flow between all consecutive frame pairs and saves as .npy files.
Flow is computed at 240x320 resolution (matching LAM training resolution).

Usage:
    python scripts/precompute_flow.py \
        --video-dir /media/data1/chan/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot \
        --output-dir /media/data1/chan/GR1_optical_flow \
        --batch-size 8
"""

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from torchvision.transforms.functional import resize
from tqdm import tqdm


TARGET_H, TARGET_W = 240, 320


def load_video_frames(video_path: str) -> np.ndarray:
    """Load all frames from video, resize to target resolution.
    Returns: (N, H, W, 3) uint8 array in RGB order.
    """
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
    if len(frames) == 0:
        return np.array([])
    return np.stack(frames)  # (N, H, W, 3)


@torch.no_grad()
def compute_flow_batch(
    model: torch.nn.Module,
    frames: np.ndarray,
    device: torch.device,
    batch_size: int = 8,
) -> np.ndarray:
    """Compute optical flow for all consecutive frame pairs.
    Args:
        frames: (N, H, W, 3) uint8 array
    Returns:
        flows: (N-1, H, W, 2) float16 array
    """
    N = len(frames)
    if N < 2:
        return np.array([])

    # Convert to float tensor, (N, 3, H, W), [0, 255] range (RAFT expects this)
    frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).float().to(device)  # (N, 3, H, W)

    all_flows = []
    for i in range(0, N - 1, batch_size):
        end = min(i + batch_size, N - 1)
        img1 = frames_t[i:end]      # (B, 3, H, W)
        img2 = frames_t[i + 1:end + 1]  # (B, 3, H, W)
        flow_preds = model(img1, img2)  # list of flow predictions at different iterations
        flow = flow_preds[-1]  # (B, 2, H, W) — last iteration is most refined
        all_flows.append(flow.cpu().permute(0, 2, 3, 1).numpy())  # (B, H, W, 2)

    return np.concatenate(all_flows, axis=0).astype(np.float16)  # (N-1, H, W, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    device = torch.device(args.device)

    # Load RAFT model
    weights = Raft_Large_Weights.DEFAULT
    model = raft_large(weights=weights).to(device).eval()

    # Find all mp4 files
    video_paths = sorted(video_dir.rglob("*.mp4"))
    print(f"Found {len(video_paths)} videos")

    skipped = 0
    for video_path in tqdm(video_paths, desc="Computing flow"):
        # Determine output path (mirror directory structure)
        rel_path = video_path.relative_to(video_dir)
        flow_path = output_dir / rel_path.with_suffix(".npy")

        if flow_path.exists():
            skipped += 1
            continue

        # Load frames
        frames = load_video_frames(str(video_path))
        if len(frames) < 2:
            print(f"Skipping {video_path}: < 2 frames")
            continue

        # Compute flow
        flows = compute_flow_batch(model, frames, device, args.batch_size)

        # Save
        flow_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(flow_path, flows)

    print(f"Done. Skipped {skipped} existing files.")


if __name__ == "__main__":
    main()
