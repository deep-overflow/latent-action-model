"""
Pre-compute optical flow for GR1 robot videos using SEA-RAFT.

Computes forward + backward flow between all consecutive frame pairs,
applies cycle consistency check, and saves flow + validity mask as .npz files.

Usage:
    python scripts/precompute_flow.py \
        --video-dir /media/data1/chan/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot \
        --output-dir /media/data1/chan/GR1_optical_flow \
        --batch-size 8 \
        --cycle-threshold 1.0
"""

import argparse
import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add SEA-RAFT to path
SEA_RAFT_DIR = Path(__file__).resolve().parents[1].parent / "SEA-RAFT"
sys.path.insert(0, str(SEA_RAFT_DIR))
sys.path.insert(0, str(SEA_RAFT_DIR / "core"))

from raft import RAFT
from utils.utils import load_ckpt

TARGET_H, TARGET_W = 240, 320


def load_sea_raft_args(cfg_path: str) -> argparse.Namespace:
    """Load SEA-RAFT config from JSON file."""
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["scale"] = 0  # No scaling for 240x320 input
    return argparse.Namespace(**cfg)


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


def warp_flow(flow_to_warp: torch.Tensor, flow_for_warp: torch.Tensor) -> torch.Tensor:
    """Warp flow_to_warp using flow_for_warp.
    Args:
        flow_to_warp: (B, 2, H, W) — the flow to be warped (backward flow)
        flow_for_warp: (B, 2, H, W) — the flow used as warp field (forward flow)
    Returns:
        warped: (B, 2, H, W)
    """
    B, _, H, W = flow_for_warp.shape
    # Create base grid  # (H, W, 2)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(H, device=flow_for_warp.device, dtype=torch.float32),
        torch.arange(W, device=flow_for_warp.device, dtype=torch.float32),
        indexing='ij'
    )
    # Apply forward flow to get sampling locations
    sample_x = grid_x.unsqueeze(0) + flow_for_warp[:, 0]  # (B, H, W)
    sample_y = grid_y.unsqueeze(0) + flow_for_warp[:, 1]  # (B, H, W)
    # Normalize to [-1, 1] for grid_sample
    sample_x = 2.0 * sample_x / (W - 1) - 1.0
    sample_y = 2.0 * sample_y / (H - 1) - 1.0
    grid = torch.stack([sample_x, sample_y], dim=-1)  # (B, H, W, 2)
    return F.grid_sample(flow_to_warp, grid, mode='bilinear', padding_mode='zeros', align_corners=True)


@torch.no_grad()
def compute_flow_batch(
    model: torch.nn.Module,
    sea_raft_args: argparse.Namespace,
    frames: np.ndarray,
    device: torch.device,
    batch_size: int = 8,
    cycle_threshold: float = 1.0,
):
    """Compute optical flow with cycle consistency check.
    Args:
        frames: (N, H, W, 3) uint8 array
        cycle_threshold: max cycle error (pixels) to consider reliable
    Returns:
        flows: (N-1, H, W, 2) float16 array
        masks: (N-1, H, W) bool array — True = reliable
    """
    N = len(frames)
    if N < 2:
        return np.array([]), np.array([])

    frames_t = torch.from_numpy(frames).permute(0, 3, 1, 2).float().to(device)  # (N, 3, H, W)

    all_flows = []
    all_masks = []
    for i in range(0, N - 1, batch_size):
        end = min(i + batch_size, N - 1)
        img1 = frames_t[i:end]          # (B, 3, H, W)
        img2 = frames_t[i + 1:end + 1]  # (B, 3, H, W)

        # Forward flow (A → B)
        fwd = model(img1, img2, iters=sea_raft_args.iters, test_mode=True)['flow'][-1]  # (B, 2, H, W)
        # Backward flow (B → A)
        bwd = model(img2, img1, iters=sea_raft_args.iters, test_mode=True)['flow'][-1]  # (B, 2, H, W)

        # Cycle consistency: warp backward flow using forward flow
        bwd_warped = warp_flow(bwd, fwd)  # (B, 2, H, W)
        cycle_error = torch.norm(fwd + bwd_warped, p=2, dim=1)  # (B, H, W)
        mask = cycle_error < cycle_threshold  # (B, H, W)

        all_flows.append(fwd.cpu().permute(0, 2, 3, 1).numpy())  # (B, H, W, 2)
        all_masks.append(mask.cpu().numpy())  # (B, H, W)

    flows = np.concatenate(all_flows, axis=0).astype(np.float16)  # (N-1, H, W, 2)
    masks = np.concatenate(all_masks, axis=0)  # (N-1, H, W)
    return flows, masks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--cfg", type=str, default=str(SEA_RAFT_DIR / "config/eval/spring-M.json"))
    parser.add_argument("--url", type=str, default="MemorySlices/Tartan-C-T-TSKH-spring540x960-M")
    parser.add_argument("--cycle-threshold", type=float, default=1.0,
                        help="Cycle consistency threshold in pixels")
    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    device = torch.device(args.device)

    # Load SEA-RAFT model
    sea_raft_args = load_sea_raft_args(args.cfg)
    model = RAFT.from_pretrained(args.url, args=sea_raft_args)
    model = model.to(device).eval()
    print(f"Loaded SEA-RAFT from {args.url}")
    print(f"Cycle consistency threshold: {args.cycle_threshold} px")

    # Find all mp4 files
    video_paths = sorted(video_dir.rglob("*.mp4"))
    print(f"Found {len(video_paths)} videos")

    skipped = 0
    for video_path in tqdm(video_paths, desc="Computing flow"):
        rel_path = video_path.relative_to(video_dir)
        flow_path = output_dir / rel_path.with_suffix(".npz")

        if flow_path.exists():
            skipped += 1
            continue

        frames = load_video_frames(str(video_path))
        if len(frames) < 2:
            print(f"Skipping {video_path}: < 2 frames")
            continue

        flows, masks = compute_flow_batch(
            model, sea_raft_args, frames, device, args.batch_size, args.cycle_threshold
        )

        flow_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(flow_path, flow=flows, mask=masks)

    print(f"Done. Skipped {skipped} existing files.")


if __name__ == "__main__":
    main()
