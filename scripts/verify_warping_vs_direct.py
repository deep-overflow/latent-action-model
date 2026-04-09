"""Verify iterative warping is correct by comparing against SEA-RAFT direct flow.

For frames f_t and f_{t+K}, the iterative warping of consecutive flows should
produce a flow field very close to running SEA-RAFT directly on (f_t, f_{t+K}).
The naive sum should differ.

This tests the algorithm in isolation from rendering / visualization.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2 as cv
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lam.flow_utils import compose_flow_and_mask

SEA_RAFT_DIR = Path(__file__).resolve().parents[2] / "SEA-RAFT"
sys.path.insert(0, str(SEA_RAFT_DIR))
sys.path.insert(0, str(SEA_RAFT_DIR / "core"))
from raft import RAFT

TARGET_H, TARGET_W = 240, 320


def load_video_frames(video_path: str) -> np.ndarray:
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--flow-dir", type=str, required=True)
    parser.add_argument("--video-root", type=str, required=True)
    parser.add_argument("--start-frame", type=int, default=140)
    parser.add_argument("--skip", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    # Load frames
    frames = load_video_frames(args.video)
    print(f"Frames: {len(frames)}")

    # Load precomputed flows
    rel = Path(args.video).relative_to(args.video_root).with_suffix("")
    flow_dir = Path(args.flow_dir) / rel
    all_flow = np.load(flow_dir / "flow.npy", mmap_mode="r")
    all_mask = np.load(flow_dir / "mask.npy", mmap_mode="r")

    K = args.skip
    s = args.start_frame
    flows_kk = np.array(all_flow[s:s + K]).astype(np.float32)  # (K, H, W, 2)
    masks_kk = np.array(all_mask[s:s + K])

    # Method 1: naive sum
    naive_flow = flows_kk.sum(axis=0)

    # Method 2: iterative warping
    warped_flow, _ = compose_flow_and_mask(flows_kk, masks_kk)

    # Method 3 (ground truth): SEA-RAFT directly on (f_t, f_{t+K})
    cfg = json.load(open(SEA_RAFT_DIR / "config/eval/spring-M.json"))
    cfg["scale"] = 0
    sea_args = argparse.Namespace(**cfg)

    device = torch.device(args.device)
    model = RAFT.from_pretrained("MemorySlices/Tartan-C-T-TSKH-spring540x960-M", args=sea_args)
    model = model.to(device).eval()

    f_t = torch.from_numpy(frames[s]).permute(2, 0, 1).float().unsqueeze(0).to(device)
    f_tk = torch.from_numpy(frames[s + K]).permute(2, 0, 1).float().unsqueeze(0).to(device)

    with torch.no_grad():
        direct_flow = model(f_t, f_tk, iters=sea_args.iters, test_mode=True)["flow"][-1]
    direct_flow = direct_flow[0].permute(1, 2, 0).cpu().numpy()  # (H, W, 2)

    # Compare each method against the direct ground truth
    def stats(label, flow):
        diff = flow - direct_flow
        l1 = np.abs(diff).mean()
        l2 = np.linalg.norm(diff, axis=-1).mean()
        max_l2 = np.linalg.norm(diff, axis=-1).max()
        epe = np.linalg.norm(diff, axis=-1)
        print(f"  {label:8s}  vs direct:  EPE mean={l2:.3f}  max={max_l2:.2f}  (L1={l1:.3f})")

    print(f"\n=== Comparison vs SEA-RAFT direct flow (start={s}, skip={K}) ===")
    print(f"  direct flow magnitude: mean={np.linalg.norm(direct_flow, axis=-1).mean():.2f}")
    print(f"  naive  flow magnitude: mean={np.linalg.norm(naive_flow, axis=-1).mean():.2f}")
    print(f"  warped flow magnitude: mean={np.linalg.norm(warped_flow, axis=-1).mean():.2f}")
    print()
    stats("naive", naive_flow)
    stats("warped", warped_flow)


if __name__ == "__main__":
    main()
