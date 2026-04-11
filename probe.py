"""
LAM Linear Probing Experiment.

Validates whether LAM's 32-dim latent actions encode meaningful robot action information
by training a linear probe to regress from LAM latents to ground-truth delta actions.

Usage:
    # Step 1: Extract features
    python external/lam_project/probe.py extract \
        --dataset-path /media/data1/chan/PhysicalAI-Robotics-GR00T-Teleop-GR1/GR1_robot \
        --ckpt-path checkpoints/DreamDojo/LAM_400k.ckpt \
        --max-episodes 500 \
        --output-dir outputs/lam_probe

    # Step 2: Train linear probe
    python external/lam_project/probe.py train \
        --feature-dir outputs/lam_probe \
        --epochs 100 \
        --lr 1e-3

    # Step 3: Evaluate
    python external/lam_project/probe.py eval \
        --feature-dir outputs/lam_probe
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2 as cv
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from lam.model import LAM

# GR-1 action indices (from shared_meta/GR1_unified_modality.json)
# Raw "action" column: 44 dims
# We extract 29 dims in the same order as groot_configs.py:
#   left_arm[0:7], right_arm[22:29], left_hand[7:13], right_hand[29:35], waist[41:44]
ACTION_SLICES = [
    ("left_arm", 0, 7),
    ("right_arm", 22, 29),
    ("left_hand", 7, 13),
    ("right_hand", 29, 35),
    ("waist", 41, 44),
]
ACTION_DIM = sum(end - start for _, start, end in ACTION_SLICES)  # 29

JOINT_GROUPS = {
    "left_arm": (0, 7),
    "right_arm": (7, 14),
    "left_hand": (14, 20),
    "right_hand": (20, 26),
    "waist": (26, 29),
}


def extract_action_subset(raw_action: np.ndarray) -> np.ndarray:
    """Extract 29-dim action subset from raw 44-dim action. (T, 44) -> (T, 29)"""
    parts = []
    for _, start, end in ACTION_SLICES:
        parts.append(raw_action[:, start:end])
    return np.concatenate(parts, axis=1)  # (T, 29)


def load_stats(stats_path: str) -> dict:
    """Load min/max stats for action normalization."""
    with open(stats_path, "r") as f:
        stats = json.load(f)
    action_stats = stats["action"]
    raw_min = np.array(action_stats["min"], dtype=np.float32)  # (44,)
    raw_max = np.array(action_stats["max"], dtype=np.float32)  # (44,)
    # Extract the same 29-dim subset
    parts_min, parts_max = [], []
    for _, start, end in ACTION_SLICES:
        parts_min.append(raw_min[start:end])
        parts_max.append(raw_max[start:end])
    return {
        "min": np.concatenate(parts_min),  # (29,)
        "max": np.concatenate(parts_max),  # (29,)
    }


def normalize_action(action: np.ndarray, stats: dict) -> np.ndarray:
    """Min-max normalize action to [-1, 1]. (T, 29) -> (T, 29)"""
    a_min = stats["min"]
    a_max = stats["max"]
    mask = a_min != a_max
    normalized = np.zeros_like(action)
    normalized[:, mask] = 2 * (action[:, mask] - a_min[mask]) / (a_max[mask] - a_min[mask]) - 1
    normalized = np.clip(normalized, -1, 1)
    return normalized


def load_video_frames(video_path: str, frame_indices: np.ndarray) -> np.ndarray:
    """Load specific frames from a video file. Returns (N, H, W, 3) float32 in [0, 1]."""
    cap = cv.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    for idx in frame_indices:
        cap.set(cv.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            # Repeat last frame if we run past the end
            if frames:
                frames.append(frames[-1])
            else:
                raise RuntimeError(f"Cannot read frame {idx} from {video_path}")
        else:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    return np.stack(frames).astype(np.float32) / 255.0  # (N, H, W, 3)


def preprocess_for_lam(frames: np.ndarray) -> torch.Tensor:
    """Resize frames to 240x320 for LAM input. (N, H, W, 3) -> (N, 240, 320, 3)"""
    # (N, H, W, C) -> (N, C, H, W) for F.interpolate
    t = torch.from_numpy(frames).permute(0, 3, 1, 2)  # (N, 3, H, W)
    t = F.interpolate(t, size=(240, 320), mode="bilinear", align_corners=False)
    t = t.permute(0, 2, 3, 1)  # (N, 240, 320, 3)
    return t


def find_video_path(dataset_path: Path, episode_id: int, chunk_size: int = 1000) -> Path:
    """Find the ego_view video path for a GR-1 episode."""
    chunk = episode_id // chunk_size
    # Try common patterns
    patterns = [
        f"videos/chunk-{chunk:03d}/observation.images.ego_view_freq20/episode_{episode_id:06d}.mp4",
        f"videos/chunk-{chunk:03d}/ego_view_freq20/episode_{episode_id:06d}.mp4",
    ]
    for p in patterns:
        full = dataset_path / p
        if full.exists():
            return full
    raise FileNotFoundError(f"No video found for episode {episode_id} in {dataset_path}")


def extract_features(args):
    """Phase 1: Extract LAM features and delta actions from the dataset."""
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load stats for normalization
    stats_path = Path(args.stats_path) if args.stats_path else dataset_path / "meta" / "stats.json"
    stats = load_stats(str(stats_path))

    # Load episode metadata
    episode_path = dataset_path / "meta" / "episodes.jsonl"
    with open(episode_path, "r") as f:
        episodes = [json.loads(line) for line in f]
    print(f"Found {len(episodes)} episodes")

    # Limit episodes
    if args.max_episodes > 0:
        episodes = episodes[: args.max_episodes]
    print(f"Using {len(episodes)} episodes")

    # Detect chunk size
    info_path = dataset_path / "meta" / "info.json"
    chunk_size = 1000
    if info_path.exists():
        with open(info_path, "r") as f:
            info = json.load(f)
        chunk_size = info.get("chunks_size", info.get("chunk_size", 1000))

    # Load LAM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading LAM from {args.ckpt_path}...")
    lam = LAM.load_from_checkpoint(args.ckpt_path, map_location=device)
    lam = lam.to(device).eval()

    all_z_reps = []
    all_delta_actions = []
    frame_stride = args.frame_stride  # Sample every N-th frame pair to reduce data volume

    for ep in tqdm(episodes, desc="Extracting features"):
        ep_id = ep["episode_index"]
        ep_len = ep["length"]

        if ep_len < 3:
            continue

        # Load parquet
        chunk = ep_id // chunk_size
        parquet_path = dataset_path / f"data/chunk-{chunk:03d}/episode_{ep_id:06d}.parquet"
        if not parquet_path.exists():
            print(f"  Skipping episode {ep_id}: parquet not found")
            continue
        df = pd.read_parquet(parquet_path)

        # Extract raw actions (T, 44)
        raw_actions = np.stack(df["action"].values)  # (T, 44)
        actions_29 = extract_action_subset(raw_actions)  # (T, 29)
        norm_actions = normalize_action(actions_29, stats)  # (T, 29)

        # Compute single-step delta actions: delta[t] = norm_action[t+1] - norm_action[t]
        delta_actions = norm_actions[1:] - norm_actions[:-1]  # (T-1, 29)

        # Find video
        try:
            video_path = find_video_path(dataset_path, ep_id, chunk_size)
        except FileNotFoundError as e:
            print(f"  Skipping episode {ep_id}: {e}")
            continue

        # Load video frames. Use timestamps from parquet to align frames.
        timestamps = df["timestamp"].values
        if np.all(timestamps == 0):
            # Fallback: assume 20 Hz
            timestamps = np.arange(len(df)) / 20.0

        # Sample frame pairs with stride
        pair_indices = list(range(0, ep_len - 1, frame_stride))
        if not pair_indices:
            continue

        # Load all needed frames
        needed_frames = sorted(set(
            idx for i in pair_indices for idx in [i, i + 1]
        ))
        try:
            loaded_frames = load_video_frames(str(video_path), np.array(needed_frames))
        except RuntimeError as e:
            print(f"  Skipping episode {ep_id}: {e}")
            continue
        frame_map = {idx: j for j, idx in enumerate(needed_frames)}

        # Preprocess all frames for LAM
        lam_frames = preprocess_for_lam(loaded_frames)  # (N, 240, 320, 3)

        # Build frame pairs and extract LAM features in batches
        batch_pairs = []
        batch_deltas = []
        for i in pair_indices:
            f0 = lam_frames[frame_map[i]]      # (240, 320, 3)
            f1 = lam_frames[frame_map[i + 1]]  # (240, 320, 3)
            pair = torch.stack([f0, f1], dim=0)  # (2, 240, 320, 3)
            batch_pairs.append(pair)
            batch_deltas.append(delta_actions[i])  # (29,)

        if not batch_pairs:
            continue

        # Process in batches through LAM
        batch_pairs_t = torch.stack(batch_pairs)  # (N_pairs, 2, 240, 320, 3)
        batch_size = args.batch_size
        z_reps = []
        with torch.no_grad():
            for start in range(0, len(batch_pairs_t), batch_size):
                batch = batch_pairs_t[start : start + batch_size].to(device)
                outputs = lam.lam.encode(batch)
                z = outputs["z_rep"].squeeze().cpu()  # (B, 32) or (32,) if B=1
                if z.dim() == 1:
                    z = z.unsqueeze(0)
                z_reps.append(z)

        z_reps = torch.cat(z_reps, dim=0)  # (N_pairs, 32)
        batch_deltas = np.stack(batch_deltas)  # (N_pairs, 29)

        all_z_reps.append(z_reps)
        all_delta_actions.append(torch.from_numpy(batch_deltas).float())

    # Concatenate all
    all_z_reps = torch.cat(all_z_reps, dim=0)  # (N_total, 32)
    all_delta_actions = torch.cat(all_delta_actions, dim=0)  # (N_total, 29)

    print(f"\nExtracted {len(all_z_reps)} samples")
    print(f"  z_rep shape: {all_z_reps.shape}")
    print(f"  delta_action shape: {all_delta_actions.shape}")

    # Split train/val (80/20)
    n = len(all_z_reps)
    perm = torch.randperm(n)
    split = int(0.8 * n)
    train_idx, val_idx = perm[:split], perm[split:]

    torch.save({
        "z_rep_train": all_z_reps[train_idx],
        "z_rep_val": all_z_reps[val_idx],
        "delta_action_train": all_delta_actions[train_idx],
        "delta_action_val": all_delta_actions[val_idx],
    }, output_dir / "features.pt")

    # Save config
    config = {
        "dataset_path": str(args.dataset_path),
        "ckpt_path": str(args.ckpt_path),
        "max_episodes": args.max_episodes,
        "frame_stride": args.frame_stride,
        "batch_size": args.batch_size,
        "n_total": n,
        "n_train": split,
        "n_val": n - split,
        "z_rep_dim": 32,
        "action_dim": ACTION_DIM,
        "joint_groups": {k: list(v) for k, v in JOINT_GROUPS.items()},
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"Saved features to {output_dir / 'features.pt'}")
    print(f"  Train: {split}, Val: {n - split}")


def train_probe(args):
    """Phase 2: Train linear probe on extracted features."""
    feature_dir = Path(args.feature_dir)
    data = torch.load(feature_dir / "features.pt", weights_only=True)
    config = json.load(open(feature_dir / "config.json"))

    z_train = data["z_rep_train"]        # (N_train, 32)
    a_train = data["delta_action_train"]  # (N_train, 29)
    z_val = data["z_rep_val"]            # (N_val, 32)
    a_val = data["delta_action_val"]     # (N_val, 29)

    # Standardize z_rep using train statistics, save for eval consistency
    z_mean = z_train.mean(dim=0)
    z_std = z_train.std(dim=0).clamp(min=1e-6)
    torch.save({"mean": z_mean, "std": z_std}, feature_dir / "z_stats.pt")
    z_train = (z_train - z_mean) / z_std
    z_val = (z_val - z_mean) / z_std

    print(f"Train: {len(z_train)}, Val: {len(z_val)}  (z standardized)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Linear probe
    probe = nn.Linear(config["z_rep_dim"], config["action_dim"]).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    train_dataset = TensorDataset(z_train, a_train)
    val_dataset = TensorDataset(z_val, a_val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    best_val_loss = float("inf")
    best_epoch = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(args.epochs):
        # Train
        probe.train()
        train_losses = []
        for z_batch, a_batch in train_loader:
            z_batch, a_batch = z_batch.to(device), a_batch.to(device)
            pred = probe(z_batch)
            loss = criterion(pred, a_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        probe.eval()
        val_losses = []
        with torch.no_grad():
            for z_batch, a_batch in val_loader:
                z_batch, a_batch = z_batch.to(device), a_batch.to(device)
                pred = probe(z_batch)
                loss = criterion(pred, a_batch)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(probe.state_dict(), feature_dir / "probe_best.pt")

        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:4d}/{args.epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}  best={best_val_loss:.6f}@{best_epoch+1}")

        # Early stopping
        if args.patience > 0 and (epoch - best_epoch) >= args.patience:
            print(f"Early stop at epoch {epoch+1} (no improvement for {args.patience} epochs)")
            break

    # Save training history
    with open(feature_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Plot training curve
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Linear Probe Training")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(feature_dir / "training_curve.png", dpi=150)
    plt.close()

    print(f"\nBest val loss: {best_val_loss:.6f}")
    print(f"Saved best probe to {feature_dir / 'probe_best.pt'}")


def evaluate_probe(args):
    """Phase 3: Evaluate probe with MSE and per-joint R²."""
    feature_dir = Path(args.feature_dir)
    data = torch.load(feature_dir / "features.pt", weights_only=True)
    config = json.load(open(feature_dir / "config.json"))

    z_val = data["z_rep_val"]          # (N_val, 32)
    a_val = data["delta_action_val"]   # (N_val, 29)

    # Apply same standardization as training (if stats available)
    z_stats_path = feature_dir / "z_stats.pt"
    if z_stats_path.exists():
        z_stats = torch.load(z_stats_path, weights_only=True)
        z_val = (z_val - z_stats["mean"]) / z_stats["std"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load best probe
    probe = nn.Linear(config["z_rep_dim"], config["action_dim"]).to(device)
    probe.load_state_dict(torch.load(feature_dir / "probe_best.pt", weights_only=True))
    probe.eval()

    # Predict
    with torch.no_grad():
        pred = probe(z_val.to(device)).cpu()  # (N_val, 29)

    # Overall MSE
    mse = ((pred - a_val) ** 2).mean().item()

    # Per-joint R² — mark zero-variance dims as NaN (constant joints)
    ss_res = ((a_val - pred) ** 2).sum(dim=0)   # (29,)
    a_var = a_val.var(dim=0)                    # (29,)
    ss_tot = ((a_val - a_val.mean(dim=0)) ** 2).sum(dim=0)  # (29,)
    valid_mask = a_var > 1e-12
    r2_per_dim = np.full(len(ss_res), np.nan, dtype=np.float32)
    r2_per_dim[valid_mask.numpy()] = (1 - ss_res[valid_mask] / ss_tot[valid_mask]).numpy()

    # Group R² by joint group (only valid dims)
    group_r2 = {}
    for name, (start, end) in JOINT_GROUPS.items():
        m = valid_mask[start:end]
        if m.any():
            group_ss_res = ss_res[start:end][m].sum().item()
            group_ss_tot = ss_tot[start:end][m].sum().item()
            group_r2[name] = 1 - group_ss_res / group_ss_tot
        else:
            group_r2[name] = float("nan")

    # Overall R² — only over valid dims
    overall_r2 = 1 - ss_res[valid_mask].sum().item() / ss_tot[valid_mask].sum().item()

    # Print results
    print(f"\n{'='*50}")
    print(f"LAM Linear Probe Results")
    print(f"{'='*50}")
    print(f"Overall MSE:  {mse:.6f}")
    print(f"Overall R²:   {overall_r2:.4f}")
    print(f"\nPer-group R²:")
    for name, r2 in group_r2.items():
        print(f"  {name:12s}: {r2:.4f}")
    print(f"\nPer-joint R² (all 29 dims):")
    joint_names = []
    for name, (start, end) in JOINT_GROUPS.items():
        for i in range(end - start):
            joint_names.append(f"{name}_{i}")
    for i, (name, r2) in enumerate(zip(joint_names, r2_per_dim)):
        print(f"  {name:18s}: {r2:.4f}")

    # Save metrics
    metrics = {
        "mse": mse,
        "overall_r2": overall_r2,
        "group_r2": group_r2,
        "per_joint_r2": {name: float(r2) for name, r2 in zip(joint_names, r2_per_dim)},
    }
    with open(feature_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Summary
    with open(feature_dir / "summary.txt", "w") as f:
        f.write("LAM Linear Probe Results\n")
        f.write(f"{'='*50}\n")
        f.write(f"Overall MSE:  {mse:.6f}\n")
        f.write(f"Overall R²:   {overall_r2:.4f}\n\n")
        f.write("Per-group R²:\n")
        for name, r2 in group_r2.items():
            f.write(f"  {name:12s}: {r2:.4f}\n")

    # Plot per-joint R² bar chart
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = []
    group_color_map = {
        "left_arm": "#2196F3",
        "right_arm": "#4CAF50",
        "left_hand": "#FF9800",
        "right_hand": "#F44336",
        "waist": "#9C27B0",
    }
    for name, (start, end) in JOINT_GROUPS.items():
        colors.extend([group_color_map[name]] * (end - start))

    r2_plot = np.where(np.isnan(r2_per_dim), 0, r2_per_dim)
    bars = ax.bar(range(len(r2_plot)), r2_plot, color=colors)
    ax.set_xticks(range(len(r2_plot)))
    ax.set_xticklabels(joint_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("R²")
    ax.set_title(f"LAM Linear Probe: Per-Joint R² (Overall R²={overall_r2:.4f})")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylim(min(np.nanmin(r2_per_dim) - 0.1, -0.2), 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=n) for n, c in group_color_map.items()]
    ax.legend(handles=legend_elements, loc="upper right")

    plt.tight_layout()
    plt.savefig(feature_dir / "r2_per_joint.png", dpi=150)
    plt.close()

    print(f"\nSaved metrics to {feature_dir / 'metrics.json'}")
    print(f"Saved plot to {feature_dir / 'r2_per_joint.png'}")


def main():
    parser = argparse.ArgumentParser(description="LAM Linear Probing Experiment")
    subparsers = parser.add_subparsers(dest="command")

    # Extract
    p_extract = subparsers.add_parser("extract", help="Extract LAM features from dataset")
    p_extract.add_argument("--dataset-path", type=str, required=True)
    p_extract.add_argument("--ckpt-path", type=str, default="checkpoints/DreamDojo/LAM_400k.ckpt")
    p_extract.add_argument("--max-episodes", type=int, default=500)
    p_extract.add_argument("--frame-stride", type=int, default=4,
                           help="Sample every N-th frame pair per episode")
    p_extract.add_argument("--batch-size", type=int, default=64)
    p_extract.add_argument("--output-dir", type=str, default="outputs/lam_probe")
    p_extract.add_argument("--stats-path", type=str, default=None,
                           help="Path to stats JSON (default: <dataset_path>/meta/stats.json)")

    # Train
    p_train = subparsers.add_parser("train", help="Train linear probe")
    p_train.add_argument("--feature-dir", type=str, default="outputs/lam_probe")
    p_train.add_argument("--epochs", type=int, default=500)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--batch-size", type=int, default=512)
    p_train.add_argument("--patience", type=int, default=50,
                         help="Early stop if val loss does not improve for N epochs (0 to disable)")

    # Eval
    p_eval = subparsers.add_parser("eval", help="Evaluate probe")
    p_eval.add_argument("--feature-dir", type=str, default="outputs/lam_probe")

    args = parser.parse_args()

    if args.command == "extract":
        extract_features(args)
    elif args.command == "train":
        train_probe(args)
    elif args.command == "eval":
        evaluate_probe(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
