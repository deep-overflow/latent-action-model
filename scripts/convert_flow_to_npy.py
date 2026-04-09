"""
Convert compressed .npz flow files to per-video directories with uncompressed
flow.npy + mask.npy. The new format supports mmap-based slicing for ~500x
faster random-access loading during training.

Usage:
    python scripts/convert_flow_to_npy.py \
        --flow-dir /media/data1/chan/GR1_optical_flow \
        --delete-original
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-dir", type=str, required=True)
    parser.add_argument("--delete-original", action="store_true",
                        help="Delete original .npz after successful conversion")
    args = parser.parse_args()

    flow_dir = Path(args.flow_dir)
    npz_files = sorted(flow_dir.rglob("*.npz"))
    print(f"Found {len(npz_files)} .npz files")

    converted = 0
    skipped = 0
    for npz_path in tqdm(npz_files, desc="Converting"):
        out_dir = npz_path.with_suffix("")  # strip .npz → directory path
        flow_npy = out_dir / "flow.npy"
        mask_npy = out_dir / "mask.npy"

        if flow_npy.exists() and mask_npy.exists():
            skipped += 1
            if args.delete_original and npz_path.exists():
                npz_path.unlink()
            continue

        data = np.load(npz_path)
        flow = data["flow"]  # (N-1, H, W, 2) float16
        mask = data["mask"]  # (N-1, H, W) bool

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(flow_npy, flow)
        np.save(mask_npy, mask)
        converted += 1

        if args.delete_original:
            npz_path.unlink()

    print(f"Done. Converted {converted}, skipped {skipped}.")


if __name__ == "__main__":
    main()
