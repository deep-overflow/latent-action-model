"""Utilities for composing optical flow over multiple frames."""

import cv2 as cv
import numpy as np


def compose_flow_and_mask(flows: np.ndarray, masks: np.ndarray):
    """Compose K consecutive optical flows into a single flow from frame 0 to frame K,
    using iterative warping.

    Given:
        flows[k] is the flow from frame k to frame k+1, in float32 (H, W, 2).
        masks[k] is the cycle-consistency mask of flows[k], bool (H, W).

    The composition rule is:
        F_{0->k+1}(x, y) = F_{0->k}(x, y) + F_{k->k+1}(x + F_{0->k}.u, y + F_{0->k}.v)

    The mask is composed by warping each step's mask along the trajectory and
    AND-ing them together — a pixel is reliable iff its corresponding pixel at
    every intermediate step is also reliable.

    Args:
        flows: (K, H, W, 2) array of consecutive flows. Float32 or float16.
        masks: (K, H, W) bool array of consecutive masks.
    Returns:
        composed_flow: (H, W, 2) float32
        composed_mask: (H, W) bool
    """
    K, H, W, _ = flows.shape
    flows = flows.astype(np.float32)

    if K == 1:
        return flows[0].copy(), masks[0].copy()

    grid_y, grid_x = np.meshgrid(
        np.arange(H, dtype=np.float32),
        np.arange(W, dtype=np.float32),
        indexing="ij",
    )

    accumulated_flow = flows[0].copy()  # (H, W, 2)
    accumulated_mask = masks[0].copy()  # (H, W)

    for k in range(1, K):
        sample_x = grid_x + accumulated_flow[..., 0]  # (H, W)
        sample_y = grid_y + accumulated_flow[..., 1]  # (H, W)

        # Sample the next-step flow at the warped positions (bilinear)
        sampled_flow = cv.remap(
            flows[k],
            sample_x,
            sample_y,
            interpolation=cv.INTER_LINEAR,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=0,
        )  # (H, W, 2)

        # Sample the next-step mask at the warped positions (nearest)
        sampled_mask = cv.remap(
            masks[k].astype(np.uint8),
            sample_x,
            sample_y,
            interpolation=cv.INTER_NEAREST,
            borderMode=cv.BORDER_CONSTANT,
            borderValue=0,
        ).astype(bool)  # (H, W)

        accumulated_flow = accumulated_flow + sampled_flow
        accumulated_mask = accumulated_mask & sampled_mask

    return accumulated_flow, accumulated_mask
