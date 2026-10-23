"""roi_localize.py - Anatomical ROI joint-crop and compartment partitioner.

Designed for RSNA Knee Abnormality Detection.
Provides square bounding box expansion and medial/lateral compartment tiling.
Used by train_knee.py when --roi is enabled.
"""

from typing import Tuple, Optional


def square_box(
    sbox: Tuple[int, int, int, int],
    W: int,
    H: int,
    pad: float = 0.06,
    mode: str = "tight",
    centroid: Optional[Tuple[float, float]] = None
) -> Tuple[int, int, int, int]:
    """Expands a tissue bounding box into a square crop centered on the joint.

    Args:
        sbox: (x0, y0, x1, y1) bounding box of tissue
        W: Width of image
        H: Height of image
        pad: Fraction of box size to add as padding
        mode: "tight" (minimal tissue frame), "safe" (conservative 1.2x scale), or "compartment"
        centroid: (cx, cy) optional center-of-mass to center the square box

    Returns:
        (x0, y0, x1, y1) integer coordinates clamped to [0, W] and [0, H]
    """
    x0, y0, x1, y1 = sbox

    # Handle degenerated or empty boxes
    if x1 <= x0 or y1 <= y0:
        return (0, 0, W, H)

    bw = x1 - x0
    bh = y1 - y0

    # Apply padding
    pad_px = int(max(bw, bh) * pad)
    if mode == "safe":
        pad_px += int(max(bw, bh) * 0.10)

    # Square dimension
    side = max(bw, bh) + 2 * pad_px
    side = min(side, min(W, H))

    # Center determination
    if centroid is not None and centroid[0] > 0 and centroid[1] > 0:
        cx, cy = centroid
    else:
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0

    # Half side
    half = side / 2.0
    new_x0 = int(round(cx - half))
    new_y0 = int(round(cy - half))

    # Shift if out of bounds
    if new_x0 < 0:
        new_x0 = 0
    elif new_x0 + side > W:
        new_x0 = W - side

    if new_y0 < 0:
        new_y0 = 0
    elif new_y0 + side > H:
        new_y0 = H - side

    new_x1 = new_x0 + side
    new_y1 = new_y0 + side

    # Final boundary clamp
    new_x0 = max(0, min(new_x0, W - 1))
    new_y0 = max(0, min(new_y0, H - 1))
    new_x1 = max(new_x0 + 1, min(new_x1, W))
    new_y1 = max(new_y0 + 1, min(new_y1, H))

    return (new_x0, new_y0, new_x1, new_y1)


def compartments(
    sq: Tuple[int, int, int, int],
    overlap: float = 0.12
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int, int]]:
    """Splits a square joint box into Medial and Lateral compartment tiles with overlap.

    Args:
        sq: (x0, y0, x1, y1) square bounding box
        overlap: Fraction of width to overlap between compartments (default 0.12 = 12%)

    Returns:
        (left_compartment, right_compartment)
    """
    x0, y0, x1, y1 = sq
    w = x1 - x0

    if w <= 2:
        return (sq, sq)

    half_w = w / 2.0
    ov_px = int(round(w * overlap))

    # Left compartment: x0 to middle + overlap
    left_x1 = int(round(x0 + half_w + ov_px))
    left_x1 = min(x1, max(x0 + 1, left_x1))
    left_box = (x0, y0, left_x1, y1)

    # Right compartment: middle - overlap to x1
    right_x0 = int(round(x0 + half_w - ov_px))
    right_x0 = max(x0, min(x1 - 1, right_x0))
    right_box = (right_x0, y0, x1, y1)

    return (left_box, right_box)
