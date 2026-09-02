from collections.abc import Iterable

import numpy as np


def encode_mask(mask: np.ndarray) -> dict[str, object]:
    """Encode only the non-empty mask crop while preserving full-frame coordinates."""
    binary = np.asarray(mask, dtype=np.uint8)
    height, width = binary.shape
    rows = np.flatnonzero(binary.any(axis=1))
    columns = np.flatnonzero(binary.any(axis=0))
    if not len(rows) or not len(columns):
        return {"size": [height, width], "bbox": [0, 0, 0, 0], "counts": []}

    x1, x2 = int(columns[0]), int(columns[-1]) + 1
    y1, y2 = int(rows[0]), int(rows[-1]) + 1
    crop = binary[y1:y2, x1:x2].reshape(-1)
    changes = np.flatnonzero(crop[1:] != crop[:-1]) + 1
    boundaries = np.concatenate(([0], changes, [crop.size]))
    counts = np.diff(boundaries).astype(int).tolist()
    if crop[0]:
        counts.insert(0, 0)
    return {
        "size": [height, width],
        "bbox": [x1, y1, x2, y2],
        "counts": counts,
    }


def decode_mask(rle: dict[str, object]) -> np.ndarray:
    height, width = (int(value) for value in rle["size"])
    counts = np.asarray(rle["counts"], dtype=np.int64)
    values = np.repeat(np.arange(len(counts), dtype=np.uint8) & 1, counts)
    bbox = rle.get("bbox")
    if bbox is None:
        if values.size != height * width:
            raise ValueError("RLE length does not match mask dimensions")
        return values.reshape(height, width)

    x1, y1, x2, y2 = (int(value) for value in bbox)
    crop_height, crop_width = y2 - y1, x2 - x1
    if not 0 <= x1 <= x2 <= width or not 0 <= y1 <= y2 <= height:
        raise ValueError("RLE crop lies outside mask dimensions")
    if values.size != crop_height * crop_width:
        raise ValueError("RLE length does not match mask dimensions")
    mask = np.zeros((height, width), dtype=np.uint8)
    if values.size:
        mask[y1:y2, x1:x2] = values.reshape(crop_height, crop_width)
    return mask


def mask_foot_point(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    cutoff = np.quantile(ys, 0.95)
    bottom_xs = xs[ys >= cutoff]
    return float(np.median(bottom_xs)), float(np.max(ys))


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
