from collections.abc import Iterable

import numpy as np


def encode_mask(mask: np.ndarray) -> dict[str, object]:
    binary = np.asarray(mask, dtype=np.uint8).reshape(-1)
    if not binary.size:
        counts: list[int] = []
    else:
        changes = np.flatnonzero(binary[1:] != binary[:-1]) + 1
        boundaries = np.concatenate(([0], changes, [binary.size]))
        counts = np.diff(boundaries).astype(int).tolist()
        if binary[0]:
            counts.insert(0, 0)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts}


def decode_mask(rle: dict[str, object]) -> np.ndarray:
    height, width = (int(value) for value in rle["size"])
    values: list[int] = []
    bit = 0
    for count in rle["counts"]:
        values.extend([bit] * int(count))
        bit = 1 - bit
    return np.asarray(values, dtype=np.uint8).reshape(height, width)


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
