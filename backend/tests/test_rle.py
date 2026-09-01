import numpy as np

from worker.rle import decode_mask, encode_mask, mask_foot_point


def test_rle_round_trip() -> None:
    mask = np.zeros((6, 7), dtype=np.uint8)
    mask[2:5, 3:6] = 1
    assert np.array_equal(decode_mask(encode_mask(mask)), mask)


def test_foot_point_uses_mask_bottom() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:9, 4:7] = 1
    assert mask_foot_point(mask) == (5.0, 8.0)
