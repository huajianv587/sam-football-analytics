import numpy as np

from worker.rle import decode_mask, encode_mask, mask_foot_point


def test_rle_round_trip() -> None:
    mask = np.zeros((6, 7), dtype=np.uint8)
    mask[2:5, 3:6] = 1
    encoded = encode_mask(mask)
    assert encoded["bbox"] == [3, 2, 6, 5]
    assert np.array_equal(decode_mask(encoded), mask)


def test_empty_mask_round_trip() -> None:
    mask = np.zeros((3, 4), dtype=np.uint8)
    assert np.array_equal(decode_mask(encode_mask(mask)), mask)


def test_decode_rejects_invalid_length() -> None:
    with np.testing.assert_raises_regex(ValueError, "RLE length"):
        decode_mask({"size": [2, 2], "counts": [1, 1]})


def test_decode_keeps_legacy_full_frame_rle_compatible() -> None:
    assert np.array_equal(
        decode_mask({"size": [2, 3], "counts": [2, 3, 1]}),
        np.asarray([[0, 0, 1], [1, 1, 0]], dtype=np.uint8),
    )


def test_foot_point_uses_mask_bottom() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:9, 4:7] = 1
    assert mask_foot_point(mask) == (5.0, 8.0)
