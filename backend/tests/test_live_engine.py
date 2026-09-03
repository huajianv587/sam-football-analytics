from live.engine import LiveInferenceEngine


def test_numeric_cuda_device_is_normalized_for_fp16_and_sam() -> None:
    assert LiveInferenceEngine._resolve_device("0") == "cuda:0"
    assert LiveInferenceEngine._resolve_device("cuda:1") == "cuda:1"
