from live.stable_tracking import RawDetection, StableTrackRegistry


def detection(raw_id: int, x: float, y: float = 10.0) -> RawDetection:
    return RawDetection(raw_id, (x, y, x + 20.0, y + 50.0), 0.9, "person", [(x, y), (x + 20, y)])


def test_public_id_survives_raw_tracker_id_change() -> None:
    registry = StableTrackRegistry(max_missing_frames=3)
    first = registry.update(0, [detection(7, 10)])
    second = registry.update(1, [detection(22, 12)])
    assert first[0].track_id == second[0].track_id == 1
    assert second[0].raw_tracker_id == 22
    assert second[0].reassociation_count >= 1


def test_short_gap_returns_predicted_track_and_translated_mask() -> None:
    registry = StableTrackRegistry(max_missing_frames=3)
    registry.update(0, [detection(7, 10)])
    predicted = registry.update(1, [])
    assert predicted[0].track_id == 1
    assert predicted[0].state == "predicted"
    assert predicted[0].polygon


def test_expired_track_is_removed_and_id_is_never_reused() -> None:
    registry = StableTrackRegistry(max_missing_frames=1)
    registry.update(0, [detection(7, 10)])
    assert registry.update(1, [])[0].track_id == 1
    assert registry.update(2, []) == []
    fresh = registry.update(3, [detection(7, 100)])
    assert fresh[0].track_id == 2


def test_raw_id_reassociation_is_counted_once() -> None:
    registry = StableTrackRegistry(max_missing_frames=3)
    registry.update(0, [detection(7, 10)])
    reassociated = registry.update(1, [detection(22, 12)])
    assert reassociated[0].track_id == 1
    assert reassociated[0].reassociation_count == 1
