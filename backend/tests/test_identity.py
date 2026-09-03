from live.identity import best_face_match, cosine_similarity


def test_cosine_similarity_is_zero_for_invalid_vectors() -> None:
    assert cosine_similarity([], []) == 0
    assert cosine_similarity([1, 0], [0]) == 0
    assert cosine_similarity([0, 0], [1, 0]) == 0


def test_best_face_match_returns_highest_candidate() -> None:
    result = best_face_match(
        [1, 0],
        [
            {"id": "alice", "label": "Alice", "embedding": [0.9, 0.1]},
            {"id": "bob", "label": "Bob", "embedding": [0, 1]},
        ],
        threshold=0.7,
    )
    assert result.profile_id == "alice"
    assert result.label == "Alice"
    assert result.score > 0.9


def test_best_face_match_keeps_unknown_when_below_threshold() -> None:
    result = best_face_match([1, 0], [{"id": "bob", "label": "Bob", "embedding": [0, 1]}])
    assert result.profile_id is None
    assert result.label is None
