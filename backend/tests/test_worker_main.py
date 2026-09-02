from worker.main import (
    fast_role_from_appearance,
    merge_jersey_predictions,
    propagation_directions,
    sam_compile_without_cudagraphs,
)


def test_sam_compile_keeps_autotune_but_disables_cudagraphs():
    calls = []

    def compile_fn(model, *args, **kwargs):
        calls.append((model, args, kwargs))
        return "compiled"

    result = sam_compile_without_cudagraphs(compile_fn)(
        "decoder", fullgraph=True, mode="max-autotune", dynamic=False
    )

    assert result == "compiled"
    assert calls[0][2] == {
        "fullgraph": True,
        "mode": "max-autotune-no-cudagraphs",
        "dynamic": False,
    }


def test_sam_compile_leaves_other_modes_unchanged():
    def compile_fn(model, *args, **kwargs):
        return kwargs

    result = sam_compile_without_cudagraphs(compile_fn)("encoder", mode="default")

    assert result["mode"] == "default"


def test_jersey_predictions_reinforce_the_same_number():
    assert merge_jersey_predictions((10, 0.7), (10, 0.8)) == (10, 0.94)


def test_jersey_predictions_do_not_average_disagreements():
    assert merge_jersey_predictions((7, 0.82), (17, 0.63)) == (7, 0.82)


def test_jersey_prediction_uses_available_source():
    assert merge_jersey_predictions((None, 0.0), (9, 0.74)) == (9, 0.74)


def test_forward_propagation_starts_at_earliest_prompt():
    prompts = {1: [{"frame": 8}], 2: [{"frame": 3}, {"frame": 20}]}
    assert propagation_directions(prompts, bidirectional=False) == [(3, False)]


def test_bidirectional_propagation_partitions_window_at_common_anchor():
    prompts = {1: [{"frame": 8}], 2: [{"frame": 3}, {"frame": 20}]}
    assert propagation_directions(prompts, bidirectional=True) == [
        (8, False),
        (8, True),
    ]


def test_fast_role_uses_referee_colour_and_goal_area_position():
    assert fast_role_from_appearance("Referee", []) == ("referee", 0.9)
    assert fast_role_from_appearance(
        "unknown", [{"pitch": {"x_bottom_middle": 46}}]
    ) == ("goalkeeper", 0.65)
    assert fast_role_from_appearance(
        "unknown", [{"pitch": {"x_bottom_middle": 3}}]
    ) == ("player", 0.5)
