from collections import Counter
from statistics import median
from typing import Any


ROLES = {"player", "goalkeeper", "referee", "other"}


def representative_detections(
    detections: list[dict[str, Any]], samples_per_track: int = 2
) -> list[dict[str, Any]]:
    ranked = sorted(
        detections,
        key=lambda item: float(item.get("confidence") or 0)
        * max(1.0, (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1])),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    for candidate in ranked:
        if all(abs(int(candidate["frame"]) - int(item["frame"])) >= 8 for item in selected):
            selected.append(candidate)
        if len(selected) == samples_per_track:
            break
    return selected or ranked[:1]


def parse_role(text: str) -> str | None:
    answer = text.strip().lower().split(maxsplit=1)[0].strip(".,:;!?") if text.strip() else ""
    return answer if answer in ROLES else None


def vote_role(votes: list[str]) -> tuple[str, float]:
    role, count = Counter(votes).most_common(1)[0]
    return role, round(count / len(votes), 3)


def resolve_role_with_pitch(role: str, detections: list[dict[str, Any]]) -> str:
    """Reject a goalkeeper label when the whole track lives in midfield.

    Dark referee kits are visually close to goalkeeper kits in wide broadcast
    crops. A goalkeeper can leave the penalty area briefly, but a representative
    short track centered more than 16.5 m from either goal is not a goalkeeper.
    """
    if role != "goalkeeper":
        return role
    pitch_x = [
        float(item["pitch"]["x_bottom_middle"])
        for item in detections
        if isinstance(item.get("pitch"), dict)
        and item["pitch"].get("x_bottom_middle") is not None
    ]
    if pitch_x and abs(median(pitch_x)) < 36.0:
        return "referee"
    return role
