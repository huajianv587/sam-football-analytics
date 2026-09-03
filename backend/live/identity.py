"""Small, model-agnostic face embedding matcher.

The face encoder is intentionally an optional deployment dependency (for
example InsightFace/ArcFace). Supabase stores the private profile metadata and
embedding; this module only performs the deterministic cosine match.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class FaceMatch:
    profile_id: str | None
    label: str | None
    score: float


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def best_face_match(
    query: Sequence[float], candidates: Sequence[dict], threshold: float = 0.45
) -> FaceMatch:
    best: tuple[float, dict] | None = None
    for candidate in candidates:
        embedding = candidate.get("embedding")
        if not isinstance(embedding, list):
            continue
        score = cosine_similarity(query, embedding)
        if best is None or score > best[0]:
            best = (score, candidate)
    if best is None or best[0] < threshold:
        return FaceMatch(None, None, round(best[0], 4) if best else 0.0)
    return FaceMatch(
        str(best[1].get("id")),
        str(best[1].get("label")),
        round(best[0], 4),
    )
