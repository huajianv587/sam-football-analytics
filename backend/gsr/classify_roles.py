import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from role_logic import parse_role, representative_detections, vote_role

PROMPT = (
    "The first image is a football broadcast frame and the second is one person cropped from it. "
    "Classify the person's role. Reply with exactly one lowercase word: "
    "player, goalkeeper, referee, or other."
)


def classify(
    video: Path,
    state_path: Path,
    model_path: Path,
    output: Path,
    selected_track_ids: set[int] | None = None,
) -> None:
    state = json.loads(state_path.read_text())
    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for frame in state["frames"]:
        for detection in frame.get("tracks", []):
            track_id = int(detection["track_id"])
            if selected_track_ids is not None and track_id not in selected_track_ids:
                continue
            by_track[track_id].append(
                {**detection, "frame": int(frame["index"])}
            )

    jobs = [
        {"track_id": track_id, **detection}
        for track_id, detections in by_track.items()
        for detection in representative_detections(detections)
    ]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda").eval()
    model.generation_config.temperature = None
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=128 * 28 * 28,
        max_pixels=512 * 28 * 28,
        use_fast=False,
    )
    processor.tokenizer.padding_side = "left"
    votes: dict[int, list[str]] = defaultdict(list)
    capture = cv2.VideoCapture(str(video))
    batch_size = 4
    for offset in range(0, len(jobs), batch_size):
        messages = []
        batch = jobs[offset : offset + batch_size]
        for job in batch:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(job["frame"]))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"unable to read role frame {job['frame']}")
            x1, y1, x2, y2 = [int(round(value)) for value in job["bbox"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                raise RuntimeError(f"empty role crop for track {job['track_id']}")
            full_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            crop_image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": full_image},
                            {"type": "image", "image": crop_image},
                            {"type": "text", "text": PROMPT},
                        ],
                    }
                ]
            )
        texts = [
            processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            for message in messages
        ]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=4, do_sample=False, use_cache=True)
        trimmed = [result[len(source) :] for source, result in zip(inputs.input_ids, generated)]
        answers = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for job, answer in zip(batch, answers):
            role = parse_role(answer)
            if role:
                votes[int(job["track_id"])].append(role)
    capture.release()

    result = {}
    for track_id, track_votes in votes.items():
        role, confidence = vote_role(track_votes)
        result[str(track_id)] = {
            "role": role,
            "confidence": confidence,
            "votes": track_votes,
        }
    output.write_text(json.dumps(result, separators=(",", ":")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("state", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("track_ids", nargs="?", default="")
    arguments = parser.parse_args()
    selected = {int(value) for value in arguments.track_ids.split(",") if value}
    classify(
        arguments.video,
        arguments.state,
        arguments.model,
        arguments.output,
        selected or None,
    )
