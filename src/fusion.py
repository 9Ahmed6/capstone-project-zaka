from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.handx_features import compact_json


class QwenVLAnnotator:
    """Use one Qwen-VL model as both the LLM and VLM."""

    def __init__(self, model_id: str, temperature: float = 0.2):
        self.model_id = model_id
        self.temperature = temperature
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

    def annotate_from_text(self, feature_json: dict, action_library: list[dict]) -> tuple[dict, str]:
        """Use Qwen-VL with a text-only prompt.

        This replaces the separate LLM from the original notebook.
        """
        prompt = TEXT_ONLY_PROMPT.format(
            action_library=json.dumps(action_library, indent=2),
            features=compact_json(feature_json),
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        raw_output = self._generate(messages, max_new_tokens=600)
        return parse_json_object(raw_output), raw_output

    def refine_with_frames(
        self,
        initial_annotation: dict,
        frames: list[np.ndarray],
        timestamps: list[float],
        chunk: dict,
        max_frames: int = 8,
    ) -> tuple[dict, str]:
        """Use Qwen-VL with sampled frames for visual refinement."""
        sampled = sample_frames_for_qwen(
            frames,
            timestamps,
            chunk["start_frame"],
            chunk["end_frame"],
            max_frames=max_frames,
        )
        metadata = [{"frame_index": item["frame_index"], "time_sec": item["time_sec"]} for item in sampled]

        prompt = VISION_PROMPT.format(
            initial_annotation=json.dumps(initial_annotation, indent=2),
            frame_metadata=json.dumps(metadata, indent=2),
        )

        content = [{"type": "text", "text": prompt}]
        content.extend({"type": "image", "image": item["image"]} for item in sampled)
        messages = [{"role": "user", "content": content}]

        raw_output = self._generate(messages, max_new_tokens=1400)
        return parse_json_object(raw_output), raw_output

    def refine_with_video_frames(
        self,
        initial_annotation: dict,
        video_path: str | Path,
        chunk: dict,
        max_frames: int = 4,
    ) -> tuple[dict, str]:
        """Reload only the sampled frames needed by Qwen-VL.

        This is the memory-friendly version used by the main pipeline. The full
        video is not kept in RAM.
        """
        sampled = sample_frames_for_qwen_from_video(video_path, chunk, max_frames=max_frames)
        metadata = [{"frame_index": item["frame_index"], "time_sec": item["time_sec"]} for item in sampled]

        prompt = VISION_PROMPT.format(
            initial_annotation=json.dumps(initial_annotation, indent=2),
            frame_metadata=json.dumps(metadata, indent=2),
        )

        content = [{"type": "text", "text": prompt}]
        content.extend({"type": "image", "image": item["image"]} for item in sampled)
        messages = [{"role": "user", "content": content}]

        raw_output = self._generate(messages, max_new_tokens=1400)
        return parse_json_object(raw_output), raw_output

    def annotate_chunk(
        self,
        feature_json: dict,
        action_library: list[dict],
        video_path: str | Path,
        chunk: dict,
        max_frames: int = 8,
    ) -> tuple[dict, str]:
        """Annotate a chunk with one Qwen-VL call.

        This sends the model everything it needs at once:
        - HandX-style motion features
        - the allowed action library
        - chunk timestamps
        - sampled frames from the original video
        """
        sampled = sample_frames_for_qwen_from_video(video_path, chunk, max_frames=max_frames)
        metadata = [{"frame_index": item["frame_index"], "time_sec": item["time_sec"]} for item in sampled]

        prompt = ONE_PASS_PROMPT.format(
            chunk=json.dumps(
                {
                    "chunk_id": chunk["chunk_id"],
                    "start_time": chunk["start_time"],
                    "end_time": chunk["end_time"],
                    "start_frame": chunk["start_frame"],
                    "end_frame": chunk["end_frame"],
                },
                indent=2,
            ),
            action_library=json.dumps(action_library, indent=2),
            features=compact_json(feature_json),
            frame_metadata=json.dumps(metadata, indent=2),
        )

        content = [{"type": "text", "text": prompt}]
        content.extend({"type": "image", "image": item["image"]} for item in sampled)
        messages = [{"role": "user", "content": content}]

        raw_output = self._generate(messages, max_new_tokens=1400)
        return parse_json_object(raw_output), raw_output

    def _generate(self, messages: list[dict], max_new_tokens: int) -> str:
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
            )

        generated_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        return self.processor.batch_decode(
            generated_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]


def sample_frames_for_qwen(
    frames: list[np.ndarray],
    timestamps: list[float],
    start_frame: int,
    end_frame: int,
    max_frames: int = 8,
) -> list[dict]:
    """Pick a small number of frames from a chunk."""
    start_frame = max(0, start_frame)
    end_frame = min(end_frame, len(frames) - 1)
    count = min(max_frames, end_frame - start_frame + 1)
    indices = np.linspace(start_frame, end_frame, num=count, dtype=int)

    sampled = []
    for frame_index in indices:
        rgb = cv2.cvtColor(frames[frame_index], cv2.COLOR_BGR2RGB)
        sampled.append(
            {
                "frame_index": int(frame_index),
                "time_sec": float(timestamps[frame_index]),
                "image": Image.fromarray(rgb),
            }
        )
    return sampled


def sample_frames_for_qwen_from_video(
    video_path: str | Path,
    chunk: dict,
    max_frames: int = 8,
    max_image_side: int = 640,
) -> list[dict]:
    """Read only a few frames from a chunk for Qwen-VL.

    The chunk stores original video frame numbers, so this function can seek
    directly to those frames without loading the whole video.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(chunk["start_frame"])
    end_frame = int(chunk["end_frame"])
    count = min(max_frames, max(1, end_frame - start_frame + 1))
    frame_indices = np.linspace(start_frame, end_frame, num=count, dtype=int)

    sampled = []
    try:
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            image.thumbnail((max_image_side, max_image_side))
            sampled.append(
                {
                    "frame_index": int(frame_index),
                    "time_sec": float(frame_index / fps),
                    "image": image,
                }
            )
    finally:
        cap.release()

    if not sampled:
        raise ValueError(f"Could not sample frames for chunk {chunk.get('chunk_id', '<unknown>')}")

    return sampled


def parse_json_object(text: str) -> dict:
    """Extract the first JSON object from a model response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output:\n{text[:1000]}")

    return json.loads(text[start : end + 1])


TEXT_ONLY_PROMPT = """You are an expert in hand-motion analysis.

Use the HandX-style feature JSON and action library to create a first action guess.
Return only valid JSON with this schema:
{{
  "action_label": "one label from the action library or unknown",
  "movement_scale": "micro, macro, bimanual, or unknown",
  "hand_side": "left, right, both, or unknown",
  "confidence": 0.0,
  "summary": "short physical motion description",
  "evidence": "which feature values support the guess"
}}

Rules:
- Describe physical motion only.
- Do not guess the person's intention.
- Use confidence below 0.6 when the features are weak.

Action library:
{action_library}

HandX-style features:
{features}
"""


VISION_PROMPT = """You are refining a hand-motion annotation using video frames.

Return only valid JSON with this schema:
{{
  "action_label": "one final action label or unknown",
  "movement_scale": "micro, macro, bimanual, or unknown",
  "hand_side": "left, right, both, or unknown",
  "confidence": 0.0,
  "summary": "visible motion across the chunk",
  "evidence": "what the sampled frames show"
}}

Rules:
- Prefer visible hand pose, finger bending, wrist movement, hand distance, and object contact.
- Do not invent object purpose or gesture meaning.
- If the frames are unclear, keep the label unknown and lower confidence.

Initial text-only annotation:
{initial_annotation}

Sampled frame metadata:
{frame_metadata}
"""


ONE_PASS_PROMPT = """You are an expert in hand-motion analysis.

You will receive:
1. A video chunk with timestamps.
2. A controlled action library.
3. HandX-style motion features for the chunk.
4. Sampled frames from that same chunk.

Return only valid JSON with this schema:
{{
  "action_label": "one label from the action library or unknown",
  "movement_scale": "micro, macro, bimanual, or unknown",
  "hand_side": "left, right, both, or unknown",
  "confidence": 0.0,
  "summary": "short visible description of the hand action",
  "evidence": "what features and frames support the label"
}}

Rules:
- Choose labels from the action library when possible.
- Use "unknown" if no action label clearly matches.
- Describe physical motion only: finger bending, hand opening, wrist movement, hand travel, contact, or two-hand relation.
- Do not guess the person's intention.
- Use confidence below 0.6 if the frames or features are unclear.

Video chunk:
{chunk}

Action library:
{action_library}

HandX-style features:
{features}

Sampled frame metadata:
{frame_metadata}
"""
