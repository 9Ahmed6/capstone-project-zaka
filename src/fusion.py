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
        raw_output = self._generate(messages, max_new_tokens=1200)
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
