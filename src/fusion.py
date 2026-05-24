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

DEFAULT_MAX_FRAMES = 6
DEFAULT_MAX_IMAGE_SIDE = 512


class QwenVLAnnotator:
    """One-shot Qwen-VL chunk annotator optimized for low GPU usage."""

    def __init__(self, model_id: str, temperature: float = 0.2):
        self.model_id = model_id
        self.temperature = temperature
        self.processor = AutoProcessor.from_pretrained(model_id)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()

    def annotate_chunk(
        self,
        feature_json: dict,
        action_library: list[dict],
        video_path: str | Path,
        chunk: dict,
        max_frames: int = DEFAULT_MAX_FRAMES,
        max_image_side: int = DEFAULT_MAX_IMAGE_SIDE,
    ) -> tuple[dict, str]:
        """Annotate a chunk with a single Qwen-VL request."""
        sampled = sample_frames_for_qwen_from_video(
            video_path=video_path,
            chunk=chunk,
            max_frames=max_frames,
            max_image_side=max_image_side,
        )
        metadata = [
            {"frame_index": item["frame_index"], "time_sec": item["time_sec"]}
            for item in sampled
        ]

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
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                use_cache=True,
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


def sample_frames_for_qwen_from_video(
    video_path: str | Path,
    chunk: dict,
    max_frames: int = DEFAULT_MAX_FRAMES,
    max_image_side: int = DEFAULT_MAX_IMAGE_SIDE,
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
