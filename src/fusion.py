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
    """Run one Qwen-VL call per chunk, with optional frame support."""

    def __init__(self, model_id: str, temperature: float = 0.2):
        self.model_id = model_id
        self.temperature = temperature
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    def annotate_chunk(
        self,
        feature_json: dict,
        action_library: list[dict],
        video_path: str | Path,
        chunk: dict,
        max_frames: int = 0,
        max_new_tokens: int = 1400,
    ) -> tuple[dict, str]:
        """Annotate one video chunk with a single Qwen-VL request."""
        sampled = self._load_sampled_frames(video_path, chunk, max_frames=max_frames) if max_frames > 0 else []

        prompt = ONE_PASS_PROMPT.format(
            chunk=json.dumps(
                {
                    "chunk_id": chunk.get("chunk_id"),
                    "start_time": chunk.get("start_time"),
                    "end_time": chunk.get("end_time"),
                    "start_frame": chunk.get("start_frame"),
                    "end_frame": chunk.get("end_frame"),
                },
                indent=2,
            ),
            action_library=json.dumps(action_library, indent=2),
            features=compact_json(feature_json),
            frame_metadata=json.dumps(
                [
                    {"frame_index": item["frame_index"], "time_sec": item["time_sec"]}
                    for item in sampled
                ],
                indent=2,
            ),
        )

        content = [{"type": "text", "text": prompt}]
        content.extend({"type": "image", "image": item["image"]} for item in sampled)
        messages = [{"role": "user", "content": content}]

        raw_output = self._generate(messages, max_new_tokens=max_new_tokens)
        return parse_json_object(raw_output), raw_output

    def _load_sampled_frames(
        self,
        video_path: str | Path,
        chunk: dict,
        max_frames: int = 4,
        max_image_side: int = 640,
    ) -> list[dict]:
        start_frame = int(chunk.get("start_frame", 0))
        end_frame = int(chunk.get("end_frame", start_frame))
        if end_frame < start_frame:
            end_frame = start_frame

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        frame_count = max(1, end_frame - start_frame + 1)
        samples = min(max_frames, frame_count)
        indices = np.linspace(start_frame, end_frame, num=samples, dtype=int).tolist()

        sampled = []
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        try:
            for frame_index in indices:
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

        return sampled

    def _generate(self, messages: list[dict], max_new_tokens: int) -> str:
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        kwargs = {"text": [prompt], "padding": True, "return_tensors": "pt"}
        if image_inputs:
            kwargs["images"] = image_inputs
        if video_inputs:
            kwargs["videos"] = video_inputs

        inputs = self.processor(**kwargs).to(self.model.device)

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


def parse_json_object(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output:\n{text[:1000]}")

    return json.loads(text[start : end + 1])


ONE_PASS_PROMPT = """You are an expert in hand-motion analysis.

You will receive:
1. A video chunk with timestamps.
2. A controlled action library.
3. HandX-style motion features for the chunk.
4. Sampled frames from that same chunk.

Return only valid JSON with this schema:
{
  "action_label": "one label from the action library or unknown",
  "movement_scale": "micro, macro, bimanual, or unknown",
  "hand_side": "left, right, both, or unknown",
  "confidence": 0.0,
  "summary": "short visible description of the hand action",
  "evidence": "what features and frames support the label"
}

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