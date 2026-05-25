"""Frame-based refinement using Qwen-VL with sampled video frames."""
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


class QwenVideoRefiner:
    """Refine annotations using sampled video frames from Qwen-VL."""

    def __init__(
        self,
        model_id: str,
        temperature: float = 0.2,
        processor=None,
        model=None,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.processor = processor or AutoProcessor.from_pretrained(model_id)
        self.model = model or Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True,
        )

    def refine(
        self,
        initial_annotation: dict,
        video_path: str | Path,
        chunk: dict,
        max_frames: int = 2,
        max_image_side: int = 320,
        max_new_tokens: int = 1000,
    ) -> tuple[dict, str]:
        """Refine annotation using sampled video frames.
        
        Args:
            initial_annotation: Output from text-only annotator
            video_path: Path to video file
            chunk: Chunk metadata with start_frame, end_frame
            max_frames: Maximum frames to sample (optimized for 10GB GPU)
            max_image_side: Maximum side length for sampled frames
            max_new_tokens: maximum tokens for visual refinement generation
            
        Returns:
            (refined_annotation_dict, raw_model_output)
        """
        sampled = self._load_sampled_frames(
            video_path,
            chunk,
            max_frames=max_frames,
            max_image_side=max_image_side,
        )
        metadata = [{"frame_index": item["frame_index"], "time_sec": item["time_sec"]} for item in sampled]

        prompt = VISION_REFINEMENT_PROMPT.format(
            initial_annotation=json.dumps(initial_annotation, indent=2),
            frame_metadata=json.dumps(metadata, indent=2),
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
        max_frames: int = 2,
        max_image_side: int = 320,
    ) -> list[dict]:
        """Load sampled frames from video, optimized for 10GB GPU."""
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
    """Extract the first JSON object from a model response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output:\n{text[:1000]}")

    return json.loads(text[start : end + 1])


VISION_REFINEMENT_PROMPT = """You are refining a hand-motion annotation using video frames.

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
- Update the label if the frames clearly show something different from the initial annotation.

Initial text-only annotation:
{initial_annotation}

Sampled frame metadata:
{frame_metadata}
"""
