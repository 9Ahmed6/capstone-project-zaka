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

    def annotate_from_text(
        self,
        feature_json: dict,
        rag_context: dict,
    ) -> tuple[dict, str]:
        """Use Qwen-VL with a text-only prompt guided by RAG retrieval."""
        prompt = TEXT_ONLY_PROMPT.format(
            rag_context=_format_rag_context(rag_context),
            features=compact_json(feature_json),
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        raw_output = self._generate(messages, max_new_tokens=256)
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

        raw_output = self._generate(messages, max_new_tokens=256)
        return parse_json_object(raw_output), raw_output

    def refine_with_video_frames(
        self,
        initial_annotation: dict,
        video_path: str | Path,
        chunk: dict,
        max_frames: int = 8,
        rag_context: dict | None = None,
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
            rag_context=_format_rag_context(rag_context),
        )

        content = [{"type": "text", "text": prompt}]
        content.extend({"type": "image", "image": item["image"]} for item in sampled)
        messages = [{"role": "user", "content": content}]

        raw_output = self._generate(messages, max_new_tokens=256)
        return parse_json_object(raw_output), raw_output

    def annotate_chunk(
        self,
        feature_json: dict,
        video_path: str | Path,
        chunk: dict,
        rag_context: dict,
        max_frames: int = 8,
    ) -> tuple[dict, str]:
        """Annotate a chunk with one Qwen-VL call.

        Labels are chosen only from RAG retrieval over confirmed_actions.json,
        reconciled with HandX features and sampled video frames.
        """
        if not rag_context:
            raise ValueError("rag_context is required; run RAG retrieval before Qwen-VL annotation.")

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
            features=compact_json(feature_json),
            frame_metadata=json.dumps(metadata, indent=2),
            rag_context=_format_rag_context(rag_context),
        )

        content = [{"type": "text", "text": prompt}]
        content.extend({"type": "image", "image": item["image"]} for item in sampled)
        messages = [{"role": "user", "content": content}]

        raw_output = self._generate(messages, max_new_tokens=256)
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


def _format_rag_context(rag_context: dict | None) -> str:
    """Turn RAG retrieval output into prompt text for Qwen-VL."""
    if not rag_context:
        return "No RAG retrieval context available."

    if rag_context.get("prompt_text"):
        best = rag_context.get("best_match") or {}
        header = ""
        if best:
            header = (
                f"RAG top hypothesis: {best.get('action_label', 'unknown')} "
                f"(confidence={best.get('confidence', 0):.2f}, "
                f"scale={best.get('movement_scale', 'unknown')}, "
                f"hand={best.get('hand_side', 'unknown')})\n\n"
            )
        return header + rag_context["prompt_text"]

    return json.dumps(rag_context, indent=2)


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


TEXT_ONLY_PROMPT = """You are an expert in hand-motion and activity analysis.

Use HandX-style features and RAG retrieval from confirmed_actions.json to create a first action guess.
Return only valid JSON with this schema:
{{
  "action_label": "one label from the RAG candidates or unknown",
  "movement_scale": "micro, macro, bimanual, or unknown",
  "hand_side": "left, right, both, or unknown",
  "confidence": 0.0,
  "summary": "2-4 sentences: inferred hand motion, plausible objects or goals if supported by features/RAG, and likely action meaning or intent; note when visuals are not available in this pass",
  "evidence": "which features and RAG candidates support the guess"
}}

Rules:
- Choose action_label only from the RAG candidate labels listed below.
- In summary, describe kinematics plus a cautious interpretation of what the person may be trying to do.
- Do not claim specific objects you cannot infer from features alone; use hedged language (e.g., "possibly reaching toward an object").
- Use confidence below 0.6 when the features are weak.

RAG retrieval (confirmed_actions dictionary):
{rag_context}

HandX-style features:
{features}
"""


VISION_PROMPT = """You are refining a hand-motion annotation using video frames.

Return only valid JSON with this schema:
{{
  "action_label": "one label from the RAG candidates or unknown",
  "movement_scale": "micro, macro, bimanual, or unknown",
  "hand_side": "left, right, both, or unknown",
  "confidence": 0.0,
  "summary": "2-5 sentences describing what you see in the frames: visible objects and scene context, hand poses and motion, contact or manipulation, and the likely meaning or intent of the action; hedge when uncertain",
  "evidence": "what the sampled frames and RAG candidates show"
}}

Rules:
- Choose action_label only from the RAG candidate labels in confirmed_actions.json.
- In summary, ground object and intent statements in visible evidence from the frames (tools, containers, surfaces, other people, etc.).
- Prefer visible hand pose, finger bending, wrist movement, hand distance, and object contact.
- Use RAG candidates as biomechanical priors, but override them when frames clearly disagree.
- Separate physical description from intent when helpful (e.g., "fingers pinch a small object" vs "likely picking up a coin").
- If the frames are unclear, keep the label unknown, lower confidence, and say what is ambiguous in the summary.

RAG retrieval (confirmed_actions dictionary):
{rag_context}

Initial text-only annotation:
{initial_annotation}

Sampled frame metadata:
{frame_metadata}
"""


ONE_PASS_PROMPT = """You are an expert in hand-motion and visual activity analysis.

You will receive:
1. A video chunk with timestamps.
2. HandX-style motion features for the chunk.
3. RAG retrieval candidates from confirmed_actions.json (the only allowed label vocabulary).
4. Sampled frames from that same chunk.

Return only valid JSON with this schema:
{{
  "action_label": "one label from the RAG candidates or unknown",
  "movement_scale": "micro, macro, bimanual, or unknown",
  "hand_side": "left, right, both, or unknown",
  "confidence": 0.0,
  "summary": "Rich visual narrative (2-5 sentences): (a) what appears in the frames—objects, surfaces, tools, and scene context; (b) what each visible hand is doing—pose, motion, contact; (c) the likely meaning or intent of the action (e.g., grasping to pick up, waving to greet, pointing to indicate). Hedge or say 'unclear' when not visible.",
  "evidence": "Bullet-style facts tying the label to frames, objects, hand motion, and RAG candidates"
}}

Rules:
- Choose action_label only from the RAG candidate labels listed below (confirmed_actions dictionary).
- Treat RAG candidates as strong priors from kinematic similarity; reconcile them with frames and features.
- Prefer the best-matching RAG label when frames are ambiguous but kinematics align.
- Use "unknown" if no RAG candidate clearly matches after checking features and frames.
- movement_scale and hand_side should align with the chosen RAG entry when possible.
- summary must reflect what you actually see in the sampled frames, not only kinematic features.
- Name objects when visible (cup, phone, door handle, table, etc.); if none are clear, say so.
- Include action meaning or intent when the visuals support a reasonable interpretation; mark guesses as likely/possible.
- Use confidence below 0.6 if the frames, objects, or intent are unclear.

Video chunk:
{chunk}

RAG retrieval candidates (confirmed_actions — label vocabulary):
{rag_context}

HandX-style features:
{features}

Sampled frame metadata:
{frame_metadata}
"""
