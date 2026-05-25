"""Text-only annotation using Qwen-VL with HandX features and action library."""
from __future__ import annotations

import json

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

from src.handx_features import compact_json


class QwenTextAnnotator:
    """Annotate chunks using only text: action library + HandX motion features."""

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

    def annotate(self, feature_json: dict, action_library: list[dict]) -> tuple[dict, str]:
        """Run text-only annotation on motion features.
        
        Args:
            feature_json: HandX-style motion features
            action_library: List of allowed action definitions
            
        Returns:
            (annotation_dict, raw_model_output)
        """
        prompt = TEXT_ONLY_PROMPT.format(
            action_library=json.dumps(action_library, indent=2),
            features=compact_json(feature_json),
        )
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        raw_output = self._generate(messages, max_new_tokens=1200)
        return parse_json_object(raw_output), raw_output

    def _generate(self, messages: list[dict], max_new_tokens: int) -> str:
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(
            text=[prompt],
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


def parse_json_object(text: str) -> dict:
    """Extract the first JSON object from a model response."""
    import re
    
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output:\n{text[:1000]}")

    return json.loads(text[start : end + 1])


def load_qwen_model(model_id: str, temperature: float = 0.2):
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    return processor, model


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
