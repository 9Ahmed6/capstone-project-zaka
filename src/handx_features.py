from __future__ import annotations

import json
import importlib
import sys
from pathlib import Path

import numpy as np


def extract_handx_motion_features(
    motion_clip: np.ndarray,
    active_slots: list[int],
    handx_diffusion_path: str | None = None,
) -> dict:
    """Extract HandX motion events.

    If the real HandX library is available, this function uses it. If not, it
    returns a small fallback feature JSON so the Week 5 pipeline can still run.
    """
    if handx_diffusion_path and Path(handx_diffusion_path).exists():
        try:
            return _extract_with_handx_library(motion_clip, active_slots, handx_diffusion_path)
        except Exception as exc:
            fallback = _extract_simple_features(motion_clip, active_slots)
            fallback["handx_error"] = str(exc)
            return fallback

    return _extract_simple_features(motion_clip, active_slots)


def compact_json(data: dict, max_chars: int = 9000) -> str:
    """Keep model prompts from becoming too long."""
    text = json.dumps(data, indent=2)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... truncated ..."


def _extract_with_handx_library(motion_clip: np.ndarray, active_slots: list[int], handx_diffusion_path: str) -> dict:
    # HandX has its own package named "src". This project is also named "src".
    # During HandX extraction, temporarily remove this project's src modules so
    # imports like "src.feature.single_motioncode" resolve to HandX instead.
    handx_path = str(Path(handx_diffusion_path).resolve())
    handx_src_path = Path(handx_path) / "src"
    handx_src_path.mkdir(parents=True, exist_ok=True)

    # Some HandX copies do not include this file. Without it, Python may prefer
    # this project's real src package over HandX's namespace folder.
    handx_init = handx_src_path / "__init__.py"
    handx_init.touch(exist_ok=True)

    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "src" or name.startswith("src.")
    }
    for name in saved_modules:
        sys.modules.pop(name, None)

    sys.path.insert(0, handx_path)
    try:
        importlib.invalidate_caches()
        from src.feature.bihand_motioncode import BihandMotionCoder
        from src.feature.single_motioncode import MotionCoder

        motion_clip = np.asarray(motion_clip, dtype=np.float32)

        if motion_clip.ndim == 3:
            slot = active_slots[0]
            coder = MotionCoder(motion_clip, isright=(slot == 1))
            coder.generate_motion_codes()
            events = coder.print_json()
            return {
                "frame_count": int(motion_clip.shape[0]),
                "detected_hands": [_slot_name(slot)],
                "left_hand_events": events if slot == 0 else {},
                "right_hand_events": events if slot == 1 else {},
                "two_hand_relationships": {},
                "source": "handx",
            }

        coder = BihandMotionCoder(motion_clip)
        coder.generate_motion_codes()
        features = coder.get_json()
        features["source"] = "handx"
        return features
    finally:
        if sys.path and sys.path[0] == handx_path:
            sys.path.pop(0)

        for name in list(sys.modules):
            if name == "src" or name.startswith("src."):
                sys.modules.pop(name, None)
        for name, module in saved_modules.items():
            sys.modules[name] = module
        importlib.invalidate_caches()


def _extract_simple_features(motion_clip: np.ndarray, active_slots: list[int]) -> dict:
    """Small readable fallback when HandX is not installed."""
    clip = np.asarray(motion_clip, dtype=np.float32)
    if clip.ndim == 3:
        clip = clip[:, None, :, :]

    per_hand = {}
    for local_index, slot in enumerate(active_slots):
        hand_points = clip[:, local_index]
        center = hand_points.mean(axis=1)
        speed = np.linalg.norm(np.diff(center, axis=0), axis=1)
        openness = _estimate_hand_openness(hand_points)
        per_hand[_slot_name(slot)] = {
            "mean_center_speed": float(speed.mean()) if len(speed) else 0.0,
            "max_center_speed": float(speed.max()) if len(speed) else 0.0,
            "openness_start": float(openness[0]),
            "openness_end": float(openness[-1]),
            "openness_change": float(openness[-1] - openness[0]),
        }

    return {
        "frame_count": int(clip.shape[0]),
        "detected_hands": [_slot_name(slot) for slot in active_slots],
        "hand_motion_summary": per_hand,
        "two_hand_relationships": _two_hand_distance(clip, active_slots),
        "source": "simple_fallback",
    }


def _estimate_hand_openness(hand_points: np.ndarray) -> np.ndarray:
    wrist = hand_points[:, 0]
    fingertip_indices = [17, 18, 19, 20]
    fingertips = hand_points[:, fingertip_indices]
    return np.linalg.norm(fingertips - wrist[:, None, :], axis=-1).mean(axis=1)


def _two_hand_distance(clip: np.ndarray, active_slots: list[int]) -> dict:
    if len(active_slots) < 2 or clip.shape[1] < 2:
        return {}

    left_center = clip[:, 0].mean(axis=1)
    right_center = clip[:, 1].mean(axis=1)
    distance = np.linalg.norm(left_center - right_center, axis=1)
    return {
        "mean_distance": float(distance.mean()),
        "start_distance": float(distance[0]),
        "end_distance": float(distance[-1]),
        "distance_change": float(distance[-1] - distance[0]),
    }


def _slot_name(slot: int) -> str:
    return "left" if slot == 0 else "right"
