from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def write_json(path: str | Path, data: dict) -> None:
    """Write a pretty JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_video_clip(
    frames: list[np.ndarray],
    start_frame: int,
    end_frame: int,
    fps: float,
    output_path: str | Path,
) -> None:
    """Save one chunk of frames as an MP4 clip."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_frames = frames[start_frame : end_frame + 1]
    if not selected_frames:
        return

    height, width = selected_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    try:
        for frame in selected_frames:
            writer.write(frame)
    finally:
        writer.release()


def build_video_output(video_id: str, segments: list[dict]) -> dict:
    """Create the final JSON shape required by the schema."""
    return {
        "video_id": video_id,
        "segments": segments,
    }

