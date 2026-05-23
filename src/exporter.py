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


def save_video_clip_from_video(
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    output_path: str | Path,
) -> None:
    """Save a clip by rereading only the needed frame range from the video."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))

    writer = None
    current_frame = int(start_frame)

    try:
        while current_frame <= int(end_frame):
            ok, frame = cap.read()
            if not ok:
                break

            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(output_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )

            writer.write(frame)
            current_frame += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def build_video_output(video_id: str, segments: list[dict]) -> dict:
    """Create the final JSON shape required by the schema."""
    return {
        "video_id": video_id,
        "segments": segments,
    }
