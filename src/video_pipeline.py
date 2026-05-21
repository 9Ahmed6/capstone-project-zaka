from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def inspect_video(video_path: str | Path) -> dict:
    """Read basic metadata before processing a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": frame_count / fps if fps else 0.0,
    }


def extract_frames(video_path: str | Path, frame_stride: int = 1) -> tuple[list[np.ndarray], list[float], float]:
    """Load frames from a video.

    Frames are returned in OpenCV BGR format because MediaPipe and clip export
    code can work directly from this format.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % frame_stride == 0:
            frames.append(frame)
            timestamps.append(frame_index / fps)

        frame_index += 1

    cap.release()
    return frames, timestamps, fps


def compute_motion_signal(keypoints: np.ndarray) -> np.ndarray:
    """Convert hand landmarks into one motion number per frame."""
    points = np.asarray(keypoints, dtype=np.float32)
    motion = np.zeros(points.shape[0], dtype=np.float32)

    for i in range(1, points.shape[0]):
        current = points[i]
        previous = points[i - 1]

        # Ignore hands that were not detected in either frame.
        active = (np.linalg.norm(current, axis=(-1, -2)) > 1e-8) & (
            np.linalg.norm(previous, axis=(-1, -2)) > 1e-8
        )
        if np.any(active):
            motion[i] = np.linalg.norm(current[active] - previous[active], axis=-1).mean()

    return motion


def segment_motion_chunks(
    motion: np.ndarray,
    timestamps: list[float],
    start_threshold: float = 0.02,
    end_threshold: float = 0.01,
    min_frames: int = 10,
) -> list[dict]:
    """Find action chunks from a motion signal.

    This is the same simple threshold idea used in the notebook: motion above
    the start threshold begins a chunk, and motion below the end threshold ends it.
    """
    chunks: list[dict] = []
    start_frame: int | None = None

    for frame_index, value in enumerate(motion):
        if start_frame is None and value > start_threshold:
            start_frame = frame_index
        elif start_frame is not None and value < end_threshold:
            if frame_index - start_frame >= min_frames:
                chunks.append(_make_chunk(len(chunks), start_frame, frame_index, timestamps))
            start_frame = None

    if start_frame is not None and len(motion) - start_frame >= min_frames:
        chunks.append(_make_chunk(len(chunks), start_frame, len(motion) - 1, timestamps))

    return chunks


def _make_chunk(chunk_number: int, start_frame: int, end_frame: int, timestamps: list[float]) -> dict:
    return {
        "chunk_id": f"chunk_{chunk_number:03d}",
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "start_time": float(timestamps[start_frame]),
        "end_time": float(timestamps[min(end_frame, len(timestamps) - 1)]),
    }

