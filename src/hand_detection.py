from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm


HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


# HandX uses a different joint order than MediaPipe.
HANDX_FROM_MEDIAPIPE = [
    0,
    5,
    6,
    7,
    9,
    10,
    11,
    17,
    18,
    19,
    13,
    14,
    15,
    1,
    2,
    3,
    4,
    8,
    12,
    16,
    20,
]


def extract_keypoints_handed(
    frames: list[np.ndarray],
    max_num_hands: int = 2,
    min_detection_confidence: float = 0.5,
    min_hand_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    model_path: str | Path = "models/hand_landmarker.task",
    auto_download_model: bool = True,
) -> np.ndarray:
    """Detect hand landmarks with the newer MediaPipe Tasks API.

    Output shape is (frames, 2, 21, 3).
    Slot 0 is left hand. Slot 1 is right hand.
    """
    model_path = ensure_hand_landmarker_model(model_path, auto_download_model)
    keypoints = np.zeros((len(frames), 2, 21, 3), dtype=np.float32)
    slot_for_label = {"Left": 0, "Right": 1}

    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=max_num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_hand_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    detector = vision.HandLandmarker.create_from_options(options)

    try:
        for frame_index, frame in enumerate(tqdm(frames, desc="Detecting hands")):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(image)

            if not result.hand_landmarks:
                continue

            used_slots: set[int] = set()
            for hand_index, hand_landmarks in enumerate(result.hand_landmarks[:2]):
                label = None
                if result.handedness and hand_index < len(result.handedness):
                    if result.handedness[hand_index]:
                        label = result.handedness[hand_index][0].category_name

                slot = slot_for_label.get(label)
                if slot is None or slot in used_slots:
                    slot = 0 if 0 not in used_slots else 1

                used_slots.add(slot)
                for joint_index, landmark in enumerate(hand_landmarks):
                    keypoints[frame_index, slot, joint_index] = [
                        landmark.x,
                        landmark.y,
                        landmark.z,
                    ]
    finally:
        detector.close()

    return keypoints


def ensure_hand_landmarker_model(model_path: str | Path, auto_download: bool = True) -> Path:
    """Make sure the MediaPipe Tasks hand model exists."""
    path = Path(model_path)
    if path.exists():
        return path

    if not auto_download:
        raise FileNotFoundError(
            f"Missing MediaPipe hand model: {path}. "
            "Download hand_landmarker.task or set auto_download_model to true."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading MediaPipe hand model to {path}...")
    urlretrieve(HAND_LANDMARKER_URL, path)
    return path


def mediapipe_to_handx_order(keypoints: np.ndarray) -> np.ndarray:
    """Reorder MediaPipe landmarks so HandX can read them."""
    keypoints = np.asarray(keypoints, dtype=np.float32)
    if keypoints.shape[-2:] != (21, 3):
        raise ValueError(f"Expected (..., 21, 3), got {keypoints.shape}")
    return keypoints[..., HANDX_FROM_MEDIAPIPE, :]


def fill_missing_hand_tracks(handx_keypoints: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Interpolate short gaps where a hand disappears for a few frames."""
    fixed = np.asarray(handx_keypoints, dtype=np.float32).copy()
    frame_count = fixed.shape[0]
    frame_numbers = np.arange(frame_count)
    active_slots: list[int] = []

    for hand_slot in range(fixed.shape[1]):
        valid = np.linalg.norm(fixed[:, hand_slot], axis=(1, 2)) > 1e-8
        if valid.sum() == 0:
            continue

        active_slots.append(hand_slot)
        for joint_index in range(21):
            for coord_index in range(3):
                fixed[:, hand_slot, joint_index, coord_index] = np.interp(
                    frame_numbers,
                    frame_numbers[valid],
                    fixed[valid, hand_slot, joint_index, coord_index],
                )

    if not active_slots:
        raise ValueError("No hands were detected in the video.")

    return fixed, active_slots


def save_keypoints(path: str | Path, keypoints: np.ndarray) -> None:
    """Save landmarks so later experiments can run without re-detecting hands."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.save(path, keypoints)
