from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.exporter import build_video_output, save_video_clip, write_json
from src.fusion import QwenVLAnnotator
from src.hand_detection import (
    extract_keypoints_handed,
    fill_missing_hand_tracks,
    mediapipe_to_handx_order,
    save_keypoints,
)
from src.handx_features import extract_handx_motion_features
from src.video_pipeline import compute_motion_signal, extract_frames, segment_motion_chunks


def run(video_path: str, settings_path: str = "configs/settings.yaml") -> dict:
    settings = yaml.safe_load(Path(settings_path).read_text(encoding="utf-8"))
    video_path_obj = Path(video_path)
    video_id = video_path_obj.stem

    frames, timestamps, fps = extract_frames(
        video_path_obj,
        frame_stride=settings["video"]["frame_stride"],
    )

    keypoints = extract_keypoints_handed(
        frames,
        max_num_hands=settings["hand_detection"]["max_num_hands"],
        min_detection_confidence=settings["hand_detection"]["min_detection_confidence"],
        min_hand_presence_confidence=settings["hand_detection"]["min_hand_presence_confidence"],
        min_tracking_confidence=settings["hand_detection"]["min_tracking_confidence"],
        model_path=settings["hand_detection"]["model_path"],
        auto_download_model=settings["hand_detection"]["auto_download_model"],
    )
    handx_keypoints, active_slots = fill_missing_hand_tracks(mediapipe_to_handx_order(keypoints))
    save_keypoints(Path("outputs/json") / f"{video_id}_handx_keypoints.npy", handx_keypoints)

    motion = compute_motion_signal(handx_keypoints)
    chunks = segment_motion_chunks(
        motion,
        timestamps,
        start_threshold=settings["chunking"]["start_threshold"],
        end_threshold=settings["chunking"]["end_threshold"],
        min_frames=settings["chunking"]["min_frames"],
    )

    action_library = json.loads(Path(settings["paths"]["action_library"]).read_text(encoding="utf-8"))
    annotator = QwenVLAnnotator(
        model_id=settings["qwen"]["model_id"],
        temperature=settings["qwen"]["temperature"],
    )

    segments = []
    for chunk in chunks:
        clip_keypoints = handx_keypoints[chunk["start_frame"] : chunk["end_frame"] + 1]
        motion_clip = clip_keypoints if len(active_slots) > 1 else clip_keypoints[:, active_slots[0]]
        features = extract_handx_motion_features(
            motion_clip,
            active_slots,
            handx_diffusion_path=settings["handx"]["diffusion_path"],
        )

        first_pass, first_raw = annotator.annotate_from_text(features, action_library)
        refined, refined_raw = annotator.refine_with_frames(
            first_pass,
            frames,
            timestamps,
            chunk,
            max_frames=settings["video"]["max_frames_for_qwen"],
        )

        segment = {
            **chunk,
            "action_label": refined.get("action_label", "unknown"),
            "movement_scale": refined.get("movement_scale", "unknown"),
            "confidence": float(refined.get("confidence", 0.0)),
            "hand_side": refined.get("hand_side", "unknown"),
            "summary": refined.get("summary", ""),
            "evidence": refined.get("evidence", ""),
            "first_pass_annotation": first_pass,
            "features": features,
            "raw_model_output": refined_raw,
        }
        segments.append(segment)

        clip_path = Path(settings["paths"]["output_clip_dir"]) / f"{video_id}_{chunk['chunk_id']}.mp4"
        save_video_clip(frames, chunk["start_frame"], chunk["end_frame"], fps, clip_path)

    output = build_video_output(video_id, segments)
    output_path = Path(settings["paths"]["output_json_dir"]) / f"{video_id}_segments.json"
    write_json(output_path, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Week 5 video understanding pipeline.")
    parser.add_argument("video_path", help="Path to an MP4/AVI video file.")
    parser.add_argument("--settings", default="configs/settings.yaml", help="Path to settings YAML.")
    args = parser.parse_args()

    output = run(args.video_path, args.settings)
    print(json.dumps({"video_id": output["video_id"], "segments": len(output["segments"])}, indent=2))


if __name__ == "__main__":
    main()
