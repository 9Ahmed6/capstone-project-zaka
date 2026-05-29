from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.exporter import build_video_output, save_video_clip_from_video, write_json
from src.fusion import QwenVLAnnotator
from src.rag_annotator import RAGAnnotator
from src.hand_detection import (
    extract_keypoints_handed_from_video,
    fill_missing_hand_tracks,
    mediapipe_to_handx_order,
    save_keypoints,
)
from src.handx_features import extract_handx_motion_features
from src.video_pipeline import compute_motion_signal, segment_motion_chunks


def run(
    video_path: str,
    settings_path: str = "configs/settings.yaml",
    annotation_mode: str = "prompt",
) -> dict:
    """
    Run video understanding pipeline.

    Args:
        video_path: Path to video file
        settings_path: Path to settings YAML
        annotation_mode: 'prompt' (RAG-augmented Qwen-VL), 'rag' (retrieval only), or 'both'

    Returns:
        Output dict with segments annotated in chosen mode(s)
    """
    settings = yaml.safe_load(Path(settings_path).read_text(encoding="utf-8"))
    video_path_obj = Path(video_path)
    video_id = video_path_obj.stem
    rag_settings = settings.get("rag", {})
    rag_top_k = int(rag_settings.get("top_k", 3))

    keypoints, timestamps, fps, frame_numbers = extract_keypoints_handed_from_video(
        video_path_obj,
        frame_stride=settings["video"]["frame_stride"],
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
        frame_numbers=frame_numbers,
        start_threshold=settings["chunking"]["start_threshold"],
        end_threshold=settings["chunking"]["end_threshold"],
        min_frames=settings["chunking"]["min_frames"],
    )

    prompt_annotator = None
    rag_annotator = None

    if annotation_mode in ("prompt", "both"):
        prompt_annotator = QwenVLAnnotator(
            model_id=settings["qwen"]["model_id"],
            temperature=settings["qwen"]["temperature"],
            max_new_tokens_text=settings["qwen"].get("max_new_tokens_text", 384),
            max_new_tokens_vision=settings["qwen"].get("max_new_tokens_vision", 768),
        )

    if annotation_mode in ("rag", "both", "prompt"):
        rag_annotator = RAGAnnotator(
            action_library_path=rag_settings.get("confirmed_actions_path"),
        )

    segments = []
    for chunk in chunks:
        clip_keypoints = handx_keypoints[chunk["start_index"] : chunk["end_index"] + 1]
        motion_clip = clip_keypoints if len(active_slots) > 1 else clip_keypoints[:, active_slots[0]]
        features = extract_handx_motion_features(
            motion_clip,
            active_slots,
            handx_diffusion_path=settings["handx"]["diffusion_path"],
        )

        segment = {**chunk, "features": features}
        rag_context = None

        if rag_annotator:
            rag_refined, rag_retrieval, rag_context = rag_annotator.annotate_chunk(
                features,
                chunk,
                top_k=rag_top_k,
                verbose=False,
            )

            segment.update({
                "rag_annotation": rag_refined,
                "rag_retrieval": rag_retrieval,
            })

            if not prompt_annotator:
                segment.update({
                    "action_label": rag_refined.get("action_label", "unknown"),
                    "movement_scale": rag_refined.get("movement_scale", "unknown"),
                    "confidence": float(rag_refined.get("confidence", 0.0)),
                    "hand_side": rag_refined.get("hand_side", "unknown"),
                    "summary": rag_refined.get("summary", ""),
                    "evidence": rag_refined.get("evidence", ""),
                })

        if prompt_annotator:
            if not rag_context:
                raise RuntimeError(
                    "Qwen-VL annotation requires RAG context from confirmed_actions.json. "
                    "Ensure RAG retrieval ran successfully for this chunk."
                )
            refined, raw_model_output = prompt_annotator.annotate_chunk(
                features,
                video_path_obj,
                chunk,
                rag_context=rag_context,
                max_frames=settings["video"]["max_frames_for_qwen"],
            )

            segment.update({
                "action_label": refined.get("action_label", "unknown"),
                "movement_scale": refined.get("movement_scale", "unknown"),
                "confidence": float(refined.get("confidence", 0.0)),
                "hand_side": refined.get("hand_side", "unknown"),
                "summary": refined.get("summary", ""),
                "evidence": refined.get("evidence", ""),
                "raw_model_output": raw_model_output,
            })

        segments.append(segment)

        clip_path = Path(settings["paths"]["output_clip_dir"]) / f"{video_id}_{chunk['chunk_id']}.mp4"
        save_video_clip_from_video(video_path_obj, chunk["start_frame"], chunk["end_frame"], clip_path)

    output = build_video_output(video_id, segments)

    if annotation_mode == "prompt":
        suffix = "_segments.json"
    elif annotation_mode == "rag":
        suffix = "_segments_rag.json"
    else:
        suffix = "_segments_both.json"

    output_path = Path(settings["paths"]["output_json_dir"]) / f"{video_id}{suffix}"
    write_json(output_path, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Week 5 video understanding pipeline.")
    parser.add_argument("video_path", help="Path to an MP4/AVI video file.")
    parser.add_argument("--settings", default="configs/settings.yaml", help="Path to settings YAML.")
    parser.add_argument(
        "--annotation-mode",
        choices=("prompt", "rag", "both"),
        default="prompt",
        help="prompt: RAG-augmented Qwen-VL; rag: retrieval only; both: store both outputs",
    )
    args = parser.parse_args()

    output = run(args.video_path, args.settings, annotation_mode=args.annotation_mode)
    print(json.dumps({"video_id": output["video_id"], "segments": len(output["segments"])}, indent=2))


if __name__ == "__main__":
    main()
