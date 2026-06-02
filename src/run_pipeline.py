from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import yaml

from src.exporter import build_video_output, save_video_clip_from_video, write_json
from src.rag_annotator import RAGAnnotator
from src.hand_detection import (
    extract_keypoints_handed_from_video,
    fill_missing_hand_tracks,
    mediapipe_to_handx_order,
    save_keypoints,
)
from src.handx_features import extract_handx_motion_features
from src.video_pipeline import compute_motion_signal, segment_dense_chunks, segment_motion_chunks


def run(
    video_path: str,
    settings_path: str = "configs/settings.yaml",
    annotation_mode: str = "prompt",
    progress_callback: Callable[[str, float, str], None] | None = None,
    chunking_mode: str = "motion",
    dense_window_sec: float | None = None,
    dense_overlap_sec: float | None = None,
) -> dict:
    """
    Run video understanding pipeline.

    Args:
        video_path: Path to video file
        settings_path: Path to settings YAML
        annotation_mode: 'prompt' (RAG-augmented Qwen-VL) or 'rag' (retrieval only)
        chunking_mode: 'motion' for threshold chunks or 'dense' for overlapping windows

    Returns:
        Output dict with segments annotated in chosen mode(s)
    """
    def report(stage: str, progress: float, message: str) -> None:
        if progress_callback:
            progress_callback(stage, min(max(progress, 0.0), 1.0), message)

    settings = yaml.safe_load(Path(settings_path).read_text(encoding="utf-8"))
    if annotation_mode not in ("prompt", "rag"):
        raise ValueError("annotation_mode must be 'prompt' or 'rag'")
    if chunking_mode not in ("motion", "dense"):
        raise ValueError("chunking_mode must be 'motion' or 'dense'")

    QwenVLAnnotator = None
    if annotation_mode == "prompt":
        try:
            from src.fusion import QwenVLAnnotator
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                f"Prompt mode requires the missing Python package '{exc.name}'. "
                "Install project dependencies with: python -m pip install -r requirements.txt"
            ) from exc

    video_path_obj = Path(video_path)
    video_id = video_path_obj.stem
    rag_settings = settings.get("rag", {})
    rag_top_k = int(rag_settings.get("top_k", 3))

    report("preprocessing", 0.02, "Starting hand landmark detection")

    def report_frame_progress(processed_frames: int, total_frames: int | None) -> None:
        if total_frames:
            progress = 0.05 + 0.50 * min(processed_frames / total_frames, 1.0)
            detail = f"Detecting hands: {processed_frames}/{total_frames} frames"
        else:
            progress = 0.30
            detail = f"Detecting hands: {processed_frames} frames"
        report("preprocessing", progress, detail)

    keypoints, timestamps, fps, frame_numbers = extract_keypoints_handed_from_video(
        video_path_obj,
        frame_stride=settings["video"]["frame_stride"],
        max_num_hands=settings["hand_detection"]["max_num_hands"],
        min_detection_confidence=settings["hand_detection"]["min_detection_confidence"],
        min_hand_presence_confidence=settings["hand_detection"]["min_hand_presence_confidence"],
        min_tracking_confidence=settings["hand_detection"]["min_tracking_confidence"],
        model_path=settings["hand_detection"]["model_path"],
        auto_download_model=settings["hand_detection"]["auto_download_model"],
        progress_callback=report_frame_progress,
    )
    report("preprocessing", 0.58, "Preparing hand tracks")
    handx_keypoints, active_slots = fill_missing_hand_tracks(mediapipe_to_handx_order(keypoints))
    save_keypoints(Path("outputs/json") / f"{video_id}_handx_keypoints.npy", handx_keypoints)

    report("chunking", 0.62, "Creating analysis segments")
    if chunking_mode == "dense":
        chunks = segment_dense_chunks(
            timestamps,
            frame_numbers=frame_numbers,
            window_sec=dense_window_sec
            if dense_window_sec is not None
            else settings["chunking"].get("dense_window_sec", 4.0),
            overlap_sec=dense_overlap_sec
            if dense_overlap_sec is not None
            else settings["chunking"].get("dense_overlap_sec", 1.0),
        )
    else:
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

    if annotation_mode == "prompt":
        report("model", 0.66, "Loading Qwen-VL model")
        prompt_annotator = QwenVLAnnotator(
            model_id=settings["qwen"]["model_id"],
            temperature=settings["qwen"]["temperature"],
            max_new_tokens_text=settings["qwen"].get("max_new_tokens_text", 384),
            max_new_tokens_vision=settings["qwen"].get("max_new_tokens_vision", 768),
        )

    if annotation_mode in ("rag", "prompt"):
        report("retrieval", 0.69, "Loading action library")
        rag_annotator = RAGAnnotator(
            action_library_path=rag_settings.get("confirmed_actions_path"),
        )

    segments = []
    chunk_count = len(chunks)
    if not chunks:
        report("annotation", 0.95, "No motion segments found")

    for chunk_index, chunk in enumerate(chunks):
        report(
            "annotation",
            0.70 + 0.25 * chunk_index / max(chunk_count, 1),
            f"Analyzing segment {chunk_index + 1}/{chunk_count}",
        )
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

    report("export", 0.97, "Writing JSON output")
    output = build_video_output(video_id, segments)

    suffix = "_segments.json" if annotation_mode == "prompt" else "_segments_rag.json"

    output_path = Path(settings["paths"]["output_json_dir"]) / f"{video_id}{suffix}"
    write_json(output_path, output)
    report("complete", 1.0, f"Analysis complete: {len(segments)} segments")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Week 5 video understanding pipeline.")
    parser.add_argument("video_path", help="Path to an MP4/AVI video file.")
    parser.add_argument("--settings", default="configs/settings.yaml", help="Path to settings YAML.")
    parser.add_argument(
        "--annotation-mode",
        choices=("prompt", "rag"),
        default="prompt",
        help="prompt: RAG-augmented Qwen-VL; rag: retrieval only",
    )
    parser.add_argument(
        "--chunking-mode",
        choices=("motion", "dense"),
        default="motion",
        help="motion: threshold-based segments; dense: overlapping full-video windows",
    )
    parser.add_argument("--dense-window-sec", type=float, default=None, help="Dense analysis window duration.")
    parser.add_argument("--dense-overlap-sec", type=float, default=None, help="Dense analysis window overlap.")
    args = parser.parse_args()

    output = run(
        args.video_path,
        args.settings,
        annotation_mode=args.annotation_mode,
        chunking_mode=args.chunking_mode,
        dense_window_sec=args.dense_window_sec,
        dense_overlap_sec=args.dense_overlap_sec,
    )
    print(json.dumps({"video_id": output["video_id"], "segments": len(output["segments"])}, indent=2))


if __name__ == "__main__":
    main()
