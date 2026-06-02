"""
RAG-based Annotator for Action Recognition

Provides an alternative to the prompt-based (Qwen-VL) approach by using
Retrieval-Augmented Generation with the action library.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

from rag.retriever import (
    ActionLibraryRetriever,
    ActionMatch,
    HandXFeatures,
    create_handx_features_from_chunk,
)


class RAGAnnotator:
    """
    RAG-based action annotator using kinematic feature similarity to action library.
    
    This approach is:
    - Faster than VLM inference (no model loading)
    - Fully deterministic (same features → same output)
    - Interpretable (exact matching criteria visible)
    - Grounded in biomechanics (HandX kinematic signals)
    """
    
    def __init__(self, action_library_path: Optional[Path | str] = None):
        """
        Initialize RAG annotator with confirmed_actions.json.
        
        Args:
            action_library_path: Path to confirmed_actions.json.
                                Defaults to rag/action_dictionary/confirmed_actions.json.
        """
        path = Path(action_library_path) if action_library_path else None
        self.retriever = ActionLibraryRetriever(path)
    
    def annotate_chunk(
        self,
        features: Dict[str, Any],
        chunk: Dict[str, Any],
        top_k: int = 6,
        verbose: bool = False,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Annotate a motion chunk using RAG retrieval.
        
        Args:
            features: HandX motion features extracted from chunk
            chunk: Chunk metadata (timestamps, indices, etc.)
            top_k: Number of candidate actions to retrieve
            verbose: Print debug information
        
        Returns:
            Tuple of:
            - refined: Best matching action label (RAG output format)
            - retrieval_results: Full list of top-k matches with scores
            - rag_context: Structured context for Qwen-VL prompts
        """
        matches = self.retrieve_matches(features, chunk, top_k=top_k, verbose=verbose)
        
        # Format output as JSON with "refined" action (best match)
        if matches:
            best_match = matches[0]
            refined = {
                "action_label": best_match.label,
                "action_id": best_match.action_id,
                "movement_scale": best_match.scale,
                "hand_side": best_match.hand,
                "confidence": best_match.confidence,
                "summary": best_match.description,
                "evidence": self._format_evidence(best_match.evidence_scores),
            }
        else:
            refined = {
                "action_label": "unknown",
                "action_id": "UNKNOWN",
                "movement_scale": "unknown",
                "hand_side": "unknown",
                "confidence": 0.0,
                "summary": "No matching action found in library",
                "evidence": "Retrieval returned no candidates",
            }
        
        # Format retrieval results for transparency
        retrieval_results = [
            {
                "rank": i + 1,
                "action_id": match.action_id,
                "label": match.label,
                "confidence": match.confidence,
                "scale": match.scale,
                "hand": match.hand,
                "evidence": match.evidence_scores,
            }
            for i, match in enumerate(matches)
        ]
        
        rag_context = self.format_prompt_context(matches, refined=refined, max_candidates=top_k)
        return refined, retrieval_results, rag_context

    def retrieve_matches(
        self,
        features: Dict[str, Any],
        chunk: Dict[str, Any],
        top_k: int = 6,
        verbose: bool = False,
    ) -> List[ActionMatch]:
        """Run retrieval only and return ActionMatch objects."""
        handx_features = create_handx_features_from_chunk(chunk, features)
        return self.retriever.retrieve(
            handx_features,
            top_k=top_k,
            confidence_threshold=0.0,
            verbose=verbose,
        )

    def format_prompt_context(
        self,
        matches: List[ActionMatch],
        refined: Optional[Dict[str, Any]] = None,
        max_candidates: int = 3,
    ) -> Dict[str, Any]:
        """
        Build structured RAG context for Qwen-VL prompts.

        Returns dict with prompt_text (for injection) and serializable metadata.
        """
        prompt_text = ActionLibraryRetriever.format_matches_for_prompt(
            matches,
            max_candidates=max_candidates,
        )
        return {
            "prompt_text": prompt_text,
            "best_match": refined or {},
            "candidates": [
                {
                    "rank": index + 1,
                    "action_id": match.action_id,
                    "label": match.label,
                    "confidence": match.confidence,
                    "scale": match.scale,
                    "hand": match.hand,
                    "description": match.description,
                    "kinematic_signal": match.kinematic_signal,
                    "evidence": match.evidence_scores,
                }
                for index, match in enumerate(matches[:max_candidates])
            ],
        }
    
    def _format_evidence(self, evidence_scores: Dict[str, float]) -> str:
        """Format evidence scores into human-readable string."""
        items = []
        for key, score in evidence_scores.items():
            # Format key name
            key_display = key.replace('_', ' ').title()
            items.append(f"{key_display}: {score:.2f}")
        
        return " | ".join(items)
    
    def annotate_chunks_batch(
        self,
        chunks: List[Dict[str, Any]],
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Annotate multiple chunks in batch.
        
        Args:
            chunks: List of chunk dicts with 'features' key
            verbose: Print progress
        
        Returns:
            List of annotated chunks with 'rag_annotation' and 'rag_retrieval' keys
        """
        results = []
        
        for i, chunk in enumerate(chunks):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Annotating chunk {i + 1}/{len(chunks)}")
            
            features = chunk.get('features', {})
            refined, retrieval_results, _rag_context = self.annotate_chunk(
                features,
                chunk,
                verbose=False,
            )
            
            annotated_chunk = {
                **chunk,
                "rag_annotation": refined,
                "rag_retrieval": retrieval_results,
            }
            results.append(annotated_chunk)
        
        return results
    
    def compare_with_prompt_baseline(
        self,
        rag_annotations: List[Dict[str, Any]],
        prompt_annotations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compare RAG results with prompt-based (Qwen-VL) baseline.
        
        Args:
            rag_annotations: Output from annotate_chunks_batch()
            prompt_annotations: Original prompt-based annotations from pipeline
        
        Returns:
            Comparison metrics and agreement analysis
        """
        if len(rag_annotations) != len(prompt_annotations):
            raise ValueError("Annotation lists must have same length")
        
        agreements = {
            'exact_label': 0,
            'same_scale': 0,
            'same_hand': 0,
            'total': len(rag_annotations),
        }
        
        detailed_comparisons = []
        
        for rag, prompt in zip(rag_annotations, prompt_annotations):
            rag_annotation = rag['rag_annotation']
            prompt_annotation = prompt
            
            # Extract labels and metadata
            rag_label = rag_annotation.get('action_label', 'unknown')
            prompt_label = prompt_annotation.get('action_label', 'unknown')
            
            rag_scale = rag_annotation.get('movement_scale', 'unknown')
            prompt_scale = prompt_annotation.get('movement_scale', 'unknown')
            
            rag_hand = rag_annotation.get('hand_side', 'unknown')
            prompt_hand = prompt_annotation.get('hand_side', 'unknown')
            
            rag_conf = rag_annotation.get('confidence', 0.0)
            prompt_conf = prompt_annotation.get('confidence', 0.0)
            
            # Count agreements
            if rag_label == prompt_label:
                agreements['exact_label'] += 1
            if rag_scale == prompt_scale:
                agreements['same_scale'] += 1
            if rag_hand == prompt_hand:
                agreements['same_hand'] += 1
            
            # Store detailed comparison
            comparison = {
                'chunk_id': rag.get('chunk_id', '?'),
                'rag_label': rag_label,
                'prompt_label': prompt_label,
                'label_match': rag_label == prompt_label,
                'rag_scale': rag_scale,
                'prompt_scale': prompt_scale,
                'scale_match': rag_scale == prompt_scale,
                'rag_hand': rag_hand,
                'prompt_hand': prompt_hand,
                'hand_match': rag_hand == prompt_hand,
                'rag_confidence': rag_conf,
                'prompt_confidence': prompt_conf,
                'rag_top_alternatives': [
                    f"{r['label']} ({r['confidence']:.2f})"
                    for r in rag.get('rag_retrieval', [])[1:4]
                ],
            }
            detailed_comparisons.append(comparison)
        
        # Compute percentages
        summary = {
            'exact_label_agreement': agreements['exact_label'] / agreements['total'] if agreements['total'] > 0 else 0,
            'scale_agreement': agreements['same_scale'] / agreements['total'] if agreements['total'] > 0 else 0,
            'hand_agreement': agreements['same_hand'] / agreements['total'] if agreements['total'] > 0 else 0,
            'total_chunks': agreements['total'],
            'agreement_counts': agreements,
        }
        
        return {
            'summary': summary,
            'detailed': detailed_comparisons,
        }


def annotate_with_rag(
    video_path: str,
    settings_path: str = "configs/settings.yaml",
    output_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Standalone function to annotate a video using only RAG (no Qwen-VL).
    
    This is a lightweight alternative to run_pipeline.run() that avoids
    loading the VLM model entirely.
    
    Args:
        video_path: Path to video file
        settings_path: Path to settings.yaml
        output_json_path: Optional output path (otherwise constructs from video)
    
    Returns:
        Output dict with video_id and segments
    """
    import yaml
    from src.hand_detection import (
        extract_keypoints_handed_from_video,
        fill_missing_hand_tracks,
        mediapipe_to_handx_order,
        save_keypoints,
    )
    from src.handx_features import extract_handx_motion_features
    from src.video_pipeline import compute_motion_signal, segment_motion_chunks
    from src.exporter import build_video_output, save_video_clip_from_video, write_json
    
    # Load settings
    settings = yaml.safe_load(Path(settings_path).read_text(encoding="utf-8"))
    video_path_obj = Path(video_path)
    video_id = video_path_obj.stem
    
    # Hand detection
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
    
    # Motion analysis
    motion = compute_motion_signal(handx_keypoints)
    chunks = segment_motion_chunks(
        motion,
        timestamps,
        frame_numbers=frame_numbers,
        start_threshold=settings["chunking"]["start_threshold"],
        end_threshold=settings["chunking"]["end_threshold"],
        min_frames=settings["chunking"]["min_frames"],
    )
    
    # RAG annotation
    annotator = RAGAnnotator()
    
    segments = []
    for chunk in chunks:
        clip_keypoints = handx_keypoints[chunk["start_index"] : chunk["end_index"] + 1]
        motion_clip = clip_keypoints if len(active_slots) > 1 else clip_keypoints[:, active_slots[0]]
        features = extract_handx_motion_features(
            motion_clip,
            active_slots,
            handx_diffusion_path=settings["handx"]["diffusion_path"],
        )
        
        refined, retrieval_results, _rag_context = annotator.annotate_chunk(
            features,
            chunk,
            top_k=int(settings.get("rag", {}).get("top_k", 6)),
            verbose=False,
        )
        
        segment = {
            **chunk,
            "action_label": refined.get("action_label", "unknown"),
            "movement_scale": refined.get("movement_scale", "unknown"),
            "confidence": float(refined.get("confidence", 0.0)),
            "hand_side": refined.get("hand_side", "unknown"),
            "summary": refined.get("summary", ""),
            "evidence": refined.get("evidence", ""),
            "features": features,
            "rag_retrieval": retrieval_results,
        }
        segments.append(segment)
        
        # Save clip
        clip_path = Path(settings["paths"]["output_clip_dir"]) / f"{video_id}_{chunk['chunk_id']}.mp4"
        save_video_clip_from_video(video_path_obj, chunk["start_frame"], chunk["end_frame"], clip_path)
    
    output = build_video_output(video_id, segments)
    
    if output_json_path is None:
        output_json_path = Path(settings["paths"]["output_json_dir"]) / f"{video_id}_segments_rag.json"
    
    write_json(output_json_path, output)
    return output
