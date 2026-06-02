"""
RAG Retriever for HandX-based Action Matching

This module implements Retrieval-Augmented Generation for action recognition
by matching HandX kinematic features to a curated action library.

Key features:
- Embedding-based similarity search for action labels
- Kinematic feature matching (contact ratio, hand side, scale)
- Confidence scoring based on multi-modal similarity
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')


@dataclass
class HandXFeatures:
    """Container for HandX kinematic features extracted from motion sequence."""
    contact_ratio: float  # [0, 1]
    hand_side: str  # 'left', 'right', 'both'
    hand_sides_detected: List[str]  # ['left'] or ['right'] or ['left', 'right']
    contact_frequency: float = 0.0  # Hz
    avg_contact_duration: float = 0.0  # seconds
    wrist_velocity: float = 0.0  # m/s
    finger_flexion_variance: float = 0.0  # flexion ratio variance
    finger_transition_count: int = 0
    wrist_motion_event_count: int = 0
    primary_joints: List[int] = None  # MANO joint indices
    description: str = ""  # Optional user description


@dataclass
class ActionMatch:
    """Result from RAG retrieval for a single action query."""
    action_id: str
    label: str
    confidence: float  # [0, 1]
    scale: str  # micro, macro, bimanual
    hand: str  # left, right, either, both
    description: str
    contact_ratio_range: List[float]
    kinematic_signal: str
    evidence_scores: Dict[str, float]  # Breakdown of confidence components


class ActionLibraryRetriever:
    """
    RAG-based retriever for matching HandX kinematic features to action labels.
    
    Uses multi-modal similarity matching:
    1. Kinematic signal semantic similarity (TF-IDF + cosine distance)
    2. Contact ratio numerical matching
    3. Hand side compatibility
    4. Movement scale classification
    """
    
    def __init__(self, action_library_path: Optional[Path] = None):
        """
        Initialize the retriever with the action library.
        
        Args:
            action_library_path: Path to confirmed_actions.json. 
                                Defaults to project standard location.
        """
        if action_library_path is None:
            # Default to project structure
            action_library_path = (
                Path(__file__).parent / 'action_dictionary' / 'confirmed_actions.json'
            )
        
        self.library_path = Path(action_library_path)
        if not self.library_path.exists():
            raise FileNotFoundError(f"Action library not found at {self.library_path}")
        
        self._load_library()
        self._build_embeddings()
    
    def _load_library(self) -> None:
        """Load action library from JSON file."""
        with open(self.library_path, 'r') as f:
            data = json.load(f)
        
        self.library_metadata = {
            'version': data.get('version', '1.0'),
            'description': data.get('description', ''),
            'schema_notes': data.get('schema_notes', {}),
        }
        
        self.actions = data['action_library']
        
        # Index by ID for fast lookup
        self.action_index = {action['id']: action for action in self.actions}
        
        # Group by scale for faster filtering
        self.by_scale = {}
        for action in self.actions:
            scale = action['scale']
            if scale not in self.by_scale:
                self.by_scale[scale] = []
            self.by_scale[scale].append(action)
        
        print(f"Loaded {len(self.actions)} actions from {self.library_path.name}")
        print(f"  Scales: {', '.join(self.by_scale.keys())}")
    
    def _build_embeddings(self) -> None:
        """Build TF-IDF embeddings for kinematic signal descriptions."""
        # Extract kinematic signal texts
        self.kinematic_texts = [
            action['handx_kinematic_signal'] for action in self.actions
        ]
        
        # Build TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(1, 3),
            min_df=1,
            max_df=len(self.actions),
            lowercase=True,
            stop_words=None,
        )
        
        self.kinematic_embeddings = self.vectorizer.fit_transform(self.kinematic_texts)
        
        print(f"Built TF-IDF embeddings ({self.kinematic_embeddings.shape[1]} features)")
    
    def _score_contact_ratio_match(
        self, 
        measured_ratio: float, 
        action: Dict[str, Any]
    ) -> float:
        """
        Score how well the measured contact ratio matches the action's range.
        
        Returns: confidence [0, 1] where 1 is perfect match
        """
        min_ratio, max_ratio = action['contact_ratio_range']
        
        # If measurement is outside range, penalize but don't zero
        if measured_ratio < min_ratio:
            # Linear penalty: 0 at 0, reaching score at min_ratio
            return max(0, measured_ratio / min_ratio) * 0.8 if min_ratio > 0 else 0.5
        elif measured_ratio > max_ratio:
            # Linear penalty: 1 at max_ratio, reaching lower value beyond
            return 0.8 if measured_ratio < max_ratio + 0.2 else 0.5
        else:
            # Perfect match within range
            return 1.0
    
    def _score_hand_compatibility(
        self,
        hands_detected: List[str],
        action_hand: str
    ) -> float:
        """
        Score how well detected hands match action requirements.
        
        Args:
            hands_detected: List of detected hands ['left'], ['right'], or ['left', 'right']
            action_hand: Required hand for action: 'left', 'right', 'either', 'both'
        
        Returns: confidence [0, 1]
        """
        if action_hand == 'either':
            # Any single hand is compatible
            return 1.0 if len(hands_detected) >= 1 else 0.0
        elif action_hand == 'both':
            # Requires both hands
            return 1.0 if len(hands_detected) == 2 else 0.3
        elif action_hand in hands_detected:
            # Specific hand required and detected
            return 1.0
        else:
            # Specific hand required but not detected
            return 0.0
    
    def _score_scale_threshold(
        self,
        contact_ratio: float,
        hand_side: str,
        action_scale: str
    ) -> float:
        """
        Score based on scale thresholds defined in library.
        
        Scale classification:
        - micro: contact_ratio <= 0.25 (isolated finger articulations)
        - macro: 0.05 <= contact_ratio <= 0.45 (wrist/arm-dominant)
        - bimanual: contact_ratio >= 0.30 AND both hands
        """
        if action_scale == "bimanual":
            return 1.0 if hand_side == "both" and contact_ratio >= 0.30 else 0.2
        if action_scale == "micro":
            return 1.0 if contact_ratio <= 0.25 else 0.2
        if action_scale == "macro":
            return 1.0 if 0.05 <= contact_ratio <= 0.45 else 0.2
        return 0.5
    
    def _score_kinematic_similarity(
        self,
        handx_description: str,
        action: Dict[str, Any],
        use_text: bool = True
    ) -> float:
        """
        Score semantic similarity between measured and expected kinematic signals.
        
        Uses TF-IDF cosine similarity on the kinematic signal descriptions.
        """
        if not use_text or not handx_description:
            return 0.5  # Neutral score if no text provided
        
        # Vectorize the measured features
        measured_vector = self.vectorizer.transform([handx_description])
        
        # Find the corresponding action's embedding
        action_id = action['id']
        action_index = next(
            (i for i, a in enumerate(self.actions) if a['id'] == action_id),
            None
        )
        
        if action_index is None:
            return 0.0
        
        # Compute cosine similarity
        similarity = cosine_similarity(measured_vector, self.kinematic_embeddings[action_index:action_index+1])
        
        # cosine_similarity returns [[score]], extract scalar
        return float(similarity[0, 0]) if similarity.size > 0 else 0.0

    def _score_temporal_compatibility(
        self,
        handx_features: HandXFeatures,
        action: Dict[str, Any],
    ) -> float:
        """Reward action classes whose temporal behavior matches the window."""
        label = action["label"]
        flexion_activity = handx_features.finger_flexion_variance
        wrist_activity = handx_features.wrist_velocity
        has_finger_changes = handx_features.finger_transition_count > 0
        has_wrist_motion = handx_features.wrist_motion_event_count > 0

        if label in ("static_hold_precision", "static_hold_power"):
            if flexion_activity > 0.20 or wrist_activity > 0.05:
                return 0.15
            return 1.0

        if label in ("finger_extension", "finger_flexion", "power_curl", "release", "button_press"):
            return 1.0 if has_finger_changes else 0.2

        if label in ("in_hand_rotation", "dial_rotation"):
            return 1.0 if has_finger_changes else 0.4

        if label in ("reach", "transport", "push", "pull", "wave"):
            return 1.0 if has_wrist_motion else 0.25

        return 0.6
    
    def retrieve(
        self,
        handx_features: HandXFeatures,
        top_k: int = 5,
        scale_filter: Optional[str] = None,
        hand_filter: Optional[str] = None,
        confidence_threshold: float = 0.0,
        verbose: bool = False
    ) -> List[ActionMatch]:
        """
        Retrieve top-K matching actions for given HandX features.
        
        Args:
            handx_features: Extracted kinematic features from motion chunk
            top_k: Number of results to return
            scale_filter: Optional filter by scale ('micro', 'macro', 'bimanual')
            hand_filter: Optional filter by hand ('left', 'right', 'both')
            confidence_threshold: Minimum confidence to include result
            verbose: Print debug information
        
        Returns:
            List of ActionMatch results sorted by confidence (descending)
        """
        candidates = self.actions
        
        # Apply optional filters
        if scale_filter:
            candidates = [a for a in candidates if a['scale'] == scale_filter]
        if hand_filter:
            candidates = [a for a in candidates if a['hand'] in (hand_filter, 'either')]
        
        if not candidates:
            return []
        
        scores = []
        
        for action in candidates:
            # Multi-modal scoring
            evidence = {}
            
            # 1. Contact ratio matching (25% weight)
            contact_score = self._score_contact_ratio_match(
                handx_features.contact_ratio,
                action
            )
            evidence['contact_ratio'] = contact_score
            
            # 2. Hand compatibility (25% weight)
            hand_score = self._score_hand_compatibility(
                handx_features.hand_sides_detected,
                action['hand']
            )
            evidence['hand_compatibility'] = hand_score
            
            # 3. Scale threshold (25% weight)
            scale_score = self._score_scale_threshold(
                handx_features.contact_ratio,
                handx_features.hand_side,
                action['scale']
            )
            evidence['scale_alignment'] = scale_score
            
            # 4. Kinematic similarity (25% weight)
            kinematic_score = self._score_kinematic_similarity(
                handx_features.description,
                action,
                use_text=True
            )
            evidence['kinematic_similarity'] = kinematic_score

            # 5. Temporal compatibility (30% weight)
            temporal_score = self._score_temporal_compatibility(handx_features, action)
            evidence['temporal_compatibility'] = temporal_score
            
            # Temporal evidence is important for separating holds from active
            # manipulation inside fixed analysis windows.
            confidence = (
                contact_score * 0.15 +
                hand_score * 0.10 +
                scale_score * 0.15 +
                kinematic_score * 0.30 +
                temporal_score * 0.30
            )
            
            scores.append({
                'action': action,
                'confidence': confidence,
                'evidence': evidence,
            })
        
        # Sort by confidence
        scores.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Filter by threshold and top_k
        results = []
        for item in scores[:top_k]:
            if item['confidence'] >= confidence_threshold:
                result = ActionMatch(
                    action_id=item['action']['id'],
                    label=item['action']['label'],
                    confidence=item['confidence'],
                    scale=item['action']['scale'],
                    hand=item['action']['hand'],
                    description=item['action']['description'],
                    contact_ratio_range=item['action']['contact_ratio_range'],
                    kinematic_signal=item['action']['handx_kinematic_signal'],
                    evidence_scores=item['evidence'],
                )
                results.append(result)
        
        if verbose and results:
            print(f"\n📊 RAG Retrieval Results (top-{len(results)}):")
            for i, match in enumerate(results, 1):
                print(f"  {i}. {match.label} ({match.action_id}) - {match.confidence:.3f}")
                print(f"     Scale: {match.scale}, Hand: {match.hand}")
                print(f"     Evidence: {', '.join(f'{k}={v:.2f}' for k, v in match.evidence_scores.items())}")
        
        return results
    
    def retrieve_by_description(
        self,
        description: str,
        top_k: int = 5,
        confidence_threshold: float = 0.0
    ) -> List[ActionMatch]:
        """
        Retrieve actions using a text description only (no numerical features).
        
        Useful for quick lookup or when kinematic features are unavailable.
        """
        features = HandXFeatures(
            contact_ratio=0.2,  # Neutral value
            hand_side='both',
            hand_sides_detected=['left', 'right'],
            description=description,
        )
        return self.retrieve(
            features,
            top_k=top_k,
            confidence_threshold=confidence_threshold
        )
    
    def get_action(self, action_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific action by ID."""
        return self.action_index.get(action_id)
    
    def list_actions(
        self,
        scale: Optional[str] = None,
        hand: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List actions with optional filtering."""
        actions = self.actions
        
        if scale:
            actions = [a for a in actions if a['scale'] == scale]
        if hand:
            actions = [a for a in actions if a['hand'] in (hand, 'either')]
        
        return actions
    
    def export_results(
        self,
        matches: List[ActionMatch],
        output_format: str = 'json'
    ) -> str:
        """
        Export retrieval results in specified format.
        
        Args:
            matches: List of ActionMatch results
            output_format: 'json' or 'dict'
        
        Returns:
            Formatted string or dict
        """
        if output_format == 'json':
            data = [asdict(m) for m in matches]
            return json.dumps(data, indent=2)
        elif output_format == 'dict':
            return [asdict(m) for m in matches]
        else:
            raise ValueError(f"Unknown format: {output_format}")

    @staticmethod
    def format_matches_for_prompt(
        matches: List[ActionMatch],
        max_candidates: int = 3,
    ) -> str:
        """Format top RAG matches as text context for Qwen-VL prompts."""
        if not matches:
            return "No RAG candidates retrieved from the action library."

        lines = []
        for index, match in enumerate(matches[:max_candidates], start=1):
            evidence = ", ".join(
                f"{key}={value:.2f}" for key, value in match.evidence_scores.items()
            )
            lines.append(
                f"{index}. {match.label} (id={match.action_id}, confidence={match.confidence:.3f})\n"
                f"   scale={match.scale}, hand={match.hand}, "
                f"contact_ratio_range={match.contact_ratio_range}\n"
                f"   description: {match.description}\n"
                f"   kinematic_signal: {match.kinematic_signal}\n"
                f"   evidence: {evidence}"
            )
        return "\n".join(lines)


def infer_kinematic_features(features: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map pipeline HandX feature JSON to numeric fields used by the retriever.

    Works with both HandX library output and the simple fallback extractor.
    """
    detected = features.get("detected_hands", [])
    if isinstance(detected, str):
        detected = [detected]
    if not detected:
        detected = [
            hand
            for hand in ("left", "right")
            if features.get(f"{hand}_hand_events")
        ]
    hand_sides = [hand for hand in detected if hand in ("left", "right")]
    if not hand_sides and detected:
        hand_sides = list(detected)

    if len(hand_sides) >= 2:
        hand_side = "both"
    elif "left" in hand_sides:
        hand_side = "left"
    elif "right" in hand_sides:
        hand_side = "right"
    else:
        hand_side = "unknown"

    finger_transition_count = _count_finger_transitions(features)
    wrist_motion_event_count = _count_wrist_motion_events(features)
    contact_ratio = float(features.get("contact_ratio", _estimate_contact_ratio(features)))
    wrist_velocity = float(features.get("wrist_velocity", _estimate_wrist_velocity(features)))
    finger_flexion_variance = float(
        features.get("finger_flexion_variance", min(1.0, finger_transition_count / 10.0))
    )

    return {
        "contact_ratio": contact_ratio,
        "hand_side": hand_side,
        "hand_sides_detected": hand_sides,
        "contact_frequency": float(features.get("contact_frequency", 0.0)),
        "avg_contact_duration": float(features.get("avg_contact_duration", 0.0)),
        "wrist_velocity": wrist_velocity,
        "finger_flexion_variance": finger_flexion_variance,
        "finger_transition_count": finger_transition_count,
        "wrist_motion_event_count": wrist_motion_event_count,
        "primary_joints": features.get("primary_joints"),
        "description": features.get("description") or build_kinematic_description(features),
    }


def build_kinematic_description(features: Dict[str, Any]) -> str:
    """Build a short text summary of motion features for TF-IDF retrieval."""
    parts: List[str] = []

    summary = features.get("hand_motion_summary", {})
    for hand_name, stats in summary.items():
        parts.append(
            f"{hand_name} speed={stats.get('mean_center_speed', 0):.4f} "
            f"openness_change={stats.get('openness_change', 0):.4f}"
        )

    relationships = features.get("two_hand_relationships", {})
    if relationships:
        parts.append(
            "two_hand "
            f"mean_distance={relationships.get('mean_distance', 0):.4f} "
            f"distance_change={relationships.get('distance_change', 0):.4f}"
        )

    for hand_name in ("left", "right"):
        events = features.get(f"{hand_name}_hand_events", {})
        if not events:
            continue

        flexing = events.get("finger_flexing", {})
        changed_joints = [
            joint_name
            for joint_name, joint_events in flexing.items()
            if any(_is_transition(event) for event in joint_events)
        ]
        if changed_joints:
            parts.append(f"{hand_name} finger flexion transitions: {', '.join(changed_joints)}")

        wrist_events = events.get("wrist_trajectory", {})
        if wrist_events:
            parts.append(f"{hand_name} wrist trajectory active: {', '.join(wrist_events)}")

        if events.get("finger_tip_contact"):
            parts.append(f"{hand_name} fingertip contact active")

    for key in ("left_hand_events", "right_hand_events", "two_hand_relationships"):
        if key in features and features[key]:
            snippet = json.dumps(features[key], default=str)
            if len(snippet) > 400:
                snippet = snippet[:400] + "..."
            parts.append(f"{key}={snippet}")

    if not parts:
        return f"motion chunk source={features.get('source', 'unknown')}"

    return "; ".join(parts)


def _estimate_contact_ratio(features: Dict[str, Any]) -> float:
    """Heuristic contact ratio from openness and two-hand proximity."""
    ratio = 0.12

    for stats in features.get("hand_motion_summary", {}).values():
        openness_change = float(stats.get("openness_change", 0.0))
        openness_start = float(stats.get("openness_start", 1.0))
        openness_end = float(stats.get("openness_end", 1.0))

        if openness_end < openness_start:
            ratio = max(ratio, min(0.45, 0.08 + abs(openness_change) * 2.5))
        elif abs(openness_change) < 0.02:
            ratio = max(ratio, 0.05)

        speed = float(stats.get("mean_center_speed", 0.0))
        if speed > 0.03:
            ratio = max(ratio, 0.10)

    relationships = features.get("two_hand_relationships", {})
    if relationships:
        mean_distance = float(relationships.get("mean_distance", 1.0))
        if mean_distance < 0.2:
            ratio = max(ratio, 0.35)

    if len(features.get("detected_hands", [])) >= 2:
        ratio = max(ratio, 0.28)

    return min(1.0, ratio)


def _estimate_wrist_velocity(features: Dict[str, Any]) -> float:
    speeds = [
        float(stats.get("mean_center_speed", 0.0))
        for stats in features.get("hand_motion_summary", {}).values()
    ]
    if speeds:
        return max(speeds)
    return min(0.30, _count_wrist_motion_events(features) * 0.04)


def _count_finger_transitions(features: Dict[str, Any]) -> int:
    count = 0
    for hand_name in ("left", "right"):
        flexing = features.get(f"{hand_name}_hand_events", {}).get("finger_flexing", {})
        for joint_events in flexing.values():
            count += sum(1 for event in joint_events if _is_transition(event))
    return count


def _count_wrist_motion_events(features: Dict[str, Any]) -> int:
    count = 0
    for hand_name in ("left", "right"):
        trajectories = features.get(f"{hand_name}_hand_events", {}).get("wrist_trajectory", {})
        count += sum(len(events) for events in trajectories.values())
    return count


def _is_transition(event: Dict[str, Any]) -> bool:
    return bool(event.get("start_des") and event.get("end_des") and event["start_des"] != event["end_des"])


def create_handx_features_from_chunk(
    chunk: Dict[str, Any],
    features: Dict[str, float] = None
) -> HandXFeatures:
    """
    Helper function to create HandXFeatures from a motion chunk.
    
    Args:
        chunk: Motion chunk metadata (timestamps, video_id, etc.)
        features: Dictionary with kinematic measurements
    
    Returns:
        HandXFeatures dataclass instance
    """
    if features is None:
        features = {}

    inferred = infer_kinematic_features(features)
    hand_sides = inferred["hand_sides_detected"]
    if isinstance(hand_sides, str):
        hand_sides = [hand_sides]

    return HandXFeatures(
        contact_ratio=inferred["contact_ratio"],
        hand_side=inferred["hand_side"],
        hand_sides_detected=hand_sides,
        contact_frequency=inferred["contact_frequency"],
        avg_contact_duration=inferred["avg_contact_duration"],
        wrist_velocity=inferred["wrist_velocity"],
        finger_flexion_variance=inferred["finger_flexion_variance"],
        finger_transition_count=inferred["finger_transition_count"],
        wrist_motion_event_count=inferred["wrist_motion_event_count"],
        primary_joints=inferred.get("primary_joints"),
        description=inferred["description"] or f"Chunk {chunk.get('chunk_id', '?')}",
    )
