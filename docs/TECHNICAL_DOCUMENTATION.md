# Technical Documentation

This document explains the project code in a structured way. It focuses on the
code used by the current CLI and Streamlit pipeline, the RAG implementation,
the active configuration and output schema, and the evaluation helpers used by
the experiment notebook. The bundled `HandX/diffusion` tree is treated as an
external dependency that this project can optionally call into.

## 1. System Purpose

The system turns a hand-activity video into structured action annotations.

At a high level:

```text
Video
 -> MediaPipe hand landmark detection
 -> MediaPipe-to-HandX joint reordering
 -> missing landmark interpolation
 -> motion chunking or dense window chunking
 -> HandX-style feature extraction
 -> RAG retrieval from confirmed_actions.json
 -> optional Qwen-VL visual refinement
 -> JSON output and per-segment MP4 clips
```

The final output is a JSON object with one `video_id` and a list of annotated
`segments`.

## 2. Repository Map

Important project files:

```text
configs/settings.yaml                         Runtime configuration
data/videos/sample.mp4                        Example input video
docs/TECHNICAL_DOCUMENTATION.md               This document
evaluation/metrics.py                         Evaluation metrics
rag/action_dictionary/confirmed_actions.json  Main action vocabulary
rag/retriever.py                              RAG retrieval and scoring
schemas/output_schema.json                    Expected output schema
src/exporter.py                               JSON and MP4 output helpers
src/fusion.py                                 Qwen-VL prompting and parsing
src/hand_detection.py                         MediaPipe hand detection
src/handx_features.py                         HandX/fallback feature extraction
src/rag_annotator.py                          RAG annotation wrapper
src/run_pipeline.py                           Main CLI/pipeline entry point
src/streamlit_app.py                          Streamlit interface
src/video_pipeline.py                         Video metadata, motion, chunking
```

## 3. Main Execution Entry Point

Main file: `src/run_pipeline.py`

Primary function:

```python
run(
    video_path: str,
    settings_path: str = "configs/settings.yaml",
    annotation_mode: str = "prompt",
    progress_callback: Callable[[str, float, str], None] | None = None,
    chunking_mode: str = "motion",
    dense_window_sec: float | None = None,
    dense_overlap_sec: float | None = None,
) -> dict
```

### 3.1 Inputs

- `video_path`: video file to analyze.
- `settings_path`: YAML configuration file.
- `annotation_mode`:
  - `rag`: deterministic retrieval-only annotation.
  - `prompt`: RAG plus Qwen-VL visual annotation.
- `progress_callback`: optional callback used by Streamlit to update progress.
- `chunking_mode`:
  - `motion`: use motion thresholds to find active segments.
  - `dense`: split the full video into overlapping windows.
- `dense_window_sec`, `dense_overlap_sec`: optional overrides for dense mode.

### 3.2 Pipeline Steps

`run()` performs the following steps:

1. Load settings from `configs/settings.yaml`.
2. Validate `annotation_mode` and `chunking_mode`.
3. Lazily import `QwenVLAnnotator` only when `annotation_mode == "prompt"`.
4. Detect hand landmarks from the video using MediaPipe.
5. Convert landmarks into HandX joint order.
6. Interpolate missing hand tracks.
7. Save keypoints to `outputs/json/<video_id>_handx_keypoints.npy`.
8. Create chunks:
   - `motion` mode uses `compute_motion_signal()` and `segment_motion_chunks()`.
   - `dense` mode uses `segment_dense_chunks()`.
9. Load Qwen-VL if prompt mode is selected.
10. Load the RAG action library.
11. For each chunk:
    - slice the keypoints for that chunk,
    - extract HandX-style features,
    - retrieve candidate action labels with RAG,
    - either use the best RAG result directly or ask Qwen-VL to refine it,
    - save a per-chunk MP4 clip.
12. Build the final output JSON.
13. Write the output under `outputs/json/`.

### 3.3 CLI Usage

```bash
python -m src.run_pipeline data/videos/sample.mp4 --annotation-mode rag
python -m src.run_pipeline data/videos/sample.mp4 --annotation-mode prompt
python -m src.run_pipeline data/videos/sample.mp4 --chunking-mode dense
```

Output suffixes:

```text
prompt mode -> outputs/json/<video_id>_segments.json
rag mode    -> outputs/json/<video_id>_segments_rag.json
```

## 4. Configuration

Main file: `configs/settings.yaml`

Important settings:

```yaml
video:
  frame_stride: 1
  max_frames_for_qwen: 4

hand_detection:
  max_num_hands: 2
  min_detection_confidence: 0.5
  min_hand_presence_confidence: 0.5
  min_tracking_confidence: 0.5
  model_path: "models/hand_landmarker.task"
  auto_download_model: true

chunking:
  start_threshold: 0.02
  end_threshold: 0.005
  min_frames: 20
  dense_window_sec: 4.0
  dense_overlap_sec: 1.0

qwen:
  model_id: "Qwen/Qwen2.5-VL-3B-Instruct"
  max_new_tokens_text: 384
  max_new_tokens_vision: 768
  temperature: 0.2

handx:
  diffusion_path: "HandX/diffusion"

rag:
  confirmed_actions_path: "rag/action_dictionary/confirmed_actions.json"
  top_k: 6
```

How these settings affect runtime:

- Increasing `frame_stride` skips frames and speeds up detection, but reduces temporal detail.
- Lowering `max_frames_for_qwen` reduces prompt image load and speeds up prompt mode.
- Lowering `start_threshold` creates more motion chunks.
- Dense chunking is useful for static holds and low-motion actions.
- `rag.top_k` controls how many candidate labels are passed to Qwen-VL.

## 5. Data Structures

### 5.1 Keypoint Array

After hand detection:

```text
shape = (processed_frames, 2, 21, 3)
```

Dimensions:

- `processed_frames`: number of analyzed frames after stride.
- `2`: hand slots.
  - slot `0`: left hand.
  - slot `1`: right hand.
- `21`: hand joints.
- `3`: normalized x, y, z landmark coordinates.

### 5.2 Chunk Object

Chunk objects are created by `_make_chunk()` in `src/video_pipeline.py`.

Example:

```json
{
  "chunk_id": "chunk_000",
  "start_index": 12,
  "end_index": 84,
  "start_frame": 12,
  "end_frame": 84,
  "start_time": 0.4,
  "end_time": 2.8
}
```

`start_index` and `end_index` refer to processed-frame indices. `start_frame`
and `end_frame` refer to original video frame numbers.

### 5.3 Feature Object

Feature objects come from `src/handx_features.py`.

When the full HandX library works, the feature object can include detailed hand
event dictionaries such as:

```text
left_hand_events
right_hand_events
two_hand_relationships
source = "handx"
```

When HandX extraction fails or is unavailable, the fallback feature object
contains:

```json
{
  "frame_count": 42,
  "detected_hands": ["left", "right"],
  "hand_motion_summary": {
    "left": {
      "mean_center_speed": 0.012,
      "max_center_speed": 0.041,
      "openness_start": 0.22,
      "openness_end": 0.17,
      "openness_change": -0.05
    }
  },
  "two_hand_relationships": {
    "mean_distance": 0.18,
    "start_distance": 0.22,
    "end_distance": 0.16,
    "distance_change": -0.06
  },
  "source": "simple_fallback"
}
```

### 5.4 RAG Context

`src/rag_annotator.py` builds a context object for Qwen-VL:

```json
{
  "prompt_text": "formatted candidate list",
  "best_match": {
    "action_label": "power_grasp",
    "action_id": "MAC_006",
    "movement_scale": "macro",
    "hand_side": "either",
    "confidence": 0.72
  },
  "candidates": [
    {
      "rank": 1,
      "action_id": "MAC_006",
      "label": "power_grasp",
      "confidence": 0.72,
      "scale": "macro",
      "hand": "either",
      "description": "...",
      "kinematic_signal": "...",
      "evidence": {
        "contact_ratio": 1.0,
        "hand_compatibility": 1.0,
        "scale_alignment": 1.0,
        "kinematic_similarity": 0.35,
        "temporal_compatibility": 0.6
      }
    }
  ]
}
```

### 5.5 Final Output Segment

Each segment contains chunk metadata, features, RAG details, and final
annotation fields:

```json
{
  "chunk_id": "chunk_000",
  "start_time": 0.4,
  "end_time": 2.8,
  "start_frame": 12,
  "end_frame": 84,
  "features": {},
  "rag_annotation": {},
  "rag_retrieval": [],
  "action_label": "power_grasp",
  "movement_scale": "macro",
  "confidence": 0.72,
  "hand_side": "either",
  "summary": "Short explanation",
  "evidence": "Short supporting facts"
}
```

In prompt mode, `raw_model_output` is also included.

## 6. Hand Detection Module

Main file: `src/hand_detection.py`

This module handles MediaPipe hand landmark extraction and converts landmarks
into the joint order expected by HandX.

### 6.1 Model Download

Constant:

```python
HAND_LANDMARKER_URL
```

Function:

```python
ensure_hand_landmarker_model(model_path, auto_download=True) -> Path
```

Behavior:

- If `model_path` exists, it returns the path.
- If the model is missing and `auto_download` is false, it raises `FileNotFoundError`.
- If the model is missing and `auto_download` is true, it downloads the model.

### 6.2 Video Streaming Detection

Function:

```python
extract_keypoints_handed_from_video(video_path, frame_stride=1, ...) -> tuple
```

This is the main production path. It reads frames one at a time and stores only
keypoints, timestamps, FPS, and original frame numbers. This avoids keeping the
whole video in memory.

Return values:

```text
keypoints, timestamps, fps, frame_numbers
```

### 6.3 Stable Hand Slots

Function:

```python
_detect_frame_keypoints(detector, frame) -> np.ndarray
```

MediaPipe returns handedness labels when available. The code maps:

```text
"Left"  -> slot 0
"Right" -> slot 1
```

If labels are missing or duplicated, it assigns the first unused slot.

### 6.4 MediaPipe to HandX Joint Order

Constant:

```python
HANDX_FROM_MEDIAPIPE
```

Function:

```python
mediapipe_to_handx_order(keypoints) -> np.ndarray
```

MediaPipe and HandX use different joint ordering. This function reindexes the
last hand-joint dimension from MediaPipe order to HandX order.

### 6.5 Missing Track Interpolation

Function:

```python
fill_missing_hand_tracks(handx_keypoints) -> tuple[np.ndarray, list[int]]
```

This finds active hand slots and linearly interpolates missing coordinates over
time. If no hands were detected at all, it raises `ValueError`.

## 7. Video Pipeline Module

Main file: `src/video_pipeline.py`

This module handles video metadata, motion signal creation, and chunk
generation.

### 7.1 Video Metadata

Function:

```python
inspect_video(video_path) -> dict
```

Returns:

```text
fps, frame_count, width, height, duration_sec
```

This is used by the Streamlit UI to preview video metadata before analysis.

### 7.2 Motion Signal

Function:

```python
compute_motion_signal(keypoints) -> np.ndarray
```

For each frame after the first, the function compares current and previous
landmarks. It ignores hands that are missing in either frame and computes the
mean landmark displacement for active hands.

Output:

```text
motion[i] = mean landmark displacement from frame i - 1 to frame i
```

### 7.3 Motion-Based Chunking

Function:

```python
segment_motion_chunks(
    motion,
    timestamps,
    frame_numbers=None,
    start_threshold=0.02,
    end_threshold=0.01,
    min_frames=10,
) -> list[dict]
```

Logic:

```text
if motion > start_threshold and no current chunk:
    start a chunk

if motion < end_threshold and a chunk is active:
    close the chunk if length >= min_frames
```

If the video ends while a chunk is active, the final chunk is closed at the last
frame if it is long enough.

### 7.4 Dense Chunking

Function:

```python
segment_dense_chunks(
    timestamps,
    frame_numbers=None,
    window_sec=4.0,
    overlap_sec=1.0,
) -> list[dict]
```

Dense mode creates overlapping windows across the full video. It is useful when
important actions involve little motion, such as static holds.

Validation:

- `window_sec` must be greater than zero.
- `overlap_sec` must be non-negative and smaller than `window_sec`.

## 8. HandX Feature Module

Main file: `src/handx_features.py`

This module converts keypoint clips into interpretable motion features.

### 8.1 Public Entry Point

Function:

```python
extract_handx_motion_features(motion_clip, active_slots, handx_diffusion_path=None) -> dict
```

Behavior:

- If `handx_diffusion_path` exists, the code tries to use the real HandX
  motion-code utilities.
- If HandX import or extraction fails, the code falls back to simple features.
- If `handx_diffusion_path` does not exist, fallback features are used directly.

### 8.2 HandX Import Protection

Function:

```python
_extract_with_handx_library(...)
```

HandX has a package named `src`, and this project also has a package named
`src`. To avoid import collisions, the function temporarily removes this
project's `src` modules from `sys.modules`, inserts the HandX path at the front
of `sys.path`, imports HandX modules, then restores the original modules.

HandX classes used:

- `MotionCoder` for one-hand clips.
- `BihandMotionCoder` for two-hand clips.

### 8.3 Fallback Feature Extraction

Function:

```python
_extract_simple_features(motion_clip, active_slots) -> dict
```

Fallback features include:

- `frame_count`
- `detected_hands`
- per-hand center speed
- per-hand openness start/end/change
- two-hand distance if two hands are present
- `source = "simple_fallback"`

Hand openness is estimated by measuring distances from wrist to selected
fingertips.

### 8.4 Prompt Size Control

Function:

```python
compact_json(data, max_chars=9000) -> str
```

This pretty-prints JSON but truncates it when it would make a prompt too long.

## 9. Action Dictionary

Main file: `rag/action_dictionary/confirmed_actions.json`

This is the controlled vocabulary for the system. It currently contains 38
action classes grouped into:

- `micro`: isolated finger or precision actions.
- `macro`: wrist, hand, or arm-dominant actions.
- `bimanual`: coordinated two-hand actions.

Each action record includes:

```text
id
label
scale
hand
contact_ratio_range
primary_joints
description
handx_kinematic_signal
notes
```

The dictionary is important because:

- RAG ranks labels from this file.
- Qwen-VL is instructed to choose only from RAG candidates from this file.
- Output labels stay consistent instead of allowing arbitrary model-generated
  labels.

## 10. RAG Retriever

Main file: `rag/retriever.py`

This file implements deterministic retrieval over the action dictionary.

### 10.1 Dataclasses

```python
@dataclass
class HandXFeatures:
    contact_ratio: float
    hand_side: str
    hand_sides_detected: list[str]
    contact_frequency: float = 0.0
    avg_contact_duration: float = 0.0
    wrist_velocity: float = 0.0
    finger_flexion_variance: float = 0.0
    finger_transition_count: int = 0
    wrist_motion_event_count: int = 0
    primary_joints: list[int] = None
    description: str = ""
```

`HandXFeatures` is the normalized feature representation used by the retriever.

```python
@dataclass
class ActionMatch:
    action_id: str
    label: str
    confidence: float
    scale: str
    hand: str
    description: str
    contact_ratio_range: list[float]
    kinematic_signal: str
    evidence_scores: dict[str, float]
```

`ActionMatch` is one retrieved candidate action.

### 10.2 Initialization

Class:

```python
ActionLibraryRetriever(action_library_path=None)
```

Initialization does two things:

1. `_load_library()` reads `confirmed_actions.json`.
2. `_build_embeddings()` builds TF-IDF embeddings for every action's
   `handx_kinematic_signal`.

### 10.3 TF-IDF Kinematic Embeddings

The retriever uses:

```python
TfidfVectorizer(
    analyzer="char",
    ngram_range=(1, 3),
    min_df=1,
    max_df=len(self.actions),
    lowercase=True,
)
```

This is not a neural embedding model. It is a lightweight character n-gram
vectorizer. It is useful for matching short technical phrases like:

```text
finger flexion ratio decreasing
thumb-index fingertip distance
wrist trajectory
```

### 10.4 Retrieval Flow

Function:

```python
retrieve(handx_features, top_k=5, scale_filter=None, hand_filter=None, confidence_threshold=0.0)
```

Steps:

1. Start with all actions.
2. Optionally filter by scale or hand.
3. Score every candidate action.
4. Compute weighted confidence.
5. Sort by descending confidence.
6. Return the top `top_k` actions above `confidence_threshold`.

### 10.5 Score 1: Contact Ratio

Function:

```python
_score_contact_ratio_match(measured_ratio, action) -> float
```

Each action has an expected range:

```json
"contact_ratio_range": [0.02, 0.18]
```

Scoring:

- If measured ratio is inside the range: `1.0`.
- If measured ratio is below the range: linearly penalized up to `0.8`.
- If measured ratio is above the range: `0.8` if only slightly above, otherwise
  `0.5`.

Purpose:

This score checks whether the amount of contact in the chunk is plausible for
the candidate action.

### 10.6 Score 2: Hand Compatibility

Function:

```python
_score_hand_compatibility(hands_detected, action_hand) -> float
```

Action hand values:

```text
left
right
either
both
```

Scoring:

- `either` with at least one detected hand: `1.0`.
- `both` with two detected hands: `1.0`.
- `both` with only one detected hand: `0.3`.
- specific required hand detected: `1.0`.
- specific required hand missing: `0.0`.

Purpose:

This prevents, for example, a bimanual action from ranking too highly when only
one hand was detected.

### 10.7 Score 3: Scale Alignment

Function:

```python
_score_scale_threshold(contact_ratio, hand_side, action_scale) -> float
```

Scale rules:

- `bimanual`: both hands and `contact_ratio >= 0.30`.
- `micro`: `contact_ratio <= 0.25`.
- `macro`: `0.05 <= contact_ratio <= 0.45`.

Scoring:

- If the candidate's scale matches the measured conditions: `1.0`.
- If it does not match: `0.2`.
- If action scale is unknown/unexpected: `0.5`.

Purpose:

This separates small finger-level actions from larger hand/arm movements and
two-hand interactions.

### 10.8 Score 4: Kinematic Similarity

Function:

```python
_score_kinematic_similarity(handx_description, action, use_text=True) -> float
```

This compares:

```text
chunk kinematic description
vs
action handx_kinematic_signal
```

It vectorizes the chunk description with the same TF-IDF vectorizer and computes
cosine similarity against the candidate action's embedded kinematic signal.

Scoring:

- Higher values mean more textual/kinematic overlap.
- If there is no description, the function returns neutral `0.5`.

Purpose:

This is the closest part to "retrieval" in classic RAG. It links observed
motion descriptions to curated action descriptions.

### 10.9 Score 5: Temporal Compatibility

Function:

```python
_score_temporal_compatibility(handx_features, action) -> float
```

This score uses:

- `finger_flexion_variance`
- `wrist_velocity`
- `finger_transition_count`
- `wrist_motion_event_count`

Examples:

- Static hold labels are penalized when flexion activity or wrist activity is
  high.
- Finger actions such as `finger_extension`, `finger_flexion`, `release`, and
  `button_press` are rewarded when finger transitions exist.
- Wrist/arm actions such as `reach`, `transport`, `push`, `pull`, and `wave`
  are rewarded when wrist motion events exist.
- Unknown/default temporal relationship returns `0.6`.

Purpose:

This helps distinguish active manipulation from static holding, especially in
dense windows.

### 10.10 Final Confidence Formula

The current retriever uses five weighted scores:

```text
confidence =
    0.15 * contact_ratio
  + 0.10 * hand_compatibility
  + 0.15 * scale_alignment
  + 0.30 * kinematic_similarity
  + 0.30 * temporal_compatibility
```

The weights show that the current system trusts kinematic text matching and
temporal behavior more than hand-side compatibility.

### 10.11 Feature Inference Helpers

Function:

```python
infer_kinematic_features(features) -> dict
```

This converts raw HandX or fallback features into the normalized fields needed
by `HandXFeatures`.

Important helper functions:

- `build_kinematic_description(features)`: creates text for TF-IDF retrieval.
- `_estimate_contact_ratio(features)`: estimates contact from hand openness and
  two-hand proximity.
- `_estimate_wrist_velocity(features)`: estimates wrist activity.
- `_count_finger_transitions(features)`: counts finger state changes in HandX
  event dictionaries.
- `_count_wrist_motion_events(features)`: counts wrist trajectory events.

Function:

```python
create_handx_features_from_chunk(chunk, features) -> HandXFeatures
```

This is the adapter used by `RAGAnnotator`.

## 11. RAG Annotator Wrapper

Main file: `src/rag_annotator.py`

This file wraps `ActionLibraryRetriever` and formats retrieval results for the
pipeline and for Qwen-VL prompts.

### 11.1 Main Class

```python
class RAGAnnotator:
    def __init__(self, action_library_path=None)
```

The constructor loads `ActionLibraryRetriever`.

### 11.2 Annotating One Chunk

Function:

```python
annotate_chunk(features, chunk, top_k=6, verbose=False)
```

Returns:

```text
refined, retrieval_results, rag_context
```

`refined` is the best match in final annotation format:

```json
{
  "action_label": "power_grasp",
  "action_id": "MAC_006",
  "movement_scale": "macro",
  "hand_side": "either",
  "confidence": 0.72,
  "summary": "...",
  "evidence": "Contact Ratio: 1.00 | ..."
}
```

`retrieval_results` is a transparent top-k list with evidence scores.

`rag_context` is structured context for Qwen-VL.

### 11.3 Retrieval Only

Function:

```python
retrieve_matches(features, chunk, top_k=6, verbose=False)
```

This returns raw `ActionMatch` objects instead of JSON dictionaries.

### 11.4 Prompt Context Formatting

Function:

```python
format_prompt_context(matches, refined=None, max_candidates=3)
```

This returns:

- `prompt_text`: a readable candidate list.
- `best_match`: RAG's first hypothesis.
- `candidates`: structured candidate metadata.

## 12. Qwen-VL Fusion Module

Main file: `src/fusion.py`

This module handles the Qwen-VL path used by prompt mode: model loading,
one-pass prompt construction, frame sampling, model generation, and JSON
parsing.

### 12.1 Main Class

```python
class QwenVLAnnotator:
    def __init__(model_id, temperature=0.2, max_new_tokens_text=384, max_new_tokens_vision=768)
```

The constructor loads:

- `AutoProcessor.from_pretrained(model_id)`
- `Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, device_map="auto")`

It uses `float16` when CUDA is available and `float32` otherwise.

### 12.2 One-Pass Chunk Annotation

Function:

```python
annotate_chunk(feature_json, video_path, chunk, rag_context, max_frames=8)
```

This is the main path used by `run_pipeline.py` in prompt mode. It sends one
prompt containing:

- chunk timestamps and frame numbers,
- RAG candidates,
- HandX-style features,
- sampled video frames.

The prompt requires the model to return only JSON.

### 12.3 Frame Sampling

Function:

```python
sample_frames_for_qwen_from_video(video_path, chunk, max_frames=8, max_image_side=640)
```

The prompt-mode pipeline uses this function because it avoids loading the full
video into memory. It seeks to evenly spaced frame indices inside the chunk and
converts frames to PIL RGB images.

### 12.4 RAG Context Formatting

Function:

```python
_format_rag_context(rag_context) -> str
```

This turns structured RAG context into readable prompt text. It also includes a
"starting hypothesis" header, while telling the model that visual evidence may
override the first RAG guess.

### 12.5 JSON Parsing and Repair

Function:

```python
parse_json_object(text) -> dict
```

The parser is defensive because model output can include code fences or can be
truncated. It:

1. strips optional Markdown code fences,
2. finds the first `{`,
3. extracts the first balanced JSON object,
4. if needed, attempts to repair truncated JSON,
5. normalizes common deviations.

Helpers:

- `_strip_code_fence(text)`
- `_first_balanced_json_object(text)`
- `_repair_truncated_json_object(text)`
- `_normalize_model_json(obj)`

Normalization currently converts list-valued `evidence` into a string and
coerces `confidence` to float.

### 12.6 Prompt

Prompt constant used by the current prompt-mode pipeline:

- `ONE_PASS_PROMPT`

The prompt enforces:

- labels must come from RAG candidates or be `unknown`,
- output must be valid JSON,
- confidence should be low when evidence is weak,
- summaries should mention visible evidence when frames are used.

## 13. Export Module

Main file: `src/exporter.py`

### 13.1 JSON Writer

Function:

```python
write_json(path, data) -> None
```

Creates parent directories and writes pretty JSON with indentation.

### 13.2 Clip Export from Source Video

Function:

```python
save_video_clip_from_video(video_path, start_frame, end_frame, output_path) -> None
```

This is the main pipeline path. It seeks to `start_frame`, writes frames through
`end_frame`, and avoids keeping the full video in memory.

### 13.3 Output Builder

Function:

```python
build_video_output(video_id, segments) -> dict
```

Returns:

```json
{
  "video_id": "sample",
  "segments": []
}
```

## 14. Streamlit App

Main file: `src/streamlit_app.py`

This provides an interactive UI for uploading videos and running the pipeline.

### 14.1 Project Path Setup

The app computes `PROJECT_ROOT`, inserts it into `sys.path`, and changes the
current working directory to the project root. This ensures imports and relative
paths work when Streamlit launches from another directory.

### 14.2 UI Flow

`main()` builds the app:

1. Set page title and layout.
2. Upload a video.
3. Choose analysis mode:
   - RAG retrieval.
   - RAG plus Qwen-VL visual analysis.
4. Choose chunking mode:
   - dense overlapping windows,
   - motion-triggered segments.
5. If dense mode is selected, configure window duration and overlap.
6. Preview video and metadata.
7. Estimate dense chunk count.
8. Run analysis on button click.
9. Display segment table.
10. Offer JSON download.
11. Show raw JSON in an expander.

### 14.3 Temporary Files

Uploaded videos are written to temporary files so OpenCV can read them. The app
removes temporary files in `finally` blocks.

### 14.4 Progress Reporting

`_analyze_video()` passes a callback to `run()`. That callback updates a
Streamlit progress bar and status message.

## 15. Evaluation Module

Main file: `evaluation/metrics.py`

This file evaluates temporal segmentation and labels against ground truth.

### 15.1 Temporal IoU

Function:

```python
temporal_iou(a_start, a_end, b_start, b_end) -> float
```

Computes time-range intersection-over-union:

```text
intersection = overlap duration
union = combined duration
iou = intersection / union
```

### 15.2 Matching Predictions to Ground Truth

Function:

```python
match_predictions(predictions, ground_truth, iou_threshold=0.5) -> list[dict]
```

For each prediction, it finds the best unused ground-truth segment by temporal
IoU. If the best IoU is above the threshold, it records a match.

### 15.3 Macro F1

Function:

```python
macro_f1(predictions, ground_truth, iou_threshold=0.5) -> float
```

It first performs temporal matching, then computes F1 per action label and
averages across labels.

### 15.4 Evaluation Summary

Function:

```python
evaluate_predictions(predictions, ground_truth, iou_threshold=0.5) -> dict
```

Returns:

```text
matched_segments
prediction_segments
ground_truth_segments
mean_temporal_iou
macro_f1
iou_threshold
```

## 16. Schemas

### 16.1 Output Schema

File: `schemas/output_schema.json`

Required top-level fields:

- `video_id`
- `segments`

Required segment fields:

- `chunk_id`
- `start_time`
- `end_time`
- `action_label`
- `movement_scale`
- `confidence`
- `hand_side`

Allowed `movement_scale` values:

```text
micro
macro
bimanual
unknown
```

Allowed `hand_side` values:

```text
left
right
both
unknown
```

Note: the action dictionary can use `either` for action definitions, while the
output schema lists `left`, `right`, `both`, and `unknown`. If schema validation
is added strictly, output normalization may need to map `either` to the detected
side or `unknown`.

## 17. Annotation Modes

### 17.1 RAG Mode

Command:

```bash
python -m src.run_pipeline data/videos/sample.mp4 --annotation-mode rag
```

Behavior:

- Does not load Qwen-VL.
- Uses the best RAG retrieval result as the final annotation.
- Faster and deterministic.
- Output includes RAG evidence and retrieval list.

Best for:

- quick experiments,
- CPU-only runs,
- debugging the action dictionary,
- comparing retrieval behavior.

### 17.2 Prompt Mode

Command:

```bash
python -m src.run_pipeline data/videos/sample.mp4 --annotation-mode prompt
```

Behavior:

- Loads Qwen-VL.
- Runs RAG first.
- Samples frames from each chunk.
- Sends features, frames, and RAG candidates to the model.
- Uses model JSON as the final annotation.

Best for:

- richer visual descriptions,
- object-aware summaries,
- resolving ambiguous RAG candidates with image evidence.

## 18. Chunking Modes

### 18.1 Motion Mode

Motion mode is event-driven. It only creates chunks when landmark displacement
crosses the configured thresholds.

Strengths:

- Fewer chunks.
- Faster.
- Good for clear movement events.

Limitations:

- Can miss quiet actions or static holds.
- Thresholds may need tuning for different videos.

### 18.2 Dense Mode

Dense mode covers the full video with overlapping fixed windows.

Strengths:

- Captures low-motion actions.
- More consistent for full-video inspection.
- Works well with temporal compatibility scoring.

Limitations:

- More chunks.
- Slower.
- Adjacent windows may produce overlapping annotations.

## 19. How RAG and Qwen-VL Work Together

RAG is used as a controlled-label retrieval layer. It narrows the possible label
set before Qwen-VL sees the chunk.

In prompt mode:

1. RAG retrieves top candidates from `confirmed_actions.json`.
2. The top candidates are formatted as prompt context.
3. Qwen-VL receives those candidates plus sampled frames and features.
4. Qwen-VL must choose one of the candidate labels or return `unknown`.

This design reduces model hallucination because Qwen-VL is not free to invent
arbitrary action labels.

## 20. Important Runtime Dependencies

Listed in `requirements.txt`:

- `opencv-python`: video reading and writing.
- `mediapipe`: hand landmark detection.
- `numpy`: array processing.
- `pyyaml`: settings loading.
- `tqdm`: progress bars.
- `pillow`: frame images for Qwen-VL.
- `scipy`: scientific utilities.
- `scikit-learn`: TF-IDF vectorizer and cosine similarity.
- `torch`: Qwen-VL model runtime.
- `transformers`: Qwen-VL model and processor loading.
- `accelerate`: device mapping support.
- `qwen-vl-utils`: image/video prompt processing.
- `streamlit`: web UI.

## 21. Error Handling and Defensive Design

Notable safeguards:

- Missing MediaPipe model can be downloaded automatically.
- If no hands are detected, `fill_missing_hand_tracks()` raises an explicit
  error.
- If HandX extraction fails, fallback features are returned and the error is
  stored as `handx_error`.
- Prompt mode lazily imports Qwen-VL dependencies and gives a clear error if a
  package is missing.
- Qwen-VL JSON output is parsed defensively and can be repaired when truncated.
- Video clips are saved by rereading the original video instead of keeping all
  frames in memory.

## 22. Mathematical Formulas And Calculations

This section collects the calculations used across the codebase. Variable names
match the source code where possible.

### 22.1 Video Time Calculations

File: `src/video_pipeline.py`

Video duration:

```text
duration_sec = frame_count / fps
```

Timestamp for an original video frame:

```text
timestamp_sec = frame_index / fps
```

When `frame_stride > 1`, the pipeline only processes frames where:

```text
frame_index mod frame_stride = 0
```

The number of processed frames is approximately:

```text
processed_frames = ceil(total_frames / frame_stride)
```

### 22.2 Active Hand Detection

Files:

- `src/video_pipeline.py`
- `src/hand_detection.py`

A hand slot is considered present when its full landmark tensor is nonzero:

```text
active_hand = norm(hand_keypoints) > 1e-8
```

For a hand with 21 joints and 3 coordinates:

```text
norm(hand_keypoints) =
sqrt(sum over joints j and coordinates c of keypoint[j, c]^2)
```

During motion computation, a hand is active for frame `i` only if it exists in
both the current and previous frames:

```text
active_i = active(current_i) AND active(previous_i)
```

### 22.3 Landmark Motion Signal

File: `src/video_pipeline.py`

Function:

```python
compute_motion_signal(keypoints)
```

For every processed frame `i > 0`, the code computes landmark displacement from
frame `i - 1` to frame `i`.

For one joint:

```text
d[i, hand, joint] =
sqrt(
    (x_i - x_{i-1})^2
  + (y_i - y_{i-1})^2
  + (z_i - z_{i-1})^2
)
```

The frame-level motion value is the mean displacement over all active hands and
their 21 joints:

```text
motion[i] = mean(d[i, active_hands, all_joints])
```

The first frame has no previous frame, so:

```text
motion[0] = 0
```

### 22.4 Motion-Based Chunking

File: `src/video_pipeline.py`

Function:

```python
segment_motion_chunks(...)
```

Chunk start condition:

```text
start chunk at frame i if:
    no chunk is active
    AND motion[i] > start_threshold
```

Chunk end condition:

```text
end chunk at frame i if:
    chunk is active
    AND motion[i] < end_threshold
    AND i - start_frame >= min_frames
```

With the current default settings:

```text
start_threshold = 0.02
end_threshold   = 0.005
min_frames      = 20
```

This is a hysteresis-style threshold design: the start threshold is higher than
the end threshold, which prevents small motion fluctuations from rapidly opening
and closing chunks.

### 22.5 Dense Window Chunking

File: `src/video_pipeline.py`

Function:

```python
segment_dense_chunks(...)
```

Dense chunking uses overlapping fixed-duration windows:

```text
step_sec = window_sec - overlap_sec
```

For default dense settings:

```text
window_sec  = 4.0
overlap_sec = 1.0
step_sec    = 3.0
```

The approximate number of dense chunks for a video of duration `T` is:

```text
chunk_count ~= ceil((T - window_sec) / step_sec) + 1
```

The code uses timestamp search rather than this approximation, so exact counts
depend on frame timestamps and the final-window handling.

### 22.6 Hand Center Speed

File: `src/handx_features.py`

Fallback feature extraction computes a hand center for every frame:

```text
center[t] = (1 / 21) * sum over joints j of point[t, j]
```

The per-frame center speed is:

```text
speed[t] = norm(center[t] - center[t - 1])
```

Expanded:

```text
speed[t] =
sqrt(
    (center_x[t] - center_x[t - 1])^2
  + (center_y[t] - center_y[t - 1])^2
  + (center_z[t] - center_z[t - 1])^2
)
```

Fallback summary values:

```text
mean_center_speed = mean(speed)
max_center_speed  = max(speed)
```

### 22.7 Hand Openness

File: `src/handx_features.py`

Fallback openness is estimated from wrist-to-fingertip distances. The code uses
HandX-ordered fingertip indices:

```text
fingertip_indices = [17, 18, 19, 20]
```

For each frame:

```text
openness[t] =
(1 / 4) * sum over fingertips f of norm(point[t, f] - wrist[t])
```

Openness change across the chunk:

```text
openness_change = openness[last] - openness[first]
```

Interpretation:

- negative `openness_change`: the hand is closing or curling.
- positive `openness_change`: the hand is opening or extending.
- near-zero `openness_change`: the hand pose is relatively stable.

### 22.8 Two-Hand Distance

File: `src/handx_features.py`

If both hands are active, the fallback extractor computes left and right hand
centers:

```text
left_center[t]  = mean(left_hand_points[t])
right_center[t] = mean(right_hand_points[t])
```

Per-frame distance:

```text
distance[t] = norm(left_center[t] - right_center[t])
```

Summary values:

```text
mean_distance   = mean(distance)
start_distance  = distance[first]
end_distance    = distance[last]
distance_change = end_distance - start_distance
```

Interpretation:

- negative `distance_change`: hands moved closer together.
- positive `distance_change`: hands moved farther apart.

### 22.9 Estimated Contact Ratio

File: `rag/retriever.py`

Function:

```python
_estimate_contact_ratio(features)
```

If a raw `contact_ratio` is not available, the retriever estimates one using a
heuristic. The initial value is:

```text
ratio = 0.12
```

For each hand summary:

```text
if openness_end < openness_start:
    ratio = max(ratio, min(0.45, 0.08 + abs(openness_change) * 2.5))

elif abs(openness_change) < 0.02:
    ratio = max(ratio, 0.05)

if mean_center_speed > 0.03:
    ratio = max(ratio, 0.10)
```

For two-hand proximity:

```text
if mean_distance < 0.2:
    ratio = max(ratio, 0.35)
```

If two hands are detected:

```text
ratio = max(ratio, 0.28)
```

Final clamp:

```text
contact_ratio = min(1.0, ratio)
```

This is a pragmatic estimate, not a physical contact measurement. It gives RAG
enough signal to compare the chunk against action dictionary ranges.

### 22.10 Estimated Wrist Velocity

File: `rag/retriever.py`

Function:

```python
_estimate_wrist_velocity(features)
```

If fallback hand speed values exist:

```text
wrist_velocity = max(mean_center_speed for each detected hand)
```

If no speed values exist, the code estimates from wrist event count:

```text
wrist_velocity = min(0.30, wrist_motion_event_count * 0.04)
```

### 22.11 Finger Flexion Variance Estimate

File: `rag/retriever.py`

When `finger_flexion_variance` is not provided directly, the retriever estimates
it from finger transition count:

```text
finger_flexion_variance = min(1.0, finger_transition_count / 10.0)
```

The transition count is computed from HandX event dictionaries. A transition is
counted when:

```text
start_des exists
AND end_des exists
AND start_des != end_des
```

### 22.12 Contact Ratio Match Score

File: `rag/retriever.py`

Function:

```python
_score_contact_ratio_match(measured_ratio, action)
```

Each action has:

```text
min_ratio = action.contact_ratio_range[0]
max_ratio = action.contact_ratio_range[1]
```

Piecewise score:

```text
if measured_ratio < min_ratio:
    score = max(0, measured_ratio / min_ratio) * 0.8
    if min_ratio == 0:
        score = 0.5

elif measured_ratio > max_ratio:
    if measured_ratio < max_ratio + 0.2:
        score = 0.8
    else:
        score = 0.5

else:
    score = 1.0
```

### 22.13 Hand Compatibility Score

File: `rag/retriever.py`

Function:

```python
_score_hand_compatibility(hands_detected, action_hand)
```

Piecewise score:

```text
if action_hand == "either":
    score = 1.0 if len(hands_detected) >= 1 else 0.0

elif action_hand == "both":
    score = 1.0 if len(hands_detected) == 2 else 0.3

elif action_hand in hands_detected:
    score = 1.0

else:
    score = 0.0
```

### 22.14 Scale Alignment Score

File: `rag/retriever.py`

Function:

```python
_score_scale_threshold(contact_ratio, hand_side, action_scale)
```

Piecewise score:

```text
if action_scale == "bimanual":
    score = 1.0 if hand_side == "both" and contact_ratio >= 0.30 else 0.2

elif action_scale == "micro":
    score = 1.0 if contact_ratio <= 0.25 else 0.2

elif action_scale == "macro":
    score = 1.0 if 0.05 <= contact_ratio <= 0.45 else 0.2

else:
    score = 0.5
```

### 22.15 TF-IDF And Cosine Similarity

File: `rag/retriever.py`

Function:

```python
_score_kinematic_similarity(handx_description, action)
```

The retriever builds TF-IDF vectors for action kinematic descriptions using
character n-grams of length 1 to 3.

Conceptually, TF-IDF weights a term by:

```text
tfidf(term, document) = term_frequency(term, document) * inverse_document_frequency(term)
```

Then the chunk description vector is compared to an action vector with cosine
similarity:

```text
cosine_similarity(A, B) = dot(A, B) / (norm(A) * norm(B))
```

Expanded:

```text
dot(A, B) = sum over dimensions k of A[k] * B[k]

norm(A) = sqrt(sum over dimensions k of A[k]^2)
norm(B) = sqrt(sum over dimensions k of B[k]^2)
```

If no kinematic description exists:

```text
kinematic_similarity = 0.5
```

### 22.16 Temporal Compatibility Score

File: `rag/retriever.py`

Function:

```python
_score_temporal_compatibility(handx_features, action)
```

Useful intermediate booleans:

```text
has_finger_changes = finger_transition_count > 0
has_wrist_motion   = wrist_motion_event_count > 0
```

Static hold labels:

```text
if label in ("static_hold_precision", "static_hold_power"):
    if finger_flexion_variance > 0.20 OR wrist_velocity > 0.05:
        score = 0.15
    else:
        score = 1.0
```

Finger-transition labels:

```text
if label in ("finger_extension", "finger_flexion", "power_curl", "release", "button_press"):
    score = 1.0 if has_finger_changes else 0.2
```

Rotation labels:

```text
if label in ("in_hand_rotation", "dial_rotation"):
    score = 1.0 if has_finger_changes else 0.4
```

Wrist/arm movement labels:

```text
if label in ("reach", "transport", "push", "pull", "wave"):
    score = 1.0 if has_wrist_motion else 0.25
```

Default:

```text
score = 0.6
```

### 22.17 Final RAG Confidence

File: `rag/retriever.py`

The current final confidence is a weighted sum:

```text
confidence =
    0.15 * contact_ratio_score
  + 0.10 * hand_compatibility_score
  + 0.15 * scale_alignment_score
  + 0.30 * kinematic_similarity_score
  + 0.30 * temporal_compatibility_score
```

The weights sum to 1:

```text
0.15 + 0.10 + 0.15 + 0.30 + 0.30 = 1.00
```

Example:

```text
contact_ratio_score        = 1.00
hand_compatibility_score   = 1.00
scale_alignment_score      = 0.20
kinematic_similarity_score = 0.40
temporal_compatibility     = 0.60

confidence =
    0.15 * 1.00
  + 0.10 * 1.00
  + 0.15 * 0.20
  + 0.30 * 0.40
  + 0.30 * 0.60

confidence = 0.58
```

### 22.18 Qwen Frame Sampling

File: `src/fusion.py`

Function:

```python
sample_frames_for_qwen_from_video(video_path, chunk, max_frames=8)
```

The number of sampled frames is:

```text
count = min(max_frames, max(1, end_frame - start_frame + 1))
```

The sampled frame numbers are evenly spaced:

```text
frame_indices = linspace(start_frame, end_frame, count)
```

The timestamp for each sampled original frame is:

```text
time_sec = frame_index / fps
```

Images are resized with aspect ratio preserved so their largest side is at most:

```text
max_image_side = 640
```

### 22.19 Temporal IoU

File: `evaluation/metrics.py`

Function:

```python
temporal_iou(a_start, a_end, b_start, b_end)
```

Intersection:

```text
intersection = max(0, min(a_end, b_end) - max(a_start, b_start))
```

Union:

```text
union = max(a_end, b_end) - min(a_start, b_start)
```

Temporal IoU:

```text
iou = intersection / union
```

If `union <= 0`, the function returns:

```text
iou = 0
```

### 22.20 Prediction Matching

File: `evaluation/metrics.py`

Function:

```python
match_predictions(predictions, ground_truth, iou_threshold=0.5)
```

For each prediction `p`, the code finds the unused ground-truth segment `g`
with highest temporal IoU:

```text
best_g = argmax over unused g of temporal_iou(p, g)
```

A match is accepted only if:

```text
temporal_iou(p, best_g) >= iou_threshold
```

Each ground-truth segment can be matched at most once.

### 22.21 Precision, Recall, And Macro F1

File: `evaluation/metrics.py`

For each action label:

```text
precision = TP / (TP + FP)
recall    = TP / (TP + FN)
```

If a denominator is zero, the code uses `0.0`.

Per-label F1:

```text
F1 = 2 * precision * recall / (precision + recall)
```

If `precision + recall == 0`, the F1 is `0.0`.

Macro F1:

```text
macro_f1 = mean(F1 for each label)
```

### 22.22 Mean Temporal IoU

File: `evaluation/metrics.py`

Function:

```python
evaluate_predictions(...)
```

Mean temporal IoU is computed over matched segments:

```text
mean_temporal_iou = sum(match.iou for all matches) / number_of_matches
```

If there are no matches:

```text
mean_temporal_iou = 0.0
```

## 23. Common Extension Points

### Add a New Action

Edit:

```text
rag/action_dictionary/confirmed_actions.json
```

Add a record with:

- unique `id`,
- `label`,
- `scale`,
- `hand`,
- `contact_ratio_range`,
- `primary_joints`,
- `description`,
- `handx_kinematic_signal`,
- optional `notes`.

The retriever will automatically include it when it reloads the dictionary.

### Change RAG Ranking Behavior

Edit:

```text
rag/retriever.py
```

Likely places:

- `_score_contact_ratio_match()`
- `_score_hand_compatibility()`
- `_score_scale_threshold()`
- `_score_kinematic_similarity()`
- `_score_temporal_compatibility()`
- final confidence weights inside `retrieve()`

### Change Prompt Behavior

Edit:

```text
src/fusion.py
```

Likely places:

- `ONE_PASS_PROMPT`
- `_format_rag_context()`
- `parse_json_object()`

### Change Chunking

Edit:

```text
src/video_pipeline.py
configs/settings.yaml
```

For threshold behavior, tune:

- `start_threshold`
- `end_threshold`
- `min_frames`

For dense behavior, tune:

- `dense_window_sec`
- `dense_overlap_sec`

### Change UI Defaults

Edit:

```text
src/streamlit_app.py
```

The UI currently defaults to RAG mode and dense chunking.

## 24. Practical Debugging Guide

### No Segments Found

Possible causes:

- Motion threshold is too high.
- Video has very little movement.
- Hands were not detected consistently.

Try:

- use `--chunking-mode dense`,
- lower `chunking.start_threshold`,
- inspect saved keypoints,
- check MediaPipe detection confidence settings.

### Prompt Mode Is Slow

Possible causes:

- Qwen-VL is running on CPU.
- Too many chunks.
- Too many sampled frames.

Try:

- use `--annotation-mode rag`,
- use motion chunking instead of dense chunking,
- lower `video.max_frames_for_qwen`,
- increase `video.frame_stride`.

### RAG Picks Static Holds Too Often

Check:

- `temporal_compatibility` evidence score,
- `finger_transition_count`,
- `wrist_motion_event_count`,
- kinematic descriptions generated by `build_kinematic_description()`.

Likely edits:

- strengthen temporal penalties in `_score_temporal_compatibility()`,
- improve HandX event extraction,
- add more specific `handx_kinematic_signal` phrases to the action dictionary.

### Output Does Not Validate Against Schema

Check:

- `hand_side` can come from an action definition as `either`.
- `evidence` should be a string.
- `confidence` should be numeric in `[0, 1]`.

Likely edits:

- normalize output fields in `src/run_pipeline.py`,
- normalize model output in `_normalize_model_json()`,
- update the schema if `either` is intended as a valid final output value.

## 25. Mental Model for the Codebase

Think of the project as four layers:

```text
Perception layer:
  src/hand_detection.py
  src/video_pipeline.py

Feature layer:
  src/handx_features.py

Reasoning layer:
  rag/retriever.py
  src/rag_annotator.py
  src/fusion.py

Product/output layer:
  src/run_pipeline.py
  src/streamlit_app.py
  src/exporter.py
  evaluation/metrics.py
```

The most important design idea is that the action dictionary controls the label
space. RAG retrieves from that dictionary, and Qwen-VL is used only after the
label space has been narrowed to grounded candidates.
