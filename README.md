# Action-Based Video Understanding System

This project turns hand-activity videos into structured action annotations. It detects hands frame by frame, creates motion-triggered or dense analysis chunks, extracts HandX-style kinematic features, retrieves candidate labels from a curated action dictionary, and can use Qwen-VL with sampled video frames to produce final JSON descriptions.

Project contributors:

- Mentor: Patrick Saade
- Guidance and project idea: Rabih Amhaz @ RA Development
- Team members: Nihal Elzubair, Abdulla Mohamed, Ahmed Mohamed

The current implementation is:

- memory-friendly video preprocessing
- MediaPipe hand landmark detection
- motion-based and dense-window chunking
- HandX-style feature extraction
- RAG retrieval over `confirmed_actions.json`
- Qwen2.5-VL visual description
- `prompt` and `rag` description modes
- local Streamlit upload interface
- JSON and clip export
- baseline temporal IoU and macro-F1 metrics

For a source-level explanation of every project module, see
[`docs/TECHNICAL_DOCUMENTATION.md`](docs/TECHNICAL_DOCUMENTATION.md).

## What The System Produces

For each input video, the pipeline writes one JSON file containing detected action segments:

```json
{
  "video_id": "sample",
  "segments": [
    {
      "chunk_id": "chunk_000",
      "start_time": 1.2,
      "end_time": 3.4,
      "start_frame": 36,
      "end_frame": 102,
      "action_label": "static_hold_power",
      "movement_scale": "macro",
      "confidence": 0.84,
      "hand_side": "both",
      "summary": "Both hands hold a remote-like device while small adjustments are made.",
      "evidence": "RAG matched power grasp features; sampled frames show sustained hand-object contact."
    }
  ]
}
```

The output schema is defined in `schemas/output_schema.json`.

## Pipeline Overview

```text
Video file
 -> Read frames one at a time
 -> Detect 21 hand landmarks per hand with MediaPipe Tasks
 -> Store keypoints, timestamps, and original frame numbers
 -> Convert MediaPipe joints to HandX joint order
 -> Interpolate missing hand detections
 -> Create analysis chunks using motion thresholds or dense windows
 -> Extract HandX-style features for each chunk
 -> Retrieve top action candidates from confirmed_actions.json
 -> Optional: sample frames and ask Qwen-VL to refine the label
 -> Save final JSON, raw model output, keypoints, and MP4 clips
```

The main entry point is `src/run_pipeline.py`.

## Local Streamlit UI

Run the local upload interface from the project root:

```bash
streamlit run src/streamlit_app.py
```

The page previews the uploaded video, shows live preprocessing and analysis
progress, displays the detected segments, and provides the final JSON as a
download. Use `rag` mode for a faster local run or `prompt` mode to include
Qwen-VL visual analysis. Dense chunking is the UI default: it analyzes the
full video with overlapping windows, including quiet holds that motion-only
chunking may skip.

## Annotation Modes

The pipeline supports two annotation modes and two chunking modes:

```bash
python -m src.run_pipeline data/videos/sample.mp4 --annotation-mode prompt
python -m src.run_pipeline data/videos/sample.mp4 --annotation-mode rag
python -m src.run_pipeline data/videos/sample.mp4 --annotation-mode rag --chunking-mode dense
python -m src.run_pipeline data/videos/sample.mp4 --chunking-mode dense --dense-window-sec 4 --dense-overlap-sec 1
```

`prompt` is the default CLI annotation mode. It runs RAG retrieval first, then sends the top candidates, HandX-style features, timestamps, and sampled frames to Qwen-VL. This gives the richest annotations.

`rag` uses only deterministic retrieval from the action dictionary. It is faster and does not load Qwen-VL.

`motion` is the default CLI chunking mode. It creates segments only when landmark motion crosses the configured thresholds.

`dense` creates overlapping windows across the full video. This is useful for quiet holds or subtle interactions that motion-only chunking may skip. The Streamlit UI defaults to `rag` annotation mode and `dense` chunking.

Output file suffixes:

- `prompt`: `outputs/json/<video_id>_segments.json`
- `rag`: `outputs/json/<video_id>_segments_rag.json`

## Action Dictionary And RAG

The current controlled vocabulary is:

```text
rag/action_dictionary/confirmed_actions.json
```

It contains 38 action classes across three movement scales:

- `micro`: isolated finger movements and precision gestures
- `macro`: wrist, hand, and arm-dominant single-hand actions
- `bimanual`: coordinated two-hand actions

Each action includes:

- action id, such as `MIC_001`, `MAC_006`, or `BIM_008`
- label, such as `fingertip_pinch`, `power_grasp`, or `bimanual_manipulation`
- movement scale
- expected hand side
- contact ratio range
- primary MANO/HandX joints
- kinematic signal description
- notes for disambiguation

`rag/retriever.py` turns extracted features into a `HandXFeatures` object and scores each action using:

- contact ratio match
- hand compatibility
- movement-scale alignment
- TF-IDF similarity between kinematic descriptions
- temporal compatibility for active motion vs static holds

The current confidence formula is:

```text
confidence =
    0.15 * contact_ratio
  + 0.10 * hand_compatibility
  + 0.15 * scale_alignment
  + 0.30 * kinematic_similarity
  + 0.30 * temporal_compatibility
```

The top matches are passed to Qwen-VL as the only allowed label candidates, which keeps labels consistent and prevents the model from inventing new class names. The default `top_k` is 6.

## Hand Detection

Hand detection lives in `src/hand_detection.py`.

The project uses the MediaPipe Tasks `HandLandmarker` model. If the model file is missing and `auto_download_model` is enabled, it downloads:

```text
models/hand_landmarker.task
```

For each processed frame, the detector stores:

```text
(frames, 2, 21, 3)
```

The hand slots are:

```text
0 = left
1 = right
```

The pipeline does not keep all raw frames in memory. It stores only landmarks, timestamps, and frame numbers, then reloads selected frames later when Qwen-VL needs visual context.

## Chunking

Chunking lives in `src/video_pipeline.py`.

In motion mode, the system computes landmark displacement between consecutive frames:

```text
motion becomes high -> chunk starts
motion becomes low  -> chunk ends
```

The thresholds are configured in `configs/settings.yaml`:

```yaml
chunking:
  start_threshold: 0.02
  end_threshold: 0.005
  min_frames: 20
  dense_window_sec: 4.0
  dense_overlap_sec: 1.0
```

In dense mode, the full video is split into overlapping windows. With the default settings, each window is 4 seconds long and overlaps the next window by 1 second.

Each chunk stores processed-frame indices, original video frame numbers, start/end times, and a stable `chunk_id`.

## HandX-Style Features

Feature extraction lives in `src/handx_features.py`.

If `HandX/diffusion` is available, the project tries to call HandX motion-code utilities. If that fails or the HandX path is missing, it falls back to lightweight features:

- detected hands
- per-hand center speed
- hand openness start/end/change
- two-hand distance and distance change
- feature source, either `handx` or `simple_fallback`

This keeps the project runnable even when the full HandX environment is not available.

## Qwen-VL Fusion

Qwen-VL annotation lives in `src/fusion.py`.

The default model is:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

For each chunk, Qwen receives:

- chunk timestamps and frame numbers
- compact HandX-style feature JSON
- top RAG candidates from `confirmed_actions.json`
- sampled frames from the same video chunk

The prompt requires JSON with:

- `action_label`
- `movement_scale`
- `hand_side`
- `confidence`
- `summary`
- `evidence`

The parser is defensive for long videos and busy frames. It accepts fenced JSON, extracts the first balanced JSON object, normalizes common model deviations, and can repair outputs truncated near the end of a string or array.

## Configuration

Main settings are in `configs/settings.yaml`.

Important options:

```yaml
video:
  frame_stride: 1
  max_frames_for_qwen: 4

qwen:
  model_id: "Qwen/Qwen2.5-VL-3B-Instruct"
  max_new_tokens_text: 384
  max_new_tokens_vision: 768
  temperature: 0.2

rag:
  confirmed_actions_path: "rag/action_dictionary/confirmed_actions.json"
  top_k: 6
```

For faster runs, increase `frame_stride`, lower `max_frames_for_qwen`, use `--annotation-mode rag`, or use motion chunking instead of dense chunking.

For richer Qwen summaries, increase `max_new_tokens_vision`, but expect slower inference and more GPU memory use.

## Folder Layout

```text
configs/settings.yaml                  Main runtime configuration
data/videos/                           Input videos
evaluation/metrics.py                  Baseline temporal IoU and macro-F1
HandX/diffusion/                       Optional bundled HandX-related code
models/hand_landmarker.task            MediaPipe hand landmark model
notebooks/experiments.ipynb            Experiment notebook
outputs/json/                          Generated JSON and keypoint outputs
outputs/clips/                         Per-chunk MP4 clips
docs/TECHNICAL_DOCUMENTATION.md         Source-level technical documentation
rag/action_dictionary/confirmed_actions.json
                                        Curated 38-class action dictionary
rag/retriever.py                       RAG action retrieval logic
schemas/output_schema.json             Expected output JSON schema
src/exporter.py                        JSON and MP4 writing
src/fusion.py                          Qwen-VL prompts, generation, JSON parsing
src/hand_detection.py                  MediaPipe hand detection and interpolation
src/handx_features.py                  HandX/fallback feature extraction
src/rag_annotator.py                   RAG-only annotation wrapper
src/run_pipeline.py                    Main CLI runner
src/streamlit_app.py                   Local upload interface
src/video_pipeline.py                  Video metadata, motion signal, chunking
```

## Local Setup

Qwen-VL inference is much smoother with a CUDA GPU. RAG-only mode can run on weaker machines because it does not load the vision-language model.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Put videos in:

```text
data/videos/
```

Run the default RAG + Qwen-VL pipeline:

```bash
python -m src.run_pipeline data/videos/your_video.mp4
```

Run retrieval-only mode:

```bash
python -m src.run_pipeline data/videos/your_video.mp4 --annotation-mode rag
```

Run dense-window analysis:

```bash
python -m src.run_pipeline data/videos/your_video.mp4 --annotation-mode rag --chunking-mode dense
```

Use a custom config:

```bash
python -m src.run_pipeline data/videos/your_video.mp4 --settings configs/settings.yaml
```

## Google Colab

Use Colab if your local machine does not have a GPU.

1. Open Colab:

```text
https://colab.research.google.com
```

2. Enable GPU:

```text
Runtime -> Change runtime type -> GPU -> Save
```

3. Clone the repository:

```python
%cd /content
!git clone https://github.com/9Ahmed6/capstone-project-zaka.git
%cd capstone-project-zaka
```

4. Install requirements:

```python
!pip install -r requirements.txt
```

5. Upload a video:

```python
from google.colab import files
from pathlib import Path
import shutil

uploaded = files.upload()
Path("data/videos").mkdir(parents=True, exist_ok=True)

for name in uploaded:
    shutil.move(name, f"data/videos/{name}")
    video_name = name

print(video_name)
```

6. Run the pipeline:

```python
!python -m src.run_pipeline data/videos/your_video.mp4
```

For a faster retrieval-only run:

```python
!python -m src.run_pipeline data/videos/your_video.mp4 --annotation-mode rag
```

For dense-window analysis:

```python
!python -m src.run_pipeline data/videos/your_video.mp4 --annotation-mode rag --chunking-mode dense
```

7. Inspect outputs:

```python
!ls outputs/json
!ls outputs/clips
```

```python
import json
from pathlib import Path

result_path = list(Path("outputs/json").glob("*_segments*.json"))[0]
result = json.loads(result_path.read_text())
result
```

## Evaluation

Baseline evaluation utilities are in `evaluation/metrics.py`.

They provide:

- temporal IoU for segment overlap
- prediction-to-ground-truth matching
- macro-F1 after temporal matching

Programmatic example:

```python
from evaluation.metrics import evaluate_predictions

metrics = evaluate_predictions(predictions, ground_truth, iou_threshold=0.5)
print(metrics)
```

Ground-truth segments should use the same timing and `action_label` fields as the pipeline output.

## Troubleshooting

No hands detected:

- Check that the video visibly contains hands.
- Lower MediaPipe confidence thresholds in `configs/settings.yaml`.
- Try `frame_stride: 1` if you were skipping frames.

Qwen-VL is slow or runs out of memory:

- Use `--annotation-mode rag`.
- Reduce `video.max_frames_for_qwen`.
- Use a shorter video, motion chunking, or wider dense windows with less overlap.
- Run on a Colab GPU or another CUDA machine.

JSON parsing errors on long videos:

- The current parser repairs common truncated Qwen outputs.
- If failures continue, increase `qwen.max_new_tokens_vision`.
- Keep prompt outputs concise; the current prompts ask for short `summary` and `evidence` strings.

MediaPipe model missing:

- Keep `hand_detection.auto_download_model: true`, or place `hand_landmarker.task` in `models/`.

## Notes

- `confirmed_actions.json` is the source of truth for action labels.
- Qwen-VL should choose labels only from the RAG candidates.
- Raw model output is stored in prompt-based results for debugging.
- Per-chunk MP4 clips are exported so annotations can be checked visually.
- The pipeline is designed to avoid loading the full video into RAM.
