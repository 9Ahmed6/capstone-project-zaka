# Action-Based Video Understanding System

This project is implemented up to the Week 5 milestone from the proposal.

It follows the same approach as the notebook:

1. Load a video and extract frames.
2. Detect left/right hand landmarks with MediaPipe.
3. Convert landmarks into the HandX joint order.
4. Segment the video using a hand-motion signal.
5. Build HandX-style motion features for each chunk.
6. Use one Qwen-VL model for both jobs:
   - text-only annotation, acting like the LLM
   - frame-aware refinement, acting like the VLM
7. Export timestamped JSON and baseline metrics.

The code is intentionally simple and commented for a junior developer.

## Folder Layout

```text
configs/settings.yaml              Main settings
schemas/action_library.json         Week 1 action vocabulary
schemas/output_schema.json          Week 1 output schema
src/video_pipeline.py               Week 2 frame extraction and chunking
src/hand_detection.py               Week 2 hand landmark detection
src/handx_features.py               Week 3 HandX-style feature extraction
src/exporter.py                     Week 3 JSON and clip export
src/fusion.py                       Week 4 Qwen-VL annotation and fusion
src/run_pipeline.py                 Week 5 end-to-end baseline runner
evaluation/metrics.py               Week 5 IoU and macro-F1 metrics
notebooks/experiments.ipynb         Week 5 prompt/baseline experiment notebook
rag/action_dictionary/              Week 5 growing action dictionary folder
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Qwen-VL is large. A GPU runtime is strongly recommended.

## Run One Video

Put a video in `data/videos/`, then run:

```bash
python -m src.run_pipeline data/videos/your_video.mp4
```

The result is written to:

```text
outputs/json/your_video_segments.json
outputs/clips/
```

## Optional HandX Library

If you clone HandX locally, set the path in `configs/settings.yaml`:

```yaml
handx:
  diffusion_path: "HandX/diffusion"
```

If HandX is not available, the project still runs with a simple fallback feature extractor so the pipeline can be tested end-to-end.

