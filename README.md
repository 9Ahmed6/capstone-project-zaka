# Action-Based Video Understanding System

This project is developed by:

Mentor: Patrick Saade
Guidance and project idea: Rabih Amhaz @ RA Development

Team Members:

    Nihal Alzubair
    Abdulla Mohamed
    Ahmed Mohamed
 
This project takes a video, finds the hand movements inside it, splits the video into action chunks, and creates a JSON description for each chunk.

It is built up to the **Week 5 milestone** from the capstone proposal and includes:

- video loading
- hand detection
- motion-based action chunking
- HandX-style movement features
- Qwen-VL annotation
- JSON and clip export
- baseline evaluation metrics


## What This Project Does

Imagine you have a video where someone is moving their hands. This system tries to answer:

- When does an action start?
- When does it end?
- Which hand moved?
- Was it a small finger movement, a larger hand movement, or a two-hand action?
- What is a short description of what happened?

The final output is a JSON file like this:

```json
{
  "video_id": "sample",
  "segments": [
    {
      "chunk_id": "chunk_000",
      "start_time": 1.2,
      "end_time": 3.4,
      "action_label": "open_hand",
      "movement_scale": "micro",
      "confidence": 0.72,
      "hand_side": "right",
      "summary": "The right hand opens during the chunk.",
      "evidence": "The sampled frames show the fingers extending away from the palm."
    }
  ]
}
```

## Pipeline Overview

The pipeline runs in this order:

```text
Video
 -> Extract frames
 -> Detect hand landmarks with MediaPipe
 -> Convert landmarks to HandX joint order
 -> Measure hand motion
 -> Split video into action chunks
 -> Extract HandX-style motion features
 -> Load the action library
 -> Use Qwen-VL as a text-only LLM
 -> Use the same Qwen-VL as a visual VLM
 -> Save JSON results and video clips
 -> Optionally calculate baseline metrics
```

## Step-By-Step Explanation

### 1. Load the Video

The project reads the video using OpenCV.

Main file:

```text
src/video_pipeline.py
```

The function `extract_frames()` turns the video into:

- a list of frames
- timestamps for each frame
- the video FPS

### 2. Detect Hands

The project uses the newer **MediaPipe Tasks HandLandmarker** model.

Main file:

```text
src/hand_detection.py
```

For every frame, MediaPipe returns 21 hand landmark points. Each point has:

```text
x, y, z
```

The system stores the results as:

```text
frames x hands x joints x coordinates
```

So the shape is:

```text
(number_of_frames, 2, 21, 3)
```

The two hand slots are:

```text
0 = left hand
1 = right hand
```

### 3. Convert to HandX Format

MediaPipe and HandX use a different order for the 21 hand joints.

The function `mediapipe_to_handx_order()` reorders the landmarks so they match the HandX-style format.

### 4. Fill Missing Hand Detections

Sometimes MediaPipe misses the hand for a few frames.

The function `fill_missing_hand_tracks()` fills short gaps using interpolation. This makes the motion signal smoother.

### 5. Measure Motion

The project compares each frame with the previous frame.

If the landmarks move a lot, the motion score is high.

If the landmarks barely move, the motion score is low.

Main function:

```text
compute_motion_signal()
```

### 6. Split the Video Into Chunks

The project uses a simple threshold method:

```text
motion becomes high -> action starts
motion becomes low  -> action ends
```

Main function:

```text
segment_motion_chunks()
```

Each chunk gets:

- chunk id
- start frame
- end frame
- start time
- end time

### 7. Extract HandX-Style Features

HandX project: https://handx-project.github.io/ is a research project from the University of Illinois Urbana-Champaign and collaborators at Snap Inc. focused on AI-generated bimanual hand motion, essentially teaching models to synthesize realistic two-hand movements with detailed finger articulation, contact timing, and interaction dynamics.

In this project we used HandX annotation strategy to extracts representative hand motion features from the videos, e.g., contact events and finger flexion, this will be helpful for the LLM later to generate accurate descriptions. 

Main file:

```text
src/handx_features.py
```

If the real HandX repo is available, the project uses it.

If HandX is not available, the project still runs with a simple fallback feature extractor. The fallback measures things like:

- hand speed
- hand openness
- whether one hand or both hands are active
- distance between hands

This makes the project easier to test.
Currently, the necessary HandX files are added to the repo. 

### 8. Use Qwen-VL as the LLM

Main file:

```text
src/fusion.py
```

This project uses **one Qwen-VL model** for both language and vision.

First, it sends only text to Qwen-VL including:

- HandX-style features
- action library

At this stage, Qwen-VL acts like the LLM.

It creates a first guess, such as:

```json
{
  "action_label": "open_hand",
  "movement_scale": "micro",
  "hand_side": "right",
  "confidence": 0.65
}
```

## Where the Action Library Fits

The action library is used during the Qwen-VL annotation stage.

Main file:

```text
schemas/action_library.json
```

The video pipeline detects hands or split the video into chunks. Then the action library is loaded after the system has already created a motion chunk and extracted HandX-style features from that chunk.

At that point, Qwen-VL receives:

- the motion features for one chunk
- the list of allowed action labels
- visual cues for each action label

The action library acts like the system's controlled vocabulary. It tells Qwen-VL which labels it should choose from.

Example action library item:

```json
{
  "action_label": "open_hand",
  "movement_scale": "micro",
  "hand_side": "left_or_right",
  "aliases": ["open palm", "fingers spread", "hand opens"],
  "visual_cues": [
    "fingers are extended",
    "palm area is visible",
    "gaps appear between fingers"
  ]
}
```

Without the action library, the model might invent inconsistent labels.

With the action library, the model is guided toward consistent labels such as:

```text
open_hand
close_hand
pinch
wrist_rotation
reach
grasp
two_hand_hold
hand_transfer
```

In simple terms:

```text
HandX-style features explain what moved.
The action library explains what labels are allowed.
Qwen-VL chooses the best label and writes the description.
```

### 9. Use the Same Qwen-VL as the VLM

Next, the project samples a few frames from the action chunk.

It sends Qwen-VL:

- the first text-only annotation
- sampled video frames
- frame timestamps

At this stage, the same Qwen-VL model acts like the VLM.

It checks what is visible in the frames and improves the annotation.

### 10. Save Results

Main file:

```text
src/exporter.py
```

The project saves:

- one final JSON file per video
- one small MP4 clip per detected chunk

Outputs are written to:

```text
outputs/json/
outputs/clips/
```

### 11. Evaluate the Week 5 Baseline

Main file:

```text
evaluation/metrics.py
```

If you provide ground-truth annotations, the project can calculate:

- temporal IoU
- macro-F1

These are the baseline metrics required for Week 5.

## Folder Layout

```text
configs/settings.yaml              Main settings
schemas/action_library.json         Action labels and visual cues
schemas/output_schema.json          Expected JSON output format
data/videos/                        Input videos
data/annotations/                   Ground-truth labels for evaluation
outputs/json/                       Output JSON files
outputs/clips/                      Exported action clips
src/video_pipeline.py               Frame extraction and chunking
src/hand_detection.py               MediaPipe hand detection
src/handx_features.py               HandX-style feature extraction
src/fusion.py                       Qwen-VL annotation and refinement
src/exporter.py                     JSON and clip saving
src/run_pipeline.py                 Main command-line runner
evaluation/metrics.py               Baseline metrics
notebooks/experiments.ipynb         Week 5 experiment notebook
rag/action_dictionary/              Starter action dictionary for later RAG work
```

## Run Locally

Qwen-VL is large, so a GPU is strongly recommended. If your laptop has no GPU, use Google Colab instead.

Local setup:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Put a video in:

```text
data/videos/
```

Run:

```bash
python -m src.run_pipeline data/videos/your_video.mp4
```

## How To Run On Google Colab

Use Colab if your machine does not have a GPU.

### 1. Open Colab

Go to:

[https://colab.research.google.com](https://colab.research.google.com)

### 2. Turn On GPU

In Colab:

```text
Runtime -> Change runtime type -> GPU -> Save
```

### 3. Clone GitHub Repo


```python
%cd /content
!git clone https://github.com/9Ahmed6/capstone-project-zaka.git
%cd capstone-project-zaka
```


### 4. Install Requirements

```python
!pip install -r requirements.txt
```

### 5. Add a Video

Option A: upload a video directly to Colab:

```python
from google.colab import files
uploaded = files.upload()
```

Move it into the project video folder:

```python
import shutil
from pathlib import Path

Path("data/videos").mkdir(parents=True, exist_ok=True)

for name in uploaded:
    shutil.move(name, f"data/videos/{name}")
    video_name = name

print(video_name)
```

Option B: use Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Copy a video from Drive:

```python
!cp "/content/drive/MyDrive/your_video.mp4" data/videos/
```

### 6. Run the Pipeline

Replace the file name with your video name:

```python
!python -m src.run_pipeline data/videos/your_video.mp4
```

### 7. View Outputs

List the output files:

```python
!ls outputs/json
!ls outputs/clips
```

Open the JSON result:

```python
import json
from pathlib import Path

result_path = list(Path("outputs/json").glob("*_segments.json"))[0]
result = json.loads(result_path.read_text())
result
```


## Notes

- Keep videos in Google Drive or upload them directly to Colab.
- The project uses `Qwen/Qwen2.5-VL-3B-Instruct` by default.
- For weaker GPUs, test with a short video first.
- You can reduce the number of sampled frames in `configs/settings.yaml`:

```yaml
video:
  max_frames_for_qwen: 4
```
