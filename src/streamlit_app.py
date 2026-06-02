from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from src.run_pipeline import run
from src.video_pipeline import inspect_video, segment_dense_chunks


def _format_duration(seconds: float) -> str:
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{int(minutes):02d}:{remaining_seconds:05.2f}"


def _segment_rows(result: dict) -> list[dict]:
    return [
        {
            "segment": segment.get("chunk_id", ""),
            "start": f"{segment.get('start_time', 0.0):.2f}s",
            "end": f"{segment.get('end_time', 0.0):.2f}s",
            "action": segment.get("action_label", "unknown"),
            "hand": segment.get("hand_side", "unknown"),
            "scale": segment.get("movement_scale", "unknown"),
            "confidence": round(float(segment.get("confidence", 0.0)), 3),
            "summary": segment.get("summary", ""),
        }
        for segment in result.get("segments", [])
    ]


def _analyze_video(
    uploaded_file,
    annotation_mode: str,
    chunking_mode: str,
    dense_window_sec: float,
    dense_overlap_sec: float,
) -> dict:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def update_progress(stage: str, progress: float, message: str) -> None:
        progress_bar.progress(progress)
        status_text.info(f"{stage.replace('_', ' ').title()}: {message}")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = Path(temp_file.name)

        result = run(
            str(temp_path),
            settings_path=str(PROJECT_ROOT / "configs" / "settings.yaml"),
            annotation_mode=annotation_mode,
            chunking_mode=chunking_mode,
            dense_window_sec=dense_window_sec,
            dense_overlap_sec=dense_overlap_sec,
            progress_callback=update_progress,
        )
        progress_bar.progress(1.0)
        status_text.success(f"Analysis complete. Found {len(result.get('segments', []))} analysis segments.")
        return result
    finally:
        if temp_path:
            temp_path.unlink(missing_ok=True)


def main() -> None:
    st.set_page_config(page_title="Hand Motion Video Analyzer", layout="wide")
    st.title("Hand Motion Video Analyzer")
    st.write("Upload a video to detect hand-motion segments and export the project JSON output.")

    uploaded_file = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])
    annotation_mode = st.radio(
        "Analysis mode",
        options=["rag", "prompt"],
        format_func=lambda mode: "RAG retrieval (faster)" if mode == "rag" else "RAG + Qwen-VL visual analysis",
        help="Qwen-VL visual analysis loads the configured local model and may take substantially longer.",
    )
    chunking_mode = st.radio(
        "Chunking mode",
        options=["dense", "motion"],
        format_func=lambda mode: "Dense overlapping windows (recommended)" if mode == "dense" else "Motion-triggered segments",
        help="Dense mode analyzes the full video, including quieter holds. Motion mode analyzes only intervals where movement crosses the configured thresholds.",
    )
    dense_window_sec = 4.0
    dense_overlap_sec = 1.0
    if chunking_mode == "dense":
        settings_columns = st.columns(2)
        dense_window_sec = settings_columns[0].slider(
            "Window duration (seconds)",
            min_value=2.0,
            max_value=10.0,
            value=4.0,
            step=0.5,
            help="Shorter windows produce more focused analysis segments.",
        )
        dense_overlap_sec = settings_columns[1].slider(
            "Window overlap (seconds)",
            min_value=0.0,
            max_value=max(dense_window_sec - 0.5, 0.0),
            value=min(1.0, dense_window_sec - 0.5),
            step=0.5,
            help="Overlap preserves context near window boundaries but increases runtime.",
        )

    if uploaded_file is None:
        st.info("Choose a video file to begin.")
        return

    st.video(uploaded_file.getvalue())

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix or ".mp4") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            preview_path = Path(temp_file.name)
        metadata = inspect_video(preview_path)
    except Exception as exc:
        st.error(f"Could not read the uploaded video: {exc}")
        return
    finally:
        if "preview_path" in locals():
            preview_path.unlink(missing_ok=True)

    columns = st.columns(4)
    columns[0].metric("Duration", _format_duration(metadata["duration_sec"]))
    columns[1].metric("Frames", metadata["frame_count"])
    columns[2].metric("FPS", f"{metadata['fps']:.2f}")
    columns[3].metric("Resolution", f"{metadata['width']} x {metadata['height']}")
    if chunking_mode == "dense" and metadata["fps"]:
        timestamps = [frame_number / metadata["fps"] for frame_number in range(metadata["frame_count"])]
        estimated_chunks = segment_dense_chunks(
            timestamps,
            window_sec=dense_window_sec,
            overlap_sec=dense_overlap_sec,
        )
        st.caption(f"Dense analysis will process approximately {len(estimated_chunks)} overlapping segments.")

    analysis_key = (
        uploaded_file.name,
        uploaded_file.size,
        annotation_mode,
        chunking_mode,
        dense_window_sec,
        dense_overlap_sec,
    )
    if st.button("Analyze video", type="primary", use_container_width=True):
        try:
            st.session_state["analysis_result"] = _analyze_video(
                uploaded_file,
                annotation_mode,
                chunking_mode,
                dense_window_sec,
                dense_overlap_sec,
            )
            suffix = "_segments_rag.json" if annotation_mode == "rag" else "_segments.json"
            st.session_state["analysis_filename"] = f"{Path(uploaded_file.name).stem}{suffix}"
            st.session_state["analysis_key"] = analysis_key
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

    result = st.session_state.get("analysis_result")
    if not result or st.session_state.get("analysis_key") != analysis_key:
        return

    st.subheader("Analysis result")
    rows = _segment_rows(result)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.warning("No analysis segments were detected. The JSON output is still available below.")

    json_text = json.dumps(result, indent=2)
    st.download_button(
        "Download JSON",
        data=json_text,
        file_name=st.session_state.get("analysis_filename", "video_segments.json"),
        mime="application/json",
        use_container_width=True,
    )
    with st.expander("View raw JSON"):
        st.json(result)


if __name__ == "__main__":
    main()
