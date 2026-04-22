# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PushUpAI is a computer vision application that analyzes push-up form from video. It compares a student's push-up performance against an expert's template using MediaPipe Pose estimation and DTW (Dynamic Time Warping) alignment.

## Tech Stack

- **Python 3.11+**
- **Streamlit** - Web UI framework
- **MediaPipe Pose** - Pose estimation
- **OpenCV** - Video processing
- **SciPy** - Signal processing (savgol_filter, find_peaks)
- **fastdtw** - Dynamic Time Warping alignment

## Project Structure

```
.
├── main.py                 # Streamlit UI entry point
├── src/
│   ├── engine.py           # PoseEngine: MediaPipe pose extraction
│   ├── processor.py        # VideoProcessor: video file handling
│   └── similarity.py       # Core analysis logic (DTW, scoring, fault detection)
├── tests/
│   ├── test_video_benchmark.py
│   └── hybrid_aqa_video_benchmark.py
└── logs/                   # Benchmark results
```

## Key Architecture

### Workflow
1. **Video Processing**: `VideoProcessor.process_video_lightweight()` reads frames, samples at ~30fps, extracts pose kinematics via `PoseEngine.extract_kinematics()`
2. **Signal Processing**: `smooth_series()` applies Savitzky-Golay filter, `segment_reps()` detects push-up cycles via elbow angle peaks
3. **Template Building**: `build_expert_templates()` creates 3 best-quality expert reps
4. **Hybrid Scoring**: `analyze_rep_hybrid()` computes 4-component score:
   - **Kinematic (35%)**: DTW alignment of pose embeddings
   - **Posture (35%)**: Rule-based fault detection (depth, lockout, hip alignment, neck, asymmetry, elbow flare)
   - **Tempo (15%)**: Duration matching against template
   - **Stability (15%)**: Motion smoothness (jerk, body wobble)

### Fault Detection
The system detects 7 fault types: shallow_depth, no_lockout, hip_sag, pike_hip, head_drop, asymmetry, elbow_flare. Each has configurable thresholds in `FAULT_THRESHOLDS`.

## Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run main.py

# Run benchmark tests
python -m tests.hybrid_aqa_video_benchmark
python -m tests.test_video_benchmark
```

## Important Notes

- **Frame sampling**: Processes every frame (sample_interval_sec=1/30 default), not skip frames
- **Template selection**: Uses up to 3 highest-quality expert reps (not just first)
- **Quality gates**: Both videos must pass `evaluate_session_quality()` before scoring
- **No video resize**: Frames processed at original resolution
- **Video files**: Stored in root directory (Push-Up correct form.mp4, Push-Up incorrect form.mp4, khong gong bung.mp4)
