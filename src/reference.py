import pickle
from pathlib import Path

import cv2

from src.processor import VideoProcessor
from src.similarity import (
    build_expert_templates,
    detect_valid_reps,
    estimate_sample_rate,
    evaluate_session_quality,
    trim_data_to_reps_window,
)


CACHE_VERSION = 3


def _cache_paths(cache_dir):
    cache_root = Path(cache_dir)
    return (
        cache_root,
        cache_root / "expert_reference.pkl",
        cache_root / "expert_preview.mp4",
    )


def _video_signature(video_path):
    stat = Path(video_path).stat()
    return {
        "path": str(Path(video_path).resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "cache_version": CACHE_VERSION,
    }


def _clip_video(source_path, target_path, start_frame, end_frame):
    source = cv2.VideoCapture(str(source_path))
    fps = source.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(source.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    writer = cv2.VideoWriter(
        str(target_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps if fps > 1e-6 else 30.0),
        (width, height),
    )

    source.set(cv2.CAP_PROP_POS_FRAMES, max(int(start_frame), 0))
    current = int(start_frame)
    while current <= int(end_frame):
        ret, frame = source.read()
        if not ret:
            break
        writer.write(frame)
        current += 1

    source.release()
    writer.release()


def load_expert_reference_cache(video_path, cache_dir):
    video_path = Path(video_path)
    cache_root, cache_file, preview_file = _cache_paths(cache_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Missing expert video: {video_path}")
    if not cache_file.exists():
        raise FileNotFoundError(
            f"Missing cached expert reference: {cache_file}"
        )

    with cache_file.open("rb") as handle:
        payload = pickle.load(handle)

    signature = _video_signature(video_path)
    if payload.get("signature") != signature:
        raise ValueError(
            "Cached expert reference is stale for the current template video."
        )

    reference = payload.get("reference")
    if not isinstance(reference, dict):
        raise ValueError("Cached expert reference payload is invalid.")

    preview_path = Path(reference.get("preview_video_path", preview_file))
    if not preview_path.exists():
        raise FileNotFoundError(
            f"Missing cached expert preview video: {preview_path}"
        )

    loaded = dict(reference)
    loaded["video_path"] = str(video_path.resolve())
    loaded["preview_video_path"] = str(preview_path.resolve())
    loaded["cache_status"] = "disk_cache_hit"
    return loaded


def build_expert_reference_cache(
    video_path,
    cache_dir,
    sample_interval_sec=1 / 30,
    force_rebuild=False,
):
    video_path = Path(video_path)
    cache_root, cache_file, preview_file = _cache_paths(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    if not force_rebuild:
        try:
            return load_expert_reference_cache(video_path, cache_dir)
        except Exception:
            pass

    signature = _video_signature(video_path)

    processor = VideoProcessor()
    full_data, full_meta = processor.process_video_path(
        video_path,
        sample_interval_sec=sample_interval_sec,
    )
    full_quality = evaluate_session_quality(full_meta, full_data)

    full_timestamps = [frame.get("timestamp", 0.0) for frame in full_data]
    full_sample_rate = estimate_sample_rate(
        full_timestamps,
        full_meta.get("processing_fps", full_meta.get("fps", 30.0)),
    )
    reps, rep_debug = detect_valid_reps(
        full_data,
        timestamps=full_timestamps,
        sample_rate=full_sample_rate,
    )
    active_data, active_offset = trim_data_to_reps_window(
        full_data,
        reps,
        full_sample_rate,
        padding_sec=0.45,
    )
    active_timestamps = [frame.get("timestamp", 0.0) for frame in active_data]
    active_sample_rate = estimate_sample_rate(
        active_timestamps,
        full_meta.get("processing_fps", full_meta.get("fps", 30.0)),
    )
    active_reps, active_rep_debug = detect_valid_reps(
        active_data,
        timestamps=active_timestamps,
        sample_rate=active_sample_rate,
    )
    templates = build_expert_templates(
        active_data,
        active_reps,
        active_sample_rate,
        max_templates=3,
    )

    preview_start_frame = 0
    preview_end_frame = int(full_meta.get("total_frames", 0)) - 1
    if reps:
        preview_start_idx = max(int(reps[0][0]) - int(round(0.45 * full_sample_rate)), 0)
        preview_end_idx = min(
            int(reps[-1][1]) + int(round(0.45 * full_sample_rate)),
            len(full_data) - 1,
        )
        preview_start_frame = int(full_data[preview_start_idx]["frame_idx"])
        preview_end_frame = int(full_data[preview_end_idx]["frame_idx"])
    _clip_video(video_path, preview_file, preview_start_frame, preview_end_frame)

    reference = {
        "video_name": video_path.name,
        "video_path": str(video_path.resolve()),
        "preview_video_path": str(preview_file.resolve()),
        "data": active_data,
        "meta": full_meta,
        "quality": full_quality,
        "sample_rate": active_sample_rate,
        "reps": active_reps,
        "templates": templates,
        "template_count": len(templates),
        "active_offset": int(active_offset),
        "rep_detection": {
            "full_pass": rep_debug,
            "active_pass": active_rep_debug,
        },
        "cache_status": "rebuilt",
    }

    with cache_file.open("wb") as handle:
        pickle.dump(
            {
                "signature": signature,
                "reference": reference,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    return reference
