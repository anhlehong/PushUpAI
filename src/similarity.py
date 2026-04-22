import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks, savgol_filter


HYBRID_WEIGHTS = {
    "kinematic": 0.35,
    "posture": 0.35,
    "tempo": 0.15,
    "stability": 0.15,
}

PUSHUP_CONTEXT_THRESHOLDS = {
    "max_torso_tilt_deg": 55.0,
    "min_session_context_ratio": 0.35,
    "min_rep_context_ratio": 0.55,
}

FAULT_THRESHOLDS = {
    "lockout_angle": 150.0,
    "bottom_depth_angle": 118.0,
    "asymmetry_angle": 42.0,
    "hip_sag_offset": 0.11,
    "hip_pike_offset": -0.18,
    "neck_min_angle": 118.0,
    "elbow_flare_ratio": 0.62,
}

FAULT_LIBRARY = {
    "shallow_depth": {
        "message": "Depth is too shallow at the bottom phase.",
        "hint": "Lower chest deeper and target elbow angle below about 95-100 deg.",
    },
    "no_lockout": {
        "message": "Top phase misses full lockout.",
        "hint": "Push to full extension and finish each rep with straight arms.",
    },
    "hip_sag": {
        "message": "Hip drops below body line.",
        "hint": "Brace core and keep shoulder-hip-ankle aligned.",
    },
    "pike_hip": {
        "message": "Hip is too high during the rep.",
        "hint": "Lower hips slightly and maintain a straight plank line.",
    },
    "head_drop": {
        "message": "Neck alignment collapses.",
        "hint": "Keep gaze slightly forward and neck neutral.",
    },
    "asymmetry": {
        "message": "Left/right elbow motion is imbalanced.",
        "hint": "Load both sides evenly and match elbow timing.",
    },
    "elbow_flare": {
        "message": "Elbows flare out too much.",
        "hint": "Tuck elbows closer to about 30-45 deg from torso.",
    },
}


def _clamp01(value):
    return float(max(0.0, min(1.0, value)))


def _torso_tilt_deg(frame):
    shoulder_x = float(frame.get("shoulder_center_x", 0.0))
    hip_x = float(frame.get("hip_center_x", 0.0))
    shoulder_y = float(frame.get("shoulder_center_y", 0.0))
    hip_y = float(frame.get("hip_center_y", 0.0))

    dx = abs(hip_x - shoulder_x)
    dy = abs(hip_y - shoulder_y)
    return float(np.degrees(np.arctan2(dy, dx + 1e-6)))


def compute_pushup_context_metrics(rep_frames, thresholds=None):
    if not rep_frames:
        return {
            "context_ratio": 0.0,
            "tilt_p50": 90.0,
            "tilt_p90": 90.0,
        }

    th = PUSHUP_CONTEXT_THRESHOLDS.copy()
    if thresholds:
        th.update(thresholds)

    tilts = np.array([_torso_tilt_deg(frame)
                     for frame in rep_frames], dtype=float)
    valid_context = tilts <= float(th["max_torso_tilt_deg"])

    return {
        "context_ratio": float(np.mean(valid_context)) if len(valid_context) else 0.0,
        "tilt_p50": float(np.percentile(tilts, 50)) if len(tilts) else 90.0,
        "tilt_p90": float(np.percentile(tilts, 90)) if len(tilts) else 90.0,
    }


def estimate_sample_rate(timestamps=None, fps_fallback=30.0):
    if timestamps is not None and len(timestamps) > 1:
        ts = np.asarray(timestamps, dtype=float)
        diffs = np.diff(ts)
        diffs = diffs[diffs > 1e-6]
        if len(diffs) > 0:
            return float(1.0 / np.median(diffs))
    return float(fps_fallback if fps_fallback and fps_fallback > 1e-6 else 30.0)


def smooth_series(data):
    if len(data) < 5:
        return data
    window = 11 if len(data) >= 11 else len(data)
    if window % 2 == 0:
        window -= 1
    if window < 5:
        return data
    poly = 3 if window > 3 else 2
    return savgol_filter(data, window, poly).tolist()


def segment_reps(
    angles,
    timestamps=None,
    fps=None,
    min_peak_distance_sec=0.45,
    min_rep_sec=0.5,
    max_rep_sec=6.0,
    action_labels=None,  # New parameter for action labels
):
    """Segment push-up reps with dynamic thresholds and time-based guards."""
    if len(angles) < 8:
        return []

    sample_rate = estimate_sample_rate(timestamps, fps_fallback=fps or 30.0)
    inv_angles = 180 - np.array(angles)

    amp = float(np.percentile(inv_angles, 90) - np.percentile(inv_angles, 10))
    baseline = float(np.percentile(inv_angles, 50))
    peak_height = baseline + max(8.0, 0.2 * amp)
    prominence = max(6.0, 0.15 * amp)
    distance = max(2, int(round(min_peak_distance_sec * sample_rate)))

    peaks, _ = find_peaks(
        inv_angles, height=peak_height, prominence=prominence, distance=distance
    )
    if len(peaks) == 0:
        peaks, _ = find_peaks(inv_angles, prominence=max(
            4.0, 0.1 * amp), distance=distance)

    reps = []
    for i, p in enumerate(peaks):
        left_limit = peaks[i - 1] if i > 0 else 0
        right_limit = peaks[i +
                            1] if i < len(peaks) - 1 else len(inv_angles) - 1

        left_bound = left_limit + \
            np.argmin(inv_angles[left_limit: p + 1]) if p > left_limit else 0
        right_bound = p + \
            np.argmin(inv_angles[p: right_limit + 1]
                      ) if right_limit > p else len(inv_angles) - 1

        # Filter segments based on action labels
        if action_labels:
            segment_action = action_labels[left_bound:right_bound]
            if "push_up" not in segment_action:
                continue

        reps.append((left_bound, right_bound))

    return reps


def filter_reps_by_quality(
    reps,
    data,
    sample_rate,
    min_duration_sec=0.5,
    max_duration_sec=6.0,
    min_confidence=0.45,
    min_rom_deg=12.0,
    context_thresholds=None,
):
    th = PUSHUP_CONTEXT_THRESHOLDS.copy()
    if context_thresholds:
        th.update(context_thresholds)

    valid_reps = []
    for start, end in reps:
        rep_frames = data[start:end]
        if len(rep_frames) < 4:
            continue

        duration = (end - start) / max(sample_rate, 1e-6)
        if duration < min_duration_sec or duration > max_duration_sec:
            continue

        elbow_seq = np.array([f.get("elbow_angle", 180.0)
                             for f in rep_frames], dtype=float)
        rom = float(np.max(elbow_seq) - np.min(elbow_seq))
        conf = float(np.mean([
            f.get("landmark_confidence", {}).get("mean", 0.0) for f in rep_frames
        ]))

        context_metrics = compute_pushup_context_metrics(
            rep_frames, thresholds=th)
        context_ratio = context_metrics["context_ratio"]

        if (
            rom >= min_rom_deg
            and conf >= min_confidence
            and context_ratio >= float(th["min_rep_context_ratio"])
        ):
            valid_reps.append((start, end))
    return valid_reps


def evaluate_session_quality(
    metadata,
    data,
    min_processed_frames=25,
    min_valid_ratio=0.33,
    min_mean_confidence=0.5,
    context_thresholds=None,
):
    th = PUSHUP_CONTEXT_THRESHOLDS.copy()
    if context_thresholds:
        th.update(context_thresholds)

    reasons = []

    processed_frames = int(metadata.get("processed_frames", 0))
    valid_ratio = float(metadata.get("valid_ratio", 0.0))
    mean_conf = float(metadata.get("mean_confidence", 0.0))
    duration = float(metadata.get("video_duration_sec", 0.0))

    if processed_frames < min_processed_frames:
        reasons.append(
            "Too few processed frames; video is likely too short or too sparse.")
    if valid_ratio < min_valid_ratio:
        reasons.append(
            "Pose visibility ratio is low; adjust camera distance/angle and lighting.")
    if mean_conf < min_mean_confidence:
        reasons.append(
            "Landmark confidence is low; keep full body in frame and avoid occlusion.")
    if duration < 3.0:
        reasons.append("Video is too short for stable rep analysis.")
    if len(data) < 10:
        reasons.append("Insufficient valid pose frames for scoring.")

    context_metrics = compute_pushup_context_metrics(data, thresholds=th)
    context_ratio = float(context_metrics["context_ratio"])
    if context_ratio < float(th["min_session_context_ratio"]):
        reasons.append(
            "Video lacks push-up context (too many standing/walking frames)."
        )

    return {
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "metrics": {
            "processed_frames": processed_frames,
            "valid_frames": int(metadata.get("valid_pose_frames", len(data))),
            "valid_ratio": valid_ratio,
            "mean_confidence": mean_conf,
            "pushup_context_ratio": context_ratio,
            "torso_tilt_p50": float(context_metrics["tilt_p50"]),
            "torso_tilt_p90": float(context_metrics["tilt_p90"]),
            "fps": float(metadata.get("fps", 0.0)),
            "duration_sec": duration,
        },
        "camera_tip": "Use a diagonal side view (~45 deg), keep full body visible, distance about 1.8-2.0m.",
    }


def _rep_duration_sec(rep_frames, sample_rate):
    if not rep_frames:
        return 0.0
    if len(rep_frames) >= 2 and "timestamp" in rep_frames[0] and "timestamp" in rep_frames[-1]:
        return max(0.0, float(rep_frames[-1]["timestamp"] - rep_frames[0]["timestamp"]))
    return len(rep_frames) / max(sample_rate, 1e-6)


def _build_fault(code, severity):
    template = FAULT_LIBRARY.get(code, {})
    return {
        "code": code,
        "severity": float(severity),
        "message": template.get("message", code),
        "hint": template.get("hint", ""),
    }


def score_posture(rep_frames, thresholds=None):
    if not rep_frames:
        return 0.0, [], {}

    th = FAULT_THRESHOLDS.copy()
    if thresholds:
        th.update(thresholds)

    elbow = np.array([f.get("elbow_angle", 180.0)
                     for f in rep_frames], dtype=float)
    elbow_l = np.array([f.get("elbow_left_angle", f.get(
        "elbow_angle", 180.0)) for f in rep_frames], dtype=float)
    elbow_r = np.array([f.get("elbow_right_angle", f.get(
        "elbow_angle", 180.0)) for f in rep_frames], dtype=float)
    neck = np.array([f.get("neck_angle", 170.0)
                    for f in rep_frames], dtype=float)
    hip_offset = np.array([f.get("hip_line_offset", 0.0)
                          for f in rep_frames], dtype=float)
    flare = np.array([f.get("elbow_flare_ratio", 0.0)
                     for f in rep_frames], dtype=float)

    metrics = {
        "elbow_min": float(np.min(elbow)),
        "elbow_max": float(np.max(elbow)),
        "asymmetry_p85": float(np.percentile(np.abs(elbow_l - elbow_r), 85)),
        "hip_offset_p80": float(np.percentile(hip_offset, 80)),
        "hip_offset_p20": float(np.percentile(hip_offset, 20)),
        "neck_p10": float(np.percentile(neck, 10)),
        "flare_p85": float(np.percentile(flare, 85)),
    }

    faults = []
    penalty = 0.0

    if metrics["elbow_min"] > th["bottom_depth_angle"]:
        faults.append(_build_fault("shallow_depth", 0.85))
        penalty += 0.22

    if metrics["elbow_max"] < th["lockout_angle"]:
        faults.append(_build_fault("no_lockout", 0.8))
        penalty += 0.2

    if metrics["hip_offset_p80"] > th["hip_sag_offset"]:
        faults.append(_build_fault("hip_sag", 0.78))
        penalty += 0.18

    if metrics["hip_offset_p20"] < th["hip_pike_offset"]:
        faults.append(_build_fault("pike_hip", 0.6))
        penalty += 0.14

    if metrics["neck_p10"] < th["neck_min_angle"]:
        faults.append(_build_fault("head_drop", 0.55))
        penalty += 0.12

    if metrics["asymmetry_p85"] > th["asymmetry_angle"]:
        faults.append(_build_fault("asymmetry", 0.65))
        penalty += 0.12

    if metrics["flare_p85"] > th["elbow_flare_ratio"]:
        faults.append(_build_fault("elbow_flare", 0.45))
        penalty += 0.08

    score = _clamp01(1.0 - penalty)
    return score, faults, metrics


def score_tempo(rep_duration_sec, template_duration_sec):
    if template_duration_sec <= 1e-6 or rep_duration_sec <= 1e-6:
        return 0.7
    ratio = max(rep_duration_sec / template_duration_sec, 1e-6)
    deviation = abs(np.log(ratio))
    return _clamp01(1.0 - (deviation / 0.7))


def score_stability(rep_frames):
    if len(rep_frames) < 4:
        return 0.75

    elbow = np.array([f.get("elbow_angle", 180.0)
                     for f in rep_frames], dtype=float)
    body_line = np.array([f.get("body_line_angle", 170.0)
                         for f in rep_frames], dtype=float)
    hip_offset = np.array([f.get("hip_line_offset", 0.0)
                          for f in rep_frames], dtype=float)

    elbow_smoothed = np.array(smooth_series(elbow.tolist()), dtype=float)
    if len(elbow_smoothed) >= 3:
        jerk = float(np.mean(np.abs(np.diff(np.diff(elbow_smoothed)))))
    else:
        jerk = 0.0

    body_wobble = float(np.std(body_line))
    hip_wobble = float(np.std(hip_offset))

    instability = (
        0.45 * min(1.0, jerk / 7.0)
        + 0.35 * min(1.0, body_wobble / 10.0)
        + 0.20 * min(1.0, hip_wobble / 0.05)
    )
    return _clamp01(1.0 - instability)


def build_expert_templates(expert_data, expert_reps, sample_rate, max_templates=3):
    templates = []
    for rep_idx, (start, end) in enumerate(expert_reps):
        rep_frames = expert_data[start:end]
        if len(rep_frames) < 5:
            continue

        posture_score, _, _ = score_posture(rep_frames)
        elbow_seq = np.array([f.get("elbow_angle", 180.0)
                             for f in rep_frames], dtype=float)
        rom = float(np.max(elbow_seq) - np.min(elbow_seq)
                    ) if len(elbow_seq) else 0.0
        confidence = float(np.mean([
            f.get("landmark_confidence", {}).get("mean", 0.0) for f in rep_frames
        ]))
        duration = _rep_duration_sec(rep_frames, sample_rate)

        template_quality = (
            0.6 * posture_score
            + 0.25 * _clamp01(confidence)
            + 0.15 * _clamp01(rom / 70.0)
        )

        templates.append(
            {
                "template_idx": rep_idx,
                "range": (start, end),
                "frames": rep_frames,
                "sigs": [f["sig"] for f in rep_frames],
                "embeddings": [f["pose_embedding"] for f in rep_frames],
                "duration_sec": duration,
                "quality": template_quality,
            }
        )

    templates.sort(key=lambda x: x["quality"], reverse=True)
    return templates[:max_templates]


def compute_pose_similarity(v1, v2):
    weights = np.ones(33)
    weights[11:17] = 4.0
    weights[23:29] = 2.0
    diff = (v1 - v2) * weights[:, np.newaxis]
    return float(max(0, 1 - (np.linalg.norm(diff) / 9.5)))


def align_and_score(st_rep_sigs, ex_template_sigs, st_rep_embeddings, ex_template_embeddings):
    """Align one student rep against one expert template using DTW."""
    if not st_rep_sigs or not ex_template_sigs or not st_rep_embeddings or not ex_template_embeddings:
        return 0.0, []

    _, path = fastdtw(st_rep_sigs, ex_template_sigs, dist=euclidean)
    if not path:
        return 0.0, []

    scores = []
    for st_idx, ex_idx in path:
        sim = compute_pose_similarity(
            st_rep_embeddings[st_idx], ex_template_embeddings[ex_idx])
        scores.append(sim)
    return float(np.mean(scores)), path


def analyze_rep_hybrid(st_rep_frames, expert_templates, sample_rate, weights=None):
    if not st_rep_frames or not expert_templates:
        return {
            "score_total": 0.0,
            "score_components": {
                "kinematic": 0.0,
                "posture": 0.0,
                "tempo": 0.0,
                "stability": 0.0,
            },
            "faults": [],
            "path": [],
            "best_template": None,
            "worst_pair": (0, 0),
        }

    st_rep_sigs = [f["sig"] for f in st_rep_frames]
    st_rep_embs = [f["pose_embedding"] for f in st_rep_frames]

    best_template = None
    best_path = []
    best_kinematic = -1.0

    for template in expert_templates:
        kinematic_score, path = align_and_score(
            st_rep_sigs,
            template["sigs"],
            st_rep_embs,
            template["embeddings"],
        )
        if kinematic_score > best_kinematic:
            best_kinematic = kinematic_score
            best_path = path
            best_template = template

    best_kinematic = _clamp01(best_kinematic)
    posture_score, faults, posture_metrics = score_posture(st_rep_frames)

    rep_duration = _rep_duration_sec(st_rep_frames, sample_rate)
    template_duration = best_template.get(
        "duration_sec", 0.0) if best_template else 0.0
    tempo_score = score_tempo(rep_duration, template_duration)
    stability_score = score_stability(st_rep_frames)

    w = HYBRID_WEIGHTS.copy()
    if weights:
        w.update(weights)

    total = (
        w["kinematic"] * best_kinematic
        + w["posture"] * posture_score
        + w["tempo"] * tempo_score
        + w["stability"] * stability_score
    )

    worst_pair = (0, 0)
    worst_local_score = 1.0
    if best_template and best_path:
        for st_idx, ex_idx in best_path:
            sim = compute_pose_similarity(
                st_rep_embs[st_idx], best_template["embeddings"][ex_idx])
            if sim < worst_local_score:
                worst_local_score = sim
                worst_pair = (st_idx, ex_idx)

    return {
        "score_total": _clamp01(total),
        "score_components": {
            "kinematic": best_kinematic,
            "posture": posture_score,
            "tempo": tempo_score,
            "stability": stability_score,
        },
        "faults": faults,
        "posture_metrics": posture_metrics,
        "path": best_path,
        "best_template": best_template,
        "worst_pair": worst_pair,
    }


def summarize_top_faults(rep_results, top_k=3):
    weighted_counts = {}
    examples = {}
    for rep in rep_results:
        for fault in rep.get("faults", []):
            code = fault.get("code", "unknown")
            weighted_counts[code] = weighted_counts.get(
                code, 0.0) + float(fault.get("severity", 1.0))
            examples[code] = fault

    ranked = sorted(weighted_counts.items(),
                    key=lambda item: item[1], reverse=True)
    output = []
    for code, score in ranked[:top_k]:
        sample = examples.get(code, {})
        output.append(
            {
                "code": code,
                "weighted_count": float(score),
                "message": sample.get("message", code),
                "hint": sample.get("hint", ""),
            }
        )
    return output
