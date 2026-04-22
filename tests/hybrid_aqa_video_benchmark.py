from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.similarity import (
    FAULT_THRESHOLDS,
    FORM_WEIGHTS,
    analyze_rep_hybrid,
    detect_valid_reps,
    estimate_sample_rate,
    evaluate_session_quality,
    summarize_top_faults,
)
from src.processor import VideoProcessor
from src.reference import build_expert_reference_cache


EXPERT_VIDEO_FILENAME = "push_up_template.mp4"

CASE_DEFINITIONS = {
    "self_baseline": {
        "student": "push_up_template.mp4",
        "description": "Template video compared with itself as the student upload.",
    },
    "neck_drop_case": {
        "student": "Push-Up incorrect form.mp4",
        "description": "Incorrect form with neck drop toward end of rep.",
    },
    "core_brace_case": {
        "student": "khong gong bung.mp4",
        "description": "Incorrect form with poor core bracing.",
    },
    "non_pushup_case": {
        "student": "video_vo_su.mp4",
        "description": "Non push-up video should be rejected before scoring.",
    },
}


def _video_path(filename: str) -> Path:
    return ROOT_DIR / filename


def _process_video(processor: VideoProcessor, path: Path) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
    with path.open("rb") as handle:
        data, temp_path, metadata = processor.process_video_lightweight(handle)
    return data, temp_path, metadata


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(mean(values))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            return str(value)
    return value


def analyze_pair(
    processor: VideoProcessor,
    expert_ref: dict[str, Any],
    student_video: Path,
    case_id: str,
    case_description: str,
) -> dict[str, Any]:
    st_data = []
    st_meta: dict[str, Any] = {}
    st_temp_path: str | None = None

    try:
        st_data, st_temp_path, st_meta = _process_video(
            processor, student_video)

        st_quality = evaluate_session_quality(st_meta, st_data)

        case_result: dict[str, Any] = {
            "case_id": case_id,
            "description": case_description,
            "expert_video": expert_ref["video_name"],
            "student_video": student_video.name,
            "expert_reference": {
                "cache_status": expert_ref.get("cache_status"),
                "rep_count": len(expert_ref.get("reps", [])),
                "template_count": expert_ref.get("template_count", 0),
            },
            "quality": {
                "expert": expert_ref["quality"],
                "student": st_quality,
            },
            "rep_detection": {
                "expert": expert_ref.get("rep_detection", {}),
            },
        }

        if not expert_ref["quality"]["passed"] or not st_quality["passed"]:
            case_result.update(
                {
                    "status": "quality_gate_failed",
                    "message": "Session quality gate failed.",
                }
            )
            return case_result

        st_timestamps = [frame.get("timestamp", 0.0) for frame in st_data]

        st_sample_rate = estimate_sample_rate(
            st_timestamps, st_meta.get(
                "processing_fps", st_meta.get("fps", 30.0))
        )

        st_reps, st_rep_debug = detect_valid_reps(
            st_data, timestamps=st_timestamps, sample_rate=st_sample_rate)
        case_result["rep_detection"]["student"] = st_rep_debug

        if not st_reps:
            case_result.update(
                {
                    "status": "no_student_rep",
                    "message": "No valid student reps found.",
                }
            )
            return case_result

        templates = expert_ref.get("templates", [])
        if not templates:
            case_result.update(
                {
                    "status": "no_template",
                    "message": "No expert templates built from reps.",
                }
            )
            return case_result

        rep_results: list[dict[str, Any]] = []
        fault_counter: Counter[str] = Counter()

        for rep_num, (start, end) in enumerate(st_reps, start=1):
            rep_eval = analyze_rep_hybrid(
                st_data[start:end], templates, st_sample_rate)
            best_template = rep_eval.get("best_template") or {}
            faults = rep_eval.get("faults", [])
            fault_codes = [fault.get("code", "unknown") for fault in faults]
            fault_counter.update(fault_codes)

            rep_results.append(
                {
                    "rep_num": rep_num,
                    "range": [int(start), int(end)],
                    "form_score": float(rep_eval.get("form_score", 0.0)),
                    "tempo_score": float(rep_eval.get("tempo_score", 0.0)),
                    "score_total": float(rep_eval.get("score_total", 0.0)),
                    "score_components": {
                        key: float(value)
                        for key, value in rep_eval.get("score_components", {}).items()
                    },
                    "fault_codes": fault_codes,
                    "faults": faults,
                    "template_idx": int(best_template.get("template_idx"))
                    if best_template.get("template_idx") is not None
                    else None,
                }
            )

        total_scores = [rep["form_score"] for rep in rep_results]
        avg_components = {
            key: _safe_mean([rep["score_components"].get(key, 0.0)
                            for rep in rep_results])
            for key in ["kinematic", "posture", "stability"]
        }
        avg_tempo_score = _safe_mean([rep["tempo_score"] for rep in rep_results])
        top_faults = summarize_top_faults(rep_results, top_k=5)

        case_result.update(
            {
                "status": "ok",
                "expert_rep_count": len(expert_ref.get("reps", [])),
                "student_rep_count": len(st_reps),
                "template_count": len(templates),
                "overall_score": _safe_mean(total_scores),
                "avg_components": avg_components,
                "avg_tempo_score": avg_tempo_score,
                "fault_counts": dict(fault_counter),
                "top_faults": top_faults,
                "rep_results": rep_results,
            }
        )
        return case_result
    finally:
        for temp_path in (st_temp_path,):
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)


def _get_case_score(cases: dict[str, dict[str, Any]], case_id: str) -> float:
    case = cases.get(case_id, {})
    return float(case.get("overall_score", 0.0)) if case.get("status") == "ok" else 0.0


def _get_fault_count(cases: dict[str, dict[str, Any]], case_id: str, fault_code: str) -> int:
    case = cases.get(case_id, {})
    if case.get("status") != "ok":
        return 0
    return int(case.get("fault_counts", {}).get(fault_code, 0))


def evaluate_checks(cases: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    self_score = _get_case_score(cases, "self_baseline")
    neck_score = _get_case_score(cases, "neck_drop_case")
    core_score = _get_case_score(cases, "core_brace_case")

    for case_id in ["self_baseline", "neck_drop_case", "core_brace_case"]:
        checks.append(
            {
                "name": f"{case_id}_status_ok",
                "passed": cases.get(case_id, {}).get("status") == "ok",
                "observed": cases.get(case_id, {}).get("status", "missing"),
                "expected": "ok",
            }
        )

    checks.append(
        {
            "name": "self_baseline_high_score",
            "passed": self_score >= 0.82,
            "observed": round(self_score, 4),
            "expected": ">= 0.82",
        }
    )

    checks.append(
        {
            "name": "neck_case_lower_than_self",
            "passed": neck_score <= self_score - 0.08,
            "observed": round(neck_score, 4),
            "expected": f"<= {max(self_score - 0.08, 0.0):.4f}",
        }
    )

    checks.append(
        {
            "name": "core_case_lower_than_self",
            "passed": core_score <= self_score - 0.08,
            "observed": round(core_score, 4),
            "expected": f"<= {max(self_score - 0.08, 0.0):.4f}",
        }
    )

    head_drop_count = _get_fault_count(cases, "neck_drop_case", "head_drop")
    checks.append(
        {
            "name": "neck_case_detect_head_drop",
            "passed": head_drop_count >= 1,
            "observed": head_drop_count,
            "expected": ">= 1",
        }
    )

    hip_fault_count = _get_fault_count(cases, "core_brace_case", "hip_sag") + _get_fault_count(
        cases, "core_brace_case", "pike_hip"
    )
    checks.append(
        {
            "name": "core_case_detect_hip_fault",
            "passed": hip_fault_count >= 1,
            "observed": hip_fault_count,
            "expected": ">= 1",
        }
    )

    core_rep_count = int(cases.get("core_brace_case", {}).get("student_rep_count", 0))
    checks.append(
        {
            "name": "core_case_detects_most_reps",
            "passed": core_rep_count >= 5,
            "observed": core_rep_count,
            "expected": ">= 5",
        }
    )

    non_pushup_status = cases.get("non_pushup_case", {}).get("status", "missing")
    checks.append(
        {
            "name": "non_pushup_case_rejected",
            "passed": non_pushup_status in {"quality_gate_failed", "no_student_rep"},
            "observed": non_pushup_status,
            "expected": "quality_gate_failed or no_student_rep",
        }
    )

    return checks


def _to_markdown(report: dict[str, Any]) -> str:
    lines = []
    lines.append("# Hybrid AQA Video Benchmark Report")
    lines.append("")
    lines.append(f"- Generated at: {report['generated_at']}")
    lines.append(f"- All checks passed: {report['all_checks_passed']}")
    lines.append("")
    lines.append("## Cases")
    lines.append("")
    lines.append("| Case | Status | Form | Pace | Reps | Top faults |")
    lines.append("|---|---|---:|---:|---:|---|")

    for case_id, result in report["cases"].items():
        if result.get("status") == "ok":
            top_fault_codes = [fault.get("code", "")
                               for fault in result.get("top_faults", [])]
            lines.append(
                "| {case} | {status} | {score:.4f} | {pace:.4f} | {reps} | {faults} |".format(
                    case=case_id,
                    status=result.get("status"),
                    score=float(result.get("overall_score", 0.0)),
                    pace=float(result.get("avg_tempo_score", 0.0)),
                    reps=int(result.get("student_rep_count", 0)),
                    faults=", ".join(
                        top_fault_codes) if top_fault_codes else "none",
                )
            )
        else:
            lines.append(
                "| {case} | {status} | - | - | - | - |".format(
                    case=case_id,
                    status=result.get("status", "missing"),
                )
            )

    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Passed | Observed | Expected |")
    lines.append("|---|---|---|---|")
    for check in report["checks"]:
        lines.append(
            "| {name} | {passed} | {observed} | {expected} |".format(
                name=check["name"],
                passed="yes" if check["passed"] else "no",
                observed=check["observed"],
                expected=check["expected"],
            )
        )

    lines.append("")
    lines.append("## Config Snapshot")
    lines.append("")
    lines.append(f"- Form weights: {report['config']['form_weights']}")
    lines.append(f"- Fault thresholds: {report['config']['fault_thresholds']}")
    lines.append("")
    return "\n".join(lines)


def write_report_files(report: dict[str, Any], log_dir: Path) -> dict[str, str]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = log_dir / f"hybrid_aqa_benchmark_{stamp}.json"
    md_path = log_dir / f"hybrid_aqa_benchmark_{stamp}.md"
    latest_json = log_dir / "hybrid_aqa_benchmark_latest.json"
    latest_md = log_dir / "hybrid_aqa_benchmark_latest.md"

    json_payload = json.dumps(_json_safe(report), indent=2, ensure_ascii=False)
    markdown_payload = _to_markdown(report)

    json_path.write_text(json_payload, encoding="utf-8")
    md_path.write_text(markdown_payload, encoding="utf-8")
    latest_json.write_text(json_payload, encoding="utf-8")
    latest_md.write_text(markdown_payload, encoding="utf-8")

    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
    }


def run_benchmark(write_logs: bool = True, log_dir: Path | None = None) -> dict[str, Any]:
    cases: dict[str, dict[str, Any]] = {}
    expert_path = _video_path(EXPERT_VIDEO_FILENAME)
    expert_ref = build_expert_reference_cache(expert_path, ROOT_DIR / "cache")

    for case_id, case_cfg in CASE_DEFINITIONS.items():
        student_path = _video_path(case_cfg["student"])

        if not expert_path.exists() or not student_path.exists():
            cases[case_id] = {
                "case_id": case_id,
                "description": case_cfg["description"],
                "status": "missing_video",
                "missing": [
                    str(path.name)
                    for path in (expert_path, student_path)
                    if not path.exists()
                ],
            }
            continue

        cases[case_id] = analyze_pair(
            VideoProcessor(),
            expert_ref=expert_ref,
            student_video=student_path,
            case_id=case_id,
            case_description=case_cfg["description"],
        )

    checks = evaluate_checks(cases)
    all_passed = all(check["passed"] for check in checks)

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "cases": cases,
        "checks": checks,
        "all_checks_passed": all_passed,
        "config": {
            "form_weights": FORM_WEIGHTS,
            "fault_thresholds": FAULT_THRESHOLDS,
        },
    }

    if write_logs:
        target_log_dir = log_dir if log_dir is not None else ROOT_DIR / "logs"
        report["log_files"] = write_report_files(report, target_log_dir)

    return report


def _print_summary(report: dict[str, Any]) -> None:
    print("Hybrid AQA benchmark summary")
    print(f"all_checks_passed={report['all_checks_passed']}")

    for case_id, case in report["cases"].items():
        if case.get("status") == "ok":
            print(
                "- {case}: status={status}, form={score:.4f}, pace={pace:.4f}, reps={reps}, top_faults={faults}".format(
                    case=case_id,
                    status=case["status"],
                    score=float(case.get("overall_score", 0.0)),
                    pace=float(case.get("avg_tempo_score", 0.0)),
                    reps=int(case.get("student_rep_count", 0)),
                    faults=[fault.get("code")
                            for fault in case.get("top_faults", [])],
                )
            )
        else:
            print(
                "- {case}: status={status}, message={message}".format(
                    case=case_id,
                    status=case.get("status", "missing"),
                    message=case.get("message", case.get("missing", "-")),
                )
            )

    print("checks:")
    for check in report["checks"]:
        print(
            "  - {name}: {state} (observed={obs}, expected={exp})".format(
                name=check["name"],
                state="PASS" if check["passed"] else "FAIL",
                obs=check["observed"],
                exp=check["expected"],
            )
        )

    if report.get("log_files"):
        print("log_files:")
        for key, value in report["log_files"].items():
            print(f"  - {key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Hybrid AQA benchmark against known push-up videos."
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Run benchmark without writing JSON/Markdown logs.",
    )
    parser.add_argument(
        "--log-dir",
        default=str(ROOT_DIR / "logs"),
        help="Directory to write benchmark logs.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code when any check fails.",
    )
    args = parser.parse_args()

    report = run_benchmark(write_logs=not args.no_log,
                           log_dir=Path(args.log_dir))
    _print_summary(report)

    if args.strict and not report["all_checks_passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
