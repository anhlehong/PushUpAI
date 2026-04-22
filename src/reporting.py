import json
from datetime import datetime
from pathlib import Path
from typing import Any


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


def _to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# User Session Analysis",
        "",
        f"- Generated at: {report.get('generated_at', '-')}",
        f"- Status: {report.get('status', '-')}",
        f"- Expert video: {report.get('expert_video', '-')}",
        f"- Student video: {report.get('student_video', '-')}",
        "",
        "## Summary",
        "",
        f"- Form score: {float(report.get('summary', {}).get('form_score', 0.0)):.4f}",
        f"- Pace score: {float(report.get('summary', {}).get('tempo_score', 0.0)):.4f}",
        f"- Student reps: {int(report.get('summary', {}).get('rep_count', 0))}",
        f"- Critical reps: {int(report.get('summary', {}).get('critical_rep_count', 0))}",
        "",
        "## Top Faults",
        "",
    ]

    top_faults = report.get("top_faults", [])
    if top_faults:
        for fault in top_faults:
            lines.append(
                "- {code}: {message} | hint={hint}".format(
                    code=fault.get("code", "unknown"),
                    message=fault.get("message", ""),
                    hint=fault.get("hint", ""),
                )
            )
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Rep Scores",
            "",
            "| Rep | Form | Pace | Stability | Faults |",
            "|---|---:|---:|---:|---|",
        ]
    )

    for rep in report.get("rep_results", []):
        component_scores = rep.get("score_components", {})
        faults = [fault.get("code", "unknown") for fault in rep.get("faults", [])]
        lines.append(
            "| {rep_num} | {form:.4f} | {pace:.4f} | {stability:.4f} | {faults} |".format(
                rep_num=int(rep.get("rep_num", 0)),
                form=float(rep.get("form_score", rep.get("score_total", 0.0))),
                pace=float(rep.get("tempo_score", 0.0)),
                stability=float(component_scores.get("stability", 0.0)),
                faults=", ".join(faults) if faults else "none",
            )
        )

    return "\n".join(lines) + "\n"


def write_session_report(report: dict[str, Any], log_dir: Path | str) -> dict[str, str]:
    target_dir = Path(log_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = target_dir / f"user_session_{stamp}.json"
    md_path = target_dir / f"user_session_{stamp}.md"
    latest_json = target_dir / "user_session_latest.json"
    latest_md = target_dir / "user_session_latest.md"

    json_payload = json.dumps(_json_safe(report), indent=2, ensure_ascii=False)
    md_payload = _to_markdown(report)

    json_path.write_text(json_payload, encoding="utf-8")
    md_path.write_text(md_payload, encoding="utf-8")
    latest_json.write_text(json_payload, encoding="utf-8")
    latest_md.write_text(md_payload, encoding="utf-8")

    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "latest_json": str(latest_json),
        "latest_markdown": str(latest_md),
    }
