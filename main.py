from datetime import datetime
from io import BytesIO
from pathlib import Path
import os

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

from src.processor import VideoProcessor
from src.reference import load_expert_reference_cache
from src.reporting import write_session_report
from src.similarity import (
    FORM_WEIGHTS,
    FAULT_THRESHOLDS,
    analyze_rep_hybrid,
    evaluate_session_quality,
    detect_valid_reps,
    estimate_sample_rate,
    summarize_top_faults,
)


ROOT_DIR = Path(__file__).resolve().parent
EXPERT_VIDEO_PATH = ROOT_DIR / "push_up_template.mp4"
LOG_DIR = ROOT_DIR / "logs"
CACHE_DIR = ROOT_DIR / "cache"
CRITICAL_FORM_THRESHOLD = 0.74
CRITICAL_FAULT_SEVERITY = 0.72

st.set_page_config(page_title="Push-Up Form Coach", layout="wide")


def inject_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {
            --bg-start: #fff8ee;
            --bg-end: #edf2f7;
            --ink: #152238;
            --muted: #546179;
            --card: rgba(255, 255, 255, 0.82);
            --accent: #d2652d;
            --accent-soft: rgba(210, 101, 45, 0.12);
            --ok: #0f766e;
            --warn: #b54708;
            --bad: #b42318;
            --line: rgba(21, 34, 56, 0.08);
        }

        html, body, [class*="css"] {
            font-family: "IBM Plex Sans", sans-serif;
            color: var(--ink);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, var(--bg-start) 0%, rgba(255, 248, 238, 0.7) 35%, transparent 60%),
                linear-gradient(135deg, var(--bg-start) 0%, var(--bg-end) 100%);
        }

        h1, h2, h3 {
            font-family: "Space Grotesk", sans-serif;
            letter-spacing: -0.02em;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        [data-testid="stMetric"] {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            box-shadow: 0 18px 40px rgba(21, 34, 56, 0.08);
        }

        .hero-card {
            background: var(--card);
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1.15rem 1.25rem;
            box-shadow: 0 20px 48px rgba(21, 34, 56, 0.08);
            margin-bottom: 1rem;
        }

        .chip-row {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-top: 0.75rem;
        }

        .chip {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            padding: 0.4rem 0.75rem;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--ink);
            font-size: 0.9rem;
            font-weight: 600;
        }

        .score-card {
            background: linear-gradient(135deg, rgba(210, 101, 45, 0.16) 0%, rgba(255, 255, 255, 0.9) 100%);
            border: 1px solid rgba(210, 101, 45, 0.18);
            border-radius: 24px;
            padding: 1.3rem 1.4rem;
            box-shadow: 0 24px 54px rgba(21, 34, 56, 0.1);
            margin-bottom: 1rem;
        }

        .score-title {
            color: var(--muted);
            font-size: 0.95rem;
            font-weight: 600;
            margin-bottom: 0.25rem;
        }

        .score-value {
            font-family: "Space Grotesk", sans-serif;
            font-size: 3rem;
            font-weight: 700;
            line-height: 1;
            margin-bottom: 0.4rem;
        }

        .score-note {
            color: var(--muted);
            font-size: 0.95rem;
        }

        .fault-card {
            background: rgba(255, 255, 255, 0.84);
            border: 1px solid var(--line);
            border-left: 4px solid var(--accent);
            border-radius: 18px;
            padding: 0.95rem 1rem;
            margin-bottom: 0.85rem;
        }

        .fault-title {
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .fault-fix {
            color: var(--muted);
            font-size: 0.95rem;
        }

        .tiny-note {
            color: var(--muted);
            font-size: 0.9rem;
        }

        .video-shell {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid var(--line);
            border-radius: 20px;
            padding: 1rem;
            box-shadow: 0 18px 36px rgba(21, 34, 56, 0.06);
        }

        .video-stage {
            min-height: 0;
        }

        [data-testid="stVideo"] {
            border: 1px solid var(--line);
            border-radius: 18px;
            overflow: hidden;
            background: rgba(21, 34, 56, 0.96);
            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.04);
        }

        [data-testid="stVideo"] video {
            width: 100%;
            aspect-ratio: 16 / 9;
            object-fit: contain;
            background: rgba(21, 34, 56, 0.96);
        }

        .video-empty {
            aspect-ratio: 16 / 9;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            border: 1px dashed rgba(21, 34, 56, 0.18);
            border-radius: 16px;
            background: rgba(237, 242, 247, 0.55);
            color: var(--muted);
            padding: 1rem;
        }

        [data-testid="stFileUploader"] {
            width: 100%;
        }

        [data-testid="stFileUploaderDropzone"] {
            min-height: 132px;
            border-radius: 18px;
            border: 1.5px dashed rgba(210, 101, 45, 0.34);
            background: linear-gradient(135deg, rgba(210, 101, 45, 0.08) 0%, rgba(255, 255, 255, 0.76) 100%);
            padding: 1rem;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] {
            text-align: center;
        }

        [data-testid="stFileUploaderDropzoneInstructions"] span {
            font-weight: 600;
            color: var(--ink);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_with_skeleton(processor, img):
    if img is None:
        return None
    disp = img.copy()
    _, lms = processor.engine.extract_kinematics(disp, is_static=True)
    if lms:
        mp.solutions.drawing_utils.draw_landmarks(
            disp, lms, mp.solutions.pose.POSE_CONNECTIONS
        )
    return cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)


def safe_mean(values):
    if not values:
        return 0.0
    return float(np.mean(values))


def score_band(score):
    if score >= 0.9:
        return "Rất tốt"
    if score >= 0.78:
        return "Khá ổn"
    if score >= 0.62:
        return "Cần chỉnh thêm"
    return "Cần sửa nhiều"


def is_critical_rep(rep):
    if float(rep.get("form_score", 0.0)) < CRITICAL_FORM_THRESHOLD:
        return True
    if float(rep.get("score_components", {}).get("posture", 0.0)) < 0.82:
        return True
    return any(
        float(fault.get("severity", 0.0)) >= CRITICAL_FAULT_SEVERITY
        for fault in rep.get("faults", [])
    )


@st.cache_data(show_spinner=False)
def load_video_bytes(path_str):
    return Path(path_str).read_bytes()


def render_empty_video_slot(message):
    st.markdown(
        f"""
        <div class="video-empty">
            {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner="Đang nạp cache video mẫu cố định...")
def load_expert_reference():
    if not EXPERT_VIDEO_PATH.exists():
        raise FileNotFoundError(f"Missing expert video: {EXPERT_VIDEO_PATH}")
    try:
        reference = load_expert_reference_cache(EXPERT_VIDEO_PATH, CACHE_DIR)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{exc} Hãy chạy `.venv/bin/python scripts/precompute_reference.py` trước khi mở app."
        ) from exc
    except ValueError as exc:
        raise RuntimeError(
            f"{exc} Hãy chạy `.venv/bin/python scripts/precompute_reference.py --force` để build lại cache."
        ) from exc
    if not reference["quality"]["passed"]:
        raise RuntimeError("Expert reference video does not pass quality gate.")
    if not reference["templates"]:
        raise RuntimeError("No expert templates were built from the fixed reference video.")
    return reference


def finalize_report(report):
    report["generated_at"] = datetime.now().isoformat(timespec="seconds")
    try:
        report["log_files"] = write_session_report(report, LOG_DIR)
    except Exception as exc:
        report["log_files"] = {}
        report["log_error"] = f"Failed to write session log: {exc}"
    return report


def analyze_student_upload(student_name, student_bytes, expert_ref):
    processor = VideoProcessor()
    student_temp_path = None
    student_data = []

    report = {
        "status": "started",
        "expert_video": expert_ref["video_name"],
        "student_video": student_name,
        "expert_reference": {
            "template_count": expert_ref["template_count"],
            "rep_count": len(expert_ref["reps"]),
            "quality": expert_ref["quality"],
        },
        "quality": {
            "expert": expert_ref["quality"],
            "student": None,
        },
        "summary": {
            "form_score": 0.0,
            "tempo_score": 0.0,
            "rep_count": 0,
            "critical_rep_count": 0,
        },
        "top_faults": [],
        "rep_results": [],
    }

    try:
        student_data, student_temp_path, student_meta = processor.process_video_lightweight(
            BytesIO(student_bytes)
        )
        student_quality = evaluate_session_quality(student_meta, student_data)
        report["quality"]["student"] = student_quality
        report["student_meta"] = student_meta

        if not student_data:
            report.update(
                {
                    "status": "no_pose_data",
                    "message": "Không đủ dữ liệu pose hợp lệ để phân tích video người tập.",
                }
            )
            return finalize_report(report), processor, student_data, student_temp_path

        if not student_quality["passed"]:
            report.update(
                {
                    "status": "quality_gate_failed",
                    "message": "Video người tập chưa đạt điều kiện kỹ thuật để chấm form.",
                }
            )
            return finalize_report(report), processor, student_data, student_temp_path

        st_timestamps = [frame.get("timestamp", 0.0) for frame in student_data]
        st_sample_rate = estimate_sample_rate(
            st_timestamps,
            student_meta.get("processing_fps", student_meta.get("fps", 30.0)),
        )
        st_reps, rep_detection = detect_valid_reps(
            student_data,
            timestamps=st_timestamps,
            sample_rate=st_sample_rate,
        )
        report["rep_detection"] = rep_detection

        if not st_reps:
            report.update(
                {
                    "status": "no_student_rep",
                    "message": "Không tìm thấy rep hít đất hợp lệ trong video người tập.",
                }
            )
            return finalize_report(report), processor, student_data, student_temp_path

        rep_results = []
        for rep_num, (start, end) in enumerate(st_reps, start=1):
            rep_eval = analyze_rep_hybrid(
                student_data[start:end],
                expert_ref["templates"],
                st_sample_rate,
            )
            best_template = rep_eval.get("best_template") or {}
            template_range = best_template.get("range", (0, 0))

            st_local, ex_local = rep_eval.get("worst_pair", (0, 0))
            st_global = min(max(start + st_local, 0), len(student_data) - 1)
            ex_global = min(
                max(template_range[0] + ex_local, 0),
                len(expert_ref["data"]) - 1,
            )

            rep_result = {
                "rep_num": rep_num,
                "range": (int(start), int(end)),
                "score_total": float(rep_eval.get("score_total", 0.0)),
                "form_score": float(rep_eval.get("form_score", 0.0)),
                "tempo_score": float(rep_eval.get("tempo_score", 0.0)),
                "score_components": {
                    key: float(value)
                    for key, value in rep_eval.get("score_components", {}).items()
                },
                "faults": rep_eval.get("faults", []),
                "template_idx": (
                    int(best_template.get("template_idx"))
                    if best_template.get("template_idx") is not None
                    else 0
                ),
                "worst_pair": (st_global, ex_global),
            }
            rep_result["critical"] = is_critical_rep(rep_result)
            rep_results.append(rep_result)

        avg_form_components = {
            key: safe_mean([rep["score_components"].get(key, 0.0) for rep in rep_results])
            for key in ["kinematic", "posture", "stability"]
        }
        critical_reps = [rep for rep in rep_results if rep["critical"]]
        top_faults = summarize_top_faults(rep_results, top_k=3)

        report.update(
            {
                "status": "ok",
                "message": "Đã chấm xong form của người tập.",
                "top_faults": top_faults,
                "rep_results": rep_results,
                "summary": {
                    "form_score": safe_mean([rep["form_score"] for rep in rep_results]),
                    "tempo_score": safe_mean([rep["tempo_score"] for rep in rep_results]),
                    "rep_count": len(rep_results),
                    "critical_rep_count": len(critical_reps),
                    "avg_form_components": avg_form_components,
                },
            }
        )
        return finalize_report(report), processor, student_data, student_temp_path
    except Exception as exc:
        report.update(
            {
                "status": "error",
                "message": str(exc),
            }
        )
        return finalize_report(report), processor, student_data, student_temp_path


def render_quality_feedback(student_quality):
    if not student_quality:
        st.error("Không đọc được quality gate của video người tập.")
        return

    if student_quality["passed"]:
        st.success("Video của bạn đủ điều kiện kỹ thuật để chấm form.")
        return

    st.error("Video của bạn chưa đạt điều kiện kỹ thuật để chấm form.")
    for reason in student_quality["reasons"]:
        st.warning(reason)
    st.caption(student_quality["camera_tip"])


def render_fault_cards(top_faults):
    if not top_faults:
        st.success("Chưa thấy lỗi form nghiêm trọng. Bạn đang giữ kỹ thuật khá ổn.")
        return

    st.subheader("Ưu tiên sửa trước")
    for fault in top_faults:
        st.markdown(
            """
            <div class="fault-card">
                <div class="fault-title">{message}</div>
                <div class="fault-fix">Cách sửa: {hint}</div>
            </div>
            """.format(
                message=fault.get("message", ""),
                hint=fault.get("hint", ""),
            ),
            unsafe_allow_html=True,
        )


def render_critical_reps(report, processor, student_data, student_path, expert_ref):
    critical_reps = [rep for rep in report["rep_results"] if rep.get("critical")]
    if not critical_reps:
        st.success("Không có rep lỗi nghiêm trọng cần soi lại từng khung hình.")
        return

    st.subheader("Rep cần xem lại")
    for rep in critical_reps:
        with st.expander(
            f"Rep {rep['rep_num']} cần sửa - Form {rep['form_score'] * 100:.1f}%",
            expanded=True,
        ):
            c1, c2, c3 = st.columns(3)
            c1.metric("Điểm form", f"{rep['form_score'] * 100:.1f}%")
            c2.metric("Độ ổn định", f"{rep['score_components']['stability'] * 100:.1f}%")
            c3.metric("Nhịp độ tham khảo", f"{rep['tempo_score'] * 100:.1f}%")

            if rep["faults"]:
                for fault in rep["faults"]:
                    st.warning(f"{fault['message']} | Cách sửa: {fault['hint']}")
            else:
                st.info("Rep này thấp điểm chủ yếu do khác quỹ đạo mẫu hoặc chưa ổn định.")

            student_idx, expert_idx = rep["worst_pair"]
            img_st = processor.get_frame(student_path, student_data[student_idx]["frame_idx"])
            img_ex = processor.get_frame(
                expert_ref["video_path"],
                expert_ref["data"][expert_idx]["frame_idx"],
            )

            p1, p2 = st.columns(2)
            p1.image(
                render_with_skeleton(processor, img_st),
                caption="Khung người tập cần sửa",
                use_container_width=True,
            )
            p2.image(
                render_with_skeleton(processor, img_ex),
                caption="Khung mẫu đúng để đối chiếu",
                use_container_width=True,
            )


def render_admin_panel(report, expert_ref):
    with st.expander("Admin log và số kỹ thuật", expanded=False):
        summary = report.get("summary", {})
        form_components = summary.get("avg_form_components", {})

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Kinematic", f"{form_components.get('kinematic', 0.0) * 100:.1f}%")
        a2.metric("Posture", f"{form_components.get('posture', 0.0) * 100:.1f}%")
        a3.metric("Stability", f"{form_components.get('stability', 0.0) * 100:.1f}%")
        a4.metric("Tempo riêng", f"{summary.get('tempo_score', 0.0) * 100:.1f}%")

        st.caption(
            f"Form weights: {FORM_WEIGHTS} | Fault thresholds: {FAULT_THRESHOLDS}"
        )

        st.markdown("**Quality gate**")
        q1, q2 = st.columns(2)
        with q1:
            st.write("Video mẫu")
            st.json(expert_ref["quality"])
        with q2:
            st.write("Video người tập")
            st.json(report["quality"]["student"])

        st.markdown("**Log files**")
        log_files = report.get("log_files", {})
        if report.get("log_error"):
            st.warning(report["log_error"])
        for key, path in log_files.items():
            st.code(f"{key}: {path}")

        st.markdown("**Reference cache**")
        st.json(
            {
                "cache_status": expert_ref.get("cache_status"),
                "preview_video_path": expert_ref.get("preview_video_path"),
                "rep_detection": expert_ref.get("rep_detection", {}),
            }
        )

        if report.get("rep_detection"):
            st.markdown("**Rep detection (student)**")
            st.json(report["rep_detection"])

        st.markdown("**Raw session payload**")
        st.json(report)


inject_styles()

st.markdown(
    """
    <div class="hero-card">
        <h1>Push-Up Form Coach</h1>
        <div class="tiny-note">
            Chỉ chấm kỹ thuật hít đất. Nhịp độ được tách riêng để tham khảo, không kéo tụt điểm form.
        </div>
        <div class="chip-row">
            <div class="chip">Mẫu chuẩn cố định</div>
            <div class="chip">Chỉ xử lý video người tập</div>
            <div class="chip">Log chi tiết vẫn lưu cho admin</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    expert_ref = load_expert_reference()
except Exception as exc:
    st.error(f"Không khởi tạo được video mẫu cố định: {exc}")
    st.stop()

expert_bytes = load_video_bytes(str(EXPERT_VIDEO_PATH))

v1, v2 = st.columns(2, gap="large")
with v1:
    with st.container():
        st.subheader("Video mẫu chuẩn")
        st.caption("Player dùng file gốc `push_up_template.mp4` để luôn xem được; cache chỉ dùng cho phân tích.")
        st.video(expert_bytes)
        st.caption(
            f"Đã sẵn sàng {len(expert_ref['reps'])} rep chuẩn, dùng {expert_ref['template_count']} template tốt nhất."
        )
        st.caption(f"Reference cache: {expert_ref.get('cache_status', 'unknown')}")

with v2:
    st.subheader("Video người tập")
    st.caption("Video học viên hiển thị cùng hàng với video mẫu; bạn có thể bấm chọn hoặc kéo thả file `.mp4` vào ô upload.")
    student_video_slot = st.empty()
    student_upload = st.file_uploader(
        "Bấm chọn hoặc kéo thả video người tập (.mp4)",
        type=["mp4"],
        accept_multiple_files=False,
        key="student_upload_right_column",
    )
    student_bytes = student_upload.getvalue() if student_upload else None
    if student_bytes:
        student_video_slot.video(student_bytes)
    else:
        with student_video_slot.container():
            render_empty_video_slot("Video người tập sẽ hiện ở đây sau khi bạn bấm chọn hoặc kéo thả file vào ô upload.")
    st.caption("Quay góc chéo khoảng 45 độ, thấy toàn thân, cách máy khoảng 1.8-2.0m.")

analyze_clicked = st.button(
    "Chấm form người tập",
    use_container_width=True,
    disabled=student_bytes is None,
)

if analyze_clicked and student_upload and student_bytes:
    student_path = None
    student_data = []
    ui_processor = None

    try:
        with st.status("Đang xử lý video người tập...", expanded=False) as status:
            report, ui_processor, student_data, student_path = analyze_student_upload(
                student_upload.name,
                student_bytes,
                expert_ref,
            )
            status.update(label="Đã phân tích xong.", state="complete")

        render_quality_feedback(report["quality"]["student"])

        if report["status"] != "ok":
            st.warning(report.get("message", "Không thể chấm form cho video này."))
            if report.get("log_files"):
                st.caption(f"Admin log: {report['log_files']['latest_json']}")
            render_admin_panel(report, expert_ref)
        else:
            summary = report["summary"]
            form_score = float(summary["form_score"])
            pace_score = float(summary["tempo_score"])
            form_components = summary["avg_form_components"]

            st.markdown(
                """
                <div class="score-card">
                    <div class="score-title">Điểm form tổng quan</div>
                    <div class="score-value">{value}</div>
                    <div class="score-note">{band} • Chấm theo kỹ thuật, không cộng tempo vào điểm chính.</div>
                </div>
                """.format(
                    value=f"{form_score * 100:.1f}%",
                    band=score_band(form_score),
                ),
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Rep hợp lệ", f"{summary['rep_count']}")
            m2.metric("Rep cần sửa", f"{summary['critical_rep_count']}")
            m3.metric("Độ ổn định", f"{form_components['stability'] * 100:.1f}%")
            m4.metric("Nhịp độ tham khảo", f"{pace_score * 100:.1f}%")

            st.caption(
                "Nhịp độ chỉ để tham khảo. Điểm form được tính từ kinematic, posture và stability."
            )

            render_fault_cards(report["top_faults"])
            render_critical_reps(
                report,
                ui_processor,
                student_data,
                student_path,
                expert_ref,
            )

            if report.get("log_files"):
                st.caption(
                    f"Admin log: {report['log_files']['latest_json']} | {report['log_files']['latest_markdown']}"
                )
            elif report.get("log_error"):
                st.caption(report["log_error"])
            render_admin_panel(report, expert_ref)
    finally:
        if student_path and os.path.exists(student_path):
            os.unlink(student_path)
