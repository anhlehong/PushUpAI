import streamlit as st
import numpy as np
import cv2
import os
import mediapipe as mp
from src.processor import VideoProcessor
from src.similarity import (
    analyze_rep_hybrid,
    build_expert_templates,
    estimate_sample_rate,
    evaluate_session_quality,
    filter_reps_by_quality,
    segment_reps,
    smooth_series,
    summarize_top_faults,
)

st.set_page_config(page_title="AI Coach Startup", layout="wide")
st.title("🚀 AI Coach: Rep-by-Rep Analysis")
st.caption("Hybrid AQA v1: DTW kinematics + posture rules + tempo + stability")

ex_file = st.file_uploader("Video Chuyên Gia (1-2 cái mẫu)", type=['mp4'])
st_file = st.file_uploader("Video Học Viên (Tập nhiều rep)", type=['mp4'])


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


def render_quality_block(title, quality):
    st.markdown(f"**{title}**")
    metrics = quality["metrics"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Valid ratio", f"{metrics['valid_ratio']*100:.1f}%")
    c2.metric("Mean confidence", f"{metrics['mean_confidence']*100:.1f}%")
    c3.metric("Processed frames", f"{metrics['processed_frames']}")

    if quality["passed"]:
        st.success("Quality gate passed")
    else:
        st.error("Quality gate failed")
        for reason in quality["reasons"]:
            st.warning(reason)
    st.caption(quality["camera_tip"])


if st.button("🚀 PHÂN TÍCH TOÀN BỘ BÀI TẬP", use_container_width=True):
    if not ex_file or not st_file:
        st.warning("Vui lòng upload đủ video chuyên gia và học viên.")
    else:
        processor = VideoProcessor()
        ex_path = None
        st_path = None

        try:
            with st.status("AI đang bóc tách từng rep...") as status:
                ex_data, ex_path, ex_meta = processor.process_video_lightweight(
                    ex_file)
                st_data, st_path, st_meta = processor.process_video_lightweight(
                    st_file)
                status.update(
                    label="Đã nhận diện xong và bắt đầu chấm Hybrid AQA v1", state="complete")

            if not ex_data or not st_data:
                st.error("Không đủ dữ liệu pose hợp lệ để phân tích.")
            else:
                ex_quality = evaluate_session_quality(ex_meta, ex_data)
                st_quality = evaluate_session_quality(st_meta, st_data)

                st.subheader("Quality Gate")
                q1, q2 = st.columns(2)
                with q1:
                    render_quality_block("Video chuyên gia", ex_quality)
                with q2:
                    render_quality_block("Video học viên", st_quality)

                if not ex_quality["passed"] or not st_quality["passed"]:
                    st.warning(
                        "Vui lòng quay lại video theo hướng dẫn camera trước khi chấm điểm.")
                else:
                    ex_timestamps = [d.get("timestamp", 0.0) for d in ex_data]
                    st_timestamps = [d.get("timestamp", 0.0) for d in st_data]

                    ex_sample_rate = estimate_sample_rate(
                        ex_timestamps, ex_meta.get(
                            "processing_fps", ex_meta.get("fps", 30.0))
                    )
                    st_sample_rate = estimate_sample_rate(
                        st_timestamps, st_meta.get(
                            "processing_fps", st_meta.get("fps", 30.0))
                    )

                    ex_angles = smooth_series(
                        [d["elbow_angle"] for d in ex_data])
                    st_angles = smooth_series(
                        [d["elbow_angle"] for d in st_data])

                    ex_reps = segment_reps(
                        ex_angles, timestamps=ex_timestamps, fps=ex_sample_rate)
                    st_reps = segment_reps(
                        st_angles, timestamps=st_timestamps, fps=st_sample_rate)

                    ex_reps = filter_reps_by_quality(
                        ex_reps, ex_data, ex_sample_rate)
                    st_reps = filter_reps_by_quality(
                        st_reps, st_data, st_sample_rate)

                    if not ex_reps:
                        st.error(
                            "Không tách được rep chuẩn từ video chuyên gia.")
                    elif not st_reps:
                        st.error(
                            "Không tách được rep hợp lệ từ video học viên.")
                    else:
                        expert_templates = build_expert_templates(
                            ex_data, ex_reps, ex_sample_rate, max_templates=3
                        )
                        if not expert_templates:
                            st.error(
                                "Không tạo được template chất lượng từ video chuyên gia.")
                        else:
                            st.header(f"Kết quả hiệp tập: {len(st_reps)} Reps")
                            st.caption(
                                f"Đang dùng {len(expert_templates)} rep mẫu tốt nhất của chuyên gia để chấm Hybrid AQA v1."
                            )

                            rep_results = []
                            for rep_num, (s_start, s_end) in enumerate(st_reps, start=1):
                                rep_eval = analyze_rep_hybrid(
                                    st_data[s_start:s_end], expert_templates, st_sample_rate
                                )
                                best_template = rep_eval.get("best_template")
                                template_range = best_template.get(
                                    "range", (0, 0)) if best_template else (0, 0)

                                st_local, ex_local = rep_eval.get(
                                    "worst_pair", (0, 0))
                                st_global = min(
                                    max(s_start + st_local, 0), len(st_data) - 1)
                                ex_global = min(
                                    max(template_range[0] + ex_local, 0), len(ex_data) - 1)

                                rep_results.append(
                                    {
                                        "rep_num": rep_num,
                                        "range": (s_start, s_end),
                                        "score": rep_eval["score_total"],
                                        "score_components": rep_eval["score_components"],
                                        "faults": rep_eval["faults"],
                                        "template_idx": best_template.get("template_idx", 0) if best_template else 0,
                                        "worst_pair": (st_global, ex_global),
                                    }
                                )

                            if rep_results:
                                overall_score = float(
                                    np.mean([r["score"] for r in rep_results]))
                                st.metric(
                                    label="🌟 Điểm tổng quan toàn bộ bài tập",
                                    value=f"{overall_score*100:.1f}%",
                                )

                                avg_components = {
                                    key: float(
                                        np.mean([r["score_components"][key] for r in rep_results]))
                                    for key in ["kinematic", "posture", "tempo", "stability"]
                                }
                                k1, k2, k3, k4 = st.columns(4)
                                k1.metric(
                                    "Kinematic", f"{avg_components['kinematic']*100:.1f}%")
                                k2.metric(
                                    "Posture", f"{avg_components['posture']*100:.1f}%")
                                k3.metric(
                                    "Tempo", f"{avg_components['tempo']*100:.1f}%")
                                k4.metric(
                                    "Stability", f"{avg_components['stability']*100:.1f}%")

                                top_faults = summarize_top_faults(
                                    rep_results, top_k=3)
                                if top_faults:
                                    st.subheader("Top lỗi ưu tiên cần sửa")
                                    for idx, fault in enumerate(top_faults, start=1):
                                        st.markdown(
                                            f"{idx}. **{fault['message']}**")
                                        st.caption(f"Fix: {fault['hint']}")

                                st.subheader("🔍 Phân tích chi tiết từng Rep")
                                for rep in rep_results:
                                    is_bad_rep = bool(rep["score"] < 0.75)
                                    emoji = "🚩" if is_bad_rep else "✅"

                                    with st.expander(
                                        f"{emoji} Rep {rep['rep_num']} - Điểm: {rep['score']*100:.1f}%",
                                        expanded=is_bad_rep,
                                    ):
                                        c1, c2, c3, c4 = st.columns(4)
                                        c1.metric(
                                            "Total", f"{rep['score']*100:.1f}%")
                                        c2.metric(
                                            "Posture", f"{rep['score_components']['posture']*100:.1f}%")
                                        c3.metric(
                                            "Tempo", f"{rep['score_components']['tempo']*100:.1f}%")
                                        c4.metric(
                                            "Stability", f"{rep['score_components']['stability']*100:.1f}%")

                                        if rep["faults"]:
                                            for fault in rep["faults"]:
                                                st.warning(
                                                    f"{fault['message']} | Fix: {fault['hint']}")
                                        else:
                                            st.success(
                                                "No critical faults detected in this rep.")

                                        student_idx, expert_idx = rep["worst_pair"]
                                        img_st = processor.get_frame(
                                            st_path, st_data[student_idx]["frame_idx"])
                                        img_ex = processor.get_frame(
                                            ex_path, ex_data[expert_idx]["frame_idx"])

                                        p1, p2 = st.columns(2)
                                        p1.image(
                                            render_with_skeleton(
                                                processor, img_st),
                                            caption="Worst frame from student in this rep",
                                        )
                                        p2.image(
                                            render_with_skeleton(
                                                processor, img_ex),
                                            caption=f"Expert reference frame (template #{rep['template_idx'] + 1})",
                                        )
        finally:
            for path in (ex_path, st_path):
                if path and os.path.exists(path):
                    os.unlink(path)
