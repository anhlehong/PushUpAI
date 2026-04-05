import streamlit as st
import numpy as np
import cv2
import os
import mediapipe as mp
from src.processor import VideoProcessor
from src.similarity import *

st.set_page_config(page_title="AI Coach Startup", layout="wide")
st.title("🚀 AI Coach: Rep-by-Rep Analysis")

ex_file = st.file_uploader("Video Chuyên Gia (1-2 cái mẫu)", type=['mp4'])
st_file = st.file_uploader("Video Học Viên (Tập nhiều rep)", type=['mp4'])

if st.button("🚀 PHÂN TÍCH TOÀN BỘ BÀI TẬP", use_container_width=True):
    if ex_file and st_file:
        processor = VideoProcessor()
        with st.status("AI đang bóc tách từng rep...") as status:
            ex_data, ex_path = processor.process_video_lightweight(ex_file)
            st_data, st_path = processor.process_video_lightweight(st_file)
            status.update(label="Đã nhận diện xong các hiệp tập!",
                          state="complete")

        if ex_data and st_data:
            # 1. XÁC ĐỊNH MẪU CHUẨN (Cắt rep đầu tiên của chuyên gia làm Template)
            ex_angles = smooth_series([d["elbow_angle"] for d in ex_data])
            ex_reps = segment_reps(ex_angles)
            if not ex_reps:
                st.error("Chuyên gia chưa thực hiện động tác chuẩn.")
                st.stop()

            # Lấy Rep 1 của chuyên gia làm 'Vàng'
            ex_start, ex_end = ex_reps[0]
            ex_template_sigs = [ex_data[i]["sig"]
                                for i in range(ex_start, ex_end)]
            ex_template_embs = [ex_data[i]["pose_embedding"]
                                for i in range(ex_start, ex_end)]

            # 2. PHÂN TÍCH 12 REP CỦA HỌC VIÊN
            st_angles = smooth_series([d["elbow_angle"] for d in st_data])
            st_reps = segment_reps(st_angles)

            st.header(f"Kết quả hiệp tập: {len(st_reps)} Reps")

            rep_results = []
            for i, (s_start, s_end) in enumerate(st_reps):
                st_rep_sigs = [st_data[j]["sig"]
                               for j in range(s_start, s_end)]
                st_rep_embs = [st_data[j]["pose_embedding"]
                               for j in range(s_start, s_end)]

                # So sánh Rep này với Template
                score, path = align_and_score(
                    st_rep_sigs, ex_template_sigs, st_rep_embs, ex_template_embs)
                rep_results.append(
                    {"rep_num": i+1, "score": score, "path": path, "range": (s_start, s_end)})

            if rep_results:
                overall_score = np.mean([r["score"] for r in rep_results])
                st.metric(label="🌟 ĐIỂM TỔNG QUAN TOÀN BỘ BÀI TẬP", value=f"{overall_score*100:.1f}%")
                
                def render(img):
                    if img is None:
                        return None
                    disp = img.copy()
                    res, lms = processor.engine.extract_kinematics(
                        disp, is_static=True)  # Skeleton dính chặt
                    if lms:
                        mp.solutions.drawing_utils.draw_landmarks(
                            disp, lms, mp.solutions.pose.POSE_CONNECTIONS)
                    return cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

                st.subheader("🔍 Phân tích chi tiết từng Rep")
                for rep in rep_results:
                    # Tính frame lỗi nhất cho rep này
                    w_score, w_pair = 1.0, (0, 0)
                    s_range = rep["range"]
                    for s_idx, e_idx in rep["path"]:
                        sim = compute_pose_similarity(
                            st_data[s_range[0] + s_idx]["pose_embedding"], ex_template_embs[e_idx])
                        if sim < w_score:
                            w_score, w_pair = sim, (s_range[0] + s_idx, ex_start + e_idx)

                    is_bad_rep = bool(rep['score'] < 0.8)  # Cảnh báo nếu điểm dưới 80%
                    emoji = "🚩" if is_bad_rep else "✅"
                    
                    with st.expander(f"{emoji} Rep {rep['rep_num']} - Điểm: {rep['score']*100:.1f}%", expanded=is_bad_rep):
                        c1, c2 = st.columns(2)
                        img_st = processor.get_frame(
                            st_path, st_data[w_pair[0]]["frame_idx"])
                        img_ex = processor.get_frame(
                            ex_path, ex_data[w_pair[1]]["frame_idx"])
                            
                        c1.image(render(img_st),
                                 caption=f"Khung hình lỗi nhất của học viên")
                        c2.image(render(img_ex), caption="Tư thế đúng của chuyên gia (Cùng pha)")

            os.unlink(ex_path)
            os.unlink(st_path)
