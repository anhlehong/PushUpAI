import streamlit as st
import numpy as np
import cv2
import os
import mediapipe as mp
from src.processor import VideoProcessor
from src.evaluator import PushUpEvaluator

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
            evaluator = PushUpEvaluator()
            result = evaluator.evaluate(ex_data, st_data)
            
            if result.get("error"):
                st.error(result["error"])
                st.stop()
                
            st.header(f"Kết quả hiệp tập: {result['st_reps_count']} Reps")
            
            if result["st_reps_count"] < result["ex_reps_count"]:
                st.warning(f"⚠️ Học viên chưa hoàn thành đủ số Rep mục tiêu ({result['st_reps_count']}/{result['ex_reps_count']} Reps).")
            elif result["st_reps_count"] > result["ex_reps_count"]:
                st.info(f"💪 Học viên thực hiện nhiều Rep hơn mục tiêu ({result['st_reps_count']}/{result['ex_reps_count']} Reps). Hãy chú ý dấu hiệu suy giảm thể lực ở các Rep cuối.")
            else:
                st.success(f"✅ Học viên hoàn thành đủ số Rep mục tiêu ({result['ex_reps_count']} Reps).")

            rep_results = result["rep_results"]
            
            if rep_results:
                st.metric(label="🌟 ĐIỂM TỔNG QUAN TOÀN BỘ BÀI TẬP", value=f"{result['overall_score']*100:.1f}%")
                
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
                    is_bad_rep = bool(rep['score'] < 0.8)  # Cảnh báo nếu điểm dưới 80%
                    emoji = "🚩" if is_bad_rep else "✅"
                    
                    with st.expander(f"{emoji} Rep {rep['rep_num']} - Điểm: {rep['score']*100:.1f}% (Rule: {rep['rule_score']*100:.0f}%, DTW: {rep['dtw_score']*100:.0f}%)", expanded=is_bad_rep):
                        c1, c2 = st.columns(2)
                        
                        st_frame_data = st_data[rep["w_pair"][0]]
                        ex_frame_data = ex_data[rep["w_pair"][1]]
                        img_st = processor.get_frame(st_path, st_frame_data["frame_idx"], flip=st_frame_data.get("flipped", False))
                        img_ex = processor.get_frame(ex_path, ex_frame_data["frame_idx"], flip=ex_frame_data.get("flipped", False))
                        
                        if is_bad_rep and rep["errors"]:
                            err = rep["errors"][0]
                            st.error(f"Lỗi chính: {err['message']} (Severity: {err['severity']})")
                            c1.image(render(img_st), caption=f"Khung hình lỗi nhất (Học viên)")
                            
                            # Show extra errors
                            for other_err in rep["errors"][1:]:
                                st.warning(f"Lỗi phụ: {other_err['message']}")
                        else:
                            st.success("Tư thế tốt, không phát hiện lỗi nghiêm trọng.")
                            c1.image(render(img_st), caption=f"Khung hình chênh lệch nhất (Học viên)")
                            
                        c2.image(render(img_ex), caption="Tư thế chuẩn của chuyên gia (Cùng pha)")

            os.unlink(ex_path)
            os.unlink(st_path)
