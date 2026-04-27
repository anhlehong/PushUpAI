import cv2
import tempfile
import shutil
import mediapipe as mp
import numpy as np
from src.engine import PoseEngine


class VideoProcessor:
    def __init__(self):
        self.engine = PoseEngine()

    def _detect_orientation(self, video_path):
        """
        Phát hiện hướng người tập trong video (đầu bên trái hay bên phải).
        Trả về True nếu cần flip (đầu bên trái), False nếu không cần (đầu bên phải).
        
        Logic: So sánh vị trí X trung bình của cổ tay (wrist) với hông (hip).
        Trong tư thế hít đất chuẩn (đầu phải), wrist.x > hip.x.
        Nếu ngược lại → cần flip.
        """
        cap = cv2.VideoCapture(video_path)
        detector = mp.solutions.pose.Pose(static_image_mode=True, model_complexity=0,
                                          min_detection_confidence=0.5)
        
        wrist_hip_diffs = []
        frame_idx = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Lấy mẫu 10 frame rải đều video
        sample_step = max(1, total_frames // 10)
        
        while cap.isOpened() and len(wrist_hip_diffs) < 10:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % sample_step == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = detector.process(rgb)
                if results.pose_landmarks:
                    lms = results.pose_landmarks.landmark
                    # Trung bình cổ tay trái + phải
                    wrist_x = (lms[15].x + lms[16].x) / 2
                    # Trung bình hông trái + phải
                    hip_x = (lms[23].x + lms[24].x) / 2
                    wrist_hip_diffs.append(wrist_x - hip_x)
            frame_idx += 1
        
        cap.release()
        detector.close()
        
        if not wrist_hip_diffs:
            return False  # Không phát hiện được → không flip
        
        avg_diff = np.mean(wrist_hip_diffs)
        # Nếu wrist nằm BÊN TRÁI hông (diff < 0) → đầu bên trái → cần flip
        return avg_diff < 0

    def process_video_lightweight(self, file_buffer):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        file_buffer.seek(0)
        shutil.copyfileobj(file_buffer, tfile)
        video_path = tfile.name
        tfile.close()

        # Phát hiện hướng và quyết định flip
        needs_flip = self._detect_orientation(video_path)

        cap = cv2.VideoCapture(video_path)
        data_list = []
        curr_frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if needs_flip:
                frame = cv2.flip(frame, 1)  # Flip ngang
            if curr_frame_idx % 2 == 0:
                res, _ = self.engine.extract_kinematics(frame)
                if res:
                    # Lọc trạng thái đứng (shoulder_heel_y_dist lớn)
                    # Chỉ bắt đầu ghi nhận dữ liệu khi ở trạng thái Plank/Hít đất
                    if res["shoulder_heel_y_dist"] < 0.4:
                        res["frame_idx"] = curr_frame_idx
                        res["flipped"] = needs_flip
                        data_list.append(res)
            curr_frame_idx += 1
        cap.release()
        return data_list, video_path

    def get_frame(self, video_path, frame_idx, flip=False):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret and flip:
            frame = cv2.flip(frame, 1)
        return frame if ret else None
