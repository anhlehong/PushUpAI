import cv2
import mediapipe as mp
import numpy as np


class PoseEngine:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        # Detector dành cho Tracking (quét video nhanh)
        self.pose = self.mp_pose.Pose(static_image_mode=False, model_complexity=1,
                                      min_detection_confidence=0.5, min_tracking_confidence=0.5)
        # Detector dành cho Static (vẽ frame lỗi dính chặt)
        self.pose_static = self.mp_pose.Pose(static_image_mode=True, model_complexity=1,
                                             min_detection_confidence=0.5)

    def extract_kinematics(self, frame, is_static=False):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detector = self.pose_static if is_static else self.pose
        results = detector.process(rgb_frame)
        if not results.pose_landmarks:
            return None, None

        lms = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.pose_landmarks.landmark])

        def get_angle_3d(p1, p2, p3):
            v1, v2 = p1 - p2, p3 - p2
            return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0)))

        # Tính toán riêng trái phải để đo độ lệch (Symmetry)
        left_elbow = get_angle_3d(lms[11], lms[13], lms[15])
        right_elbow = get_angle_3d(lms[12], lms[14], lms[16])
        elbow_angle = (left_elbow + right_elbow) / 2
        left_right_symmetry = abs(left_elbow - right_elbow)

        # Đặc trưng 2: Độ sâu (Chiều dọc vai so với hông)
        shoulder_y = (lms[11][1] + lms[12][1]) / 2
        hip_y = (lms[23][1] + lms[24][1]) / 2
        depth_sig = shoulder_y - hip_y

        # Góc hông trung bình (để kiểm tra lưng thẳng)
        hip_angle = (get_angle_3d(lms[11], lms[23], lms[25]) + get_angle_3d(lms[12], lms[24], lms[26])) / 2

        # Góc vai trung bình (Khuỷu tay - Vai - Hông)
        shoulder_angle = (get_angle_3d(lms[13], lms[11], lms[23]) + get_angle_3d(lms[14], lms[12], lms[24])) / 2

        # Góc đường cơ thể (Vai - Hông - Gót chân)
        body_line_angle = (get_angle_3d(lms[11], lms[23], lms[29]) + get_angle_3d(lms[12], lms[24], lms[30])) / 2
        
        # Góc đầu (Tai - Vai - Hông)
        head_angle = (get_angle_3d(lms[7], lms[11], lms[23]) + get_angle_3d(lms[8], lms[12], lms[24])) / 2

        # Khoảng cách dọc Vai - Gót chân (Dùng cho lọc trạng thái đứng/plank)
        heel_y = (lms[29][1] + lms[30][1]) / 2
        shoulder_heel_y_dist = abs(shoulder_y - heel_y)

        # Chuẩn hóa Pose Embedding (Xoay và Scale về gốc Hông)
        hip_center = (lms[23] + lms[24]) / 2
        spine_dist = np.linalg.norm((lms[11]+lms[12])/2 - hip_center)
        norm_lms = (lms - hip_center) / (spine_dist if spine_dist > 0 else 1)

        return {
            "pose_embedding": norm_lms,
            "elbow_angle": elbow_angle,
            "hip_angle": hip_angle,
            "shoulder_angle": shoulder_angle,
            "body_line_angle": body_line_angle,
            "head_angle": head_angle,
            "left_right_symmetry": left_right_symmetry,
            "depth_sig": depth_sig,
            "shoulder_heel_y_dist": shoulder_heel_y_dist,
            "sig": [elbow_angle, depth_sig * 100]  # Signature đa đặc trưng
        }, results.pose_landmarks
