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

        # Đặc trưng 1: Góc khuỷu tay trung bình
        elbow_angle = (get_angle_3d(
            lms[11], lms[13], lms[15]) + get_angle_3d(lms[12], lms[14], lms[16])) / 2

        # Đặc trưng 2: Độ sâu (Chiều dọc vai so với hông) - Quan trọng để khớp pha
        shoulder_y = (lms[11][1] + lms[12][1]) / 2
        hip_y = (lms[23][1] + lms[24][1]) / 2
        depth_sig = shoulder_y - hip_y

        # Chuẩn hóa Pose Embedding (Xoay và Scale về gốc Hông)
        hip_center = (lms[23] + lms[24]) / 2
        spine_dist = np.linalg.norm((lms[11]+lms[12])/2 - hip_center)
        norm_lms = (lms - hip_center) / (spine_dist if spine_dist > 0 else 1)

        return {
            "pose_embedding": norm_lms,
            "elbow_angle": elbow_angle,
            "depth_sig": depth_sig,
            "sig": [elbow_angle, depth_sig * 100]  # Signature đa đặc trưng
        }, results.pose_landmarks
