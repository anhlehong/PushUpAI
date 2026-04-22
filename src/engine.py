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

    @staticmethod
    def _safe_angle_3d(p1, p2, p3):
        v1, v2 = p1 - p2, p3 - p2
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 180.0
        cos_angle = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    @staticmethod
    def _safe_div(num, den, default=0.0):
        return float(num / den) if den else float(default)

    def extract_kinematics(self, frame, is_static=False):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detector = self.pose_static if is_static else self.pose
        results = detector.process(rgb_frame)
        if not results.pose_landmarks:
            return None, None

        lms = np.array([[lm.x, lm.y, lm.z]
                       for lm in results.pose_landmarks.landmark])

        vis = np.array(
            [lm.visibility for lm in results.pose_landmarks.landmark])

        # Core elbow kinematics
        elbow_left = self._safe_angle_3d(lms[11], lms[13], lms[15])
        elbow_right = self._safe_angle_3d(lms[12], lms[14], lms[16])
        elbow_angle = (elbow_left + elbow_right) / 2.0

        # Torso and neck line control
        body_left = self._safe_angle_3d(lms[11], lms[23], lms[27])
        body_right = self._safe_angle_3d(lms[12], lms[24], lms[28])
        body_line_angle = (body_left + body_right) / 2.0

        neck_left = self._safe_angle_3d(lms[7], lms[11], lms[23])
        neck_right = self._safe_angle_3d(lms[8], lms[12], lms[24])
        neck_angle = (neck_left + neck_right) / 2.0

        shoulder_center = (lms[11] + lms[12]) / 2.0
        hip_center = (lms[23] + lms[24]) / 2.0
        ankle_center = (lms[27] + lms[28]) / 2.0

        shoulder_y = float(shoulder_center[1])
        hip_y = float(hip_center[1])
        ankle_y = float(ankle_center[1])
        depth_sig = shoulder_y - hip_y

        torso_len = np.linalg.norm(shoulder_center - hip_center)
        torso_len = torso_len if torso_len > 1e-6 else 1.0
        hip_line_offset = self._safe_div(
            hip_y - ((shoulder_y + ankle_y) / 2.0), torso_len)

        shoulder_width = np.linalg.norm(lms[11] - lms[12])
        shoulder_width = shoulder_width if shoulder_width > 1e-6 else 1.0
        elbow_flare_left = abs(lms[13][0] - lms[11][0]) / shoulder_width
        elbow_flare_right = abs(lms[14][0] - lms[12][0]) / shoulder_width
        elbow_flare_ratio = float((elbow_flare_left + elbow_flare_right) / 2.0)

        # Symmetry and confidence gates
        elbow_asymmetry = float(abs(elbow_left - elbow_right))
        critical_idx = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 27, 28]
        critical_visibility = vis[critical_idx]
        confidence_mean = float(np.mean(critical_visibility))
        confidence_min = float(np.min(critical_visibility))

        # Normalize pose embedding relative to hip center and torso scale
        spine_dist = np.linalg.norm(shoulder_center - hip_center)
        norm_lms = (lms - hip_center) / \
            (spine_dist if spine_dist > 1e-6 else 1.0)

        return {
            "pose_embedding": norm_lms,
            "elbow_angle": elbow_angle,
            "elbow_left_angle": elbow_left,
            "elbow_right_angle": elbow_right,
            "elbow_asymmetry": elbow_asymmetry,
            "body_line_angle": body_line_angle,
            "neck_angle": neck_angle,
            "elbow_flare_ratio": elbow_flare_ratio,
            "depth_sig": depth_sig,
            "hip_line_offset": hip_line_offset,
            "shoulder_center_y": shoulder_y,
            "hip_center_y": hip_y,
            "shoulder_center_x": float(shoulder_center[0]),
            "hip_center_x": float(hip_center[0]),
            "landmark_confidence": {
                "mean": confidence_mean,
                "min": confidence_min
            },
            "sig": [
                elbow_angle,
                depth_sig * 100,
                body_line_angle,
                elbow_asymmetry
            ],
        }, results.pose_landmarks
