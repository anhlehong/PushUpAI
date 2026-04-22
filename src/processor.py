import cv2
import tempfile
import shutil
from pathlib import Path
from src.engine import PoseEngine


class VideoProcessor:
    def __init__(self):
        self.engine = PoseEngine()

    def _process_video_capture(
        self,
        cap,
        sample_interval_sec=1 / 30,
        min_visibility=0.45,
        min_keypoint_visibility=0.25,
        analysis_min_visibility=0.15,
        analysis_min_keypoint_visibility=0.05,
    ):
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        fps = float(fps) if fps and fps > 1e-6 else 30.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        data_list = []
        curr_frame_idx = 0
        processed_frames = 0
        valid_pose_frames = 0
        analysis_pose_frames = 0
        conf_sum = 0.0
        analysis_conf_sum = 0.0
        last_processed_ts = -1e9

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = curr_frame_idx / fps
            should_process = sample_interval_sec <= 0 or (
                (timestamp - last_processed_ts) + 1e-9 >= sample_interval_sec
            )
            if should_process:
                last_processed_ts = timestamp
                processed_frames += 1
                res, _ = self.engine.extract_kinematics(frame)
                if res:
                    frame_conf = res.get("landmark_confidence", {})
                    conf_mean = float(frame_conf.get("mean", 0.0))
                    conf_min = float(frame_conf.get("min", 0.0))

                    if (
                        conf_mean >= analysis_min_visibility
                        and conf_min >= analysis_min_keypoint_visibility
                    ):
                        res["frame_idx"] = curr_frame_idx
                        res["timestamp"] = timestamp
                        data_list.append(res)
                        analysis_pose_frames += 1
                        analysis_conf_sum += conf_mean

                    if conf_mean >= min_visibility and conf_min >= min_keypoint_visibility:
                        valid_pose_frames += 1
                        conf_sum += conf_mean

            curr_frame_idx += 1

        total_frames = frame_count if frame_count > 0 else curr_frame_idx
        duration_sec = total_frames / fps if fps > 1e-6 else 0.0

        metadata = {
            "fps": fps,
            "total_frames": total_frames,
            "video_duration_sec": duration_sec,
            "processed_frames": processed_frames,
            "valid_pose_frames": valid_pose_frames,
            "valid_ratio": valid_pose_frames / max(processed_frames, 1),
            "mean_confidence": conf_sum / max(valid_pose_frames, 1),
            "analysis_pose_frames": analysis_pose_frames,
            "analysis_ratio": analysis_pose_frames / max(processed_frames, 1),
            "analysis_mean_confidence": analysis_conf_sum / max(analysis_pose_frames, 1),
            "sample_interval_sec": sample_interval_sec,
            "processing_fps": 1.0 / sample_interval_sec if sample_interval_sec > 1e-9 else fps,
            "quality_gate_thresholds": {
                "min_visibility": min_visibility,
                "min_keypoint_visibility": min_keypoint_visibility,
            },
            "analysis_thresholds": {
                "min_visibility": analysis_min_visibility,
                "min_keypoint_visibility": analysis_min_keypoint_visibility,
            },
        }
        return data_list, metadata

    def process_video_lightweight(
        self,
        file_buffer,
        sample_interval_sec=1 / 30,
        min_visibility=0.45,
        min_keypoint_visibility=0.25,
        analysis_min_visibility=0.15,
        analysis_min_keypoint_visibility=0.05,
    ):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        file_buffer.seek(0)
        shutil.copyfileobj(file_buffer, tfile)
        video_path = tfile.name
        tfile.close()

        cap = cv2.VideoCapture(video_path)
        data_list, metadata = self._process_video_capture(
            cap,
            sample_interval_sec=sample_interval_sec,
            min_visibility=min_visibility,
            min_keypoint_visibility=min_keypoint_visibility,
            analysis_min_visibility=analysis_min_visibility,
            analysis_min_keypoint_visibility=analysis_min_keypoint_visibility,
        )
        cap.release()
        return data_list, video_path, metadata

    def process_video_path(
        self,
        video_path,
        sample_interval_sec=1 / 30,
        min_visibility=0.45,
        min_keypoint_visibility=0.25,
        analysis_min_visibility=0.15,
        analysis_min_keypoint_visibility=0.05,
    ):
        cap = cv2.VideoCapture(str(Path(video_path)))
        data_list, metadata = self._process_video_capture(
            cap,
            sample_interval_sec=sample_interval_sec,
            min_visibility=min_visibility,
            min_keypoint_visibility=min_keypoint_visibility,
            analysis_min_visibility=analysis_min_visibility,
            analysis_min_keypoint_visibility=analysis_min_keypoint_visibility,
        )
        cap.release()
        return data_list, metadata

    def get_frame(self, video_path, frame_idx):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
