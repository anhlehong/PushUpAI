import cv2
import tempfile
import shutil
from src.engine import PoseEngine


class VideoProcessor:
    def __init__(self):
        self.engine = PoseEngine()

    def process_video_lightweight(self, file_buffer):
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        file_buffer.seek(0)
        shutil.copyfileobj(file_buffer, tfile)
        video_path = tfile.name
        tfile.close()

        cap = cv2.VideoCapture(video_path)
        data_list = []
        curr_frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if curr_frame_idx % 2 == 0:
                # SỬA LỖI: Gọi đúng tên hàm extract_kinematics
                res, _ = self.engine.extract_kinematics(frame)
                if res:
                    res["frame_idx"] = curr_frame_idx
                    data_list.append(res)
            curr_frame_idx += 1
        cap.release()
        return data_list, video_path

    def get_frame(self, video_path, frame_idx):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
