import numpy as np
from src.processor import VideoProcessor

processor = VideoProcessor()
for vid in ["data/videos/push_up_template.mp4", "data/videos/khong gong bung.mp4", "data/videos/video_vo_su.mp4"]:
    with open(vid, "rb") as f:
        data, _ = processor.process_video_lightweight(f)
    if not data:
        print(f"{vid}: No data")
        continue
    
    # wrist Y coordinates
    wrist_y = []
    for d in data:
        # lms = d["pose_embedding"]? No, we don't have raw lms in return dict.
        pass
