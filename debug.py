import numpy as np
from src.processor import VideoProcessor
from src.similarity import smooth_series, segment_reps

processor = VideoProcessor()
with open("data/videos/khong gong bung.mp4", 'rb') as f:
    st_data, st_path = processor.process_video_lightweight(f)

print(f"Total valid frames in data_list: {len(st_data)}")

if len(st_data) > 0:
    st_depths = smooth_series([d["depth_sig"] * 100 for d in st_data])
    st_elbows = smooth_series([d["elbow_angle"] for d in st_data])
    print(f"Max depth diff: {max(st_depths) - min(st_depths)}")
    print(f"Max elbow diff: {max(st_elbows) - min(st_elbows)}")
    
    reps = segment_reps(st_depths, st_elbows)
    print(f"Reps found: {reps}")

