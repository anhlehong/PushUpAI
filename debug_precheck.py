import numpy as np
from src.processor import VideoProcessor
from src.similarity import smooth_series, segment_reps, get_golden_template, compute_pose_similarity

processor = VideoProcessor()

with open("data/videos/push_up_template.mp4", "rb") as f:
    ex_data, _ = processor.process_video_lightweight(f)
ex_depths = smooth_series([d["depth_sig"] * 100 for d in ex_data])
ex_elbows = smooth_series([d["elbow_angle"] for d in ex_data])
ex_reps = segment_reps(ex_depths, ex_elbows)
golden_rep = get_golden_template(ex_data, ex_reps)
plank_pose = ex_data[golden_rep[0]]["pose_embedding"]

for vid in ["data/videos/khong gong bung.mp4", "data/videos/Push-Up incorrect form.mp4", "data/videos/video_vo_su.mp4"]:
    with open(vid, "rb") as f:
        st_data, _ = processor.process_video_lightweight(f)
    
    if not st_data:
        print(f"{vid}: No data")
        continue
        
    max_sim = 0
    for d in st_data:
        sim = compute_pose_similarity(d["pose_embedding"], plank_pose)
        if sim > max_sim:
            max_sim = sim
            
    print(f"Max similarity to plank for {vid}: {max_sim*100:.1f}%")

