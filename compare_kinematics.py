import os, warnings, numpy as np
warnings.filterwarnings('ignore')
from src.processor import VideoProcessor
from src.similarity import smooth_series, segment_reps, get_golden_template

processor = VideoProcessor()

videos = [
    ("TEMPLATE", "data/templates/push_up_template.mp4"),
    ("HV01_CUOI_DAU_THAP", "data/tests/hv01_cuoi_dau_thap.mp4"),
    ("HV01_TAP_DUNG", "data/tests/hv01_tap_dung.mp4"),
    ("HV02_TAP_DUNG", "data/tests/hv02_tap_dung.mp4"),
    ("HV01_MONG_CAO", "data/tests/hv01_mong_cao.mp4"),
    ("HV01_REP_SAI_REP_DUNG", "data/tests/hv01_rep_sai_rep_dung.mp4")
]

for name, path in videos:
    print(f"\n{'='*60}")
    print(f"VIDEO: {name} ({path})")
    print(f"{'='*60}")
    
    with open(path, 'rb') as f:
        data, temp_path = processor.process_video_lightweight(f)
    os.unlink(temp_path)
    
    if not data:
        continue
        
    depths = smooth_series([d["depth_sig"] * 100 for d in data])
    elbows = smooth_series([d["elbow_angle"] for d in data])
    reps = segment_reps(depths, elbows)
    
    print(f"  Số reps phát hiện: {len(reps)}")
    for i, (start, end) in enumerate(reps):
        rep_frames = data[start:end+1]
        hips = [f["hip_angle"] for f in rep_frames]
        heads = [f["head_angle"] for f in rep_frames]
        elbows_rep = [f["elbow_angle"] for f in rep_frames]
        head_drop = [-f.get("ear_shoulder_y_diff", 0) for f in rep_frames] # Positive = head lower than shoulder
        
        bottom_idx = np.argmin(elbows_rep)
        
        bodies = [f["body_line_angle"] for f in rep_frames]
        depth_sigs = [f["depth_sig"] for f in rep_frames]
        
        print(f"    Rep {i+1}:")
        print(f"      head_angle start: {heads[0]:.1f}, min: {min(heads):.1f}, diff: {heads[0] - min(heads):.1f}, std: {np.std(heads):.1f}")
        print(f"      head_drop at bottom: {head_drop[bottom_idx]:.3f} (max drop: {max(head_drop):.3f})")
        print(f"      hip_angle min: {min(hips):.1f}, mean: {np.mean(hips):.1f}")
        print(f"      body_line min: {min(bodies):.1f}, mean: {np.mean(bodies):.1f}")
        print(f"      depth_sig at bottom: {depth_sigs[bottom_idx]:.4f}, mean: {np.mean(depth_sigs):.4f}")

