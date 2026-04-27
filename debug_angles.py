"""Debug script: Xuất dữ liệu góc thực tế của từng video để hiểu vấn đề."""
import os, warnings, numpy as np
warnings.filterwarnings('ignore')
from src.processor import VideoProcessor
from src.similarity import smooth_series, segment_reps, get_golden_template

processor = VideoProcessor()

videos = {
    "TEMPLATE": "data/templates/push_up_template.mp4",
    "HV01_TAP_DUNG": "data/tests/hv01_tap_dung.mp4",
    "HV01_CUOI_DAU": "data/tests/hv01_cuoi_dau_thap.mp4",
    "HV02_TAP_DUNG": "data/tests/hv02_tap_dung.mp4",
}

for name, path in videos.items():
    print(f"\n{'='*60}")
    print(f"VIDEO: {name} ({path})")
    print(f"{'='*60}")
    
    with open(path, 'rb') as f:
        data, temp_path = processor.process_video_lightweight(f)
    os.unlink(temp_path)
    
    if not data:
        print("  KHÔNG CÓ DỮ LIỆU!")
        continue
    
    flipped = data[0].get("flipped", False)
    print(f"  Đã flip: {flipped}")
    print(f"  Tổng frames hợp lệ: {len(data)}")
    
    # Thống kê góc
    angles = {
        "elbow_angle": [d["elbow_angle"] for d in data],
        "hip_angle": [d["hip_angle"] for d in data],
        "body_line_angle": [d["body_line_angle"] for d in data],
        "head_angle": [d["head_angle"] for d in data],
        "depth_sig": [d["depth_sig"] for d in data],
    }
    
    print(f"\n  --- Thống kê góc trung bình (mean ± std) ---")
    for k, v in angles.items():
        arr = np.array(v)
        print(f"    {k:20s}: {arr.mean():7.1f} ± {arr.std():5.1f}  (min={arr.min():.1f}, max={arr.max():.1f})")
    
    # Segmentation
    depths = smooth_series([d["depth_sig"] * 100 for d in data])
    elbows = smooth_series([d["elbow_angle"] for d in data])
    reps = segment_reps(depths, elbows)
    
    print(f"\n  --- Rep Segmentation ---")
    print(f"  Số reps phát hiện: {len(reps)}")
    
    for i, (start, end) in enumerate(reps):
        rep_depths = depths[start:end+1]
        rep_elbows = elbows[start:end+1]
        peak_idx = start + np.argmax(rep_depths)  # Đỉnh depth = điểm thấp nhất (bottom)
        
        # Kiểm tra pha: depth tăng trước peak (going down), giảm sau peak (going up)
        pre_peak = depths[start:peak_idx+1] if peak_idx > start else [0]
        post_peak = depths[peak_idx:end+1] if end > peak_idx else [0]
        
        print(f"\n    Rep {i+1}: frames [{start}-{end}] ({end-start} frames)")
        print(f"      depth range: {min(rep_depths):.1f} -> peak {max(rep_depths):.1f} -> {rep_depths[-1]:.1f}")
        print(f"      elbow range: {max(rep_elbows):.1f} (plank) -> {min(rep_elbows):.1f} (bottom) -> {rep_elbows[-1]:.1f}")
        print(f"      Peak (bottom) at relative frame: {peak_idx - start}/{end - start}")
    
    # So sánh sig pattern nếu có golden template
    if len(reps) >= 1:
        golden = get_golden_template(data, reps)
        g_start, g_end = golden
        golden_sigs = [data[i]["sig"] for i in range(g_start, g_end)]
        print(f"\n  --- Golden Template ---")
        print(f"  Rep range: [{g_start}-{g_end}]")
        print(f"  Sig[0] (start): elbow={golden_sigs[0][0]:.1f}, depth={golden_sigs[0][1]:.1f}")
        mid = len(golden_sigs) // 2
        print(f"  Sig[mid] (bottom): elbow={golden_sigs[mid][0]:.1f}, depth={golden_sigs[mid][1]:.1f}")
        print(f"  Sig[-1] (end): elbow={golden_sigs[-1][0]:.1f}, depth={golden_sigs[-1][1]:.1f}")

print("\n\nDone!")
