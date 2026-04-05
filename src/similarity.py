import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks, savgol_filter


def smooth_series(data):
    if len(data) < 11:
        return data
    return savgol_filter(data, 11, 3).tolist()


def segment_reps(angles):
    """Tìm điểm bắt đầu và kết thúc của từng Rep (động)."""
    inv_angles = 180 - np.array(angles)
    
    # Tìm các đỉnh (khoảnh khắc chạm đáy)
    peaks, _ = find_peaks(inv_angles, height=40, distance=15)

    reps = []
    for i, p in enumerate(peaks):
        # Xác định khoảng an toàn giữa các đỉnh nhịp
        left_limit = peaks[i-1] if i > 0 else 0
        right_limit = peaks[i+1] if i < len(peaks)-1 else len(inv_angles)-1
        
        # Tìm giới hạn đáy (min điểm của inv_angles)
        left_bound = left_limit + np.argmin(inv_angles[left_limit:p+1]) if p > left_limit else 0
        right_bound = p + np.argmin(inv_angles[p:right_limit+1]) if right_limit > p else len(inv_angles)-1
        
        # Dóng từ đỉnh giũa ra hai biên, dừng lại nếu inv_angles nhỏ (tay duỗi ~ 160 độ)
        start = p
        while start > left_bound and inv_angles[start] > 20: 
            start -= 1
        
        end = p
        while end < right_bound and inv_angles[end] > 20: 
            end += 1
            
        start = max(left_bound, start - 2)
        end = min(right_bound, end + 2)
            
        if end - start >= 5:
            reps.append((start, end))
    return reps


def compute_pose_similarity(v1, v2):
    weights = np.ones(33)
    weights[11:17] = 4.0  # Tay
    weights[23:25] = 2.0  # Hông
    diff = (v1 - v2) * weights[:, np.newaxis]
    return float(max(0, 1 - (np.linalg.norm(diff) / 9.0)))


def align_and_score(st_rep_sigs, ex_template_sigs, st_rep_embeddings, ex_template_embeddings):
    """So sánh 1 rep học viên với 1 rep chuẩn của chuyên gia."""
    # 1. Khớp pha nội bộ trong 1 Rep
    _, path = fastdtw(st_rep_sigs, ex_template_sigs, dist=euclidean)

    # 2. Chấm điểm trên các cặp đã khớp
    scores = []
    for st_idx, ex_idx in path:
        sim = compute_pose_similarity(
            st_rep_embeddings[st_idx], ex_template_embeddings[ex_idx])
        scores.append(sim)
    return np.mean(scores), path
