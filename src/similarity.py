import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks, savgol_filter


def smooth_series(data):
    if len(data) < 11:
        return data
    return savgol_filter(data, 11, 3).tolist()


def segment_reps(primary_signals, aux_signals=None):
    """Tìm điểm bắt đầu và kết thúc của từng Rep (động)."""
    # primary_signals là depth_sig (nhân 100), tăng khi đi xuống, đỉnh dương.
    peaks, _ = find_peaks(primary_signals, distance=15)

    reps = []
    for i, p in enumerate(peaks):
        # Xác định khoảng an toàn giữa các đỉnh nhịp
        left_limit = peaks[i-1] if i > 0 else 0
        right_limit = peaks[i+1] if i < len(peaks)-1 else len(primary_signals)-1
        
        # Tìm giới hạn đáy (min điểm của primary_signals)
        left_bound = left_limit + np.argmin(primary_signals[left_limit:p+1]) if p > left_limit else 0
        right_bound = p + np.argmin(primary_signals[p:right_limit+1]) if right_limit > p else len(primary_signals)-1
        
        # Lọc nhiễu: Biên độ aux_signal (elbow_angle) phải thay đổi ít nhất 15 độ
        if aux_signals is not None:
            max_aux = max(aux_signals[left_bound:right_bound+1])
            min_aux = min(aux_signals[left_bound:right_bound+1])
            if max_aux - min_aux < 15:
                continue
                
        # Dóng từ đỉnh giữa ra hai biên, dừng lại khi tín hiệu giảm gần đáy (plank)
        start = p
        thresh_left = primary_signals[left_bound] + 0.2 * (primary_signals[p] - primary_signals[left_bound])
        while start > left_bound and primary_signals[start] > thresh_left: 
            start -= 1
        
        end = p
        thresh_right = primary_signals[right_bound] + 0.2 * (primary_signals[p] - primary_signals[right_bound])
        while end < right_bound and primary_signals[end] > thresh_right: 
            end += 1
            
        start = max(left_bound, start - 2)
        end = min(right_bound, end + 2)
            
        if end - start >= 5:
            reps.append((start, end))
    return reps


def compute_pose_similarity(v1, v2):
    weights = np.ones(33)
    weights[11:13] = 4.0  # Vai
    weights[13:17] = 2.0  # Tay
    weights[23:25] = 4.0  # Hông
    weights[25:27] = 2.0  # Đầu gối
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


def check_is_valid_exercise(st_data, ex_template_embs):
    """
    Kiểm tra nhanh xem video học viên có chứa tư thế giống bài tập không
    bằng cách so sánh ngẫu nhiên các frame với tư thế chuẩn bị (frame đầu tiên của template).
    """
    if not st_data or not ex_template_embs:
        return False
        
    plank_pose = ex_template_embs[0]
    
    # Lấy mẫu 20 frame từ video học viên để quét nhanh
    step = max(1, len(st_data) // 20)
    sample_embs = [st_data[i]["pose_embedding"] for i in range(0, len(st_data), step)]
    
    max_sim = 0
    for emb in sample_embs:
        sim = compute_pose_similarity(emb, plank_pose)
        if sim > max_sim:
            max_sim = sim
            
    # Nếu frame giống nhất với plank mà vẫn < 5% thì chắc chắn không phải đang tập hít đất
    return max_sim > 0.05


def get_golden_template(ex_data, ex_reps):
    """
    Tìm Rep trung bình/ổn định nhất của chuyên gia bằng cách tính 
    tổng khoảng cách DTW từ mỗi Rep tới tất cả các Rep khác,
    và chọn Rep có độ lệch chuẩn/khoảng cách trung bình thấp nhất.
    """
    if not ex_reps:
        return None
    
    if len(ex_reps) == 1:
        return ex_reps[0]
        
    rep_sigs = []
    for start, end in ex_reps:
        rep_sigs.append([ex_data[i]["sig"] for i in range(start, end)])
        
    n_reps = len(ex_reps)
    dist_matrix = np.zeros((n_reps, n_reps))
    
    for i in range(n_reps):
        for j in range(i + 1, n_reps):
            dist, _ = fastdtw(rep_sigs[i], rep_sigs[j], dist=euclidean)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
            
    avg_dists = np.mean(dist_matrix, axis=1)
    best_rep_idx = np.argmin(avg_dists)
    
    return ex_reps[best_rep_idx]
