import numpy as np
from src.similarity import segment_reps, smooth_series, get_golden_template, align_and_score, check_is_valid_exercise, compute_pose_similarity
from src.rules import PushUpRuleEngine

def _normalize_sig(sigs):
    """
    Chuẩn hóa sig (elbow_angle, depth*100) về khoảng [0,1] theo min-max
    trong phạm vi 1 rep. Giúp DTW không bị ảnh hưởng bởi scale camera.
    """
    arr = np.array(sigs)
    mins = arr.min(axis=0)
    maxs = arr.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1  # tránh chia 0
    return ((arr - mins) / ranges).tolist()


class PushUpEvaluator:
    def __init__(self):
        pass

    def evaluate(self, ex_data, st_data):
        # 1. Expert setup
        ex_depths = smooth_series([d["depth_sig"] * 100 for d in ex_data])
        ex_elbows = smooth_series([d["elbow_angle"] for d in ex_data])
        ex_reps = segment_reps(ex_depths, ex_elbows)
        
        if not ex_reps:
            return {"error": "Chuyên gia chưa thực hiện động tác chuẩn."}

        golden_rep = get_golden_template(ex_data, ex_reps)
        ex_start, ex_end = golden_rep
        ex_template_sigs = [ex_data[i]["sig"] for i in range(ex_start, ex_end)]
        ex_template_embs = [ex_data[i]["pose_embedding"] for i in range(ex_start, ex_end)]

        # Tính baseline angles của mentor từ golden rep
        mentor_frames = [ex_data[i] for i in range(ex_start, ex_end)]
        mentor_min_hip = np.min([f["hip_angle"] for f in mentor_frames])
        mentor_max_head_drop = np.max([f.get("head_drop_norm", 0) for f in mentor_frames])
        mentor_context = {
            "mentor_head_angle": np.mean([f["head_angle"] for f in mentor_frames]),
            "mentor_head_std": np.std([f["head_angle"] for f in mentor_frames]),
            "mentor_min_head_angle": np.min([f["head_angle"] for f in mentor_frames]),
            "mentor_body_line_angle": np.mean([f["body_line_angle"] for f in mentor_frames]),
            "mentor_hip_angle": np.mean([f["hip_angle"] for f in mentor_frames]),
            "mentor_min_elbow": np.min([f["elbow_angle"] for f in mentor_frames]),
            "mentor_min_hip": mentor_min_hip,
            "mentor_max_head_drop": mentor_max_head_drop,
            "mentor_depth_sig_mean": np.mean([f["depth_sig"] for f in mentor_frames])
        }
        
        # Chuẩn hóa sig của template
        ex_template_sigs_norm = _normalize_sig(ex_template_sigs)

        # Pre-check
        if not check_is_valid_exercise(st_data, ex_template_embs):
            return {"error": "❌ Hệ thống nhận thấy tư thế trong video hoàn toàn KHÔNG PHẢI là động tác Hít đất. Vui lòng tải lên video đúng bài tập!"}

        # 2. Student setup
        st_depths = smooth_series([d["depth_sig"] * 100 for d in st_data])
        st_elbows = smooth_series([d["elbow_angle"] for d in st_data])
        st_reps = segment_reps(st_depths, st_elbows)

        # Khởi tạo Rule Engine VỚI mentor context
        rule_engine = PushUpRuleEngine(mentor_context=mentor_context)

        rep_results = []
        for i, (s_start, s_end) in enumerate(st_reps):
            st_rep_sigs = [st_data[j]["sig"] for j in range(s_start, s_end)]
            st_rep_embs = [st_data[j]["pose_embedding"] for j in range(s_start, s_end)]

            # Chuẩn hóa sig trước khi DTW để loại bỏ ảnh hưởng scale camera
            st_rep_sigs_norm = _normalize_sig(st_rep_sigs)
            
            dtw_score, path = align_and_score(
                st_rep_sigs_norm, ex_template_sigs_norm, st_rep_embs, ex_template_embs)
                
            w_score, w_pair = 1.0, (0, 0)
            for s_idx, e_idx in path:
                sim = compute_pose_similarity(
                    st_data[s_start + s_idx]["pose_embedding"], ex_template_embs[e_idx])
                if sim < w_score:
                    w_score, w_pair = sim, (s_start + s_idx, ex_start + e_idx)

            st_rep_frames = [dict(d) for d in st_data[s_start:s_end]]
            errors = rule_engine.evaluate_rep(st_rep_frames)
            rule_score = rule_engine.calculate_score(errors)
            
            # Rule Engine đã được calibrate với mentor context → tin cậy hơn DTW
            # DTW bị ảnh hưởng bởi tỷ lệ cơ thể/góc camera khác nhau
            final_score = 0.6 * rule_score + 0.4 * dtw_score
            
            rep_results.append({
                "rep_num": i+1, 
                "score": final_score, 
                "dtw_score": dtw_score,
                "rule_score": rule_score,
                "errors": errors,
                "w_pair": w_pair,
                "range": (s_start, s_end),
                "kinematics": {
                    "min_hip": min(f["hip_angle"] for f in st_rep_frames),
                    "mean_hip": np.mean([f["hip_angle"] for f in st_rep_frames]),
                    "min_elbow": min(f["elbow_angle"] for f in st_rep_frames),
                    "mean_body": np.mean([f["body_line_angle"] for f in st_rep_frames])
                }
            })

        overall_score = 0.0
        if rep_results:
            overall_score = np.mean([r["score"] for r in rep_results])
            if len(st_reps) < len(ex_reps):
                penalty = (len(ex_reps) - len(st_reps)) * 0.05
                overall_score = max(0.0, overall_score - penalty)

        return {
            "error": None,
            "st_reps_count": len(st_reps),
            "ex_reps_count": len(ex_reps),
            "rep_results": rep_results,
            "overall_score": overall_score,
            "mentor_context": mentor_context
        }
