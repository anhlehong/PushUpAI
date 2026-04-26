import numpy as np
from src.similarity import segment_reps, smooth_series, get_golden_template, align_and_score, check_is_valid_exercise, compute_pose_similarity
from src.rules import PushUpRuleEngine

class PushUpEvaluator:
    def __init__(self):
        self.rule_engine = PushUpRuleEngine()

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

        # Pre-check
        if not check_is_valid_exercise(st_data, ex_template_embs):
            return {"error": "❌ Hệ thống nhận thấy tư thế trong video hoàn toàn KHÔNG PHẢI là động tác Hít đất. Vui lòng tải lên video đúng bài tập!"}

        # 2. Student setup
        st_depths = smooth_series([d["depth_sig"] * 100 for d in st_data])
        st_elbows = smooth_series([d["elbow_angle"] for d in st_data])
        st_reps = segment_reps(st_depths, st_elbows)

        rep_results = []
        for i, (s_start, s_end) in enumerate(st_reps):
            st_rep_sigs = [st_data[j]["sig"] for j in range(s_start, s_end)]
            st_rep_embs = [st_data[j]["pose_embedding"] for j in range(s_start, s_end)]

            dtw_score, path = align_and_score(
                st_rep_sigs, ex_template_sigs, st_rep_embs, ex_template_embs)
                
            w_score, w_pair = 1.0, (0, 0)
            for s_idx, e_idx in path:
                sim = compute_pose_similarity(
                    st_data[s_start + s_idx]["pose_embedding"], ex_template_embs[e_idx])
                if sim < w_score:
                    w_score, w_pair = sim, (s_start + s_idx, ex_start + e_idx)

            st_rep_frames = [dict(d) for d in st_data[s_start:s_end]]
            errors = self.rule_engine.evaluate_rep(st_rep_frames)
            rule_score = self.rule_engine.calculate_score(errors)
            
            final_score = 0.4 * rule_score + 0.6 * dtw_score
            
            rep_results.append({
                "rep_num": i+1, 
                "score": final_score, 
                "dtw_score": dtw_score,
                "rule_score": rule_score,
                "errors": errors,
                "w_pair": w_pair,
                "range": (s_start, s_end)
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
            "overall_score": overall_score
        }
