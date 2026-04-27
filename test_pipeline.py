import os
import datetime
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from src.processor import VideoProcessor
from src.evaluator import PushUpEvaluator

def run_evaluation(expert_path, student_path, log_file, details_dir, run_timestamp, test_name):
    processor = VideoProcessor()
    
    with open(expert_path, 'rb') as f_ex:
        ex_data, ex_temp_path = processor.process_video_lightweight(f_ex)
        
    with open(student_path, 'rb') as f_st:
        st_data, st_temp_path = processor.process_video_lightweight(f_st)
        
    if not ex_data or not st_data:
        log_file.write("ERROR: Không thể xử lý video.\n")
        if ex_data: os.unlink(ex_temp_path)
        if st_data: os.unlink(st_temp_path)
        return

    # Ghi nhận thông tin flip
    st_flipped = st_data[0].get("flipped", False) if st_data else False
    ex_flipped = ex_data[0].get("flipped", False) if ex_data else False
    if st_flipped:
        log_file.write("  [INFO] Video học viên đã được tự động lật ngang (phát hiện đầu bên trái).\n")
    if ex_flipped:
        log_file.write("  [INFO] Video chuyên gia đã được tự động lật ngang (phát hiện đầu bên trái).\n")

    evaluator = PushUpEvaluator()
    result = evaluator.evaluate(ex_data, st_data)
    
    if result.get("error"):
        log_file.write(f"ERROR: {result['error']}\n")
        os.unlink(ex_temp_path)
        os.unlink(st_temp_path)
        return
        
    log_file.write(f"Expert Reps: {result['ex_reps_count']}\n")
    log_file.write(f"Student Reps: {result['st_reps_count']}\n")
    
    for rep in result['rep_results']:
        score = rep['score']
        rule_score = rep['rule_score']
        dtw_score = rep['dtw_score']
        errors = rep['errors']
        
        log_file.write(f"  Rep {rep['rep_num']}: {score*100:.1f}% (Rule: {rule_score*100:.0f}%, DTW: {dtw_score*100:.0f}%)\n")
        
        # In thêm thông số kinematics cho dễ debug
        k = rep.get('kinematics', {})
        if k:
            log_file.write(f"    [Chi tiết] Hip min: {k.get('min_hip',0):.1f}°, mean: {k.get('mean_hip',0):.1f}° | ")
            log_file.write(f"Elbow min: {k.get('min_elbow',0):.1f}° | Body mean: {k.get('mean_body',0):.1f}°\n")
            
        if errors:
            for e in errors:
                log_file.write(f"    - Lỗi: {e['message']} (Nghiêm trọng: {e['severity']})\n")
        else:
            log_file.write(f"    - Hoàn hảo!\n")
            
    if result['rep_results']:
        overall_score = result['overall_score']
        if result['st_reps_count'] < result['ex_reps_count']:
            penalty = (result['ex_reps_count'] - result['st_reps_count']) * 0.05
            log_file.write(f"  Penalty applied for missing reps: -{penalty*100:.1f}%\n")
            
        log_file.write(f"OVERALL SCORE: {overall_score*100:.1f}%\n")
    else:
        log_file.write("Không phát hiện Rep nào từ video học viên.\n")

    # Lưu detail parameters ra JSON
    detail_path = os.path.join(details_dir, f"{run_timestamp}_{test_name.replace(' ', '_')}.json")
    
    # Loại bỏ object w_pair không serializable được để dump json
    dump_data = result.copy()
    for r in dump_data.get('rep_results', []):
        if 'w_pair' in r: del r['w_pair']
        if 'range' in r: del r['range']
        
    with open(detail_path, 'w', encoding='utf-8') as df:
        json.dump(dump_data, df, ensure_ascii=False, indent=2)

    os.unlink(ex_temp_path)
    os.unlink(st_temp_path)

if __name__ == "__main__":
    os.makedirs("logs/results", exist_ok=True)
    os.makedirs("logs/details", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/results/test_run_{timestamp}.log"
    
    expert_vid = "data/templates/push_up_template.mp4"
    
    test_cases = [
        # 1. Template vs chính nó
        ("Template vs Template", expert_vid),
        # 2. Các video test của học viên
        ("Template vs HV01 Tập Đúng", "data/tests/hv01_tap_dung.mp4"),
        ("Template vs HV02 Tập Đúng (Đầu bên trái)", "data/tests/hv02_tap_dung.mp4"),
        ("Template vs HV01 Cúi Đầu Thấp", "data/tests/hv01_cuoi_dau_thap.mp4"),
        ("Template vs HV01 Hít Nửa Rep", "data/tests/hv01_hit_nua_rep.mp4"),
        ("Template vs HV01 Không Gồng Bụng", "data/tests/hv01_khong_gong_bung.mp4"),
        ("Template vs HV01 Mông Cao", "data/tests/hv01_mong_cao.mp4"),
        ("Template vs HV01 Rep Sai Rep Đúng", "data/tests/hv01_rep_sai_rep_dung.mp4"),
        ("Template vs Võ Teakwondo (Không phải hít đất)", "data/tests/vo_teakwondo.mp4"),
    ]
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Test Run: {timestamp}\n")
        f.write(f"Expert Video: {expert_vid}\n")
        f.write("="*60 + "\n")
        
        for name, student_vid in test_cases:
            f.write(f"\n{'='*60}\n")
            f.write(f"TEST: {name}\n")
            f.write(f"Expert: {expert_vid}\n")
            f.write(f"Student: {student_vid}\n")
            f.write("-" * 40 + "\n")
            if os.path.exists(student_vid) and os.path.exists(expert_vid):
                print(f"Running {name}...")
                run_evaluation(expert_vid, student_vid, f, "logs/details", timestamp, name)
            else:
                f.write(f"FILE NOT FOUND: {student_vid}\n")
            f.write("="*60 + "\n")
            
    print(f"\n✅ Tests complete. Log saved to {log_path}")
