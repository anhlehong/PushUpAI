import os
import datetime
import numpy as np
import warnings
warnings.filterwarnings('ignore') # Ignore tensorflow/mediapipe warnings for clean output
from src.processor import VideoProcessor
from src.evaluator import PushUpEvaluator

def run_evaluation(expert_path, student_path, log_file):
    processor = VideoProcessor()
    
    with open(expert_path, 'rb') as f_ex:
        ex_data, ex_temp_path = processor.process_video_lightweight(f_ex)
        
    with open(student_path, 'rb') as f_st:
        st_data, st_temp_path = processor.process_video_lightweight(f_st)
        
    if not ex_data or not st_data:
        log_file.write("ERROR: Could not process videos.\n")
        if ex_data: os.unlink(ex_temp_path)
        if st_data: os.unlink(st_temp_path)
        return

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
        log_file.write("No reps detected for student.\n")

    os.unlink(ex_temp_path)
    os.unlink(st_temp_path)

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/test_run_{timestamp}.log"
    
    expert_vid = "data/videos/push_up_template.mp4"
    test_cases = [
        ("Expert vs Expert", expert_vid),
        ("Expert vs Khong Gong Bung", "data/videos/khong gong bung.mp4"),
        ("Expert vs Incorrect Form", "data/videos/Push-Up incorrect form.mp4"),
        ("Expert vs Irrelevant (Vo Su)", "data/videos/video_vo_su.mp4")
    ]
    
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write(f"Test Run: {timestamp}\n")
        f.write("="*40 + "\n")
        
        for name, student_vid in test_cases:
            f.write(f"\nRunning test: {name}\n")
            f.write(f"Student Video: {student_vid}\n")
            f.write("-" * 20 + "\n")
            if os.path.exists(student_vid) and os.path.exists(expert_vid):
                print(f"Running {name}...")
                run_evaluation(expert_vid, student_vid, f)
            else:
                f.write(f"File not found for either {expert_vid} or {student_vid}\n")
            f.write("="*40 + "\n")
            
    print(f"Tests complete. Log saved to {log_path}")
