import numpy as np

def detect_phases(depth_series):
    phases = []
    max_idx = np.argmax(depth_series)
    for i in range(len(depth_series)):
        if i < max_idx - 2:
            phases.append("down")
        elif i <= max_idx + 2:
            phases.append("bottom")
        else:
            phases.append("up")
    return phases


class Rule:
    def __init__(self, name, severity):
        self.name = name
        self.severity = severity
        self.description = ""

    def check(self, frame, context=None):
        return None


# === FRAME-LEVEL RULES (kiểm tra từng frame) ===

class HipSagRule(Rule):
    """So sánh hip_angle với mentor. Chỉ bắt lỗi khi lệch > 15° dưới mức mentor."""
    def __init__(self):
        super().__init__("hip_sag", "high")
        self.description = "Lưng bị võng xuống"

    def check(self, frame, context=None):
        mentor_avg = context.get("mentor_hip_angle", 165.0) if context else 165.0
        # Widen from 15 to 22 degrees below mentor to prevent false positives for different camera angles
        if frame["hip_angle"] < mentor_avg - 22:
            return {
                "type": self.name, "message": self.description,
                "frame_idx": frame["frame_idx"], "severity": self.severity
            }
        return None


class PikeRule(Rule):
    """Phát hiện mông nhô cao (pike) bằng depth_sig.
    depth_sig = shoulder_y - hip_y. Dương = vai thấp hơn hông = mông cao.
    So sánh trung bình depth_sig của học viên với mentor."""
    def __init__(self):
        super().__init__("hip_pike", "high")
        self.description = "Nhô mông quá cao"

    def check(self, frame, context=None):
        # Không check per-frame nữa, dùng check_rep
        return None

    def check_rep(self, frames, context=None):
        mentor_depth_sig = context.get("mentor_depth_sig_mean", -0.015) if context else -0.015
        
        student_depth_sigs = [f["depth_sig"] for f in frames]
        student_mean = np.mean(student_depth_sigs)
        
        # Nếu depth_sig trung bình của học viên dương hơn mentor > 0.03 → mông cao
        # Template: ~-0.015, Mông cao: ~+0.04, Sai biệt: ~0.055
        if student_mean > mentor_depth_sig + 0.03:
            worst = max(frames, key=lambda f: f["depth_sig"])
            return {
                "type": self.name, "message": self.description,
                "severity": self.severity, "frames": [worst["frame_idx"]]
            }
        return None


class BodyAlignmentRule(Rule):
    """So sánh body_line_angle với mentor. Chỉ bắt khi body_line thấp hơn mentor
    VÀ depth_sig âm (loại bỏ trường hợp mông cao đã bắt bởi PikeRule)."""
    def __init__(self):
        super().__init__("body_not_straight", "high")
        self.description = "Cơ thể không giữ thẳng (Vai-Hông-Gót)"

    def check(self, frame, context=None):
        mentor_avg = context.get("mentor_body_line_angle", 172.0) if context else 172.0
        # Chỉ bắt khi body_line thấp hơn mentor > 20° VÀ depth_sig âm (không phải pike)
        if frame["body_line_angle"] < mentor_avg - 20 and frame["depth_sig"] < 0:
            return {
                "type": self.name, "message": self.description,
                "frame_idx": frame["frame_idx"], "severity": self.severity
            }
        return None


# === REP-LEVEL RULES (kiểm tra toàn bộ rep 1 lần) ===

class DepthRule(Rule):
    """So sánh MIN elbow angle trong toàn bộ rep với mentor's min.
    Không check per-frame nữa để tránh bắt frame đang xuống dở."""
    def __init__(self):
        super().__init__("not_deep_enough", "high")
        self.description = "Chưa hạ người đủ sâu"

    def check_rep(self, frames, context=None):
        mentor_min_elbow = context.get("mentor_min_elbow", 75.0) if context else 75.0
        student_min_elbow = min(f["elbow_angle"] for f in frames)
        # Cho phép lệch 45° so với mentor (tính đến ROM khác nhau và camera)
        if student_min_elbow > mentor_min_elbow + 45:
            worst = min(frames, key=lambda f: f["elbow_angle"])
            return {
                "type": self.name, "message": self.description,
                "severity": self.severity, "frames": [worst["frame_idx"]]
            }
        return None


class HeadMisalignedRule(Rule):
    """Phát hiện cúi đầu bằng cách kiểm tra khoảng cách rơi của đầu so với vai."""
    def __init__(self):
        super().__init__("head_misaligned", "medium")
        self.description = "Gập cổ hoặc cúi đầu quá mức"

    def check_rep(self, frames, context=None):
        # Mặc định max_drop của mentor khoảng 0.085, cộng thêm biên độ sai số (threshold)
        mentor_max_drop = context.get("mentor_max_head_drop", 0.085) if context else 0.085
        
        student_drops = [f.get("head_drop_norm", 0) for f in frames]
        max_drop = max(student_drops)

        # Nếu đầu gập/rơi xuống quá sâu so với biên độ của mentor (thêm 0.02 để tránh false positive)
        # Các rep tập đúng của học viên thường có max_drop < 0.05
        # Các rep cúi đầu sẽ có max_drop từ 0.08 đến 0.13
        if max_drop > mentor_max_drop + 0.02:
            worst = max(frames, key=lambda f: f.get("head_drop_norm", 0))
            return {
                "type": self.name, "message": self.description,
                "severity": self.severity, "frames": [worst["frame_idx"]]
            }
        return None


class PushUpRuleEngine:
    def __init__(self, mentor_context=None):
        self.mentor_context = mentor_context or {}
        self.frame_rules = [
            HipSagRule(),
            BodyAlignmentRule(),
        ]
        self.rep_rules = [
            DepthRule(),
            HeadMisalignedRule(),
            PikeRule(),
        ]

    def evaluate_rep(self, frames):
        depth_series = [f["depth_sig"] * 100 for f in frames]
        phases = detect_phases(depth_series)
        for i, f in enumerate(frames):
            f["phase"] = phases[i]

        # Frame-level rules → aggregate
        frame_errors = []
        for f in frames:
            for rule in self.frame_rules:
                res = rule.check(f, context=self.mentor_context)
                if res:
                    frame_errors.append(res)

        results = self.aggregate(frame_errors, total_frames=len(frames))

        # Rep-level rules → add directly (đã tự aggregate)
        for rule in self.rep_rules:
            res = rule.check_rep(frames, context=self.mentor_context)
            if res:
                results.append(res)

        return results

    def aggregate(self, errors, total_frames):
        grouped = {}
        for e in errors:
            if e["type"] not in grouped:
                grouped[e["type"]] = {
                    "message": e["message"], "severity": e["severity"], "frames": []
                }
            grouped[e["type"]]["frames"].append(e["frame_idx"])

        results = []
        for err_type, data in grouped.items():
            threshold_ratio = 0.2
            if len(data["frames"]) / total_frames > threshold_ratio:
                step = max(1, len(data["frames"]) // 3)
                sample_frames = data["frames"][::step][:3]
                results.append({
                    "type": err_type, "message": data["message"],
                    "severity": data["severity"], "frames": sample_frames
                })
        return results

    def calculate_score(self, aggregated_errors):
        score = 1.0
        for err in aggregated_errors:
            if err["severity"] == "high":
                score -= 0.20
            elif err["severity"] == "medium":
                score -= 0.10
            elif err["severity"] == "low":
                score -= 0.05
        return max(0.0, score)
