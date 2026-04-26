import numpy as np

def detect_phases(depth_series):
    """
    Phân loại từng frame trong 1 rep vào 3 phase: down, bottom, up
    Dựa trên tín hiệu depth_sig (nhân 100).
    Khi depth tăng (vai thấp hơn) -> down. Đạt đỉnh -> bottom. Giảm -> up.
    """
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
        pass


class HipSagRule(Rule):
    def __init__(self):
        super().__init__("hip_sag", "high")
        self.description = "Lưng bị võng xuống"

    def check(self, frame, context=None):
        if frame["hip_angle"] < 155:  # Góc hông nhỏ hơn 155 độ (võng lưng rõ rệt)
            return {
                "type": self.name,
                "message": self.description,
                "frame_idx": frame["frame_idx"],
                "severity": self.severity
            }
        return None


class PikeRule(Rule):
    def __init__(self):
        super().__init__("hip_pike", "medium")
        self.description = "Nhô mông quá cao"

    def check(self, frame, context=None):
        if frame["hip_angle"] > 185:  # Góc hông lớn hơn 185 độ (nhô mông lên)
            return {
                "type": self.name,
                "message": self.description,
                "frame_idx": frame["frame_idx"],
                "severity": self.severity
            }
        return None


class DepthRule(Rule):
    def __init__(self):
        super().__init__("not_deep_enough", "high")
        self.description = "Chưa hạ người đủ sâu"

    def check(self, frame, context=None):
        if frame.get("phase") == "bottom":
            if frame["elbow_angle"] > 100:  # Không xuống tới mức 90 độ
                return {
                    "type": self.name,
                    "message": self.description,
                    "frame_idx": frame["frame_idx"],
                    "severity": self.severity
                }
        return None


class BodyAlignmentRule(Rule):
    def __init__(self):
        super().__init__("body_not_straight", "high")
        self.description = "Cơ thể không giữ thẳng (Vai-Hông-Gót)"

    def check(self, frame, context=None):
        # Trục cơ thể thẳng tắp thường quanh 180 độ. Nới lỏng ngưỡng lên 25 độ
        if abs(frame["body_line_angle"] - 180) > 25:
            return {
                "type": self.name,
                "message": self.description,
                "frame_idx": frame["frame_idx"],
                "severity": self.severity
            }
        return None


class HeadMisalignedRule(Rule):
    def __init__(self):
        super().__init__("head_misaligned", "low")
        self.description = "Gập cổ hoặc ngửa đầu quá mức"

    def check(self, frame, context=None):
        # Góc đầu - cổ so với thân. Nới lỏng ngưỡng lên 40 độ
        if abs(frame["head_angle"] - 180) > 40:
            return {
                "type": self.name,
                "message": self.description,
                "frame_idx": frame["frame_idx"],
                "severity": self.severity
            }
        return None


class AsymmetryRule(Rule):
    def __init__(self):
        super().__init__("asymmetry", "medium")
        self.description = "Hai tay dùng lực không đều (Lệch trái/phải)"

    def check(self, frame, context=None):
        if frame["left_right_symmetry"] > 25:  # Khuỷu tay 2 bên lệch nhau > 25 độ
            return {
                "type": self.name,
                "message": self.description,
                "frame_idx": frame["frame_idx"],
                "severity": self.severity
            }
        return None


class PushUpRuleEngine:
    def __init__(self):
        self.rules = [
            HipSagRule(),
            PikeRule(),
            DepthRule(),
            BodyAlignmentRule(),
            HeadMisalignedRule()
            # AsymmetryRule() - Đã vô hiệu hóa do Side View gây nhiễu
        ]

    def evaluate_rep(self, frames):
        errors = []
        depth_series = [f["depth_sig"] * 100 for f in frames]
        phases = detect_phases(depth_series)

        # Gán phase cho từng frame
        for i, f in enumerate(frames):
            f["phase"] = phases[i]

        for f in frames:
            for rule in self.rules:
                res = rule.check(f)
                if res:
                    errors.append(res)

        return self.aggregate(errors, total_frames=len(frames))

    def aggregate(self, errors, total_frames):
        """
        Gộp lỗi để tránh spam. Lỗi chỉ được công nhận nếu xảy ra trên một 
        tỷ lệ frame nhất định (trừ lỗi Depth chỉ check ở bottom).
        """
        grouped = {}
        for e in errors:
            if e["type"] not in grouped:
                grouped[e["type"]] = {
                    "message": e["message"],
                    "severity": e["severity"],
                    "frames": []
                }
            grouped[e["type"]]["frames"].append(e["frame_idx"])

        results = []
        for err_type, data in grouped.items():
            # DepthRule chỉ bị 1-2 frame ở bottom nên không cần rule tỷ lệ % cao
            threshold_ratio = 0.05 if err_type == "not_deep_enough" else 0.2
            
            if len(data["frames"]) / total_frames > threshold_ratio:
                # Chỉ lấy 3 frame đại diện
                step = max(1, len(data["frames"]) // 3)
                sample_frames = data["frames"][::step][:3]
                
                results.append({
                    "type": err_type,
                    "message": data["message"],
                    "severity": data["severity"],
                    "frames": sample_frames
                })

        return results
        
    def calculate_score(self, aggregated_errors):
        """
        Tính điểm 1 Rep dựa trên Weighted Penalty
        """
        score = 1.0
        for err in aggregated_errors:
            if err["severity"] == "high":
                score -= 0.20
            elif err["severity"] == "medium":
                score -= 0.10
            elif err["severity"] == "low":
                score -= 0.05
        return max(0.0, score)
