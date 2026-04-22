# Hybrid AQA v1 - Technical Checklist

Muc tieu file nay: bien de xuat thanh task cu the theo tung file, co the code ngay.

## 0) Nguyen tac trien khai

- [x] Khong research lai noi dung Improve.md cu, chi chuyen thanh task ky thuat.
- [x] Giu luong app hien tai (Streamlit + MediaPipe + DTW), nang cap theo huong hybrid.
- [x] Uu tien code chay duoc truoc, sau do moi tune threshold.

## 1) Checklist theo file

### main.py

- [x] Them luong Hybrid AQA v1 trong UI thay cho chi-so global similarity.
- [x] Them quality gate truoc khi cham diem (expert + student).
- [x] Dung nhieu rep template cua chuyen gia (top template quality), khong chi rep dau.
- [x] Hien thi score component: kinematic, posture, tempo, stability.
- [x] Hien thi top fault uu tien va huong sua ngan gon.
- [x] Hien thi worst-frame student vs expert theo rep.
- [ ] Them bieu do time-series theo rep (elbow/body/hip) de debug nhanh.
- [ ] Them panel "why low score" theo rule threshold (numerical explainability).

### src/engine.py

- [x] Bo sung elbow_left_angle, elbow_right_angle, elbow_asymmetry.
- [x] Bo sung body_line_angle (shoulder-hip-ankle), neck_angle.
- [x] Bo sung hip_line_offset de bat hip sag / pike.
- [x] Bo sung elbow_flare_ratio.
- [x] Bo sung landmark confidence (mean/min) tren keypoints quan trong.
- [x] Van giu key cu de khong vo backward compatibility (`elbow_angle`, `depth_sig`, `pose_embedding`, `sig`).
- [ ] Tach logic camera-view suitability (side/diagonal) thanh score rieng.

### src/processor.py

- [x] Chuyen sampling tu skip-frame cung sang timestamp-based sampling.
- [x] Gan `timestamp` cho moi frame hop le.
- [x] Loc frame theo confidence gate (mean/min visibility).
- [x] Tra ve metadata session: fps, processed_frames, valid_ratio, mean_confidence, duration.
- [ ] Them blur/motion quality gate (Laplacian variance).
- [ ] Them optional auto-trim warmup truoc khi vao bai.

### src/similarity.py

- [x] Nang cap segmentation theo dynamic threshold + time-based constraints.
- [x] Them bo loc rep quality (duration, confidence, ROM).
- [x] Them session quality gate helper.
- [x] Them template builder cho nhieu rep expert.
- [x] Giu DTW alignment va kinematic similarity lam tang 1.
- [x] Them posture rule engine + fault taxonomy:
  - shallow_depth
  - no_lockout
  - hip_sag
  - pike_hip
  - head_drop
  - asymmetry
  - elbow_flare
- [x] Them tempo score va stability score.
- [x] Hop nhat diem theo cong thuc Hybrid AQA v1:

  S_rep = 0.35 * S_kinematic + 0.35 * S_posture + 0.15 * S_tempo + 0.15 * S_stability

- [x] Tra ve output co cau truc: score_total, score_components, faults, path, best_template, worst_pair.
- [ ] Tach threshold thanh config file de tune nhanh tren dataset.

## 2) Milestone trien khai

### Milestone A - Foundation (DONE)

- [x] Feature extraction cho posture + confidence
- [x] Session quality gate
- [x] Time-based segmentation
- [x] Hybrid scoring component
- [x] UI summary + rep-level feedback

### Milestone B - Calibration (NEXT)

- [ ] Tao file config threshold theo profile (beginner/intermediate)
- [ ] Tune threshold tren tap video gan nhan
- [ ] Add metric logging de benchmark qua cac version

### Milestone C - Productization

- [ ] Xuat JSON report sau moi session
- [ ] Them regression tests cho segmentation + fault engine
- [ ] Add CI smoke test cho pipeline upload -> score

## 3) Definition of Done cho Hybrid AQA v1

- [x] App van chay voi `streamlit run main.py`
- [x] Co quality gate truoc scoring
- [x] Co score component theo cong thuc hybrid
- [x] Co fault feedback theo rep
- [x] Co multi-template expert matching
- [ ] Co test tu dong cho logic scoring
- [ ] Co bo threshold duoc tune tren data thuc

## 4) Command de verify nhanh

- Chay app:
  - `streamlit run main.py`
- Kiem tra syntax:
  - `python -m compileall main.py src`
