# Giai thich bao cao ket qua Hybrid AQA v1

Tai lieu nay giai thich dung theo code hien tai cua du an, bo qua `.venv/` va `__pycache__/`.

> Cap nhat: file nay mo ta phien ban Hybrid AQA v1 cu. Tu refactor ngay 2026-04-22, UI chinh da chuyen sang cham `form` la diem chinh va tach `tempo` ra thanh chi so tham khao rieng.

## 1. Cac file va ham tham gia vao bao cao

| Thanh phan | File / ham chinh | Vai tro |
|---|---|---|
| Xu ly video | `src/processor.py` - `VideoProcessor.process_video_lightweight()` | Doc video, lay frame theo moc thoi gian, goi pose engine, tinh metadata session |
| Trich xuat dac trung pose | `src/engine.py` - `PoseEngine.extract_kinematics()` | Tinh elbow angle, body line, neck, hip offset, confidence, pose embedding |
| Quality gate | `src/similarity.py` - `evaluate_session_quality()` | Kiem tra video co du dieu kien de cham hay khong |
| Tach rep | `src/similarity.py` - `smooth_series()`, `segment_reps()`, `filter_reps_by_quality()` | Lam muot tin hieu, tim rep, loc rep xau |
| Chon rep mau chuyen gia | `src/similarity.py` - `build_expert_templates()` | Xep hang rep chuyen gia va giu lai toi da 3 rep tot nhat |
| Cham diem tung rep | `src/similarity.py` - `analyze_rep_hybrid()` | Tinh 4 thanh phan: kinematic, posture, tempo, stability |
| Tong hop loi uu tien | `src/similarity.py` - `summarize_top_faults()` | Cong don severity cua loi de xep hang top loi |
| Hien thi UI | `main.py` | Lay cac gia tri tren va render thanh bao cao |

## 2. Quality Gate trong bao cao co nghia la gi

Phan UI hien thi o `main.py`:

- `render_quality_block()` tai `main.py:38-52`
- goi sau khi da xu ly xong video o `main.py:75-83`

No hien 3 chi so:

### 2.1 `Valid ratio`

Nguon tinh:

- `src/processor.py:66-76`

Cong thuc:

```text
valid_ratio = valid_pose_frames / processed_frames
```

Trong do:

- `processed_frames`: so frame da duoc dem la "mang di xu ly pose"
- `valid_pose_frames`: so frame ma `extract_kinematics()` tim thay pose va frame do qua duoc confidence gate

Confidence gate cho tung frame nam o `src/processor.py:48-58`:

- `conf_mean >= 0.45`
- `conf_min >= 0.25`

Hai gia tri `conf_mean` va `conf_min` duoc tinh trong `src/engine.py:76-106` tu visibility cua cac keypoint quan trong:

```python
critical_idx = [0, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 27, 28]
```

Y nghia:

- `Valid ratio` cao: nhieu frame co skeleton du ro de dung tiep
- `Valid ratio` thap: nhieu frame bi mat pose, khuat nguoi, dung qua xa, anh sang kem, hoac goc quay xau

Voi so lieu ban dua:

- Video chuyen gia: `35.6%`
- Video hoc vien: `35.3%`

Hai video van `passed` vi nguong quality gate chi yeu cau:

```text
valid_ratio >= 0.33
```

Nguong nay nam o `src/similarity.py:218-223`.

### 2.2 `Mean confidence`

Nguon tinh:

- `src/processor.py:50-58`
- `src/processor.py:72-73`
- `src/engine.py:76-106`

Cong thuc:

```text
mean_confidence = tong conf_mean cua cac valid frame / valid_pose_frames
```

Trong do `conf_mean` la trung binh visibility cua cac keypoint quan trong tren moi frame hop le.

Y nghia:

- Gia tri nay khong tinh tren toan bo `processed_frames`
- No chi tinh tren `valid_pose_frames`
- Vi vay no do "do chac cua frame hop le", khong phai do phu pose toan video

Voi so lieu ban dua:

- Chuyen gia: `93.6%`
- Hoc vien: `88.8%`

Hai video deu vuot xa nguong `0.5`.

### 2.3 `Processed frames`

Nguon tinh:

- `src/processor.py:31-47`
- `src/processor.py:66-76`

Cong thuc:

```text
processed_frames = so frame duoc lay mau va dem da dua vao buoc pose extraction
```

Luu y:

- Day khong phai tong so frame goc cua video
- He thong lay mau theo thoi gian voi `sample_interval_sec = 1/30` (`src/processor.py:14`)
- Neu video goc la 30 FPS, thuong se xu ly xap xi moi frame
- Neu video goc FPS cao hon, he thong van chi lay xap xi 30 frame/giay

### 2.4 `Quality gate passed`

Nguon tinh:

- `src/similarity.py:218-273`

`passed = True` khi khong vi pham bat ky dieu kien nao sau day:

```text
processed_frames >= 25
valid_ratio >= 0.33
mean_confidence >= 0.5
video_duration_sec >= 3.0
len(data) >= 10
pushup_context_ratio >= 0.35
```

`pushup_context_ratio` duoc tinh boi:

- `compute_pushup_context_metrics()` tai `src/similarity.py:77-97`

Cong thuc:

```text
torso_tilt_deg = atan2(abs(hip_y - shoulder_y), abs(hip_x - shoulder_x))
pushup_context_ratio = ty le frame co torso_tilt_deg <= 55 do
```

Y nghia:

- Video phai co du frame "dang nam cheo nghieng giong boi canh hit dat"
- Neu co nhieu frame dung, di bo, xoay nguoi thi quality gate co the fail du `valid_ratio` cao

### 2.5 Câu: `Use a diagonal side view (~45 deg), keep full body visible, distance about 1.8-2.0m.`

Day la loi khuyen co dinh, khong phai chi so duoc tinh dong.

Nguon:

- `src/similarity.py:272`

## 3. `Ket qua hiep tap: 2 Reps` duoc tinh the nao

Nguon UI:

- `main.py:130`

So rep hien thi la:

```text
len(st_reps)
```

Trong do `st_reps` duoc tao qua 3 buoc:

1. Lay chuoi `elbow_angle` tu du lieu pose
2. Lam muot bang `smooth_series()` - `src/similarity.py:110-119`
3. Tach rep bang `segment_reps()` - `src/similarity.py:122-171`
4. Loc rep bang `filter_reps_by_quality()` - `src/similarity.py:174-215`

Dieu kien de 1 rep duoc giu lai sau loc:

- do dai trong `[0.5s, 6.0s]`
- bien do ROM cua elbow `>= 12 deg`
- confidence trung binh cua rep `>= 0.45`
- `context_ratio` cua rep `>= 0.55`

Vi vay:

- `2 Reps` co nghia la video hoc vien sau khi tach va loc chat luong con lai 2 rep hop le de cham

## 4. `Dang dung 2 rep mau tot nhat cua chuyen gia...` duoc tinh the nao

Nguon UI:

- `main.py:123-133`

Danh sach template duoc tao boi:

- `build_expert_templates()` tai `src/similarity.py:396-432`

Ham nay:

- duyet tung rep hop le cua chuyen gia
- tinh `template_quality`
- sap xep giam dan theo `template_quality`
- giu lai toi da `max_templates=3`

Cong thuc xep hang rep chuyen gia:

```text
template_quality =
    0.6 * posture_score
  + 0.25 * confidence
  + 0.15 * clamp01(rom / 70.0)
```

Voi:

- `posture_score`: diem posture cua rep chuyen gia
- `confidence`: mean landmark confidence cua rep
- `rom`: bien do chuyen dong cua `elbow_angle`

Vay cau:

```text
Dang dung 2 rep mau tot nhat cua chuyen gia de cham Hybrid AQA v1.
```

co nghia la:

- sau khi tach rep va loc rep cho video chuyen gia
- ham `build_expert_templates()` da giu lai 2 template trong danh sach `expert_templates`

## 5. Cach tinh diem tong quan va 4 thanh phan

Nguon:

- `analyze_rep_hybrid()` tai `src/similarity.py:460-538`
- trong UI tong hop o `main.py:163-184`

### 5.1 Diem cho moi rep

Cong thuc:

```text
score_total =
    0.35 * kinematic
  + 0.35 * posture
  + 0.15 * tempo
  + 0.15 * stability
```

Trong do weight nam o:

- `src/similarity.py:7-12`

### 5.2 Diem tong quan toan bai tap

Nguon:

- `main.py:163-169`

Cong thuc:

```text
overall_score = mean(score_total cua tat ca rep)
```

Voi so lieu ban dua:

```text
overall_score = mean(42.9%, 44.5%) = 43.7%
```

Dung voi UI.

### 5.3 Diem thanh phan toan bai

Nguon:

- `main.py:171-184`

Cong thuc:

```text
avg_components[key] = mean(score_components[key] cua tat ca rep)
```

Ap vao bo so cua ban:

- `Posture = mean(60.0, 60.0) = 60.0%`
- `Tempo = mean(50.2, 59.5) = 54.85% ~= 54.9%`
- `Stability = mean(84.0, 87.1) = 85.55% ~= 85.5%`

`Kinematic = 4.7%` cung la trung binh tu gia tri goc cua 2 rep. UI chi khong hien `Kinematic` trong khung chi tiet tung rep, nhung van co trong `rep['score_components']['kinematic']`.

## 6. Y nghia va cach tinh tung thanh phan diem

### 6.1 `Kinematic`

Nguon:

- `align_and_score()` - `src/similarity.py:443-457`
- `compute_pose_similarity()` - `src/similarity.py:435-440`
- `PoseEngine.extract_kinematics()` - `src/engine.py:29-113`

Pipeline:

1. Moi frame duoc trich `sig`:

```text
sig = [
  elbow_angle,
  depth_sig * 100,
  body_line_angle,
  elbow_asymmetry
]
```

Nguon:

- `src/engine.py:107-112`

2. Moi frame cung co `pose_embedding` da duoc chuan hoa theo tam hong va do dai than:

```text
norm_lms = (landmarks - hip_center) / spine_dist
```

Nguon:

- `src/engine.py:83-86`

3. `align_and_score()` dung `fastdtw(..., dist=euclidean)` de can pha rep hoc vien voi tung template chuyen gia.

4. Sau khi co `path`, moi cap frame `(st_idx, ex_idx)` duoc tinh do giong pose bang:

```text
sim = max(0, 1 - norm(((v1 - v2) * weights)) / 9.5)
```

Trong do:

- tay (`11:17`) co weight `4.0`
- hong/chan (`23:29`) co weight `2.0`
- cac diem con lai co weight `1.0`

5. `Kinematic score` cua rep = trung binh `sim` tren toan bo duong DTW.

Y nghia:

- diem nay do muc do giong dong tac theo quy dao pose va pha chuyen dong so voi rep mau chuyen gia
- trong bo ket qua cua ban, diem `Kinematic` rat thap (`4.7%`), tuc la tuy posture/tempo/stability khong qua te, hinh dang chuyen dong va can pha van lech rat xa rep mau

### 6.2 `Posture`

Nguon:

- `score_posture()` - `src/similarity.py:294-357`

Ham nay khong so khop voi chuyen gia, ma cham truc tiep tu rep hoc vien bang bo rule.

No tinh cac metric noi bo:

- `elbow_min`
- `elbow_max`
- `asymmetry_p85`
- `hip_offset_p80`
- `hip_offset_p20`
- `neck_p10`
- `flare_p85`

Roi cong phat neu vi pham rule:

| Fault code | Dieu kien kich hoat | Penalty | Message trong UI |
|---|---|---:|---|
| `shallow_depth` | `elbow_min > 118` | `0.22` | `Depth is too shallow at the bottom phase.` |
| `no_lockout` | `elbow_max < 150` | `0.20` | `Top phase misses full lockout.` |
| `hip_sag` | `hip_offset_p80 > 0.11` | `0.18` | `Hip drops below body line.` |
| `pike_hip` | `hip_offset_p20 < -0.18` | `0.14` | `Hip is too high during the rep.` |
| `head_drop` | `neck_p10 < 118` | `0.12` | `Neck alignment collapses.` |
| `asymmetry` | `asymmetry_p85 > 42` | `0.12` | `Left/right elbow motion is imbalanced.` |
| `elbow_flare` | `flare_p85 > 0.62` | `0.08` | `Elbows flare out too much.` |

Cuoi cung:

```text
posture_score = clamp01(1.0 - tong_penalty)
```

Voi ket qua ban dua:

- moi rep deu bi 2 loi:
  - `shallow_depth`
  - `hip_sag`
- tong penalty = `0.22 + 0.18 = 0.40`
- nen `posture_score = 1 - 0.40 = 0.60 = 60.0%`

Vi vay ca Rep 1 va Rep 2 deu ra `Posture = 60.0%`.

### 6.3 `Tempo`

Nguon:

- `_rep_duration_sec()` - `src/similarity.py:276-281`
- `score_tempo()` - `src/similarity.py:360-365`

Ham nay so thoi luong rep hoc vien voi template chuyen gia duoc chon tot nhat cho rep do.

Cong thuc:

```text
ratio = rep_duration_sec / template_duration_sec
deviation = abs(log(ratio))
tempo_score = clamp01(1.0 - deviation / 0.7)
```

Y nghia:

- bang toc do thi diem cao
- nhanh hon hoac cham hon nhieu thi diem giam
- neu thieu duration hop le thi mac dinh tra `0.7`

### 6.4 `Stability`

Nguon:

- `score_stability()` - `src/similarity.py:368-393`

Ham nay do do "run" cua rep hoc vien.

No lay 3 thanh phan:

- `jerk`: dao ham bac 2 trung binh cua chuoi `elbow_angle` da lam muot
- `body_wobble`: do lech chuan cua `body_line_angle`
- `hip_wobble`: do lech chuan cua `hip_line_offset`

Cong thuc:

```text
instability =
    0.45 * min(1, jerk / 7.0)
  + 0.35 * min(1, body_wobble / 10.0)
  + 0.20 * min(1, hip_wobble / 0.05)

stability_score = clamp01(1.0 - instability)
```

Y nghia:

- rep cang it rung, it giat, it vo song than nguoi thi diem cang cao
- bo ket qua cua ban co `84.0%` va `87.1%`, nghia la rep kha on dinh, du form chua dung va chua giong rep mau

## 7. Vi sao top loi uu tien lai ra dung 2 loi nay

Nguon:

- `summarize_top_faults()` - `src/similarity.py:541-564`

Cong thuc:

```text
weighted_counts[fault_code] += fault.severity
```

Severity duoc gan trong `_build_fault()` tu `FAULT_LIBRARY`.

Trong code hien tai:

- `shallow_depth` co `severity = 0.85`
- `hip_sag` co `severity = 0.78`

Neu 2 loi nay cung xuat hien o ca 2 rep, diem cong don se la:

- `shallow_depth`: `0.85 + 0.85 = 1.70`
- `hip_sag`: `0.78 + 0.78 = 1.56`

Nen chung duoc dua len phan:

- `Top loi uu tien can sua`

## 8. Giai thich tung dong trong phan chi tiet Rep

### 8.1 `Rep 1 - Diem: 42.9%` va `Rep 2 - Diem: 44.5%`

Nguon:

- `main.py:135-161`
- `main.py:195-212`

Moi rep la 1 phan tu trong `rep_results`, trong do:

- `score` = `rep_eval["score_total"]`
- `score_components` = `rep_eval["score_components"]`
- `faults` = `rep_eval["faults"]`
- `template_idx` = rep chuyen gia duoc chon de so khop tot nhat
- `worst_pair` = cap frame hoc vien/chuyen gia lech nhat tren DTW path

### 8.2 Tai sao trong chi tiet rep khong thay `Kinematic`

Do UI hien tai chi render:

- `Total`
- `Posture`
- `Tempo`
- `Stability`

o `main.py:204-212`.

`Kinematic` van co trong data, nhung khong duoc render trong o chi tiet rep.

### 8.3 `Worst frame from student in this rep`

Nguon:

- `src/similarity.py:515-523`
- `main.py:144-159`
- `main.py:222-237`

Ham `analyze_rep_hybrid()` duyet toan bo cap frame tren `best_path`, tinh lai `compute_pose_similarity()` cho tung cap, roi lay cap co similarity thap nhat:

```text
worst_pair = cap frame co local similarity thap nhat
```

Sau do UI doi chi so local thanh chi so global cua ca video de lay dung frame goc.

### 8.4 `Expert reference frame (template #2)`

Nguon:

- `main.py:237`

Caption duoc render bang:

```text
template #{rep['template_idx'] + 1}
```

Can hieu dung:

- `template_idx` la chi so rep goc cua chuyen gia truoc khi doi sang hien thi 1-based
- no khong nhat thiet la "template xep hang thu 2"
- no la rep chuyen gia co `kinematic_score` tot nhat cho rep hoc vien dang xet

## 9. Dien giai ngan gon bo so ban da dua

Co the doc bo ket qua nay nhu sau:

- Hai video deu qua `Quality Gate`, nhung `Valid ratio` deu sat nguong 33%, tuc la chat luong pose chi vua du de cham.
- Video hoc vien tach duoc `2 rep` hop le.
- He thong giu `2 rep` chuyen gia lam template so sanh.
- Diem tong `43.7%` la trung binh cua 2 rep `42.9%` va `44.5%`.
- `Posture = 60.0%` vi ca 2 rep deu dinh 2 loi: xuong chua sau (`shallow_depth`) va vo duong thang hong (`hip_sag`).
- `Tempo = 54.9%` cho thay nhip rep khong khop tot voi template.
- `Stability = 85.5%` cho thay dong tac khong rung qua nhieu.
- `Kinematic = 4.7%` cho thay quy dao pose va pha chuyen dong tong the van rat khac rep mau chuyen gia.

## 10. Tom tat mapping nhanh: metric nao do ham nao

| Chi so tren bao cao | Ham tinh chinh | Cong thuc ngan |
|---|---|---|
| `Valid ratio` | `process_video_lightweight()` | `valid_pose_frames / processed_frames` |
| `Mean confidence` | `process_video_lightweight()` + `extract_kinematics()` | `avg(conf_mean tren valid frames)` |
| `Processed frames` | `process_video_lightweight()` | dem so frame da lay mau de xu ly |
| `Quality gate passed` | `evaluate_session_quality()` | pass neu vuot tat ca nguong session |
| `Kết quả hiệp tập: 2 Reps` | `segment_reps()` + `filter_reps_by_quality()` | `len(st_reps)` |
| `2 rep mẫu tốt nhất` | `build_expert_templates()` | top rep chuyen gia theo `template_quality` |
| `Điểm tổng quan` | `main.py` + `analyze_rep_hybrid()` | mean diem tung rep |
| `Kinematic` | `align_and_score()` + `compute_pose_similarity()` | DTW + mean pose similarity |
| `Posture` | `score_posture()` | `1 - tong_penalty` |
| `Tempo` | `score_tempo()` | `1 - abs(log(ratio))/0.7` |
| `Stability` | `score_stability()` | `1 - instability` |
| `Top lỗi ưu tiên` | `summarize_top_faults()` | cong don `severity` theo loi |
| `Worst frame` | `analyze_rep_hybrid()` | cap frame co similarity thap nhat tren `best_path` |

## 11. Cac line code de doi chieu nhanh

- `main.py:38-52` - render block Quality Gate
- `main.py:75-184` - pipeline chinh cua UI
- `main.py:195-237` - render chi tiet tung rep va worst frame
- `src/processor.py:11-77` - xu ly video + metadata
- `src/engine.py:29-113` - trich xuat dac trung pose
- `src/similarity.py:77-97` - push-up context
- `src/similarity.py:122-171` - tach rep
- `src/similarity.py:174-215` - loc rep
- `src/similarity.py:218-273` - quality gate
- `src/similarity.py:294-357` - posture score
- `src/similarity.py:360-365` - tempo score
- `src/similarity.py:368-393` - stability score
- `src/similarity.py:396-432` - build expert templates
- `src/similarity.py:443-457` - DTW alignment + kinematic score
- `src/similarity.py:460-538` - cham diem hybrid cho 1 rep
- `src/similarity.py:541-564` - top faults
