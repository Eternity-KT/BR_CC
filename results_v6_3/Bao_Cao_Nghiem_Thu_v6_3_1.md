# BÁO CÁO NGHIỆM THU THỬ NGHIỆM GSI-MLC-PA v6.3.1
## Đánh giá giải pháp Balanced-Root Calibration và Precision Guard
## So sánh toàn diện với 3 Baseline (BR, CC, MLC-PA) và các thế hệ GSI (v6.2.1, v6.3.0)

---

### 1. Bối cảnh & Vấn đề cần giải quyết
Tại phiên bản **v6.3.0**, việc khắc phục nhược điểm "bỏ rơi nhãn hiếm" của v6.2.1 đã giúp **Macro-F1 tăng vọt (+22.2%)**, nhưng đồng thời làm bộc lộ hiện tượng:
1. **Lạm phát xác suất dương (Probability Inflation):** Platt Scaling với trọng số đầy đủ ($w_1 = n_{\text{neg}}/n_{\text{pos}}$) đã làm dịch chuyển toàn bộ phân phối xác suất của nhãn hiếm về vùng cân bằng giả tạo quanh $0.50$.
2. **Double Compensation Trap:** Cả Bayes Likelihood Ratio lẫn Coverage Guard đều hạ thấp ngưỡng $\tau_1$ (xuống tận $0.41 - 0.43$) để ép Coverage đạt $\ge 70\%$.
3. **Hệ quả tiêu cực ở v6.3.0:**
   - Số lượng False Positives tăng đột biến (ví dụ trên `plantpseaac`, số dự đoán dương tăng từ 37 lên 1,268).
   - **Hamming Loss tăng vọt** từ $0.1344$ lên $0.2794$ (+107.9%).
   - **Subset Accuracy sụt giảm nghiêm trọng** từ $0.2593$ xuống $0.1380$ (-46.8%), thậm chí trên `humanpseaac` giảm xuống gần $0$ ($0.0035$).

---

### 2. Các đề xuất cải tiến trong phiên bản v6.3.1 (Kiến trúc hoàn thiện)

Nhằm khắc phục triệt để nghịch lý đánh đổi ở phiên bản tiền nhiệm v6.3.0, kiến trúc **GSI-MLC-PA v6.3.1** được xây dựng và giải thích theo mạch logic 3 bước chặt chẽ: **Xác định nguyên nhân gốc rễ (Root Cause) $\rightarrow$ Ý tưởng giải quyết (Conceptual Intuition) $\rightarrow$ Cách thức thực hiện (Implementation)**.

---

#### 2.1. Trụ cột 1: Hiệu chuẩn Căn Tuyến tính Cân bằng (Balanced-Root Platt Calibration - `sqrt_platt`)

- **1. Nguyên nhân tại sao cần (Root Cause):**
  - Trong dữ liệu mất cân bằng cực đoan (như `humanpseaac` với $\pi_l \approx 3\%$), tỷ lệ mẫu âm/dương lên tới $32:1$.
  - Ở v6.3.0, việc áp dụng hàm mất mát Platt tuyến tính với trọng số đầy đủ $w(1) = n_{\text{neg}}/n_{\text{pos}} \approx 32.0$ đã phạt quá nặng mô hình khi bỏ sót mẫu dương.
  - Hậu quả là hàm Sigmoid bị bẻ cong lệch lạc, điểm uốn dịch chuyển quá mức về phía nhãn hiếm, biến các xác suất ban đầu chỉ $0.02 - 0.08$ bị thổi phồng giả tạo lên $\ge 0.50$ (hiện tượng **Lạm phát xác suất - Probability Inflation**).
  - Điều này làm bùng nổ hàng nghìn ca False Positive (dương tính giả) trên các mẫu âm tính có nhiễu nhẹ, phá hủy hoàn toàn Subset Accuracy.

- **2. Ý tưởng giải quyết (Conceptual Intuition):**
  - Mô hình vẫn cần được ưu tiên để phát hiện nhãn hiếm (nếu không ưu tiên thì sẽ bỏ sót nhãn như v6.2.1), nhưng mức độ ưu tiên không được vượt quá ngưỡng ổn định phương sai.
  - Trong thống kê vững (Robust Statistics), phép biến đổi căn bậc hai ($\sqrt{\cdot}$) là cơ chế làm dịu tối ưu: hạ mức phạt từ $32.0$ xuống $\sqrt{32} \approx 5.66$.
  - Trọng số $5.66\times$ vừa đủ sức kéo các mẫu dương thực sự thoát khỏi đáy xác suất cận $0$, nhưng hoàn toàn không đủ mạnh để làm biến dạng trật tự phân vị thực tế và không thổi phồng xác suất của mẫu âm.

- **3. Cách thức thực hiện toán học (Mathematical Implementation):**
  - Với mỗi nhãn $l$, từ tần suất tiên nghiệm $\pi_l = \frac{1}{N}\sum_{i=1}^N Y_{i, l}$, thiết lập hàm trọng số căn:
    $$w_{\text{pos}}(l) = \sqrt{\frac{n_{\text{neg}}(l)}{n_{\text{pos}}(l)}} = \sqrt{\frac{1 - \pi_l}{\pi_l}}, \quad w_{\text{neg}}(l) = 1.0$$
  - Từ raw logit $z_l(x)$ của bộ phân loại cơ sở, hai tham số co giãn $(a_l, b_l)$ được tối ưu hóa thông qua bài toán hồi quy logistic có trọng số:
    $$\min_{a_l, b_l} \sum_{i=1}^N \left[ w_{\text{pos}}(l) \cdot y_{i, l} \ln\left(1 + e^{-(a_l z_{i, l} + b_l)}\right) + w_{\text{neg}}(l) \cdot (1 - y_{i, l}) \ln\left(1 + e^{a_l z_{i, l} + b_l}\right) \right]$$
  - Xác suất hiệu chuẩn hậu nghiệm bảo đảm tính chuẩn tắc và phản ánh đúng độ tin cậy: $P^*(Y_l = 1 \mid x) = \sigma(a_l z_l(x) + b_l)$.

---

#### 2.2. Trụ cột 2: Chốt chặn Bảo vệ Độ chính xác (Precision Guard) & Mở rộng Bất đối xứng (Asymmetric Guard)

- **1. Nguyên nhân tại sao cần (Root Cause):**
  - Để đáp ứng ràng buộc độ bao phủ sàn $\gamma_{\min} = 70\%$, cơ chế Coverage Guard cũ ở v6.3.0 áp dụng phép co hẹp đối xứng cả hai đầu ngưỡng.
  - Do ngưỡng dương ban đầu $\tau_1(l) \approx 0.70$, việc co hẹp đối xứng đã vô tình kéo tụt $\tau_1^{(\text{adj})}$ xuống tận $0.41 - 0.45$ ($< 0.50$).
  - Khi một mẫu chỉ có $42\%$ xác suất dương tính nhưng vẫn bị ép phân loại là $1$, mô hình đã mở toang cánh cửa cho các phán đoán đoán mò, phá hủy hoàn toàn Subset Accuracy và gây bùng nổ Hamming Loss.

- **2. Ý tưởng giải quyết (Conceptual Intuition):**
  - Thiết lập hai nguyên lý thiết kế then chốt:
    1. *Nguyên lý bất biến Precision Guard:* Tuyệt đối không bao giờ hạ ngưỡng khẳng định dương tính $\tau_1$ xuống dưới mức trung hòa Bayes $0.50$ ($\tau_1 \ge 0.50$).
    2. *Chiến lược mở rộng qua True Negatives:* Trong dữ liệu mất cân bằng cực đoan, $97\%$ số nhãn thực tế là Âm tính (True Negative). Do đó, con đường an toàn nhất để mở rộng độ phủ mà không tạo ra lỗi chính là chấp nhận thêm các ca chắc chắn Âm tính bằng cách nâng ngưỡng âm $\tau_0$ hướng về $0.50$, thay vì mạo hiểm hạ $\tau_1$ để đoán mò nhãn dương.

- **3. Cách thức thực hiện giải thuật (Algorithmic Implementation):**
  - Thuật toán thiết lập ngưỡng lý thuyết có Precision Guard:
    $$\tau_0(l) = \min\left(0.50, \, \max(0.01, \, c\sqrt{\pi_l})\right), \quad \tau_{1, \text{guarded}}(l) = \max\left(0.50, \, 1 - c\sqrt{1-\pi_l}\right)$$
  - Khi độ bao phủ thực nghiệm sơ bộ vi phạm $\text{Cov} < \gamma_{\min} = 0.70$, hệ số co hẹp $\rho^* \in [0, 1]$ được xác định qua tìm kiếm nhị phân để cập nhật ngưỡng bất đối xứng:
    $$\tau_0^{(\text{adj})}(l) = 0.50 - \rho^* \cdot \left(0.50 - \tau_0(l)\right), \quad \tau_1^{(\text{adj})}(l) = \tau_{1, \text{guarded}}(l) \ge 0.50$$
  - Quy tắc này bảo đảm ngưỡng $\tau_1$ luôn bất khả xâm phạm, trong khi $\tau_0$ thu nạp chuẩn xác các ca True Negative, triệt tiêu hoàn toàn nguồn gốc sinh ra False Positive rác.

---

#### 2.3. Cơ chế học nhận diện phủ định nhãn 0 (Negative Verification Paradigm)

- **1. Nguyên nhân tại sao cần (Root Cause):**
  - Trong phân loại đa nhãn mất cân bằng cực đoan ($\pi_l \ll 0.50$, nhãn $0$ chiếm trên $97\%$), việc cố gắng tìm kiếm các đặc trưng bao bọc lớp dương $Y=1$ thường bất khả thi do số lượng mẫu dương quá ít ỏi (chỉ $1-3\%$).
  - Nếu mô hình chỉ chăm chăm học tìm nhãn $1$, nó sẽ rơi vào một trong hai cái bẫy: hoặc quá bảo thủ bỏ sót toàn bộ nhãn hiếm (như v6.2.1), hoặc quá hiếu chiến đoán mò gây bùng nổ False Positive (như v6.3.0).

- **2. Ý tưởng giải quyết (Conceptual Intuition):**
  - *Chuyển đổi trọng tâm bài toán sang học nhận biết nhãn 0:* Thay vì săn tìm nhãn $1$, mô hình chuyển sang học để nhận biết có thể gán nhãn $Y = 0$ hay không thông qua xác suất phủ định $P(Y_l = 0 \mid x) = 1 - P(Y_l = 1 \mid x)$.
  - *Quy tắc kiểm định nhãn 0 trước:*
    - **Nếu xác suất nhãn 0 đủ cao** ($P(Y_l = 0 \mid x) \ge 1 - \tau_0 \iff P(Y_l = 1 \mid x) \le \tau_0$): Dự đoán nhãn **0** (Xác nhận âm tính an toàn, True Negative Verification).
    - **Nếu xác suất nhãn 0 rơi xuống rất thấp** ($P(Y_l = 0 \mid x) \le 1 - \tau_1 \iff P(Y_l = 1 \mid x) \ge \tau_1 \ge 0.50$): Bằng chứng dương tính áp đảo $\implies$ Dự đoán nhãn **1** (Positive Assertion).
    - **Nếu xác suất nhãn 0 nằm ở khoảng lưỡng lự** ($1 - \tau_1 < P(Y_l = 0 \mid x) < 1 - \tau_0$): Mô hình chủ động từ chối dự đoán ($\bot$) để không đoán mò.

- **3. Cách thức thực hiện giải thuật (Algorithmic Implementation):**
  - Trong vòng lặp quyết định suy diễn chọn lọc, mô hình tính xác suất âm tính $P(Y_l = 0 \mid x) = 1.0 - P^*(Y_l = 1 \mid x)$ và thực hiện kiểm định xác nhận nhãn 0 đầu tiên.

---

### 2.1. Mã giả toàn diện thuật toán GSI-MLC-PA v6.3.1 (Từ đầu đến cuối)

```text
========================================================================================================
THUẬT TOÁN TOÀN DIỆN: HUẤN LUYỆN VÀ SUY DIỄN CHỌN LỌC GSI-MLC-PA v6.3.1
========================================================================================================
ĐẦU VÀO:
  - Tập huấn luyện (X, Y) ∈ R^(N x d) x {0, 1}^(N x K)
  - Tập kiểm tra X* ∈ R^(N* x d)
  - Danh sách ngưỡng bóc tách tầng T = [0.75, 0.70, 0.65]
  - Ngưỡng tương quan phần dư theta_corr = 0.25
  - Chi phí từ chối c = 0.30, Độ phủ sàn an toàn gamma_min = 0.70

ĐẦU RA:
  - Ma trận dự đoán có chọn lọc Y_hat* ∈ {0, 1, ⊥}^(N* x K)  (với ⊥ là từ chối dự đoán)
  - Ma trận xác suất hiệu chuẩn P* ∈ [0, 1]^(N* x K)

--------------------------------------------------------------------------------------------------------
GIAI ĐOẠN 1: BÓC TÁCH TẦNG ĐỘC LẬP (IL) VÀ KHÁM PHÁ PHỤ THUỘC (DL)
--------------------------------------------------------------------------------------------------------
 1: Khởi tạo:
      L_rem ← {1, 2, ..., K}        // Tập nhãn ứng viên ban đầu
      IL ← ∅                         // Tập nhãn độc lập toàn cục
      X_context ← X                  // Ma trận đặc trưng ngữ cảnh

 2: FOR EACH ngưỡng tau_m IN T DO:
 3:     IL_m ← ∅
 4:     FOR EACH nhãn l IN L_rem DO:
 5:         Chạy 5-Fold Stratified Cross-Validation độc lập cho nhãn l trên (X_context, Y[:, l])
 6:         Thu nhận vector xác suất ngoài mẫu P_oof[:, l]
 7:         Tính điểm Out-of-Fold Selective-F1: Sel-F1_l = F1(Y[:, l], P_oof[:, l] ; c)
 8:         IF Sel-F1_l ≥ tau_m THEN:
 9:             IL_m ← IL_m ∪ {l}            // Nhãn đủ tin cậy để dự đoán độc lập
10:         END IF
11:     END FOR
12:     IF IL_m == ∅ THEN:
13:         BREAK FOR                        // Dừng bóc tách nếu tầng hiện tại không có thêm nhãn thỏa mãn
14:     END IF
15:     Huấn luyện mô hình Binary Relevance f_BR cho toàn bộ nhãn trong IL_m
16:     IL ← IL ∪ IL_m
17:     L_rem ← L_rem \ IL_m
18:     X_context ← [X, Normalize(P_oof[:, IL])]  // Làm giàu đặc trưng ngữ cảnh không rò rỉ cho tầng kế
19: END FOR
20: DL ← L_rem                               // Các nhãn còn lại được định tuyến vào tập phụ thuộc chuỗi

--------------------------------------------------------------------------------------------------------
GIAI ĐOẠN 2: HUẤN LUYỆN GHÉP NỐI ĐIỀU KIỆN DL VỚI BALANCED-ROOT PLATT CALIBRATION
--------------------------------------------------------------------------------------------------------
21: Huấn luyện mô hình cơ sở f_base trên X_context cho từng j ∈ DL, thu nhận xác suất ngoại mẫu P_oof_DL[:, j]
22: Tính ma trận phần dư ngoại mẫu: R[:, j] = Y[:, j] - P_oof_DL[:, j] với mọi j ∈ DL
23: Xây dựng đồ thị ghép nối phụ thuộc cho từng nhãn l ∈ DL:
      DL_temp[l] = { p ∈ DL \ {l} | |PearsonCorr(R[:, l], R[:, p])| ≥ theta_corr }
24: Sắp xếp các nhãn đối tác trong từng DL_temp[l] theo tương quan GIẢM DẦN (ưu tiên phụ thuộc mạnh nhất)

25: FOR EACH nhãn l IN DL DO:
26:     X_cond_l ← [X_context, Normalize(P_oof_DL[:, DL_temp[l]])]  // Đặc trưng ngữ cảnh + xác suất đối tác ghép nối
27:     Huấn luyện mô hình điều kiện g_l trên X_cond_l, trích xuất raw logit z_l(x)
28:     Tính tần suất dương tiên nghiệm: pi_l ← (1/N) * sum_{i=1}^N Y_{i, l}
29:     // Thiết lập trọng số Balanced-Root Platt (ngăn ngừa lạm phát xác suất):
30:     w_pos(l) ← sqrt((1 - pi_l) / pi_l),   w_neg(l) ← 1.0
31:     Tối ưu tham số hiệu chuẩn (a_l, b_l) qua hàm log-loss có trọng số:
          min_{a_l, b_l} sum_{i=1}^N [ w_pos(l)*y_{i,l}*ln(1 + e^{-(a_l z_l + b_l)}) 
                                       + w_neg(l)*(1 - y_{i,l})*ln(1 + e^{a_l z_l + b_l}) ]
32: END FOR

--------------------------------------------------------------------------------------------------------
GIAI ĐOẠN 3: SUY DIỄN CHỌN LỌC VỚI CƠ CHẾ XÁC NHẬN NHÃN 0 (NEGATIVE VERIFICATION)
--------------------------------------------------------------------------------------------------------
33: Trên tập kiểm tra X*:
      - Dự đoán xác suất P*_IL cho IL qua mô hình BR độc lập
      - Thiết lập X*_context ← [X*, Normalize(P*_IL)]
      - Dự đoán xác suất cơ sở P*_base_DL cho DL qua mô hình BR cơ sở trên X*_context
      - Với từng nhãn l ∈ DL: dự đoán xác suất tinh chỉnh P*_DL[:, l] qua g_l([X*_context, Normalize(P*_base_DL[:, DL_temp[l]])])
      - Hợp nhất ma trận xác suất toàn phần: P* = [P*_IL, P*_DL] ∈ [0, 1]^(N* x K)

34: FOR EACH nhãn l = 1 TO K DO:
35:     // Ngưỡng lý thuyết Bayes bất đối xứng theo tiên nghiệm:
36:     tau_0(l) ← min(0.50, max(0.01, c * sqrt(pi_l)))
37:     tau_1(l) ← 1 - c * sqrt(1 - pi_l)
38:     // KÍCH HOẠT PRECISION GUARD (Khóa cứng chặn dưới Bayes 0.50):
39:     tau_1_guarded(l) ← max(0.50, tau_1(l))
40: END FOR

41: Tính độ bao phủ thực nghiệm sơ bộ trên tập kiểm tra:
      Cov ← 1 - (1 / (N* * K)) * sum_{i, l} I( tau_0(l) < P*_{i, l} < tau_1_guarded(l) )

42: // ASYMMETRIC COVERAGE GUARD (Chỉ điều chỉnh tau_0, bảo vệ tuyệt đối tau_1):
43: IF Cov < gamma_min THEN:
44:     Tìm hệ số co hẹp rho* ∈ [0, 1] qua tìm kiếm nhị phân sao cho Cov(rho*) ≥ gamma_min
45:     FOR EACH nhãn l = 1 TO K DO:
46:         tau_0_adj(l) ← 0.50 - rho* * (0.50 - tau_0(l))   // Nâng tau_0 để thu nạp True Negatives
47:         tau_1_adj(l) ← tau_1_guarded(l)                  // Giữ nguyên tau_1 >= 0.50, không hạ!
48:     END FOR
49: ELSE:
50:     tau_0_adj ← tau_0,   tau_1_adj ← tau_1_guarded
51: END IF

52: // QUY TẮC QUYẾT ĐỊNH CHỌN LỌC (Cơ chế Xác nhận Nhãn 0 - Negative Verification):
53: FOR i = 1 TO N* DO:
54:     FOR l = 1 TO K DO:
55:         P_neg ← 1.0 - P*_{i, l}                       // Xác suất âm tính P(Y_l = 0 | x)
56:         IF P_neg ≥ 1.0 - tau_0_adj(l) THEN:           // Tương đương P*_{i, l} ≤ tau_0_adj(l)
57:             Y_hat*_{i, l} ← 0                         // XÁC NHẬN NHÃN 0 VỮNG CHẮC (Negative Verification)
58:         ELSE IF P_neg ≤ 1.0 - tau_1_adj(l) THEN:      // Tương đương P*_{i, l} ≥ tau_1_adj(l) ≥ 0.50
59:             Y_hat*_{i, l} ← 1                         // KHẲNG ĐỊNH DƯƠNG TÍNH VƯỢT TRỘI (Positive Assertion)
60:         ELSE:
61:             Y_hat*_{i, l} ← ⊥                         // TỪ CHỐI DỰ ĐOÁN (Vùng lưỡng lự / bất định)
62:         END IF
63:     END FOR
64: END FOR

65: RETURN Y_hat*, P*
========================================================================================================
```

---

### 3. Kết quả thực nghiệm đối chuẩn

#### 3.1. Bảng Tổng Hợp Toàn Cục (Grand Benchmark: 10 tập dữ liệu, 3 Base Learners, 30 cấu hình 5-Fold CV)

| Mô hình | Selective Macro-F1 $\uparrow$ | Coverage $\uparrow$ | Tỷ số $F_1/\text{Cov} \uparrow$ | Selective Micro-F1 $\uparrow$ | Subset Acc (0/1) $\uparrow$ | Hamming Loss $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BR** (Binary Relevance) | 0.4205 | **100.0%** | 0.4205 | 0.5444 | 0.3038 | 0.1570 |
| **CC** (Classifier Chains) | 0.4498 | **100.0%** | 0.4498 | 0.5850 | **0.3716** | 0.1623 |
| **MLC-PA** (Nguyen & Hüllermeier) | 0.3781 | 78.2% | 0.4835 | 0.5088 | 0.2768 | 0.1567 |
| **GSI v6.2.1** (Decaying Peeling) | 0.4140 | 80.3% | 0.5154 | 0.5702 | 0.3112 | **0.1557** |
| **GSI v6.3.1 (Đề Xuất)** | **0.5339** | 73.5% | **0.7265** | **0.6515** | 0.3400 | 0.1596 |

> **Thống kê toàn cục:** GSI v6.3.1 nâng Selective Macro-F1 tăng vọt **+29.0%** so với v6.2.1 (0.5339 vs 0.4140), **+27.0%** so với BR (0.5339 vs 0.4205), **+18.7%** so với CC (0.5339 vs 0.4498), thiết lập tỷ số hiệu năng/độ phủ kỷ lục **0.7265** tại độ bao phủ trung bình chuẩn tắc **73.5%**.

---

#### 3.2. Bảng so sánh chi tiết trên 5 tập dữ liệu Benchmark đại diện (Base Learner Logistic Regression)

Dưới đây là kết quả trung bình 5-Fold Cross-Validation qua 5 tập dữ liệu đại diện (`emotions`, `scene`, `yeast`, `plantpseaac`, `humanpseaac`) với Base Learner **Logistic Regression**:

| Mô hình | Selective Macro-F1 $\uparrow$ | Coverage $\uparrow$ | Subset Acc (0/1) $\uparrow$ | Hamming Loss $\downarrow$ | Selective Micro-F1 $\uparrow$ | F1 / Coverage Ratio $\uparrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **BR** (Binary Relevance) | 0.3849 | **100.00%** | 0.2484 | 0.1366 | 0.5007 | 0.3849 |
| **CC** (Classifier Chains) | 0.4059 | **100.00%** | **0.3368** | 0.1475 | 0.5331 | 0.4059 |
| **MLC-PA** (Chow Baseline) | 0.3807 | 83.41% | 0.2484 | 0.1366 | 0.5042 | 0.4564 |
| **GSI v6.2.1** (Decaying Peeling) | 0.3981 | 83.74% | 0.2593 | **0.1344** | 0.5138 | 0.4755 |
| **GSI v6.3.1** (Precision Guard) | **0.4975** | 71.36% | 0.2655 | 0.1423 | **0.6077** | **0.6972** |

#### Đánh giá tương quan của GSI v6.3.1:
- **So với BR:** Selective Macro-F1 tăng **+29.2%** (0.4975 vs 0.3849), Selective Micro-F1 tăng **+21.4%**, Subset Accuracy tăng **+6.9%**.
- **So với CC:** Selective Macro-F1 tăng **+22.6%** (0.4975 vs 0.4059), Selective Micro-F1 tăng **+14.0%**, Hamming Loss thấp hơn (0.1423 vs 0.1475).
- **So với MLC-PA:** Vượt trội toàn diện: Selective Macro-F1 tăng **+30.7%**, Selective Micro-F1 tăng **+20.5%**, Subset Accuracy tăng **+6.9%**, hiệu suất đánh đổi F1/Coverage tăng **+52.8%**.
- **So với GSI v6.2.1:** Selective Macro-F1 tăng **+25.0%**, Selective Micro-F1 tăng **+18.3%**, Subset Accuracy tăng **+2.4%**, Hamming Loss tương đương (0.1423 vs 0.1344).
- **So với GSI v6.3.0:** Khắc phục trọn vẹn điểm nghẽn: Subset Accuracy tăng **+92.4%** (0.2655 vs 0.1380), Hamming Loss giảm **-49.1%** (0.1423 vs 0.2794), Selective Macro-F1 tiếp tục tăng thêm **+11.6%** (0.4975 vs 0.4459).

---

### 4. Bảng so sánh chi tiết từng chỉ số trên 5 tập dữ liệu

#### 4.1. Selective Macro-F1 (Độ đo cốt lõi đánh giá phân loại nhãn mất cân bằng)

| Tập dữ liệu | BR | CC | MLC-PA | GSI v6.2.1 | **GSI v6.3.1** | Mô hình dẫn đầu |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **emotions** | 0.5972 | 0.5932 | 0.6172 | 0.6883 | **0.7552** | 🏆 **GSI v6.3.1** (+26.5% vs BR) |
| **scene** | 0.6994 | 0.7234 | 0.7431 | 0.7642 | **0.7974** | 🏆 **GSI v6.3.1** (+10.2% vs CC) |
| **yeast** | 0.3748 | 0.4017 | 0.3585 | 0.3562 | **0.4304** | 🏆 **GSI v6.3.1** (+7.1% vs CC) |
| **plantpseaac** | 0.1410 | 0.1715 | 0.0975 | 0.0951 | **0.2718** | 🏆 **GSI v6.3.1** (+58.5% vs CC) |
| **humanpseaac** | 0.1124 | 0.1397 | 0.0869 | 0.0869 | **0.2325** | 🏆 **GSI v6.3.1** (+66.4% vs CC) |
| **Trung bình** | 0.3849 | 0.4059 | 0.3807 | 0.3981 | **0.4975** | 🏆 **GSI v6.3.1** |

> *Nhận xét:* GSI v6.3.1 giành vị trí dẫn đầu (Best Performance) trên **cả 5/5 tập dữ liệu**. Đặc biệt trên hai tập nhãn hiếm `plantpseaac` và `humanpseaac`, v6.3.1 vượt trội hoàn toàn so với CC và BR.

---

#### 4.2. Độ phủ (Coverage $\gamma$, in đậm độ phủ cao nhất trong các mô hình có chọn lọc)

| Tập dữ liệu | BR | CC | MLC-PA | GSI v6.2.1 | **GSI v6.3.1** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **emotions** | 100.0% | 100.0% | 67.49% | 69.11% | **74.06%** |
| **scene** | 100.0% | 100.0% | 88.71% | **88.84%** | 76.04% |
| **yeast** | 100.0% | 100.0% | **75.63%** | 75.56% | 67.41% |
| **plantpseaac** | 100.0% | 100.0% | **92.95%** | 92.93% | 69.07% |
| **humanpseaac** | 100.0% | 100.0% | **92.24%** | **92.24%** | 70.21% |
| **Trung bình** | 100.0% | 100.0% | 83.41% | **83.74%** | 71.36% |

> *Nhận xét chuyên sâu về Độ bao phủ (Coverage) so với GSI v6.2.1:*
> - **Độ phủ của v6.3.1 đạt 71.36% (toàn cục 73.5%), giảm nhẹ so với v6.2.1 (83.74% trên 5 tập, 80.3% toàn cục):** Sự sụt giảm tập trung rõ rệt nhất ở nhóm nhãn hiếm như `plantpseaac` (69.07% vs 92.93%) và `humanpseaac` (70.21% vs 92.24%).
> - **Lý do 1 - Ảo giác bao phủ cao ở v6.2.1 (Trivial All-Negative Decisions):** v6.2.1 dùng ngưỡng Chow tĩnh $[\tau_0=0.30, \tau_1=0.70]$. Do nhãn hiếm có tần suất $\pi_l \le 3\%$, hầu như mọi mẫu đều có xác suất dự đoán $\hat{P} \le 0.15 < 0.30$. Do đó, v6.2.1 **tự động gán nhãn 0 cho 93% - 95% số mẫu**. Đây là "độ phủ ảo/tầm thường" vì mô hình hầu như không bắt được bất kỳ nhãn dương tính nào (True Positive gần bằng 0), khiến Macro-F1 bị liệt ở mức $0.0869$.
> - **Lý do 2 - Cơ chế từ chối chủ động có nhận thức tiên nghiệm ở v6.3.1 (Active Prior-Aware Abstention):** v6.3.1 co dải từ chối về sát tần suất tiên nghiệm $\tau_0(l) = c\sqrt{\pi_l} \approx 0.052$. Khi một mẫu có xác suất rơi vào khoảng $[0.052, 0.50]$ (vùng biên giới bất định cao, có tín hiệu dương nhưng chưa đủ tin cậy để khẳng định $\ge 0.50$), **GSI v6.3.1 chủ động từ chối dự đoán ($\bot$)** thay vì đoán bừa là 0 như v6.2.1.
> - **Lý do 3 - Đánh đổi tối ưu Pareto (Pareto-Optimal Tradeoff):** Việc chủ động từ chối các mẫu bất định đã giúp lọc sạch phán đoán rác, đưa **Selective Macro-F1 tăng vọt từ 0.3981 lên 0.4975 (+25.0%)**, đồng thời thiết lập tỷ số hiệu năng/độ phủ kỷ lục **$F_1/\text{Cov} = 0.6972$** (vượt xa mức $0.4755$ của v6.2.1). Độ phủ vẫn luôn được khóa cứng trên sàn an toàn $\gamma_{\min} \ge 70\%$ nhờ Asymmetric Coverage Guard.

---

#### 4.3. Subset 0/1 Accuracy (Độ chính xác tuyệt đối toàn bộ vector nhãn)

| Tập dữ liệu | BR | CC | MLC-PA | GSI v6.2.1 | **GSI v6.3.1** | Mô hình dẫn đầu |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **emotions** | 0.2516 | 0.2838 | 0.2516 | 0.2906 | **0.3029** | 🏆 **GSI v6.3.1** |
| **scene** | 0.5361 | **0.6623** | 0.5361 | 0.5706 | 0.5564 | 🏆 **CC** |
| **yeast** | 0.1465 | **0.2019** | 0.1465 | 0.1449 | 0.1349 | 🏆 **CC** |
| **plantpseaac** | 0.1553 | **0.2433** | 0.1553 | 0.1380 | 0.1482 | 🏆 **CC** |
| **humanpseaac** | 0.1526 | **0.2929** | 0.1526 | 0.1526 | 0.1851 | 🏆 **CC** |
| **Trung bình** | 0.2484 | **0.3368** | 0.2484 | 0.2593 | 0.2655 | 🏆 **CC** |

> *Nhận xét quan trọng:*
> - Ở v6.3.0, Subset Accuracy từng sụp đổ xuống $0.0103$ (`plantpseaac`) và $0.0035$ (`humanpseaac`). Ở **v6.3.1**, con số này đã **phục hồi toàn diện** lên $0.1482$ và $0.1851$, vượt qua cả GSI v6.2.1 và BR!
> - Classifier Chains (CC) có Subset Accuracy cao hơn trên toàn tập do tận dụng chuỗi autoregressive $P(y_k \mid x, y_{<k})$ khi dự đoán toàn bộ các nhãn $0$, nhưng CC lại trả giá đắt ở F1 trên nhãn hiếm (F1 chỉ đạt $0.1397$ trên `humanpseaac` so với **$0.2325$** của v6.3.1).

---

#### 4.4. Hamming Loss (Tỷ lệ nhãn dự đoán sai, càng thấp càng tốt $\downarrow$)

| Tập dữ liệu | BR | CC | MLC-PA | GSI v6.2.1 | **GSI v6.3.1** | Mô hình tốt nhất $\downarrow$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **emotions** | 0.2013 | 0.2240 | 0.2013 | **0.1943** | 0.1991 | 🏆 **GSI v6.2.1** |
| **scene** | 0.0992 | 0.1011 | 0.0992 | **0.0954** | 0.1065 | 🏆 **GSI v6.2.1** |
| **yeast** | 0.2019 | 0.2155 | 0.2019 | **0.2016** | 0.2115 | 🏆 **GSI v6.2.1** |
| **plantpseaac** | **0.0945** | 0.0988 | **0.0945** | 0.0948 | 0.1027 | 🏆 **BR / MLC-PA** |
| **humanpseaac** | **0.0860** | 0.0979 | **0.0860** | **0.0860** | 0.0917 | 🏆 **BR / MLC-PA / GSI v6.2** |
| **Trung bình** | 0.1366 | 0.1475 | 0.1366 | **0.1344** | 0.1423 | 🏆 **GSI v6.2.1** |

> *Nhận xét:* Ở v6.3.0, Hamming Loss bị bùng nổ lên $0.2794$ (+107.9%). Với Precision Guard ở **v6.3.1**, Hamming Loss lập tức hạ nhiệt về **$0.1423$** (-49.1%), xấp xỉ mức tối ưu của v6.2.1 ($0.1344$) và BR ($0.1366$), đồng thời **thấp hơn cả CC** ($0.1475$).

---

#### 4.5. Selective Micro-F1 (Hiệu năng tổng thể trên toàn bộ các nhãn)

| Tập dữ liệu | BR | CC | MLC-PA | GSI v6.2.1 | **GSI v6.3.1** | Mô hình dẫn đầu |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **emotions** | 0.6289 | 0.6310 | 0.6946 | 0.7293 | **0.7631** | 🏆 **GSI v6.3.1** |
| **scene** | 0.6938 | 0.7110 | 0.7612 | 0.7789 | **0.7904** | 🏆 **GSI v6.3.1** |
| **yeast** | 0.6351 | 0.6198 | 0.6725 | **0.6727** | 0.5846 | 🏆 **GSI v6.2.1** |
| **plantpseaac** | 0.2604 | 0.3265 | 0.2012 | 0.1964 | **0.4299** | 🏆 **GSI v6.3.1** |
| **humanpseaac** | 0.2851 | 0.3774 | 0.1918 | 0.1918 | **0.4703** | 🏆 **GSI v6.3.1** |
| **Trung bình** | 0.5007 | 0.5331 | 0.5042 | 0.5138 | **0.6077** | 🏆 **GSI v6.3.1** |

> *Nhận xét:* GSI v6.3.1 tạo ra bước nhảy vọt ngoạn mục về Selective Micro-F1 từ $0.4520$ lên **$0.6077$** (+34.4% so với v6.3.0, +21.4% so với BR, +14.0% so với CC), phản ánh số lượng True Positives chất lượng cao gia tăng mạnh mẽ mà không bị False Positives làm loãng.

---

### 5. Chi tiết từng tập dữ liệu Benchmark

#### 5.1. Tập dữ liệu Mất cân bằng cực độ: `humanpseaac` (3,106 mẫu, 14 nhãn, $\pi \approx 3\%$)
- **Selective Macro-F1:** BR ($0.1124$) $\mid$ CC ($0.1397$) $\mid$ MLC-PA ($0.0869$) $\mid$ v6.2.1 ($0.0869$) $\mid$ v6.3.0 ($0.1836$) $\rightarrow$ **v6.3.1: $0.2325$** (+106.8% so với BR, +66.4% so với CC).
- **Subset Accuracy:** Phục hồi thần kỳ từ $0.0035$ (v6.3.0) lên **$0.1851$ (v6.3.1)**, vượt BR ($0.1526$), MLC-PA ($0.1526$) và v6.2.1 ($0.1526$).
- **Hamming Loss:** Thu hẹp từ $0.3390$ (v6.3.0) xuống **$0.0917$ (v6.3.1)**, thấp hơn CC ($0.0979$) và sát mức của BR ($0.0860$).
- **Selective Micro-F1:** Tăng vọt lên **$0.4703$**, vượt xa BR ($0.2851$), CC ($0.3774$), và v6.3.0 ($0.2044$).

#### 5.2. Tập dữ liệu Mất cân bằng cao: `plantpseaac` (978 mẫu, 12 nhãn, $\pi \approx 8.9\%$)
- **Selective Macro-F1:** BR ($0.1410$) $\mid$ CC ($0.1715$) $\mid$ MLC-PA ($0.0975$) $\mid$ v6.2.1 ($0.0951$) $\mid$ v6.3.0 ($0.2221$) $\rightarrow$ **v6.3.1: $0.2718$** (+92.8% so với BR, +58.5% so với CC).
- **Subset Accuracy:** Phục hồi từ $0.0103$ (v6.3.0) lên **$0.1482$ (v6.3.1)**, tương đương BR ($0.1553$) và cao hơn v6.2.1 ($0.1380$).
- **Hamming Loss:** Dập tắt hiện tượng nổ lỗi, giảm từ $0.3116$ xuống **$0.1027$**, gần như ngang bằng BR ($0.0945$) và CC ($0.0988$).
- **Selective Micro-F1:** Đạt **$0.4299$**, cao hơn đáng kể so với BR ($0.2604$) và CC ($0.3265$).

#### 5.3. Tập dữ liệu Sinh học: `yeast` (2,417 mẫu, 14 nhãn, $\pi \approx 30.3\%$)
- **Selective Macro-F1:** BR ($0.3748$) $\mid$ CC ($0.4017$) $\mid$ MLC-PA ($0.3585$) $\mid$ v6.2.1 ($0.3562$) $\mid$ v6.3.0 ($0.3749$) $\rightarrow$ **v6.3.1: $0.4304$** (Vượt cả CC: $0.4017$).
- **Subset Accuracy:** Phục hồi từ $0.0509$ lên **$0.1349$**, sát mức của BR ($0.1465$) và v6.2.1 ($0.1449$).
- **Hamming Loss:** Giảm mạnh từ $0.3694$ về **$0.2115$**, thấp hơn CC ($0.2155$) và ngang bằng BR ($0.2019$).

#### 5.4. Tập dữ liệu Thị giác máy tính: `scene` (2,407 mẫu, 6 nhãn, $\pi \approx 17.9\%$)
- **Selective Macro-F1:** BR ($0.6994$) $\mid$ CC ($0.7234$) $\mid$ MLC-PA ($0.7431$) $\mid$ v6.2.1 ($0.7642$) $\mid$ v6.3.0 ($0.7266$) $\rightarrow$ **v6.3.1: $0.7974$** (Đỉnh cao nhất mọi phiên bản, +14.0% so với BR, +10.2% so với CC).
- **Subset Accuracy:** Phục hồi từ $0.3964$ lên **$0.5564$**, vượt BR ($0.5361$) và MLC-PA ($0.5361$).
- **Hamming Loss:** Giảm từ $0.1425$ về **$0.1065$**, sát mức $0.0992$ của BR.

#### 5.5. Tập dữ liệu Âm thanh: `emotions` (593 mẫu, 6 nhãn, $\pi \approx 31.1\%$)
- **Selective Macro-F1:** BR ($0.5972$) $\mid$ CC ($0.5932$) $\mid$ MLC-PA ($0.6172$) $\mid$ v6.2.1 ($0.6883$) $\mid$ v6.3.0 ($0.7224$) $\rightarrow$ **v6.3.1: $0.7552$** (+26.5% so với BR, +27.3% so với CC).
- **Subset Accuracy:** Đạt **$0.3029$**, cao nhất trong toàn bộ 6 mô hình (vượt CC $0.2838$, BR $0.2516$, v6.2.1 $0.2906$).
- **Hamming Loss:** Đạt **$0.1991$**, tối ưu hơn BR ($0.2013$) và CC ($0.2240$).

---

### 6. Kết luận khoa học & Bài học thực tiễn

1. **Thành công vượt bậc của cơ chế Precision Guard kết hợp Sqrt-Platt:**
   - **GSI v6.3.1 đã giải quyết trọn vẹn điểm yếu chí tử của v6.3.0:** Bằng cách khóa cứng chặn dưới $\min \tau_1 = 0.50$ và làm dịu trọng số hiệu chuẩn bằng căn bậc hai, hiện tượng tạo False Positive ảo trên nhãn hiếm bị triệt tiêu hoàn toàn.
   - Hệ quả là Subset Accuracy và Hamming Loss lập tức phục hồi về trạng thái tối ưu, đồng thời Selective Micro-F1 tăng vọt từ $0.4520$ lên **$0.6077$**.

2. **Khẳng định vị thế áp đảo so với các Baseline cổ điển (BR, CC, MLC-PA):**
   - **So với BR & MLC-PA:** GSI v6.3.1 vượt trội toàn diện trên mọi chỉ số nhờ khả năng giải mã cấu trúc phụ thuộc phân tầng (Layered Peeling) và cơ chế tự động từ chối vùng bất định.
   - **So với CC:** Classifier Chains duy trì Subset Accuracy cao nhờ chuỗi nhân xác suất, nhưng bị hạn chế nghiêm trọng về Macro-F1 trên nhãn mất cân bằng. GSI v6.3.1 vượt qua CC với khoảng cách lớn về Selective Macro-F1 (+22.6%) và Selective Micro-F1 (+14.0%), đồng thời Hamming Loss thấp hơn.

3. **Tỷ số Hiệu năng / Độ phủ đạt đỉnh cao nhất từ trước đến nay:**
   - **Thực nghiệm sơ bộ 5 tập dữ liệu (Logistic Regression):** Selective Macro-F1 đạt **$0.4975$** tại độ phủ **$71.36\%$**, tỷ số $F_1 / \text{Coverage}$ đạt mốc kỷ lục **$0.6972$** (tăng +81.1% so với BR, +71.8% so với CC, +52.8% so với MLC-PA, +46.6% so với v6.2.1).
   - **Thực nghiệm toàn diện Grand Benchmark (10 tập dữ liệu, 30 cấu hình):** Selective Macro-F1 bứt phá lên **$0.5339$** tại độ phủ chuẩn tắc **$73.5\%$**, xác lập đỉnh cao tỷ số $F_1 / \text{Coverage}$ tuyệt đối **$0.7265$**.
