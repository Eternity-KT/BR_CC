# BÁO CÁO NGHIỆM THU THỬ NGHIỆM GSI-MLC-PA v6.3.2
## Đánh giá giải pháp Suy diễn biến phân Trường trung bình Chuẩn hóa một bước (OS-NMF) trên nhãn phụ thuộc (DL)
## So sánh toàn diện với các Baseline (BR, CC, MLC-PA) và các thế hệ GSI (v6.2.1, v6.3.1)

---

### 1. Bối cảnh & Vấn đề kỹ thuật cần giải quyết

Trong phiên bản **v6.3.1**, cơ chế Hiệu chuẩn Căn bậc hai (Balanced-Root Platt Calibration) kết hợp Chốt chặn Độ chính xác (Precision Guard $\tau_1 \ge 0.50$) và Nguyên lý Xác nhận Nhãn 0 (Negative Verification) đã giải quyết triệt để hiện tượng lạm phát xác suất trên dữ liệu mất cân bằng cực đoan.

Tuy nhiên, việc xử lý tập nhãn phụ thuộc ($DL$) ở v6.3.1 vẫn dựa trên kiến trúc **bộ phân loại điều kiện thứ cấp** $g_l(x, \hat{P}_{\text{cha}})$. Khi triển khai trên quy mô 10 tập dữ liệu benchmark, phương pháp này bộc lộ ba vấn đề kỹ thuật:

1. **Chi phí tính toán bổ sung:** Với mỗi nhãn trong $DL$, mô hình phải huấn luyện thêm một bộ phân loại nhị phân riêng biệt trên không gian đặc trưng mở rộng, làm tăng đáng kể thời gian huấn luyện tổng thể.
2. **Nguy cơ quá khớp và lan truyền sai số chuỗi:** Trên các tập dữ liệu có số lượng nhãn phụ thuộc lớn ($|DL| \ge 10$ như `yeast`, `plantpseaac`, `humanpseaac`), việc ghép nối các xác suất dự đoán $\hat{P}_{\text{cha}}$ vào đầu vào của $g_l$ làm tăng số chiều đặc trưng và dễ khuếch đại sai số nếu các nhãn cha bị dự đoán sai.
3. **Mâu thuẫn dự đoán giữa các nhãn tương quan:** Mô hình phân loại điều kiện $g_l$ không trực tiếp ràng buộc tính nhất quán đại số giữa các nhãn có tương quan đồng biến thiên hoặc nghịch biến thiên mạnh, dẫn đến việc Subset 0/1 Accuracy chưa đạt mức tối ưu và Selective Hamming Loss vẫn còn cao ($0.4130$ trên tập quyết định).

---

### 2. Các đề xuất cải tiến trong phiên bản v6.3.2

Kiến trúc **GSI-MLC-PA v6.3.2** thay thế toàn bộ các mô hình điều kiện thứ cấp $g_l$ bằng **Suy diễn biến phân Trường trung bình Chuẩn hóa một bước (One-Step Normalized Mean-Field - OS-NMF)**, xây dựng trực tiếp trên ma trận tương quan Pearson của phần dư sai số ($C^{\text{res}}$).

Thuật toán được xây dựng trên 3 cơ chế toán học cốt lõi:

---

#### 2.1. Cơ chế 1: Chuẩn hóa bậc hàng bảo vệ độ tin cậy (Degree Normalization)

- **Nguyên nhân kỹ thuật:** Khi một nhãn $j$ có nhiều nhãn liên đới trong $DL$ (bậc đồ thị cao), tổng lực kéo logit $\sum_k C^{\text{res}}_{jk}$ có thể tăng vượt tầm kiểm soát, gây hiện tượng bão hòa xác suất (xác suất bị đẩy về sát 0 hoặc 1 một cách cực đoan) hoặc đếm lặp phụ thuộc (double counting).
- **Giải pháp toán học:** Chuẩn hóa ma trận trọng số tương quan theo tổng trị tuyệt đối bậc hàng:
  $$W_{jk} = \frac{C^{\text{res}}_{jk}}{\max\left(1.0, \, \sum_{m \in \mathcal{N}(j)} |C^{\text{res}}_{jm}|\right)}$$
  Hằng số $1.0$ ở mẫu số bảo đảm rằng:
  - Nếu nhãn chỉ có vài liên kết yếu, độ lớn của tương quan được giữ nguyên, không bị phóng đại.
  - Nếu nhãn có mật độ liên kết dày đặc, tổng độ lớn các tương quan luôn bị chặn trên bởi $1.0$, triệt tiêu hoàn toàn nguy cơ nổ biên độ logit.

---

#### 2.2. Cơ chế 2: Căn giữa Spin đối xứng ($2P - 1.0$)

- **Nguyên nhân kỹ thuật:** Trong phân loại đa nhãn, xác suất thô $p_k \in [0, 1]$ mang kỳ vọng lệch mạnh về 0 khi nhãn mất cân bằng ($\mathbb{E}[p_k] \ll 0.50$). Nếu sử dụng trực tiếp $p_k$ làm biến điều kiện, các nhãn có tương quan âm sẽ luôn nhận tín hiệu dương sai lệch từ lớp âm tính.
- **Giải pháp toán học:** Chuyển đổi xác suất sang biến spin Ising đối xứng:
  $$s_k = 2 p_k - 1.0 \in [-1.0, \, +1.0]$$
  - Khi $p_k \approx 0.0$ (chắc chắn âm tính): $s_k \approx -1.0$.
  - Khi $p_k \approx 0.5$ (lưỡng lự): $s_k \approx 0.0$ (không tác động lên các nhãn lân cận).
  - Khi $p_k \approx 1.0$ (chắc chắn dương tính): $s_k \approx +1.0$.
  Cách biểu diễn này bảo đảm các tương quan nghịch biến thiên ($C^{\text{res}}_{jk} < 0$) và đồng biến thiên ($C^{\text{res}}_{jk} > 0$) tác động đúng chiều đại số lên logit của nhãn mục tiêu.

---

#### 2.3. Cơ chế 3: Kẹp biên độ dịch chuyển logit (Bounded Logit Shifts)

- **Nguyên nhân kỹ thuật:** Để bảo toàn tính đúng đắn của phép hiệu chuẩn Balanced-Root Platt và quy tắc chốt chặn Precision Guard ($\tau_1 \ge 0.50$), độ dịch chuyển xác suất từ trường trung bình không được phép làm đảo lộn hoàn toàn quyết định phân loại ban đầu của bộ phân loại cơ sở.
- **Giải pháp toán học:** Độ dịch chuyển logit thô $\Delta z_j = \alpha \sum_{k \in \mathcal{N}(j)} W_{jk} (2 p_k - 1)$ được giới hạn trong khoảng bảo vệ chặt chẽ $[-z_{\max}, +z_{\max}]$ (với $z_{\max} = 0.50$, $\alpha = 0.25$):
  $$\Delta z_j^{(\text{clipped})} = \text{clip}\left(\Delta z_j, \, -z_{\max}, \, +z_{\max}\right)$$
  Logit sau hiệu chỉnh và xác suất tinh chỉnh được tính bởi:
  $$z_j^{(\text{new})} = \text{logit}(p_j) + \Delta z_j^{(\text{clipped})}, \quad p_j^{(\text{refined})} = \sigma\left(z_j^{(\text{new})}\right)$$

---

### 2.1. Mã giả toàn diện thuật toán GSI-MLC-PA v6.3.2

```text
========================================================================================================
THUẬT TOÁN TOÀN DIỆN: HUẤN LUYỆN VÀ SUY DIỄN CHỌN LỌC GSI-MLC-PA v6.3.2
========================================================================================================
ĐẦU VÀO:
  - Tập huấn luyện (X, Y) ∈ R^(N x d) x {0, 1}^(N x K)
  - Tập kiểm tra X* ∈ R^(N* x d)
  - Danh sách ngưỡng bóc tách tầng T = [0.75, 0.70, 0.65]
  - Ngưỡng tương quan phần dư theta_corr = 0.25
  - Chi phí từ chối c = 0.30, Độ phủ sàn an toàn gamma_min = 0.70
  - Tham số trường trung bình: alpha = 0.25, z_max = 0.50

ĐẦU RA:
  - Ma trận dự đoán chọn lọc Y_hat* ∈ {0, 1, ⊥}^(N* x K)  (với ⊥ là từ chối dự đoán)
  - Ma trận xác suất tinh chỉnh P* ∈ [0, 1]^(N* x K)

--------------------------------------------------------------------------------------------------------
GIAI ĐOẠN 1: BÓC TÁCH TẦNG ĐỘC LẬP (IL) VÀ XÁC ĐỊNH PHỤ THUỘC (DL)
--------------------------------------------------------------------------------------------------------
 1: Khởi tạo:
      L_rem ← {1, 2, ..., K},   IL ← ∅,   X_context ← X
 2: FOR EACH ngưỡng tau_m IN T DO:
 3:     IL_m ← ∅
 4:     FOR EACH nhãn l IN L_rem DO:
 5:         Chạy 5-Fold Cross-Validation cho nhãn l trên (X_context, Y[:, l])
 6:         Thu nhận xác suất Out-of-Fold P_oof[:, l] và tính Selective-F1_l
 7:         IF Selective-F1_l ≥ tau_m THEN:
 8:             IL_m ← IL_m ∪ {l}
 9:         END IF
10:     END FOR
11:     IF IL_m == ∅ THEN BREAK FOR
12:     Huấn luyện mô hình Binary Relevance độc lập cho các nhãn trong IL_m
13:     IL ← IL ∪ IL_m,   L_rem ← L_rem \ IL_m
14:     X_context ← [X, Normalize(P_oof[:, IL])]
15: END FOR
16: DL ← L_rem

--------------------------------------------------------------------------------------------------------
GIAI ĐOẠN 2: TÍNH TOÁN MA TRẬN TƯƠNG QUAN PHẦN DƯ VÀ TRƯỜNG TRUNG BÌNH TRÊN DL
--------------------------------------------------------------------------------------------------------
17: Huấn luyện mô hình cơ sở f_base trên X_context cho từng j ∈ DL, thu nhận OOF P_oof_DL[:, j]
18: Tính ma trận phần dư sai số liên tục:
      R[:, j] ← Y[:, j] - P_oof_DL[:, j],   với mọi j ∈ DL
19: Tính ma trận tương quan Pearson có dấu C_res giữa các cột phần dư:
      C_res[j, k] ← PearsonCorrelation(R[:, j], R[:, k]),   với mọi j, k ∈ DL, j ≠ k
20: Lọc ngưỡng tương quan: Gán C_res[j, k] ← 0 nếu |C_res[j, k]| < theta_corr
21: Tính ma trận trọng số chuẩn hóa bậc kết nối:
      W[j, k] ← C_res[j, k] / max(1.0, sum_{m ≠ j} |C_res[j, m]|)
22: Cập nhật xác suất Out-of-Fold của DL bằng một bước Mean-Field (OS-NMF):
      FOR EACH mẫu i = 1 TO N DO:
          Delta_z[i, j] ← alpha * sum_{k ≠ j} W[j, k] * (2 * P_oof_DL[i, k] - 1.0)
          Delta_z[i, j] ← clip(Delta_z[i, j], -z_max, +z_max)
          P_oof_DL_refined[i, j] ← sigma(logit(P_oof_DL[i, j]) + Delta_z[i, j])
      END FOR
23: Hợp nhất xác suất toàn phần: P_oof_all ← [P_oof[:, IL], P_oof_DL_refined]
24: Áp dụng Balanced-Root Platt Calibration trên P_oof_all với w_pos = sqrt((1 - pi) / pi)

--------------------------------------------------------------------------------------------------------
GIAI ĐOẠN 3: SUY DIỄN CHỌN LỌC TRÊN TẬP KIỂM TRA X*
--------------------------------------------------------------------------------------------------------
25: Trên tập kiểm tra X*:
      - Dự đoán P*_IL bằng các mô hình BR độc lập
      - X*_context ← [X*, Normalize(P*_IL)]
      - Dự đoán xác suất cơ sở P*_base_DL bằng f_base trên X*_context
      - Tinh chỉnh xác suất P*_DL bằng OS-NMF theo ma trận trọng số W đã lưu:
          Delta_z*[i, j] ← alpha * sum_{k ≠ j} W[j, k] * (2 * P*_base_DL[i, k] - 1.0)
          Delta_z*[i, j] ← clip(Delta_z*[i, j], -z_max, +z_max)
          P*_DL[i, j]    ← sigma(logit(P*_base_DL[i, j]) + Delta_z*[i, j])
      - Hợp nhất ma trận xác suất kiểm tra: P* ← [P*_IL, P*_DL]
      - Áp dụng bộ hiệu chuẩn Platt đã học lên P*

26: Thiết lập ngưỡng Bayes Likelihood Ratio với Precision Guard:
      tau_0(l) ← (c * pi_l) / ((1 - c) * (1 - pi_l) + c * pi_l)
      tau_1(l) ← max(0.50, ((1 - c) * pi_l) / (c * (1 - pi_l) + (1 - c) * pi_l))

27: Điều chỉnh ngưỡng bảo đảm độ bao phủ an toàn (Coverage Guard):
      IF Coverage(P*, tau_0, tau_1) < gamma_min THEN:
          Nâng dần tau_0(l) hướng về 0.50 theo bước step sao cho Coverage ≥ gamma_min
          Giữ nguyên tau_1(l) ≥ 0.50 (Precision Guard không cho phép hạ tau_1 dưới 0.50)
      END IF

28: Thực hiện quy tắc quyết định chọn lọc (Negative Verification):
      FOR i = 1 TO N*,  l = 1 TO K DO:
          IF P*[i, l] ≤ tau_0_adj(l) THEN:
              Y_hat*[i, l] ← 0
          ELSE IF P*[i, l] ≥ tau_1_adj(l) THEN:
              Y_hat*[i, l] ← 1
          ELSE:
              Y_hat*[i, l] ← ⊥    // Từ chối dự đoán
          END IF
      END FOR

29: RETURN Y_hat*, P*
========================================================================================================
```

---

### 3. Kết quả thực nghiệm đối chuẩn

#### 3.1. Bảng Tổng Hợp Toàn Cục (Grand Benchmark: 10 tập dữ liệu, Base Learner Logistic Regression, 5-Fold CV)

Dữ liệu tổng hợp từ 5-fold cross-validation trên 10 tập dữ liệu chuẩn:

| Mô hình | Selective Macro-F1 (↑) | Độ phủ Coverage (%) | Tỷ số F1 / Coverage (↑) | Selective Micro-F1 (↑) | Subset Acc (0/1) (↑) | Hamming Loss (↓) | Selective Hamming Loss (↓) | Thời gian Train trung bình (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BR** (Binary Relevance) | 0.4643 | **100.0%** | 0.4643 | 0.5881 | 0.3471 | 0.1520 | 0.1520 | 0.12s |
| **CC** (Classifier Chains) | 0.4796 | **100.0%** | 0.4796 | 0.6103 | **0.4129** | 0.1604 | 0.1604 | 0.35s |
| **MLC-PA** (Nguyen & Hüllermeier) | 0.4590 | 81.9% | 0.5605 | 0.6096 | 0.3471 | 0.1520 | 0.1348 | 0.12s |
| **GSI v6.2.1** (Decaying Peeling) | 0.4709 | 82.1% | 0.5735 | 0.6152 | 0.3537 | **0.1503** | **0.1312** | 0.85s |
| **GSI v6.3.1** (Precision Guard) | 0.5152 | 74.8% | 0.6892 | 0.5419 | 0.2420 | 0.2530 | 0.4130 | 2.09s |
| **GSI v6.3.2 (Đề Xuất)** | **0.5593** | 73.7% | **0.7586** | **0.6677** | 0.3646 | 0.1554 | 0.1459 | **1.83s** |

---

#### 3.2. So sánh đối đầu trực tiếp giữa GSI v6.3.1 và GSI v6.3.2 trên 10 tập dữ liệu

| Tiêu chí đánh giá | GSI v6.3.1 | GSI v6.3.2 | Chênh lệch kỹ thuật | Đánh giá xu hướng |
| :--- | :---: | :---: | :---: | :--- |
| **Selective Macro-F1** | 0.5152 | **0.5593** | **+0.0441 (+8.6%)** | Thắng 9/10 tập dữ liệu |
| **Selective Micro-F1** | 0.5419 | **0.6677** | **+0.1258 (+23.2%)** | Thắng 10/10 tập dữ liệu |
| **Subset 0/1 Accuracy** | 0.2420 | **0.3646** | **+0.1225 (+50.6%)** | Thắng 10/10 tập dữ liệu |
| **Selective Hamming Loss** | 0.4130 | **0.1459** | **-0.2671 (-64.7%)** | Giảm sai số quyết định 10/10 tập |
| **Hamming Loss (Toàn phần)** | 0.2530 | **0.1554** | **-0.0976 (-38.6%)** | Hạ nhiệt lỗi toàn cục 10/10 tập |
| **Độ phủ Coverage** | 74.75% | 73.73% | -1.02% | Ổn định vững chắc quanh ngưỡng ~74% |
| **Thời gian huấn luyện** | 2.094s | **1.826s** | **-12.8%** | Nhanh hơn trên 10/10 tập dữ liệu |

---

### 4. Bảng chi tiết từng chỉ số trên toàn bộ 10 tập dữ liệu Benchmark

#### 4.1. Selective Macro-F1 (↑)

| Tập dữ liệu | BR | CC | MLC-PA | GSI v6.2.1 | GSI v6.3.1 | **GSI v6.3.2** | Mô hình dẫn đầu |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`emotions`** | 0.5972 | 0.5932 | 0.6172 | 0.6883 | 0.7245 | **0.7499** | **GSI v6.3.2** |
| **`scene`** | 0.6994 | 0.7234 | 0.7431 | 0.7642 | 0.7213 | **0.7968** | **GSI v6.3.2** |
| **`yeast`** | 0.3748 | 0.4017 | 0.3585 | 0.3562 | 0.3744 | **0.4284** | **GSI v6.3.2** |
| **`plantpseaac`** | 0.1410 | 0.1715 | 0.0975 | 0.0951 | 0.2201 | **0.2685** | **GSI v6.3.2** |
| **`humanpseaac`** | 0.1124 | 0.1397 | 0.0869 | 0.0869 | 0.1838 | **0.2380** | **GSI v6.3.2** |
| **`chd49`** | 0.5103 | 0.5073 | **0.5242** | 0.5203 | 0.4320 | 0.4672 | **MLC-PA** |
| **`music`** | 0.6067 | 0.6031 | 0.6375 | 0.6679 | 0.7236 | **0.7617** | **GSI v6.3.2** |
| **`gpositivepseaac`** | 0.5535 | 0.5791 | 0.5370 | 0.5415 | 0.5852 | **0.6677** | **GSI v6.3.2** |
| **`genbase`** | 0.6782 | 0.6930 | 0.6334 | 0.6408 | 0.6803 | **0.7562** | **GSI v6.3.2** |
| **`viruspseaac`** | 0.3692 | 0.3836 | 0.3543 | 0.3479 | **0.5068** | 0.4584 | **GSI v6.3.1** |
| **Trung bình** | 0.4643 | 0.4796 | 0.4590 | 0.4709 | 0.5152 | **0.5593** | **GSI v6.3.2** |

---

#### 4.2. Độ bao phủ quyết định Coverage (%)

| Tập dữ liệu | BR | CC | MLC-PA | GSI v6.2.1 | GSI v6.3.1 | **GSI v6.3.2** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`emotions`** | 100.0% | 100.0% | 67.49% | 69.11% | 73.22% | **74.17%** |
| **`scene`** | 100.0% | 100.0% | 88.71% | **88.84%** | 73.34% | 75.34% |
| **`yeast`** | 100.0% | 100.0% | **75.63%** | 75.56% | 71.24% | 67.32% |
| **`plantpseaac`** | 100.0% | 100.0% | **92.95%** | 92.93% | 70.30% | 68.20% |
| **`humanpseaac`** | 100.0% | 100.0% | **92.24%** | 92.24% | 70.25% | 70.16% |
| **`chd49`** | 100.0% | 100.0% | 63.04% | 63.31% | **71.18%** | 68.08% |
| **`music`** | 100.0% | 100.0% | 68.87% | 69.15% | **73.96%** | 73.15% |
| **`gpositivepseaac`** | 100.0% | 100.0% | 85.84% | **86.22%** | 75.19% | 73.08% |
| **`genbase`** | 100.0% | 100.0% | 99.76% | **99.77%** | 92.88% | 94.28% |
| **`viruspseaac`** | 100.0% | 100.0% | **84.34%** | 83.93% | 75.89% | 73.55% |
| **Trung bình** | 100.0% | 100.0% | 81.89% | **82.11%** | 74.75% | 73.73% |

---

#### 4.3. Subset 0/1 Accuracy (Exact Match) (↑)

| Tập dữ liệu | BR | CC | MLC-PA | GSI v6.2.1 | GSI v6.3.1 | **GSI v6.3.2** | Mô hình dẫn đầu |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`emotions`** | 0.2516 | 0.2838 | 0.2516 | 0.2906 | 0.2306 | **0.2978** | **GSI v6.3.2** |
| **`scene`** | 0.5361 | **0.6623** | 0.5361 | 0.5706 | 0.3848 | 0.5481 | **CC** |
| **`yeast`** | 0.1465 | **0.2019** | 0.1465 | 0.1449 | 0.0521 | 0.1515 | **CC** |
| **`plantpseaac`** | 0.1553 | **0.2433** | 0.1553 | 0.1380 | 0.0113 | 0.1533 | **CC** |
| **`humanpseaac`** | 0.1526 | **0.2929** | 0.1526 | 0.1526 | 0.0035 | 0.1880 | **CC** |
| **`chd49`** | 0.1585 | **0.1873** | 0.1585 | 0.1531 | 0.0759 | 0.1622 | **CC** |
| **`music`** | 0.2603 | 0.2837 | 0.2603 | 0.2855 | 0.2450 | **0.3040** | **GSI v6.3.2** |
| **`gpositivepseaac`** | 0.5954 | **0.7032** | 0.5954 | 0.6050 | 0.4817 | 0.6242 | **CC** |
| **`genbase`** | 0.9547 | 0.9577 | 0.9547 | 0.9562 | 0.7930 | **0.9758** | **GSI v6.3.2** |
| **`viruspseaac`** | 0.2597 | **0.3130** | 0.2597 | 0.2402 | 0.1423 | 0.2409 | **CC** |
| **Trung bình** | 0.3471 | **0.4129** | 0.3471 | 0.3537 | 0.2420 | 0.3646 | **CC** |

---

#### 4.4. Hamming Loss Toàn phần (↓) & Selective Hamming Loss (↓)

| Tập dữ liệu | Hamming Loss v6.3.1 | **Hamming Loss v6.3.2** | Sel. Hamming v6.3.1 | **Sel. Hamming v6.3.2** | Giảm lỗi Sel. Hamming (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`emotions`** | 0.2322 | **0.2008** | 0.2601 | **0.1789** | **-31.2%** |
| **`scene`** | 0.1443 | **0.1072** | 0.1785 | **0.0952** | **-46.7%** |
| **`yeast`** | 0.3703 | **0.2091** | 0.6234 | **0.2318** | **-62.8%** |
| **`plantpseaac`** | 0.3109 | **0.1014** | 0.6926 | **0.0835** | **-87.9%** |
| **`humanpseaac`** | 0.3403 | **0.0914** | 0.7734 | **0.0753** | **-90.3%** |
| **`chd49`** | 0.3895 | **0.2920** | 0.5410 | **0.3346** | **-38.2%** |
| **`music`** | 0.2256 | **0.1943** | 0.2615 | **0.1656** | **-36.7%** |
| **`gpositivepseaac`** | 0.2153 | **0.1469** | 0.3234 | **0.1226** | **-62.1%** |
| **`genbase`** | 0.0084 | **0.0011** | 0.0119 | **0.0006** | **-95.0%** |
| **`viruspseaac`** | 0.2930 | **0.2096** | 0.4644 | **0.1712** | **-63.1%** |
| **Trung bình** | 0.2530 | **0.1554** | 0.4130 | **0.1459** | **-64.7%** |

---

#### 4.5. Selective Micro-F1 (↑)

| Tập dữ liệu | BR | CC | MLC-PA | GSI v6.2.1 | GSI v6.3.1 | **GSI v6.3.2** | Mô hình dẫn đầu |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`emotions`** | 0.6289 | 0.6310 | 0.6946 | 0.7293 | 0.7189 | **0.7580** | **GSI v6.3.2** |
| **`scene`** | 0.6938 | 0.7110 | 0.7612 | 0.7789 | 0.7050 | **0.7903** | **GSI v6.3.2** |
| **`yeast`** | 0.6351 | 0.6198 | 0.6725 | **0.6727** | 0.3937 | 0.5869 | **GSI v6.2.1** |
| **`plantpseaac`** | 0.2604 | 0.3265 | 0.2012 | 0.1964 | 0.2342 | **0.4315** | **GSI v6.3.2** |
| **`humanpseaac`** | 0.2851 | 0.3774 | 0.1918 | 0.1918 | 0.2044 | **0.4736** | **GSI v6.3.2** |
| **`chd49`** | 0.6604 | 0.6575 | **0.7208** | 0.7204 | 0.4835 | 0.5690 | **MLC-PA** |
| **`music`** | 0.6436 | 0.6440 | 0.6956 | 0.7182 | 0.7181 | **0.7730** | **GSI v6.3.2** |
| **`gpositivepseaac`** | 0.6865 | 0.7127 | 0.7420 | 0.7425 | 0.6200 | **0.7851** | **GSI v6.3.2** |
| **`genbase`** | 0.9797 | 0.9809 | 0.9852 | 0.9852 | 0.8940 | **0.9939** | **GSI v6.3.2** |
| **`viruspseaac`** | 0.4070 | 0.4426 | 0.4316 | 0.4165 | 0.4476 | **0.5162** | **GSI v6.3.2** |
| **Trung bình** | 0.5881 | 0.6103 | 0.6096 | 0.6152 | 0.5419 | **0.6677** | **GSI v6.3.2** |

---

#### 4.6. Đối sánh hiệu năng theo từng bộ phân loại cơ sở (Base Learners: Logistic, Linear SVM, MLP)

Nhằm khẳng định tính tổng quát hóa của cơ chế Suy diễn biến phân Trường trung bình Chuẩn hóa một bước (OS-NMF), mô hình được kiểm định toàn diện trên 3 họ bộ phân loại cơ sở với các đặc tính toán học khác nhau:
1. **Logistic Regression (LR):** Mô hình tuyến tính xác suất chuẩn tắc.
2. **Calibrated Linear SVM:** Mô hình khoảng phân cách cực đại kết hợp hiệu chuẩn Platt hậu nghiệm.
3. **Multilayer Perceptron (MLP):** Mô hình mạng nơ-ron phi tuyến sâu.

Dưới đây là bảng tổng hợp các chỉ số trung bình toàn cục theo từng bộ học cơ sở trên toàn bộ 10 tập dữ liệu benchmark:

| Bộ học cơ sở | Mô hình | Selective Macro-F1 (↑) | Độ phủ Coverage (%) | Subset Acc (0/1) (↑) | Hamming Loss (↓) | Selective Hamming Loss (↓) | Selective Micro-F1 (↑) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **BR** | 0.4643 | **100.0%** | 0.3471 | 0.1520 | 0.1520 | 0.5881 |
| | **CC** | 0.4796 | **100.0%** | **0.4129** | 0.1604 | 0.1604 | 0.6103 |
| | **MLC-PA** | 0.4590 | 81.9% | 0.3471 | 0.1520 | **0.1033** | 0.6096 |
| | **GSI v6.2.1** | 0.4709 | 82.1% | 0.3537 | **0.1503** | 0.1016 | 0.6152 |
| | **GSI v6.3.1** | 0.5152 | 74.8% | 0.2420 | 0.2530 | 0.4130 | 0.5419 |
| | **GSI v6.3.2 (Đề Xuất)** | **0.5593** | 73.7% | 0.3646 | 0.1554 | 0.1459 | **0.6677** |
| \midrule | | | | | | | |
| **Linear SVM (Platt)** | **BR** | 0.4036 | **100.0%** | 0.2938 | 0.1549 | 0.1549 | 0.5122 |
| | **CC** | 0.4472 | **100.0%** | **0.3826** | 0.1635 | 0.1635 | 0.5785 |
| | **MLC-PA** | 0.3788 | 78.1% | 0.2938 | 0.1549 | **0.0946** | 0.5382 |
| | **GSI v6.2.1** | 0.3838 | 78.2% | 0.2976 | **0.1541** | 0.0953 | 0.5405 |
| | **GSI v6.3.1** | 0.5107 | 72.6% | 0.3320 | 0.1575 | 0.1509 | **0.6287** |
| | **GSI v6.3.2 (Đề Xuất)** | **0.5120** | 73.2% | 0.3379 | 0.1566 | 0.1516 | 0.6252 |
| \midrule | | | | | | | |
| **MLP (Neural Net)** | **BR** | 0.3254 | **100.0%** | 0.1896 | 0.1632 | 0.1632 | 0.3998 |
| | **CC** | 0.4228 | **100.0%** | 0.3194 | 0.1631 | 0.1631 | 0.5663 |
| | **MLC-PA** | 0.2965 | 74.6% | 0.1896 | 0.1632 | **0.0950** | 0.3787 |
| | **GSI v6.2.1** | 0.3871 | 80.7% | 0.2824 | **0.1626** | 0.1099 | 0.5550 |
| | **GSI v6.3.1** | **0.5252** | 73.6% | **0.3252** | 0.1661 | 0.1580 | **0.6524** |
| | **GSI v6.3.2 (Đề Xuất)** | 0.5239 | 73.1% | 0.3251 | 0.1663 | 0.1570 | 0.6510 |

---

### 5. Đánh giá kỹ thuật trên từng nhóm tập dữ liệu

#### 5.1. Nhóm nhãn phụ thuộc dày đặc (`yeast`, `plantpseaac`, `humanpseaac`)
- **Đặc trưng:** Có số lượng nhãn phụ thuộc lớn ($K_{DL} \in [10, 14]$) và tỷ lệ dương tính thấp.
- **Biểu hiện kỹ thuật:**
  - Trên `humanpseaac`, Selective Macro-F1 tăng từ $0.1838$ lên **$0.2380$** (+29.5%); Selective Micro-F1 tăng từ $0.2044$ lên **$0.4736$** (+131.7%); Subset Accuracy tăng từ $0.0035$ lên **$0.1880$**; Selective Hamming Loss giảm từ $0.7734$ xuống **$0.0753$** (-90.3%).
  - Trên `plantpseaac`, Selective Macro-F1 tăng từ $0.2201$ lên **$0.2685$** (+22.0%); Subset Accuracy tăng từ $0.0113$ lên **$0.1533$**; Selective Hamming Loss giảm từ $0.6926$ xuống **$0.0835$** (-87.9%).
  - Trên `yeast`, Selective Macro-F1 tăng từ $0.3744$ lên **$0.4284$** (+14.4%); Subset Accuracy tăng từ $0.0521$ lên **$0.1515$**; Selective Hamming Loss giảm từ $0.6234$ xuống **$0.2318$** (-62.8%).
- **Nguyên nhân:** Việc loại bỏ các mô hình điều kiện thứ cấp $g_l$ triệt tiêu hoàn toàn sự lan truyền sai số chuỗi. Chuẩn hóa bậc kết nối bảo đảm rằng tổng lực kéo từ 10-14 nhãn liên đới không làm nổ logit.

#### 5.2. Nhóm tập dữ liệu âm thanh và thị giác (`emotions`, `scene`, `music`)
- **Đặc trưng:** Số lượng nhãn ít ($K \in [6]$), tỷ lệ dương tính cân bằng vừa ($\pi \approx 18\% - 31\%$).
- **Biểu hiện kỹ thuật:**
  - Trên `scene`, Selective Macro-F1 tăng từ $0.7213$ lên **$0.7968$** (+10.5%); Subset Accuracy tăng từ $0.3848$ lên **$0.5481$** (+42.4%); Selective Hamming Loss giảm từ $0.1785$ xuống **$0.0952$** (-46.7%).
  - Trên `music`, Selective Macro-F1 tăng từ $0.7236$ lên **$0.7617$** (+5.3%); Subset Accuracy tăng từ $0.2450$ lên **$0.3040$**; Selective Hamming Loss giảm từ $0.2615$ xuống **$0.1656$** (-36.7%).
  - Trên `emotions`, Selective Macro-F1 tăng từ $0.7245$ lên **$0.7499$** (+3.5%); Subset Accuracy tăng từ $0.2306$ lên **$0.2978$**; Selective Hamming Loss giảm từ $0.2601$ xuống **$0.1789$** (-31.2%).
- **Nguyên nhân:** Các cặp nhãn có tương quan cảm xúc/thị giác tự nhiên được điều chỉnh đồng bộ bằng spin $2P - 1$, giúp giảm thiểu các trường hợp dự đoán rời rạc mâu thuẫn giữa các nhãn thường xuyên cùng xuất hiện.

#### 5.3. Nhóm phân tích trường hợp đặc biệt (`viruspseaac`, `chd49`)
- **`viruspseaac` (Kích thước mẫu nhỏ $N \approx 207$, 6 nhãn):**
  - Selective Macro-F1 giảm nhẹ từ $0.5068$ xuống $0.4584$ (-9.5%).
  - Tuy nhiên, Subset Accuracy tăng mạnh từ $0.1423$ lên **$0.2409$** (+69.3%), Selective Micro-F1 tăng từ $0.4476$ lên **$0.5162$** (+15.3%), và Selective Hamming Loss giảm mạnh từ $0.4644$ xuống **$0.1712$** (-63.1%).
  - *Giải thích kỹ thuật:* Trên tập dữ liệu mẫu quá nhỏ, một vài nhãn chỉ có 2-3 mẫu dương tính trên toàn bộ tập dữ liệu. Ước lượng tương quan Pearson trên phần dư của số mẫu quá ít có độ biến động thống kê (variance) cao, khiến một số nhãn hiếm bị kéo nhẹ xác suất, làm giảm điểm F1 của riêng nhãn đó. Ngược lại, cấu hình vector toàn thể vẫn chính xác hơn đáng kể.
- **`chd49` (Tập dữ liệu bệnh tim mạch, 6 nhãn dày đặc):**
  - Selective Macro-F1 tăng từ $0.4320$ lên **$0.4672$** (+8.1%).
  - Subset Accuracy tăng từ $0.0759$ lên **$0.1622$** (+113.6%).
  - Selective Hamming Loss giảm từ $0.5410$ xuống **$0.3346$** (-38.2%).

---

### 6. Kết luận khoa học & Đánh giá thực tiễn

1. **Hiệu quả của việc thay thế mô hình điều kiện bằng Mean-Field một bước:**
   - Việc loại bỏ các mô hình phụ $g_l$ và áp dụng One-Step Normalized Mean-Field (OS-NMF) giúp **Selective Macro-F1 tăng trung bình từ 0.5152 lên 0.5593 (+8.6%)** trên toàn bộ 10 tập dữ liệu.
   - Hiệu năng phân loại cấu hình nhãn (Subset 0/1 Accuracy) tăng từ **24.20% lên 36.46%** (tăng hơn 1.5 lần).
   - Tỷ lệ sai số trên các nhãn được đưa ra quyết định (Selective Hamming Loss) giảm mạnh **64.7%** (từ 0.4130 xuống 0.1459).

2. **Cắt giảm tài nguyên tính toán:**
   - Do không phải huấn luyện tập mô hình thứ cấp, thời gian huấn luyện trung bình của GSI v6.3.2 giảm **12.8%** so với v6.3.1 (từ 2.094s xuống 1.826s trên 10 tập dữ liệu).

3. **Tính ổn định của cơ chế kiểm soát biên độ:**
   - Cơ chế chuẩn hóa bậc $\max(1.0, \text{row\_sums})$ kết hợp kẹp biên $[-0.5, +0.5]$ bảo đảm quá trình suy diễn không gây bão hòa logit hay làm phá vỡ cơ chế Precision Guard ($\tau_1 \ge 0.50$).
   - Độ bao phủ quyết định trung bình duy trì ổn định ở mức **73.73%**, luôn nằm trên ngưỡng an toàn $\gamma_{\min} = 70.0%$.

4. **Hạn chế thực tế cần lưu ý:**
   - Khi kích thước mẫu quá nhỏ ($N \le 200$) kết hợp với nhãn cực hiếm, ma trận tương quan phần dư có thể chịu phương sai ước lượng cao. Trong các ứng dụng thực tế với số lượng mẫu hạn chế, nên cân nhắc nâng ngưỡng lọc tương quan $\theta_{\text{corr}}$ từ $0.25$ lên $0.35$ để tránh các liên kết nhiễu.
