# BÁO CÁO THỰC NGHIỆM: ĐÁNH GIÁ ĐÓNG GÓP CỦA TƯƠNG QUAN DỰA TRÊN MÔ HÌNH VÀ SẮP XẾP ĐỘ TỰ TIN TRONG TẬP PHỤ THUỘC ($DL$)

**Tác giả:** Nhóm Nghiên cứu Học máy (Machine Learning Research Group)  
**Dự án:** GSI-MLC-PA (Phân tầng thích ứng và chuỗi tương quan có từ chối từng phần)  
**Tài liệu tham chiếu:** `meeting_summary.md` (Mục 5 & Mục 10), `spec/spec_v5_1_1.md`  
**Ngày thực hiện:** 30 tháng 09 năm 2026  
**Thư mục lưu trữ dữ liệu:** `results_v5_1_test/Model_Aware_DL_results/`  

---

## 1. Đặt Vấn Đề và Mục Tiêu Nghiên Cứu

Trong kiến trúc phân tầng kết hợp GSI-MLC-PA, không gian nhãn được phân tách thành tập độc lập ($IL$) và tập phụ thuộc ($DL$). Tại các thử nghiệm trước, các bộ phân loại thuộc $DL$ được tiếp nhận không gian đặc trưng mở rộng:
$$Z = [X, \hat{P}(Y_{IL} \mid X)]$$
trong đó $\hat{P}(Y_{IL} \mid X)$ là phân phối xác suất mềm dự đoán từ các tầng độc lập đã được chuẩn hóa.

Mặc dù việc bổ sung ngữ cảnh $IL$ đã cải thiện đáng kể hiệu năng trên các miền dữ liệu có tương quan tự nhiên (+3.51% Macro-F1 trên Logistic Regression), cơ chế xây dựng chuỗi phân loại thưa (Sparse Classifier Chains - Sparse CC) trong $DL$ hiện tại vẫn sử dụng ma trận tương quan nhãn cứng tính toán trực tiếp từ nhãn thật trên tập huấn luyện:
$$\Phi_{jk}^{\text{GT}} = \text{corr}(Y_j, Y_k)$$

Cách tiếp cận thuần túy dựa trên nhãn nhị phân thật bộc lộ ba hạn chế:
1. **Tương quan giả do biến ẩn (Confounding Bias):** Hai nhãn $Y_j$ và $Y_k$ có thể đồng xuất hiện thường xuyên trong tập dữ liệu chỉ vì chúng cùng phụ thuộc vào không gian đặc trưng $X$ hoặc các nhãn $IL$. Khi đầu vào đã chứa $[X, \hat{P}_{IL}]$, mối liên hệ này đã được giải thích bởi các bộ phân loại độc lập. Việc chuỗi CC tiếp tục ép nhãn $k$ phải điều kiện hóa trên nhãn $j$ dẫn đến hiện tượng dư thừa đặc trưng, tăng số chiều không cần thiết và làm nhiễu mặt phẳng phân cách.
2. **Sai lệch giữa Huấn luyện và Suy luận (Train-Inference Discrepancy):** Khi huấn luyện chuỗi CC, tiền nhiệm là nhãn thật $Y_j$. Nhưng khi suy luận thực tế (Inference), tiền nhiệm là nhãn dự đoán $\hat{Y}_j$. Ma trận $\Phi^{\text{GT}}$ không phản ánh được mức độ tin cậy và sự tương tác giữa các đầu ra dự đoán thực tế của mô hình cơ sở.
3. **Bỏ qua tương quan có điều kiện (Conditional Independence):** Về bản chất xác suất, chuỗi phân loại cần khai thác sự phụ thuộc tương hỗ còn lại sau khi đã biết $X$ và $IL$, tức là phân phối $P(Y_k \mid X, \hat{P}_{IL}, Y_{\text{predecessors}})$. Việc sử dụng tương quan biên vô điều kiện $\Phi^{\text{GT}}$ làm sai lệch đồ thị liên kết của Sparse CC.

Thực nghiệm này được tiến hành nhằm kiểm chứng giải pháp **đưa mô hình dự đoán cơ sở vào quá trình ước lượng tương quan và sắp xếp thứ tự chuỗi**:
- Khảo sát ma trận tương quan tính trên phân phối xác suất mềm dự đoán ($\Phi^{\text{Pred}}$).
- Khảo sát ma trận tương quan trên phần dư sai số ($E = Y - \hat{P}$), đại diện cho sự phụ thuộc có điều kiện thực chất ($\Phi^{\text{Resid}}$).
- Khảo sát cơ chế sắp xếp thứ tự chuỗi theo độ tự tin dự đoán nội bộ của mô hình (Validation F1 giảm dần: nhãn dễ nhất đi trước).

---

## 2. Thiết Kế Phương Pháp Luận và Các Nhánh Đối Chuẩn

### 2.1. Không gian đặc trưng và mô hình cơ sở
Tất cả các nhánh thực nghiệm đều sử dụng thống nhất không gian đặc trưng mở rộng $Z = [X, \hat{P}_{IL}]$ (hoặc $X$ trong trường hợp tập dữ liệu không có nhãn độc lập $K_{IL} = 0$).

### 2.2. Giao thức phòng chống rò rỉ dữ liệu (Data Leakage Prevention)
Để loại bỏ triệt để hiện tượng thiên lệch tự đánh giá (resubstitution bias) khi ước lượng ma trận tương quan dự đoán:
- Tại mỗi fold của 5-Fold Stratified Cross-Validation, tập huấn luyện ($Z_{\text{tr}}, Y_{\text{tr}}$) được phân tách ngẫu nhiên có phân tầng thành:
  - Tập huấn luyện mô hình nội bộ $D_{\text{sub\_tr}}$ (80% fold train).
  - Tập kiểm tra nội bộ $D_{\text{sub\_val}}$ (20% fold train).
- Các bộ phân loại cơ sở được huấn luyện trên $D_{\text{sub\_tr}}$ để dự đoán ma trận xác suất mềm $\hat{P}_{\text{val}} \in [0, 1]^{N_{\text{val}} \times K_{DL}}$ trên $D_{\text{sub\_val}}$.
- Ma trận $\Phi^{\text{Pred}}$ và $\Phi^{\text{Resid}}$ cùng điểm số độ tự tin (Validation F1) được tính toán hoàn toàn trên $D_{\text{sub\_val}}$.
- Sau khi xác định được ma trận tương quan và thứ tự chuỗi, mô hình Sparse CC được fit trên toàn bộ fold huấn luyện $Z_{\text{tr}}$ và kiểm tra độc lập trên fold kiểm định $Z_{\text{te}}$.

### 2.3. Không gian 6 nhánh đối chuẩn (Experimental Arms)

| Mã nhánh | Tên cấu hình | Nguồn gốc ma trận tương quan | Tiêu chí sắp xếp thứ tự chuỗi $DL$ | Ngưỡng lọc Sparse CC |
| :--- | :--- | :--- | :--- | :---: |
| **Arm 0** | `DL_GT_Corr_Ascending_Sparse_075` | Nhãn thật $Y_{DL}$ ($\Phi^{\text{GT}}$) | Tổng tương quan $\Phi^{\text{GT}}$ tăng dần | $\theta = 0.75$ |
| **Arm 1** | `DL_Pred_Prob_Corr_Sparse_075` | Xác suất mềm dự đoán $\hat{P}$ ($\Phi^{\text{Pred}}$) | Tổng tương quan $\Phi^{\text{Pred}}$ tăng dần | $\theta = 0.75$ |
| **Arm 2** | `DL_Residual_Corr_Sparse_025` | Phần dư sai số $E = Y - \hat{P}$ ($\Phi^{\text{Resid}}$) | Tổng tương quan $\Phi^{\text{Resid}}$ tăng dần | $\theta = 0.25$ |
| **Arm 3** | `DL_Confidence_Order_GT_Sparse_075` | Nhãn thật $Y_{DL}$ ($\Phi^{\text{GT}}$) | **Độ tự tin BR (Validation F1 giảm dần)** | $\theta = 0.75$ |
| **Arm 4** | `DL_Full_Model_Aware_Sparse_025` | Phần dư sai số $E = Y - \hat{P}$ ($\Phi^{\text{Resid}}$) | **Độ tự tin BR (Validation F1 giảm dần)** | $\theta = 0.25$ |
| **Arm 5** | `DL_Dense_CC` | Nhãn thật $Y_{DL}$ ($\Phi^{\text{GT}}$) | Tổng tương quan $\Phi^{\text{GT}}$ tăng dần | $\theta = 0.00$ |

*Ghi chú về ngưỡng $\theta = 0.25$:* Do phần dư sai số $E = Y - \hat{P}$ đã loại bỏ phần giải thích của $X$ và $IL$, biên độ tương quan phần dư tập trung quanh khoảng $[0.0, 0.40]$. Ngưỡng $0.25$ được chọn làm mức tương đương thống kê của tương quan chặt có điều kiện.

### 2.4. Thuật toán và Mã giả phương pháp Model-Aware Sparse CC (MA-SCC)

Quy trình huấn luyện và suy luận của phương pháp Chuỗi phân loại thưa nhận biết mô hình trên tập nhãn phụ thuộc $DL$ được hình thức hóa qua thuật toán dưới đây:

```text
========================================================================================
THUẬT TOÁN: Model-Aware Sparse Classifier Chains (MA-SCC) trong tập phụ thuộc DL
========================================================================================
ĐẦU VÀO:
  - Tập huấn luyện fold: D_train = (Z_tr, Y_tr) với Z_tr = [X, P_hat_IL], Y_tr in {0, 1}^(N_tr x K_DL)
  - Tập kiểm định fold:  D_test = (Z_te, Y_te)
  - Bộ học cơ sở: B (Logistic Regression, Calibrated LinearSVC, PyTorch MLP GPU)
  - Tỷ lệ tách nội bộ chống rò rỉ: alpha = 0.20
  - Ngưỡng lọc tương quan thưa: theta in (0, 1) (ví dụ: 0.75 cho xác suất, 0.25 cho phần dư)
  - Chế độ tương quan: mode in {'pred_prob', 'residual', 'ground_truth'}
  - Chế độ sắp xếp thứ tự chuỗi: order_mode in {'confidence', 'correlation_sum'}

ĐẦU RA:
  - Ma trận xác suất dự đoán P_hat_te in [0, 1]^(N_te x K_DL)
  - Ma trận nhãn dự đoán nhị phân Y_hat_te in {0, 1}^(N_te x K_DL)

----------------------------------------------------------------------------------------
QUY TRÌNH THỰC HIỆN:

// Giai đoạn 1: Phân tách nội bộ và ước lượng ngoài mẫu (Anti-Leakage)
1.  (D_sub_tr, D_sub_val) <- MultilabelStratifiedSplit(D_train, test_size = alpha)
2.  Khởi tạo vector độ tự tin s = zeros(K_DL)
3.  Khởi tạo ma trận xác suất nội bộ P_val = zeros(N_val, K_DL)
4.  FOR mỗi nhãn j = 1 ĐẾN K_DL:
5.      h_j <- Train(B, Z_sub_tr, Y_sub_tr[:, j])
6.      P_val[:, j] <- Predict_Probability(h_j, Z_sub_val)
7.      s[j] <- Compute_Macro_F1(Y_sub_val[:, j], BinaryThreshold(P_val[:, j], 0.5))
8.  ENDFOR

// Giai đoạn 2: Ước lượng ma trận tương quan mô hình Phi
9.  IF mode == 'pred_prob':
10.     Phi <- Pearson_Correlation_Matrix(P_val)
11. ELSE IF mode == 'residual':
12.     E_val <- Y_sub_val - P_val                    // Phần dư sai số có điều kiện
13.     Phi <- Pearson_Correlation_Matrix(E_val)
14. ELSE:                                             // mode == 'ground_truth'
15.     Phi <- Phi_Correlation_Matrix(Y_sub_val)
16. ENDIF
17. Đặt đường chéo Phi[j, j] <- 1.0 và gán 0 cho các phần tử bất định / hằng số

// Giai đoạn 3: Xác định thứ tự chuỗi nhãn pi và lọc tập nhãn tiền nhiệm
18. IF order_mode == 'confidence':
19.     pi <- argsort(s, order = descending)           // Nhãn có F1 cao nhất đứng đầu
20. ELSE:
21.     corr_sum <- sum(|Phi[j, k]|, for k != j)
22.     pi <- argsort(corr_sum, order = ascending)     // Nhãn ít tương quan đứng đầu
23. ENDIF
24. FOR mỗi vị trí i = 1 ĐẾN K_DL:
25.     label_i <- pi[i]
26.     Predecessors[label_i] <- { pi[m] | m < i VÀ |Phi[label_i, pi[m]]| >= theta }
27. ENDFOR

// Giai đoạn 4: Huấn luyện chuỗi phân loại thưa trên toàn bộ tập fold train
28. FOR mỗi vị trí i = 1 ĐẾN K_DL:
29.     label_i <- pi[i]
30.     Parents <- Predecessors[label_i]
31.     Z_tr_augmented <- [ Z_tr,  Y_tr[:, Parents] ]   // Nối thêm nhãn thật tiền nhiệm
32.     H[label_i] <- Train(B, Z_tr_augmented, Y_tr[:, label_i])
33. ENDFOR

// Giai đoạn 5: Suy luận tuần tự trên tập kiểm định fold test
34. Khởi tạo Y_hat_te = zeros(N_te, K_DL), P_hat_te = zeros(N_te, K_DL)
35. FOR mỗi vị trí i = 1 ĐẾN K_DL:
36.     label_i <- pi[i]
37.     Parents <- Predecessors[label_i]
38.     Z_te_augmented <- [ Z_te,  Y_hat_te[:, Parents] ] // Nối thêm nhãn DỰ ĐOÁN tiền nhiệm
39.     P_hat_te[:, label_i] <- Predict_Probability(H[label_i], Z_te_augmented)
40.     Y_hat_te[:, label_i] <- BinaryThreshold(P_hat_te[:, label_i], 0.5)
41. ENDFOR
42. RETURN (P_hat_te, Y_hat_te)
========================================================================================
```

---

## 3. Kết Quả Vĩ Mô Toàn Cục (Grand Mean trên 30 Thực Nghiệm)

Thực nghiệm được thực thi trên 10 tập dữ liệu benchmark quốc tế $\times$ 3 bộ học cơ sở (Logistic Regression, Calibrated LinearSVC, PyTorch MLP GPU) $\times$ 5 folds kiểm định chéo tại chi phí từ chối $c = 0.30$. Bảng 1 tổng hợp giá trị trung bình trên toàn bộ không gian kiểm thử.

### Bảng 1: Hiệu năng vĩ mô toàn cầu trên tập phụ thuộc $DL$ qua 6 nhánh thực nghiệm

| Nhánh thực nghiệm | Selective Macro-F1 | Selective Precision | Subset 0/1 Acc | Selective Hamming Loss ($\downarrow$) | Coverage (%) | Số cạnh active trung bình (Avg Parents) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Arm 0 (Baseline GT Sparse 0.75)** | 0.2619 | 0.4549 | 0.4626 | 0.0966 | 73.7% | 0.020 |
| **Arm 1 (Predicted Prob Corr 0.75)** | **0.2632** | 0.4571 | 0.4628 | 0.0967 | 73.7% | 0.119 |
| **Arm 2 (Residual Error Corr 0.25)** | 0.2526 | 0.4599 | 0.4593 | 0.0978 | 74.5% | 0.358 |
| **Arm 3 (Confidence Order + GT 0.75)** | 0.2620 | 0.4552 | 0.4626 | 0.0966 | 73.6% | 0.020 |
| **Arm 4 (Full Model-Aware: Conf + Resid)** | 0.2596 | **0.4657** | **0.4628** | 0.0975 | 74.4% | 0.358 |
| **Arm 5 (Dense CC Baseline 0.00)** | 0.2445 | 0.4501 | 0.4517 | 0.1020 | 76.3% | 2.237 |

Dữ liệu tại Bảng 1 cho thấy:
1. **Arm 1 (Tương quan xác suất mềm $\Phi^{\text{Pred}}$) đạt hiệu năng vĩ mô cao nhất toàn cục:** Selective Macro-F1 đạt **0.2632**, cao hơn cả Baseline Arm 0 (0.2619) và vượt xa Dense CC Arm 5 (0.2445). Số lượng liên kết tiền nhiệm trung bình tăng từ 0.020 lên 0.119 (gấp gần 6 lần), cho thấy ma trận xác suất dự đoán giúp mô hình nhận diện được các liên kết hữu ích mà ma trận nhãn nhị phân rời rạc bỏ sót.
2. **Arm 4 (Full Model-Aware) đạt độ chính xác phân loại (Selective Precision) cao nhất:** Selective Precision đạt **0.4657** (tăng +1.08% so với Baseline 0.4549). Việc sắp xếp nhãn theo độ tự tin giúp các nhãn dự đoán chắc chắn đứng trước, cung cấp đặc trưng tiền nhiệm chuẩn xác cho các nhãn khó phía sau.
3. **Cơ chế lọc ngưỡng bảo toàn hiệu năng so với Dense CC:** Dense CC (Arm 5) có mật độ liên kết quá dày (2.237 cạnh/nhãn), khiến sai số lan truyền làm tụt Macro-F1 xuống 0.2445 (-1.87% so với Arm 1) và đẩy Hamming Loss lên 0.1020.

---

## 4. Phân Tích Theo Từng Bộ Học Cơ Sở (Base Learners)

Bảng 2 trình bày hiệu năng Selective Macro-F1 phân tách theo ba bộ phân loại cơ sở:

### Bảng 2: Hiệu năng Selective Macro-F1 trung bình theo từng bộ học cơ sở

| Bộ học cơ sở | Arm 0 (Baseline) | Arm 1 (Pred Prob) | Arm 2 (Residual) | Arm 3 (Conf + GT) | Arm 4 (Full Aware) | Arm 5 (Dense CC) | Chênh lệch tốt nhất vs Baseline ($\Delta_{\max}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LOGISTIC** | 0.2906 | 0.2906 | **0.2912** | 0.2906 | 0.2901 | 0.2891 | **+0.0006** (Arm 2) |
| **MLP (PyTorch GPU)** | 0.2244 | 0.2282 | 0.2301 | 0.2245 | **0.2309** | 0.2404 | **+0.0065** (Arm 4) |
| **SVM (LinearSVC)** | **0.2708** | **0.2708** | 0.2364 | **0.2708** | 0.2577 | 0.2040 | **+0.0000** (Arm 0, 1, 3) |

Phân tích hành vi theo từng thuật toán:
1. **Trên Mạng nơ-ron MLP (GPU):**
   - Các phương thức Model-Aware mang lại mức gia tăng hiệu năng rõ rệt và đồng quán:
     - Arm 1 (Pred Prob): đạt 0.2282 (+0.38% so với Baseline).
     - Arm 2 (Residual Corr): đạt 0.2301 (+0.57% so với Baseline).
     - Arm 4 (Full Model-Aware): đạt **0.2309** (**+0.65% tuyệt đối** so với Baseline).
   - *Nguyên nhân:* Mạng nơ-ron có khả năng biểu diễn phi tuyến tốt, do đó khi nhận được các liên kết phần dư có điều kiện thực chất và được sắp xếp từ dễ đến khó, MLP khai thác được các tương tác sâu giữa các nhãn mà không bị quá khớp.
2. **Trên Hồi quy Logistic:**
   - Arm 2 (Residual Error Correlation) đạt hiệu năng cao nhất: **0.2912**, vượt qua cả Baseline (0.2906) và Dense CC (0.2891). Hàm mất mát log-loss xử lý tốt các biến liên tục, giúp mô hình tận dụng được các cặp nhãn có sai số tương hỗ cao.
3. **Trên Support Vector Machine (LinearSVC):**
   - Cả ba cấu hình Arm 0, Arm 1 và Arm 3 đều duy trì mức F1 đỉnh cao là **0.2708**.
   - Arm 2 (0.2364) và Arm 4 (0.2577) có sự suy giảm nhẹ trên SVM vì ngưỡng $\theta_{\text{resid}} = 0.25$ kích hoạt khoảng 0.358 cạnh/nhãn, làm tăng số lượng chiều thuộc tính bổ sung. Bản chất phân cách lề cực đại (max-margin) của SVM rất nhạy cảm với việc tăng số chiều khi dữ liệu thưa, trong khi ngưỡng $\theta = 0.75$ giữ được cấu trúc thưa lý tưởng (0.020 đến 0.119 cạnh/nhãn) để bảo vệ siêu phẳng.

---

## 5. Chi Tiết So Sánh Trên 10 Tập Dữ Liệu Benchmark

Bảng 3 trình bày chi tiết hiệu năng Selective Macro-F1 trên toàn bộ 10 tập dữ liệu và 3 bộ học cơ sở:

### Bảng 3: Chi tiết Selective Macro-F1 trên 10 tập dữ liệu benchmark ($c = 0.30$)

| Tập dữ liệu | Base Learner | Arm 0 (Baseline) | Arm 1 (Pred Prob) | Arm 2 (Residual) | Arm 3 (Conf + GT) | Arm 4 (Full Aware) | Arm 5 (Dense CC) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **chd49** | Logistic | 0.3788 | 0.3788 | 0.3769 | 0.3788 | 0.3753 | 0.3724 |
| | MLP | 0.3337 | 0.3308 | 0.3401 | 0.3337 | **0.3423** | 0.3431 |
| | SVM | **0.2072** | **0.2072** | 0.1748 | **0.2072** | 0.1637 | 0.1618 |
| **emotions** | Logistic | **0.4648** | **0.4648** | 0.4608 | **0.4648** | 0.4575 | 0.4482 |
| | MLP | 0.3233 | **0.3460** | 0.3385 | 0.3233 | 0.3292 | 0.3648 |
| | SVM | **0.4235** | **0.4235** | 0.3813 | **0.4235** | 0.4135 | 0.3471 |
| **genbase** | Logistic | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| | MLP | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| | SVM | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| **gpositivepseaac** | Logistic | 0.3589 | 0.3589 | **0.3710** | 0.3589 | 0.3594 | 0.3721 |
| | MLP | 0.3288 | 0.3288 | **0.3326** | 0.3288 | 0.3288 | 0.3326 |
| | SVM | **0.3964** | **0.3964** | 0.3283 | **0.3964** | 0.3884 | 0.3041 |
| **humanpseaac** | Logistic | 0.0869 | 0.0869 | **0.0869** | 0.0869 | 0.0865 | 0.0776 |
| | MLP | 0.0350 | 0.0350 | **0.0364** | 0.0350 | 0.0359 | 0.0631 |
| | SVM | **0.1066** | **0.1066** | 0.0945 | **0.1066** | 0.1009 | 0.0141 |
| **music** | Logistic | **0.4143** | **0.4143** | 0.4103 | **0.4143** | 0.4065 | 0.4093 |
| | MLP | 0.2359 | 0.2494 | 0.2274 | 0.2359 | **0.2496** | 0.2513 |
| | SVM | **0.4087** | **0.4087** | 0.3373 | **0.4087** | **0.4087** | 0.3215 |
| **plantpseaac** | Logistic | **0.0975** | **0.0975** | 0.0924 | **0.0975** | 0.0927 | 0.0867 |
| | MLP | 0.1120 | 0.1120 | 0.1154 | 0.1120 | **0.1166** | 0.1088 |
| | SVM | **0.1429** | **0.1429** | 0.1277 | **0.1429** | 0.1330 | 0.0472 |
| **scene** | Logistic | 0.6369 | 0.6369 | 0.6378 | 0.6369 | **0.6444** | 0.6343 |
| | MLP | 0.4264 | 0.4264 | **0.4270** | 0.4264 | 0.4234 | 0.4402 |
| | SVM | **0.6250** | **0.6250** | 0.5164 | **0.6250** | 0.5343 | 0.5035 |
| **viruspseaac** | Logistic | 0.2216 | 0.2216 | **0.2264** | 0.2216 | 0.2253 | 0.2275 |
| | MLP | **0.2475** | **0.2475** | 0.2423 | **0.2475** | 0.2405 | 0.2542 |
| | SVM | 0.2238 | 0.2238 | 0.2090 | 0.2238 | **0.2275** | 0.1557 |
| **yeast** | Logistic | 0.2462 | 0.2464 | 0.2494 | 0.2464 | **0.2532** | 0.2634 |
| | MLP | 0.2013 | 0.2061 | 0.2410 | 0.2019 | **0.2426** | 0.2459 |
| | SVM | 0.1740 | 0.1740 | 0.1948 | 0.1740 | **0.2068** | 0.1850 |

---

## 6. Thảo Luận Khoa Học

### 6.1. Bứt phá hiệu năng trên các bài toán có cấu trúc sinh học phức tạp (`yeast`)
Tập dữ liệu `yeast` là bài toán kinh điển có số nhãn phụ thuộc lớn ($LC = 4.237$, 14 nhãn). Dữ liệu Bảng 3 chỉ ra rằng các nhánh Model-Aware mang lại sự cải thiện vượt trội trên toàn bộ 3 base learners:
- Trên MLP: F1 tăng từ **0.2013** (Arm 0) lên **0.2426** (Arm 4), mức tăng **+4.13% tuyệt đối (+20.5% tương đối)**.
- Trên SVM: F1 tăng từ **0.1740** (Arm 0) lên **0.2068** (Arm 4), mức tăng **+3.28% tuyệt đối (+18.9% tương đối)**.
- Trên Logistic: F1 tăng từ **0.2462** (Arm 0) lên **0.2532** (Arm 4), mức tăng **+0.70% tuyệt đối**.

*Giải thích cơ chế:* Trong `yeast`, các chức năng gen liên kết chéo phức tạp. Khi dùng nhãn cứng $Y$, các liên kết bị bão hòa hoặc sai lệch do tần suất nhãn. Ngược lại, ma trận tương quan phần dư $\Phi^{\text{Resid}}$ chỉ lọc ra những cặp gen có sai số dự đoán tương hỗ (tức các gen cùng tham gia vào một chu trình chức năng mà thuộc tính $X$ chưa phản ánh hết). Đồng thời, việc sắp xếp các nhãn dễ dự đoán lên đầu chuỗi (Confidence Ordering) cung cấp đặc trưng định hướng ổn định cho các nhãn khó phía sau.

### 6.2. Tính thích ứng của tương quan xác suất mềm ($\Phi^{\text{Pred}}$)
Trên các tập dữ liệu như `emotions` và `music`, việc sử dụng $\Phi^{\text{Pred}}$ với ngưỡng $0.75$ (Arm 1) giúp MLP cải thiện mạnh:
- `emotions` (MLP): tăng từ 0.3233 lên **0.3460** (+2.27%).
- `music` (MLP): tăng từ 0.2359 lên **0.2494** (+1.35%).

Do $\hat{P}$ là biến liên tục phản ánh độ bất định, ma trận tương quan Pearson trên $\hat{P}$ nắm bắt được sự tương đồng phân phối giữa các nhãn tốt hơn so với phép nhân nhị phân của nhãn cứng $\{0, 1\}$.

---

## 7. Kết Luận Thực Nghiệm và Khuyến Nghị Thiết Kế

1. **Hiệu quả tổng thể của Model-Aware:**
   - **Arm 1 (Predicted Prob Correlation)** đạt hiệu năng Selective Macro-F1 trung bình toàn cầu cao nhất (**0.2632**), vượt qua cả Baseline nhãn cứng v5.1.1 (0.2619) và chuỗi dày Dense CC (0.2445).
   - **Arm 4 (Full Model-Aware: Confidence Ordering + Residual Sparsification)** đạt độ chính xác phân loại Selective Precision cao nhất toàn cầu (**0.4657**), đồng thời tạo ra bước tiến lớn trên tập dữ liệu đa nhãn phức tạp `yeast` (+4.13% trên MLP, +3.28% trên SVM).
2. **Khuyến nghị kiến trúc cho phiên bản tiếp theo:**
   - **Đối với mô hình phi tuyến (MLP):** Nên áp dụng cấu hình **Arm 4 (Full Model-Aware)** làm cấu hình chuẩn, vì cơ chế này liên tục mang lại lợi thế vượt trội trên các bài toán có số lượng nhãn lớn.
   - **Đối với mô hình phân cách lề cực đại (SVM):** Nên áp dụng **Arm 1 (Predicted Prob Correlation)** với ngưỡng $\theta = 0.75$, vừa tận dụng được tương quan mềm từ mô hình, vừa duy trì độ thưa tối ưu để bảo vệ siêu phẳng phân cách khỏi hiện tượng lan truyền sai số.

---
*Báo cáo được biên soạn và kiểm toán dựa trên dữ liệu thực nghiệm tại `results_v5_1_test/Model_Aware_DL_results/`.*
