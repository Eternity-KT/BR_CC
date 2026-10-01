# TÀI LIỆU ĐẶC TẢ KỸ THUẬT PHIÊN BẢN v6: MULTI-STAGE 5-FOLD CV PEELING PARTITIONING (GSI-MLC-PA v6)

**Tài liệu tham chiếu:** `meeting_summary.md`  
**Nhánh Git:** `v6`  
**Ngày lập đặc tả:** 01/10/2026  
**Trạng thái:** Triển khai thử nghiệm (Experimental)

---

## 1. Bối Cảnh và Mục Tiêu Cốt Lõi (Core Motivation)

Trong phiên bản v5.1, thuật toán Stratified Peeling phân tách nhãn độc lập ($IL$) và nhãn phụ thuộc ($DL$) dựa trên một tập validation nội bộ duy nhất (`validation_size=0.2`). Mặc dù mang lại cải thiện đáng kể so với phương pháp Greedy trước đó, việc sử dụng một phân chia validation tĩnh có một số hạn chế:
1. **Thiếu tính đại diện trên các tập dữ liệu nhỏ hoặc mất cân bằng cao:** Các nhãn hiếm có thể chỉ có 1-2 mẫu dương trong tập validation 20%, dẫn đến ước lượng F1 thiếu ổn định.
2. **Rủi ro rò rỉ hoặc thiên lệch thông tin khi tăng cường đặc trưng:** Trong v5.1, khi chuyển từ Tầng 1 sang Tầng 2, các xác suất $\hat{P}_{IL_1}$ được dự đoán trên toàn bộ tập train bằng mô hình huấn luyện trên chính tập train đó (in-sample prediction), tiềm ẩn nguy cơ quá khớp.

### Định Hướng Cải Tiến Cốt Lõi của v6 (`meeting_summary.md`):
- **Phân tách nhãn độc lập từng bước tuần tự:** Kiểm tra từng nhãn $y_1, y_2, \dots, y_K$ xem có thuộc tập độc lập ($IL$) hay không.
- **Học mô hình phân lớp cơ sở đa dạng:** Hỗ trợ cả 3 họ mô hình phân lớp phổ biến: Logistic Regression, Support Vector Machine (LinearSVC calibrated), và Multi-Layer Perceptron (MLP).
- **Kiểm định chéo 5-Fold Cross Validation:** Thay vì 1 split train/val, dùng 5-Fold CV cho từng nhãn.
- **Tính toán Selective-F1 trên dự đoán ngoài mẫu (Out-Of-Fold - OOF):** Nhãn $y_i$ có Selective-F1 $\ge 0.75$ (hoặc ngưỡng $\tau$ được cấu hình) sẽ được kết luận là độc lập và đưa vào tập $IL_1$.
- **Tăng cường đặc trưng Out-Of-Fold cho tầng kế tiếp:** Toàn bộ xác suất dự đoán OOF của $IL_1$ được bổ sung vào không gian đặc trưng $[X, \hat{P}^{\text{OOF}}_{IL_1}]$ cho tầng thứ 2. Do là OOF, các đặc trưng này hoàn toàn phi rò rỉ (leakage-free).
- **Lặp lại cho tập DL còn lại:** Tiếp tục 5-Fold CV trên không gian đặc trưng mới để tìm tập $IL_2$, tiếp tục cho đến khi đạt độ sâu tối đa hoặc không còn nhãn nào đạt ngưỡng.
- **Tập phụ thuộc dư thừa ($DL_{\text{residual}}$):** Các nhãn còn lại được sắp xếp theo thứ tự tương quan (Ascending Correlation Order) để đưa vào chuỗi phân loại CC.

---

## 2. Thuật Toán Cốt Lõi (Core Algorithm)

### Thuật toán 1: 5-Fold CV Multi-Stage Peeling Partitioning
```text
Input: 
  - Ma trận đặc trưng X kích thước (N, d)
  - Ma trận nhãn nhị phân Y kích thước (N, K)
  - Bộ học cơ sở base_learner ∈ {"logistic", "svm", "mlp"}
  - Ngưỡng đánh giá τ (mặc định = 0.75)
  - Chi phí từ chối c (mặc định = 0.30)
  - Số fold k_folds = 5
  - Độ sâu bóc tách tối đa max_depth = 3
  - Cơ chế giảm ngưỡng decaying_threshold (True/False, decay_step = 0.05)

Output:
  - Danh sách các tầng độc lập: IL = [IL_1, IL_2, ...]
  - Tập nhãn phụ thuộc dư thừa: DL_residual
  - Thứ tự thực thi chuỗi toàn phần: Execution_Order = [IL_1, IL_2, ..., DL_residual]
  - Nhật ký kiểm toán chẩn đoán (audit diagnostics) cho từng tầng và từng nhãn

Các bước thực hiện:
1. Khởi tạo:
   - Tập ứng viên phụ thuộc: Candidate_DL = {0, 1, ..., K - 1}
   - Không gian đặc trưng tầng hiện tại: X^(1) = X
   - IL_layers = []
   - stage = 1

2. Lặp qua từng tầng stage = 1, 2, ..., max_depth:
   a. Xác định ngưỡng tầng hiện tại:
      τ_stage = max(0.50, τ - (stage - 1) * decay_step) nếu decaying_threshold else τ
   
   b. Khởi tạo danh sách thăng hạng tầng: Promoted_stage = []
   c. Lưu trữ xác suất OOF: OOF_Probs_stage = {}
   
   d. Với mỗi nhãn j ∈ Candidate_DL:
      i.   Chia 5 fold bằng MultilabelStratifiedKFold (hoặc StratifiedKFold trên nhãn j).
      ii.  Với mỗi fold k ∈ {1..5}:
           - Huấn luyện mô hình cơ sở M_j,k trên (X^(stage)[train_k], Y[train_k, j])
           - Dự đoán xác suất p_val_k trên X^(stage)[val_k]
           - Lưu vào vector dự đoán OOF P_hat_OOF[j, val_k] = p_val_k
      iii. Đánh giá Selective-F1 trên (Y[:, j], P_hat_OOF[j]):
           - Quyết định tại chi phí c = 0.30: decided = (min(p, 1-p) <= c)
           - y_pred = (p >= 0.5) trên các mẫu decided
           - Score(j) = F1(Y[decided, j], y_pred[decided])
      iv.  Nếu Score(j) >= τ_stage:
           - Promoted_stage.append(j)
   
   e. Kiểm tra điều kiện dừng:
      - Nếu Promoted_stage rỗng: Dừng thuật toán (Stopping reason: "no_promotion_in_stage").
   
   f. Cập nhật tập nhãn:
      - IL_layers.append(Promoted_stage)
      - Candidate_DL = Candidate_DL \ Promoted_stage
      - Nếu len(Candidate_DL) == 0: Dừng (Stopping reason: "all_labels_independent").
      - Nếu stage == max_depth: Dừng (Stopping reason: "max_depth_reached").
   
   g. Chuẩn bị đặc trưng cho tầng kế tiếp stage + 1:
      - Thu thập các cột xác suất OOF của toàn bộ các nhãn đã thuộc IL (Accumulated_IL).
      - Chuẩn hóa xác suất nếu cần (Augmentation Normalization: "matching" hoặc "centered").
      - X^(stage + 1) = [X, Normalize(P_hat_OOF[:, Accumulated_IL])]

3. Sắp xếp tập DL dư thừa:
   - Tính ma trận tương quan Phi-coefficient trên Y.
   - Sắp xếp Candidate_DL theo tổng tương quan tăng dần (Ascending Correlation Order):
     DL_residual = Sort_Ascending(Candidate_DL)

4. Xây dựng thứ tự thực thi thống nhất:
   - Execution_Order = Flatten(IL_layers) + DL_residual
   - Trả về CVPeelingResult đầy đủ.
```

---

## 3. Các Lựa Chọn Nâng Cao (Options từ Meeting Summary)

1. **Đo lường hiệu năng của Binary Relevance (BR) trên các nhãn độc lập ($IL$):**
   - Đánh giá Selective-F1, Full F1, Precision, Accuracy và Coverage của mô hình BR độc lập trên các nhãn thuộc $IL_1$ và $IL_2$.
   - Xuất bảng so sánh hiệu năng của BR trên từng nhãn độc lập qua 3 bộ học cơ sở.

2. **Giảm dần ngưỡng phân tầng theo các lần lặp (Decaying Threshold across Iterations):**
   - Thay vì giữ cố định 0.75, có thể giảm dần:
     - Tầng 1: $\tau_1 = 0.75$ (hoặc $0.80 - 0.85$)
     - Tầng 2: $\tau_2 = 0.70$
     - Tầng 3: $\tau_3 = 0.65$

3. **Cơ chế Ensemble Classifier Chains (ECC) khi CC trong DL vẫn thấp:**
   - Trong trường hợp cấu trúc DL quá phức tạp hoặc số nhãn thưa thớt, kích hoạt Ensemble Classifier Chains với nhiều thứ tự chuỗi ngẫu nhiên nhằm giảm phương sai tích lũy lỗi.

---

## 4. Cấu Trúc Mã Nguồn Dự Kiến

- `src/selection/cv_peeling.py`:
  - `CVPeelingConfig`: Dataclass cấu hình tham số.
  - `CVPeelingResult`: Dataclass lưu trữ kết quả và nhật ký kiểm toán.
  - `evaluate_label_5fold_cv`: Hàm kiểm định chéo 5-fold tính Selective-F1 và trích xuất xác suất OOF.
  - `CVStratifiedPeelingSelector`: Lớp thực thi thuật toán bóc tách đa tầng dựa trên 5-fold CV.
- `src/selection/__init__.py`: Export các lớp và hàm mới.
- `src/selection/partition.py`: Đăng ký partition mode `"cv_stratified_peeling"`, `"v6"`.
- `src/models/gsi_mlc_pa.py`: Tích hợp phân tầng v6 vào mô hình tổng thể.
- `tests/test_v6_cv_peeling.py`: Bộ kiểm thử tự động toàn diện.
- `scripts/run_v6_experiment.py`: Kịch bản thực nghiệm đo đạc trên các tập dữ liệu benchmark.
