# Báo Cáo Nghiên Cứu & So Sánh Thực Nghiệm Các Phương Pháp Phân Loại Đa Nhãn (Multi-Label Classification)

> **Tài liệu tham khảo lý thuyết:**
> 1. Zhang, M.-L., et al. (2018). *Binary relevance for multi-label learning: an overview.* Frontiers of Computer Science. (`FCS'17.pdf`)
> 2. Read, J., et al. (2011). *Classifier chains for multi-label classification.* Machine Learning. (`s10994-011-5256-5.pdf`)
> 3. Popescu, M.-C., et al. *Multilayer Perceptron and Neural Networks.* (`Multilayer_perceptron_and_neural_networks.pdf`)

---

## 1. Tổng Quan Thực Nghiệm

Nghiên cứu này đánh giá và so sánh toàn diện **5 phương pháp phân loại đa nhãn** trên **10 tập dữ liệu benchmark quốc tế** đa dạng về miền dữ liệu (âm nhạc, hình ảnh, văn bản, sinh học) bằng quy trình **5-Fold Multilabel Stratified Cross-Validation**:

1. **BR (LinearSVC)**: Binary Relevance truyền thống với Linear Support Vector Classifier (Zhang et al., 2018).
2. **BR (Logistic Reg)**: Binary Relevance với Logistic Regression (mô hình xác suất lồi, hàm mất mát Log-Loss).
3. **BR (MLP)**: Multi-Label Multi-Layer Perceptron (Mạng nơ-ron đa tầng chạy song song trên GPU RTX 4060, hàm kích hoạt ReLU, tối ưu hóa AdamW, điều chỉnh độ lệch nhãn `pos_weight` theo Popescu et al.).
4. **CC (LinearSVC)**: Classifier Chains với LinearSVC (Read et al., 2011) mô hình hóa tương quan nhãn theo chuỗi.
5. **CC (MLP)**: Classifier Chains với mạng nơ-ron đa tầng (MLP) làm bộ phân loại cơ sở trong từng mắt xích chuỗi.

---

## 2. Bảng Tổng Hợp Kết Quả Thực Nghiệm Toàn Diện

*Dưới đây là giá trị trung bình qua 5 fold (Mean ± Std). Giá trị in đậm đại diện cho mô hình tốt nhất trên từng tiêu chí.*

### 2.1. Macro-F1 (Tiêu chí bắt buộc, trung bình không trọng số trên từng nhãn ↑)

| Tập dữ liệu | BR (LinearSVC) | BR (Logistic) | **BR (MLP)** | CC (LinearSVC) | **CC (MLP)** | Mô hình tốt nhất |
|---|---|---|---|---|---|---|
| **EMOTIONS** | 0.630 ± 0.008 | 0.597 ± 0.015 | **0.678 ± 0.015** | 0.619 ± 0.023 | 0.666 ± 0.022 | **BR (MLP)** |
| **MUSIC** | 0.643 ± 0.018 | 0.607 ± 0.020 | 0.652 ± 0.041 | 0.635 ± 0.025 | **0.670 ± 0.027** | **CC (MLP)** |
| **SCENE** | 0.683 ± 0.023 | 0.699 ± 0.023 | **0.782 ± 0.020** | 0.697 ± 0.027 | 0.710 ± 0.023 | **BR (MLP)** |
| **YEAST** | 0.361 ± 0.008 | 0.375 ± 0.009 | **0.490 ± 0.010** | 0.407 ± 0.014 | 0.481 ± 0.014 | **BR (MLP)** |
| **CAL500** | 0.095 ± 0.006 | 0.074 ± 0.003 | **0.204 ± 0.004** | 0.131 ± 0.006 | 0.200 ± 0.009 | **BR (MLP)** |
| **BIBTEX** | **0.330 ± 0.010** | 0.283 ± 0.012 | 0.291 ± 0.006 | 0.328 ± 0.010 | 0.256 ± 0.010 | **BR (LinearSVC)** |
| **ENRON** | 0.226 ± 0.014 | 0.216 ± 0.011 | **0.234 ± 0.009** | 0.225 ± 0.010 | 0.222 ± 0.011 | **BR (MLP)** |
| **GENBASE** | **0.791 ± 0.016** | 0.678 ± 0.033 | 0.765 ± 0.040 | **0.791 ± 0.016** | 0.740 ± 0.032 | **BR / CC (SVC)** |
| **MEDICAL** | 0.394 ± 0.041 | 0.284 ± 0.009 | 0.374 ± 0.024 | **0.403 ± 0.045** | 0.305 ± 0.045 | **CC (LinearSVC)** |
| **REUTERS-K500** | 0.273 ± 0.017 | 0.165 ± 0.009 | 0.239 ± 0.006 | **0.283 ± 0.016** | 0.259 ± 0.018 | **CC (LinearSVC)** |
| **TRUNG BÌNH** | **0.4426** | **0.3979** | **0.4710 (Top 1 🏆)** | **0.4519** | **0.4509** | **BR (MLP)** |

---

### 2.2. Bảng So Sánh Tổng Thể Trên Tất Cả 7 Metrics (Trung bình qua 10 Datasets)

| Metric | Hướng tối ưu | BR (LinearSVC) | BR (Logistic) | **BR (MLP)** | CC (LinearSVC) | **CC (MLP)** | Mô hình dẫn đầu |
|---|---|---|---|---|---|---|---|
| **Macro-F1** | ↑ Cao hơn | 0.4426 | 0.3979 | **0.4710** | 0.4519 | 0.4509 | **BR (MLP) 🏆** |
| **Micro-F1** | ↑ Cao hơn | 0.6207 | 0.6105 | **0.6341** | 0.6193 | 0.6132 | **BR (MLP) 🏆** |
| **Hamming Loss** | ↓ Thấp hơn | 0.0952 | **0.0921** | 0.1080 | 0.1027 | 0.1129 | **BR (Logistic) 🏆** |
| **Subset Accuracy** | ↑ Cao hơn | 0.3427 | 0.3337 | 0.3422 | **0.3764** | 0.3249 | **CC (LinearSVC) 🏆** |
| **Example-F1** | ↑ Cao hơn | 0.5895 | 0.5632 | **0.6283** | 0.6046 | 0.6003 | **BR (MLP) 🏆** |
| **Macro Precision** | ↑ Cao hơn | 0.5033 | **0.5057** | 0.4638 | 0.4919 | 0.4504 | **BR (Logistic) 🏆** |
| **Macro Recall** | ↑ Cao hơn | 0.4207 | 0.3607 | **0.5011** | 0.4409 | 0.4792 | **BR (MLP) 🏆** |

---

## 3. Phân Tích Chuyên Sâu Các Phương Pháp

### 3.1. Đánh giá Mạng Nơ-ron Đa Tầng (MLP: BR_MLP và CC_MLP)
- **Ưu điểm vượt trội về biểu diễn phi tuyến**:
  - `BR_MLP` đạt **Macro-F1 trung bình cao nhất (0.4710)** và giành chiến thắng trên 5/10 datasets (`emotions`, `scene`, `yeast`, `cal500`, `enron`).
  - Đặc biệt trên các tập dữ liệu phi tuyến tính phức tạp như `scene` (F1 tăng từ 0.6829 lên **0.7822**, tăng +14.5%) và `yeast` (F1 tăng từ 0.3607 lên **0.4902**, tăng +35.9%), mạng MLP với hàm kích hoạt ReLU và cấu trúc ẩn đa lớp đã học được các ranh giới phân lớp phi tuyến mà mô hình tuyến tính (LinearSVC, Logistic) không thể phân tách được.
  - Trên tập dữ liệu cực kỳ mất cân bằng và nhiều nhãn như `cal500` (174 nhãn), MLP cải thiện Macro-F1 hơn **gấp đôi** (từ 0.0952 lên **0.2040**) nhờ cơ chế cân bằng trọng số nhãn dương (`pos_weight`) và khả năng khái quát hóa thông qua chia sẻ biểu diễn ẩn (Shared Representation).

- **Khả năng nắm bắt nhãn hiếm (Macro Recall)**:
  - MLP đạt Macro Recall **0.5011** (vượt xa LinearSVC 0.4207 và Logistic 0.3607). Điều này chứng minh mạng MLP không bị thiên lệch dự đoán toàn bộ là 0 đối với các nhãn hiếm.

### 3.2. Đánh giá Classifier Chains (CC)
- **Ưu thế về Subset Accuracy (Khớp toàn bộ tập nhãn)**:
  - `CC (LinearSVC)` đạt Subset Accuracy cao nhất (**0.3764** so với BR **0.3427**).
  - Điều này hoàn toàn khớp với lý thuyết của Read et al. (2011): việc truyền các nhãn đã dự đoán trước đó làm đặc trưng bổ sung giúp mô hình nắm bắt được **tương quan nhãn co-occurrence**, từ đó giảm thiểu các dự đoán mâu thuẫn giữa các nhãn trong cùng một mẫu.
  - Tuy nhiên, trong CC xuất hiện hiện tượng **lan truyền lỗi (error propagation)**: nếu một mắt xích đầu chuỗi dự đoán sai, sai số sẽ truyền sang các mắt xích phía sau, khiến Hamming Loss của CC cao hơn BR một chút.

### 3.3. Đánh giá Binary Relevance (BR) Tuyến Tính (LinearSVC & Logistic Regression)
- **Hiệu quả cao trên dữ liệu văn bản thưa số chiều lớn (Text/Bag-of-Words)**:
  - Trên các tập dữ liệu văn bản có số chiều đặc trưng rất lớn ($d > 1000$) như `bibtex` ($d=1836$), `genbase` ($d=1186$), `medical` ($d=1449$), `reuters-k500` ($d=500$), `LinearSVC` hoạt động cực kỳ hiệu quả (thắng trên `bibtex` và `genbase`).
  - Lý do: Theo lý thuyết học máy thống kê, trong không gian số chiều rất cao, dữ liệu thường có xu hướng phân tách tuyến tính tốt, do đó mô hình biên cực đại SVM tuyến tính vừa tránh overfitting vừa tối ưu hóa lồi toàn cục.
- **Hamming Loss & Precision**:
  - `BR_Logistic` và `BR_LinearSVC` đạt Hamming Loss thấp nhất (~0.092 - 0.095) và Precision cao nhất (~0.505) do có xu hướng dự đoán thận trọng ở ngưỡng xác suất 0.5.

---

## 4. Kết Luận & Khuyến Nghị Thực Tiễn

1. **Khi nào nên chọn BR_MLP / Multi-Label MLP**:
   - Khi bài toán có đặc trưng liên tục, phi tuyến tính (Audio, Image, Biological Data như `emotions`, `scene`, `yeast`, `cal500`).
   - Khi số lượng nhãn lớn và dữ liệu mất cân bằng nhãn cao (MLP với hàm mất mát có trọng số giải quyết rất tốt).
   - Tận dụng tính toán song song trên GPU cho tốc độ huấn luyện dưới 1 giây.

2. **Khi nào nên chọn Classifier Chains (CC)**:
   - Khi mục tiêu ưu tiên hàng đầu là **Subset Accuracy / Exact Match Ratio** (cần dự đoán đúng chính xác toàn bộ tổ hợp nhãn của mẫu).

3. **Khi nào nên chọn LinearSVC / Logistic Regression**:
   - Khi không gian đặc trưng là văn bản thưa (Sparse Bag-of-Words/TF-IDF) có số chiều rất lớn ($d \ge 1,000$).
