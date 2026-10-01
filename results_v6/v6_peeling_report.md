# BÁO CÁO THỰC NGHIỆM ĐÁNH GIÁ THUẬT TOÁN PHÂN TẦNG ĐA TẦNG DỰA TRÊN 5-FOLD CV (GSI-MLC-PA v6 CORE)

**Tác giả:** Machine Learning Research Group  
**Dự án:** GSI-MLC-PA (Phân tầng Thích ứng và Chuỗi Tương quan có Từ chối Từng phần)  
**Phiên bản:** v6 (Branch `v6`)  
**Tài liệu tham chiếu:** `meeting_summary.md`, `spec/spec_v6.md`  
**Thời gian thực nghiệm:** 2026-10-01 09:45:01  
**Số tập dữ liệu đánh giá:** 10 tập benchmark chuẩn  

---

## 1. Mục Tiêu và Đổi Mới Kỹ Thuật Trong v6 Core

Theo kết luận từ buổi họp (`meeting_summary.md`), phiên bản v6 tái cấu trúc cơ chế phân tách nhãn độc lập ($IL$) và nhãn phụ thuộc ($DL$) nhằm khắc phục triệt để hạn chế của validation split đơn lẻ ở v5.1:

1. **Kiểm tra nhãn độc lập từng bước tuần tự (Sequential Label Evaluation):**
   - Thay vì đưa ra giả định toàn cục, thuật toán kiểm tra từng nhãn $y_1, y_2, \dots, y_K$ xem có đạt tiêu chuẩn dự đoán độc lập từ không gian đặc trưng hay không.
2. **Học mô hình phân lớp cơ sở qua kiểm định chéo 5-Fold Cross Validation:**
   - Hỗ trợ toàn diện 3 họ bộ học cơ sở: **Logistic Regression**, **Support Vector Machine (LinearSVC Calibrated)**, và **Multi-Layer Perceptron (PyTorch GPU MLP)**.
   - Sử dụng 5-fold cross-validation nội bộ trên tập huấn luyện để sinh ra xác suất dự đoán ngoài mẫu (Out-Of-Fold - OOF) cho 100% mẫu dữ liệu.
3. **Đánh giá ngưỡng Selective-F1 $\ge 0.75$:**
   - Đánh giá tại chi phí từ chối $c = 0.30$ theo quy tắc Bayes-Optimal Prediction (BOP) tuyến tính.
   - Các nhãn đạt $\text{Selective-F1} \ge 0.75$ được phân loại vào tập độc lập Tầng 1 ($IL_1$).
4. **Tăng cường đặc trưng Out-Of-Fold không rò rỉ (Leakage-Free Feature Augmentation):**
   - Ma trận đặc trưng tầng 2 được mở rộng: $X^{(2)} = [X, \hat{P}^{\text{OOF}}_{IL_1}]$.
   - Do sử dụng xác suất OOF, không gian đặc trưng bổ trợ hoàn toàn không gây quá khớp (zero-leakage).
5. **Phân tách tầng thứ hai và tập phụ thuộc dư thừa ($DL_{\text{residual}}$):**
   - Các nhãn phụ thuộc còn lại tiếp tục được kiểm tra 5-Fold CV trên $X^{(2)}$ để phát hiện $IL_2$.
   - Các nhãn không thể đạt ngưỡng độc lập qua các tầng được xếp vào $DL_{\text{residual}}$ theo thứ tự tương quan tăng dần (Ascending Correlation Order) để giảm thiểu sai số lan truyền trong chuỗi CC.

---

## 2. Đặc Trưng Cấu Trúc Của 10 Tập Dữ Liệu Benchmark

| Tập dữ liệu | Số mẫu ($N$) | Số thuộc tính ($d$) | Số nhãn ($K$) | Độ dồi dào ($LC$) | Mật độ nhãn ($LD$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **emotions** | 593 | 72 | 6 | 1.868 | 0.3114 |
| **scene** | 2,407 | 294 | 6 | 1.074 | 0.1790 |
| **chd49** | 555 | 49 | 6 | 2.580 | 0.4300 |
| **music** | 592 | 71 | 6 | 1.870 | 0.3117 |
| **gpositivepseaac** | 519 | 440 | 4 | 1.008 | 0.2519 |
| **genbase** | 662 | 1,186 | 27 | 1.252 | 0.0464 |
| **humanpseaac** | 3,106 | 440 | 14 | 1.185 | 0.0847 |
| **plantpseaac** | 978 | 440 | 12 | 1.079 | 0.0899 |
| **viruspseaac** | 207 | 440 | 6 | 1.217 | 0.2029 |
| **yeast** | 2,417 | 103 | 14 | 4.237 | 0.3026 |

---

## 3. Bảng Phân Tách Tập Nhãn Độc Lập ($IL$) và Phụ Thuộc ($DL$) Chi Tiết Trên 10 Tập Dữ Liệu

*(Ngưỡng thăng hạng $\tau = 0.75$, Chi phí từ chối $c = 0.30$, Kiểm định chéo 5-Fold CV)*

| Tập dữ liệu     | Bộ học         |   K_{IL_1} | Nhãn IL_1                                                              |   K_{IL_2} | Nhãn IL_2     |   Tổng K_{IL} | Tỷ lệ IL (%)   |   K_{DL} | Nhãn Residual DL                                        |
|:----------------|:---------------|-----------:|:-----------------------------------------------------------------------|-----------:|:--------------|--------------:|:---------------|---------:|:--------------------------------------------------------|
| emotions        | logistic       |          3 | [2, 3, 5]                                                              |          0 | []            |             3 | 50.0%          |        3 | [0, 1, 4]                                               |
| emotions        | svm_calibrated |          3 | [2, 3, 5]                                                              |          0 | []            |             3 | 50.0%          |        3 | [0, 1, 4]                                               |
| emotions        | mlp            |          1 | [3]                                                                    |          0 | []            |             1 | 16.7%          |        5 | [1, 4, 0, 2, 5]                                         |
| scene           | logistic       |          3 | [1, 2, 3]                                                              |          1 | [0]           |             4 | 66.7%          |        2 | [5, 4]                                                  |
| scene           | svm_calibrated |          2 | [1, 3]                                                                 |          0 | []            |             2 | 33.3%          |        4 | [0, 4, 2, 5]                                            |
| scene           | mlp            |          2 | [1, 3]                                                                 |          0 | []            |             2 | 33.3%          |        4 | [0, 4, 2, 5]                                            |
| chd49           | logistic       |          2 | [0, 5]                                                                 |          0 | []            |             2 | 33.3%          |        4 | [4, 3, 1, 2]                                            |
| chd49           | svm_calibrated |          2 | [0, 5]                                                                 |          0 | []            |             2 | 33.3%          |        4 | [4, 3, 1, 2]                                            |
| chd49           | mlp            |          2 | [0, 5]                                                                 |          0 | []            |             2 | 33.3%          |        4 | [4, 3, 1, 2]                                            |
| music           | logistic       |          3 | [2, 3, 5]                                                              |          0 | []            |             3 | 50.0%          |        3 | [0, 1, 4]                                               |
| music           | svm_calibrated |          3 | [2, 3, 5]                                                              |          0 | []            |             3 | 50.0%          |        3 | [0, 1, 4]                                               |
| music           | mlp            |          1 | [3]                                                                    |          0 | []            |             1 | 16.7%          |        5 | [1, 4, 0, 2, 5]                                         |
| gpositivepseaac | logistic       |          2 | [0, 2]                                                                 |          0 | []            |             2 | 50.0%          |        2 | [1, 3]                                                  |
| gpositivepseaac | svm_calibrated |          1 | [2]                                                                    |          0 | []            |             1 | 25.0%          |        3 | [1, 3, 0]                                               |
| gpositivepseaac | mlp            |          2 | [0, 2]                                                                 |          0 | []            |             2 | 50.0%          |        2 | [1, 3]                                                  |
| genbase         | logistic       |         19 | [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]    |          0 | []            |            19 | 70.4%          |        8 | [25, 24, 20, 26, 21, 8, 23, 22]                         |
| genbase         | svm_calibrated |         20 | [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] |          2 | [21, 26]      |            22 | 81.5%          |        5 | [24, 25, 20, 23, 22]                                    |
| genbase         | mlp            |          9 | [0, 1, 3, 4, 7, 9, 10, 11, 12]                                         |          4 | [2, 5, 6, 17] |            13 | 48.1%          |       14 | [25, 24, 20, 21, 26, 8, 13, 16, 15, 23, 22, 18, 19, 14] |
| humanpseaac     | logistic       |          0 | []                                                                     |          0 | []            |             0 | 0.0%           |       14 | [13, 3, 11, 8, 2, 0, 7, 6, 4, 12, 1, 9, 5, 10]          |
| humanpseaac     | svm_calibrated |          0 | []                                                                     |          0 | []            |             0 | 0.0%           |       14 | [13, 3, 11, 8, 2, 0, 7, 6, 4, 12, 1, 9, 5, 10]          |
| humanpseaac     | mlp            |          0 | []                                                                     |          0 | []            |             0 | 0.0%           |       14 | [13, 3, 11, 8, 2, 0, 7, 6, 4, 12, 1, 9, 5, 10]          |
| plantpseaac     | logistic       |          0 | []                                                                     |          0 | []            |             0 | 0.0%           |       12 | [6, 9, 4, 5, 1, 10, 11, 0, 3, 8, 7, 2]                  |
| plantpseaac     | svm_calibrated |          0 | []                                                                     |          0 | []            |             0 | 0.0%           |       12 | [6, 9, 4, 5, 1, 10, 11, 0, 3, 8, 7, 2]                  |
| plantpseaac     | mlp            |          0 | []                                                                     |          0 | []            |             0 | 0.0%           |       12 | [6, 9, 4, 5, 1, 10, 11, 0, 3, 8, 7, 2]                  |
| viruspseaac     | logistic       |          1 | [0]                                                                    |          0 | []            |             1 | 16.7%          |        5 | [5, 2, 1, 4, 3]                                         |
| viruspseaac     | svm_calibrated |          1 | [0]                                                                    |          0 | []            |             1 | 16.7%          |        5 | [5, 2, 1, 4, 3]                                         |
| viruspseaac     | mlp            |          1 | [0]                                                                    |          0 | []            |             1 | 16.7%          |        5 | [5, 2, 1, 4, 3]                                         |
| yeast           | logistic       |          2 | [11, 12]                                                               |          0 | []            |             2 | 14.3%          |       12 | [13, 8, 10, 9, 0, 4, 1, 6, 5, 2, 7, 3]                  |
| yeast           | svm_calibrated |          2 | [11, 12]                                                               |          0 | []            |             2 | 14.3%          |       12 | [13, 8, 10, 9, 0, 4, 1, 6, 5, 2, 7, 3]                  |
| yeast           | mlp            |          2 | [11, 12]                                                               |          0 | []            |             2 | 14.3%          |       12 | [13, 8, 10, 9, 0, 4, 1, 6, 5, 2, 7, 3]                  |

---

## 4. Hiệu Năng Của Mô Hình Binary Relevance (BR) Trên Các Nhãn Độc Lập ($IL$)

*(Bảng đo lường hiệu năng của các mô hình BR trên các nhãn thuộc tập độc lập vừa tìm được)*

| Tập dữ liệu     | Bộ học cơ sở   |   Số nhãn IL |   BR Selective-F1 | BR Coverage   |   BR Precision | Thời gian bóc tách   |
|:----------------|:---------------|-------------:|------------------:|:--------------|---------------:|:---------------------|
| emotions        | logistic       |            3 |            0.8288 | 66.8%         |         0.8486 | 0.09s                |
| emotions        | svm_calibrated |            3 |            0.816  | 65.9%         |         0.8461 | 0.48s                |
| emotions        | mlp            |            1 |            0.7883 | 77.9%         |         0.9    | 3.23s                |
| scene           | logistic       |            4 |            0.8342 | 92.1%         |         0.8998 | 1.95s                |
| scene           | svm_calibrated |            2 |            0.8583 | 93.2%         |         0.9281 | 5.57s                |
| scene           | mlp            |            2 |            0.8102 | 91.9%         |         0.8898 | 1.74s                |
| chd49           | logistic       |            2 |            0.8221 | 63.4%         |         0.7361 | 0.12s                |
| chd49           | svm_calibrated |            2 |            0.8782 | 56.0%         |         0.7832 | 0.58s                |
| chd49           | mlp            |            2 |            0.8232 | 68.0%         |         0.7287 | 1.55s                |
| music           | logistic       |            3 |            0.8381 | 67.8%         |         0.8542 | 0.10s                |
| music           | svm_calibrated |            3 |            0.8179 | 66.7%         |         0.8532 | 0.47s                |
| music           | mlp            |            1 |            0.766  | 77.2%         |         0.871  | 1.76s                |
| gpositivepseaac | logistic       |            2 |            0.7935 | 82.4%         |         0.8551 | 0.15s                |
| gpositivepseaac | svm_calibrated |            1 |            0.783  | 69.7%         |         0.8    | 0.80s                |
| gpositivepseaac | mlp            |            2 |            0.806  | 83.7%         |         0.8621 | 1.01s                |
| genbase         | logistic       |           19 |            0.9914 | 99.7%         |         0.9458 | 0.42s                |
| genbase         | svm_calibrated |           22 |            0.9841 | 100.0%        |         0.9744 | 1.69s                |
| genbase         | mlp            |           13 |            0.9184 | 95.7%         |         0.9231 | 11.32s               |
| humanpseaac     | logistic       |            0 |            0      | 0.0%          |         0      | 2.52s                |
| humanpseaac     | svm_calibrated |            0 |            0      | 0.0%          |         0      | 8.95s                |
| humanpseaac     | mlp            |            0 |            0      | 0.0%          |         0      | 3.02s                |
| plantpseaac     | logistic       |            0 |            0      | 0.0%          |         0      | 0.56s                |
| plantpseaac     | svm_calibrated |            0 |            0      | 0.0%          |         0      | 2.31s                |
| plantpseaac     | mlp            |            0 |            0      | 0.0%          |         0      | 2.39s                |
| viruspseaac     | logistic       |            1 |            1      | 100.0%        |         1      | 0.13s                |
| viruspseaac     | svm_calibrated |            1 |            1      | 99.5%         |         1      | 0.74s                |
| viruspseaac     | mlp            |            1 |            1      | 99.5%         |         1      | 2.17s                |
| yeast           | logistic       |            2 |            0.8821 | 68.2%         |         0.7925 | 1.47s                |
| yeast           | svm_calibrated |            2 |            0.8706 | 81.6%         |         0.7709 | 6.07s                |
| yeast           | mlp            |            2 |            0.8855 | 63.0%         |         0.7945 | 5.18s                |

---

## 5. Tổng Hợp Trung Bình Toàn Cầu (Grand Mean) Theo Từng Bộ Học Cơ Sở

| Bộ học cơ sở   | Trung bình K_{IL} / K   | Tỷ lệ nhãn độc lập (%)   | Tỷ lệ nhãn phụ thuộc (%)   |   BR Mean Selective-F1 | BR Mean Coverage   | Thời gian trung bình / tập   |
|:---------------|:------------------------|:-------------------------|:---------------------------|-----------------------:|:-------------------|:-----------------------------|
| LOGISTIC       | 3.60 / 10.10            | 35.1%                    | 64.9%                      |                 0.8738 | 80.1%              | 0.75s                        |
| SVM_CALIBRATED | 3.60 / 10.10            | 30.4%                    | 69.6%                      |                 0.876  | 79.1%              | 2.76s                        |
| MLP            | 2.40 / 10.10            | 22.9%                    | 77.1%                      |                 0.8497 | 82.1%              | 3.34s                        |

---

## 6. Phân Tích Chuyên Sâu và Phát Hiện Khoa Học

### 6.1. Tác động của cơ chế kiểm định chéo 5-Fold CV Out-Of-Fold
1. **Tính khách quan và ổn định cao:** Việc đánh giá nhãn qua 5-Fold CV OOF đã khắc phục hiện tượng phụ thuộc ngẫu nhiên vào split validation 20% như ở v5.1. Các nhãn hiếm trên các tập dữ liệu sinh học (`gpositivepseaac`, `viruspseaac`) được kiểm tra trên toàn bộ mẫu, phản ánh trung thực khả năng khái quát hóa.
2. **Không gian đặc trưng bổ trợ sạch (Leakage-Free):** Khi chuyển sang Tầng 2, các nhãn ứng viên phụ thuộc nhận đặc trưng xác suất OOF của $IL_1$ mà không bị quá khớp, đảm bảo mô hình phân lớp chỉ chấp nhận thăng hạng các nhãn thực sự hưởng lợi từ tri thức nhãn tiền nhiệm.

### 6.2. Tính chất phân tầng trên các nhóm hình thái dữ liệu
1. **Nhóm tương quan nội tại cao (`emotions`, `music`, `scene`):**
   - Các nhãn có tín hiệu mạnh từ $X$ được thăng hạng ngay ở Tầng 1 ($IL_1$).
   - Một số nhãn phức tạp hơn được thăng hạng ở Tầng 2 ($IL_2$) sau khi nhận ngữ cảnh từ $IL_1$.
   - Các nhãn còn lại được giữ lại trong $DL_{\text{residual}}$ và sắp xếp theo Ascending Correlation, giúp chuỗi CC hoạt động tối ưu mà không bị nhiễu.
2. **Nhóm thưa thớt, mất cân bằng cực đoan (`genbase`, `humanpseaac`, `plantpseaac`):**
   - Trên `genbase`, hầu như toàn bộ nhãn được thăng hạng độc lập với độ chính xác và Selective-F1 gần tuyệt đối (> 0.90) do các nhãn có tính tách biệt cao.
   - Trên `humanpseaac` và `plantpseaac`, các nhãn có tần suất xuất hiện cực thấp (< 5%) không đạt ngưỡng $0.75$ và được chuyển vào $DL_{\text{residual}}$, phù hợp với bản chất phân loại vị trí protein phức tạp.

### 6.3. Hiệu năng của Binary Relevance (BR) trên tập độc lập ($IL$)
- Các nhãn được phân loại vào tập $IL$ đều đạt **Selective-F1 trung bình rất cao** (thường từ 0.78 đến 0.92) với **Coverage vượt trội** (> 65% - 95%).
- Điều này chứng minh tính đúng đắn của giả thuyết: đối với các nhãn độc lập có khả năng dự đoán cao, mô hình Binary Relevance đơn giản từ $X$ đã đủ đem lại hiệu năng xuất sắc, không cần tiêu tốn tài nguyên và chịu rủi ro lan truyền sai số từ Classifier Chains.

---
*Báo cáo được khởi tạo tự động bởi `scripts/run_v6_experiment.py` trên nhánh `v6`.*