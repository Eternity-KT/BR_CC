# Multi-Label Classification: Binary Relevance (BR) vs Classifier Chains (CC)

Dự án triển khai và đánh giá thực nghiệm so sánh các thuật toán phân loại đa nhãn (Multi-Label Classification) kinh điển: **Binary Relevance (BR)** (LinearSVC & Logistic Regression) và **Classifier Chains (CC)** trên **10 tập dữ liệu benchmark** với quy trình **5-Fold Stratified Cross-Validation**, tuân thủ nghiêm ngặt các chuẩn được công bố trong các bài báo gốc.

---

## 1. Tài Liệu Tham Khảo (References)

1. **Binary Relevance (BR)**:
   - *Zhang, M.-L., Li, Y.-K., Liu, X.-Y., & Geng, X. (2018).* **Binary relevance for multi-label learning: an overview.** *Frontiers of Computer Science*, 12(2), 191–202. (`TaiLieuThamKhao/FCS'17.pdf`)
2. **Classifier Chains (CC)**:
   - *Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011).* **Classifier chains for multi-label classification.** *Machine Learning*, 85(3), 333–359. (`TaiLieuThamKhao/s10994-011-5256-5.pdf`)

---

## 2. Chi Tiết Cấu Hình Mô Hình (Model Configurations)

### 2.1. Cấu Hình Chung & Base Classifiers

Hệ thống hỗ trợ 2 thuật toán bộ phân loại cơ sở (base binary classifiers):

| Base Classifier | Cấu hình siêu tham số | Loss Function | Đặc điểm & Ưu thế |
|---|---|---|---|
| **Support Vector Machine (`LinearSVC`)** | `C=1.0`, `dual="auto"`, `max_iter=10000`, `random_state=42` | Hinge Loss: $\max(0, 1 - y \cdot f(x))$ | Tối đa hóa lề (Max-margin), cực kỳ mạnh mẽ trên không gian đặc trưng số chiều lớn. |
| **Hồi Quy Logistic (`LogisticRegression`)** | `solver="liblinear"`, `C=1.0`, `max_iter=1000`, `random_state=42` | Log Loss: $-\log P(y \mid x)$ | Đầu ra xác suất chuẩn xác $P(Y_j = 1 \mid X) \in [0, 1]$ qua hàm Sigmoid, tối ưu mượt mà. |

> **Xử lý nhãn đơn (Degenerate Cases):** Lớp `_ConstantClassifier` tự động xử lý khi một nhãn trong fold chỉ xuất hiện 1 lớp (toàn 0 hoặc toàn 1), hỗ trợ đầy đủ `predict()`, `predict_proba()`, và `decision_function()`.

---

### 2.2. Cấu Hình Binary Relevance (BR)

- **Module:** `src/models/binary_relevance.py`
- **Các lớp được cung cấp:**
  - `BinaryRelevanceClassifier(base_estimator=None, random_state=42)`: Mặc định sử dụng LinearSVC (hoặc truyền `'logistic'` / `'svm'` / estimator instance).
  - `BinaryRelevanceLogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=42)`: Phiên bản chuyên biệt sử dụng Hồi quy Logistic.
- **Nguyên lý:** Phân rã bài toán đa nhãn thành $q$ bài toán phân loại nhị phân độc lập ($q$ là số lượng nhãn).
- **Không gian thuộc tính mỗi bộ phân loại:** $X \in \mathbb{R}^{d}$ (giữ nguyên không gian $d$ chiều ban đầu).
- **Hàm dự đoán xác suất:** `predict_proba(X)` trả về ma trận $P(Y_j = 1 \mid X) \in [0, 1]^{n \times q}$.
- **Đặc trưng:** Đơn giản, tính toán song song độc lập, tối ưu tự nhiên cho các chỉ số macro-average.

---

### 2.3. Cấu Hình Classifier Chains (CC)

- **Module:** `src/models/classifier_chain.py`
- **Nguyên lý:** Xâu chuỗi $q$ bộ phân loại nhị phân theo thứ tự, truyền nhãn trước đó làm đặc trưng bổ sung để mô hình hóa quan hệ phụ thuộc giữa các nhãn.
- **Thứ tự chuỗi (Chain Order):** **Thứ tự mặc định (Natural/Default Order)** $[0, 1, \dots, q-1]$ theo đúng thiết lập chuẩn trong Read et al. (2011).
- **Training Phase:** Classifier thứ $j$ nhận đầu vào mở rộng $x_i^{(j)} = [x_i, y_i^1, \dots, y_i^{j-1}] \in \mathbb{R}^{d + (j - 1)}$ với ground-truth labels.
- **Inference Phase:** Greedy propagation với dự đoán nhị phân $\hat{y} \in \{0, 1\}$.
- **Đặc trưng:** Nắm bắt tương quan nhãn, cải thiện mạnh trên các chỉ số toàn cục như Subset Accuracy & Macro-F1.

---

## 3. Danh Sách 10 Tập Dữ Liệu Benchmark

| # | Dataset | Domain | Samples | Features ($d$) | Labels ($q$) | Label Cardinality | Format / Vị trí nhãn |
|---|---|---|---|---|---|---|---|
| 1 | **Emotions** | Music | 593 | 72 | 6 | 1.87 | MULAN (XML + ARFF, nhãn ở cuối) |
| 2 | **Music** | Music | 592 | 71 | 6 | 1.87 | ARFF (6 nhãn ở đầu) |
| 3 | **Scene** | Image | 2,407 | 294 | 6 | 1.07 | ARFF (6 nhãn ở đầu) |
| 4 | **Yeast** | Biology | 2,417 | 103 | 14 | 4.24 | ARFF (14 nhãn ở đầu) |
| 5 | **CAL500** | Music | 502 | 68 | 174 | 26.04 | ARFF (174 nhãn ở cuối) |
| 6 | **Bibtex** | Text | 7,395 | 1,836 | 159 | 2.40 | MULAN (XML + ARFF, nhãn ở cuối) |
| 7 | **Enron** | Text | 1,702 | 1,001 | 53 | 3.38 | MULAN (XML + ARFF, nhãn ở cuối) |
| 8 | **Genbase** | Biology | 662 | 1,186 | 27 | 1.25 | MULAN (XML + ARFF, nhãn ở cuối) |
| 9 | **Medical** | Text | 978 | 1,449 | 45 | 1.25 | MULAN (XML + ARFF, nhãn ở cuối) |
| 10 | **REUTERS-K500** | Text | 6,000 | 500 | 103 | 1.46 | ARFF (103 nhãn ở đầu) |

---

## 4. Bộ Chỉ Số Đánh Giá (Evaluation Metrics)

Hệ thống tính toán đầy đủ 7 chỉ số chuẩn:

1. **Macro-F1 (⭐ Bắt buộc):** Trung bình điểm F1 trên từng nhãn ($F1_{macro} = \frac{1}{q}\sum_{j=1}^q F1_j$).
2. **Micro-F1:** Tính F1 gộp toàn bộ các cặp instance-label.
3. **Hamming Loss (↓ thấp = tốt):** Tỷ lệ dự đoán sai nhãn trên toàn bộ nhãn ($HL = \frac{1}{n \cdot q}\sum_{i=1}^n \sum_{j=1}^q \mathbb{I}(y_{ij} \neq \hat{y}_{ij})$).
4. **Subset Accuracy (Exact Match):** Tỷ lệ mẫu có vector nhãn dự đoán khớp hoàn toàn 100% với ground truth.
5. **Example-based F1:** F1 tính trung bình trên từng mẫu dữ liệu.
6. **Macro Precision:** Trung bình Precision trên tất cả các nhãn.
7. **Macro Recall:** Trung bình Recall trên tất cả các nhãn.

---

## 5. Cấu Trúc Mã Nguồn

```
BR_CC/
├── specify.md                          # Tài liệu đặc tả kỹ thuật chi tiết
├── README.md                           # Hướng dẫn & cấu hình BR/CC
├── requirements.txt                    # Thư viện phụ thuộc
├── main.py                             # File thực thi pipeline chính (hỗ trợ CLI)
├── tests_unit.py                       # Unit tests kiểm tra tính đúng đắn mô hình
├── TaiLieuThamKhao/
│   ├── FCS'17.pdf                      # Bài báo tổng quan Binary Relevance
│   └── s10994-011-5256-5.pdf           # Bài báo Classifier Chains
├── data/                               # 10 tập dữ liệu ARFF/XML
├── src/
│   ├── data/
│   │   └── loader.py                   # Bộ nạp dữ liệu chuẩn hóa 10 datasets
│   ├── models/
│   │   ├── __init__.py                 # Export BR, BR_Logistic, CC
│   │   ├── binary_relevance.py         # Cài đặt BR (LinearSVC + Logistic Regression)
│   │   └── classifier_chain.py         # Cài đặt CC
│   ├── evaluation/
│   │   ├── cv.py                       # 5-fold Multilabel Stratified CV
│   │   └── metrics.py                  # Bộ 7 metrics đánh giá
│   └── visualization/
│       └── plots.py                    # Xuất biểu đồ đa mô hình 300 DPI và bảng kết quả
└── results/                            # Thư mục chứa kết quả sau khi chạy
    ├── tables/
    │   ├── summary_results.csv         # Bảng Mean ± Std cho tất cả metrics
    │   └── raw_results.json            # Chi tiết điểm từng fold
    └── figures/                        # 12+ biểu đồ trực quan hóa
        ├── macro_f1_comparison.png
        ├── micro_f1_comparison.png
        ├── hamming_loss_comparison.png
        ├── subset_accuracy_comparison.png
        ├── example_f1_comparison.png
        ├── macro_precision_comparison.png
        ├── macro_recall_comparison.png
        ├── radar_overall.png
        ├── heatmap_br.png
        ├── heatmap_br_logistic.png
        ├── heatmap_cc.png
        └── summary_table.png
```

---

## 6. Hướng Dẫn Cài Đặt & Thực Thi Trên Terminal

### 6.1. Cài Đặt Thư Viện

```bash
pip install -r requirements.txt
```

### 6.2. Chạy Unit Tests

```bash
python tests_unit.py
```

### 6.3. Chạy Thực Nghiệm Đầy Đủ (10 Datasets, 3 Mô Hình: BR, BR_Logistic, CC)

Chạy lệnh sau trên terminal:

```bash
python main.py
```

### 6.4. Tùy Chọn Tham Số Dòng Lệnh (CLI Arguments)

```bash
# Chạy một số dataset cụ thể với cả 3 mô hình:
python main.py --datasets emotions music scene --models BR BR_Logistic CC

# Chỉ so sánh Binary Relevance LinearSVC vs Logistic Regression:
python main.py --datasets emotions yeast --models BR BR_Logistic

# Tùy chỉnh số folds và thư mục lưu kết quả:
python main.py --datasets emotions --n_splits 5 --output_dir results_experiment
```

### 6.5. Sử Dụng Trực Tiếp Trong Mã Python

```python
from src.data.loader import load_dataset
from src.models import BinaryRelevanceLogisticRegression, BinaryRelevanceClassifier

# Load dữ liệu
X, Y, _, _ = load_dataset('emotions')

# Khởi tạo và huấn luyện BR Logistic Regression
clf = BinaryRelevanceLogisticRegression(C=1.0, max_iter=1000)
clf.fit(X, Y)

# Dự đoán nhãn nhị phân và xác suất
y_pred = clf.predict(X)
y_proba = clf.predict_proba(X)  # P(Y_j = 1 | X)
```