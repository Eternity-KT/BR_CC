# Đặc Tả Cài Đặt Bộ Phân Loại Binary Relevance (BR) và Classifier Chains (CC)

> **Tài liệu tham khảo:**
> 1. Zhang, M.-L., Li, Y.-K., Liu, X.-Y., & Geng, X. (2018). *Binary relevance for multi-label learning: an overview.* Frontiers of Computer Science, 12(2), 191–202. (`FCS'17.pdf`)
> 2. Read, J., Pfahringer, B., Holmes, G., & Frank, E. (2011). *Classifier chains for multi-label classification.* Machine Learning, 85(3), 333–359. (`s10994-011-5256-5.pdf`)

---

## 1. Tổng Quan

Dự án này triển khai và so sánh 2 phương pháp phân loại đa nhãn (multi-label classification) kinh điển:

| Phương pháp | Viết tắt | Paper gốc |
|---|---|---|
| Binary Relevance | **BR** | Zhang et al. (FCS 2018) |
| Classifier Chains | **CC** | Read et al. (ML 2011) |

Mục tiêu: Đánh giá hiệu năng trên **10 tập dữ liệu benchmark** sử dụng **5-fold Stratified Cross-Validation**, sau đó trực quan hóa kết quả bằng các biểu đồ chi tiết.

---

## 2. Mô Tả Thuật Toán

### 2.1. Binary Relevance (BR)

**Nguồn:** Zhang et al. (FCS 2018), Section 2 — "Basic settings for multi-label learning"

#### Ý tưởng cốt lõi
BR phân rã bài toán phân loại đa nhãn thành `q` bài toán phân loại nhị phân độc lập, trong đó `q` là số lượng nhãn. Mỗi bộ phân loại nhị phân chỉ học cho một nhãn duy nhất, hoàn toàn **không xét quan hệ giữa các nhãn**.

#### Thuật toán

```
Algorithm: Binary Relevance (BR)
────────────────────────────────────────────────
Input:
  - D = {(x_i, Y_i)} : tập huấn luyện, mỗi x_i ∈ ℝ^d, Y_i ⊆ L = {l_1, ..., l_q}
  - x_new : mẫu cần dự đoán

Training Phase:
  FOR j = 1 TO q:
    1. Tạo tập huấn luyện nhị phân D_j:
       - Với mỗi (x_i, Y_i) ∈ D:
         - Nếu l_j ∈ Y_i → gán nhãn y_j = +1
         - Nếu l_j ∉ Y_i → gán nhãn y_j = −1
    2. Huấn luyện bộ phân loại nhị phân h_j trên D_j

Prediction Phase:
  FOR j = 1 TO q:
    ŷ_j = h_j(x_new)
  
  Output: Ŷ = {l_j | ŷ_j = +1, j = 1, ..., q}
────────────────────────────────────────────────
```

#### Đặc điểm chính (theo Zhang et al.):
- **Ưu điểm:** Đơn giản, dễ triển khai, mở rộng tốt với số lượng nhãn lớn, tối ưu tự nhiên cho macro-averaged metrics
- **Nhược điểm:** Giả định nhãn độc lập → bỏ qua tương quan nhãn (label correlation)
- **Phức tạp thời gian:** O(q × T_base), với T_base là thời gian huấn luyện base classifier

#### Phiên bản Binary Relevance với Hồi Quy Logistic (Logistic Regression)
- **Base Classifier:** `sklearn.linear_model.LogisticRegression(solver='liblinear', C=1.0, max_iter=1000, random_state=42)`
- **Mô hình xác suất:** Với mỗi nhãn $j \in \{1, \dots, q\}$, mô hình học vector trọng số $w_j$ và bias $b_j$ sao cho:
  $$P(Y_j = 1 \mid x) = \sigma(w_j^T x + b_j) = \frac{1}{1 + e^{-(w_j^T x + b_j)}}$$
- **Hàm mất mát (Binary Cross-Entropy / Log Loss):**
  $$\mathcal{L}(w_j, b_j) = -\sum_{i=1}^n \left[ y_{ij} \log P(Y_j=1 \mid x_i) + (1 - y_{ij}) \log (1 - P(Y_j=1 \mid x_i)) \right] + \frac{1}{2C} \|w_j\|_2^2$$
- **Quy tắc ra quyết định:** $\hat{y}_j = \mathbb{I}(P(Y_j = 1 \mid x) \ge 0.5) = \mathbb{I}(w_j^T x + b_j \ge 0)$.
- **Ưu điểm so với SVM:** Cung cấp xác suất hậu nghiệm $P(Y_j = 1 \mid X)$ được hiệu chuẩn (calibrated), trực tiếp phục vụ các tác vụ xếp hạng (ranking) và threshold tuning.

### 2.2. Classifier Chains (CC)

**Nguồn:** Read et al. (ML 2011), Section 3 — "Classifier Chains"

#### Ý tưởng cốt lõi
CC mở rộng BR bằng cách **xâu chuỗi (chain)** các bộ phân loại nhị phân theo một thứ tự cố định. Mỗi bộ phân loại nhận thêm **dự đoán của các nhãn trước đó trong chuỗi** làm feature bổ sung, từ đó mô hình hóa **tương quan giữa các nhãn**.

#### Thuật toán

```
Algorithm: Classifier Chains (CC)
────────────────────────────────────────────────
Input:
  - D = {(x_i, Y_i)} : tập huấn luyện
  - L = {l_1, ..., l_q} : tập nhãn theo thứ tự chuỗi
  - x_new : mẫu cần dự đoán

Training Phase:
  FOR j = 1 TO q:
    1. Tạo tập huấn luyện mở rộng D_j:
       - Với mỗi (x_i, Y_i) ∈ D:
         - Feature mở rộng: x_i^(j) = [x_i, y_i^1, y_i^2, ..., y_i^(j-1)]
           (nối thêm giá trị thực {0,1} của j-1 nhãn trước đó)
         - Nhãn: y_j = 1 nếu l_j ∈ Y_i, ngược lại y_j = 0
    2. Huấn luyện bộ phân loại nhị phân h_j trên D_j

Prediction Phase (greedy inference):
  FOR j = 1 TO q:
    ŷ_j = h_j([x_new, ŷ_1, ŷ_2, ..., ŷ_{j-1}])
    (sử dụng dự đoán ŷ của các nhãn trước, KHÔNG phải giá trị thực)

  Output: Ŷ = {l_j | ŷ_j = +1, j = 1, ..., q}
────────────────────────────────────────────────
```

#### Đặc điểm chính (theo Read et al.):
- **Ưu điểm:** Mô hình hóa tương quan nhãn, cải thiện đáng kể trên Subset Accuracy & Macro-F1 so với BR
- **Nhược điểm:** Phụ thuộc vào thứ tự chuỗi, lỗi lan truyền (error propagation) theo chuỗi
- **Thứ tự nhãn trong chuỗi:** Sử dụng **thứ tự mặc định** (default order) như trong dataset — đúng với thiết lập gốc trong paper Read et al. (2011)
- **Phức tạp thời gian:** O(q × T_base), tương đương BR (feature space tăng dần từ d đến d+q-1)

---

## 3. Thiết Lập Thí Nghiệm

### 3.1. Base Classifier

Theo chuẩn của cả hai paper:

| Thuộc tính | Giá trị |
|---|---|
| **Base classifier** | **Support Vector Machine (SVM)** — SMO (Sequential Minimal Optimization) |
| **Kernel** | Linear kernel (tương ứng `LinearSVC` hoặc `SVC(kernel='linear')` trong scikit-learn) |
| **Lý do chọn** | Cả 2 paper đều sử dụng SVM/SMO làm base classifier chính trong thí nghiệm. SVM được Zhang et al. liệt kê là lựa chọn phổ biến nhất cho BR framework, và Read et al. sử dụng SMO trong các thí nghiệm benchmark. |
| **Thư viện** | `scikit-learn` (Python) |
| **Tham số** | Mặc định (C=1.0), đúng theo chuẩn cài đặt trong paper |

**Triển khai cụ thể:**
```python
from sklearn.svm import LinearSVC

# Base classifier cho mỗi nhãn
base_classifier = LinearSVC(
    C=1.0,           # Regularization parameter mặc định
    max_iter=10000,  # Đảm bảo hội tụ cho datasets lớn
    random_state=42  # Reproducibility
)
```

### 3.2. Chiến Lược Đánh Giá: 5-Fold Stratified Cross-Validation

| Thuộc tính | Giá trị |
|---|---|
| **Số folds** | **5** |
| **Stratification** | Sử dụng **Iterative Stratification** cho multi-label (thư viện `scikit-multilearn`) để đảm bảo phân bố nhãn đồng đều giữa các fold |
| **Seed/Random state** | `random_state=42` cho reproducibility |
| **Report** | Mean ± Std trên 5 folds cho mỗi metric |

**Lý do chọn Iterative Stratification:**
- Standard `StratifiedKFold` chỉ hỗ trợ single-label
- Iterative Stratification (Sechidis et al., 2011) giữ cân bằng phân bố label trên tất cả folds — chuẩn trong multi-label research

```python
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

cv = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

---

## 4. Tập Dữ Liệu

### 4.1. Tổng Quan 10 Datasets

| # | Dataset | Domain | Samples | Features | Labels | Label Cardinality | File format |
|---|---|---|---|---|---|---|---|
| 1 | **Emotions** | Music | 593 | 72 | 6 | 1.87 | MULAN (XML + ARFF) |
| 2 | **Music** | Music | 593 | 72 | 6 | 1.87 | Single ARFF (labels ở đầu) |
| 3 | **Scene** | Image | 2,407 | 294 | 6 | 1.07 | Single ARFF |
| 4 | **Yeast** | Biology | 2,417 | 103 | 14 | 4.24 | Single ARFF |
| 5 | **CAL500** | Music | 502 | 68 | 174 | 26.04 | Single ARFF |
| 6 | **Bibtex** | Text | 7,395 | 1,836 | 159 | 2.40 | MULAN (XML + ARFF) |
| 7 | **Enron** | Text | 1,702 | 1,001 | 53 | 3.38 | MULAN (XML + ARFF) |
| 8 | **Genbase** | Biology | 662 | 1,186 | 27 | 1.25 | MULAN (XML + ARFF) |
| 9 | **Medical** | Text | 978 | 1,449 | 45 | 1.25 | MULAN (XML + ARFF) |
| 10 | **REUTERS-K500** | Text | 6,000 | 500 | 103 | ~1.46 | Single ARFF (labels ở đầu) |

> **Lưu ý:** Cột "Label Cardinality" = trung bình số nhãn mỗi mẫu — giá trị tham khảo, sẽ được tính tự động bởi data loader.

### 4.2. Vị Trí File Dữ Liệu

```
data/
├── emotions/
│   ├── emotions.arff          ← Dữ liệu chính (labels ở cuối, 6 nhãn cuối)
│   └── emotions.xml           ← MULAN label descriptor
├── Music.arff                 ← Labels ở ĐẦU file (6 nhãn đầu)
├── Scene.arff                 ← Labels ở CUỐI file (6 nhãn cuối)
├── Yeast.arff                 ← Labels ở CUỐI file (14 nhãn cuối)
├── CAL500.arff                ← Labels ở CUỐI file (174 nhãn cuối)
├── bibtex/
│   ├── bibtex.arff
│   └── bibtex.xml
├── enron/
│   ├── enron.arff
│   └── enron.xml
├── genbase/
│   ├── genbase.arff
│   └── genbase.xml
├── medical/
│   ├── medical.arff
│   └── medical.xml
└── REUTERS-K500-EX2.arff      ← Labels ở ĐẦU file (103 nhãn đầu)
```

### 4.3. Data Loading Logic

Do các dataset có **vị trí nhãn khác nhau** (đầu vs cuối file), cần một data loader linh hoạt:

```python
# Cấu hình cho mỗi dataset
DATASET_CONFIG = {
    "emotions": {
        "path": "data/emotions/emotions.arff",
        "xml_path": "data/emotions/emotions.xml",  # Dùng XML để xác định label
        "label_location": "end",
        "num_labels": 6
    },
    "music": {
        "path": "data/Music.arff",
        "xml_path": None,
        "label_location": "start",    # Labels nằm ở 6 cột đầu
        "num_labels": 6
    },
    "scene": {
        "path": "data/Scene.arff",
        "xml_path": None,
        "label_location": "end",
        "num_labels": 6
    },
    "yeast": {
        "path": "data/Yeast.arff",
        "xml_path": None,
        "label_location": "end",
        "num_labels": 14
    },
    "cal500": {
        "path": "data/CAL500.arff",
        "xml_path": None,
        "label_location": "end",
        "num_labels": 174
    },
    "bibtex": {
        "path": "data/bibtex/bibtex.arff",
        "xml_path": "data/bibtex/bibtex.xml",
        "label_location": "end",
        "num_labels": 159
    },
    "enron": {
        "path": "data/enron/enron.arff",
        "xml_path": "data/enron/enron.xml",
        "label_location": "end",
        "num_labels": 53
    },
    "genbase": {
        "path": "data/genbase/genbase.arff",
        "xml_path": "data/genbase/genbase.xml",
        "label_location": "end",
        "num_labels": 27
    },
    "medical": {
        "path": "data/medical/medical.arff",
        "xml_path": "data/medical/medical.xml",
        "label_location": "end",
        "num_labels": 45
    },
    "reuters-k500": {
        "path": "data/REUTERS-K500-EX2.arff",
        "xml_path": None,
        "label_location": "start",   # Labels nằm ở 103 cột đầu
        "num_labels": 103
    }
}
```

#### Quy trình load dữ liệu:
1. **Parse ARFF** → Đọc attributes và data
2. **Xác định nhãn:**
   - Nếu có file `.xml` (MULAN format) → parse XML để lấy tên nhãn → tìm vị trí tương ứng trong ARFF
   - Nếu không có XML → dựa vào `label_location` và `num_labels` để tách feature/label
3. **Tách X (features) và Y (labels):**
   - `label_location="end"` → `Y = data[:, -num_labels:]`, `X = data[:, :-num_labels]`
   - `label_location="start"` → `Y = data[:, :num_labels]`, `X = data[:, num_labels:]`
4. **Xử lý missing values:** Thay NaN bằng 0 (nếu có)
5. **Đảm bảo Y ∈ {0, 1}** (binary matrix)

---

## 5. Metrics Đánh Giá

### 5.1. Bảng Metrics

Dựa trên chuẩn đánh giá trong cả hai paper, sử dụng bộ metrics sau:

| # | Metric | Loại | Hướng | Ký hiệu trong paper | Mô tả |
|---|---|---|---|---|---|
| 1 | **Macro-F1** ⭐ | Label-based | ↑ cao = tốt | Macro F-measure | Trung bình F1 trên tất cả nhãn (bắt buộc) |
| 2 | **Micro-F1** | Label-based | ↑ cao = tốt | Micro F-measure | F1 tính gộp trên tất cả nhãn |
| 3 | **Hamming Loss** | Example-based | ↓ thấp = tốt | HL | Tỷ lệ nhãn bị dự đoán sai |
| 4 | **Subset Accuracy** | Example-based | ↑ cao = tốt | Accuracy / 0-1 Loss | Tỷ lệ mẫu có tập nhãn dự đoán khớp hoàn toàn |
| 5 | **Example-based F1** | Example-based | ↑ cao = tốt | — | F1 tính trung bình trên từng mẫu |
| 6 | **Macro Precision** | Label-based | ↑ cao = tốt | — | Trung bình precision trên tất cả nhãn |
| 7 | **Macro Recall** | Label-based | ↑ cao = tốt | — | Trung bình recall trên tất cả nhãn |

> ⭐ **Macro-F1 là metric bắt buộc** theo yêu cầu.

### 5.2. Công Thức Chi Tiết

#### Macro-F1 (Bắt buộc)
```
Cho mỗi nhãn j (j = 1, ..., q):
  TP_j = số mẫu có y_j = 1 VÀ ŷ_j = 1
  FP_j = số mẫu có y_j = 0 VÀ ŷ_j = 1
  FN_j = số mẫu có y_j = 1 VÀ ŷ_j = 0

  Precision_j = TP_j / (TP_j + FP_j)
  Recall_j    = TP_j / (TP_j + FN_j)
  F1_j        = 2 × Precision_j × Recall_j / (Precision_j + Recall_j)

  Macro-F1 = (1/q) × Σ_{j=1}^{q} F1_j
```

#### Micro-F1
```
TP_total = Σ_{j=1}^{q} TP_j
FP_total = Σ_{j=1}^{q} FP_j
FN_total = Σ_{j=1}^{q} FN_j

Precision_micro = TP_total / (TP_total + FP_total)
Recall_micro    = TP_total / (TP_total + FN_total)
Micro-F1        = 2 × Precision_micro × Recall_micro / (Precision_micro + Recall_micro)
```

#### Hamming Loss
```
Hamming Loss = (1 / (n × q)) × Σ_{i=1}^{n} Σ_{j=1}^{q} I(y_ij ≠ ŷ_ij)
```

#### Subset Accuracy (Exact Match Ratio)
```
Subset Accuracy = (1/n) × Σ_{i=1}^{n} I(Y_i = Ŷ_i)
```

#### Example-based F1
```
Cho mỗi mẫu i:
  F1_i = 2 × |Y_i ∩ Ŷ_i| / (|Y_i| + |Ŷ_i|)

Example-F1 = (1/n) × Σ_{i=1}^{n} F1_i
```

### 5.3. Triển Khai Metrics (scikit-learn)

```python
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    accuracy_score,
    precision_score,
    recall_score
)
import numpy as np

def compute_all_metrics(y_true, y_pred):
    """
    Tính toàn bộ metrics cho multi-label classification.
    
    Parameters:
        y_true: np.ndarray, shape (n_samples, n_labels), ground truth
        y_pred: np.ndarray, shape (n_samples, n_labels), predictions
    
    Returns:
        dict: Tên metric → giá trị
    """
    results = {}
    
    # 1. Macro-F1 (BẮT BUỘC)
    results["Macro-F1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    
    # 2. Micro-F1
    results["Micro-F1"] = f1_score(y_true, y_pred, average="micro", zero_division=0)
    
    # 3. Hamming Loss
    results["Hamming Loss"] = hamming_loss(y_true, y_pred)
    
    # 4. Subset Accuracy (Exact Match)
    results["Subset Accuracy"] = accuracy_score(y_true, y_pred)
    
    # 5. Example-based F1
    n_samples = y_true.shape[0]
    example_f1s = []
    for i in range(n_samples):
        if y_true[i].sum() == 0 and y_pred[i].sum() == 0:
            example_f1s.append(1.0)
        elif y_true[i].sum() == 0 or y_pred[i].sum() == 0:
            example_f1s.append(0.0)
        else:
            intersection = np.logical_and(y_true[i], y_pred[i]).sum()
            example_f1s.append(
                2 * intersection / (y_true[i].sum() + y_pred[i].sum())
            )
    results["Example-F1"] = np.mean(example_f1s)
    
    # 6. Macro Precision
    results["Macro Precision"] = precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    # 7. Macro Recall
    results["Macro Recall"] = recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    return results
```

---

## 6. Quy Trình Thí Nghiệm (Pipeline)

### 6.1. Workflow Tổng Thể

```
┌──────────────────────────────────────────────────────────────────────┐
│                         MAIN PIPELINE                                │
│                                                                      │
│  FOR each dataset in [emotions, music, scene, yeast, cal500,         │
│                        bibtex, enron, genbase, medical, reuters]:     │
│    ┌──────────────────────────────────────────────────────────────┐   │
│    │ 1. Load data (X, Y) từ ARFF (+XML nếu có)                  │   │
│    │ 2. Print dataset stats (samples, features, labels)          │   │
│    │ 3. Create 5-fold MultilabelStratifiedKFold                  │   │
│    │                                                              │   │
│    │ FOR each fold k = 1..5:                                      │   │
│    │   ┌──────────────────────────────────────────────────────┐   │   │
│    │   │ a. Split: X_train, X_test, Y_train, Y_test           │   │   │
│    │   │ b. Train BR classifier → predict → compute metrics   │   │   │
│    │   │ c. Train CC classifier → predict → compute metrics   │   │   │
│    │   │ d. Store results per fold                             │   │   │
│    │   └──────────────────────────────────────────────────────┘   │   │
│    │                                                              │   │
│    │ 4. Aggregate: Mean ± Std across 5 folds                     │   │
│    │ 5. Store aggregated results for this dataset                 │   │
│    └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Generate comparison charts & tables                                 │
└──────────────────────────────────────────────────────────────────────┘
```

### 6.2. Chi Tiết Triển Khai BR

```python
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC

class BinaryRelevanceClassifier:
    """
    Binary Relevance: q bộ phân loại nhị phân độc lập.
    Tuân thủ: Zhang et al. (FCS 2018)
    """
    def __init__(self, base_classifier=None):
        if base_classifier is None:
            base_classifier = LinearSVC(C=1.0, max_iter=10000, random_state=42)
        self.model = MultiOutputClassifier(base_classifier)
    
    def fit(self, X, Y):
        self.model.fit(X, Y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
```

### 6.3. Chi Tiết Triển Khai CC

```python
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.base import clone

class ClassifierChainClassifier:
    """
    Classifier Chains: chuỗi bộ phân loại nhị phân có truyền dự đoán.
    Tuân thủ: Read et al. (ML 2011), Section 3
    
    - Thứ tự chuỗi: default order (theo thứ tự cột label trong dataset)
    - Inference: greedy (dùng hard prediction ŷ, không dùng probability)
    """
    def __init__(self, base_classifier=None, order=None):
        if base_classifier is None:
            base_classifier = LinearSVC(C=1.0, max_iter=10000, random_state=42)
        self.base_classifier = base_classifier
        self.order = order  # None → default order
        self.classifiers = []
    
    def fit(self, X, Y):
        n_labels = Y.shape[1]
        if self.order is None:
            self.order = list(range(n_labels))
        
        self.classifiers = []
        for i, label_idx in enumerate(self.order):
            clf = clone(self.base_classifier)
            
            # Nối features gốc + dự đoán các nhãn trước đó trong chuỗi
            if i == 0:
                X_extended = X
            else:
                # Dùng giá trị THỰC (ground truth) trong training
                prev_labels = Y[:, self.order[:i]]
                X_extended = np.hstack([X, prev_labels])
            
            clf.fit(X_extended, Y[:, label_idx])
            self.classifiers.append(clf)
        
        return self
    
    def predict(self, X):
        n_samples = X.shape[0]
        n_labels = len(self.order)
        predictions = np.zeros((n_samples, n_labels))
        
        for i, (label_idx, clf) in enumerate(zip(self.order, self.classifiers)):
            if i == 0:
                X_extended = X
            else:
                # Dùng DỰ ĐOÁN (predicted) của các nhãn trước trong inference
                prev_preds = predictions[:, self.order[:i]]
                X_extended = np.hstack([X, prev_preds])
            
            predictions[:, label_idx] = clf.predict(X_extended)
        
        return predictions
```

### 6.4. Main Experiment Loop

```python
import pandas as pd

def run_experiment():
    all_results = {}
    
    for dataset_name, config in DATASET_CONFIG.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
        
        # 1. Load data
        X, Y = load_dataset(config)
        print(f"  Samples: {X.shape[0]}, Features: {X.shape[1]}, Labels: {Y.shape[1]}")
        
        # 2. Setup CV
        cv = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        fold_results = {"BR": [], "CC": []}
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, Y)):
            print(f"  Fold {fold_idx + 1}/5...")
            
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            
            # 3a. BR
            br = BinaryRelevanceClassifier()
            br.fit(X_train, Y_train)
            y_pred_br = br.predict(X_test)
            metrics_br = compute_all_metrics(Y_test, y_pred_br)
            fold_results["BR"].append(metrics_br)
            
            # 3b. CC
            cc = ClassifierChainClassifier()
            cc.fit(X_train, Y_train)
            y_pred_cc = cc.predict(X_test)
            metrics_cc = compute_all_metrics(Y_test, y_pred_cc)
            fold_results["CC"].append(metrics_cc)
        
        # 4. Aggregate results
        all_results[dataset_name] = {}
        for method in ["BR", "CC"]:
            df_folds = pd.DataFrame(fold_results[method])
            all_results[dataset_name][method] = {
                "mean": df_folds.mean().to_dict(),
                "std": df_folds.std().to_dict()
            }
    
    return all_results
```

---

## 7. Trực Quan Hóa (Visualization)

### 7.1. Danh Sách Biểu Đồ Bắt Buộc

| # | Loại biểu đồ | Mô tả | Tên file |
|---|---|---|---|
| 1 | **Grouped Bar Chart** | So sánh BR vs CC trên Macro-F1 cho 10 datasets | `macro_f1_comparison.png` |
| 2 | **Grouped Bar Chart** | So sánh BR vs CC trên Micro-F1 cho 10 datasets | `micro_f1_comparison.png` |
| 3 | **Grouped Bar Chart** | So sánh BR vs CC trên Hamming Loss cho 10 datasets | `hamming_loss_comparison.png` |
| 4 | **Grouped Bar Chart** | So sánh BR vs CC trên Subset Accuracy cho 10 datasets | `subset_accuracy_comparison.png` |
| 5 | **Grouped Bar Chart** | So sánh BR vs CC trên Example-F1 cho 10 datasets | `example_f1_comparison.png` |
| 6 | **Grouped Bar Chart** | So sánh BR vs CC trên Macro Precision cho 10 datasets | `macro_precision_comparison.png` |
| 7 | **Grouped Bar Chart** | So sánh BR vs CC trên Macro Recall cho 10 datasets | `macro_recall_comparison.png` |
| 8 | **Radar/Spider Chart** | So sánh tổng thể BR vs CC trên tất cả metrics (trung bình qua 10 datasets) | `radar_overall.png` |
| 9 | **Heatmap** | Ma trận metrics × datasets cho BR | `heatmap_br.png` |
| 10 | **Heatmap** | Ma trận metrics × datasets cho CC | `heatmap_cc.png` |
| 11 | **Summary Table** | Bảng tổng hợp Mean±Std cho tất cả metrics trên tất cả datasets | `summary_table.png` |

### 7.2. Yêu Cầu Thiết Kế Biểu Đồ

#### Grouped Bar Chart (Biểu đồ 1–7):
```
- Trục X: 10 dataset names (xoay 45° nếu cần)
- Trục Y: Giá trị metric
- 2 bars cho mỗi dataset: BR (màu xanh dương) vs CC (màu cam)
- Error bars: ±1 standard deviation (5 folds)
- Title: tên metric
- Legend: "BR", "CC"
- Grid: bật trên trục Y
- Ghi giá trị trung bình lên đỉnh mỗi bar (font nhỏ)
- DPI: 300
- Size: 14×6 inches
```

#### Radar Chart (Biểu đồ 8):
```
- Các trục: 7 metrics
- 2 đường: BR (xanh dương, nét liền) vs CC (cam, nét đứt)
- Fill alpha: 0.25
- Normalize tất cả metrics về [0, 1] (đảo Hamming Loss vì lower = better)
```

#### Heatmap (Biểu đồ 9–10):
```
- Rows: 10 datasets
- Columns: 7 metrics
- Color: viridis colormap
- Annotate: giá trị mean (2 chữ số thập phân)
- Title: "BR Performance Heatmap" / "CC Performance Heatmap"
```

### 7.3. Code Template Trực Quan Hóa

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_grouped_bar(all_results, metric_name, output_path):
    """
    Vẽ grouped bar chart so sánh BR vs CC trên một metric cụ thể.
    """
    datasets = list(all_results.keys())
    br_means = [all_results[d]["BR"]["mean"][metric_name] for d in datasets]
    cc_means = [all_results[d]["CC"]["mean"][metric_name] for d in datasets]
    br_stds = [all_results[d]["BR"]["std"][metric_name] for d in datasets]
    cc_stds = [all_results[d]["CC"]["std"][metric_name] for d in datasets]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, br_means, width, yerr=br_stds,
                   label='BR', color='#4A90D9', capsize=3, edgecolor='white')
    bars2 = ax.bar(x + width/2, cc_means, width, yerr=cc_stds,
                   label='CC', color='#E8854A', capsize=3, edgecolor='white')
    
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(f'{metric_name}: BR vs CC Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Ghi giá trị lên bar
    for bar, mean_val in zip(bars1, br_means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=7)
    for bar, mean_val in zip(bars2, cc_means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=7)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

# Gọi cho từng metric
METRICS = [
    "Macro-F1", "Micro-F1", "Hamming Loss", "Subset Accuracy",
    "Example-F1", "Macro Precision", "Macro Recall"
]

for metric in METRICS:
    filename = metric.lower().replace("-", "_").replace(" ", "_")
    plot_grouped_bar(all_results, metric, f"results/{filename}_comparison.png")
```

---

## 8. Cấu Trúc Thư Mục Dự Án

```
BR_CC/
├── README.md
├── specify.md                      ← Tài liệu đặc tả này
├── requirements.txt                ← Dependencies
├── TaiLieuThamKhao/
│   ├── FCS'17.pdf                  ← Paper BR (Zhang et al.)
│   └── s10994-011-5256-5.pdf       ← Paper CC (Read et al.)
├── data/                           ← 10 datasets
│   ├── emotions/
│   ├── Music.arff
│   ├── Scene.arff
│   ├── Yeast.arff
│   ├── CAL500.arff
│   ├── bibtex/
│   ├── enron/
│   ├── genbase/
│   ├── medical/
│   └── REUTERS-K500-EX2.arff
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py               ← Data loader (ARFF + XML parser)
│   ├── models/
│   │   ├── __init__.py
│   │   ├── binary_relevance.py     ← BR implementation
│   │   └── classifier_chain.py     ← CC implementation
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py              ← Tất cả metrics
│   └── visualization/
│       ├── __init__.py
│       └── plots.py                ← Tất cả biểu đồ
├── main.py                         ← Entry point: chạy toàn bộ pipeline
└── results/                        ← Output directory
    ├── tables/
    │   └── summary_results.csv     ← Bảng tổng hợp kết quả
    └── figures/
        ├── macro_f1_comparison.png
        ├── micro_f1_comparison.png
        ├── hamming_loss_comparison.png
        ├── subset_accuracy_comparison.png
        ├── example_f1_comparison.png
        ├── macro_precision_comparison.png
        ├── macro_recall_comparison.png
        ├── radar_overall.png
        ├── heatmap_br.png
        ├── heatmap_cc.png
        └── summary_table.png
```

---

## 9. Dependencies (requirements.txt)

```
scikit-learn>=1.3.0
scikit-multilearn>=0.2.0
iterative-stratification>=0.1.7
numpy>=1.24.0
pandas>=2.0.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
liac-arff>=2.5.0
```

---

## 10. Bảng Tóm Tắt Thiết Lập Thí Nghiệm

| Thuộc tính | Giá trị |
|---|---|
| Phương pháp so sánh | BR, CC |
| Base classifier | LinearSVC (C=1.0, linear kernel) |
| Cross-validation | 5-fold Iterative Stratified |
| Random seed | 42 |
| Số datasets | 10 |
| Số metrics | 7 (Macro-F1 bắt buộc) |
| Số biểu đồ | 11 |
| Ngôn ngữ | Python 3.10+ |
| Thư viện chính | scikit-learn, matplotlib, seaborn |
| Format dữ liệu | ARFF (MULAN compatible) |
| Output format | PNG (300 DPI) + CSV |

---

## 11. Ghi Chú Quan Trọng

### 11.1. Khác biệt giữa Training và Prediction trong CC
- **Training:** Sử dụng **giá trị thực** (ground truth labels) của các nhãn trước → đảm bảo không bị ảnh hưởng bởi lỗi dự đoán
- **Prediction:** Sử dụng **dự đoán** (predicted labels) của các nhãn trước → phản ánh hoạt động thực tế của mô hình (greedy inference)

### 11.2. Xử Lý Zero Division
- Khi tính F1 cho nhãn hiếm (rare label), có thể xảy ra TP=FP=FN=0 → F1 = 0 (zero_division=0)
- Đặc biệt quan trọng với CAL500 (174 labels) và REUTERS-K500 (103 labels)

### 11.3. Reproducibility
- Cố định `random_state=42` cho:
  - Base classifier (LinearSVC)
  - Cross-validation splitter (MultilabelStratifiedKFold)
- Ghi lại phiên bản thư viện trong `requirements.txt`

### 11.4. Thứ Tự Chuỗi trong CC
- Paper gốc (Read et al. 2011) sử dụng **default order** (thứ tự tự nhiên của nhãn trong dataset)
- Trong đặc tả này, chúng ta tuân thủ đúng paper gốc → dùng default order
- Không sử dụng random order hay heuristic order

### 11.5. Hiệu Năng Tính Toán
- Datasets lớn (bibtex: 7,395 samples × 1,836 features × 159 labels) có thể mất thời gian đáng kể
- LinearSVC nhanh hơn nhiều so với SVC(kernel='rbf') → phù hợp cho thí nghiệm quy mô lớn
- Cân nhắc thêm progress bar (`tqdm`) để theo dõi tiến trình
