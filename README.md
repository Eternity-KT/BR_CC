# Multi-Label Classification: BR, CC, MLC-PA, and GSI-MLC-PA

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

Schema v2/legacy hỗ trợ trực tiếp các classifier bên dưới. Schema v3 bổ sung
MLP PyTorch và calibrated SVM qua shared registry; xem mục 8 để chạy ma trận
12 baseline có cấu hình khớp nhau.

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

Schema v2 lưu 5 chỉ số dự đoán đầy đủ tương thích ngược:

1. **Macro-F1 (⭐ Bắt buộc):** Trung bình điểm F1 trên từng nhãn ($F1_{macro} = \frac{1}{q}\sum_{j=1}^q F1_j$).
2. **Micro-F1:** Tính F1 gộp toàn bộ các cặp instance-label.
3. **Hamming Loss (↓ thấp = tốt):** Tỷ lệ dự đoán sai nhãn trên toàn bộ nhãn ($HL = \frac{1}{n \cdot q}\sum_{i=1}^n \sum_{j=1}^q \mathbb{I}(y_{ij} \neq \hat{y}_{ij})$).
4. **Subset Accuracy (Exact Match):** Tỷ lệ mẫu có vector nhãn dự đoán khớp hoàn toàn 100% với ground truth.
5. **Example-based F1:** F1 tính trung bình trên từng mẫu dữ liệu.

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

`requirements.txt` installs the PyTorch CUDA 13.0 wheel used by the MLP
backend. Verify the installation and GPU access with:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 6.2. Chạy Unit Tests

```bash
python tests_unit.py
```

### 6.3. Chạy thực nghiệm resumable trên toàn bộ dataset

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

---

## 7. Resumable CV and MLC-PA

The benchmark now checkpoints each model independently in
`results_pa/tables/<MODEL>.json` (for example, `MLC_PA.json`). Before loading a
dataset, the runner checks every requested model/dataset pair:

- completed pairs are read from their model cache;
- missing BR/CC-style caches are imported from the existing
  `raw_results.json` when available (including `BR_MLP` and `CC_MLP`);
- only missing pairs enter k-fold CV;
- plots and CSV tables are rebuilt from cached plus newly computed results.

`MLC_PA.json` also records the number of folds, random seed, base probability
estimator, the complete rejection-cost grid, reporting cost, and penalty. A
cache with different decision-loss settings is not reused because its partial
predictions are not comparable. One fitted probability model is reused across
all rejection costs.

`MLCPartialAbstentionClassifier` implements the Bayes-optimal generalized
Hamming-loss rules from Nguyen and Huellermeier (JAIR 2021). It is a decision
layer over marginal label probabilities:

- SEP / linear penalty: `g(a) = a*c`; decide label `k` exactly when
  `min(p_k, 1-p_k) <= c`;
- PAR / concave penalty: `g(a) = a*K*c/(K+a)`; sort marginal decision risks
  and choose the globally risk-minimizing number of decided labels;
- `predict()` returns `{0, -1, 1}`, where `-1` means abstain;
- `predict_full()` returns the corresponding complete binary prediction.

Partial predictions are reported with Generalized Loss, Selective Hamming
Loss, Selective Macro-F1, Selective Micro-F1, coverage, ABS (fraction of samples
with at least one abstention), and AABS (fraction of all label positions that
are abstained). Selective metrics discard abstained (`-1`) positions. Complete
Macro-F1 and Micro-F1 are retained separately and always come from
`predict_full()`; partial metrics never overwrite them.

In each selective-model JSON cache, the full metrics are stored under
`datasets.<dataset>.full`, while the rejection-dependent metrics are stored
separately for every operating cost under
`datasets.<dataset>.costs.<cost>.mean/std/raw_folds`. Therefore Selective
Macro-F1 and Selective Micro-F1 appear in `MLC_PA.json` and `GSI_MLC_PA.json`,
but not in the rejection-free BR/CC caches.

```bash
# Default: reuse cached BR_MLP/CC_MLP and run only missing selective models
python main.py

# SEP with BR-MLP marginal probabilities
python main.py --models BR_MLP CC_MLP MLC_PA \
  --abstention_penalty linear \
  --abstention_costs 0.2 0.25 0.3 0.35 0.4 --mlc_pa_base mlp

# PAR with logistic marginal probabilities
python main.py --models MLC_PA \
  --abstention_penalty concave \
  --abstention_costs 0.2 0.25 0.3 0.35 0.4 --mlc_pa_base logistic
```

### 7.1. GSI-MLC-PA

`GSI_MLC_PA` uses the same MLP mechanisms as `BR_MLP` and `CC_MLP`. Training,
IL/DL selection, and correlation are independent of the rejection cost:

1. Each outer CV training fold is split internally into selection-train and
   validation subsets. The outer test fold is not used during selection.
2. BR-MLP predicts direct marginal probabilities for every label without
   abstention. CC-MLP learns each label from the original features plus its
   ground-truth predecessors on selection-train.
3. A one-parent DL probability uses exact two-state marginalization:
   `(1-p_parent)*P(y=1|parent=0) + p_parent*P(y=1|parent=1)`.
4. Multiple parents use a factorized mean-field approximation by replacing
   their binary values with their already-finalized soft probabilities under
   the candidate IL/DL configuration.
5. Starting with `IL={}` and `DL=all labels`, labels are tested sequentially.
   A label remains in IL only when complete validation Macro-F1 (threshold
   `0.5`, no abstention) strictly increases; otherwise it remains in DL.
6. Only after IL/DL is frozen, Phi/Pearson label correlation is computed from
   outer-training labels and used to derive the final correlation chain order.
7. Both probability models are refit on the complete outer-training fold. The
   test probabilities are computed once.
8. BOP is then applied at each requested decision-time cost. For linear SEP,
   `p<=c -> 0`, `c<p<1-c -> abstain`, and `p>=1-c -> 1`.

The cache is `results_pa/tables/GSI_MLC_PA.json` and includes the internal
validation fraction and selection/marginalization settings. A mismatched cache
is recomputed rather than silently reused.

```bash
python main.py --models BR_MLP CC_MLP MLC_PA GSI_MLC_PA \
  --abstention_costs 0.2 0.25 0.3 0.35 0.4 \
  --report_cost 0.3 --gsi_validation_size 0.2
```

The four requested figures are written to `results_pa/plots_pa/`:

- `rejection_cost_comparison.png`;
- `selective_macro_f1_comparison.png`;
- `selective_micro_f1_comparison.png`;
- `generalized_loss_comparison.png`.

---

## 8. Schema v3: matched baselines, deployment audit và resumable folds

Schema v3 là pipeline dùng cho thí nghiệm mới; schema v2 vẫn là mặc định để
không đọc/ghi nhầm cache cũ. Mọi lệnh v3 phải truyền `--result_schema 3` và một
output directory riêng.

### 8.1. Ma trận 12 baseline chính

| Family | Logistic | PyTorch MLP | Calibrated SVM |
|---|---|---|---|
| BR | `BR_Logistic` | `BR_MLP` | `BR_SVM` |
| CC | `CC_Logistic` | `CC_MLP` | `CC_SVM` |
| MLC-PA | `MLC_PA_Logistic` | `MLC_PA_MLP` | `MLC_PA_SVM` |
| GSI-MLC-PA | `GSI_MLC_PA_Logistic` | `GSI_MLC_PA_MLP` | `GSI_MLC_PA_SVM` |

`BR_SVM`/`CC_SVM` và các selective SVM dùng nested sigmoid/Platt calibration
chỉ trong outer-training fold. Các alias `BR`/`CC` cũ vẫn là raw LinearSVC và
không thuộc ma trận primary comparison.

Schema v3 tách riêng các scope `Full`, `Selective`, `Rejected`, `Optimistic`
và `Deployment`. Ngoài complete metrics, output có coverage, risk-at-coverage,
AURC, ABS/AABS, error capture, optimistic gain, per-label/IL-DL metrics,
calibration và reviewer scenarios. Optimistic metrics là upper bound giả định
reviewer đúng 100%, không phải chất lượng tự động của model.

### 8.2. Smoke benchmark tái lập và audit tự động

Từ repository root, chạy:

```bash
python -m unittest discover -s tests -v
python tests_unit.py

python main.py --datasets emotions \
  --models BR_Logistic BR_MLP CC_Logistic CC_MLP MLC_PA_Logistic MLC_PA_MLP GSI_MLC_PA_Logistic GSI_MLC_PA_MLP BR_SVM CC_SVM MLC_PA_SVM GSI_MLC_PA_SVM \
  --n_splits 2 --random_state 42 \
  --abstention_costs 0.3 0.5 --report_cost 0.3 \
  --result_schema 3 --output_dir results_pa_v3_smoke \
  --label_policy_path configs/label_policy.json \
  --operating_point_rule min_generalized_loss

python scripts/audit_v3_results.py results_pa_v3_smoke \
  --datasets emotions --expect-primary-models --folds 2 --costs 0.3 0.5
```

Smoke chuẩn Q11 có config hash `2a8c08daeccbf1b4`. Audit kiểm tra strict JSON,
run/checkpoint hashes, đủ folds/model matrix, `Hamming Accuracy + Hamming Loss =
1`, sanity point `c=0.5`, chín CSV và bảy figures. Chạy lại cùng lệnh sẽ dùng
checkpoint hoàn thành thay vì fit lại fold.

Artifacts nằm dưới directory có config hash:

```text
results_pa_v3_smoke/
├── checkpoints/<MODEL>/emotions.<pair_hash>.json
├── tables/2a8c08daeccbf1b4/results_v3.json
├── tables/2a8c08daeccbf1b4/*.csv
└── figures/2a8c08daeccbf1b4/*.png
```

### 8.3. Label policy và chọn operating point

`configs/label_policy.json` chứa critical labels, weights, FP/FN cost, review
cost và reviewer accuracies theo dataset. File mặc định cố ý rỗng; khi domain
policy chưa được khai báo, critical metrics/utility trả `N/A` và pipeline không
suy importance từ test prevalence.

Ba rule được hỗ trợ:

- `min_generalized_loss`;
- `max_utility_at_coverage` với `--operating_coverage_gamma`;
- `max_coverage_at_risk` với `--operating_risk_epsilon`.

Cost chỉ được chọn trên inner-validation của outer-training fold. Outer-test
chỉ dùng để báo cáo sau khi operating point đã freeze.

### 8.4. Full run đã freeze và chạy theo quota

Primary config nằm ở `configs/full_run.json`, SHA-256:

```text
4036af255fcdc462f9d5307b924cc0d1a1004b02309f749ac66826a9bc61fd1e
```

Kiểm tra config mà chưa chạy model:

```bash
python scripts/run_frozen_experiment.py --config configs/full_run.json --dry-run
```

Chạy/resume theo quota bằng fold checkpoint (ví dụ tối đa 20 fold mới trong
một phiên):

```bash
python scripts/run_frozen_experiment.py \
  --config configs/full_run.json --max-new-folds 20
```

Không sửa `configs/full_run.json` sau khi bắt đầu primary run. Mọi thay đổi
scientific setting phải tạo config/checksum mới và output directory mới; tham
số `--max-new-folds` chỉ là giới hạn vận hành, không thay đổi scientific hash.
