# Đặc tả cải tiến sau buổi họp với giảng viên hướng dẫn

> Nguồn yêu cầu: [`tmp.md`](tmp.md)
>
> Phạm vi: đặc tả công việc cần làm, chưa phải kết quả thực nghiệm.
> Nguyên tắc chính: **giữ nguyên core BR, CC và cơ chế sinh xác suất của GSI-MLC-PA; các decision rule, objective, metric, calibration và phân tích mới phải được bổ sung dưới dạng module có thể bật/tắt.**

## 1. Mục tiêu và tiêu chí thành công

Sau cải tiến, thí nghiệm phải trả lời được bốn câu hỏi khác nhau, không dùng riêng Macro-F1 để thay cho cả bốn:

1. Mô hình dự đoán đầy đủ tốt đến đâu khi **bắt buộc phải quyết định ngay**?
2. Trên các vị trí mô hình chấp nhận dự đoán, chất lượng tăng bao nhiêu và đổi lại phải từ chối bao nhiêu?
3. Các vị trí bị từ chối có thật sự tập trung lỗi và các nhãn quan trọng hay không?
4. Việc chia nhãn thành IL/DL có tạo ra lợi ích riêng, sau khi đã kiểm soát base learner, fold, siêu tham số, chain order và decision policy hay không?

Chỉ được kết luận mô hình có tiềm năng ứng dụng thực tế khi đồng thời có bằng chứng về chất lượng, coverage/khối lượng cần duyệt, khả năng bắt lỗi, nhãn quan trọng và chi phí. Việc Selective Macro-F1 tăng trong khi coverage giảm **không đủ** để đưa ra kết luận này.

## 2. Kết luận nghiên cứu dùng để chốt thiết kế

### 2.1. BOP, quy hoạch động và MDP là ba khái niệm khác nhau

- MLC với partial abstention hiện tại là một bài toán quyết định Bayes: mô hình sinh xác suất, sau đó decision rule chọn `0`, `1` hoặc `abstain` để tối ưu expected utility/loss. Đây là kiến trúc hai tầng đúng với Nguyen và Hüllermeier ([paper trong repo](TaiLieuThamKhao/sminton,+12610-Article+(PDF)-28712-1-11-20211029.pdf), [DOI](https://doi.org/10.1613/jair.1.12610)).
- BOP cho generalized Hamming loss có thể dùng threshold như code hiện tại. BOP cho instance-based F-measure và Jaccard **không tương đương** với threshold `0.5` hoặc BOP Hamming; paper đưa ra thuật toán quy hoạch động độ phức tạp `O(K^3)` dưới giả định conditional label independence.
- “Quy hoạch động” trong các thuật toán F1/Jaccard không phải “Markov Decision Process”. MDP cần có state, action, transition và reward qua nhiều bước. Pipeline hiện tại chỉ ra quyết định một lần, không quan sát phản hồi mới giữa các nhãn, vì vậy chưa có transition thực tế để biện minh cho MDP.
- MDP chỉ trở nên phù hợp nếu quy trình triển khai là tuần tự, ví dụ: mô hình chọn một nhãn để dự đoán/yêu cầu người duyệt, nhận nhãn thật, cập nhật posterior của các nhãn còn lại rồi tiếp tục. Các nghiên cứu dùng reinforcement learning cho thứ tự nhãn động cũng giả định một quá trình tuần tự như vậy ([Nam et al., ICML 2019](https://proceedings.mlr.press/v97/nam19a.html)).
- Với bài toán biên hóa xác suất trong classifier chain, lựa chọn đúng để nghiên cứu trước là: exact enumeration ở số cha nhỏ, mean-field hiện tại và Monte Carlo/beam search. PCC biểu diễn joint probability bằng chain rule; exact inference tăng theo `2^K`, còn Monte Carlo là hướng xấp xỉ đã được nghiên cứu cho classifier chain ([Read, Martino và Luengo](https://arxiv.org/abs/1211.2190)).

**Quyết định thiết kế:** triển khai BOP F1/Jaccard dưới dạng decision-policy module ở P1. MDP chỉ là research spike P2 và không được nối vào core model trước khi thỏa gate ở mục 7.

### 2.2. “Instant base F1” cần được chuẩn hóa thuật ngữ

Thuật ngữ chuẩn trong multi-label classification là **instance-based F1** (còn gọi example-based F1), không phải “instant-based F1”. `Example-F1` hiện có trong repo chính là metric này.

Để vẫn bao phủ cách hiểu “cần quyết định ngay lập tức”, báo cáo sẽ dùng hai chế độ rõ ràng:

- **Immediate/automatic mode:** dùng complete prediction, không cho phép `-1`; metric chính là `Instance-F1 (Immediate)`.
- **Human-assisted mode:** cho phép từ chối và điền nhãn bị từ chối bằng reviewer/oracle; metric là `Optimistic Instance-F1` cùng coverage và review load.

Trước khi nộp báo cáo cần xác nhận lại với giảng viên rằng cụm trong note đúng là “instance-based F1”. Trong code và bảng kết quả chỉ dùng tên chuẩn; có thể giữ `Example-F1` làm alias tương thích ngược.

### 2.3. Không được tối ưu Selective Hamming Accuracy một cách không ràng buộc

Với complete prediction, tối đa hóa Hamming Accuracy tương đương tối thiểu hóa Hamming Loss. Với partial prediction, tối đa hóa accuracy chỉ trên phần đã quyết định có nghiệm suy biến là từ chối gần như tất cả. Vì vậy:

- báo cáo dùng Hamming Accuracy để dễ đọc;
- code vẫn giữ Hamming Loss/Generalized Loss ở tầng decision vì công thức BOP được định nghĩa theo loss;
- mọi tối ưu selective accuracy phải đi kèm `coverage >= gamma`, review budget, hoặc abstention cost;
- phải báo cáo risk–coverage curve, không chọn mô hình bằng một điểm selective metric đơn lẻ. Đây là cách đánh giá chuẩn của selective classification ([Geifman và El-Yaniv, NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html)).

### 2.4. Macro-F1 phải khóa đúng định nghĩa

Trong toàn bộ code và báo cáo, Macro-F1 được định nghĩa là **trung bình số học của F1 dương trên từng nhãn**:

```text
F1_k = 2 TP_k / (2 TP_k + FP_k + FN_k)
Macro-F1 = (1/K) * sum_k F1_k
```

Không tính bằng harmonic mean của Macro-Precision và Macro-Recall vì hai cách có thể cho kết quả và thứ hạng mô hình khác nhau ([Opitz và Burst](https://arxiv.org/abs/1911.03347)). Quy ước `zero_division=0` phải được ghi rõ.

## 3. Audit hiện trạng và khoảng trống cần sửa

| Hạng mục | Hiện trạng | Khoảng trống/rủi ro |
|---|---|---|
| Complete metrics | Có Macro-F1, Micro-F1, Hamming Loss, Subset Accuracy, Example-F1 | Thiếu Hamming Accuracy, Jaccard, Macro Precision/Recall và per-label metrics |
| Partial metrics | Có generalized loss, selective F1, coverage, ABS/AABS | Chưa có optimistic/oracle completion, rejected-set audit, error capture, critical-label metrics |
| GSI IL/DL selection | Hard-code complete Macro-F1 sau threshold `0.5` | Chưa áp dụng BOP instance-F1/Jaccard trong bước chọn IL/DL; chưa ablation objective |
| Baseline | Runner mặc định chỉ chạy BR-MLP, CC-MLP, MLC-PA và GSI-MLC-PA | Chưa có ma trận đầy đủ cùng base learner; MLC-PA CLI mặc định logistic nhưng GSI cố định MLP |
| Hyperparameter | BR và CC có factory trùng lặp | Dễ lệch cấu hình; fallback MLP im lặng có thể thay backend/kiến trúc |
| SVM probability | Dùng sigmoid trực tiếp trên `decision_function` | Không phải xác suất đã calibration; không đủ tin cậy cho abstention threshold |
| Cache | Schema v2; cố ý loại Macro Precision/Recall | Metric mới có thể bị mất hoặc dùng nhầm cache cũ |
| Plot/report | Dùng Hamming Loss, so selective F1 ở một số cost | Chưa thể hiện trade-off coverage, optimistic upper bound, IL/DL contribution và nhãn quan trọng |
| Báo cáo cuộc họp | Chỉ có note thô `tmp.md` | Chưa có file tóm tắt cuộc họp theo yêu cầu |

## 4. Quy ước metric bắt buộc

Ký hiệu: `Y` là ground truth, `Y_full` là complete prediction, `Y_pa` thuộc `{0, -1, 1}`, `D` là mask đã quyết định và `A = not D` là mask từ chối.

### 4.1. Complete/immediate metrics

| Tên output chuẩn | Công thức/diễn giải | Mục đích |
|---|---|---|
| `Macro-F1` | Trung bình F1 trên từng nhãn | Cân bằng ảnh hưởng giữa nhãn phổ biến và nhãn hiếm |
| `Micro-F1` | Gộp TP/FP/FN trên toàn ma trận | Hiệu suất tổng thể theo label-position |
| `Hamming Accuracy` | `1 - Hamming Loss` | Tỷ lệ label-position đúng, dễ đọc hơn loss |
| `Subset Accuracy` | Tỷ lệ instance có toàn bộ vector nhãn đúng | Đánh giá yêu cầu exact match, rất nghiêm ngặt |
| `Instance-F1` | Trung bình `2TP_i/(2TP_i+FP_i+FN_i)` theo instance | Chất lượng tập nhãn của từng quyết định ngay lập tức |
| `Instance Jaccard` | Trung bình `TP_i/(TP_i+FP_i+FN_i)` theo instance | Mức giao/ hợp của hai tập nhãn |
| `Macro Precision` | Trung bình precision từng nhãn | Kiểm soát false positive |
| `Macro Recall` | Trung bình recall từng nhãn | Kiểm soát false negative |

Quy ước khi cả tập nhãn thật và dự đoán của một instance đều rỗng: `Instance-F1 = 1` và `Instance Jaccard = 1`. Mọi bảng phải có support/prevalence để tránh diễn giải Hamming Accuracy cao do quá nhiều nhãn âm.

### 4.2. Partial/selective metrics

| Tên output chuẩn | Định nghĩa | Quy ước biên |
|---|---|---|
| `Coverage` | `sum(D)/(N*K)` | `AABS = 1 - Coverage` |
| `ABS` | Tỷ lệ instance có ít nhất một abstention | Giữ để đo số hồ sơ cần can thiệp |
| `Selective Hamming Accuracy` | Số dự đoán đúng trên `D` chia `sum(D)` | `NaN` nếu không có quyết định; không trả `1.0` |
| `Selective Macro-F1` | Tính F1 từng nhãn chỉ trên vị trí đã quyết định | Nhãn không có quyết định đóng góp `0`, đồng thời phải xuất per-label coverage |
| `Selective Micro-F1` | Micro-F1 trên tất cả vị trí đã quyết định | Trả `0` nếu all-abstain để không thưởng nghiệm suy biến |
| `Generalized Loss` | Lỗi trên phần quyết định + penalty abstention | Metric chính để đánh giá policy BOP Hamming |
| `Risk at Coverage` | `1 - Selective Hamming Accuracy` tại coverage xác định | So sánh các policy ở cùng workload |
| `AURC` | Diện tích dưới risk–coverage curve | Thấp hơn tốt hơn; không thay cho các operating point |

Selective metrics không được ghi đè complete metrics. Tên metric phải chứa `Full`, `Selective`, `Rejected` hoặc `Optimistic` khi xuất bảng để tránh so sánh sai mẫu số.

### 4.3. Optimistic/oracle metrics

Giả định reviewer gán lại mọi nhãn bị từ chối và đúng 100%:

```text
Y_oracle[i,k] = Y[i,k]       nếu Y_pa[i,k] == -1
                Y_pa[i,k]    nếu mô hình đã quyết định
```

Từ `Y_oracle`, tính lại toàn bộ complete metrics, ít nhất gồm:

- `Optimistic Macro-F1`;
- `Optimistic Micro-F1`;
- `Optimistic Hamming Accuracy`;
- `Optimistic Instance-F1`;
- `Optimistic Instance Jaccard`.

Đây là **upper bound của hệ human-in-the-loop**, không phải điểm tự động của mô hình. All-abstain sẽ trở thành perfect oracle completion; phần lớn accuracy/set metrics bằng `1`. Riêng positive-class Macro-F1 có thể nhỏ hơn `1` nếu một fold có nhãn không xuất hiện dương và quy ước `zero_division=0`. Vì vậy mỗi optimistic score bắt buộc đi cùng `Coverage`, `AABS`, `ABS`, support và generalized cost. Không dùng optimistic score làm objective duy nhất hoặc để xếp hạng mô hình.

Các delta cần xuất:

```text
Oracle gain       = Optimistic complete metric - Full complete metric
Errors avoided    = số lỗi của Y_full nằm trong tập abstain
Review load       = AABS
```

Không lấy hiệu giữa optimistic complete metric và selective metric vì hai đại lượng dùng mẫu số khác nhau.

### 4.4. Hai tập decided/rejected và hai nhóm IL/DL

Cụm “tính Macro-F1 trên cả hai tập” trong note chưa xác định rõ đối tượng. Để không bỏ sót ý, đặc tả yêu cầu cả hai lát cắt sau:

1. **Decided/rejected:**
   - `Decided Macro-F1` chính là Selective Macro-F1.
   - `Rejected Counterfactual Macro-F1` dùng `Y_full` tại các vị trí bị từ chối để đo mức khó nếu hệ thống buộc phải tự quyết định.
   - Xuất thêm `Rejected Error Rate` và `Error Capture Rate`.
2. **IL/DL:**
   - `IL Full Macro-F1` là trung bình F1 của các nhãn thuộc IL.
   - `DL Full Macro-F1` tương tự cho DL.
   - Tính thêm group-level coverage, precision, recall, F1 và Jaccard khi phù hợp.

Nếu một fold không có nhãn trong IL hoặc DL, group metric là `NaN` và phải kèm `group_label_count=0`; không thay bằng `0`. Nếu một nhãn không có rejected position, rejected diagnostic của nhãn đó là `NaN` và không được giả là hoàn hảo.

Hai metric đo khả năng triage:

```text
Rejected Error Rate = số lỗi của Y_full nằm trong A / số vị trí trong A
Error Capture Rate   = số lỗi của Y_full nằm trong A / tổng số lỗi của Y_full
```

Mẫu số bằng `0` thì metric tương ứng là `NaN` và phải xuất count gốc.

Ở cùng AABS, một rejector hữu ích phải có Error Capture Rate cao hơn baseline từ chối ngẫu nhiên; baseline ngẫu nhiên được lặp tối thiểu 1.000 lần với seed cố định để tạo khoảng tin cậy.

### 4.5. Nhãn quan trọng

Không được đồng nhất “nhãn hiếm” với “nhãn quan trọng”. Importance phải đến từ domain/user config, không suy ra từ test set.

Tạo file cấu hình `configs/label_policy.json` theo dataset, hỗ trợ:

```json
{
  "dataset_name": {
    "critical_labels": ["label_a", "label_b"],
    "weights": {"label_a": 3.0, "label_b": 2.0},
    "false_negative_cost": {"label_a": 10.0},
    "false_positive_cost": {"label_a": 2.0},
    "review_cost": 0.25
  }
}
```

Khi có config, xuất `Critical-label Recall/F1/Coverage`, `Critical Error Capture Rate`, `Optimistic Critical-label F1` và cost-sensitive utility. Khi không có config, đánh dấu `N/A`; chỉ được phân tích theo strata rare/medium/common và gọi đó là phân tích theo prevalence, không gọi là importance.

## 5. Phần cần sửa trong code

### C0. Ranh giới thay đổi core

Giữ nguyên mặc định và regression-test các phần sau:

- công thức huấn luyện BR trong `src/models/binary_relevance.py`;
- công thức huấn luyện/greedy inference CC trong `src/models/classifier_chain.py`;
- cách GSI sinh direct probability, conditional probability, mean-field probability và greedy IL/DL mặc định;
- external API hiện có của `predict`, `predict_proba`, `predict_full_from_proba`.

Chỉ thêm injection point cho `base_learner`, `decision_policy`, `partition_objective`, `partition_provider` và `metric_registry`. Default phải tái tạo hành vi hiện tại. Không sửa trực tiếp core để nhúng công thức F1/Jaccard/MDP.

### C1. Tạo decision-policy layer độc lập

Tạo package mới:

```text
src/decision/
├── __init__.py
├── base.py              # protocol/interface chung
├── hamming.py           # wrap SEP/PAR hiện tại
├── fbeta.py             # complete và partial BOP cho instance F-beta
└── jaccard.py           # complete và partial BOP cho instance Jaccard
```

Interface tối thiểu:

```python
class DecisionPolicy:
    def predict(self, probabilities, *, cost=None, penalty=None): ...
    def expected_utility(self, probabilities, *, cost=None, penalty=None): ...
    def get_config(self): ...
```

Yêu cầu:

- `HammingBOPPolicy` tái sử dụng đúng SEP/PAR hiện có; model cũ chỉ delegate sang module này.
- `FbetaBOPPolicy(beta=1)` cài Algorithm 2 của Nguyen–Hüllermeier, cho output `{0,-1,1}` và generalized utility `F_beta(Y_D, Yhat_D) - g(|A|)`.
- `JaccardBOPPolicy` cài Algorithm 3.
- Có chế độ `allow_abstention=False` cho BOP complete dùng trong immediate mode.
- Tie-breaking xác định và ghi rõ: ưu tiên nhiều quyết định hơn, sau đó thứ tự label index để reproducible.
- Với `K` lớn, vector hóa/cache các bảng xác suất count-distribution; ghi train/inference time riêng.
- Tính Bayes-optimal của Algorithm 2/3 dựa trên CLI. Khi đưa các dependent marginal của GSI vào policy này, phải ghi rõ đây là **BOP dưới xấp xỉ CLI**, không phải exact BOP của joint distribution phụ thuộc; chỉ marginal probability là chưa đủ cho exact non-CLI F1-BOP.
- Không gọi threshold `0.5` là F1-BOP. Tối ưu Hamming không được trình bày như tối ưu F1; hai loss có thể có Bayes action khác nhau ([Waegeman et al., JMLR 2014](https://www.jmlr.org/papers/v15/waegeman14a.html)).

### C2. Module hóa objective chọn IL/DL

Tạo `src/selection/objectives.py` và `src/selection/partition.py`.

`GSIMLCPartialAbstentionClassifier` nhận thêm tham số nhưng giữ default tương thích:

```text
selection_objective="full_macro_f1"
decision_policy="hamming"
partition_mode="learned"   # learned | all_il | all_dl | fixed | random_matched
```

Quy trình đánh giá một candidate partition phải là:

```text
candidate partition
  -> candidate probability matrix trên inner validation
  -> decision policy tương ứng
  -> objective score
  -> accept/reject việc chuyển nhãn IL/DL
```

Các objective bắt buộc:

| ID | Decision output dùng để score | Score |
|---|---|---|
| `full_macro_f1` | threshold `0.5`, complete | Macro-F1 hiện tại; baseline tương thích |
| `immediate_instance_f1` | complete F1-BOP | Mean instance-F1 khi phải quyết định ngay |
| `bop_instance_f1` | partial F1-BOP | Mean generalized F1 utility, đã trừ abstention penalty |
| `bop_jaccard` | partial Jaccard-BOP | Mean generalized Jaccard utility |
| `macro_precision` | complete policy cố định | Ablation thiên về FP; kèm predicted-positive rate |
| `macro_recall` | complete policy cố định | Ablation thiên về FN; kèm predicted-positive rate |
| `f_beta_0_5` | F-beta BOP | Precision-oriented nhưng tránh objective precision thuần |
| `f_beta_2` | F-beta BOP | Recall-oriented nhưng tránh objective recall thuần |

Precision/Recall thuần chỉ dùng ablation vì dễ tạo nghiệm cực đoan. Objective chính để kết luận là Macro-F1, generalized instance-F1, Jaccard hoặc cost-sensitive utility.

Không dùng outer test fold để chọn objective, cost, threshold, partition hoặc label importance. Mọi lựa chọn nằm trong outer-train/inner-validation. Cache phải ghi toàn bộ `selection_history`, objective, policy, cost và inner split seed.

### C3. Bổ sung metric mà không làm `metrics.py` phình thêm

Tách thành:

```text
src/evaluation/
├── metrics.py             # facade tương thích ngược
├── complete_metrics.py
├── abstention_metrics.py
├── group_metrics.py       # decided/rejected, IL/DL, critical labels
└── calibration_metrics.py
```

API dự kiến:

```python
compute_complete_metrics(y_true, y_full)
compute_abstention_metrics(y_true, y_partial, y_full, cost, penalty)
compute_group_metrics(y_true, y_full, y_partial, label_groups)
compute_per_label_metrics(y_true, y_full, y_partial)
compute_calibration_metrics(y_true, y_proba)
```

Thay đổi bắt buộc:

- thêm `Hamming Accuracy`, `Instance Jaccard`, `Macro Precision`, `Macro Recall`;
- đổi canonical name `Example-F1` thành `Instance-F1`, giữ alias ở lớp facade;
- thêm optimistic/oracle metrics và triage metrics ở mục 4;
- bỏ `UNPERSISTED_METRICS = {"Macro Precision", "Macro Recall"}`;
- không lưu hai alias thành hai cột trong bảng mới;
- validate shape, giá trị `{0,1}`/`{0,-1,1}`, empty-set convention và `NaN` convention thống nhất.

### C4. Đầy đủ baseline và công bằng hyperparameter

Tạo một registry duy nhất, ví dụ `src/experiments/model_registry.py`, và cấu hình `configs/experiment.json`. Không để BR/CC tự định nghĩa hai bản `_get_base_estimator` có thể lệch nhau.

Ma trận baseline tối thiểu:

| Model family | Linear SVM | Logistic Regression | MLP |
|---|---:|---:|---:|
| BR complete | Có | Có | Có |
| CC complete | Có | Có | Có |
| MLC-PA | Có, phải calibration | Có | Có |
| GSI-MLC-PA | Có, phải calibration | Có | Có |

Canonical model ID phải chứa đủ family/base/policy, ví dụ:

```text
BR__logistic
CC__logistic
MLC_PA__logistic__hamming
GSI_MLC_PA__logistic__hamming
GSI_MLC_PA__logistic__f1
```

Quy tắc công bằng:

- cùng outer folds, scaler, seed và dataset version;
- cùng base-learner factory và mọi hyperparameter có cùng ý nghĩa;
- cùng tuning budget; nếu không tuning thì cố định cấu hình cho tất cả family;
- cùng probability calibration và threshold/decision policy khi so kiến trúc;
- CC được phép có thêm previous-label features vì đó là định nghĩa mô hình, nhưng không được có training budget lớn hơn mà không ghi rõ;
- MLC-PA và BR cùng base phải có `Y_full` giống nhau; đây là invariant để phát hiện cấu hình lệch;
- mọi khác biệt bất khả kháng phải xuất trong `run_manifest.json`.

MLP benchmark phải **fail fast** nếu backend PyTorch không dùng được; không được im lặng fallback sang `sklearn.MLPClassifier` vì cấu trúc `(64,)`, epoch và optimizer sẽ thay đổi. Nếu cần CPU backend, đặt ID/config riêng và không trộn kết quả.

### C5. Calibration xác suất

Abstention phụ thuộc trực tiếp vào độ tin cậy của `p(y_k=1|x)`, vì vậy sigmoid thủ công trên LinearSVC margin không đủ để gọi là calibrated probability.

- Thêm `ProbabilityAdapter` module.
- Logistic và MLP dùng native probability nhưng vẫn phải đánh giá calibration.
- LinearSVC dùng sigmoid/Platt calibration chỉ trên outer-train, có inner CV; tuyệt đối không fit calibrator trên outer test. `CalibratedClassifierCV` hỗ trợ CV calibration ([tài liệu scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)).
- Với nhãn quá hiếm không đủ mẫu cho calibration folds, giảm số fold theo support hoặc dùng constant classifier; ghi fallback theo từng nhãn.
- Xuất Macro Brier score, log loss và ECE kèm reliability plot. Brier vừa phản ánh calibration vừa phản ánh discrimination nên không được diễn giải như calibration-only.
- Tách `svm_raw` legacy baseline và `svm_calibrated` probability baseline nếu cần giữ kết quả cũ.

### C6. Ablation chứng minh lợi ích IL/DL

Trên cùng outer fold, base learner và decision policy, chạy:

1. `all_il`: mọi nhãn dùng direct/BR probability;
2. `all_dl`: mọi nhãn dùng conditional-chain probability;
3. `learned`: partition do GSI chọn;
4. `random_matched`: partition ngẫu nhiên có cùng số IL với learned, lặp tối thiểu 30 seed;
5. `learned_no_correlation_order`: learned partition nhưng natural/fixed order, để tách lợi ích partition khỏi lợi ích reorder.

Để ablation hợp lệ, các biến thể phải dùng cùng final order khi câu hỏi chỉ là partition; hoặc phải tách thêm factor `order`. Báo cáo:

- delta learned so với `all_il`, `all_dl`, và mean/CI của `random_matched`;
- IL/DL group metrics;
- tỷ lệ nhãn vào IL/DL;
- Jaccard similarity của tập IL giữa các folds để đo stability;
- thời gian selection và inference.

Chỉ được kết luận “chia IL/DL giúp tăng hiệu quả” khi learned vượt ít nhất hai endpoint `all_il` và `all_dl` trên metric mục tiêu, không đánh đổi coverage/utility quá ngưỡng đã định và có kết quả nhất quán qua datasets/folds.

### C7. Đánh giá lợi ích ứng dụng thực tế

Tạo `src/evaluation/deployment.py` để tính ở từng operating point:

- immediate complete quality;
- selective quality ở coverage cố định;
- optimistic human-assisted upper bound;
- review load theo label-position (`AABS`) và theo case (`ABS`);
- error capture/review efficiency;
- critical-label quality nếu có policy config;
- utility với FP cost, FN cost và review cost;
- sensitivity analysis reviewer accuracy `h in {0.80, 0.90, 0.95, 1.00}` nếu có đủ giả định domain.

Operating point phải được chọn trên inner validation theo một trong ba rule đã khai báo trước:

```text
min Generalized Loss
max utility subject to Coverage >= gamma
max Coverage subject to selective risk <= epsilon
```

Không chọn cost tốt nhất trên test rồi báo lại cùng test. Với grid cost hiện tại, thêm `c=0.5` làm no-abstention sanity point.

### C8. Output schema, cache và biểu đồ

Nâng cache lên schema v3 hoặc dùng output directory mới `results_pa_v3/`; không ghi đè kết quả v2. Cache key/hash phải gồm:

```text
dataset + fold + seed + scaler
model family + base learner + all hyperparameters + backend
calibration method
partition mode + selection objective + chain order
decision policy + beta + penalty + cost grid
metric version + label-policy hash
```

Output tối thiểu:

```text
results_pa_v3/
├── run_manifest.json
├── tables/
│   ├── summary_complete.csv
│   ├── summary_selective.csv
│   ├── per_label_metrics.csv
│   ├── il_dl_ablation.csv
│   ├── objective_ablation.csv
│   ├── calibration.csv
│   └── raw_folds.json
└── figures/
    ├── risk_coverage.png
    ├── optimistic_gain_vs_review_load.png
    ├── error_capture_vs_review_load.png
    ├── il_dl_ablation.png
    ├── objective_comparison.png
    ├── per_label_critical_metrics.png
    └── calibration_reliability.png
```

Trong CSV, lưu số thực `[0,1]`; chỉ đổi sang phần trăm ở plot/report. Hamming Accuracy là cột hiển thị chính; vẫn lưu Hamming Loss nội bộ để audit `accuracy + loss = 1` trên complete prediction.

### C9. Kiểm thử bắt buộc

Bổ sung test nhỏ, deterministic trước khi chạy lại 10 datasets:

1. `Hamming Accuracy == 1 - Hamming Loss` cho complete prediction.
2. All-abstain: `Coverage=0`, `Selective Hamming Accuracy=NaN`, selective F1 không được thành `1`, oracle completion trùng `Y`, và generalized loss vẫn phản ánh abstention cost. Kiểm riêng rằng optimistic positive-class Macro-F1 tuân theo `zero_division=0` khi một nhãn không có positive support.
3. Oracle completion chỉ thay đúng vị trí `-1`.
4. Empty true/predicted label set cho Instance-F1 và Jaccard bằng `1`.
5. Per-label Macro-F1 khớp `sklearn.f1_score(..., average="macro", zero_division=0)`.
6. Group IL/DL rỗng trả `NaN` cùng count bằng `0`.
7. F1/Jaccard BOP khớp exhaustive search trên mọi action `{0,-1,1}^K` với synthetic `K <= 6`.
8. BOP F1 có ít nhất một counterexample mà output khác threshold `0.5`, chứng minh pipeline thật sự gọi đúng policy.
9. Calibration và IL/DL selector không quan sát outer test bằng spy estimator/split indices.
10. BR và MLC-PA cùng base có probability/full-prediction bằng nhau.
11. Registry sinh đúng đủ `4 families x 3 base learners` và config tương ứng bằng nhau.
12. Cache v2 không được dùng như v3; cache v3 đổi khi objective/policy/config đổi.
13. Không silent MLP fallback.
14. Mọi plot/table xử lý được `NaN`, nhãn/group rỗng và model chưa chạy đủ.

Sau unit tests, chạy smoke test trên `emotions`, `2 folds`, một base learner; sau đó mới chạy full benchmark.

## 6. Phần cần sửa trong báo cáo

### R1. Tạo file tóm tắt cuộc họp

Tạo `meeting_summary.md`, không dùng `tmp.md` làm tài liệu chính thức. Nội dung gồm:

- ngày/bối cảnh cuộc họp;
- vấn đề giảng viên chỉ ra;
- quyết định đã chốt;
- điểm còn phải xác nhận: “instance-based” hay “instant-based”, ý nghĩa “cả hai tập”, danh sách nhãn quan trọng và review cost;
- backlog chia Code/Report, người phụ trách, trạng thái và bằng chứng hoàn thành.

### R2. Sửa phần phương pháp và BOP

Phải mô tả kiến trúc hai tầng:

```text
X -> probability estimator (BR/CC/GSI) -> decision policy -> {0, -1, 1}
```

Nêu rõ:

- training/ước lượng xác suất độc lập với rejection cost;
- Hamming-BOP, F1-BOP và Jaccard-BOP là các decision policy khác nhau;
- F1/Jaccard BOP dùng dynamic programming dưới giả định conditional label independence;
- mean-field nhiều parent trong GSI là xấp xỉ, không được gọi là exact marginalization;
- calibration là điều kiện quan trọng khi threshold xác suất quyết định abstention.

### R3. Sửa phần MDP và biên hóa xác suất

Thêm một tiểu mục “MDP có cần thiết không?” với kết luận:

- BOP hiện tại là one-step Bayes decision, không phải MDP.
- Dynamic programming của Algorithm 2/3 chỉ là kỹ thuật tính expected F1/Jaccard.
- Probability marginalization của classifier chain dựa trên chain rule; exact inference, mean-field và Monte Carlo là ba mức so sánh phù hợp.
- MDP/POMDP chỉ là hướng mở rộng khi có quan sát tuần tự hoặc human feedback làm thay đổi state.

Không được viết rằng mô hình hiện tại “dùng MDP”. Nếu chưa chạy research spike ở mục 7, ghi đây là hướng nghiên cứu tương lai.

### R4. Viết lại phần metrics: công thức, ý nghĩa và lý do dùng

Mỗi metric phải có: công thức, đơn vị/mẫu số, hướng tốt, câu hỏi nó trả lời và hạn chế.

Các diễn giải bắt buộc:

- Macro-F1 cho mỗi nhãn trọng số ngang nhau, phù hợp mất cân bằng nhãn nhưng có variance cao ở nhãn rất hiếm.
- Micro-F1 phản ánh hiệu suất tổng thể nhưng bị nhãn phổ biến chi phối.
- Hamming Accuracy dễ hiểu nhưng có thể cao do true negatives; luôn đọc cùng F1/Jaccard.
- Subset Accuracy đánh giá toàn vector, rất nghiêm ngặt khi `K` lớn.
- Instance-F1/Jaccard đánh giá chất lượng tập nhãn của từng case, phù hợp yêu cầu quyết định ngay.
- Generalized Loss đo đúng trade-off lỗi–abstention của decision policy.
- Coverage/ABS/AABS đo workload; selective F1 nếu thiếu coverage có thể gây hiểu nhầm.
- Optimistic metrics là upper bound giả định reviewer đúng 100%, không phải khả năng tự động.
- Per-label/critical metrics trả lời liệu mô hình đúng các nhãn quan trọng hay không.

Trong phần thân báo cáo dùng `Hamming Accuracy`; Hamming Loss chỉ xuất hiện khi trình bày hàm mục tiêu/lý thuyết hoặc phụ lục đối chiếu.

### R5. Sửa phần Macro-F1 và trình bày “cả hai tập”

Bảng partial-abstention chính ở mỗi dataset/operating point phải có tối thiểu:

| Model | Full Macro-F1 | Decided Macro-F1 | Rejected CF Macro-F1 | Optimistic Macro-F1 | Coverage | ABS | AABS |
|---|---:|---:|---:|---:|---:|---:|---:|

Thêm bảng IL/DL:

| Model | #IL | #DL | IL Macro-F1 | DL Macro-F1 | IL Coverage | DL Coverage | Partition stability |
|---|---:|---:|---:|---:|---:|---:|---:|

Không đưa training Macro-F1 vào bảng kết quả chính. Nếu “cả hai tập” thực sự được xác nhận là train/test, training score chỉ để phân tích overfit ở phụ lục; mọi claim vẫn dựa trên outer test.

### R6. Bổ sung baseline và bảng hyperparameter

Báo cáo đầy đủ 12 tổ hợp family/base ở mục C4 hoặc ghi rõ tổ hợp nào không chạy được và lý do. Mỗi bảng so sánh phải nhóm theo base learner; không so GSI-MLP với MLC-PA-Logistic rồi quy chênh lệch cho kiến trúc.

Bảng hyperparameter phải có:

- estimator/backend;
- preprocessing;
- architecture/hidden units;
- optimizer, learning rate, weight decay, epochs/early stopping;
- SVM `C`, tolerance, calibration method/folds;
- seed, CV folds, inner validation size;
- chain order;
- abstention penalty/cost grid;
- objective và decision policy.

Nêu rõ tham số nào giống nhau và tham số nào không thể đồng nhất vì bản chất learner khác nhau. “Giống hyperparameter” có nghĩa cùng cấu hình cho **cùng base learner qua các model family**, không phải ép SVM và MLP dùng tham số cùng tên.

### R7. Bổ sung objective/IL-DL ablation

Trình bày hai ablation riêng:

1. **Selection objective:** Macro-F1, immediate instance-F1, partial F1-BOP, Jaccard-BOP, precision-oriented F0.5 và recall-oriented F2.
2. **Partition:** all-IL, all-DL, learned, random-matched và learned không reorder.

Với mỗi objective, báo cáo không chỉ objective đó mà toàn bộ guardrail metrics. Ví dụ objective recall phải kèm precision, predicted-positive rate, Hamming Accuracy, coverage và cost.

### R8. Chứng minh giá trị thực tế bằng metric, không chỉ bằng nhận định

Phần thảo luận ứng dụng phải trả lời:

- Ở coverage/review budget thực tế, error rate trên phần tự động giảm bao nhiêu?
- Bao nhiêu phần trăm tổng lỗi được đưa sang tập review?
- Bao nhiêu case và label-position phải review?
- Nhãn critical có recall/F1/error-capture tốt hơn không?
- Optimistic gain có đủ lớn so với review cost không?
- Nếu reviewer không hoàn hảo, kết luận còn giữ ở accuracy nào?
- Lợi ích đến từ IL/DL partition, chain order, base learner hay chỉ do abstention?

Ngôn ngữ claim:

- Nếu chỉ có optimistic 100%: “tiềm năng/lợi ích tối đa giả định”.
- Nếu có cost và reviewer sensitivity: “lợi ích kỳ vọng trong kịch bản đã giả định”.
- Chỉ dùng “có thể áp dụng thực tế” khi có domain label priority, review budget/cost và operating constraint cụ thể.

### R9. Thống kê và cách kết luận

- Báo cáo mean ± std trên outer folds cho từng dataset.
- So model bằng paired dataset-level results; không xem mọi fold của mọi dataset là mẫu độc lập.
- Khi so nhiều model: Friedman test trên dataset ranks, sau đó post-hoc pairwise Wilcoxon với Holm correction; báo effect size và confidence interval, không chỉ p-value.
- Tách kết quả “primary” đã đăng ký trước khỏi exploratory ablation.
- Kết luận theo số dataset thắng/thua/hòa và magnitude, không chỉ grand average.

## 7. Research spike MDP/marginalization — P2, có gate

Tạo tài liệu `docs/research/mdp_probability_marginalization.md` trước khi viết code. Tài liệu phải trả lời:

1. State tối thiểu có chứa đủ thông tin Markov không?
2. Action là predict 0/1, abstain, chọn nhãn tiếp theo hay request review?
3. Transition nào làm posterior thay đổi, và lấy dữ liệu transition ở đâu?
4. Reward có FP/FN/review/time cost thật hay chỉ là metric tùy ý?
5. Có feedback giữa episode không, hay chỉ có một batch prediction?
6. MDP cải thiện gì so với F1/Jaccard DP, mean-field, exact-small-K hoặc Monte Carlo ở cùng budget?

Chỉ tạo `src/sequential/mdp_policy.py` khi cả bốn điều kiện sau đúng:

- có quy trình tuần tự và feedback làm thay đổi state;
- xác định được transition/reward từ domain;
- có simulator hoặc log tương tác để train/evaluate không dùng test leakage;
- pilot trên synthetic/small dataset vượt decision-policy tĩnh ở cùng review budget.

Nếu thiếu một điều kiện, kết luận research spike là “MDP chưa phù hợp với phạm vi hiện tại”; giữ ở future work. Đây không phải thất bại mà là kết luận mô hình hóa cần thiết để tránh thêm độ phức tạp không có bằng chứng.

Song song, có thể tạo module ablation nhẹ hơn `src/inference/marginalization.py`:

```text
mean_field        # default hiện tại
exact_small_k     # oracle kiểm tra đúng cho số parent nhỏ
monte_carlo       # xấp xỉ scalable, seed và sample budget cố định
```

Module này chỉ thay strategy inference qua injection point, không sửa cách train BR/CC/GSI.

## 8. Kế hoạch thực hiện theo quota 5 giờ cho mỗi phase

### 8.1. Quy ước quota

- **Mỗi phase là một phiên độc lập, tối đa 5 giờ của model.** Toàn bộ đặc tả cần nhiều phase; không ép hoàn thành trong một quota.
- Mỗi phase chỉ phân bổ khoảng `3h50` cho đọc handoff và implementation. `1h10` cuối bắt buộc dành cho test, sửa lỗi, rà diff và viết handoff; đây là phần dự phòng để phase không vượt quota.
- Phase sau chỉ bắt đầu khi phase trước đạt exit criteria hoặc handoff ghi rõ phần chưa đạt và cách tiếp tục.
- Không gộp một hạng mục đang dở sang phase mới mà không cập nhật phạm vi phase mới. Nếu task vượt quota, dừng ở checkpoint an toàn và chia nhỏ tiếp.
- Thời gian GPU chạy dài được quản lý bằng cache/checkpoint, nhưng không được giả định một model-dataset pair sẽ hoàn tất ngoài quota mà không có khả năng resume.

### 8.2. Mẫu vận hành chung của một phase

| Khoảng thời gian | Việc bắt buộc |
|---|---|
| `00:00–00:20` | Đọc handoff phase trước, kiểm tra `git status`, chạy test nền liên quan và khóa phạm vi file |
| `00:20–03:50` | Thực hiện mục tiêu chính; tạo checkpoint nhỏ sau từng hạng mục hoàn chỉnh |
| `03:50–04:30` | Unit/integration/smoke tests và sửa lỗi blocker/high |
| `04:30–05:00` | Buffer, `git diff --check`, kiểm tra thay đổi ngoài phạm vi và viết handoff phase sau |

Tổng mỗi phase: `20 + 210 + 40 + 30 = 300 phút`.

Mỗi phase tạo `docs/progress/phase_<ID>.md` gồm:

```text
Objective đã hoàn thành
Files đã sửa/tạo
Quyết định kỹ thuật và giả định
Test/lệnh đã chạy + kết quả
Output/cache được tạo
Việc chưa hoàn thành hoặc blocker
Git status và thay đổi ngoài phạm vi
Entry point/lệnh đầu tiên cho phase tiếp theo
```

### 8.3. Roadmap các phase, mỗi phase không quá 5 giờ

#### Phase Q0 — Audit, hợp đồng metric và tài liệu cuộc họp

**Trạng thái:** ✅ Đã hoàn thành ngày 2026-08-30 — xem [`docs/progress/phase_Q0.md`](docs/progress/phase_Q0.md).

**Mục tiêu:** khóa nền tảng trước khi sửa production code.

Phạm vi:

1. Chạy test hiện tại, audit source/cache/result schema và ghi baseline.
2. Tạo `meeting_summary.md` từ `tmp.md`, đánh dấu thuật ngữ cần xác nhận.
3. Khóa canonical names, công thức, `zero_division`, empty-set và `NaN` conventions ở mục 4.
4. Tạo skeleton `complete_metrics.py`, `abstention_metrics.py`, `group_metrics.py`, facade và test fixtures.
5. Viết test contract trước cho complete, all-abstain, group rỗng và zero support; chưa cần làm chúng pass trong Q0.

Exit criteria:

- baseline test log và audit hiện trạng tồn tại;
- metric contract không còn tên/công thức mơ hồ;
- module skeleton import được nhưng chưa thay production output;
- `meeting_summary.md` và `docs/progress/phase_Q0.md` tồn tại.

#### Phase Q1 — Complete, partial, optimistic và group metrics

**Trạng thái:** ✅ Đã hoàn thành ngày 2026-08-31 — xem [`docs/progress/phase_Q1.md`](docs/progress/phase_Q1.md).

**Phụ thuộc:** Q0.

Phạm vi:

1. Cài complete metrics: Macro/Micro-F1, Hamming Accuracy, Instance-F1/Jaccard, Macro Precision/Recall.
2. Cài selective và optimistic/oracle completion metrics.
3. Cài decided/rejected error metrics, error capture, IL/DL groups và per-label coverage/support.
4. Critical-label hook trả `N/A` khi chưa có domain config.
5. Giữ `Example-F1` alias qua facade nhưng không tạo cột trùng.

Exit criteria:

- metric unit tests pass cho perfect, all-abstain, no-abstain, no-error, group rỗng, zero support và zero denominator;
- complete metrics khớp scikit-learn ở trường hợp chuẩn;
- Full/Selective/Rejected/Optimistic APIs có mẫu số và `NaN` convention rõ;
- chưa thay cache/schema production.

#### Phase Q2 — Pipeline integration, output schema v3 và resumability

**Trạng thái:** ✅ Đã hoàn thành ngày 2026-08-31 — xem [`docs/progress/phase_Q2.md`](docs/progress/phase_Q2.md).

**Phụ thuộc:** Q1.

Phạm vi:

1. Nối metric mới vào `_evaluate_model` và summary pipeline.
2. Nâng cache/output lên schema v3, thêm settings/config hash và migration guard.
3. Thêm fold-level checkpoint để model-dataset pair có thể dừng/resume giữa quota.
4. Xuất JSON/CSV cho complete, selective, per-label và group scopes.
5. Chạy synthetic/tiny integration và interruption/resume test.

Exit criteria:

- tiny run sinh đủ scopes với canonical names;
- cache v2 không bị ghi đè hoặc đọc nhầm như v3;
- interruption sau một fold resume mà không chạy lại fold đã hoàn thành;
- schema/audit tests pass và `docs/progress/phase_Q2.md` ghi migration rule.

#### Phase Q3 — Decision-policy interface và Hamming regression

**Trạng thái:** ✅ Đã hoàn thành ngày 2026-08-31 — xem [`docs/progress/phase_Q3.md`](docs/progress/phase_Q3.md).

**Phụ thuộc:** Q2.

Phạm vi:

1. Tạo `src/decision/base.py` và policy registry tối thiểu.
2. Wrap Hamming SEP/PAR hiện tại vào `HammingBOPPolicy`.
3. Giữ API `predict`, `predict_from_proba`, `decision_mask` tương thích.
4. Thêm deterministic tie/threshold boundary tests và config serialization.
5. Chạy regression trên fixture/cache nhỏ để chứng minh default output không đổi.

Exit criteria:

- mọi Hamming unit/regression test pass;
- model cũ delegate sang policy module nhưng kết quả bitwise/numerically tương đương;
- decision policy được ghi trong cache hash/manifest;
- chưa cài F1/Jaccard trong cùng phase.

#### Phase Q4 — F1-BOP hoàn chỉnh

**Trạng thái:** ✅ Đã hoàn thành ngày 2026-08-31 — xem [`docs/progress/phase_Q4.md`](docs/progress/phase_Q4.md).

**Phụ thuộc:** Q3.

Phạm vi:

1. Cài complete/partial `FbetaBOPPolicy`, ưu tiên `beta=1`.
2. Cài count-distribution dynamic programming và deterministic tie-breaking.
3. Ghi CLI assumption/metadata và phân biệt dependent-marginal approximation.
4. Exhaustive validation trên `{0,-1,1}^K`, `K <= 6`.
5. Thêm counterexample F1-BOP khác threshold `0.5`/Hamming-BOP và benchmark runtime nhỏ.

Exit criteria:

- F1-BOP khớp exhaustive optimum trên toàn bộ synthetic fixtures;
- complete/no-abstention và partial modes đều pass;
- dependent marginals được gắn nhãn “BOP dưới xấp xỉ CLI”;
- chưa nối policy vào GSI selection.

#### Phase Q5 — Jaccard-BOP hoàn chỉnh

**Trạng thái:** ✅ Đã hoàn thành ngày 2026-08-31 — xem [`docs/progress/phase_Q5.md`](docs/progress/phase_Q5.md).

**Phụ thuộc:** Q4.

Phạm vi:

1. Cài complete/partial `JaccardBOPPolicy` theo Algorithm 3.
2. Tái sử dụng hạ tầng count-distribution an toàn từ Q4.
3. Cài empty-union convention, penalty và deterministic tie-breaking.
4. Exhaustive validation `K <= 6` và runtime/memory smoke với `K` lớn hơn.
5. Rà API/config/cache parity với F1/Hamming policies.

Exit criteria:

- Jaccard-BOP khớp exhaustive optimum;
- không regression Hamming/F1;
- policy registry tạo và serialize được cả ba objective families;
- `docs/progress/phase_Q5.md` ghi độ phức tạp và giới hạn CLI.

#### Phase Q6 — Objective injection cho GSI

**Trạng thái:** ✅ Đã hoàn thành ngày 2026-08-31 — xem `docs/progress/phase_Q6.md`.

**Phụ thuộc:** Q5.

Phạm vi:

1. Tạo `src/selection/objectives.py` với `full_macro_f1`, `immediate_instance_f1`, `bop_instance_f1`, `bop_jaccard`, F0.5 và F2.
2. Inject objective/decision policy vào GSI, giữ default `full_macro_f1` tương thích.
3. Score mọi candidate trên inner validation; cấm outer-test access.
4. Cache selection history, objective, policy, beta, cost và seed.
5. Thêm spy leakage test, default regression fixture và synthetic objective smoke.

Exit criteria:

- spy test chứng minh không có outer-test leakage;
- default GSI tái tạo kết quả trước thay đổi;
- thay objective cập nhật score/history đúng policy;
- selector không chứa công thức decision policy hard-code.

#### Phase Q7 — Partition modes và IL/DL ablation infrastructure

**Trạng thái:** ✅ Đã hoàn thành ngày 2026-09-02 — xem `docs/progress/phase_Q7.md`.

**Phụ thuộc:** Q6.

Phạm vi:

1. Tạo `partition_provider` cho `all_il`, `all_dl`, `learned`, `fixed`, `random_matched`.
2. Thêm `learned_no_correlation_order` và control final order để tách partition khỏi reorder.
3. Xuất IL/DL group metrics, partition size, stability và selection/inference time.
4. Test empty IL/DL, fixed partition, random seed và same-order invariants.
5. Smoke ablation nhỏ trên `emotions`, 2 folds, một base learner.

Exit criteria:

- năm partition modes chạy qua cùng evaluation API;
- random-matched tái lập theo seed và giữ đúng số IL;
- ablation output đủ để so learned với all-IL/all-DL mà không confound order;
- chưa chạy full 30-seed/10-dataset ablation.

#### Phase Q8 — Shared registry, matched Logistic/MLP baselines

**Trạng thái:** ✅ Đã hoàn thành ngày 2026-09-02 — xem `docs/progress/phase_Q8.md`.

**Phụ thuộc:** Q2; nên thực hiện sau Q7 để registry bao phủ API cuối.

Phạm vi:

1. Tạo shared base-learner factory và `configs/experiment.json`.
2. Loại hai factory BR/CC bị trùng mà không thay thuật toán fit/predict.
3. Đăng ký BR, CC, MLC-PA, GSI-MLC-PA cho Logistic và MLP.
4. Loại silent MLP fallback; backend khác phải có ID/config khác.
5. Thêm manifest và invariant BR/MLC-PA cùng base có cùng full probabilities/predictions.

Exit criteria:

- 8 model IDs Logistic/MLP được tạo từ một registry;
- factory/config tests và invariant tests pass;
- smoke mỗi family trên tiny data pass;
- hyperparameter/backend xuất đầy đủ vào manifest.

#### Phase Q9 — SVM calibration và hoàn chỉnh 12 baselines

**Trạng thái:** ✅ Đã hoàn thành ngày 2026-09-02 — xem `docs/progress/phase_Q9.md`.

**Phụ thuộc:** Q8.

Phạm vi:

1. Tạo `ProbabilityAdapter` và `svm_calibrated` bằng Platt/sigmoid calibration trong outer-train.
2. Xử lý rare-label calibration folds/constant labels và ghi fallback.
3. Thêm Brier, log loss, ECE và reliability data.
4. Đăng ký bốn family dùng calibrated SVM; giữ `svm_raw` chỉ như legacy full baseline nếu cần.
5. Spy/split tests bảo đảm calibrator không thấy outer test.

Exit criteria:

- đủ 12 matched family/base IDs;
- calibration leakage tests pass;
- rare-label fallback deterministic và được ghi manifest;
- smoke SVM family pass trên dataset nhỏ.

#### Phase Q10 — Deployment metrics, critical labels và visualization

**Trạng thái:** ✅ Đã hoàn thành ngày 2026-09-02 — xem `docs/progress/phase_Q10.md`.

**Phụ thuộc:** Q2, Q7, Q9.

Phạm vi:

1. Cài `deployment.py`, risk-at-coverage, AURC, review load, error-capture efficiency và optimistic gain.
2. Tạo/validate `configs/label_policy.json`; không suy importance từ test data.
3. Cài cost-sensitive utility và optional reviewer accuracy scenarios.
4. Chọn operating point chỉ trên inner validation.
5. Sinh các CSV/plot ở C8 và test `NaN`/missing model/incomplete cache.

Exit criteria:

- risk–coverage và deployment tables sinh từ fixture/smoke results;
- label-policy thiếu thì critical metrics là `N/A`, không crash;
- operating-point leakage test pass;
- plot/table không so sai Full với Selective denominator.

#### Phase Q11 — System verification và reproducible smoke benchmark

**Trạng thái:** ✅ Đã hoàn thành ngày 2026-09-02 — xem `docs/progress/phase_Q11.md`.

**Phụ thuộc:** Q0–Q10.

Phạm vi:

1. Chạy toàn bộ C9 unit/integration tests và sửa blocker/high.
2. Rà cache hash, schema migration, aliases, deterministic seed và run manifest.
3. Chạy smoke benchmark `emotions`, 2 folds, 12 baseline IDs, cost `0.3` và `0.5`; decision-objective ablation có thể giới hạn Logistic.
4. Audit output JSON/CSV/figures bằng script, không chỉ nhìn thủ công.
5. Cập nhật README/lệnh chạy và freeze experiment config cho full run.

Lệnh smoke dự kiến:

```powershell
python tests_unit.py
python main.py --datasets emotions `
  --n_splits 2 --abstention_costs 0.3 0.5 `
  --report_cost 0.3 --output_dir results_pa_v3_smoke
```

Exit criteria:

- test suite pass hoặc chỉ còn issue medium/low được ghi rõ;
- smoke run hoàn tất, audit invariants pass và không sửa kết quả v2;
- config cho full run được khóa bằng hash;
- `docs/progress/phase_Q11.md` chứa lệnh resume chính xác.

#### Phase Q12 — Full experiments, phase lặp lại theo quota

**Phụ thuộc:** Q11.

**Tiến độ:**

- ✅ Q12.1 hoàn thành ngày 2026-09-02: `emotions`, 12/12 pairs, 60/60 folds — xem `docs/progress/phase_Q12.1.md`.

Q12 là phase **repeatable** (`Q12.1`, `Q12.2`, ...), mỗi lần vẫn tối đa 5 giờ. Không gộp toàn bộ 10 datasets và mọi ablation vào một quota.

Mỗi lần Q12:

1. Đọc job queue/cache và ước lượng runtime từ các run trước.
2. Chọn số model-dataset pairs có thể hoàn thành/checkpoint trong khoảng 4 giờ.
3. Không dispatch job mới sau mốc `03:30`; thời gian còn lại dành cho job hiện tại, cache audit và handoff.
4. Sau mỗi fold/pair, xác minh cache có thể load và settings hash khớp.
5. Khi có lỗi, ưu tiên chẩn đoán/retry một pair; không thay code lớn trong phase experiment.

Gợi ý thứ tự queue:

```text
small:  emotions, music, scene, yeast
medium: genbase, medical, enron
large:  cal500, bibtex, reuters-k500
primary matched baselines trước -> objective ablation -> random-matched repetitions
```

Exit criteria của mỗi Q12.x:

- mọi job đã dispatch có completed cache hoặc fold checkpoint hợp lệ;
- summary completeness report ghi pairs completed/missing/failed;
- không có cache khác config bị trộn;
- handoff chỉ rõ job queue kế tiếp.

Thoát Q12 khi đủ primary runs, cost grid và ablations đã đăng ký trước. Full benchmark không bắt buộc hoàn tất trong một Q12.x.

#### Phase Q13 — Statistical analysis và bảng/biểu đồ cuối

**Phụ thuộc:** Q12 hoàn tất primary runs.

Phạm vi:

1. Audit completeness và loại run invalid theo rule đã đăng ký, không chọn theo kết quả đẹp/xấu.
2. Tổng hợp dataset-level paired results, effect sizes và confidence intervals.
3. Friedman + post-hoc Wilcoxon-Holm cho so sánh nhiều model.
4. Tạo bảng complete/selective/rejected/optimistic, IL/DL và hyperparameter.
5. Chốt figure captions và claim matrix: claim nào được/không được dữ liệu hỗ trợ.

Exit criteria:

- script analysis tái lập từ raw cache;
- mọi số trong bảng truy ngược được dataset/fold/config;
- không pseudo-replicate folds như mẫu độc lập;
- có danh sách kết luận thắng/thua/hòa và giới hạn.

#### Phase Q14 — Cập nhật báo cáo hoàn chỉnh

**Phụ thuộc:** Q13.

Phạm vi:

1. Hoàn thiện R2–R9: phương pháp, metric, baseline, objective/partition ablation, deployment và thống kê.
2. Dùng Hamming Accuracy trong phần thân, giữ generalized loss ở phần lý thuyết.
3. Phân biệt immediate, selective, rejected và optimistic claims.
4. Ghi limitations: CLI approximation, calibration, mean-field, reviewer/domain assumptions.
5. Cross-check mọi claim với bảng/figure và cập nhật `meeting_summary.md`.

Exit criteria:

- báo cáo không còn claim chỉ dựa trên Selective Macro-F1;
- công thức/tên metric khớp code;
- baseline/hyperparameter tables đầy đủ;
- tài liệu có reproduction commands và reference đúng.

#### Phase Q15 — MDP/marginalization research spike, tùy chọn

**Phụ thuộc:** báo cáo chính không phụ thuộc phase này; chỉ chạy khi cần trả lời hướng nghiên cứu mục 7.

Phạm vi một quota:

1. Viết `docs/research/mdp_probability_marginalization.md` và trả lời sáu gate questions.
2. So sánh one-step BOP, dynamic programming, exact-small-K, mean-field và Monte Carlo về giả định/độ phức tạp.
3. Chỉ thiết kế synthetic pilot nếu xác định được state/action/transition/reward.
4. Không nối MDP vào core trong cùng phase nghiên cứu.

Exit criteria:

- kết luận rõ `not applicable`, `needs data/domain definition`, hoặc `pilot justified`;
- nếu pilot justified, tách implementation thành phase Q16 mới, cũng tối đa 5 giờ;
- không mô tả mô hình hiện tại là MDP khi chưa có sequential feedback.

### 8.4. Quy tắc ưu tiên và dừng phase

Thứ tự mặc định:

```text
Q0 -> Q1 -> Q2 -> Q3 -> Q4 -> Q5 -> Q6 -> Q7 -> Q8
   -> Q9 -> Q10 -> Q11 -> Q12.x (lặp) -> Q13 -> Q14
Q15 là tùy chọn; Q16 chỉ tồn tại nếu Q15 qua gate.
```

Quy tắc khi phase gần hết quota:

1. Không bắt đầu subtask mới sau `04:00`.
2. Không cắt test bảo toàn cache, leakage, metric edge cases hoặc reproducibility.
3. Có thể hoãn plot/prose/performance optimization sang phase sau.
4. Nếu implementation chính chưa xong, tạo checkpoint có test cho phần đã hoàn tất; không để API nửa cũ nửa mới mà không có feature flag.
5. Handoff là deliverable bắt buộc, không phải phần tùy chọn khi còn thời gian.

## 9. Definition of Done

### 9.1. Definition of Done áp dụng cho từng quota phase

- [ ] Phase không vượt quá một quota 5 giờ và không bắt đầu subtask mới sau mốc dừng ở mục 8.4.
- [ ] Exit criteria riêng của phase ở mục 8.3 đã đạt; nếu chưa đạt phải ghi trạng thái `partial` thay vì `complete`.
- [ ] Test liên quan đến thay đổi của phase đã chạy và kết quả được ghi nguyên văn/tóm tắt có thể kiểm chứng.
- [ ] Không ghi đè cache/kết quả cũ và không trộn output khác config hash.
- [ ] `git diff --check` pass; mọi thay đổi ngoài phạm vi được bảo toàn và giải thích.
- [ ] `docs/progress/phase_<ID>.md` có files, decisions, tests, outputs, blockers, git status và entry point cho phase sau.
- [ ] Không claim hạng mục của phase tương lai là đã hoàn thành; đặc biệt với exact BOP, đủ baselines, full benchmark, hiệu quả thực tế và MDP.

### 9.2. Definition of Done cho toàn bộ đặc tả

Đợt cải tiến hoàn thành khi:

- [ ] Tất cả unit/smoke tests ở C9 pass.
- [ ] Kết quả cũ không bị ghi đè; mọi run có manifest và config hash.
- [ ] Có đủ baseline theo base learner hoặc có log lý do thiếu.
- [ ] Không có so sánh family bị confound bởi base learner/hyperparameter/calibration khác nhau.
- [ ] Complete, selective, rejected và optimistic metrics có tên/mẫu số tách biệt.
- [ ] Hamming Accuracy là metric hiển thị; generalized Hamming Loss vẫn được giữ cho BOP/audit.
- [ ] Có Jaccard, Instance-F1, Macro Precision/Recall và per-label table.
- [ ] Có error-capture/review-load và critical-label analysis hoặc ghi `N/A` vì thiếu domain config.
- [ ] Có ablation chứng minh hoặc bác bỏ lợi ích IL/DL một cách độc lập với order/base/policy.
- [ ] Có objective ablation, trong đó GSI thực sự gọi BOP instance-F1/Jaccard trên inner validation.
- [ ] Không chọn cost/objective/threshold trên outer test.
- [ ] Báo cáo mô tả đúng giới hạn của optimistic metrics và không suy diễn selective Macro-F1 thành hiệu quả thực tế.
- [ ] `meeting_summary.md` tồn tại và các điểm chưa xác nhận đã được cập nhật.
- [ ] MDP được ghi đúng là future/research direction, trừ khi đã qua gate và có kết quả pilot.

## 10. Tài liệu tham khảo chính

1. Nguyen, V.-L. và Hüllermeier, E. *Multilabel Classification with Partial Abstention: Bayes-Optimal Prediction under Label Independence*. JAIR 72 (2021), 613–665. [Bản local](TaiLieuThamKhao/sminton,+12610-Article+(PDF)-28712-1-11-20211029.pdf), [DOI](https://doi.org/10.1613/jair.1.12610).
2. Read, J., Pfahringer, B., Holmes, G. và Frank, E. *Classifier Chains for Multi-label Classification*. Machine Learning 85 (2011). [Bản local](TaiLieuThamKhao/s10994-011-5256-5.pdf), [DOI](https://doi.org/10.1007/s10994-011-5256-5).
3. Waegeman, W. et al. *On the Bayes-Optimality of F-Measure Maximizers*. JMLR 15 (2014). [JMLR](https://www.jmlr.org/papers/v15/waegeman14a.html).
4. Dembczyński, K. et al. *Optimizing the F-Measure in Multi-Label Classification*. ICML 2013. [PMLR](https://proceedings.mlr.press/v28/dembczynski13.html).
5. Geifman, Y. và El-Yaniv, R. *Selective Classification for Deep Neural Networks*. NeurIPS 2017. [Proceedings](https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html).
6. Nam, J. et al. *Learning Context-dependent Label Permutations for Multi-label Classification*. ICML 2019. [PMLR](https://proceedings.mlr.press/v97/nam19a.html).
7. Read, J., Martino, L. và Luengo, D. *Efficient Monte Carlo Methods for Multi-Dimensional Learning with Classifier Chains*. [arXiv](https://arxiv.org/abs/1211.2190).
8. Opitz, J. và Burst, S. *Macro F1 and Macro F1*. [arXiv](https://arxiv.org/abs/1911.03347).
9. scikit-learn. *CalibratedClassifierCV*. [Official documentation](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html).
