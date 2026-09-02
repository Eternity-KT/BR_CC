# Handoff Phase Q9 — SVM calibration và 12 matched baselines

**Trạng thái:** Hoàn thành ngày 2026-09-02.

## Objective đã hoàn thành

- Tạo `ProbabilityAdapter` cho LinearSVC + sigmoid/Platt calibration bằng `CalibratedClassifierCV` nằm hoàn toàn trong training fold của model sở hữu nó.
- Tự giảm calibration CV từ 3 xuống minority-class count khi cần.
- Xử lý constant label và one-minority-sample deterministically, lưu strategy/reason/probability/class counts vào calibration audit.
- Tổng hợp calibration audit cho từng label trong BR/CC, chuyển tiếp qua MLC-PA, và lưu cả selection/final calibration scopes trong GSI.
- Đăng ký thêm bốn calibrated-SVM IDs: `BR_SVM`, `CC_SVM`, `MLC_PA_SVM`, `GSI_MLC_PA_SVM`.
- Hoàn chỉnh registry thành 12 matched Logistic/MLP/calibrated-SVM IDs; `svm_raw` chỉ còn legacy uncalibrated preset.
- Cài aggregate/per-label Brier Score, binary Log Loss, ECE và 10-bin reliability data.
- Xuất `calibration_metrics.csv`, `reliability_data.csv`, runtime calibration audit và experiment manifest qua schema-v3.
- Thêm spy test chứng minh calibrator không được gọi khi outer-test được evaluate và chưa từng nhận outer-test features.
- Smoke cả bốn calibrated-SVM families trên synthetic small dataset.

## Files đã sửa/tạo

| File | Thay đổi Q9 |
|---|---|
| `src/models/probability_adapter.py` | Calibrated binary adapter, rare/constant fallback và audit aggregation |
| `src/evaluation/calibration_metrics.py` | Brier, log loss, ECE, per-label và reliability bins |
| `configs/experiment.json` | `svm_calibrated`, bốn SVM IDs và 12-ID default |
| `src/models/base_learners.py` | Tạo `ProbabilityAdapter` từ shared factory |
| `src/models/registry.py` | 12 matched IDs và calibrated-SVM aliases |
| `src/models/binary_relevance.py` | Cho adapter tự xử lý constant labels và thu calibration audit |
| `src/models/classifier_chain.py` | Tương tự cho từng chain node/label order |
| `src/models/mlc_pa.py` | Chuyển tiếp calibration audit của marginal estimator |
| `src/models/gsi_mlc_pa.py` | Lưu selection/final BR+CC calibration audits |
| `src/evaluation/pipeline_v3.py` | Calibration summary, raw Model Metadata và hash-safe manifest |
| `src/evaluation/export_v3.py` | Calibration/reliability CSV exports |
| `src/evaluation/__init__.py` | Export calibration metric API |
| `main.py` | Probability evaluation cho mọi family và 12-ID CLI default |
| `tests/test_model_registry.py` | Giữ Q8 contract riêng cho 8 Logistic/MLP IDs |
| `tests/test_svm_calibration.py` | 8 test calibration/fallback/registry/smoke/leakage Q9 |
| `specify.md` | Đánh dấu Q9 hoàn thành |

## Calibration contract

Luồng calibrated SVM:

```text
outer-train supplied to family model
  -> per-label ProbabilityAdapter.fit
     -> nested stratified calibration folds inside supplied train data
     -> LinearSVC margins + sigmoid/Platt calibration
  -> fitted probability estimator
outer-test -> predict_proba only; never fit/calibrate
```

Với GSI:

- selection BR/CC adapters chỉ fit trên inner-train;
- selected partition chỉ score trên inner-validation;
- final BR/CC adapters refit/calibrate trên toàn outer-train;
- outer-test chỉ đi qua `predict_proba` sau khi partition/order/calibration đã freeze.

Fallback contract:

| Training label state | Strategy | Probability |
|---|---|---|
| Chỉ quan sát class 0 | `constant_label` | `0` |
| Chỉ quan sát class 1 | `constant_label` | `1` |
| Minority class count = 1 | `smoothed_prior` | `(positive_count + 1)/(n + 2)` |
| Minority count >= 2 | `calibrated_sigmoid_cv` | Platt calibration với `effective_cv=min(3, minority_count)` |

Audit không lưu feature/target arrays; chỉ lưu counts, sample count, method, requested/effective CV, fallback reason/probability và `outer_test_access=false`.

## 12 primary matched IDs

| Family | Logistic | MLP | Calibrated SVM |
|---|---|---|---|
| BR | `BR_Logistic` | `BR_MLP` | `BR_SVM` |
| CC | `CC_Logistic` | `CC_MLP` | `CC_SVM` |
| MLC-PA | `MLC_PA_Logistic` | `MLC_PA_MLP` | `MLC_PA_SVM` |
| GSI-MLC-PA | `GSI_MLC_PA_Logistic` | `GSI_MLC_PA_MLP` | `GSI_MLC_PA_SVM` |

`BR`/`CC` legacy aliases vẫn dùng `svm_raw`; chúng không được đánh tráo thành calibrated models và không nằm trong 12-ID primary comparison.

## Calibration metric contract

- Brier Score và binary Log Loss được tính trên toàn bộ sample-label pairs.
- ECE dùng 10 equal-width probability bins mặc định, weighted theo count của sample-label pairs.
- Reliability data ghi bin bounds, count, mean predicted probability, observed positive rate và absolute gap.
- Per-label Brier/Log Loss/ECE dùng cùng công thức và label names của fold.
- Empty reliability bins giữ count `0`; các giá trị không xác định được strict-JSON exporter chuyển thành `null`, không ghi `NaN` literal.

Schema-v3 thêm:

```text
Calibration.Metrics
Calibration.Per Label
Calibration.Reliability
Model Metadata.Calibration Audit
calibration_metrics.csv
reliability_data.csv
```

## Test/lệnh đã chạy + kết quả

```text
python -m unittest tests.test_svm_calibration -v
```

- PASS 8/8 test Q9.
- Balanced fixture dùng sigmoid CV; rare/constant fixtures deterministic.
- Registry đủ 12 IDs và bốn SVM family manifests cùng ghi `scope=nested_in_training_fold`.
- Bốn SVM families fit/predict/evaluate được trên small synthetic data; calibration/reliability CSV đều non-empty.
- `BR_SVM` và `MLC_PA_SVM` có probability/full predictions giống hệt (`rtol=0`, `atol=0`).
- Spy test GSI xác nhận outer-test sentinel `999` không xuất hiện trong bất kỳ `ProbabilityAdapter.fit` input nào và evaluation không phát sinh fit call mới.

```text
python -m unittest discover -s tests -v
```

- PASS 84/84 test Q1–Q9.

```text
python tests_unit.py
```

- PASS toàn bộ 15 nhóm legacy regression/unit/integration.

```text
python -m compileall -q main.py src tests
python main.py --help
git diff --check
```

- PASS syntax/import, 12-ID CLI default description và whitespace check; Git chỉ cảnh báo LF/CRLF trên Windows.

## Output/cache được tạo

- Q9 smoke/tests dùng small synthetic arrays và temporary directories tự xóa.
- Không chạy dataset benchmark, không sửa `results/`, `results_pa/` hoặc Q7 smoke output.
- Registered SVM model signatures chứa calibration settings nên không thể dùng nhầm cache `svm_raw`.

## Việc chưa hoàn thành hoặc blocker

- Chưa có deployment metrics, label-policy config, operating-point selection hoặc deployment plots; đây là Q10.
- Chưa chạy reproducible `emotions` smoke đủ 12 IDs; đây là Q11.
- Chưa chạy full benchmark/cost grid; đây là Q12.
- Reliability/ECE hiện dùng fixed equal-width bins; không claim đây là duy nhất hoặc tối ưu cho mọi dataset size.
- Không có blocker kỹ thuật cho Q10.

## Git status và thay đổi ngoài phạm vi

- Q9 bắt đầu từ commit Q8 `3c99f29` trên nhánh `Khanh_1` và được gom trong một phase commit riêng sau handoff này.
- `note.txt` vẫn ở trạng thái deleted có từ trước Q9, không thuộc phase và không được stage.
- Nhánh local đang đi trước `origin/Khanh_1`; chưa push Q7–Q9.

## Entry point/lệnh đầu tiên cho Phase Q10

```powershell
python -m unittest discover -s tests -v
```

Sau baseline, tạo `src/evaluation/deployment.py` và validate `configs/label_policy.json` mà không suy label importance từ outer test. Viết fixture cho risk-at-coverage/AURC, review load, error-capture efficiency và optimistic gain trước; sau đó mới nối operating-point selection trên inner validation và CSV/plots.
