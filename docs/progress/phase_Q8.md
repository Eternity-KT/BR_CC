# Handoff Phase Q8 — Shared registry và matched Logistic/MLP baselines

**Trạng thái:** Hoàn thành ngày 2026-09-02.

## Objective đã hoàn thành

- Tạo shared base-learner factory dùng chung cho BR, CC, MLC-PA và GSI-MLC-PA.
- Tạo `configs/experiment.json` làm nguồn duy nhất cho model IDs, backend và hyperparameters.
- Loại hai factory binary estimator trùng nhau trong BR/CC; compatibility wrapper hiện cùng delegate về shared factory.
- Đăng ký đủ tám matched Logistic/MLP IDs:
  - `BR_Logistic`, `BR_MLP`;
  - `CC_Logistic`, `CC_MLP`;
  - `MLC_PA_Logistic`, `MLC_PA_MLP`;
  - `GSI_MLC_PA_Logistic`, `GSI_MLC_PA_MLP`.
- Cho GSI nhận `base_learner` module hóa, không thay selection, marginalization, partition hay decision-policy core.
- Loại silent PyTorch-to-sklearn fallback của MLP. Backend thiếu gây `BackendUnavailableError`; sklearn MLP chỉ còn preset tường minh `mlp_sklearn` và không dùng cùng registered ID.
- Lưu experiment manifest JSON-safe vào model metadata, schema-v2 settings và schema-v3 model signature/config hash.
- Đổi CLI default sang tám matched IDs; aliases cũ và legacy `BR`, `CC`, `MLC_PA`, `GSI_MLC_PA` vẫn hoạt động.
- Chứng minh BR và MLC-PA cùng base sinh probability/full predictions giống hệt nhau trên tiny fixture.

## Files đã sửa/tạo

| File | Thay đổi Q8 |
|---|---|
| `configs/experiment.json` | Preregister base learners, 8 model IDs, backend và hyperparameters |
| `src/models/base_learners.py` | Shared binary/multilabel factory, config loader và explicit backend errors |
| `src/models/registry.py` | Matched model registry, aliases, factory, family/base lookup và manifest |
| `src/models/binary_relevance.py` | Delegate binary factory; loại silent MLP fallback |
| `src/models/classifier_chain.py` | Delegate factory trùng sang shared module |
| `src/models/mlc_pa.py` | Delegate marginal estimator sang shared factory |
| `src/models/gsi_mlc_pa.py` | Inject shared `base_learner` cho BR/CC components |
| `src/models/__init__.py` | Export registry API |
| `src/evaluation/pipeline_v3.py` | Family-aware policy và manifest-backed model signature |
| `main.py` | Registered factory/aliases, eight-ID defaults, family-aware evaluator/cache |
| `requirements.txt` | Khai báo PyTorch cho registered MLP backend |
| `tests/test_model_registry.py` | 6 test registry/config/backend/invariant/smoke Q8 |
| `specify.md` | Đánh dấu Q8 hoàn thành |

## Registry và compatibility contract

| Base learner | Backend registered | Shared parameters chính |
|---|---|---|
| `logistic` | `sklearn.linear_model.LogisticRegression` | `C=1`, `solver=liblinear`, `tol=1e-3`, `max_iter=1000` |
| `mlp` | PyTorch | binary `(64,)`; multilabel `(128,64)`; `lr=1e-3`, `weight_decay=1e-3`, `epochs=30` |

Các quyết định:

1. Registered `*_MLP` luôn có backend `pytorch`. Không đổi ID theo package availability hoặc device; CPU/CUDA vẫn là hai devices của cùng PyTorch implementation.
2. `mlp_sklearn` là preset legacy tường minh với backend/config khác và không nằm trong tám primary IDs.
3. `svm_raw` giữ LinearSVC legacy để không phá aliases cũ, nhưng không nằm trong matched primary registry vì chưa calibration; Q9 sẽ đăng ký `svm_calibrated`.
4. BR/MLC/GSI marginal component gọi cùng `create_multilabel_estimator`; CC/GSI chain component gọi cùng `create_binary_estimator`.
5. MLP vẫn dùng multi-output network cho BR marginal estimator và binary network cho từng CC node; cả hai architecture/hyperparameters được ghi riêng trong cùng base manifest, không bị mô tả như một estimator giống hệt.
6. Registered model manifest chứa model ID, family, selective flag, exact base backend, binary/multilabel parameters và wrapper parameters. Estimator objects được chuyển thành class + parameter tree JSON-safe trước khi hash.
7. Legacy aliases được canonicalize nhưng không trộn với registered IDs có semantics khác. `MLC_PA` tiếp tục nhận `--mlc_pa_base`; `GSI_MLC_PA` giữ MLP legacy behavior.
8. Probability algorithm của BR/CC, MLC-PA decision layer và GSI selection/inference không đổi; Q8 chỉ thống nhất construction/configuration.

## Primary model IDs

```text
BR_Logistic              BR_MLP
CC_Logistic              CC_MLP
MLC_PA_Logistic          MLC_PA_MLP
GSI_MLC_PA_Logistic      GSI_MLC_PA_MLP
```

CLI không truyền `--models` sẽ chạy danh sách trên. Có thể tiếp tục gọi legacy IDs trực tiếp khi cần kiểm tra cache cũ.

## Test/lệnh đã chạy + kết quả

```text
python -m unittest tests.test_model_registry -v
```

- PASS 6/6 test Q8.
- Factory config/hyperparameters đúng manifest.
- Backend-unavailable test chứng minh `mlp` raise rõ ràng và không fallback; `mlp_sklearn` chỉ tạo khi gọi tường minh.
- Cả tám IDs tạo JSON-safe manifest, config hash và fit/predict/evaluate thành công trên tiny data.

BR/MLC invariant cho cả Logistic và MLP:

```text
max absolute probability difference = 0
full prediction mismatch count      = 0
assert_allclose rtol=0, atol=0       = PASS
```

```text
python -m unittest discover -s tests -v
```

- PASS 76/76 test Q1–Q8.

```text
python tests_unit.py
```

- PASS toàn bộ 15 nhóm legacy regression/unit/integration.

```text
python -m compileall -q main.py src tests
python main.py --help
git diff --check
```

- PASS syntax/import, eight-ID CLI default description và whitespace check; Git chỉ cảnh báo LF/CRLF trên Windows.

## Output/cache được tạo

- Q8 smoke dùng tiny arrays trong bộ nhớ; không chạy dataset benchmark và không tạo output/cache nghiên cứu.
- Không sửa `results/`, `results_pa/` hoặc Q7 smoke output.
- Config/model signatures mới làm schema-v3 hashes khác đúng chủ đích; legacy caches không bị ghi đè.

## Việc chưa hoàn thành hoặc blocker

- Chưa có calibrated SVM hoặc 12 matched IDs; đây là Q9.
- `svm_raw` hiện chỉ là legacy margin-to-sigmoid behavior và không được dùng làm calibrated probability baseline.
- Chưa có Brier/log-loss/ECE/reliability export; đây là Q9.
- Chưa chạy `emotions` 2-fold với đủ 8 IDs; system smoke 12 IDs thuộc Q11, Q8 chỉ dùng tiny smoke theo đặc tả.
- Không có blocker kỹ thuật cho Q9.

## Git status và thay đổi ngoài phạm vi

- Q8 bắt đầu từ commit Q7 `58c4bc8` trên nhánh `Khanh_1` và được gom trong một phase commit riêng sau handoff này.
- `note.txt` vẫn ở trạng thái deleted có từ trước Q8, không thuộc phase và không được stage.
- Nhánh local đang đi trước `origin/Khanh_1`; chưa push trong Q7/Q8 vì yêu cầu hiện tại chỉ yêu cầu commit theo phase.

## Entry point/lệnh đầu tiên cho Phase Q9

```powershell
python -m unittest discover -s tests -v
```

Sau baseline, tạo `ProbabilityAdapter` cho `svm_calibrated` dùng sigmoid/Platt calibration chỉ trong outer-train. Đăng ký bốn SVM IDs vào cùng registry, thêm deterministic fallback cho constant/rare labels và calibration manifest. Viết split spy test trước khi nối Brier, log loss, ECE và reliability data vào schema-v3.
