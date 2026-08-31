# Handoff Phase Q2 — Pipeline integration, schema v3 và resumability

Ngày hoàn thành: 2026-08-31

Phase tiếp theo: Q3 — Decision-policy interface và Hamming regression

## Objective đã hoàn thành

- Nối metric bundle v3 vào `_evaluate_model` qua chế độ opt-in `metric_schema=3`; đường chạy mặc định vẫn là schema v2.
- Thêm `result_schema=3` cho `run_experiment` và CLI, mặc định dùng output riêng `results_pa_v3`.
- Cài checkpoint atomically sau từng fold cho mỗi `model × dataset × config_hash`.
- Cài resume: fold đã có trong checkpoint bị skip; checkpoint chỉ được đánh dấu complete khi đủ chính xác `1..n_splits`.
- Khóa config hash bằng dataset fingerprint, exact CV split hash, seed, scaler, costs, penalty, model class/parameters/backend, label names, critical labels và metric contract.
- Cài migration/isolation guard: không đọc hoặc ghi đè schema v2 như v3, không dùng `results/`, `results_pa/` hoặc thư mục có `raw_results.json`.
- Cài strict JSON (`NaN/Infinity → null`) và bốn CSV: complete, selective, per-label và group.
- Thêm `--max_new_folds` để dừng ở checkpoint an toàn theo quota và resume bằng cùng lệnh/config.
- Thêm tiny integration, strict-schema, export và interruption/resume tests.

## Files đã sửa/tạo

| File | Thay đổi Q2 |
|---|---|
| `main.py` | Schema dispatcher, evaluator v3 và CLI opt-in; mặc định v2 giữ nguyên |
| `src/evaluation/cache_v3.py` | Config hash, strict serializer, atomic fold checkpoint và migration guard |
| `src/evaluation/export_v3.py` | JSON manifest và scope-specific CSV exports |
| `src/evaluation/pipeline_v3.py` | Isolated CV runner, fold resume, summary và output guard |
| `tests/test_pipeline_v3.py` | 10 schema/integration/export/resume tests |
| `specify.md` | Đánh dấu Q2 hoàn thành |

Các file metric Q1 và `docs/progress/phase_Q1.md` vẫn là thay đổi chưa commit từ phase trước.

## Output schema v3

Mỗi fold evaluator trả:

```text
Schema Version
Metric Contract Version
Full
Per Label
Groups
Critical Labels
Model Metadata
Costs
  <cost>
    Selective
    Rejected
    Optimistic
    Diagnostics
    Per Label
    Groups
    Critical Labels
```

Summary giữ `Mean`, `Std` và `Raw Folds` cho các scalar scopes; per-label/group giữ `Raw Records` có fold index. Alias v2 không được serialize thành cột trùng.

Artifacts chuẩn dưới `<output_dir>/tables/<RUN_CONFIG_HASH>/`:

```text
results_v3.json
complete_metrics.csv
selective_metrics.csv
per_label_metrics.csv
group_metrics.csv
```

Checkpoint nằm tại:

```text
<output_dir>/checkpoints/<MODEL>/<DATASET>.<CONFIG_HASH>.json
```

## Migration và isolation rules

1. Schema v2 vẫn là mặc định của `main.py`; không truyền `--result_schema 3` thì hành vi/cache/plot cũ không đổi.
2. Schema v3 mặc định ghi vào `results_pa_v3`; tuyệt đối không ghi vào `results/` hoặc `results_pa/`.
3. Nếu target có `tables/raw_results.json`, v3 dừng bằng `V3OutputIsolationError` trước khi chạy model.
4. Nếu file ở đúng checkpoint path không có `schema_version=3`, đúng metric contract, config hash, model và dataset, loader dừng bằng `IncompatibleV3CacheError`; file không bị backup, migrate hay overwrite.
5. Thay đổi data, fold indices, seed, preprocessing, cost, model settings/backend hoặc metric contract tạo pair config hash/checkpoint path mới. Checkpoint cũ vẫn được giữ nguyên; artifacts của các run settings khác nhau nằm ở thư mục run hash khác nhau.
6. Fold được ghi bằng temporary file + `fsync` + atomic replace. Duplicate fold bị từ chối thay vì ghi đè.
7. JSON v3 dùng `allow_nan=False`; metric không xác định được biểu diễn bằng `null`. CSV để ô trống tương ứng.
8. `max_new_folds` và output path không thuộc scientific config hash, nên có thể đổi quota/đường dẫn khi resume mà không đổi kết quả khoa học.

## Lệnh chạy/resume

Chạy v3 và chủ động dừng sau một fold mới:

```powershell
python main.py --result_schema 3 --output_dir results_pa_v3 --datasets emotions --models BR_Logistic --n_splits 5 --abstention_costs 0.20 0.25 0.30 0.35 0.40 --report_cost 0.30 --max_new_folds 1
```

Resume cùng scientific config, chạy thêm tối đa một fold:

```powershell
python main.py --result_schema 3 --output_dir results_pa_v3 --datasets emotions --models BR_Logistic --n_splits 5 --abstention_costs 0.20 0.25 0.30 0.35 0.40 --report_cost 0.30 --max_new_folds 1
```

Hoặc bỏ `--max_new_folds` để chạy đến hết. Thay seed, folds, costs hoặc model settings sẽ tạo checkpoint mới thay vì resume checkpoint cũ.

## Quyết định kỹ thuật và giả định

1. Thay vì nâng `src/evaluation/cache.py` tại chỗ, Q2 tạo ba module v3 riêng để giới hạn ảnh hưởng đến core/v2.
2. Acceptance confidence cho AURC hiện là `2 * abs(p - 0.5)` trên probability output. Đây là ranking confidence chung; calibration chính thức vẫn thuộc Q9.
3. Với GSI, evaluator tự map `independent_labels_ → IL` và `dependent_labels_ → DL`; model khác có group rỗng trừ khi phase sau cung cấp group config.
4. `critical_labels` của CLI hiện là danh sách tên dùng chung cho các dataset được chọn. Cấu hình theo dataset trong `configs/label_policy.json` vẫn thuộc Q10; không dùng option này cho nhiều dataset có label namespace khác nhau.
5. V3 không gọi plot schema v2. Visualization riêng cho các scope mới thuộc Q10.
6. Checkpoint lưu train/test size và thời gian fold làm metadata; các giá trị thời gian không tham gia metric summary hoặc so sánh resume-equivalence.

## Test/lệnh đã chạy + kết quả

```text
python -m unittest discover -s tests -v
```

- PASS 30/30 tests: 5 contract, 15 metric Q1 và 10 pipeline Q2.

```text
python -m unittest tests.test_pipeline_v3 -v
```

- PASS 10/10 Q2 tests.
- Tiny selective evaluator sinh Full/Selective/Rejected/Optimistic/Diagnostics/Per Label/Groups/Critical Labels với canonical names.
- Schema-v2 payload ở v3 path bị từ chối và nội dung file giữ nguyên.
- Strict JSON không chứa token `NaN`/`Infinity`.
- Interruption sau fold 1 rồi resume chỉ fit thêm fold 2–3; rerun lần ba fit thêm `0` fold.
- Metric/record summary của resumed run bằng uninterrupted run; checkpoint chứa đúng `(1, 2, 3)` không trùng.
- `main.run_experiment(..., result_schema=3)` dispatch đúng pipeline và sinh artifacts.

```text
python tests_unit.py
```

- PASS toàn bộ 15 nhóm regression/unit/integration cũ.
- Schema-v2 evaluation, cache, GSI selection/refit và plots vẫn hoạt động.

```text
python main.py --help
python -m compileall -q main.py src\evaluation tests
git diff --check
```

- PASS CLI/import/syntax/whitespace; chỉ có cảnh báo LF/CRLF của Git.

## Output/cache được tạo

- Integration tests chỉ tạo artifacts/checkpoints trong temporary directories và tự xóa sau test.
- Không tạo `results_pa_v3/` thật trong workspace.
- Không sửa file nào trong `results/` hoặc `results_pa/`.

## Việc chưa hoàn thành hoặc blocker

- Chưa có decision-policy interface; evaluator v3 vẫn gọi trực tiếp decision rule hiện tại. Đây là Q3.
- Chưa có fold-level plot/visualization v3. Đây là Q10 sau khi policy/calibration/ablation ổn định.
- Chưa có dataset-specific critical-label config loader. Đây là Q10.
- Chưa có calibration; acceptance confidence hiện chỉ là margin quanh `0.5`. Đây là Q9.
- Không có blocker kỹ thuật để bắt đầu Q3.

## Git status và thay đổi ngoài phạm vi

- Nhánh hiện tại: `Khanh_1`, tracking `origin/Khanh_1` trước các thay đổi Q1–Q2.
- `note.txt` vẫn ở trạng thái deleted có từ trước, không thuộc Q1/Q2 và không được chỉnh sửa.
- Các thay đổi Q1–Q2 chưa commit/push tại thời điểm viết handoff này.

## Entry point/lệnh đầu tiên cho Phase Q3

```powershell
python -m unittest discover -s tests -v
```

Sau baseline, đọc `src/models/mlc_pa.py::predict_from_proba`, `src/models/gsi_mlc_pa.py::predict_from_proba` và `tests_unit.py` nhóm 7/8/11. Tạo `src/decision/base.py` cùng `HammingBOPPolicy` wrapper trước, rồi cho evaluator v3 gọi policy qua registry; default SEP/PAR phải giữ output bit-for-bit trên fixture hiện tại.
