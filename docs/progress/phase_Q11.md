# Handoff Phase Q11 — System verification và reproducible smoke benchmark

**Trạng thái:** Hoàn thành ngày 2026-09-02.

## Objective đã hoàn thành

- Chạy toàn bộ test Q1–Q11 và legacy regression; không còn blocker/high issue.
- Rà schema v3, config/pair hash, model aliases/registry, deterministic seed, checkpoint completeness và run manifest bằng test + audit script.
- Chạy smoke thật `emotions`, 2 folds, đủ 12 matched IDs, costs `0.3/0.5` và operating-point selection trên inner validation.
- Chạy lại cùng lệnh và xác nhận 12 checkpoint files không đổi timestamp: không fold nào bị fit lại.
- Audit output tự động thay vì chỉ nhìn thủ công: strict JSON, 12 pairs, 24 fold results, 9 CSV, 7 figures và các metric invariants đều pass.
- Cập nhật README với schema-v3 matrix, smoke/resume/audit commands, label-policy/operating-point contract và full-run workflow.
- Freeze primary 10-dataset/12-model config cùng SHA-256; runner từ chối chạy nếu checksum lệch.

## Files đã sửa/tạo

| File | Thay đổi Q11 |
|---|---|
| `scripts/audit_v3_results.py` | Audit strict manifest/checkpoints/hash/folds/costs/CSV/figures/invariants |
| `scripts/run_frozen_experiment.py` | Verify SHA-256 và chạy/resume public pipeline từ frozen config |
| `scripts/__init__.py` | Khai báo package helpers |
| `configs/full_run.json` | Scientific settings primary full run đã freeze |
| `configs/full_run.sha256` | Companion checksum của đúng bytes config |
| `tests/test_system_verification.py` | Frozen-config contract và end-to-end tiny audit tests |
| `README.md` | 12-ID schema-v3 smoke, audit, resume và full-run commands |
| `.gitignore` | Loại schema-v3 benchmark outputs tái tạo được khỏi Git |
| `specify.md` | Đánh dấu Q11 hoàn thành |

## Frozen full-run contract

File: `configs/full_run.json`

```text
SHA-256 = 4036af255fcdc462f9d5307b924cc0d1a1004b02309f749ac66826a9bc61fd1e
schema/status = 1/frozen
datasets = 10
models = 12 matched Logistic/MLP/calibrated-SVM IDs
folds/seed = 5/42
costs = 0.20, 0.25, 0.30, 0.35, 0.40, 0.50
selection objective/policy = full_macro_f1/hamming
partition/order = learned/correlation
operating rule/scope = min_generalized_loss/inner_validation
output = results_pa_v3_full
```

`--max-new-folds` là giới hạn vận hành do runner thêm sau checksum verification và không nằm trong scientific config/hash. Không sửa file frozen sau khi bắt đầu Q12; setting mới phải có config/checksum/output directory mới.

## Smoke command và kết quả

```powershell
python main.py --datasets emotions `
  --models BR_Logistic BR_MLP CC_Logistic CC_MLP MLC_PA_Logistic MLC_PA_MLP GSI_MLC_PA_Logistic GSI_MLC_PA_MLP BR_SVM CC_SVM MLC_PA_SVM GSI_MLC_PA_SVM `
  --n_splits 2 --random_state 42 `
  --abstention_costs 0.3 0.5 --report_cost 0.3 `
  --result_schema 3 --output_dir results_pa_v3_smoke `
  --label_policy_path configs/label_policy.json `
  --operating_point_rule min_generalized_loss `
  --operating_validation_size 0.2
```

Kết quả:

```text
Status                         complete
Run config hash                2a8c08daeccbf1b4
Dataset/model pairs            12/12
Completed fold results         24/24
Selective pairs                6 (MLC-PA + GSI × 3 bases)
c=0.5 coverage                 1.0 trên mọi selective fold
Operating selection scope      inner_validation
Critical-label deployment      N/A (policy file chưa có domain labels)
Checkpoint resume              PASS, 12 files không đổi
```

Smoke chỉ là system verification trên 2 folds; các Macro-F1 quan sát được không dùng để xếp hạng hoặc kết luận nghiên cứu. BR và MLC-PA cùng base có cùng Full Macro-F1 trong smoke như invariant đã thiết kế; GSI operating choices được lưu riêng cho từng outer fold.

## Machine audit

```powershell
python scripts/audit_v3_results.py results_pa_v3_smoke `
  --datasets emotions --expect-primary-models --folds 2 --costs 0.3 0.5
```

Output:

```json
{
  "status": "PASS",
  "config_hash": "2a8c08daeccbf1b4",
  "datasets": 1,
  "models": 12,
  "folds_per_pair": 2,
  "model_dataset_pairs": 12,
  "selective_pairs": 6,
  "csv_artifacts": 9,
  "figure_artifacts": 7
}
```

Audit kiểm tra:

- JSON parse strict, không chấp nhận `NaN`/`Infinity` literal;
- manifest hash bằng canonical settings hash và directory hash;
- pair checkpoint hash/config/status/fold set khớp summary;
- checkpoint có dataset/split fingerprints, model signature, label-policy hash, operating point và evaluation policy;
- model/dataset matrix và cost grid đúng expectation;
- `Hamming Accuracy + Hamming Loss = 1` trên từng Full fold;
- `c=0.5` có `Coverage=1`, `AABS=0` cho mọi selective fold;
- deployment CSV không gắn scope `Full`, coverage luôn có và risk chỉ được `N/A` khi coverage bằng 0;
- đủ 9 non-empty CSV files và 7 non-empty PNG files.

## Test/lệnh đã chạy + kết quả

```text
python -m unittest tests.test_system_verification -v
```

- PASS 2/2 Q11 tests, gồm end-to-end tiny complete/selective run qua audit thật.

```text
python -m unittest discover -s tests -v
```

- PASS 95/95 tests Q1–Q11.

```text
python tests_unit.py
```

- PASS toàn bộ 15 nhóm legacy regression/unit/integration.

```text
python -m compileall -q main.py src scripts tests
python main.py --help
python scripts/run_frozen_experiment.py --config configs/full_run.json --dry-run
git diff --check
```

- PASS syntax/import, CLI, frozen checksum/settings và whitespace check; Git chỉ cảnh báo LF/CRLF trên Windows.

## Output/cache được tạo

- `results_pa_v3_smoke/checkpoints/`: 12 complete pair checkpoints, khoảng 676 KB.
- `results_pa_v3_smoke/tables/2a8c08daeccbf1b4/`: strict JSON + 9 CSV, khoảng 950 KB.
- `results_pa_v3_smoke/figures/2a8c08daeccbf1b4/`: 7 PNG, khoảng 661 KB.
- Output được giữ trong workspace để kiểm chứng/resume nhưng bị `.gitignore`; không commit binary/generated benchmark files.
- `results/` và `results_pa/` schema-v2 không thay đổi trong Git.

## Việc chưa hoàn thành hoặc blocker

- Chưa chạy primary full benchmark 10 datasets; Q12 thực hiện theo nhiều phase quota/checkpoint.
- Chưa chạy objective/partition/random-matched grid đầy đủ; xếp sau primary matched baselines trong Q12 queue.
- Chưa có domain owner content trong `configs/label_policy.json`; critical/cost-sensitive claims tiếp tục là `N/A` cho tới khi được cung cấp.
- Chưa làm statistical analysis/report final; thuộc Q13/Q14.
- Không có blocker kỹ thuật cho Q12.1.

## Git status và thay đổi ngoài phạm vi

- Q11 bắt đầu từ commit Q10 `d53869f` trên nhánh `Khanh_1` và được gom thành một phase commit riêng sau handoff này.
- `note.txt` vẫn ở trạng thái deleted có từ trước Q7–Q11, không thuộc phase và không được stage.
- Local branch đang đi trước `origin/Khanh_1`; chưa push vì lượt này người dùng chỉ yêu cầu commit theo phase.

## Entry point/lệnh đầu tiên cho Phase Q12.1

Xác minh frozen config rồi dispatch tối đa một quota folds:

```powershell
python scripts/run_frozen_experiment.py `
  --config configs/full_run.json --dry-run

python scripts/run_frozen_experiment.py `
  --config configs/full_run.json --max-new-folds 20
```

Sau phiên:

```powershell
python scripts/audit_v3_results.py results_pa_v3_full `
  --expect-primary-models --folds 5 `
  --costs 0.2 0.25 0.3 0.35 0.4 0.5
```

Audit đầy đủ sẽ chỉ PASS khi toàn bộ run complete. Nếu Q12.1 dừng partial đúng `--max-new-folds`, kiểm tra checkpoint JSON trực tiếp, ghi completed/missing queue trong `docs/progress/phase_Q12.1.md`, rồi Q12.2 resume cùng frozen config.
