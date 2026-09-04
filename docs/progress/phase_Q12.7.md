# Handoff Phase Q12.7 — Hoàn tất primary queue và mở objective ablation

**Trạng thái:** Hoàn thành checkpoint ngày 2026-09-03; có thể resume an toàn.

## Objective đã hoàn thành

- Resume ba pair SVM còn lại của `bibtex` và đủ 12 pair của `reuters-k500`.
- Hoàn tất primary benchmark: 120/120 model-dataset pairs, 600/600 outer folds.
- Chạy strict full audit trên run hash `2a300c396384c5e9`: PASS.
- Tách `gsi_partition_random_state` khỏi outer/model `random_state`; default `None` giữ nguyên model signature và cache hash cũ.
- Khóa ablation manifest trước khi đọc metric values: 7 objective Logistic, ba endpoint/order mode trên ba base và 30 random-matched seeds trên ba base.
- Vector hóa exact F-beta/Jaccard count-distribution tables. Exhaustive oracle tests xác nhận action/utility không đổi.
- Chạy 8/10 dataset pairs đầu tiên của objective `immediate_instance_f1`: 40/50 folds; không có partial pair.

## Primary audit

Lệnh:

```powershell
python scripts/audit_v3_results.py results_pa_v3_full `
  --config-hash 2a300c396384c5e9 `
  --expect-primary-models --folds 5 `
  --costs 0.2 0.25 0.3 0.35 0.4 0.5
```

Kết quả:

```json
{
  "status": "PASS",
  "config_hash": "2a300c396384c5e9",
  "datasets": 10,
  "models": 12,
  "folds_per_pair": 5,
  "model_dataset_pairs": 120,
  "selective_pairs": 60,
  "csv_artifacts": 9,
  "figure_artifacts": 7
}
```

Không có process primary còn chạy. Generated artifacts tiếp tục nằm trong thư mục ignored `results_pa_v3_full/`.

## Ablation preregistration

- Config: `configs/ablation_run.json`.
- SHA-256: `6033e030a5e45c22cb40f2d080167f9ac5dab4614ef94b253889b0d85e772ee8`.
- Expanded jobs: 40.
- Outer folds, inner split và estimator seed giữ cố định `42`.
- 30 repetition chỉ đổi `gsi_partition_random_state=1001..1030`.
- Objective ablation giới hạn Logistic để tránh confound và giữ chi phí khả thi; final decision policy vẫn là Hamming.
- Partition ablation dùng đủ Logistic/MLP/calibrated SVM với cùng cost grid, policy và outer folds.

## Tối ưu exact BOP

Lần chạy Cal500 đầu tiên cho `immediate_instance_f1` dùng recurrence Python cũ đã vượt 15 phút mà chưa hoàn tất một fold. Process được dừng, nhưng child PID còn chạy nền đã được phát hiện và terminate sau khi xác minh command line để tránh ghi đè checkpoint mới.

Implementation mới vẫn dùng đúng Algorithm 2/3, exact dưới CLI và vẫn O(K^3), nhưng:

- cache prefix/suffix Poisson-binomial count distributions;
- lập bảng utility cho mọi boundary;
- dùng NumPy/BLAS thay vòng lặp Python lồng nhau;
- giữ nguyên tie tolerance, ưu tiên nhiều quyết định hơn và stable label-index order.

Benchmark policy 100 rows × 174 labels:

| Policy | Thời gian |
|---|---:|
| Complete F1-BOP | 0.697 s |
| Partial F1-BOP | 0.716 s |
| Partial Jaccard-BOP | 0.257 s |

Cal500 end-to-end giảm còn khoảng 107–119 giây/fold cho objective này. Năm fold đều checkpoint complete.

## Tests/checks

```text
python -m unittest tests.test_partition_modes tests.test_model_registry tests.test_pipeline_v3 -v
25/25 PASS

python -m unittest tests.test_ablation_runner tests.test_partition_modes -v
12/12 PASS

python -m unittest tests.test_decision_fbeta tests.test_decision_jaccard tests.test_selection_objectives -v
27/27 PASS
```

Default GSI signature được so trực tiếp với primary checkpoint: `DefaultSignatureCompatible=True` (1136 bytes ở cả hai phía). Strict primary audit vẫn PASS sau thay đổi.

## Queue còn lại

1. Hoàn tất `immediate_instance_f1`: `bibtex`, `reuters-k500` (10 folds).
2. Chạy sáu objective jobs còn lại (300 folds).
3. Chạy partition endpoints/order (450 folds).
4. Chạy 30 random-matched repetitions (4,500 folds).
5. Audit/merge ablation artifacts, sau đó mới mở Q13.

## Entry point Q12.8

```powershell
python scripts/run_frozen_ablation.py `
  --config configs/ablation_run.json --max-new-folds 5
```

Lệnh tự resume `objective/immediate_instance_f1` tại `bibtex` và không chạy lại 40 fold đã hoàn tất.

## Git/worktree

- Production/test/config/docs thuộc Q12.7 được liệt kê trong `git diff`; generated results bị ignore.
- `note.txt` đã bị xóa từ trước và không thuộc phạm vi phase.
- Base commit khi bắt đầu phase: `aeda47f`.
