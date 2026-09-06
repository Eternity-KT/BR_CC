# Progress Phase Q12.10 — Hoàn tất Objective Ablation (6/6) & Chuyển sang Partition Ablation

**Trạng thái:** Sẵn sàng chạy Partition Ablation, ngày 2026-09-05.

## Hoàn thành trong checkpoint này

- Đã hoàn tất toàn bộ 6 selection objective jobs (10 datasets × 5 folds = 50 folds/job):
  1. `immediate_instance_f1`: **50/50 folds** [PASS, hash `e30c6f137f743e2b`]
  2. `bop_instance_f1`: **50/50 folds** [PASS, hash `c4efe74b55fe3ca5`]
  3. `bop_jaccard`: **50/50 folds** [tables hash `6560e8e34273a1e5`]
  4. `macro_precision`: **50/50 folds** [tables hash `fdabcc85b180ff96`]
  5. `macro_recall`: **50/50 folds** [tables hash `32dc79ca45098584`]
  6. `f_beta_0_5`: **50/50 folds** [PASS, hash `5aa3261904aa5157`]
- Đã loại bỏ `f_beta_2` khỏi `configs/ablation_run.json` theo yêu cầu.
- Checksum SHA-256 mới của `configs/ablation_run.json`: `df236c86c778369ef15153f7b20350aa894ee3741161ebca6e5fc2a8136678af`.
- Toàn bộ unit test runner trong `tests/test_ablation_runner.py` (39 jobs) đều PASS.

## Queue tiếp theo: Partition Ablation (3 jobs = 450 folds)

Job tiếp theo ngay lập tức trong queue là **`partition/all_il`**:
- Models: `GSI_MLC_PA_Logistic`, `GSI_MLC_PA_MLP`, `GSI_MLC_PA_SVM`
- Mode: `all_il` (toàn bộ là Independent Labels)
- 3 models × 10 datasets × 5 folds = **150 folds**.

Lệnh khởi động runner để chạy tiếp:

```powershell
# Chạy 1 job trọn vẹn (150 folds của all_il) rồi dừng an toàn:
python scripts/run_frozen_ablation.py --config configs/ablation_run.json --max-new-folds 150

# Hoặc chạy liên tục không giới hạn quota:
python scripts/run_frozen_ablation.py --config configs/ablation_run.json
```
