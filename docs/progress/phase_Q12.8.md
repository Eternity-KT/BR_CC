# Progress Phase Q12.8 — Objective ablation tiếp tục

**Trạng thái:** Đang chạy, checkpoint-safe, ngày 2026-09-03.

## Hoàn thành trong checkpoint này

- Hoàn tất hai dataset pair còn lại (`bibtex`, `reuters-k500`) của
  `objective/immediate_instance_f1`, Logistic GSI.
- Job objective đầu tiên nay đủ 10 datasets × 5 outer folds = **50/50 folds**.
- Audit độc lập trên output riêng của job PASS, hash `e30c6f137f743e2b`:
  10 dataset pairs, 5 folds/pair, cost grid đầy đủ, 9 CSV và 7 figures.

Lệnh audit đã chạy:

```powershell
python scripts/audit_v3_results.py results_pa_v3_ablation/objective/immediate_instance_f1 `
  --datasets emotions music scene yeast genbase medical enron cal500 bibtex reuters-k500 `
  --models GSI_MLC_PA_Logistic --folds 5 `
  --costs 0.2 0.25 0.3 0.35 0.4 0.5
```

## Đang chạy

`objective/bop_instance_f1` đã checkpoint **8/10 dataset pairs = 40/50
folds**: `emotions`, `music`, `scene`, `yeast`, `genbase`, `medical`,
`enron`, và `cal500`. Runner đang tiếp tục `bibtex`, sau đó là
`reuters-k500`; output hiện tại chỉ được audit khi job đủ 50/50.

Đây là policy F1-BOP partial exact, nên `bibtex` có thể không in dòng mới
trong thời gian dài giữa hai checkpoint fold. Không thay đổi policy, datasets,
seeds hoặc cost grid để rút ngắn run.

```powershell
python scripts/run_frozen_ablation.py `
  --config configs/ablation_run.json --max-new-folds 5
```

## Queue tiếp theo

1. Hoàn tất và audit sáu objective jobs còn lại.
2. Chạy/audit ba partition endpoint/order mode cho Logistic, MLP và SVM.
3. Chạy 30 random-matched partition seeds trên cùng ba base learners.
4. Chỉ mở Q13 sau khi toàn bộ ablation đã audit.

## Bảo toàn tái lập

- Manifest: `configs/ablation_run.json`.
- SHA-256: `6033e030a5e45c22cb40f2d080167f9ac5dab4614ef94b253889b0d85e772ee8`.
- Outer/model/inner seeds giữ `42`; chỉ random-matched được phép đổi
  `gsi_partition_random_state` (1001–1030).
- Generated outputs nằm trong `results_pa_v3_ablation/` và được git-ignore.
