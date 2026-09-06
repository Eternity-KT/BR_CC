# Progress Phase Q12.9 — Hoàn tất BOP Instance F1 & Chuyển sang BOP Jaccard

**Trạng thái:** Checkpoint-safe, ngày 2026-09-05.

## Hoàn thành trong checkpoint này

- Hoàn tất toàn bộ 10 dataset pairs (`emotions`, `music`, `scene`, `yeast`, `genbase`, `medical`, `enron`, `cal500`, `bibtex`, `reuters-k500`) của `objective/bop_instance_f1`, Logistic GSI.
- Job objective thứ hai đã đủ 10 datasets × 5 outer folds = **50/50 folds**.
- Audit độc lập trên thư mục output `results_pa_v3_ablation/objective/bop_instance_f1` đã **PASS** (exit code 0):
  - Config hash: `c4efe74b55fe3ca5`
  - Model: `GSI_MLC_PA_Logistic`
  - 10 dataset pairs, 5 folds/pair, đầy đủ cost grid (0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
  - 9 CSV artifacts và 7 figure artifacts
  - Manifest: `results_pa_v3_ablation/objective/bop_instance_f1/tables/c4efe74b55fe3ca5/results_v3.json`

Lệnh audit đã chạy thành công:

```powershell
python scripts/audit_v3_results.py results_pa_v3_ablation/objective/bop_instance_f1 `
  --datasets emotions music scene yeast genbase medical enron cal500 bibtex reuters-k500 `
  --models GSI_MLC_PA_Logistic --folds 5 `
  --costs 0.2 0.25 0.3 0.35 0.4 0.5
```

## Job tiếp theo trong queue

Job tiếp theo theo thứ tự preregistered trong `configs/ablation_run.json` là **`objective/bop_jaccard`** (0/50 folds):
- Base model: `GSI_MLC_PA_Logistic`
- Selection objective: `bop_jaccard`
- 10 datasets × 5 folds = 50 folds.
- Output directory: `results_pa_v3_ablation/objective/bop_jaccard`

Lệnh chạy tiếp:

```powershell
# Chạy 5 folds đầu tiên (emotions) hoặc theo quota kiểm soát:
python scripts/run_frozen_ablation.py --config configs/ablation_run.json --max-new-folds 5

# Hoặc chạy trọn vẹn cả job bop_jaccard (50 folds) rồi dừng an toàn:
python scripts/run_frozen_ablation.py --config configs/ablation_run.json --max-new-folds 50

# Hoặc chạy liên tục không giới hạn quota:
python scripts/run_frozen_ablation.py --config configs/ablation_run.json
```

Sau khi `bop_jaccard` đủ 50/50 folds, thực hiện audit:

```powershell
python scripts/audit_v3_results.py results_pa_v3_ablation/objective/bop_jaccard `
  --datasets emotions music scene yeast genbase medical enron cal500 bibtex reuters-k500 `
  --models GSI_MLC_PA_Logistic --folds 5 `
  --costs 0.2 0.25 0.3 0.35 0.4 0.5
```

## Toàn bộ tiến độ Objective Ablation (2/7 hoàn thành)

1. `immediate_instance_f1`: **50/50 folds** — [PASS] Hash `e30c6f137f743e2b`
2. `bop_instance_f1`: **50/50 folds** — [PASS] Hash `c4efe74b55fe3ca5`
3. `bop_jaccard`: **0/50 folds** — *Sẵn sàng chạy*
4. `macro_precision`: 0/50 folds
5. `macro_recall`: 0/50 folds
6. `f_beta_0_5`: 0/50 folds
7. `f_beta_2`: 0/50 folds
