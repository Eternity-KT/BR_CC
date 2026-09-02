# Handoff Phase Q12.5 — Primary full run: genbase

**Trạng thái:** Hoàn thành ngày 2026-09-02.

## Objective đã hoàn thành

- Resume canonical frozen run, bỏ qua 240 folds đã hoàn thành.
- Hoàn tất `genbase`, đủ 12 matched IDs × 5 folds = 60 folds mới.
- Audit 60 complete pairs/300 folds của năm dataset; đạt đúng 50% primary queue.
- Không có partial/failed/duplicate checkpoint.

## Lệnh và runtime

```powershell
python scripts/run_frozen_experiment.py `
  --config configs/full_run.json --max-new-folds 60
```

- Checksum frozen config: PASS.
- 60 newly checkpointed folds: PASS.
- Runtime khoảng 100 giây. `genbase` có 27 labels/1,186 features nên GSI/calibrated-SVM chậm hơn block nhỏ, vẫn nằm an toàn trong quota.

## Audit

```json
{
  "status": "PASS",
  "config_hash": "2a300c396384c5e9",
  "completed_pairs": 60,
  "completed_folds": 300,
  "partial_pairs": 0,
  "missing_pairs": 60,
  "next_dataset": "medical"
}
```

Audit recompute canonical run hash, parse strict 60 checkpoints, xác minh schema/status/pair hashes và fold set `[1,2,3,4,5]`. Result objects được so theo key set vì strict exporter sắp xếp JSON object keys; dataset execution order tiếp tục lấy từ `Settings.datasets` frozen.

## Queue state

| Dataset | Pairs | Folds | Trạng thái |
|---|---:|---:|---|
| emotions | 12/12 | 60/60 | complete |
| music | 12/12 | 60/60 | complete |
| scene | 12/12 | 60/60 | complete |
| yeast | 12/12 | 60/60 | complete |
| genbase | 12/12 | 60/60 | complete |
| 5 datasets còn lại | 0/60 | 0/300 | missing |

Primary queue còn lại: `medical -> enron -> cal500 -> bibtex -> reuters-k500`.

## Output/cache

- `results_pa_v3_full/checkpoints/`: 60 complete files, khoảng 19.49 MB.
- Canonical partial tables/figures ở config hash `2a300c396384c5e9`.
- Generated results local/ignored; schema-v2 output không đổi.

## Test/code/commit scope

- Không đổi production/test code trong Q12.5.
- Strict inventory audit PASS 60/60 pairs, 300/300 folds.
- Regression baseline: 96/96 tests và 15/15 legacy groups pass từ Q12.2.
- Phase commit chỉ gồm `docs/progress/phase_Q12.5.md` và dòng tiến độ trong `specify.md`.
- `git diff --check` pass trước commit.

## Blocker/việc còn lại

- Còn 60 pairs/300 primary folds; không có blocker hoặc failed job.
- `medical` có 45 labels; Q12.6 vẫn dispatch một block 60 folds nhưng phải theo dõi runtime trước các dataset lớn hơn.
- Full artifact audit/ablation/statistics chưa bắt đầu.
- `note.txt` deleted từ trước tiếp tục nằm ngoài stage.

## Entry point Phase Q12.6

```powershell
python scripts/run_frozen_experiment.py `
  --config configs/full_run.json --max-new-folds 60
```

Expected: hoàn tất `medical`, đạt 72/120 pairs và 360/600 folds; dataset kế tiếp `enron`.
