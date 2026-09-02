# Handoff Phase Q12.4 — Primary full run: yeast

**Trạng thái:** Hoàn thành ngày 2026-09-02.

## Objective đã hoàn thành

- Resume canonical frozen run, bỏ qua 180 folds của ba dataset trước.
- Hoàn tất `yeast` với 12 matched IDs, 5 folds/model: 60 folds mới.
- Audit strict 48 complete pair checkpoints/240 folds trên `emotions`, `music`, `scene`, `yeast`.
- Không có failed, duplicate hoặc partial pair.

## Lệnh và runtime

```powershell
python scripts/run_frozen_experiment.py `
  --config configs/full_run.json --max-new-folds 60
```

- Frozen checksum: PASS.
- 60/60 newly dispatched folds: PASS.
- Runtime khoảng 90 giây; dataset 14 labels chậm hơn ba block trước nhưng vẫn thấp hơn nhiều quota 5 giờ.

## Audit

```json
{
  "status": "PASS",
  "config_hash": "2a300c396384c5e9",
  "completed_pairs": 48,
  "completed_folds": 240,
  "partial_pairs": 0,
  "missing_pairs": 72,
  "next_dataset": "genbase"
}
```

Manifest giữ `Status=partial` đúng frozen queue và tiếp tục dùng canonical hash Q12.2/Q12.3. Mọi pair checkpoint có schema v3, status complete, recomputed config hash đúng và fold keys `[1,2,3,4,5]`.

## Queue state

| Dataset | Pairs | Folds | Trạng thái |
|---|---:|---:|---|
| emotions | 12/12 | 60/60 | complete |
| music | 12/12 | 60/60 | complete |
| scene | 12/12 | 60/60 | complete |
| yeast | 12/12 | 60/60 | complete |
| 6 datasets còn lại | 0/72 | 0/360 | missing |

Queue: `genbase -> medical -> enron -> cal500 -> bibtex -> reuters-k500`.

## Output/cache

- `results_pa_v3_full/checkpoints/`: 48 files, khoảng 12.55 MB.
- Canonical partial tables/figures tiếp tục ở directory hash `2a300c396384c5e9`.
- Output local/ignored và schema-v2 results không đổi.

## Test và code

- Phase không đổi code; strict inventory audit PASS 48/48 pairs.
- Regression baseline gần nhất: 96/96 tests và 15/15 legacy groups pass ở Q12.2.
- `git diff --check` pass trước commit.

## Files commit

- `docs/progress/phase_Q12.4.md`.
- `specify.md` ghi tiến độ, chưa đánh dấu Q12 tổng thể complete.

## Blocker/việc còn lại

- Còn 72 pairs/360 primary folds; không có blocker hay failed job.
- Full artifact audit và ablation queue vẫn chờ primary completion.
- `note.txt` deleted từ trước vẫn ngoài phase/staging.

## Entry point Phase Q12.5

```powershell
python scripts/run_frozen_experiment.py `
  --config configs/full_run.json --max-new-folds 60
```

Expected: hoàn tất `genbase`, đạt 60/120 pairs và 300/600 folds; dataset tiếp theo `medical`.
