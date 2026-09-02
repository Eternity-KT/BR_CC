# Handoff Phase Q12.1 — Primary full run: emotions

**Trạng thái:** Hoàn thành ngày 2026-09-02.

## Objective đã hoàn thành

- Xác minh checksum frozen config trước khi chạy.
- Dùng đúng primary 10-dataset/12-model settings đã khóa ở Q11.
- Dispatch một dataset block trọn vẹn: `emotions`, 12 matched IDs, 5 outer folds/model.
- Hoàn tất 60/60 folds đã dispatch; không có pair/fold dở hoặc failed.
- Audit strict JSON, run hash, pair config hashes, schema/status và fold indices của cả 12 checkpoints.
- Ghi queue còn lại để Q12.2 resume mà không thay scientific config.

## Lệnh đã chạy

```powershell
python scripts/run_frozen_experiment.py `
  --config configs/full_run.json --max-new-folds 60
```

Runner xác minh trước khi dispatch:

```text
Frozen config SHA-256
4036af255fcdc462f9d5307b924cc0d1a1004b02309f749ac66826a9bc61fd1e
```

`60` được chọn vì đúng một dataset block (`12 models × 5 folds`), không để pair dở ở ranh giới phase. Smoke Q11 hoàn thành 24 folds trong khoảng 14 giây; Q12.1 thực tế hoàn thành 60 folds trong khoảng 22 giây, thấp hơn nhiều so với quota 5 giờ.

## Kết quả checkpoint audit

```json
{
  "status": "PASS",
  "run_status": "partial",
  "config_hash": "7b6798f8aa9dc934",
  "completed_pairs": 12,
  "completed_folds": 60,
  "partial_pairs": 0,
  "missing_pairs": 108
}
```

`run_status=partial` là đúng vì manifest mô tả toàn bộ frozen queue 120 pairs nhưng mới dispatch `emotions`. Mỗi checkpoint đã được parse strict và xác nhận:

- `schema_version=3`, `status=complete`;
- canonical `compute_config_hash(config)` bằng `config_hash` đã lưu;
- dataset/model đúng pair path;
- fold keys chính xác `[1,2,3,4,5]`;
- không có duplicate/partial checkpoint.

## Queue state

| Trạng thái | Pair | Folds |
|---|---:|---:|
| Complete | 12 | 60 |
| Partial | 0 | 0 |
| Missing | 108 | 540 |

Dataset hoàn thành:

```text
emotions: 12/12 models, 60/60 folds
```

Dataset chưa dispatch theo frozen order:

```text
music -> scene -> yeast -> genbase -> medical -> enron
      -> cal500 -> bibtex -> reuters-k500
```

Bốn pair kế tiếp: `music/BR_Logistic`, `music/BR_MLP`, `music/CC_Logistic`, `music/CC_MLP`; Q12.2 dùng block 60 folds để hoàn tất toàn bộ `music`, không chỉ bốn pair này.

## Output/cache được tạo

- `results_pa_v3_full/checkpoints/`: 12 checkpoint files, tổng khoảng 2.68 MB.
- `results_pa_v3_full/tables/7b6798f8aa9dc934/`: partial manifest và CSV hiện có.
- `results_pa_v3_full/figures/7b6798f8aa9dc934/`: figures từ phần kết quả đã hoàn thành.
- Generated v3 output bị `.gitignore`, được giữ trong workspace để Q12.2 resume và không đưa vào phase commit.
- Không sửa/ghi đè `results/` hoặc `results_pa/` schema-v2.

## Test/lệnh kiểm tra

```text
python scripts/run_frozen_experiment.py --config configs/full_run.json --max-new-folds 60
```

- PASS checksum verification; 60 dòng fold checkpoint; process exit code `0`.

```text
inline strict checkpoint inventory audit
```

- PASS run config hash, 12 pair hashes, 12 complete statuses và 60 fold keys.

```text
git diff --check
```

- Chạy trước commit Q12.1; chỉ có cảnh báo line ending nếu Git phát hiện LF/CRLF.

Full unit/regression suite không chạy lại vì Q12.1 không đổi production/test code; baseline Q11 ngay trước phase là 95/95 tests và 15/15 legacy groups pass.

## Files thuộc phase commit

- `docs/progress/phase_Q12.1.md` — checkpoint/job queue handoff.
- `specify.md` — ghi tiến độ Q12.1; không đánh dấu Q12 tổng thể hoàn thành.

## Việc chưa hoàn thành hoặc blocker

- 108 primary pairs/540 folds còn thiếu; đây là expected queue, không phải blocker.
- Full-run audit yêu cầu `Status=complete` chỉ chạy ở Q12 cuối; mỗi phase trung gian audit checkpoint inventory.
- Objective/partition/random-matched ablations chưa dispatch cho tới khi primary queue hoàn thành.
- Không có failed pair và không có blocker kỹ thuật cho Q12.2.

## Git status và thay đổi ngoài phạm vi

- Q12.1 bắt đầu từ commit Q11 `d23696b` trên nhánh `Khanh_1`.
- `note.txt` vẫn ở trạng thái deleted có từ trước và không được stage.
- Benchmark output được giữ local/ignored; phase commit chỉ chứa handoff + tiến độ.

## Entry point/lệnh đầu tiên cho Phase Q12.2

Resume cùng checksum-frozen config và hoàn tất block `music`:

```powershell
python scripts/run_frozen_experiment.py `
  --config configs/full_run.json --max-new-folds 60
```

Expected sau Q12.2: `completed_pairs=24`, `completed_folds=120`, `partial_pairs=0`, `missing_pairs=96`; dataset tiếp theo là `scene`.
