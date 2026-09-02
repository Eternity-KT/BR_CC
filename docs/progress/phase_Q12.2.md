# Handoff Phase Q12.2 — Primary full run: music và run-hash stability fix

**Trạng thái:** Hoàn thành ngày 2026-09-02.

## Objective đã hoàn thành

- Resume cùng checksum-frozen config, tự bỏ qua 60 folds `emotions` đã complete.
- Dispatch và hoàn tất `music`, 12 matched IDs × 5 folds = 60 folds mới.
- Lũy kế 24 complete pairs/120 folds; không có partial hoặc failed checkpoint.
- Phát hiện run-manifest hash thay đổi theo progress dù frozen scientific settings không đổi.
- Tạm dừng dispatch dataset tiếp theo, xác định root cause, sửa và regression-test trước khi tiếp tục queue.
- Chạy lại toàn bộ unit/integration và legacy regression sau production fix.

## Lệnh experiment đã chạy

```powershell
python scripts/run_frozen_experiment.py `
  --config configs/full_run.json --max-new-folds 60
```

Kết quả: 60 dòng `music/<MODEL>` fold checkpoint, exit code `0`; runtime khoảng 23 giây.

## Checkpoint inventory

```json
{
  "status": "PASS",
  "completed_pairs": 24,
  "completed_folds": 120,
  "partial_pairs": 0,
  "missing_pairs": 96,
  "next_dataset": "scene"
}
```

Mỗi checkpoint `emotions`/`music` được parse strict và xác nhận schema v3, status complete, canonical pair config hash và fold keys `[1,2,3,4,5]`.

## High issue phát hiện và cách sửa

Hai partial manifest trước fix có settings chỉ khác một trường:

```text
Q12.1 hash 7b6798f8aa9dc934: label_policy_hashes = {emotions}
Q12.2 hash c111f264dfbb554e: label_policy_hashes = {emotions, music}
```

Root cause: `dataset_policy_hashes` được bổ sung bên trong dataset loop. Khi `max_new_folds` dừng sớm, run config chỉ chứa policy hash của dataset đã đi qua; progress state vô tình trở thành scientific hash input.

Fix trong `src/evaluation/pipeline_v3.py`:

1. Chuẩn hóa/hash policy cho toàn bộ requested dataset queue trước khi dispatch fold, không load dataset/test targets.
2. Khi dataset thực sự load, validate policy với label names và assert content hash không đổi.
3. Không thay pair checkpoint config/hash hoặc model prediction; 120 folds đã tạo vẫn hợp lệ và được resume.

Canonical stable run hash của frozen settings sau fix:

```text
2a300c396384c5e9
```

Hai historical partial manifest directories được giữ local/ignored làm audit trail; từ Q12.3 phải dùng canonical directory `2a300c396384c5e9`. Final audit truyền `--config-hash 2a300c396384c5e9` để không chọn nhầm historical manifest.

## Regression mới

`test_run_hash_is_stable_across_multi_dataset_quota_resume` tạo run hai dataset:

```text
call 1: stop sau dataset first  -> Status=partial
call 2: resume dataset second   -> Status=complete
```

Test yêu cầu:

- hai `Config Hash` bằng nhau;
- run settings chứa policy-hash keys cho cả `first` và `second` ngay từ call đầu;
- output chỉ có đúng một `tables/<config_hash>/results_v3.json`.

Trước fix test này tạo hai manifest directories; sau fix pass.

## Files sửa/tạo

| File | Thay đổi Q12.2 |
|---|---|
| `src/evaluation/pipeline_v3.py` | Precompute full-queue label-policy hashes để run hash không phụ thuộc quota progress |
| `tests/test_system_verification.py` | Multi-dataset partial/resume run-hash regression |
| `docs/progress/phase_Q12.2.md` | Experiment inventory, incident/root cause/fix và queue handoff |
| `specify.md` | Ghi tiến độ Q12.2; Q12 tổng thể vẫn đang chạy |

## Test/lệnh đã chạy + kết quả

```text
python -m unittest tests.test_system_verification tests.test_pipeline_v3 -v
```

- PASS 13/13 targeted resume/hash/pipeline tests.

```text
python -m unittest discover -s tests -v
```

- PASS 96/96 tests Q1–Q12.2.

```text
python tests_unit.py
```

- PASS toàn bộ 15 nhóm legacy regression/unit/integration.

```text
python -m compileall -q main.py src scripts tests
git diff --check
```

- PASS syntax và whitespace; chỉ có cảnh báo LF/CRLF trên Windows.

## Queue state

| Dataset | Complete pairs | Complete folds | Trạng thái |
|---|---:|---:|---|
| emotions | 12/12 | 60/60 | complete |
| music | 12/12 | 60/60 | complete |
| 8 datasets còn lại | 0/96 | 0/480 | missing |

Frozen order tiếp theo:

```text
scene -> yeast -> genbase -> medical -> enron
      -> cal500 -> bibtex -> reuters-k500
```

## Output/cache

- `results_pa_v3_full/checkpoints/`: 24 complete files, tổng khoảng 5.36 MB.
- Historical partial tables/figures giữ local và ignored; không stage/delete.
- Pair checkpoints không phụ thuộc lỗi run-summary hash và đã được audit lại sau khi phát hiện issue.
- Không thay `results/` hoặc `results_pa/` schema-v2.

## Việc chưa hoàn thành hoặc blocker

- 96 primary pairs/480 folds còn thiếu; expected Q12 queue.
- Canonical manifest `2a300c396384c5e9` sẽ được pipeline sinh ở Q12.3 khi resume bằng code đã fix.
- Full run audit chưa thể PASS cho đến dataset cuối.
- Không còn blocker kỹ thuật; high issue đã có regression và full suite pass.

## Git status và thay đổi ngoài phạm vi

- Q12.2 bắt đầu từ commit Q12.1 `8404095` trên nhánh `Khanh_1`.
- `note.txt` vẫn deleted từ trước, không thuộc phase và không được stage.
- Generated benchmark data bị ignore và được giữ để resume.

## Entry point/lệnh đầu tiên cho Phase Q12.3

```powershell
python scripts/run_frozen_experiment.py `
  --config configs/full_run.json --max-new-folds 60
```

Expected: resume 120 folds cũ, hoàn tất `scene`, sinh/update canonical partial manifest `tables/2a300c396384c5e9/results_v3.json`, đạt 36/120 pairs và 180/600 folds; dataset kế tiếp `yeast`.
