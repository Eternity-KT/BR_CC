# Handoff Phase Q12.3 — Primary full run: scene

**Trạng thái:** Hoàn thành ngày 2026-09-02.

## Objective đã hoàn thành

- Resume frozen queue, bỏ qua 120 folds `emotions`/`music` đã complete.
- Hoàn tất `scene`, 12 matched IDs × 5 folds = 60 folds mới.
- Xác minh production fix Q12.2: manifest mới dùng đúng canonical run hash ổn định `2a300c396384c5e9`.
- Audit 36 pair checkpoints/180 folds; không có partial, duplicate hoặc failed pair.

## Lệnh và runtime

```powershell
python scripts/run_frozen_experiment.py `
  --config configs/full_run.json --max-new-folds 60
```

- Frozen SHA-256 verification: PASS.
- 60 fold checkpoints mới: PASS.
- Runtime: khoảng 54 giây; nằm an toàn trong quota 5 giờ.

## Audit kết quả

```json
{
  "status": "PASS",
  "run_status": "partial",
  "config_hash": "2a300c396384c5e9",
  "completed_pairs": 36,
  "completed_folds": 180,
  "partial_pairs": 0,
  "missing_pairs": 84,
  "next_dataset": "yeast"
}
```

Audit parse strict canonical manifest, recompute run config hash và kiểm toàn bộ checkpoint `emotions`/`music`/`scene`:

- `schema_version=3`, `status=complete`;
- pair config hash khớp canonical serialization;
- fold keys đúng `[1,2,3,4,5]`;
- manifest chỉ tổng hợp ba dataset đã hoàn thành và giữ `Status=partial` đúng queue state.

Hai historical progress-dependent manifests từ trước fix Q12.2 không được pipeline cập nhật. Canonical directory đã xuất hiện đúng như regression dự kiến; các phase sau dùng `2a300c396384c5e9`.

## Queue state

| Dataset | Pairs | Folds | Trạng thái |
|---|---:|---:|---|
| emotions | 12/12 | 60/60 | complete |
| music | 12/12 | 60/60 | complete |
| scene | 12/12 | 60/60 | complete |
| 7 datasets còn lại | 0/84 | 0/420 | missing |

Queue tiếp theo:

```text
yeast -> genbase -> medical -> enron -> cal500 -> bibtex -> reuters-k500
```

## Output/cache

- `results_pa_v3_full/checkpoints/`: 36 complete files, khoảng 8.07 MB.
- Canonical partial tables/figures: `results_pa_v3_full/{tables,figures}/2a300c396384c5e9/`.
- Generated output được giữ local/ignored để resume; schema-v2 output không đổi.

## Test và thay đổi code

- Q12.3 không đổi production/test code.
- Strict inventory audit: PASS 36/36 pairs và 180/180 folds.
- Baseline ngay trước phase: 96/96 tests + 15/15 legacy groups pass ở Q12.2.
- `git diff --check` chạy trước phase commit.

## Files thuộc phase commit

- `docs/progress/phase_Q12.3.md`.
- `specify.md` chỉ ghi tiến độ; Q12 tổng thể chưa complete.

## Việc chưa hoàn thành hoặc blocker

- 84 pairs/420 primary folds còn thiếu.
- Full artifact audit chỉ chạy khi dataset cuối hoàn thành.
- Không có blocker; canonical hash drift đã được xác nhận hết lỗi trên run thật.

## Git status

- Phase bắt đầu từ commit Q12.2 `c8a82f9` trên `Khanh_1`.
- `note.txt` vẫn deleted từ trước và không được stage.

## Entry point Phase Q12.4

```powershell
python scripts/run_frozen_experiment.py `
  --config configs/full_run.json --max-new-folds 60
```

Expected: hoàn tất `yeast`, đạt 48/120 pairs và 240/600 folds; dataset kế tiếp `genbase`.
