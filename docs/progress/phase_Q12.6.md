# Handoff Phase Q12.6 — Primary full run: medical, enron, cal500 và Bibtex (partial)

**Trạng thái:** Dừng chủ động ngày 2026-09-02; có thể resume an toàn.

## Objective đã hoàn thành

- Resume canonical frozen run từ 300/600 folds.
- Hoàn tất toàn bộ `medical`, `enron` và `cal500`: mỗi dataset có 12 matched IDs × 5 folds = 60 folds.
- Hoàn tất 9/12 model pairs của `bibtex`, gồm tất cả Logistic/MLP và `BR_SVM`.
- Dừng sau checkpoint hoàn chỉnh; không có pair ở trạng thái `in_progress` hay checkpoint bị ghi dở.

## Lệnh đã chạy

```powershell
python scripts/run_frozen_experiment.py `
  --config configs/full_run.json --max-new-folds 5
```

- Frozen checksum: `4036af255fcdc462f9d5307b924cc0d1a1004b02309f749ac66826a9bc61fd1e` — PASS.
- Bibtex GSI variants là phần tốn thời gian nhất; runner vẫn checkpoint độc lập sau từng fold.
- Việc dừng không làm mất kết quả vì chỉ các pair có đủ 5 folds mới được đánh dấu `complete`.

## Checkpoint audit

Inventory checkpoint schema-v3 sau khi dừng:

| Dataset | Complete pairs | Complete folds | Trạng thái |
|---|---:|---:|---|
| emotions | 12/12 | 60/60 | complete |
| music | 12/12 | 60/60 | complete |
| scene | 12/12 | 60/60 | complete |
| yeast | 12/12 | 60/60 | complete |
| genbase | 12/12 | 60/60 | complete |
| medical | 12/12 | 60/60 | complete |
| enron | 12/12 | 60/60 | complete |
| cal500 | 12/12 | 60/60 | complete |
| bibtex | 9/12 | 45/60 | partial queue, no partial pair |
| reuters-k500 | 0/12 | 0/60 | missing |
| **Total** | **105/120** | **525/600** | **resume-ready** |

- 105 checkpoint files đều có `status=complete` và đủ fold set `[1,2,3,4,5]`.
- Không có process Python còn chạy sau lệnh dừng.
- Full artifact audit chưa chạy vì queue primary còn thiếu 15 pairs/75 folds.

## Queue còn lại

1. `bibtex`: `CC_SVM`, `MLC_PA_SVM`, `GSI_MLC_PA_SVM` (15 folds).
2. `reuters-k500`: toàn bộ 12 models (60 folds).

## Entry point lần chạy sau

Chạy từng block 5 folds để tiếp tục theo queue frozen:

```powershell
python scripts/run_frozen_experiment.py `
  --config configs/full_run.json --max-new-folds 5
```

Lần gọi kế tiếp tự chọn `bibtex/CC_SVM`; không cần truyền dataset/model hoặc dọn cache. Lặp lại lệnh đến khi hết queue, rồi chạy full audit:

```powershell
python scripts/audit_v3_results.py results_pa_v3_full `
  --config-hash 2a300c396384c5e9 `
  --expect-primary-models `
  --folds 5 `
  --costs 0.2 0.25 0.3 0.35 0.4 0.5
```

## Commit scope

- Không đổi production/test code trong Q12.6.
- Generated results vẫn local/ignored; commit chỉ lưu handoff này và cập nhật tiến độ trong `specify.md`.
- `note.txt` đã bị xóa từ trước và tiếp tục nằm ngoài stage.
