# Progress Phase v6.3.03 — Metric, result store và kế thừa

Ngày cập nhật: 2026-10-11  
Trạng thái: completed

## Công việc đã làm được

- Cài complete, selective, calibration và IL/DL layer audit theo schema đặc tả.
- Cài result store nguyên tử cho JSON/NPZ/CSV, checksum prediction và resume theo config/split hash.
- Cài kế thừa BR, CC, MLC_PA và GSI_v6_2, không huấn luyện lại.
- Metric không có trong artifact v6.2 được xuất `-`, provenance là `inherited_unmatched` vì không có split hash/prediction cũ để chứng minh ghép cặp.
- Tạo đúng cây `hamming_bop` và scaffold rỗng `f_measure_bop`.

## File đã tạo hoặc thay đổi

- `src/evaluation/layer_metrics.py`
- `src/evaluation/result_store_v6.py`
- `src/evaluation/__init__.py`
- `scripts/run_v6_3_ablation.py`
- `tests/test_v6_3.py`

## Lệnh/test đã chạy và kết quả

- `python -m unittest tests.test_v6_3 -v`: 7/7 pass.
- Test checksum/resume và mismatch config hash: pass.
- `python tests_unit.py`: 15/15 nhóm test hiện hữu pass.

## Thông tin cần kiểm chứng

- Selective Subset Accuracy tạm tính trên các mẫu có ít nhất một quyết định: bằng 1 khi toàn bộ vị trí đã quyết định đều đúng; mẫu all-abstain bị loại khỏi mẫu số; không có mẫu hợp lệ trả `NaN`. Công thức nằm ở `src/evaluation/layer_metrics.py` và ảnh hưởng `selective_metric.csv` cùng hình selective subset.

## Công việc bị block

- Không có.

## Artifact/kết quả được tạo

- Năm bảng CSV theo learner và bảng nối tại `all` trong smoke run.
- Checkpoint chứa partition, IL/DL, chain order, sampling metadata, metric và timing.

## Công việc cho phase sau

- Smoke test cả ba learner.
