# Phase status — BSS-UG-SPCC-PA

- Updated at: 2026-09-20
- Current phase: Phase 6 — Baseline Logistic và hình so sánh
- Status: complete
- Objective of this phase: Tái sử dụng kết quả Logistic tương thích, tính chính xác Selective Instance-F1 của MLC-PA từ dự đoán từng fold và dựng 15 hình so sánh BR–CC–MLC-PA–BSS theo mẫu ESWA.

## Completed

- Đã giữ primary queue ở 9 dataset; `bibtex` không nằm trong queue.
- Đã xác minh `results_pa/tables/raw_results.json` có đủ 9 dataset `BR_Logistic`.
- Đã xác minh `results_pa/tables/MLC_PA.json` dùng `base_estimator=logistic`, 5 folds, seed 42, linear penalty và đúng cost grid; cache này có 8/9 dataset primary, thiếu `genbase`.
- Đã xác minh dự án không có kết quả `CC_Logistic` tương thích; key `CC` hiện có là LinearSVC và không được dùng/đổi nhãn.
- Đã thêm adapter strict chuẩn hóa đúng 5 Full metric và các selective metric cần cho hình; `Hamming Accuracy = 1 - Hamming Loss`, `Instance-F1 = Example-F1` là ánh xạ chính xác.
- Đã xác định cache MLC-PA schema-v2 không thể suy ra Selective Instance-F1; bổ sung run schema-v3 riêng để tính trực tiếp từ prediction `-1/0/1`, không nội suy và không thay bằng full Instance-F1.
- Runner hiện tái sử dụng BR/MLC-PA đã có, tạo-resume riêng `CC_Logistic` cho 9 dataset và `MLC_PA_Logistic` cho `genbase`, sau đó tiến triển BSS. `--max-new-folds N` áp dụng cho từng queue còn thiếu và queue BSS.
- Đã bổ sung `--skip-comparison-preparation` cho chẩn đoán; audit chính thức vẫn yêu cầu comparison đủ dataset.
- Đã mở rộng 10 hình dataset và 5 hình rejection-cost theo đúng cấu trúc hình mẫu: full prediction cho bốn model, cột “with rejection” và ABS/AABS chỉ cho MLC-PA/BSS.
- Đã thêm `comparison_audit.json` với source path, SHA-256, schema, model key, settings, missing datasets và cảnh báo metric thiếu; file được đưa vào artifact manifest.
- Baseline comparison không được trộn vào `results.json`, `fold_results.csv` hoặc `summary_results.csv` của BSS.
- Đã cập nhật `specify.md` và `README.md` với quy trình mới.
- Đã chạy hoàn tất 45 outer fold `CC_Logistic` và 5 outer fold `MLC_PA_Logistic/genbase`; 45 fold BSS đã có được tái sử dụng, không huấn luyện lại.
- Đã dựng bộ artifact chính thức tại run hash `183eb14632e3dca1`: 10 hình dataset, 5 hình rejection-cost và 5 table/audit artifact.
- Theo yêu cầu bổ sung, đã chạy lại 45 fold `MLC_PA_Logistic` với `include_selective_instance_f1=true`, tính đủ metric ở 5 cost cho cả 9 dataset và dựng artifact thay thế tại run hash `179e728c3cd3f066`.

## Files created/modified

- `src/evaluation/bss_comparison.py` — mới; adapter/provenance/status cho BR, CC, MLC-PA Logistic.
- `tests/test_bss_comparison.py` — mới; fixture nguồn, settings, metric thiếu và render 15 hình.
- `configs/bss_ug_spcc_pa.json` — thêm cấu hình comparison và output checkpoint baseline thiếu.
- `scripts/run_bss_ug_spcc_pa.py` — phát hiện nguồn và tạo-resume phần baseline Logistic còn thiếu.
- `main.py`, `src/evaluation/pipeline_v3.py` — truyền/nạp comparison bundle trong schema-v3 BSS.
- `src/evaluation/export_v3.py` — xuất `comparison_audit.json`.
- `src/visualization/plots.py` — hình bốn model và rejection-cost theo mẫu.
- `scripts/audit_v3_results.py` — audit model identity, base learner, completeness và source artifact.
- `specify.md`, `README.md`, `phase_status.md` — đặc tả và hướng dẫn vận hành.

## Verification/tests and evidence

- `python -m py_compile` cho toàn bộ module/script bị tác động: pass.
- `python scripts/run_bss_ug_spcc_pa.py --dry-run`: pass; báo BR 9/9, CC 0/9, MLC-PA 8/9 và đúng dataset thiếu `genbase`; không nạp dataset/không huấn luyện.
- `python -m unittest tests.test_bss_comparison`: 3/3 pass, gồm fail-closed khi settings lệch và synthetic render đủ 15 PNG.
- Đã mở và kiểm tra trực quan hình dataset/rejection-cost synthetic: legend đủ bốn model, MLC-PA Selective Instance-F1 hiển thị `N/A`, nhãn và error bar không bị cắt.
- `python -m unittest tests.test_bss_comparison tests.test_bss_artifacts`: 4/4 pass.
- Registry/comparison/artifact regression chọn lọc: 10/10 pass.
- Full suite cuối: 109/111 pass; chỉ còn đúng hai lỗi checksum frozen đã tồn tại trước ở `configs/full_run.json` và `configs/ablation_run.json`.
- Audit chính thức run `183eb14632e3dca1`: PASS với 9 dataset, 5 folds, 5 costs, 15 hình và comparison không còn dataset thiếu.
- Dry-run sau thực nghiệm xác nhận BR/CC/MLC-PA Logistic đều đủ 9/9 dataset, `missing_datasets=[]`.
- Dry-run sau lần tính bổ sung xác nhận `selective_instance_f1_available_datasets` đủ 9/9 và `missing_selective_instance_f1_datasets=[]`.
- Đã kiểm tra trực quan hình thật mới `dataset_selective_instance_f1_comparison.png` và `rejection_cost_selective_instance_f1.png`: đủ bốn model, nhãn/error bar rõ và MLC-PA có giá trị ở mọi dataset/cost, không còn `N/A`.
- Audit chính thức run `179e728c3cd3f066`: PASS; `Missing Selective Instance-F1 -> MLC_PA_Logistic=[]`.

## Leakage/reproducibility audit

- Baseline chỉ được nhận khi model key và settings Logistic tương thích; `CC` LinearSVC bị loại.
- Mọi baseline thiếu được chạy bằng cùng 5 outer folds, seed 42, `MaxAbsScaler`, cost grid/report cost/penalty với primary run.
- Comparison chỉ đọc kết quả outer-fold đã hoàn tất và không tham gia chọn alpha, DAG, parent, calibration hoặc cost của BSS.
- Không dùng test để chọn mô hình hoặc tối ưu tham số.

## Remaining issues/blockers

- Repository vẫn có hai lỗi checksum frozen đã tồn tại trước ở `configs/full_run.json` và `configs/ablation_run.json`; hai file này không bị sửa.

## Next phase and exact first task

- Không còn công việc bắt buộc cho phase thực nghiệm hiện tại.
- Phase sau tùy chọn: phân tích kết quả/bảng để viết báo cáo; không thay đổi scientific settings của run `179e728c3cd3f066`.

## Experiment status

- complete_and_audited
- Run hash chính thức: `179e728c3cd3f066`.
