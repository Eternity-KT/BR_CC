# Progress Phase v6.3.06 — Tổng hợp và xác thực cuối

Ngày cập nhật: 2026-10-12  
Trạng thái: completed

## Công việc đã làm được

- Tái tạo toàn bộ hình từ kết quả full run, thay cho bộ hình smoke chỉ có `emotions`.
- Tạo báo cáo ablation Markdown từ năm bảng kết quả.
- Sửa report generator để xuất bảng Markdown không phụ thuộc gói tùy chọn `tabulate`.
- Chạy validator toàn bộ checksum, schema, số lượng artifact, phạm vi xác suất, shared DL order và cấu hình PCC.
- Kiểm tra trực quan `complete_macro_f1.png`: đủ 10 dataset theo MeanIR trong cả ba panel learner.

## File đã tạo hoặc thay đổi

- `scripts/generate_v6_3_report.py`
- `result_pa_v6_3/v6_3_ablation_report.md`
- `result_pa_v6_3/validation_report.json`
- 20 CSV dưới các thư mục `tables/`
- 64 PNG dưới các thư mục `figures/`

## Lệnh/test đã chạy và kết quả

- `venv/Scripts/python.exe scripts/generate_v6_3_figures.py`: thành công, 64/64 PNG.
- `venv/Scripts/python.exe scripts/generate_v6_3_report.py`: thành công.
- `venv/Scripts/python.exe scripts/validate_v6_3_results.py`: `status=pass`, không có lỗi.
- Validator: 150 checkpoint tính mới, 120 checkpoint kế thừa, 750 model-fold tính mới, 600 model-fold kế thừa, 150 prediction NPZ, 20 bảng và 64 hình.
- `venv/Scripts/python.exe -m unittest tests.test_v6_3 -v`: 7/7 pass.
- `venv/Scripts/python.exe tests_unit.py`: 15/15 nhóm test pass.

## Thông tin cần kiểm chứng

- `V63-C04`: Selective Subset Accuracy hiện loại mẫu all-abstain khỏi mẫu số, theo quy ước đã ghi trong đặc tả.

## Công việc bị block

- Không có.

## Artifact/kết quả được tạo

- Bộ kết quả full v6.3 hoàn chỉnh tại `result_pa_v6_3/`.
- Báo cáo và validation report đã sẵn sàng để kiểm tra hoặc dùng cho bước viết bài.

## Công việc cho phase sau

- Không có bước bắt buộc; chỉ cần đổi quy ước `V63-C04` và tái tổng hợp nếu người dùng chọn định nghĩa khác.
