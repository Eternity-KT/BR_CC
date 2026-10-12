# Progress Phase v6.3.00 — Audit và đóng băng cấu hình

Ngày cập nhật: 2026-10-11  
Trạng thái: completed

## Công việc đã làm được

- Đối chiếu `comment.md`, `ablation.md`, các đặc tả v6 trước và artifact v6.2.
- Khóa danh sách 10 dataset theo `configs/v6_3_run.json` và đúng danh sách v6.2.
- Khóa CC/PCC dùng chung thứ tự DL theo tổng tương quan tăng dần.
- Khóa PCC dùng 1.000 ancestral samples cho mỗi mẫu kiểm tra.
- Xác định nguồn kế thừa thực tế là `results_v6_2/v6_2_detailed_folds.csv`.
- Bổ sung ARFF fallback để cả 10 dataset chính thức đều nạp được.

## File đã tạo hoặc thay đổi

- `spec/spec_v6_3.md`
- `configs/v6_3_run.json`
- `src/data/loader.py`

## Lệnh/test đã chạy và kết quả

- Nạp thử toàn bộ 10 dataset: thành công, đúng số mẫu/nhãn dự kiến.
- `python tests_unit.py`: 15/15 nhóm test pass.

## Thông tin cần kiểm chứng

- Quy ước tạm cho Selective Subset Accuracy vẫn cần xác nhận riêng; full run dùng quy ước đã ghi rõ trong đặc tả và phase 03.

## Công việc bị block

- Không có.

## Artifact/kết quả được tạo

- Đặc tả v6.3 và cấu hình frozen.

## Công việc cho phase sau

- Cài selector single-stage và đối chứng ResidualBR.
