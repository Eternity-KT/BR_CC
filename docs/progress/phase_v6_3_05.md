# Progress Phase v6.3.05 — Full outer 5-fold

Ngày cập nhật: 2026-10-12  
Trạng thái: completed

## Công việc đã làm được

- Khởi động full run theo đúng `configs/v6_3_run.json`.
- Runner chỉ huấn luyện năm mô hình v6.3 mới; bốn mô hình cũ được nhập từ v6.2.
- Ghi checkpoint/prediction sau từng model-fold để có thể resume.
- Hoàn tất 10 dataset × 3 learner × 5 outer folds × 5 mô hình mới.
- Xác nhận full-run log kết thúc bằng marker `[DONE]` và stderr rỗng.

## File đã tạo hoặc thay đổi

- `result_pa_v6_3/`
- `v6_3_full.stdout.log`
- `v6_3_full.stderr.log`
- `v6_3_full.pid`

## Lệnh/test đã chạy và kết quả

- `venv/Scripts/python.exe scripts/run_v6_3_ablation.py`: hoàn tất thành công.
- Artifact tính mới: 150 checkpoint, 750 model-fold và 150 prediction NPZ.
- Artifact kế thừa: 120 checkpoint và 600 model-fold.

## Thông tin cần kiểm chứng

- Không có.

## Công việc bị block

- Không có.

## Artifact/kết quả được tạo

- Artifact được cập nhật nguyên tử sau từng fold và đã đủ toàn bộ ma trận chạy.

## Công việc cho phase sau

- Tái tạo bảng/hình, viết báo cáo ablation và chạy validator cuối.
