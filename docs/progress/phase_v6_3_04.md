# Progress Phase v6.3.04 — Smoke test ba learner

Ngày cập nhật: 2026-10-11  
Trạng thái: completed

## Công việc đã làm được

- Chạy dataset `emotions` với Logistic, calibrated SVM và PyTorch MLP.
- Mỗi learner hoàn tất ít nhất một fold cho cả năm mô hình mới.
- Kiểm tra artifact chứa đúng shared DL order giữa CC/PCC và `num_samples_per_test_x=1000` cho PCC.
- Kiểm tra resume thực tế sau khi một tiến trình bị ngắt sau lúc lưu fold.

## File đã tạo hoặc thay đổi

- `result_pa_v6_3/` (artifact smoke; full runner sẽ upsert/resume theo hash)
- `v6_3_smoke.stdout.log`
- `v6_3_smoke_mlp.stdout.log`

## Lệnh/test đã chạy và kết quả

- Logistic: đủ 5 variant trên fold 1, thành công.
- calibrated SVM: đủ 5 variant trên folds 1–2, thành công.
- MLP: đủ 5 variant trên fold 1, thành công.
- Figure generator tạo đủ 16 PNG cho từng scope hiện có.

## Thông tin cần kiểm chứng

- Không có thông tin bắt buộc mới trước full run.

## Công việc bị block

- Không có.

## Artifact/kết quả được tạo

- Checkpoint JSON, predictions NPZ, năm CSV và hình smoke dưới `result_pa_v6_3`.

## Công việc cho phase sau

- Chạy full 10 dataset × 3 learner × 5 outer folds cho năm mô hình mới.
