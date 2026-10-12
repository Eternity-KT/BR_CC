# Progress Phase v6.3.01 — Single-stage peeling và ResidualBR

Ngày cập nhật: 2026-10-11  
Trạng thái: completed

## Công việc đã làm được

- Cài wrapper selector chỉ chạy đúng một stage.
- Tắt singleton promotion; singleton còn lại trong DL.
- Cài đối chứng `GSI_v6_3_1L_ResidualBR` trên phân hoạch một tầng.
- Dùng OOF IL probabilities cho đặc trưng huấn luyện `x_il` và predicted IL probabilities khi suy luận.

## File đã tạo hoặc thay đổi

- `src/selection/single_stage_peeling.py`
- `src/selection/__init__.py`
- `src/models/gsi_v6_3.py`
- `src/models/__init__.py`

## Lệnh/test đã chạy và kết quả

- `python -m unittest tests.test_v6_3 -v`: pass.
- Test singleton DL và số stage: pass.

## Thông tin cần kiểm chứng

- Không có giả định mới ngoài cấu hình frozen.

## Công việc bị block

- Không có.

## Artifact/kết quả được tạo

- Selector một tầng và mô hình ResidualBR dùng chung partition v6.3.

## Công việc cho phase sau

- Cài CC/PCC và hai feature space cho DL.
