# Progress Phase v6.3.02 — CC/PCC và feature-space ablation

Ngày cập nhật: 2026-10-11  
Trạng thái: completed

## Công việc đã làm được

- Cài PCC với factorization theo chain và Monte-Carlo ancestral inference.
- PCC dùng đúng 1.000 mẫu tổ tiên trên mỗi test x; sampling theo batch chỉ là tối ưu bộ nhớ, không đổi số mẫu.
- Cài đủ `CC_X`, `PCC_X`, `CC_X_IL`, `PCC_X_IL`.
- CC/PCC nhận cùng danh sách DL đã được selector sắp theo tương quan tăng dần và dùng natural local order trên danh sách đó.
- Xác suất PCC được trả về thứ tự cột nhãn gốc.

## File đã tạo hoặc thay đổi

- `src/models/probabilistic_classifier_chain.py`
- `src/models/gsi_v6_3.py`
- `tests/test_v6_3.py`

## Lệnh/test đã chạy và kết quả

- Test PCC reproducibility, shape/range và lưới xác suất 1/1000: pass.
- Test CC/PCC cùng partition và chain order: pass.

## Thông tin cần kiểm chứng

- Không có.

## Công việc bị block

- Không có.

## Artifact/kết quả được tạo

- Năm cấu hình mô hình mới, gồm đối chứng ResidualBR và bốn ablation yêu cầu.

## Công việc cho phase sau

- Cài metric, layer audit, result store và kế thừa v6.2.
