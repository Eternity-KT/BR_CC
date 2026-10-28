# Thử nghiệm chỉ từ chối trên DL

Ngày chạy: 2026-10-14

## Cơ chế

Biến thể v6.3.5 kế thừa v6.3.4/SAMG2. Dự đoán IL luôn là quyết định đầy đủ 0/1; dự đoán DL tiếp tục dùng hai ngưỡng và có thể trả về `-1`. Xác suất, quá trình peeling, SAMG và ngưỡng không thay đổi.

## Kết quả 10 dataset

| Cấu hình | Selective Macro-F1 | Selective Micro-F1 | Coverage |
|---|---:|---:|---:|
| v6.3.3, từ chối mọi nhãn | **0.575900** | **0.683835** | 0.731756 |
| v6.3.4, từ chối mọi nhãn | 0.571757 | 0.683274 | 0.731277 |
| v6.3.5, chỉ từ chối DL | 0.556034 | 0.671737 | **0.821233** |

So với v6.3.4, IL coverage tăng từ 0.767707 lên 1.0; DL coverage và DL selective Macro-F1 giữ nguyên hoàn toàn. IL selective Macro-F1 trung bình giảm từ 0.830181 xuống 0.803170 do các trường hợp khó trước đây bị từ chối nay bắt buộc phải quyết định.

| Dataset | Delta Macro-F1 | Delta coverage |
|---|---:|---:|
| emotions | -0.044387 | +0.145336 |
| scene | -0.064617 | +0.227062 |
| yeast | -0.012793 | +0.119791 |
| plantpseaac | 0 | 0 |
| humanpseaac | 0 | 0 |
| chd49 | **+0.037813** | +0.155197 |
| music | -0.035890 | +0.127511 |
| gpositivepseaac | -0.034883 | +0.115105 |
| genbase | -0.002469 | +0.008728 |
| viruspseaac | 0 | +0.000833 |

`plantpseaac` và `humanpseaac` không đổi vì không có nhãn IL trong các fold chạy thử. Full Hamming loss và Subset Accuracy của v6.3.4/v6.3.5 giống nhau vì biến thể này chỉ thay đổi việc xuất `-1`, không thay đổi quyết định full.

## Kiểm tra chd49

Mức tăng tổng thể của `chd49` đến từ hai nhãn IL. Nhãn hiếm dương 3 vẫn thuộc DL nên không bị tác động: support dương 16, coverage 0.731532, F1 dương 0, F1 âm 0.985 và end-to-end positive recall 0.

## Quyết định

Không promote v6.3.5 nếu mục tiêu chính là tối đa selective F1. Cơ chế này chỉ phù hợp khi yêu cầu nghiệp vụ ưu tiên coverage và bắt buộc IL phải luôn có kết quả. Nên giữ nó như một chế độ tùy chọn `rejection_scope="dl_only"`, còn mặc định vẫn là `all_labels`.
