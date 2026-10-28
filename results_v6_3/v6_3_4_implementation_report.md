# Báo cáo xử lý SAMG cho v6.3.3

Ngày chạy: 2026-10-13

## Kết luận

Không nên thay v6.3.3 bằng v6.3.4/SAMG2 ở thời điểm này. SAMG2 loại bỏ các lần hạ ngưỡng không vượt qua kiểm tra OOF và không làm giảm Macro-F1 so với cấu hình không SAMG trên 10 tập, nhưng Macro-F1 trung bình vẫn thấp hơn SAMG cũ.

| Cấu hình | Macro-F1 trung bình | Coverage | Hamming loss |
|---|---:|---:|---:|
| v6.3.3 không SAMG | 0.569270 | 0.731246 | 0.155678 |
| v6.3.3 SAMG cũ | **0.575900** | 0.731756 | **0.155626** |
| v6.3.4 SAMG2 | 0.571757 | 0.731277 | 0.155746 |

SAMG2 tăng so với không SAMG trên `gpositivepseaac` (+0.011111) và `genbase` (+0.013757), không đổi trên 8 tập còn lại. So với SAMG cũ, SAMG2 bỏ mất một phần lợi ích trên `gpositivepseaac` và toàn bộ lợi ích trên `viruspseaac`.

## Vì sao F1 dương của chd49 bằng 0

Nhãn 3 của `chd49` chỉ có 16/555 mẫu dương (2.88%); mỗi fold huấn luyện chỉ có 12–13 mẫu dương và test có 3–4 mẫu. Phân phối xác suất dương và âm chồng lấn ở cả 5 fold nên điều kiện tạo khoảng cách an toàn của SAMG không đạt:

| Fold | min xác suất mẫu dương | percentile 99.5% mẫu âm | SAMG |
|---:|---:|---:|---|
| 1 | 0.2253 | 0.2686 | Không kích hoạt |
| 2 | 0.1571 | 0.1738 | Không kích hoạt |
| 3 | 0.1638 | 0.1988 | Không kích hoạt |
| 4 | 0.1618 | 0.2019 | Không kích hoạt |
| 5 | 0.0084 | 0.2709 | Không kích hoạt |

Kết quả pooled: F1 dương = 0, F1 âm = 0.985, positive coverage = 68.75%, end-to-end positive recall = 0. Đây là vấn đề thiếu tín hiệu xếp hạng chứ không chỉ là ngưỡng: test AUC trung bình khoảng 0.536 và AP khoảng 0.096. Thử `class_weight=balanced` và trọng số dương nhẹ hơn chỉ tạo thêm false positive, không cải thiện xếp hạng ổn định, nên không đưa vào mã chính.

F1 âm cao chủ yếu đến từ mất cân bằng lớp. Với tỷ lệ dương 2.88%, mô hình luôn đoán âm đã có F1 âm xấp xỉ 0.985; vì vậy không thể dùng F1 âm cao làm bằng chứng rằng nhãn này được học tốt.

## Thay đổi đã thực hiện

- Giữ nguyên hành vi mặc định của v6.3.3 để không phá kết quả cũ.
- Thêm v6.3.4 thử nghiệm với SAMG `oof_guarded_margin`: khoảng cách được phát hiện trên xác suất fit chỉ được dùng nếu không làm giảm F1 dương và không làm giảm F1 âm quá 0.01 trên OOF.
- Bổ sung audit cho từng nhãn: ngưỡng trước/sau, nguồn xác suất, lý do SAMG bật/tắt, F1 dương/âm trước và sau ứng viên.
- Sửa coverage để được tính lại sau ngưỡng SAMG cuối cùng.
- Bổ sung metric pooled: support, positive/negative coverage, confusion counts, selective recall và end-to-end recall.
- Thêm runner so sánh không SAMG, SAMG cũ và SAMG2.

## Quyết định thực nghiệm

Giữ v6.3.3 SAMG hiện tại làm phiên bản chính. Giữ v6.3.4 ở trạng thái thử nghiệm/audit, chưa promote. Không tiếp tục mở rộng sang SVM/MLP vì Logistic gate chưa thắng v6.3.3; làm thêm ở thời điểm này sẽ tăng độ phức tạp nhưng chưa giải quyết được tín hiệu rất yếu của `chd49`.

## Kiểm thử

Đã chạy 32 test và 15 subtest liên quan tới v6.3.3, v6.3.4 và metric contract; tất cả đều pass. Python 3.13 trên Windows có in stack trace access violation khi import `pyarrow`, nhưng pytest kết thúc với exit code 0.
