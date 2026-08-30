# Tóm tắt cuộc họp với giảng viên hướng dẫn

> Nguồn ghi chú thô: [`tmp.md`](tmp.md)
> Đặc tả triển khai: [`specify.md`](specify.md)

## 1. Mục tiêu điều chỉnh

Hoàn thiện cách đánh giá mô hình multi-label classification có partial abstention để kết quả không chỉ tốt trên Macro-F1 mà còn phản ánh được khả năng quyết định ngay, mức độ từ chối, khối lượng cần con người xử lý và lợi ích thực tế của việc chia nhãn IL/DL.

Nguyên tắc triển khai đã thống nhất trong đặc tả là giữ nguyên core BR, CC và cơ chế sinh xác suất của GSI-MLC-PA. Decision rule, objective, metric, calibration và phân tích mới được thêm dưới dạng module có thể bật/tắt.

## 2. Các góp ý chính

### 2.1. Lý thuyết và decision rule

- Nghiên cứu vai trò của Markov Decision Process trong bài toán biên hóa xác suất.
- Phân biệt MDP với Bayes-optimal prediction và quy hoạch động dùng để tính expected F1/Jaccard.
- Ở bước chia nhãn IL/DL, nghiên cứu BOP cho instance-based F1 thay vì chỉ threshold `0.5` và complete Macro-F1.
- So sánh nhiều objective: precision-oriented, recall-oriented, F1 và Jaccard.

### 2.2. Metrics

- Ghi rõ ý nghĩa, công thức, lý do sử dụng và hạn chế của từng metric.
- Trong phần trình bày dùng Hamming Accuracy thay cho Hamming Loss để dễ đọc; vẫn giữ loss cho BOP và audit.
- Chuẩn hóa “instant base F1” thành instance-based F1 nếu xác nhận đúng ý giảng viên.
- Bổ sung Instance Jaccard, Macro Precision/Recall và per-label metrics.
- Macro-F1 phải được báo cáo tách biệt cho complete, decided/selective, rejected counterfactual và optimistic scopes.
- Optimistic Macro-F1 giả định toàn bộ nhãn bị từ chối được con người gán lại đúng 100%; đây chỉ là upper bound và phải đi cùng coverage/review load.
- Kiểm tra riêng các nhãn quan trọng thay vì suy luận hiệu quả thực tế chỉ từ điểm trung bình.

### 2.3. Baseline và tính công bằng

- Bổ sung đầy đủ BR, CC, MLC-PA và GSI-MLC-PA với các base learner tương ứng.
- Dùng cùng fold, preprocessing, seed, base-learner hyperparameter, calibration và decision policy khi so sánh kiến trúc.
- Không so GSI-MLP với MLC-PA-Logistic rồi quy toàn bộ chênh lệch cho phương pháp IL/DL.

### 2.4. Bằng chứng cho khả năng ứng dụng

- Selective Macro-F1 tăng chưa đủ chứng minh mô hình tốt trong thực tế vì mô hình có thể bỏ qua các ca khó.
- Cần báo coverage, ABS/AABS, risk-at-coverage, error capture, review load, optimistic gain và critical-label metrics.
- Cần ablation `all_il`, `all_dl`, `learned`, `random_matched` và no-reorder để tách đóng góp của IL/DL partition khỏi base learner và chain order.

## 3. Quyết định kỹ thuật hiện tại

1. Kiến trúc đánh giá được tách thành `probability estimator -> decision policy -> metrics`.
2. Hamming, F1 và Jaccard là các decision policy khác nhau; không gọi threshold Hamming là F1-BOP.
3. F1/Jaccard BOP được nghiên cứu bằng dynamic programming dưới giả định conditional label independence.
4. MDP chỉ phù hợp nếu có sequential feedback làm thay đổi state/posterior; hiện được giữ như research spike, chưa nối vào core.
5. Kết quả mới dùng cache/output schema mới và không ghi đè kết quả hiện tại.
6. Công việc được chia thành các phase độc lập, mỗi phase tối đa một quota 5 giờ và có handoff bắt buộc.

## 4. Điểm cần xác nhận lại với giảng viên

- “Instant base F1” có đúng là **instance-based F1** hay là một metric/quy trình thời gian thực khác?
- “Macro-F1 trên cả hai tập” chỉ decided/rejected, IL/DL hay train/test?
- Nhãn nào được xem là quan trọng trên từng dataset/application?
- False-positive cost, false-negative cost, review cost và reviewer accuracy thực tế là bao nhiêu?
- Quy trình triển khai có nhận human feedback tuần tự để cập nhật xác suất các nhãn còn lại hay không? Nếu không, MDP chưa có transition thực tế.

## 5. Hành động tiếp theo

- Phase Q0: audit, khóa metric contract, dựng module skeleton và test fixtures.
- Phase Q1–Q2: cài metrics và tích hợp schema/cache v3.
- Phase Q3–Q6: tách decision policy, cài F1/Jaccard BOP và nối objective vào GSI.
- Phase Q7–Q11: IL/DL ablation, matched baselines, calibration, deployment metrics và system verification.
- Phase Q12–Q14: chạy benchmark theo batch resumable, phân tích thống kê và cập nhật báo cáo.
- Phase Q15: nghiên cứu MDP/marginalization tùy chọn.
