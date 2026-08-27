Nghiên cứu thêm về Markov decision processes trong biên hóa xác suất
Sửa lại báo cáo phần tính macro f1 trên cả 2 tập
Nói rõ hơn về ý nghĩa các metrics, tại sao lại dùng
Trong báo cáo sử dụng hamming accuracy thay vì hamming loss để cho dễ đọc
Bổ sung đầy đủ các mô hình baseline với đủ các base learner
Cố gắng fix sao cho hyperparameter giống nhau so sánh được giữa các mô hình
Mô hình nâng cao được điểm macro f1 nhưng chưa đảm bảo tốt trong thực tế
Macro F1 chưa thực sự thể hiện rõ ràng về hiệu suất của mô hình, nhất là khi từ chối
Nghiên cứu về metrics Optimistic macro F1: Giả định các nhãn từ chối được người gán nhãn lại và chính xác 100%
Từ chỉ số optimistic macro F1, nghiên cứu tiếp xem liệu mình đã dự đoán đúng các nhãn quan trọng chưa, từ đó đánh giá lợi ích của mô hình abtension đem lại hiệu quả như nào trong ứng dụng thực tế
Thử tối ưu thêm hamming accuracy. instant base f1
Thực tế khi có instant thì cần đưa ra luôn được quyết định -> Cần instant base f1 để đánh giá hiệu quả của mô hình trong việc đưa ra quyết định ngay lập tức
Ở bước chia các nhãn DL và IL cần BOP cho instant base f1
So sánh với nhiều hàm mục tiêu khác nhau để đánh giá hiệu quả của mô hình, ví dụ như precision, recall, và F1-score cho từng nhãn
Chỉ ra việc chia nhãn IL/DL giúp tăng hiệu quả
Cần chứng minh mô hình này có ưu điểm tốt, có thể áp dụng thực tế (Chứng minh bằng metrics)
Bổ sung thêm jaccard metric 
Tạo 1 file ghi lại tóm tắt nội dung cuộc họp 