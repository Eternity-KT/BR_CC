**core**

Dùng 5 fold tính bộ phân loại của BR để tách nhãn IL, sau đó lại dùng 5 fold train lại bộ phân loại của CC (DL) với data = X + IL. Có bảng chi tiết quá trình phân tách nhãn IL qua các tầng trong báo cáo.
Khi tập DL chỉ còn 1 nhãn thì đưa luôn vào IL
2 Hướng xử lý tập DL
- Hướng 1 cho tập DL: Thay CC bằng ECC để k cần dùng tương quan phi để sắp xếp - Ưu tiên thực hiện trước, gọi là v6.1
- Hướng 2 cho tập DL: Tính tương quan giữa |y1-y1*| với |yn-yn*| với yn* là dự đoán của mô hình BR. Show bảng chi tiết quá tình tính tương quan này trong báo cáo - Thực hiện sau khi hoàn thành hướng 1, gọi là v6.2

Khi thực hiện hướng 2 thì không tách nhánh mới mà cô lập kết quả chạy cũng như những thành phần khác của nó dể không ảnh hưởng đến hướng 1

Sau khi chạy xong thử nghiệm với ECC (v6.1) thì làm 3 việc:
- Giải thích tại sao humanspeaac và plantspeaac hiệu năng tổng thấp
- Giải thích tại sao genbase lại bị 0 điểm khi tách riêng DL ra tính
- Giải thích tại sao ở bộ phân loại MLP tách được ít nhãn hơn logistic và svm, hiệu năng trên tập IL cũng thấp hơn đáng kể

**option**
Thử giảm threshold theo các lần lặp - Làm sau khi chạy thử nghiệm bước Sử dụng xác suất từ BR để sắp xếp nhãn trong CC
Trong quá trình tính xác suất trong CC, thay vì chỉ đặt threshold 0.75 thì sẽ giảm dần theo thứ tự nhãn: Ví dụ nhãn ở đầu sẽ là 0.8-0.85 sau đó giảm dần đến ngưỡng 0.7-0.
Tổng hợp lại kết quả của các mô hình BR, CC, ECC đã public