**core**
Tách các bảng IL và DL khi chạy thử nghiệm phân tích
Thử nghiệm khi tách riêng DL thì ko dung IL, chỉ dung tập data X
Nếu CC trong DL vẫn thấp thì có thể dung ECC
Đẩy mô hình dự đoán vào tính mối tương quan giữa các nhãn trong DL


**option**
Xem các mô hình BR hiệu năng như nào trên các nhãn độc lập
Thử giảm threshold theo các lần lặp - Làm sau khi chạy thử nghiệm bước Sử dụng xác suất từ BR để sắp xếp nhãn trong CC
Trong quá trình tính xác suất trong CC, thay vì chỉ đặt threshold 0.75 thì sẽ giảm dần theo thứ tự nhãn: Ví dụ nhãn ở đầu sẽ là 0.8-0.85 sau đó giảm dần đến ngưỡng 0.7-0.75