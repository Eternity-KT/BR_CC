**core**
giờ tách ra, kiểm tra nhãn y1 có IL không sau đó tiếp tục kiểm tra y2 có DL không
từng bước vậy
muốn kiểm tra y1 có DL không thì học mô hình phân lớp (logistic regression, SVM, MLP)
cách học mô hình là dùng 5-fold cross validation
dựa vào kiểm định chéo này tính Selective-F1 nếu lớn hơn 0.75 thì KL nhãn y1 là độc lập cho vào tập IL
làm tương tự như thế cho các nhãn còn lại
kết thúc bước này được tập IL ở tầng 1;
bổ sung các nhãn ở tập IL này vào làm đặc trưng cho tầng thứ 2
Làm tương tự như bước trước đối với tập DL còn lại để tìm tập IL ở tầng thứ 2
**option**
Xem các mô hình BR hiệu năng như nào trên các nhãn độc lập
Thử giảm threshold theo các lần lặp - Làm sau khi chạy thử nghiệm bước Sử dụng xác suất từ BR để sắp xếp nhãn trong CC
Trong quá trình tính xác suất trong CC, thay vì chỉ đặt threshold 0.75 thì sẽ giảm dần theo thứ tự nhãn: Ví dụ nhãn ở đầu sẽ là 0.8-0.85 sau đó giảm dần đến ngưỡng 0.7-0.
Nếu CC trong DL vẫn thấp thì có thể dung ECC