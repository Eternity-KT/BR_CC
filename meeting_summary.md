Chỉnh lại sắp xếp chuỗi CC trong tập DL đưa các nhãn tổng tương quan nhỏ lên đầu thay vì tổng lớn hơn
Làm lại phần phân hoạch IL DL theo logic thông thường, ko dùng tham lam macro f1 nữa, chỉ dựa vào dữ liệu x để phân chia nhãn y vào IL hay DL
Chỉ tính đến việc 1 nhãn cung cấp thông tin cho nhãn khác sau khi chia nhãn
Vd: 1 bệnh có dữ liệu, từ bệnh A này sinh ra bệnh B khác thì k có nghĩa bệnh B quyết định bệnh A
Khi biết nhãn độc lập r thì dùng BR chạy trước để dự đoán sau đó tính đến DL sau
Nếu 1 nhãn có thể dự đoán tốt chỉ dựa vào data với high perform thì có thể coi là độc lập ( có thể tính thêm accuracy hoặc f1 score nếu vượt ngưỡng thì đưa vào IL)
Trong DL cần trả lời câu hỏi: Nhãn nào phụ thuộc nhãn nào? Sắp xếp nhãn trong CC như nào cho phù hợp?
Cần chọn threshold để chọn nhãn vào IL phù hợp (khoảng 0,7-0,8 điểm f1)
Khi dự đoán đc nhãn IL thì có thể dùng làm data để dự đoán nhãn DL. Khi đó lặp lại quá trình chọn nhãn IL bên trên với tập DL để tách tiếp các nhãn ra 1 tập IL_2 chỉ phụ thuộc data và IL ban đầu, lặp lại nhiều lần đến khi không còn nhãn nào có xác suất lớn hơn threshold, sau khi dừng quá trình này thì dùng CC cho các nhãn còn lại / Dùng sort coefficent và CC như hiện tại
Nếu trong quá trình làm mà mô hình cũ bị sụp đổ thì cần phải sửa mô hình
Nghiên cứu thêm nhóm phụ thuộc và nhóm độc lập (Làm sau khi test các phần trên)
Nghiên cứu thêm về hàm phạt cho quá trình phân hoạch nhãn để tránh độ phức tạp thời gian quá lớn, lãng phí thời gian chạy