**core**
Đặt lại ngưỡng tương quan về 0.75, 
Khảo sát xem độ mất cân bằng của các nhãn trong DL
Hạ threshold trên human và plant xuống 0.5
Phân tích chi tiết về mất cân bằng nhãn và các nguyên nhân gây giảm hiệu năng
Hạ threshold tầng IL 2 xuống 0,7, tầng 3 xuống 0,65
Trên human và plant thì dùng BR trên tập X tính f1 score, sắp xếp từ lớn đến bé, lấy median/mean (phải lớn hơn 0.5) làm threshold

**option**
Nghiên cứu thêm sự phụ thuộc từng phần do CC hiệu năng vẫn thấp trên 1 vài tập dữ liệu