**Phần Introduction**
- Thiếu kết quả thu được, cần thêm các gạch đều dòng về tóm tắt đóng góp chính của phương pháp đề xuất, đóng góp của abtension trước sau đó mới đến tóm tắt bài báo
- Có 1 phần tóm tắt kết quả thực nghiệm để chỉ ra bài báo đáng được review
- Thêm timeline của các nghiên cứu trước đây, bổ sung câu chuyện vào giới thiệu, lỗ hổng các nghiên cứu trước
- Nhấn mạnh nhược điểm của nghiên cứu trước và cách mình khắc phục
- Bỏ các chỉ số thực nghiệm khỏi giới thiệu
- Thêm các đóng góp của mình để giải quyết bài toán
- Link sourecode của nghiên cứu
- Nói qua về những mô hình trước đây được dùng để so sánh với mô hình của mình

**Chung**
- Thống nhất các thuật ngữ đúng chuẩn
- Bổ sung các ưu điểm, nhược điểm của các kĩ thuật trước đây
- Tầm quan trọng của phương pháp của mình trong thực tế: Ví dụ: Tầm quan trọng của phương pháp từ chối trong lĩnh vực yêu cầu chính xác cao như y sinh, y học, tài chính
- Viết theo đúng quy trình deep understand mô hình => Viết bài báo

**Chi tiết về mô hình**
- Nói rõ dùng những phương pháp, lý thuyết gì
- Chuẩn hóa kí hiệu toán học
- Methology: Chi tiết các bước thực hiện, ý nghĩa của từng bước, tại sao lại làm vậy
- Thuật toán: Code hoặc giả mã thuật toán
- Công thức: Công thức toán học
- Lý thuyết: Giải thích lí thuyết nền tảng
- Không viết quá nhiều về lựa chọn nhãn tham lam, nên tích hợp phần này vào lựa chọn nhãn
- Viết sao cho dễ hiểu, không dùng quá nhiều từ chuyên ngành khó hiểu
- Nói rõ hơn các tính phụ thuộc giữa nhãn trong tập IL và DL: 1 nhãn thuộc DL phụ thuộc vào các nhãn tập IL như thế nào, cách tính như nào,...
- Các siêu tham số chọn trước cần phải giải thích tại sao lại chọn như vậy
- Nói rõ hơn cách chọn chi phí từ chối
- Giải thích tại sao chọn SEP thay vì PAR
- Nếu không chọn PAR thì bỏ qua, không thêm vào
- Có gì trong mô hình thì giải thích hết tại sao lại chọn như vậy (Giải thích một cách tự nhiên, không tách riêng khỏi các phần được giải thích)

**Phần thực nghiệm**
- Đưa các công thức metric lên phần background thay vì để ở thực nghiệm
- Tên các bảng/biểu đồ phải rõ ràng, không gây nhầm lẫn
- Thêm giải thích tại sao có các tập thấp hơn mô hình baseline MLC PA
- Nếu đã chọn c tự động thì không nên đưa vào bảng/biểu đồ
- Phân tích kết quả, lí do tại sao đạt được
- Có thể thêm 1 phần để tối ưu thêm cho F1 thay vì hamming
- Đếm số lần các thuật toán khác kém hơn so với phương pháp đề xuất để đưa vào paper
- Có thể chỉ số instance based F1 bị giảm do mất cân bằng dữ liệu
- Nghiên cứu thêm phần instance jaccard 
- Thêm instance based f1 khi tính nhãn từ chối
- Nêu rõ chỉ số coverage để so sánh với mlcpa khi loss tương tự nhau trong các bảng so sánh
- Nếu được thì Bỏ cái bảng hiệu năng trên cost vì rất khó giải thích
- Thêm 1 phần chuyên tối ưu instance based f1 tương tự phần tối ưu hamming hiện tại để có thể cạnh tranh với mlcpa
- Thiết kế thực nghiệm chuyên sâu hơn để thuyết phục hơn

**core**
- Nghiên cứu lại phần sắp xếp nhãn IL và DL, cần rõ ràng phần này để cải thiện performance 
- Tách các nhãn IL ra riêng để đánh giá bằng BR, các nhãn thuộc DL sử dụng CC

**bên lề**
- Check các phương pháp hiện tại tương tự có mạnh hơn của mình không
- Lên deadline sửa paper
