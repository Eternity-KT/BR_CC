**core**

Khởi tạo: 
   - Không gian đặc trưng: FS = X; DL = {tập tất cả các nhãn); IL = {rỗng}
   - tang doc lap: i=1;

Do { 
         IL[i]=rỗng; // lưu các nhãn độc lập ở tầng i; 
	 
   if (|DL|=1) // chi co 1 nhan trong DL
	  1. Huan luyen ham BR cho nhan trong DL trong khong gian dac trung FS;
          2. IL[i]=IL[i]U DL; DL=rong;
          2. Break; thoat khoi viec thuc hien tim cac nhan độc lập.
 
   // Tim cac nhan doc lap tren khong gian dac trung FS
    for (mỗi nhãn l thuộc DL)
            1. Huấn luyện hàm BR (f) trên không gian đặc trưng FS;
	    2. if (F1_score (f, l) >= threshold)
                   - IL[i]=IL[i]U{l}; DL=DL/{l}
                   - mark nhan l duoc huan luyen tren khong gian dac trung FS;
             
    // Bo sung cac nhan doc lap tren khong gian dac trung vao khong gian FS de tim cac nhan doc lap o tang khac
             if (|IL[i]|<> rong){
                       FS = FS U IL[i];
                       i=i+1; tăng lên tầng tiếp theo;
             }
              
             else  break; // thoat khoi vong tim cac nhan doc lap
  while (1);
// buoc tiep la du ly cac nhan trong DL
 if (DL<> rong){
           
           for (mỗi nhãn l thuộc DL){
              - FS[l]=FS;// xác định không gian đặc trưng tạm thời để học hàm đoán nhận nhãn l;
              - DL_temp[l]=rong; // luu cac nhan trong DL ma nhan l phu thuoc
              - for (mỗi nhãn p<>l thuộc DL)
                    + Tính tương quan của l và p (gọi là Corr(l,p)) theo cach tính PCC giua (l-f(l)) va (p - f(p)// trong đó: f(l) và f(p) lần lượt là BR classifier được học từ không gian FS;
                    + if Corr(l,p) >=threshold)
                         DL_temp[l]=DL_temp[l] U {p}
              - FS[l]=FS[l] U DL_temp[l]; cập nhận không gian đặc trưng của để đoán nhận nhãn l trong DL
              - dung BR để học hàm phân lớp cho l trong không gian đặc trưng FS[l]
            }
                  
 // Để đẩy cơ chế từ chối thì lúc học hàm BR cho từng lớp có thể dùng cơ chế từ chối ở đó. Ví dụ tại các mục

**option**
Thử giảm threshold theo các lần lặp - Làm sau khi chạy thử nghiệm bước Sử dụng xác suất từ BR để sắp xếp nhãn trong CC
Trong quá trình tính xác suất trong CC, thay vì chỉ đặt threshold 0.75 thì sẽ giảm dần theo thứ tự nhãn: Ví dụ nhãn ở đầu sẽ là 0.8-0.85 sau đó giảm dần đến ngưỡng 0.7-0.
Tổng hợp lại kết quả của các mô hình BR, CC, ECC đã public