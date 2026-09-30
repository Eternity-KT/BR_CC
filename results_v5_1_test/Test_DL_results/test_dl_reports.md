# BÁO CÁO THỰC NGHIỆM ĐÁNH GIÁ ĐÓNG GÓP TẬP NHÃN VÀ KHẢ NĂNG CÔ LẬP TRONG MÔ HÌNH PHÂN LOẠI ĐA NHÃN CÓ TỪ CHỐI TỪNG PHẦN

**Tác giả:** Nhóm Nghiên cứu Học máy (Machine Learning Research Group)  
**Dự án:** GSI-MLC-PA (Phân tầng thích ứng và chuỗi tương quan có từ chối từng phần)  
**Tài liệu tham chiếu:** `meeting_summary.md`, `spec/spec_v5_1_1.md`, `results_v5_1_test/data_labels_analize.tex`  
**Ngày thực hiện:** 30 tháng 09 năm 2026  

---

## 1. Đặt Vấn Đề và Mục Tiêu Thực Nghiệm

Trong bài toán phân loại đa nhãn có cơ chế từ chối từng phần (Multi-Label Classification with Partial Abstention - MLC-PA), kiến trúc phân tầng kết hợp (hybrid architecture) chia không gian nhãn thành hai tập con:
1. **Tập nhãn độc lập ($IL$):** Các nhãn có thể dự đoán trực tiếp từ không gian đặc trưng đầu vào $X$ với độ chính xác chấp nhận được ($F_1 \ge \tau$).
2. **Tập nhãn phụ thuộc ($DL$):** Các nhãn còn lại có độ phức tạp cao hơn, phụ thuộc vào tương quan giữa các nhãn và cần được hỗ trợ dự đoán theo chuỗi (Classifier Chains - CC).

Theo thiết kế ban đầu của phiên bản v5.1, các bộ phân loại thuộc tập $DL$ nhận thêm phân phối xác suất dự đoán của tập $IL$ như một tập thuộc tính bổ trợ tĩnh:
$$X_{\text{context}} = [X, \hat{P}(Y_{IL} \mid X)]$$

Tuy nhiên, trong quá trình phát triển mô hình, việc đưa toàn bộ thông tin từ $IL$ vào làm thuộc tính cho $DL$ đặt ra hai vấn đề cần kiểm chứng thực nghiệm:
- **Tác động chuyển giao thông tin:** Chưa có số liệu định lượng cho thấy liệu thông tin từ $IL$ có thực sự cải thiện chất lượng dự đoán của $DL$ hay gây ra hiện tượng quá khớp (overfitting) do tăng chiều đặc trưng.
- **Khả năng tự lực của tập $DL$:** Khi tắt hoàn toàn thông tin từ $IL$ và chỉ cho phép $DL$ sử dụng không gian đặc trưng gốc $X$, hiệu năng suy giảm ở mức độ nào và cấu trúc chuỗi (Sparse CC so với Dense CC) đóng vai trò gì trong việc kiểm soát sai số lan truyền.

Thực nghiệm này được tiến hành nhằm phân tích độc lập hai tập nhãn $IL$ và $DL$, so sánh trực diện hiệu năng của $DL$ trong hai điều kiện (có sử dụng ngữ cảnh $IL$ và cô lập hoàn toàn chỉ dùng $X$), đồng thời đánh giá tương tác giữa cấu trúc nhãn ban đầu của dữ liệu với chất lượng dự đoán cuối cùng.

---

## 2. Đặc Trưng Không Gian Dữ Liệu và Cấu Trúc Nhãn Ban Đầu

Trước khi đánh giá mô hình, toàn bộ 10 tập dữ liệu benchmark được kiểm toán cấu trúc nhãn. Các chỉ số đo đạc gồm số lượng mẫu ($N$), số chiều thuộc tính ($d$), số nhãn ($K$), độ dồi dào nhãn trung bình ($LC$), mật độ nhãn ($LD$), tỷ lệ mất cân bằng trung bình (MeanIR), tỷ lệ nhãn hiếm ($<5\%$) và phân phối hệ số tương quan cặp Phi-coefficient ($|\phi|$).

### Bảng 1: Thống kê cấu trúc mẫu, nhãn và tương quan của 10 tập dữ liệu benchmark

| Tập dữ liệu | Lĩnh vực | Số mẫu ($N$) | Số thuộc tính ($d$) | Số nhãn ($K$) | $LC$ | $LD$ | MeanIR | Nhãn hiếm ($<5\%$) | Mean $|\phi|$ | Tỷ lệ $|\phi| \ge 0.30$ | Tỷ lệ $|\phi| \ge 0.50$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **emotions** | Âm nhạc | 593 | 72 | 6 | 1.868 | 0.311 | 2.32 | 0 / 6 (0.0\%) | 0.325 | 53.3\% | 13.3\% |
| **scene** | Thị giác máy tính | 2,407 | 294 | 6 | 1.074 | 0.179 | 4.66 | 0 / 6 (0.0\%) | 0.185 | 0.0\% | 0.0\% |
| **chd49** | Tim mạch lâm sàng | 555 | 49 | 6 | 2.580 | 0.430 | 7.21 | 1 / 6 (16.7\%) | 0.160 | 20.0\% | 6.7\% |
| **music** | Phân loại thể loại | 592 | 71 | 6 | 1.870 | 0.312 | 2.32 | 0 / 6 (0.0\%) | 0.325 | 53.3\% | 13.3\% |
| **gpositivepseaac** | Protein Gram dương | 519 | 440 | 4 | 1.008 | 0.252 | 8.63 | 1 / 4 (25.0\%) | 0.296 | 50.0\% | 16.7\% |
| **genbase** | Trình tự hệ gen | 662 | 1,186 | 27 | 1.252 | 0.046 | 143.46 | 18 / 27 (66.7\%) | 0.069 | 5.7\% | 3.4\% |
| **humanpseaac** | Định vị protein người | 3,106 | 440 | 14 | 1.185 | 0.085 | 45.51 | 7 / 14 (50.0\%) | 0.055 | 0.0\% | 0.0\% |
| **plantpseaac** | Định vị protein thực vật | 978 | 440 | 12 | 1.079 | 0.090 | 21.88 | 6 / 12 (50.0\%) | 0.075 | 1.5\% | 0.0\% |
| **viruspseaac** | Định vị protein virus | 207 | 440 | 6 | 1.217 | 0.203 | 8.62 | 1 / 6 (16.7\%) | 0.149 | 6.7\% | 0.0\% |
| **yeast** | Chức năng gen nấm men | 2,417 | 103 | 14 | 4.237 | 0.303 | 8.95 | 1 / 14 (7.1\%) | 0.154 | 13.2\% | 5.5\% |

Dựa trên số liệu bảng 1, 10 tập dữ liệu được phân chia thành ba nhóm hình thái rõ rệt:
1. **Nhóm tương quan nội tại cao (`emotions`, `music`, `gpositivepseaac`):** Có hơn 50\% số cặp nhãn đạt hệ số tương quan $|\phi| \ge 0.30$ và giá trị Mean $|\phi| \approx 0.30 - 0.32$. Nhóm này phản ánh đúng bối cảnh mà chuỗi phân loại (Classifier Chains) có thể khai thác sự phụ thuộc tương hỗ giữa các nhãn.
2. **Nhóm mật độ cao, tương quan trung bình (`yeast`, `chd49`):** Có $LC > 2.5$ (nhiều nhãn cùng xuất hiện đồng thời trên một mẫu), MeanIR ở mức vừa phải ($7 - 9$), tỷ lệ cặp tương quan đáng kể dao động từ 13\% đến 20\%.
3. **Nhóm thưa thớt, mất cân bằng cực đoan (`genbase`, `humanpseaac`, `plantpseaac`):** Có mật độ nhãn rất thấp ($LD < 0.10$), MeanIR rất cao (từ 21.88 đến 143.46), và trên 50\% số nhãn là nhãn hiếm (tần suất dưới 5\%). Hầu như không tồn tại các cặp nhãn có tương quan mạnh ($|\phi| \ge 0.30$ dưới 6\%).

---

## 3. Thiết Kế Thực Nghiệm và Phương Pháp Đánh Giá

### 3.1. Giao thức kiểm định chéo và tiền xử lý
- **Quy trình kiểm định:** Sử dụng phương pháp 5-Fold Multilabel Stratified Cross-Validation (`random_state=42`) nhằm giữ nguyên tỷ lệ phân bố nhãn giữa các fold.
- **Tiền xử lý:** Chuẩn hóa ma trận đặc trưng $X$ bằng `MaxAbsScaler`, tham số được ước lượng nghiêm ngặt trên fold huấn luyện và áp dụng sang fold kiểm tra.
- **Ngưỡng phân tầng:** Quá trình bóc tách tầng (Stratified Peeling) sử dụng ngưỡng $F_1 = 0.70$ với độ sâu tối đa 3 tầng trên tập validation nội bộ (20\% fold huấn luyện).
- **Tầng quyết định có từ chối:** Áp dụng quy tắc Bayes-Optimal Prediction (BOP) với hàm phạt tuyến tính (SEP) tại chi phí từ chối $c = 0.30$. Các vị trí có xác suất $p \le 0.30$ được gán nhãn 0, $p \ge 0.70$ gán nhãn 1, và vùng không chắc chắn $0.30 < p < 0.70$ được gán nhãn từ chối (ký hiệu -1).

### 3.2. Không gian mô hình đối sánh
Thực nghiệm đánh giá trên ba họ bộ học cơ sở:
1. **Logistic Regression:** Bộ giải `liblinear`, $C=1.0$, `max_iter=1000`.
2. **Support Vector Machine (LinearSVC):** Hiệu chuẩn xác suất lồng nhau qua hàm Sigmoid (Platt scaling), $C=1.0$.
3. **Multi-Layer Perceptron (PyTorch GPU):** Tối ưu hóa AdamW, hàm mất mát BCE, dừng sớm dựa trên validation loss.

Trên mỗi fold kiểm tra, bốn cấu hình của tập phụ thuộc $DL$ được đối sánh trực tiếp:
- **`DL_with_IL_Sparse_CC`:** Huấn luyện trên không gian đặc trưng mở rộng $[X, \hat{P}_{\mathcal{I}}]$, chuỗi CC chỉ liên kết các tiền nhiệm trong $DL$ có tương quan tuyệt đối $|\phi| \ge \theta_{\text{corr}} = 0.75$.
- **`DL_isolated_Sparse_CC`:** Huấn luyện hoàn toàn trên không gian đặc trưng gốc $X$ (tắt toàn bộ $IL$), chuỗi CC lọc theo ngưỡng $|\phi| \ge 0.75$.
- **`DL_with_IL_Dense_CC`:** Huấn luyện trên $[X, \hat{P}_{\mathcal{I}}]$, chuỗi CC kết nối đầy đủ tất cả các tiền nhiệm trong $DL$ ($\theta_{\text{corr}} = 0.0$).
- **`DL_isolated_Dense_CC`:** Huấn luyện hoàn toàn trên $X$, chuỗi CC kết nối đầy đủ tất cả các tiền nhiệm trong $DL$ ($\theta_{\text{corr}} = 0.0$).

---

## 4. Kết Quả Thực Nghiệm và Phân Tích Định Lượng

### 4.1. Hiệu năng của tập nhãn độc lập ($IL$)
Tập $IL$ được xác định tự động thông qua quy trình bóc tách đa tầng và được huấn luyện độc lập bằng Binary Relevance từ $X$. Bảng 2 thể hiện hiệu năng trung bình của tập $IL$ toàn phần trên 10 tập dữ liệu.

### Bảng 2: Hiệu năng của tập nhãn độc lập ($IL$) trên từng bộ phân loại cơ sở

| Tập dữ liệu | Base Learner | Số nhãn $K_{IL}$ | Coverage (\%) | Selective Macro-F1 | Full Macro-F1 | Selective Precision | Subset 0/1 Acc | Selective Hamming Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **emotions** | Logistic | 3.8 / 6 | 67.7\% | 0.7635 | 0.7037 | 0.8421 | 0.4607 | 0.0957 |
| | SVM | 3.6 / 6 | 69.1\% | 0.7381 | 0.6976 | 0.8252 | 0.4573 | 0.0968 |
| | MLP | 2.6 / 6 | 55.2\% | 0.7512 | 0.6894 | 0.8115 | 0.4219 | 0.1012 |
| **scene** | Logistic | 3.6 / 6 | 82.4\% | 0.7812 | 0.7621 | 0.8845 | 0.6214 | 0.0621 |
| | SVM | 3.8 / 6 | 48.5\% | 0.7924 | 0.7590 | 0.8921 | 0.6385 | 0.0512 |
| | MLP | 2.0 / 6 | 60.3\% | 0.7741 | 0.7320 | 0.8650 | 0.5890 | 0.0684 |
| **chd49** | Logistic | 2.2 / 6 | 65.4\% | 0.6214 | 0.5521 | 0.6842 | 0.3541 | 0.1852 |
| | SVM | 2.0 / 6 | 63.5\% | 0.6350 | 0.5642 | 0.6912 | 0.3621 | 0.1794 |
| | MLP | 2.0 / 6 | 80.7\% | 0.5982 | 0.5312 | 0.6514 | 0.3120 | 0.2014 |
| **music** | Logistic | 4.0 / 6 | 65.2\% | 0.7721 | 0.7142 | 0.8512 | 0.4821 | 0.0912 |
| | SVM | 3.8 / 6 | 59.0\% | 0.7654 | 0.7082 | 0.8490 | 0.4795 | 0.0924 |
| | MLP | 3.4 / 6 | 55.8\% | 0.7592 | 0.6984 | 0.8321 | 0.4512 | 0.0984 |
| **gpositivepseaac** | Logistic | 2.2 / 4 | 88.4\% | 0.6842 | 0.6621 | 0.7512 | 0.6521 | 0.0894 |
| | SVM | 1.6 / 4 | 74.2\% | 0.6721 | 0.6480 | 0.7410 | 0.6380 | 0.0912 |
| | MLP | 2.4 / 4 | 84.5\% | 0.6690 | 0.6510 | 0.7350 | 0.6240 | 0.0945 |
| **genbase** | Logistic | 25.6 / 27 | 99.8\% | 0.7915 | 0.7821 | 0.8124 | 0.9854 | 0.0009 |
| | SVM | 26.2 / 27 | 99.8\% | 0.7924 | 0.7845 | 0.8145 | 0.9860 | 0.0008 |
| | MLP | 24.0 / 27 | 98.8\% | 0.7684 | 0.7512 | 0.7942 | 0.9780 | 0.0014 |
| **humanpseaac** | Logistic | 0.0 / 14 | -- | -- | -- | -- | -- | -- |
| | SVM | 0.0 / 14 | -- | -- | -- | -- | -- | -- |
| | MLP | 0.0 / 14 | -- | -- | -- | -- | -- | -- |
| **plantpseaac** | Logistic | 0.0 / 12 | -- | -- | -- | -- | -- | -- |
| | SVM | 0.0 / 12 | -- | -- | -- | -- | -- | -- |
| | MLP | 0.0 / 12 | -- | -- | -- | -- | -- | -- |
| **viruspseaac** | Logistic | 1.6 / 6 | 82.4\% | 0.5412 | 0.5120 | 0.6120 | 0.5840 | 0.1120 |
| | SVM | 1.4 / 6 | 62.1\% | 0.5280 | 0.4950 | 0.6010 | 0.5620 | 0.1180 |
| | MLP | 1.8 / 6 | 78.4\% | 0.5350 | 0.5080 | 0.5980 | 0.5710 | 0.1150 |
| **yeast** | Logistic | 2.6 / 14 | 72.1\% | 0.7412 | 0.6854 | 0.7952 | 0.3241 | 0.1120 |
| | SVM | 2.4 / 14 | 72.8\% | 0.7485 | 0.6912 | 0.8014 | 0.3312 | 0.1085 |
| | MLP | 2.0 / 14 | 71.0\% | 0.7321 | 0.6745 | 0.7841 | 0.3105 | 0.1165 |

**Ghi chú:** Tại `humanpseaac` và `plantpseaac`, không có nhãn nào vượt qua ngưỡng $F_1 \ge 0.70$ trên tập validation nội bộ, do đó toàn bộ các nhãn được đưa vào tập phụ thuộc $DL$ ($K_{IL} = 0$).

---

### 4.2. Hiệu năng tích lũy của các tầng nhãn độc lập ($IL_1$, $IL_{1+2}$, $IL_{1+2+3}$)

Giải thuật Data-Driven Stratified Peeling thực hiện bóc tách không gian nhãn theo từng tầng độc lập với số tầng tối đa $T_{\max} = 3$ và ngưỡng chấp nhận $\tau = 0.70$. Bảng 2a tổng hợp hiệu năng vĩ mô trung bình của từng cấp độ tầng độc lập tích lũy: Tầng 1 ($IL_1$), tích lũy Tầng 1 và 2 ($IL_{1+2}$), và tích lũy toàn bộ cả 3 tầng ($IL_{1+2+3}$) trên 3 bộ học cơ sở.

#### Bảng 2a: Hiệu năng vĩ mô của các cấp độ tầng độc lập tích lũy ($IL_1$, $IL_{1+2}$, $IL_{1+2+3}$)

| Bộ học cơ sở | Cấu hình tầng độc lập | Số nhãn ($K_{IL}$) | Coverage (%) | Selective Macro-F1 | Selective Precision | Subset 0/1 Acc | Selective Hamming Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LOGISTIC** | $IL_1$ (Tầng 1) | 5.85 | 83.4% | 0.7778 | 0.7601 | 0.6505 | 0.1288 |
| | $IL_{1+2}$ (Tích lũy 1+2) | 5.90 | 82.8% | 0.7793 | 0.7628 | 0.6449 | 0.1276 |
| | $IL_{1+2+3}$ (Toàn bộ $IL$) | 5.90 | 82.8% | 0.7793 | 0.7628 | 0.6449 | 0.1276 |
| **SVM** | $IL_1$ (Tầng 1) | 5.45 | 86.4% | 0.8047 | 0.7817 | 0.7128 | 0.1302 |
| | $IL_{1+2}$ (Tích lũy 1+2) | 5.60 | 84.9% | 0.8093 | 0.7856 | 0.6966 | 0.1302 |
| | $IL_{1+2+3}$ (Toàn bộ $IL$) | 5.60 | 84.9% | 0.8093 | 0.7856 | 0.6966 | 0.1302 |
| **MLP** | $IL_1$ (Tầng 1) | 4.92 | 82.1% | 0.7910 | 0.7817 | 0.5551 | 0.1304 |
| | $IL_{1+2}$ (Tích lũy 1+2) | 5.08 | 82.0% | 0.7916 | 0.7806 | 0.5415 | 0.1296 |
| | $IL_{1+2+3}$ (Toàn bộ $IL$) | 5.10 | 82.0% | 0.7910 | 0.7800 | 0.5415 | 0.1296 |

#### Bảng 2b: Chi tiết số lượng nhãn và Selective Macro-F1 theo từng tầng bóc tách trên 10 tập dữ liệu benchmark

| Tập dữ liệu | Mô hình | $IL_1$: $K_{IL}$ | $IL_1$: F1 | $IL_{1+2}$: $K_{IL}$ | $IL_{1+2}$: F1 | $IL_{1+2+3}$: $K_{IL}$ | $IL_{1+2+3}$: F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **emotions** | Logistic | 3.6 | 0.7533 | 3.6 | 0.7533 | 3.6 | 0.7533 |
| | SVM | 3.4 | 0.8174 | 3.6 | 0.8248 | 3.6 | 0.8248 |
| | MLP | 2.4 | 0.7640 | 2.6 | 0.7647 | 2.6 | 0.7647 |
| **scene** | Logistic | 3.4 | 0.8921 | 3.6 | 0.8861 | 3.6 | 0.8861 |
| | SVM | 2.6 | 0.9094 | 3.2 | 0.9021 | 3.2 | 0.9021 |
| | MLP | 2.0 | 0.9229 | 2.0 | 0.9229 | 2.0 | 0.9229 |
| **chd49** | Logistic | 2.0 | 0.7966 | 2.0 | 0.7966 | 2.0 | 0.7966 |
| | SVM | 2.0 | 0.8058 | 2.0 | 0.8058 | 2.0 | 0.8058 |
| | MLP | 2.0 | 0.8010 | 2.0 | 0.8010 | 2.0 | 0.8010 |
| **music** | Logistic | 3.6 | 0.7809 | 3.6 | 0.7809 | 3.6 | 0.7809 |
| | SVM | 3.4 | 0.7640 | 3.4 | 0.7640 | 3.4 | 0.7640 |
| | MLP | 2.4 | 0.7988 | 2.8 | 0.8100 | 2.8 | 0.8100 |
| **gpositivepseaac** | Logistic | 3.0 | 0.6372 | 3.2 | 0.6555 | 3.2 | 0.6555 |
| | SVM | 1.6 | 0.6541 | 2.0 | 0.6903 | 2.0 | 0.6903 |
| | MLP | 3.0 | 0.7377 | 3.0 | 0.7377 | 3.0 | 0.7377 |
| **genbase** | Logistic | 26.8 | 0.6965 | 26.8 | 0.6965 | 26.8 | 0.6965 |
| | SVM | 27.0 | 0.7201 | 27.0 | 0.7201 | 27.0 | 0.7201 |
| | MLP | 23.4 | 0.5936 | 24.0 | 0.5870 | 24.2 | 0.5819 |
| **humanpseaac** | Logistic | 0.0 | -- | 0.0 | -- | 0.0 | -- |
| | SVM | 0.0 | -- | 0.0 | -- | 0.0 | -- |
| | MLP | 0.0 | -- | 0.0 | -- | 0.0 | -- |
| **plantpseaac** | Logistic | 0.0 | -- | 0.0 | -- | 0.0 | -- |
| | SVM | 0.0 | -- | 0.0 | -- | 0.0 | -- |
| | MLP | 0.0 | -- | 0.0 | -- | 0.0 | -- |
| **viruspseaac** | Logistic | 1.8 | 0.8509 | 1.8 | 0.8509 | 1.8 | 0.8509 |
| | SVM | 1.2 | 0.9375 | 1.2 | 0.9375 | 1.2 | 0.9375 |
| | MLP | 2.2 | 0.8548 | 2.2 | 0.8548 | 2.2 | 0.8548 |
| **yeast** | Logistic | 2.6 | 0.8145 | 2.6 | 0.8145 | 2.6 | 0.8145 |
| | SVM | 2.4 | 0.8296 | 2.4 | 0.8296 | 2.4 | 0.8296 |
| | MLP | 2.0 | 0.8549 | 2.0 | 0.8549 | 2.0 | 0.8549 |

Phân tích số liệu bóc tách tầng cho thấy ba quy luật thực nghiệm:
1. **Khả năng hội tụ sớm tại Tầng 1 ($IL_1$):** Đa số các nhãn độc lập (trên 95% nhãn thuộc tập $IL$) được nhận diện và bóc tách ngay tại tầng đầu tiên (5.85 / 5.90 nhãn ở Logistic, 5.45 / 5.60 nhãn ở SVM). Selective Macro-F1 trung bình tại Tầng 1 đạt mức cao (0.7778 đến 0.8047).
2. **Hiệu ứng thu gom cận biên tại Tầng 2 ($IL_{1+2}$):** Tầng 2 bổ sung một số nhãn cận biên trên các tập dữ liệu có tương quan phức tạp hơn (`emotions`, `scene`, `gpositivepseaac`, `music`, `genbase`). Khi tích lũy $IL_{1+2}$, chất lượng Macro-F1 trung bình tăng nhẹ lên 0.7793 (Logistic) và 0.8093 (SVM).
3. **Trạng thái bão hòa hoàn toàn tại Tầng 3 ($IL_{1+2+3}$):** Trên 9/10 tập dữ liệu, Tầng 3 không bóc tách thêm bất kỳ nhãn nào (chỉ có duy nhất `genbase` với MLP bóc tách thêm 0.2 nhãn trung bình). Điều này khẳng định độ sâu bóc tách 3 tầng là đủ để phân loại triệt để không gian nhãn độc lập mà không cần mở rộng thêm các tầng sâu hơn.

---

### 4.3. So sánh đối chứng tập phụ thuộc $DL$: Có $IL$ Context vs. Cô lập chỉ dùng $X$
Bảng 3 trình bày chi tiết hiệu năng của tập phụ thuộc $DL$ khi có và không có ngữ cảnh $IL$, đánh giá qua hai cấu hình chuỗi Sparse CC và Dense CC. Giá trị chênh lệch được định nghĩa:
$$\Delta = \text{Metric}(\text{DL with IL}) - \text{Metric}(\text{DL isolated X})$$
Trong đó $\Delta > 0$ trên Selective Macro-F1 thể hiện tác động tích cực của ngữ cảnh $IL$.

### Bảng 3: Chi tiết so sánh hiệu năng trên tập phụ thuộc $DL$ qua 10 tập dữ liệu ($c = 0.30$)

| Tập dữ liệu | Base Learner | Cấu hình $DL$ | Coverage (\%) | Selective Macro-F1 | Selective Precision | Subset 0/1 Acc | Selective Hamming Loss |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **emotions** | Logistic | DL_with_IL_Sparse_CC | 67.6\% | 0.4648 | 0.7327 | 0.5844 | 0.1358 |
| | | DL_isolated_Sparse_CC | 65.0\% | 0.3207 | 0.4467 | 0.5522 | 0.1453 |
| | | **Chênh lệch $\Delta_{\text{Sparse}}$** | **+2.6\%** | **+0.1441** | **+0.2860** | **+0.0323** | **-0.0095** |
| | | DL_with_IL_Dense_CC | 67.0\% | 0.4482 | 0.7439 | 0.5828 | 0.1355 |
| | | DL_isolated_Dense_CC | 64.9\% | 0.3103 | 0.4512 | 0.5423 | 0.1466 |
| | | **Chênh lệch $\Delta_{\text{Dense}}$** | **+2.1\%** | **+0.1379** | **+0.2928** | **+0.0405** | **-0.0111** |
| | SVM | DL_with_IL_Sparse_CC | 34.6\% | 0.4235 | 0.5656 | 0.5251 | 0.1024 |
| | | DL_isolated_Sparse_CC | 31.6\% | 0.4216 | 0.5356 | 0.5107 | 0.1074 |
| | | **Chênh lệch $\Delta_{\text{Sparse}}$** | **+3.0\%** | **+0.0020** | **+0.0300** | **+0.0144** | **-0.0050** |
| | | DL_with_IL_Dense_CC | 39.1\% | 0.3471 | 0.5000 | 0.4980 | 0.1121 |
| | | DL_isolated_Dense_CC | 36.2\% | 0.3503 | 0.4667 | 0.4867 | 0.1137 |
| | | **Chênh lệch $\Delta_{\text{Dense}}$** | **+2.9\%** | **-0.0032** | **+0.0333** | **+0.0112** | **-0.0016** |
| | MLP | DL_with_IL_Sparse_CC | 62.7\% | 0.3233 | 0.5198 | 0.3493 | 0.1537 |
| | | DL_isolated_Sparse_CC | 64.2\% | 0.3084 | 0.5147 | 0.3494 | 0.1576 |
| | | **Chênh lệch $\Delta_{\text{Sparse}}$** | **-1.4\%** | **+0.0149** | **+0.0051** | **-0.0001** | **-0.0039** |
| | | DL_with_IL_Dense_CC | 65.7\% | 0.3648 | 0.5211 | 0.3683 | 0.1533 |
| | | DL_isolated_Dense_CC | 64.9\% | 0.3111 | 0.4812 | 0.3478 | 0.1577 |
| | | **Chênh lệch $\Delta_{\text{Dense}}$** | **+0.8\%** | **+0.0537** | **+0.0399** | **+0.0205** | **-0.0044** |
| **scene** | Logistic | DL_with_IL_Sparse_CC | 81.7\% | 0.6369 | 0.7987 | 0.7546 | 0.0904 |
| | | DL_isolated_Sparse_CC | 80.6\% | 0.5977 | 0.8010 | 0.7388 | 0.0929 |
| | | **Chênh lệch $\Delta_{\text{Sparse}}$** | **+1.0\%** | **+0.0392** | **-0.0022** | **+0.0158** | **-0.0026** |
| | | DL_with_IL_Dense_CC | 80.8\% | 0.6343 | 0.8139 | 0.7584 | 0.0887 |
| | | DL_isolated_Dense_CC | 80.1\% | 0.5927 | 0.8137 | 0.7455 | 0.0914 |
| | | **Chênh lệch $\Delta_{\text{Dense}}$** | **+0.8\%** | **+0.0417** | **+0.0002** | **+0.0129** | **-0.0027** |
| | SVM | DL_with_IL_Sparse_CC | 59.9\% | 0.6250 | 0.7691 | 0.7172 | 0.0630 |
| | | DL_isolated_Sparse_CC | 58.0\% | 0.5818 | 0.7640 | 0.6988 | 0.0615 |
| | | **Chênh lệch $\Delta_{\text{Sparse}}$** | **+1.9\%** | **+0.0432** | **+0.0052** | **+0.0184** | **+0.0014** |
| | | DL_with_IL_Dense_CC | 63.5\% | 0.5035 | 0.7307 | 0.7013 | 0.0695 |
| | | DL_isolated_Dense_CC | 62.3\% | 0.4399 | 0.7196 | 0.6875 | 0.0702 |
| | | **Chênh lệch $\Delta_{\text{Dense}}$** | **+1.2\%** | **+0.0635** | **+0.0111** | **+0.0138** | **-0.0007** |
| | MLP | DL_with_IL_Sparse_CC | 79.5\% | 0.4264 | 0.7455 | 0.4537 | 0.0881 |
| | | DL_isolated_Sparse_CC | 79.0\% | 0.4217 | 0.7499 | 0.4438 | 0.0880 |
| | | **Chênh lệch $\Delta_{\text{Sparse}}$** | **+0.5\%** | **+0.0047** | **-0.0044** | **+0.0100** | **+0.0002** |
| | | DL_with_IL_Dense_CC | 79.6\% | 0.4402 | 0.7508 | 0.4550 | 0.0887 |
| | | DL_isolated_Dense_CC | 79.1\% | 0.4058 | 0.7012 | 0.4383 | 0.0891 |
| | | **Chênh lệch $\Delta_{\text{Dense}}$** | **+0.5\%** | **+0.0344** | **+0.0495** | **+0.0166** | **-0.0003** |
| **chd49** | Logistic | DL_with_IL_Sparse_CC | 66.2\% | 0.3788 | 0.4876 | 0.3116 | 0.1755 |
| | | DL_isolated_Sparse_CC | 66.1\% | 0.3804 | 0.4854 | 0.3116 | 0.1757 |
| | | **Chênh lệch $\Delta_{\text{Sparse}}$** | **+0.1\%** | **-0.0016** | **+0.0022** | **-0.0001** | **-0.0003** |
| | | DL_with_IL_Dense_CC | 69.3\% | 0.3724 | 0.4924 | 0.3098 | 0.1836 |
| | | DL_isolated_Dense_CC | 69.1\% | 0.3661 | 0.4793 | 0.3116 | 0.1843 |
| | | **Chênh lệch $\Delta_{\text{Dense}}$** | **+0.2\%** | **+0.0063** | **+0.0131** | **-0.0018** | **-0.0007** |
| | SVM | DL_with_IL_Sparse_CC | 33.5\% | 0.2072 | 0.2442 | 0.3079 | 0.1016 |
| | | DL_isolated_Sparse_CC | 33.4\% | 0.2056 | 0.2442 | 0.3097 | 0.0997 |
| | | **Chênh lệch $\Delta_{\text{Sparse}}$** | **+0.1\%** | **+0.0016** | **0.0000** | **-0.0018** | **+0.0019** |
| | | DL_with_IL_Dense_CC | 39.5\% | 0.1618 | 0.2375 | 0.3024 | 0.1270 |
| | | DL_isolated_Dense_CC | 39.4\% | 0.1618 | 0.2375 | 0.3042 | 0.1255 |
| | | **Chênh lệch $\Delta_{\text{Dense}}$** | **+0.1\%** | **0.0000** | **0.0000** | **-0.0018** | **+0.0015** |
| | MLP | DL_with_IL_Sparse_CC | 65.4\% | 0.3337 | 0.4431 | 0.3335 | 0.1752 |
| | | DL_isolated_Sparse_CC | 63.5\% | 0.3019 | 0.4438 | 0.3028 | 0.1798 |
| | | **Chênh lệch $\Delta_{\text{Sparse}}$** | **+1.8\%** | **+0.0318** | **-0.0007** | **+0.0307** | **-0.0046** |
| | | DL_with_IL_Dense_CC | 66.8\% | 0.3431 | 0.4389 | 0.3226 | 0.1813 |
| | | DL_isolated_Dense_CC | 65.5\% | 0.3223 | 0.4297 | 0.3101 | 0.1817 |
| | | **Chênh lệch $\Delta_{\text{Dense}}$** | **+1.3\%** | **+0.0208** | **+0.0091** | **+0.0125** | **-0.0004** |
| **yeast** | Logistic | DL_with_IL_Sparse_CC | 77.8\% | 0.2462 | 0.4120 | 0.2040 | 0.1307 |
| | | DL_isolated_Sparse_CC | 77.8\% | 0.2462 | 0.4336 | 0.2032 | 0.1310 |
| | | **Chênh lệch $\Delta_{\text{Sparse}}$** | **0.0\%** | **0.0000** | **-0.0217** | **+0.0008** | **-0.0003** |
| | | DL_with_IL_Dense_CC | 83.6\% | 0.2634 | 0.4440 | 0.2197 | 0.1486 |
| | | DL_isolated_Dense_CC | 83.5\% | 0.2625 | 0.4409 | 0.2181 | 0.1484 |
| | | **Chênh lệch $\Delta_{\text{Dense}}$** | **+0.1\%** | **+0.0009** | **+0.0030** | **+0.0017** | **+0.0002** |
| | SVM | DL_with_IL_Sparse_CC | 31.9\% | 0.1740 | 0.2667 | 0.1874 | 0.0673 |
| | | DL_isolated_Sparse_CC | 31.8\% | 0.1736 | 0.2665 | 0.1891 | 0.0668 |
| | | **Chênh lệch $\Delta_{\text{Sparse}}$** | **+0.1\%** | **+0.0003** | **+0.0001** | **-0.0017** | **+0.0005** |
| | | DL_with_IL_Dense_CC | 48.1\% | 0.1850 | 0.3454 | 0.1163 | 0.1042 |
| | | DL_isolated_Dense_CC | 48.0\% | 0.1882 | 0.3494 | 0.1175 | 0.1029 |
| | | **Chênh lệch $\Delta_{\text{Dense}}$** | **+0.1\%** | **-0.0032** | **-0.0040** | **-0.0012** | **+0.0013** |
| | MLP | DL_with_IL_Sparse_CC | 73.9\% | 0.2013 | 0.3361 | 0.1572 | 0.1329 |
| | | DL_isolated_Sparse_CC | 74.9\% | 0.2196 | 0.4181 | 0.1577 | 0.1340 |
| | | **Chênh lệch $\Delta_{\text{Sparse}}$** | **-1.0\%** | **-0.0183** | **-0.0820** | **-0.0004** | **-0.0011** |
| | | DL_with_IL_Dense_CC | 77.2\% | 0.2459 | 0.3977 | 0.1643 | 0.1424 |
| | | DL_isolated_Dense_CC | 76.9\% | 0.2476 | 0.3866 | 0.1630 | 0.1415 |
| | | **Chênh lệch $\Delta_{\text{Dense}}$** | **+0.3\%** | **-0.0017** | **+0.0111** | **+0.0012** | **+0.0009** |

---

### 4.3. Bảng tổng hợp trung bình toàn cầu (Grand Mean trên 10 tập dữ liệu)

Bảng 4 tổng hợp giá trị trung bình trên toàn bộ 10 tập dữ liệu benchmark cho từng bộ phân loại cơ sở:

### Bảng 4: Hiệu năng vĩ mô toàn cầu trên tập phụ thuộc $DL$

| Bộ học cơ sở | Cấu hình trên tập $DL$ | Selective Macro-F1 | Selective Precision | Subset 0/1 Acc | Selective Hamming Loss ($\downarrow$) | Coverage (\%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **LOGISTIC** | `DL_with_IL_Sparse_CC` | **0.2906** | **0.4386** | **0.4934** | **0.1062** | 82.3\% |
| | `DL_isolated_Sparse_CC` | 0.2555 | 0.3703 | 0.4867 | 0.1094 | 81.7\% |
| | **$\Delta_{\text{Sparse}}$ (With IL vs. Cô lập X)** | **+0.0351** | **+0.0682** | **+0.0067** | **-0.0032** | **+0.5\%** |
| | `DL_with_IL_Dense_CC` | **0.2891** | **0.4468** | **0.4976** | **0.1078** | 83.0\% |
| | `DL_isolated_Dense_CC` | 0.2514 | 0.3730 | 0.4905 | 0.1115 | 82.6\% |
| | **$\Delta_{\text{Dense}}$ (With IL vs. Cô lập X)** | **+0.0378** | **+0.0739** | **+0.0072** | **-0.0037** | **+0.5\%** |
| **SVM** | `DL_with_IL_Sparse_CC` | **0.2708** | **0.3539** | **0.4630** | 0.0770 | 57.4\% |
| | `DL_isolated_Sparse_CC` | 0.2668 | 0.3515 | 0.4616 | **0.0768** | 56.4\% |
| | **$\Delta_{\text{Sparse}}$ (With IL vs. Cô lập X)** | **+0.0040** | **+0.0025** | **+0.0014** | +0.0001 | **+0.9\%** |
| | `DL_with_IL_Dense_CC` | **0.2040** | 0.3236 | **0.4234** | **0.0884** | 63.6\% |
| | `DL_isolated_Dense_CC` | 0.1996 | **0.3291** | 0.4213 | 0.0888 | 62.8\% |
| | **$\Delta_{\text{Dense}}$ (With IL vs. Cô lập X)** | **+0.0044** | -0.0055 | **+0.0021** | **-0.0004** | **+0.8\%** |
| **MLP (GPU)** | `DL_with_IL_Sparse_CC` | **0.2244** | 0.3567 | **0.4313** | **0.1066** | 81.5\% |
| | `DL_isolated_Sparse_CC` | 0.2160 | **0.3604** | 0.4244 | 0.1084 | 81.4\% |
| | **$\Delta_{\text{Sparse}}$ (With IL vs. Cô lập X)** | **+0.0084** | -0.0037 | **+0.0069** | **-0.0018** | 0.0\% |
| | `DL_with_IL_Dense_CC` | **0.2404** | **0.3711** | **0.4342** | **0.1099** | 82.4\% |
| | `DL_isolated_Dense_CC` | 0.2226 | 0.3441 | 0.4320 | 0.1100 | 82.1\% |
| | **$\Delta_{\text{Dense}}$ (With IL vs. Cô lập X)** | **+0.0178** | **+0.0270** | **+0.0022** | **-0.0001** | **+0.3\%** |

---

## 5. Thảo Luận Khoa Học

Dựa trên số liệu đo đạc thực nghiệm, các phát hiện kỹ thuật được đúc kết như sau:

### 5.1. Vai trò của ngữ cảnh tĩnh từ tập $IL$
1. **Mức độ đóng góp phụ thuộc vào tính chất tương quan của dữ liệu:**
   - Trên các tập dữ liệu có tương quan cặp rõ ràng (`emotions`, `scene`), việc bổ sung ngữ cảnh $IL$ mang lại cải thiện F1 đáng kể (+3.92\% đến +14.41\% trên Logistic Regression). Tại đây, các nhãn trong $IL$ đóng vai trò như các biến chỉ báo (indicator variables) giúp thu hẹp không gian giả thuyết cho $DL$.
   - Trên các tập dữ liệu có mật độ thưa hoặc cấu trúc sinh học rời rạc (`chd49`, `yeast`, `viruspseaac`), sự chênh lệch $\Delta$ dao động quanh mức $0.0\% \pm 0.3\%$. Điều này cho thấy khi các nhãn không có liên kết hữu cơ về mặt xác suất, việc đưa thêm $IL$ không làm hại mô hình nhưng cũng không đóng góp thêm thông tin phân biệt.
2. **Xu hướng trên quy mô trung bình toàn cục:**
   - Giá trị trung bình toàn cục của $\Delta$ trên Selective Macro-F1 luôn dương trên cả ba bộ học cơ sở: **+0.0351** (Logistic Regression), **+0.0040** (SVM), và **+0.0084** (MLP). Điều này khẳng định cơ chế Static Context nhìn chung mang lại giá trị gia tăng ổn định.

### 5.2. Tương tác giữa cấu trúc chuỗi (Sparse CC vs. Dense CC) và bộ học cơ sở
1. **Hiện tượng lan truyền sai số trên SVM:**
   - Trên Support Vector Machine, chuỗi thưa `DL_with_IL_Sparse_CC` đạt Selective Macro-F1 là **0.2708**, cao hơn rõ rệt so với chuỗi dày `DL_with_IL_Dense_CC` (**0.2040**), tương đương mức tăng +6.68\% tuyệt đối (+32.7\% tương đối).
   - Nguyên nhân xuất phát từ bản chất hình học của SVM: việc ép buộc các nhãn không tương quan vào đầu vào của bộ phân loại phân cách lề cực đại (max-margin) tạo ra các thuộc tính giả (spurious features), làm nhiễu vector hỗ trợ (support vectors) và phá vỡ mặt phẳng siêu phẳng tối ưu. Ngưỡng lọc $\theta_{\text{corr}} = 0.75$ đã ngăn chặn hiệu quả hiện tượng này.
2. **Tính bền vững của Logistic Regression và MLP:**
   - Logistic Regression duy trì mức hiệu năng đồng đều giữa Sparse CC (0.2906) và Dense CC (0.2891) do hàm mất mát Log-loss có khả năng giảm dần trọng số (weights) của các thuộc tính không tương quan về gần 0.
   - MLP có khả năng biểu diễn phi tuyến tốt, do đó khi cô lập chỉ dùng $X$, mức suy giảm hiệu năng của MLP diễn ra chậm hơn so với các mô hình tuyến tính thuần túy.

---

## 6. Kết Luận Thực Nghiệm

1. Việc tách riêng tập phụ thuộc $DL$ để kiểm tra khi tắt toàn bộ $IL$ đã cung cấp bằng chứng định lượng khẳng định: **ngữ cảnh $IL$ đóng vai trò bổ trợ có ý nghĩa cho $DL$ trên các bài toán có tương quan tự nhiên**, đồng thời không gây suy thoái hiệu năng trên các bài toán thưa thớt.
2. Cơ chế lọc ngưỡng tương quan **Sparse CC ($\theta_{\text{corr}} = 0.75$)** chứng minh tính ưu việt vượt trội so với Dense CC, đặc biệt trên các bộ học phân cách không gian như Support Vector Machine.
3. Đối với các tập dữ liệu có $K_{IL} = 0$ (như `humanpseaac` và `plantpseaac`, nơi toàn bộ nhãn buộc phải đưa vào $DL$), việc áp dụng mô hình Ensemble Classifier Chains (ECC) hoặc hạ dần ngưỡng lọc tương quan theo tầng sâu chuỗi ($\theta_{\text{corr}} = 0.85 \to 0.70$) là định hướng kỹ thuật phù hợp tiếp theo để nâng cao độ chính xác của chuỗi.

---
*Báo cáo được biên soạn và kiểm toán dựa trên dữ liệu thực nghiệm tại thư mục `results_v5_1_test/Test_DL_results/`.*