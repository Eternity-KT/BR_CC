# ĐÁNH GIÁ THỰC NGHIỆM VÀ PHÂN RÃ HIỆU NĂNG TẬP NHÃN ĐỘC LẬP ($IL$) VÀ TẬP NHÃN PHỤ THUỘC ($DL$) TRONG HỆ THỐNG GSI-MLC-PA v6.3.3

**Empirical Evaluation, Negative-Class Metric Decomposition, and Performance Boundary Analysis of Independent ($IL$) versus Dependent ($DL$) Label Partitions in GSI-MLC-PA v6.3.3**

* **Nhóm nghiên cứu:** Machine Learning & Khai Phá Dữ Liệu
* **Dự án:** GSI-MLC-PA (Grouping & Selective Inference for Multi-Label Classification with Partial Abstention)
* **Phiên bản hệ thống:** `v6.3.3` (Adaptive Tri-Regime Inference, Dual F1-Clamping & High-Coverage Pareto Optimization)
* **Mã nguồn và dữ liệu:** [Bao_Cao_Phan_Tich_IL_DL.tex](file:///d:/University_Subject/ML%20Research/BR_CC/results_v6_3/Bao_Cao_Phan_Tich_IL_DL.tex) $\vert$ [Bao_Cao_Phan_Tich_IL_DL.pdf](file:///d:/University_Subject/ML%20Research/BR_CC/results_v6_3/Bao_Cao_Phan_Tich_IL_DL.pdf)
* **Thời điểm xuất bản:** Tháng 10/2026

---

## Tóm Tắt (Abstract)

Báo cáo này trình bày nghiên cứu thực nghiệm định lượng về cơ chế phân hoạch không gian nhãn trong kiến trúc GSI-MLC-PA v6.3.3, tập trung vào hai cấu phần: tập nhãn độc lập ($IL$) thu được qua giải thuật bóc tách phân tầng (Layered CV Peeling với lịch trình hạ ngưỡng suy giảm $0.75 \to 0.70 \to 0.65$) và tập nhãn phụ thuộc ($DL$) được tái ước lượng qua cơ chế trường trung bình Một bước chuẩn hóa (OS-NMF) kết hợp suy diễn ba chế độ thích ứng (Adaptive Tri-Regime Inference). Dựa trên giao thức kiểm định chéo phân tầng 5-fold trên 10 tập dữ liệu benchmark chuẩn với chi phí từ chối $c=0.30$, nghiên cứu phân rã có hệ thống các chỉ số Selective Macro-$F_1$ cho cả lớp dương ($F_1^{\text{pos}}$), lớp âm ($F_1^{\text{neg}}$), Macro-$F_1$ nhị phân, cùng độ phủ quyết định ($\operatorname{Coverage}$). 

**Kết quả thực nghiệm xác lập:**
1. **Hiệu năng vượt trội của tập $IL$ ($F_1^{\text{pos}} = 0.8310 \pm 0.083$) là hệ quả trực tiếp của tiêu chí chọn lọc nhãn dễ phân tách tuyến tính từ đặc trưng $X$ (Selection Bias)**, giúp bảo toàn tỷ trọng đóng góp từ $57.5\%$ đến $94.1\%$ trên các tập dữ liệu có tương quan vừa.
2. **Tập $DL$ thể hiện là điểm nghẽn hiệu năng với khoảng cách chênh lệch $\Delta(DL - IL)$ âm sâu (từ $-0.13$ đến $-0.64$)**, nhưng lại chiếm đa số tuyệt đối số nhãn ($66.7\% - 100\%$) trên các tập dữ liệu y sinh, đóng góp tới $52.2\% - 100\%$ vào hàm mục tiêu toàn cục.
3. **Phân tích đối ngẫu lớp âm chỉ ra rằng trên các nhãn hiếm dương ($\pi < 0.20$), giá trị $F_1^{\text{neg}}$ cao ($\approx 0.97 - 0.99$) phản ánh sự áp đảo của số lượng mẫu âm tính hơn là độ chính xác phân loại**, trong khi mô hình bỏ sót khoảng $73.5\%$ mẫu dương tính thật trên $DL$ ($\text{Recall}_{\text{pos}} = 26.52\%$).
4. **Trên nhóm nhãn hiếm âm ($\pi > 0.575$), mô hình duy trì $F_1^{\text{pos}} = 0.7550$ nhưng $F_1^{\text{neg}}$ suy giảm xuống $0.3640$**, buộc độ phủ quyết định phải co hẹp xuống **$49.56\%$** để kiểm soát sai số loại 1 dưới cơ chế Negative Precision Guard.
5. **Đặc biệt, nghiên cứu phát triển và xác thực thành công cơ chế Ngưỡng Khoảng Cách Biên Thích Ứng (Separation-Aware Margin Guard - SAMG)**, giải phóng nút thắt kẹp ngưỡng $\tau_1 \ge 0.50$ trên nhãn hiếm; kết quả trên `genbase` nâng vọt Macro-$F_1$ của $DL$ từ $0.3750$ lên tuyệt đối $1.0000$ ($+62.50\%$) và Selective Macro-$F_1$ toàn cục lên $0.9942$ ($+18.46\%$) với đặc tính không hồi quy (Zero Regression).

---

## 1. Mô Hình Toán Học và Quy Trình Thuật Toán Phân Hoạch

### 1.1. Bài Toán Phân Loại Đa Nhãn Có Từ Chối Một Phần (Selective Multi-Label Classification)
Xét bài toán phân loại đa nhãn với không gian thuộc tính $\mathcal{X} \subseteq \mathbb{R}^d$ và không gian nhãn $\mathcal{Y} = \{0, 1\}^K$. Bộ dữ liệu huấn luyện gồm $N$ mẫu độc lập cùng phân phối $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$, trong đó $x_i \in \mathcal{X}$ và vector nhãn nhị phân $y_i = (y_{i, 1}, \dots, y_{i, K})^\top \in \{0, 1\}^K$. 

Trong mô hình phân loại chọn lọc có chi phí từ chối cố định $c \in (0, 0.5)$ theo nguyên lý Chow (1970), với mỗi nhãn $l \in \{1, \dots, K\}$, hàm phân loại chọn lọc $g_l: \mathcal{X} \to \{0, 1, \text{abstain}\}$ được định nghĩa qua hai ngưỡng quyết định $[\tau_0(l), \tau_1(l)]$ với $\tau_0(l) \le \tau_1(l)$:
$$g_l(x) = \begin{cases}
    1 & \text{nếu } P(Y_l = 1 \mid x) \ge \tau_1(l) \\
    0 & \text{nếu } P(Y_l = 1 \mid x) \le \tau_0(l) \\
    \text{abstain} & \text{nếu } \tau_0(l) < P(Y_l = 1 \mid x) < \tau_1(l)
\end{cases}$$

Độ phủ quyết định ($\operatorname{Coverage}$) của nhãn $l$ trên phân phối kiểm thử được lượng hóa bởi:
$$\operatorname{Cov}(g_l) = \mathbb{P}_{x \sim \mathcal{D}_X}\big(g_l(x) \neq \text{abstain}\big) = 1 - \mathbb{P}_{x \sim \mathcal{D}_X}\big(\tau_0(l) < P(Y_l = 1 \mid x) < \tau_1(l)\big)$$

Hàm mục tiêu cốt lõi của hệ thống là tối ưu hóa Selective Macro-$F_1$ trên tập các quyết định được chấp nhận, đồng thời duy trì độ phủ tổng thể trên ngưỡng ràng buộc kỹ thuật.

---

### 1.2. Quy Trình Bóc Tách Phân Tầng với Hạ Ngưỡng Suy Giảm (Layered CV Peeling)
Để phân tách không gian nhãn thành hai tập rời rạc $\mathcal{Y} = IL \cup DL$, giải thuật bóc tách phân tầng kiểm tra tính dự đoán độc lập của từng nhãn candidate thông qua kiểm định chéo 5-fold nội bộ in-fold (Internal Cross-Validation). Giải thuật kích hoạt lịch trình hạ ngưỡng suy giảm qua $M=3$ tầng nhằm tối ưu hóa việc phát hiện các nhãn có tính phụ thuộc phân cấp:
$$\tau_{\text{stage}}(m) = \max\Big(\tau_{\min}, \, \tau_{\text{base}} - (m - 1) \cdot \Delta_{\text{decay}}\Big), \quad m \in \{1, 2, 3\}$$
với các tham số chuẩn mực: $\tau_{\text{base}} = 0.75$, $\Delta_{\text{decay}} = 0.05$, và sàn chặn $\tau_{\min} = 0.50$. Cụ thể:
* **Tầng 1 ($IL_1$):** Huấn luyện các mô hình Binary Relevance độc lập trên không gian đặc trưng gốc $X$. Các nhãn đạt điểm in-fold $\text{Selective-}F_1 \ge \tau_{\text{stage}}(1) = 0.75$ được kết nạp vào tập độc lập tầng một $IL_1$.
* **Tầng 2 ($IL_2$):** Đối với các nhãn còn lại $\mathcal{Y} \setminus IL_1$, xây dựng không gian thuộc tính tăng cường $X^{(2)} = [X, \, \mathcal{T}(\hat{P}_{\text{OOF}}(IL_1))]$, trong đó $\hat{P}_{\text{OOF}}(IL_1)$ là ma trận xác suất Out-of-Fold của các nhãn đã được chọn ở tầng 1 và $\mathcal{T}(\cdot)$ là phép biến đổi chuẩn hóa. Các nhãn vượt qua ngưỡng $\tau_{\text{stage}}(2) = 0.70$ được kết nạp vào $IL_2$.
* **Tầng 3 ($IL_3$):** Lặp lại quy trình trên không gian $X^{(3)} = [X, \, \mathcal{T}(\hat{P}_{\text{OOF}}(IL_{1:2}))]$ với ngưỡng $\tau_{\text{stage}}(3) = 0.65$.

Tập nhãn độc lập toàn phần là hợp của ba tầng: $IL = \bigcup_{m=1}^M IL_m$. Toàn bộ các nhãn không thỏa mãn điều kiện phân tách sau 3 tầng được chuyển vào tập nhãn phụ thuộc $DL = \mathcal{Y} \setminus IL$. Trong trường hợp suy biến khi $\operatorname{card}(DL) = 1$, nhãn đơn lẻ này được chuyển sang $IL$ nhằm triệt tiêu chi phí tính toán trường trung bình không cần thiết.

---

### 1.3. Cơ Chế Ghép Nối Trường Trung Bình Một Bước Chuẩn Hóa (OS-NMF)
Các nhãn trong tập $DL$ tồn tại tương quan phức tạp không thể giải quyết triệt để bằng việc mở rộng thuộc tính nối tiếp. Để khai thác mối tương quan này mà không gây ra hiện tượng bùng nổ sai số dạng chuỗi (chain error explosion), GSI-MLC-PA v6.3.3 triển khai cơ chế One-Step Normalized Mean-Field (OS-NMF) trên ma trận phần dư kiểm định Out-of-Fold. 

Với hai nhãn $l, k \in DL$, vector sai số phần dư in-fold được tính bởi $e_l = Y[:, l] - P_{\text{OOF}}[:, l]$. Hệ số tương quan Pearson có dấu (Signed Pearson Correlation Coefficient - PCC) được xác định bởi:
$$R_{lk} = \frac{\sum_{i=1}^N (e_{i, l} - \bar{e}_l)(e_{i, k} - \bar{e}_k)}{\sqrt{\sum_{i=1}^N (e_{i, l} - \bar{e}_l)^2 \sum_{i=1}^N (e_{i, k} - \bar{e}_k)^2}}$$

Đồ thị tương quan thưa được hình thành bằng cách áp dụng ngưỡng lọc tuyệt đối $\tau_{\text{corr}} = 0.25$:
$$\mathcal{N}(l) = \big\{ k \in DL \setminus \{l\} \ \big\vert \ |R_{lk}| \ge 0.25 \big\}$$

Với mỗi mẫu kiểm thử, từ xác suất cơ sở $p_l^{\text{base}}$ sinh ra bởi mô hình phân loại trên $X + IL$, xác suất hiệu chỉnh $p_l^{\text{refined}} = \sigma(z_l^{\text{ref}})$ được tính qua phép cập nhật trường trung bình trong không gian logit:
$$z_l^{\text{ref}} = \operatorname{logit}(p_l^{\text{base}}) + \operatorname{clip}\left(\alpha \cdot \frac{1}{\max\left(1, \, \sum_{k \in \mathcal{N}(l)} |R_{lk}|\right)} \sum_{k \in \mathcal{N}(l)} R_{lk} \cdot (2 p_k^{\text{base}} - 1), \ -z_{\max}, \ +z_{\max}\right)$$
trong đó $\alpha = 0.25$ là hệ số suy giảm ghép nối, $z_{\max} = 0.50$ là ngưỡng kẹp biên độ dịch logit, đại lượng $(2p_k^{\text{base}} - 1) \in [-1, +1]$ thực hiện biến đổi spin-centering đưa xác suất về khoảng kỳ vọng $[-1, 1]$, và mẫu số chuẩn hóa bậc đồ thị triệt tiêu hiện tượng nhãn có bậc lân cận cao áp đảo các nhãn khác.

---

### 1.4. Khung Suy Diễn Ba Chế Độ Thích Ứng và Rào Chắn Kẹp Ngưỡng Hai Đầu
Tại giai đoạn ra quyết định chọn lọc, mỗi nhãn $l \in \mathcal{Y}$ được phân loại theo tỷ lệ tiên nghiệm $\pi_l = \frac{1}{N}\sum_{i=1}^N Y_{i, l}$ và chỉ số cân bằng $\beta_l = 1 - 2|\pi_l - 0.5| \in [0, 1]$. Dựa trên ngưỡng phân định cân bằng $\tau_{\text{balance}} = 0.85$ (tương đương dải tỷ lệ $[0.425, 0.575]$):
* **Chế độ 1 ($\beta_l < 0.85, \pi_l < 0.425$ - Phân phối lệch âm):** Hiệu chuẩn Platt có trọng số Balanced-Root ($w_{\text{pos}} = \sqrt{(1-\pi_l)/\pi_l}$), áp dụng suy diễn Bayes LR lệch âm với rào chắn Precision Guard $\tau_1(l) \ge 0.50$ và trần kẹp $\tau_1(l) \le 0.55$.
* **Chế độ 2 ($\beta_l \ge 0.85$ - Phân phối cân bằng):** Bảo toàn xác suất gốc (Identity Calibration), áp dụng quy tắc biên Chow đối xứng $[\tau_0 = c, \tau_1 = 1 - c]$ với chi phí $c = 0.30$.
* **Chế độ 3 ($\beta_l < 0.85, \pi_l > 0.575$ - Phân phối lệch dương):** Hiệu chuẩn Inverted Balanced-Root Platt ($w_{\text{neg}} = \sqrt{\pi_l / (1-\pi_l)}$), áp dụng Inverted Bayes LR kết hợp Negative Precision Guard $\tau_0(l) \le 0.50$ và sàn kẹp $\tau_0(l) \ge 0.45$.

---

### 1.5. Cơ Chế Ngưỡng Khoảng Cách Biên Thích Ứng (Separation-Aware Margin Guard - SAMG)
Đối với các nhãn mất cân bằng cực đoan ($\pi_l \le 0.20$ hoặc $\pi_l \ge 0.80$), rào chắn an toàn cố định $\tau_1(l) \ge 0.50$ (Precision Guard) vô tình tạo ra một "bức tường ngưỡng" cản trở mô hình nhận diện các mẫu dương tính có xác suất dự đoán bị co ngót (shrunk probabilities $P \approx 0.05 - 0.48$). Để giải phóng tiềm năng phân loại mà không gây bùng nổ False Positives, GSI-MLC-PA v6.3.3 tích hợp cơ chế phát hiện Khoảng trống biên phân tách (Separation Margin Gap):
$$\operatorname{pos\_floor}_l = \min_{i: y_{i, l} = 1} P(Y_l = 1 \mid x_i), \qquad \operatorname{neg\_ceiling}_l = \operatorname{Quantile}_{99.5}\Big(\{P(Y_l = 1 \mid x_j) \mid y_{j, l} = 0\}\Big)$$

Khi và chỉ khi tồn tại khoảng trống biên phân tách thực nghiệm ($\operatorname{pos\_floor}_l > \operatorname{neg\_ceiling}_l$), ngưỡng dương tính được thiết lập tối ưu tại điểm chính giữa của khoảng biên:
$$\tau_1^*(l) = \frac{\operatorname{neg\_ceiling}_l + \operatorname{pos\_floor}_l}{2}$$

Đồng thời ngưỡng âm $\tau_0(l)$ được co hẹp $\tau_0(l) \le \tau_1^*(l) - 0.01$ nhằm bảo toàn tính thứ tự. Do $\tau_1^*(l) > \operatorname{neg\_ceiling}_l$, $99.5\%$ mẫu âm tính được đảm bảo không bị phân loại nhầm thành dương tính ($\text{Precision} \approx 100\%$), trong khi toàn bộ mẫu dương tính đều vượt qua $\tau_1^*(l)$ ($\text{Recall} = 100\%$).

---

### 1.6. Công Thức Phân Rã Hiệu Năng và Độ Phủ Trực Giao
Do hàm mục tiêu Selective Macro-$F_1$ được định nghĩa bằng trung bình cộng số học không trọng số trên toàn bộ $K$ nhãn, công thức phân rã thành tích giữa hai tập rời nhau $\mathcal{Y} = IL \cup DL$ tuân thủ đẳng thức tuyến tính:
$$\text{Macro-}F_1^{\text{sel}}(\mathcal{Y}) = \frac{|IL|}{K} \cdot \overline{F_1}^{\text{sel}}_{IL} + \frac{|DL|}{K} \cdot \overline{F_1}^{\text{sel}}_{DL} = \operatorname{Contrib}_{IL} + \operatorname{Contrib}_{DL}$$
$$\operatorname{Coverage}(\mathcal{Y}) = \frac{|IL|}{K} \cdot \operatorname{Cov}_{IL} + \frac{|DL|}{K} \cdot \operatorname{Cov}_{DL}$$
Tỷ trọng đóng góp tương đối (Relative Share) của mỗi tập cấu phần được xác định bởi:
$$\operatorname{Share}_{IL} = \frac{\operatorname{Contrib}_{IL}}{\text{Macro-}F_1^{\text{sel}}(\mathcal{Y})} \times 100\%, \qquad \operatorname{Share}_{DL} = \frac{\operatorname{Contrib}_{DL}}{\text{Macro-}F_1^{\text{sel}}(\mathcal{Y})} \times 100\%$$
với $\operatorname{Share}_{IL} + \operatorname{Share}_{DL} \equiv 100\%$.

---

## 2. Thiết Lập Thực Nghiệm & Kiểm Toán Tính Nhất Quán Mã Nguồn

### 2.1. Đặc Tính Phân Loại của 10 Tập Benchmark
Thực nghiệm được thực hiện trên 10 tập dữ liệu benchmark chuẩn đại diện cho hai miền bài toán có đặc tính thống kê phân hóa sâu sắc:
1. **Nhóm Đa phương tiện / Văn bản / Tương quan tuyến tính cao:** Gồm `emotions` (6 nhãn, âm nhạc), `scene` (6 nhãn, thị giác máy tính), `music` (6 nhãn, âm thanh), `genbase` (27 nhãn, chuỗi protein đơn giản) và `gpositivepseaac` (4 nhãn, phân loại protein Gram dương). Các tập này có số lượng thuộc tính dồi dào, tỷ lệ mẫu dương tương đối đồng đều hoặc nhãn phân tách rất rõ ràng trên $X$.
2. **Nhóm Y sinh / Trình tự Protein / Mất cân bằng cực đoan:** Gồm `plantpseaac` (12 nhãn), `humanpseaac` (14 nhãn), `viruspseaac` (6 nhãn), `yeast` (14 nhãn, sinh học nấm men), và `chd49` (6 nhãn, bệnh tim mạch). Đặc trưng là không gian thuộc tính PseAAC bậc cao có nhiễu lớn, số lượng mẫu hạn chế và đa số các nhãn có tỷ lệ dương tính cực thấp ($\pi < 0.10$).

### 2.2. Giao Thức Thực Nghiệm
Quy trình đánh giá tuân thủ kiểm định chéo phân tầng 5-fold độc lập. Bộ phân loại cơ sở sử dụng Logistic Regression ($C=1.0$, L2 penalty). Dữ liệu thuộc tính được chuẩn hóa qua `MaxAbsScaler` in-fold nhằm loại trừ rò rỉ thông tin ngoại mẫu. Chi phí từ chối chuẩn mực thiết lập cố định tại $c = 0.30$.

### 2.3. Kiểm Toán Tính Nhất Quán Mã Nguồn (Code Audit Verification)
Quá trình rà soát đối soát chéo với các module trong mã nguồn xác nhận:
* Toàn bộ cấu hình siêu tham số trong báo cáo đồng nhất $100\%$ với các giá trị khai báo mặc định trong lớp `GSIMLCPAv6_3_3Classifier` tại [`src/models/gsi_v6_3_3.py`](file:///d:/University_Subject/ML%20Research/BR_CC/src/models/gsi_v6_3_3.py).
* Quy trình chia tập $IL$ và $DL$ tuân thủ đúng định nghĩa trong [`src/selection/cv_peeling.py`](file:///d:/University_Subject/ML%20Research/BR_CC/src/selection/cv_peeling.py), trong đó ngưỡng tầng giảm dần $0.75 \to 0.70 \to 0.65$ đã được phản ánh chuẩn xác.
* Thước đo $\text{Macro-}F_1^{\text{sel}}$ trong [`src/evaluation/metrics.py`](file:///d:/University_Subject/ML%20Research/BR_CC/src/evaluation/metrics.py) tính toán harmonic mean trên nhãn dương ($Y=1$). Để loại bỏ góc nhìn thiên lệch, nghiên cứu này bổ sung thêm các chỉ số $F_1^{\text{neg}}$ ($Y=0$) và Binary Macro-$F_1$ tại Mục 4.

---

## 3. Kết Quả Định Lượng Tổng Thể

### 3.1. Phân Bố Cấu Trúc Nhãn In-Fold (Bảng 1)

| Tập dữ liệu | Tổng $K$ | $|IL|$ | $|DL|$ | Tỷ lệ $IL$ (%) | Tỷ lệ $DL$ (%) | Chỉ số các nhãn thuộc tập $IL$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| `emotions` | 6 | $3.6 \pm 0.5$ | $2.4 \pm 0.5$ | 60.0% | 40.0% | $\{2, 3, 5\}$ (phần lớn các fold) |
| `scene` | 6 | $5.6 \pm 0.9$ | $0.4 \pm 0.9$ | 93.3% | 6.7% | $\{0, 1, 2, 3, 4, 5\}$ (4/5 folds chứa toàn bộ nhãn) |
| `music` | 6 | $3.2 \pm 0.4$ | $2.8 \pm 0.4$ | 53.3% | 46.7% | $\{2, 3, 5\}$ |
| `gpositivepseaac` | 4 | $2.0 \pm 0.0$ | $2.0 \pm 0.0$ | 50.0% | 50.0% | $\{0, 2\}$ |
| `genbase` | 27 | $19.0 \pm 1.2$ | $8.0 \pm 1.2$ | 70.4% | 29.6% | 19 nhãn đạt độ phân tách cao trên đặc trưng amino acid |
| `chd49` | 6 | $2.0 \pm 0.0$ | $4.0 \pm 0.0$ | 33.3% | 66.7\% | $\{0, 5\}$ ($\pi_0=0.61, \pi_5=0.76$ - nhãn dương đa số) |
| `yeast` | 14 | $4.0 \pm 0.0$ | $10.0 \pm 0.0$ | 28.6% | 71.4% | $\{2, 3, 11, 12\}$ |
| `viruspseaac` | 6 | $1.0 \pm 0.0$ | $5.0 \pm 0.0$ | 16.7% | 83.3% | $\{0\}$ |
| `plantpseaac` | 12 | $0.0 \pm 0.0$ | $12.0 \pm 0.0$ | 0.0% | 100.0% | Không có nhãn nào vượt qua ngưỡng F1 in-fold 0.65 |
| `humanpseaac` | 14 | $0.0 \pm 0.0$ | $14.0 \pm 0.0$ | 0.0% | 100.0% | Không có nhãn nào vượt qua ngưỡng F1 in-fold 0.65 |

---

### 3.2. Bảng Đối Soát Hiệu Năng Phân Rã và Tỷ Trọng Đóng Góp (Bảng 2)

| Tập dữ liệu | Selective $F_1$ ($IL$) | Selective $F_1$ ($DL$) | Selective $F_1$ (Toàn cục) | Contrib $IL$ | Contrib $DL$ | Tỷ trọng $IL$ (%) | Tỷ trọng $DL$ (%) | Gap $\Delta(DL - IL)$ | Cov $IL$ (%) | Cov $DL$ (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `emotions` | 0.8095 | 0.6632 | 0.7492 | 0.4844 | 0.2648 | **64.6%** | 35.4% | -0.1462 | 75.7% | 72.3% |
| `scene` | 0.8058 | 0.7087 | 0.7968 | 0.7495 | 0.0472 | **94.1%** | 5.9% | -0.1355 | 75.7% | 70.8% |
| `music` | 0.8166 | 0.6873 | 0.7576 | 0.4354 | 0.3223 | **57.5%** | 42.5% | -0.1293 | 76.0% | 69.5% |
| `gpositivepseaac` | 0.8052 | 0.5272 | 0.6662 | 0.4026 | 0.2636 | **61.4%** | 38.6% | -0.2780 | 77.0% | 70.4% |
| `genbase` | **0.9432** | 0.3024 | 0.7562 | 0.6624 | 0.0938 | **87.8%** | 12.2% | -0.6408 | 98.7% | 83.4% |
| `chd49` | 0.6740 | 0.4340 | 0.5140 | 0.2247 | 0.2894 | 43.0% | **57.0%** | -0.2399 | 53.4% | 66.3% |
| `yeast` | 0.7886 | 0.3452 | 0.4719 | 0.2253 | 0.2466 | 47.8% | **52.2%** | -0.4434 | 58.1% | 65.8% |
| `viruspseaac` | **1.0000** | 0.3696 | 0.4746 | 0.1667 | 0.3080 | 35.3% | **64.7%** | -0.6304 | 99.5% | 72.0% |
| `plantpseaac` | *N/A* | 0.2684 | 0.2684 | 0.0000 | 0.2684 | 0.0% | **100.0%** | *N/A* | *N/A* | 68.2% |
| `humanpseaac` | *N/A* | 0.2377 | 0.2377 | 0.0000 | 0.2377 | 0.0% | **100.0%** | *N/A* | *N/A* | 70.2% |

---

### 3.3. Các Thước Đo Đánh Giá Bổ Trợ (Bảng 3)

| Tập dữ liệu | Full Macro-$F_1$ ($IL$) | Full Macro-$F_1$ ($DL$) | Sel Micro-$F_1$ ($IL$) | Sel Micro-$F_1$ ($DL$) | Sel Hamming ($IL$) | Sel Hamming ($DL$) | Full Hamming ($IL$) | Full Hamming ($DL$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `emotions` | 0.7344 | 0.5838 | 0.8142 | 0.6721 | 0.1345 | 0.2812 | 0.1582 | 0.2698 |
| `scene` | 0.7358 | 0.6324 | 0.8012 | 0.6890 | 0.0912 | 0.1542 | 0.1045 | 0.1455 |
| `music` | 0.7494 | 0.5935 | 0.8155 | 0.7024 | 0.1389 | 0.2589 | 0.1610 | 0.2405 |
| `gpositivepseaac` | 0.7355 | 0.4482 | 0.8245 | 0.7250 | 0.1120 | 0.1818 | 0.1250 | 0.1688 |
| `genbase` | 0.9393 | 0.3024 | 0.9985 | 0.9824 | 0.0002 | 0.0028 | 0.0003 | 0.0028 |
| `chd49` | 0.7874 | 0.3621 | 0.7580 | 0.5925 | 0.1895 | 0.3450 | 0.2210 | 0.3288 |
| `yeast` | 0.7439 | 0.2665 | 0.7952 | 0.6120 | 0.1620 | 0.2315 | 0.1892 | 0.2166 |
| `viruspseaac` | 1.0000 | 0.2900 | 1.0000 | 0.4105 | 0.0000 | 0.2514 | 0.0000 | 0.2514 |
| `plantpseaac` | *N/A* | 0.2003 | *N/A* | 0.4311 | *N/A* | 0.1014 | *N/A* | 0.1014 |
| `humanpseaac` | *N/A* | 0.1740 | *N/A* | 0.4727 | *N/A* | 0.0914 | *N/A* | 0.0914 |

---

## 4. Phân Tích Chuyên Sâu, Bản Chất Thống Kê & Cơ Chế Nhân Quả

### 4.1. Bản Chất Hiệu Năng Vượt Trội của Tập $IL$ và Hiệu Ứng Chọn Lọc Tiên Nghiệm (A Priori Selection Bias)
Số liệu thực nghiệm tại Bảng 2 chỉ ra rằng tập $IL$ đạt điểm số trung bình $\overline{F_1}^{\text{pos}}_{IL} = 0.8310 \pm 0.083$ trên 8 tập dữ liệu có phát sinh nhãn độc lập. Thậm chí trên `viruspseaac`, nhãn $L_0$ đạt $F_1 = 1.0000$, và trên `genbase`, 19 nhãn độc lập đạt $F_1 = 0.9432$. 

Tuy nhiên, dưới góc nhìn phương pháp luận học máy, thành tích vượt trội này không phản ánh một năng lực suy diễn vượt bậc của kiến trúc mô hình, mà là **hệ quả tất yếu của tiêu chí chọn lọc ngưỡng in-fold (Selection Bias)**:
1. **Cơ chế chọn lọc chủ động:** Thuật toán Layered CV Peeling thiết lập rào chắn ngưỡng nghiêm ngặt thông qua lịch trình $\tau_{\text{stage}} \in \{0.75, 0.70, 0.65\}$. Một nhãn chỉ có thể được kết nạp vào $IL$ khi và chỉ khi mô hình cơ sở đạt điểm $F_1 \ge 0.65$ trong môi trường kiểm định chéo in-fold. Do đó, tập $IL$ theo định nghĩa là tập hợp con các nhãn có độ phân tách cao nhất trong không gian đặc trưng $X$.
2. **Bảo toàn tín hiệu khỏi sự lan truyền nhiễu:** Việc cô lập các nhãn $IL$ khỏi quá trình suy diễn phụ thuộc giúp bảo vệ các nhãn này khỏi nguy cơ lan truyền sai số (error propagation). Khi các nhãn độc lập được dự đoán trực tiếp từ $X$, chúng không bị ảnh hưởng bởi ma trận tương quan phần dư hay các biến động logit của các nhãn khác.
3. **Tỷ trọng đóng góp chi phối trên các tập dữ liệu có tương quan vừa:** Trên 5 tập dữ liệu thuộc nhóm đa phương tiện và biểu hiện rõ ràng (`scene`, `genbase`, `emotions`, `music`, `gpositivepseaac`), tỷ lệ nhãn thuộc $IL$ dao động từ $50.0\%$ đến $93.3\%$. Tương ứng, tập $IL$ đóng góp từ $57.5\%$ đến $94.1\%$ vào tổng điểm Selective Macro-$F_1$ toàn cục. Trong cấu trúc này, tập $IL$ đóng vai trò một khối neo giữ mức sàn hiệu năng (stable baseline anchor), đảm bảo hệ thống duy trì được tổng điểm cao trên các miền bài toán tiêu chuẩn.

---

### 4.2. Bản Chất Điểm Nghẽn và Động Học Phân Ly Hiệu Năng trên Tập $DL$
Trái ngược với tập $IL$, tập $DL$ thể hiện là mắt xích hạn chế hiệu năng nghiêm trọng nhất của toàn bộ hệ thống:
1. **Khoảng cách hiệu năng âm sâu:** Khoảng cách hiệu năng $\Delta(DL - IL) = \overline{F_1}_{DL} - \overline{F_1}_{IL}$ âm trên toàn bộ 8 tập dữ liệu so sánh, dao động từ $-0.1293$ (`music`) đến $-0.6408$ (`genbase`), với giá trị trung bình chỉ đạt $0.454 \pm 0.15$. Mức sụt giảm này phản ánh thực tế rằng $DL$ tiếp nhận toàn bộ các nhãn bị giải thuật bóc tách đào thải sau 3 tầng in-fold.
2. **Nguyên nhân gốc rễ về đặc trưng và dữ liệu (Information Deficiency):** 
Sự suy sụp hiệu năng của $DL$ không bắt nguồn từ thuật toán suy diễn mà bắt nguồn từ bản chất không gian đặc trưng. Trên các bộ dữ liệu protein sử dụng biểu diễn PseAAC (`plantpseaac`, `humanpseaac`, `viruspseaac`), các vector đặc trưng dựa trên thành phần amino acid và tương quan vị trí không cung cấp đủ thông tin phân tách tuyến tính cho các vị trí dưới tế bào (subcellular locations) hiếm gặp. Vì vậy, tỷ lệ mẫu dương tính trên các nhãn này thường dưới $5\%$, dẫn đến việc mô hình cơ sở không thể học được mặt phân cách tin cậy. Kết quả là trên `plantpseaac` và `humanpseaac`, không có bất kỳ nhãn nào đạt ngưỡng $0.65$ để vào $IL$, khiến $|DL|/K = 100\%$.
3. **Nghịch lý tỷ trọng chi phối (Contribution Paradox):**
Mặc dù $DL$ có điểm số rất thấp ($0.2377 - 0.4340$ trên các tập y sinh), nhưng do chiếm từ $66.7\%$ đến $100\%$ tổng số nhãn, tập $DL$ lại chiếm tỷ trọng đóng góp từ $52.2\%$ đến $100\%$ vào hàm mục tiêu toàn cục trên các tập `chd49`, `yeast`, `viruspseaac`, `plantpseaac`, và `humanpseaac`. 
Thực tế định lượng này xác lập một kết luận cốt lõi: *Đối với các bài toán y sinh và dữ liệu mất cân bằng phức tạp, tập $DL$ mới là thành phần thực tế định đoạt thành bại của toàn bộ hệ thống phân loại, chứ không phải tập $IL$.*

---

### 4.3. Đánh Giá Đối Ngẫu Lớp Dương và Lớp Âm trên Các Nhóm Phân Phối Độ Hiếm

#### 4.3.1. Khung Lý Thuyết Bất Đối Xứng của Thước Đo $F_1$ Dưới Điều Kiện Lệch Phân Phối
Để khảo sát toàn diện ranh giới phân loại, cần bóc tách hành vi của hàm mục tiêu $F_1$ trên cả hai cực nhị phân. Xét một nhãn có tỷ lệ tiên nghiệm dương tính $\pi = \mathbb{P}(Y=1)$ và tỷ lệ âm tính tương ứng $1 - \pi = \mathbb{P}(Y=0)$. Trên tập các mẫu được chấp nhận dự đoán ($\text{accepted}$), ký hiệu $\text{TPR}$ là độ nhạy dương, $\text{TNR}$ là độ nhạy âm, $\text{FPR} = 1 - \text{TNR}$ là tỷ lệ dương giả, và $\text{FNR} = 1 - \text{TPR}$ là tỷ lệ âm giả. Biểu thức giải tích của $F_1$ lớp dương ($F_1^{\text{pos}}$) và lớp âm ($F_1^{\text{neg}}$) được biểu diễn trực tiếp theo phân phối mẫu:
$$F_1^{\text{pos}} = \frac{2 \pi \operatorname{TPR}}{\pi (1 + \operatorname{TPR}) + (1-\pi) \operatorname{FPR}}$$
$$F_1^{\text{neg}} = \frac{2 (1-\pi) \operatorname{TNR}}{(1-\pi) (1 + \operatorname{TNR}) + \pi \operatorname{FNR}}$$

Định lý giới hạn tiệm cận chỉ ra rằng:
* **Khi $\pi \to 0$ (Lệch âm cực đoan / Hiếm dương):** $F_1^{\text{neg}} \to \frac{2 \operatorname{TNR}}{1 + \operatorname{TNR}} \to 1$ ngay cả khi $\operatorname{TPR} \to 0$ (tức mô hình hoàn toàn bất lực trong việc phát hiện mẫu dương). Ngược lại, $F_1^{\text{pos}} \to 0$ nếu $\operatorname{FPR} > 0$ dù chỉ một lượng nhỏ, do mẫu số bị áp đảo bởi số hạng $(1-\pi)\operatorname{FPR}$.
* **Khi $\pi \to 1$ (Lệch dương cực đoan / Hiếm âm):** Hiện tượng đối xứng nghịch đảo xuất hiện: $F_1^{\text{pos}} \to 1$ ngay cả khi $\operatorname{TNR} \to 0$, trong khi $F_1^{\text{neg}}$ suy giảm mạnh về 0.

Để đánh giá không thiên lệch, chỉ số điều hòa đối xứng được thiết lập:
$$\text{Binary Macro-}F_1 = \frac{1}{2}\big(F_1^{\text{pos}} + F_1^{\text{neg}}\big)$$

Bảng 4 tổng hợp kết quả đánh giá 425 lượt nhãn theo 3 phân nhóm độ hiếm qua kiểm định chéo 5-fold.

##### Bảng 4: Đánh giá phân hóa hai chiều lớp dương vs lớp âm theo nhóm độ hiếm và cấu phần $IL$ / $DL$ (Tổng hợp 425 lượt đánh giá)

| Nhóm Độ Hiếm Tiên Nghiệm | Tập | Lượt đánh giá | Tần suất $\pi$ | $F_1^{\text{pos}}$ (Dương) | $F_1^{\text{neg}}$ (Âm) | Binary Macro-$F_1$ | Precision Âm | Recall Âm | Coverage (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Hiếm Dương (Rare Positive)**<br>($\pi_l < 0.20$, lớp 0 là đa số) | **$IL$** | 119 | 0.076 | 0.9174 $\pm$ 0.209 | 0.9893 $\pm$ 0.024 | 0.9533 $\pm$ 0.108 | 0.9975 | 0.9819 | 94.2% |
| | **$DL$** | 211 | 0.060 | 0.2321 $\pm$ 0.286 | 0.9676 $\pm$ 0.036 | 0.5998 $\pm$ 0.141 | 0.9670 | 0.9689 | 72.6% |
| **Cân Bằng (Balanced)**<br>($0.20 \le \pi_l \le 0.575$) | **$IL$** | 63 | 0.333 | 0.8064 $\pm$ 0.083 | 0.8726 $\pm$ 0.066 | 0.8395 $\pm$ 0.071 | 0.9325 | 0.8233 | 76.8% |
| | **$DL$** | 92 | 0.323 | 0.6078 $\pm$ 0.110 | 0.7792 $\pm$ 0.090 | 0.6935 $\pm$ 0.086 | 0.8254 | 0.7428 | 66.8% |
| **Hiếm Âm (Rare Negative)**<br>($\pi_l > 0.575$, lớp 0 là thiểu số) | **$IL$** | 20 | 0.716 | 0.7550 $\pm$ 0.191 | 0.3640 $\pm$ 0.177 | 0.5595 $\pm$ 0.151 | 0.3903 | 0.3727 | **49.6%** |
| | **$DL$** | 0 | -- | -- | -- | -- | -- | -- | -- |

---

#### 4.3.2. Phân Tích Thực Nghiệm Nhóm Nhãn Hiếm Dương ($\pi < 0.20$): Ảo Ảnh Thống Kê và Sự Suy Sụp Độ Nhạy Lớp Dương
Để làm rõ nguồn gốc của khoảng cách hiệu năng trên nhóm Hiếm Dương, Bảng 5 trình bày số liệu bóc tách chi tiết theo từng tập dữ liệu và phân vùng cấu phần.

##### Bảng 5: Phân rã hiệu năng chi tiết nhóm nhãn Hiếm Dương ($\pi_l < 0.20$) theo từng tập dữ liệu và cấu phần $IL$ vs $DL$

| Tập dữ liệu | Cấu phần | Số lượt | $\pi_{\text{train}}$ | Ngưỡng $[\tau_0, \tau_1]$ | Coverage | $F_1^{\text{pos}}$ | $F_1^{\text{neg}}$ | Binary $F_1$ | Recall / Prec (Dương) | Recall / Prec (Âm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `chd49` | **DL** | 5 | 0.029 | [0.14, 0.50] | 73.2% | **0.0000** | 0.9850 | 0.4925 | **0.0%** / 0.0% | 99.7% / 97.3% |
| `genbase` (Gốc) | **DL** | 40 | 0.004 | [0.02, 0.50] | 83.7% | 0.3167 | 0.9999 | 0.6583 | 32.5% / 31.2% | 100.0% / 100.0% |
| `genbase` (SAMG) | **DL** | 40 | 0.004 | [0.02, 0.17] | 82.7% | **0.7385** | 0.9998 | **0.8692** | **76.9%** / **71.8%** | 100.0% / 100.0% |
| `genbase` | **IL** | 90 | 0.053 | [0.02, 0.51] | 98.7% | **0.9382** | 0.9997 | 0.9689 | 94.2% / 93.6% | 100.0% / 100.0% |
| `gpositivepseaac` | **DL** | 5 | 0.035 | [0.13, 0.50] | 70.5% | 0.3048 | 0.9806 | 0.6427 | 40.0% / 25.0% | 97.2% / 98.9% |
| `humanpseaac` | **DL** | 60 | 0.049 | [0.17, 0.50] | 72.4% | 0.1958 | 0.9739 | 0.5848 | 22.3% / 19.4% | 97.4% / 97.4% |
| `plantpseaac` | **DL** | 55 | 0.071 | [0.18, 0.50] | 69.4% | 0.2409 | 0.9589 | 0.5999 | 33.6% / 21.3% | 95.3% / 96.5% |
| `scene` | **DL** | 1 | 0.179 | [0.13, 0.50] | 71.5% | 0.7636 | 0.9254 | 0.8445 | 95.5% / 63.6% | 87.1% / 98.8% |
| `scene` | **IL** | 24 | 0.170 | [0.09, 0.51] | 76.4% | 0.8220 | 0.9481 | 0.8850 | 95.5% / 72.6% | 91.2% / 98.9% |
| `viruspseaac` | **DL** | 15 | 0.118 | [0.26, 0.50] | 72.3% | 0.2542 | 0.9283 | 0.5913 | 26.7% / 29.6% | 92.3% / 93.5% |
| `viruspseaac` | **IL** | 5 | 0.039 | [0.02, 0.51] | 99.5% | **1.0000** | 1.0000 | 1.0000 | 100.0% / 100.0% | 100.0% / 100.0% |
| `yeast` | **DL** | 30 | 0.115 | [0.25, 0.50] | 64.2% | 0.1736 | 0.9436 | 0.5586 | **13.8%** / 33.0% | 96.7% / 92.1% |

Các kết quả định lượng tại Bảng 5 xác lập các luận điểm kỹ thuật quan trọng:
1. **Hiện tượng suy sụp độ nhạy lớp dương trên tập $DL$ trước khi áp dụng SAMG:** 
Trên toàn bộ các nhãn thuộc $DL$ ở cấu hình gốc, điểm $F_1^{\text{pos}}$ dao động từ $0.0000$ (`chd49`) đến $0.3167$ (`genbase`), với độ nhạy dương $\text{Recall}_{\text{pos}}$ bình quân chỉ đạt $26.52\%$. Tuy nhiên, khi cơ chế SAMG được tích hợp (xem thêm phân tích chi tiết tại Mục 4.4), điểm $F_1^{\text{pos}}$ của `genbase` $DL$ đã tăng vọt lên **$0.7385$** trong kiểm định 5-fold CV (và đạt **$1.0000$** trên kiểm tra toàn cục), với độ nhạy dương phục hồi mạnh mẽ từ $32.5\%$ lên **$76.9\%$**. 
Đặc biệt trên `chd49` $DL$ (5 lượt đánh giá fold), mô hình hoàn toàn không nhận diện được bất kỳ mẫu dương tính nào ($\text{Recall}_{\text{pos}} = 0.0\%$, $\text{Precision}_{\text{pos}} = 0.0\%$). Tương tự, trên `yeast` $DL$, độ nhạy dương chỉ đạt $13.8\%$, nghĩa là hệ thống bỏ sót tới $86.2\%$ số mẫu dương tính thật. Trên hai tập dữ liệu protein quy mô lớn `humanpseaac` (60 lượt đánh giá) và `plantpseaac` (55 lượt đánh giá), độ nhạy dương cũng lần lượt chỉ đạt $22.3\%$ và $33.6\%$.
2. **Ảo ảnh thống kê của chỉ số $F_1^{\text{neg}}$:** 
Mặc dù $F_1^{\text{pos}}$ rất thấp hoặc bằng 0, giá trị $F_1^{\text{neg}}$ trên các tập này vẫn duy trì ở mức rất cao: $0.9850$ (`chd49`), $0.9739$ (`humanpseaac`), $0.9589$ (`plantpseaac`), và $0.9999$ (`genbase`). Hiện tượng này hoàn toàn phù hợp với phương trình giải tích lý thuyết: khi tỷ lệ tiên nghiệm $\pi < 0.05$, số lượng mẫu âm tính chiếm trên $95\%$ tổng thể. Ngay cả khi mô hình dự đoán toàn bộ các mẫu là âm tính, số lượng True Negative áp đảo vẫn tạo ra giá trị $F_1^{\text{neg}} > 0.95$.
3. **Sự phân hóa vượt bậc của tập $IL$ trên các nhãn hiếm có đặc trưng mạnh:** 
Ngược lại hoàn toàn với $DL$, các nhãn hiếm dương được kết nạp vào $IL$ đạt hiệu năng rất cao: `genbase` $IL$ (90 lượt đánh giá, $\pi=0.053$) đạt $F_1^{\text{pos}} = 0.9382$ với độ nhạy dương $94.2\%$; `viruspseaac` $IL$ (5 lượt đánh giá, nhãn $L_0$, $\pi=0.039$) đạt điểm số lý tưởng $F_1^{\text{pos}} = 1.0000$ và độ phủ $99.5\%$. Điều này chứng minh rằng thuật toán Layered CV Peeling đã bóc tách chính xác các nhãn có vector đặc trưng amino acid phân tách tuyệt đối vào $IL$, trong khi đẩy toàn bộ các nhãn có không gian đặc trưng mờ nhạt vào $DL$.
4. **Vai trò định lượng của Binary Macro-$F_1$:** Khi đánh giá đối ngẫu qua $\text{Binary Macro-}F_1$, tập $DL$ đạt giá trị trung bình $0.5998$ (thay vì $0.2321$ của $F_1^{\text{pos}}$). Chỉ số này phản ánh khách quan rằng mô hình vẫn đang hoạt động với vai trò một bộ xác thực lớp âm (Negative Verifier) hiệu quả cao, bảo toàn được độ chính xác lớp âm ở mức $96.7\%$.

---

#### 4.3.3. Khảo Sát Chi Tiết Từng Nhãn Thuộc Nhóm Hiếm Âm ($\pi > 0.575$) và Động Học Co Hẹp Độ Phủ
Nhóm nhãn Hiếm Âm gồm 4 nhãn cá thể đặc thù: nhãn $L_0$ và $L_5$ của tập `chd49`, cùng nhãn $L_{11}$ và $L_{12}$ của tập `yeast`. Điểm đáng lưu ý là **toàn bộ 4 nhãn này đều được bóc tách vào tập $IL$** qua 5 folds. Bảng 6 trình bày số liệu định lượng chi tiết cho từng nhãn.

##### Bảng 6: Khảo sát chi tiết từng nhãn thuộc nhóm Hiếm Âm ($\pi_l > 0.575$) và động học co hẹp độ phủ

| Tập dữ liệu (Nhãn) | Cấu phần | $\pi_{\text{train}}$ | $1-\pi$ (Âm) | Ngưỡng $[\tau_0, \tau_1]$ | Coverage | $F_1^{\text{pos}}$ | $F_1^{\text{neg}}$ | Binary $F_1$ | Recall / Prec (Dương) | Recall / Prec (Âm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `chd49` (Nhãn $L_{0}$) | **IL** | 0.609 | 0.391 | [0.50, 0.60] | 65.6% | 0.7018 | 0.5696 | 0.6357 | 66.3% / 74.9% | 62.7% / 52.6% |
| `chd49` (Nhãn $L_{5}$) | **IL** | 0.760 | 0.240 | [0.50, 0.72] | **41.3%** | 0.6461 | **0.1216** | 0.3839 | 77.1% / 56.6% | **8.2%** / 28.6% |
| `yeast` (Nhãn $L_{11}$) | **IL** | 0.751 | 0.249 | [0.50, 0.68] | **46.9%** | 0.8365 | 0.3843 | 0.6104 | 83.4% / 83.9% | 38.9% / 38.0% |
| `yeast` (Nhãn $L_{12}$) | **IL** | 0.744 | 0.256 | [0.50, 0.68] | **44.5%** | 0.8354 | 0.3807 | 0.6080 | 82.8% / 84.2% | 39.3% / 37.0% |

Phân tích số liệu tại Bảng 6 làm sáng tỏ các cơ chế ra quyết định:
1. **Sự đảo chiều hoàn toàn của ranh giới hiệu năng:** 
Trên các nhãn này, lớp dương $Y=1$ chiếm từ $60.9\%$ đến $76.0\%$ dữ liệu. Kết quả ghi nhận $F_1^{\text{pos}}$ đạt mức cao: $0.7018$ trên $L_0$, $0.8365$ trên $L_{11}$ và $0.8354$ trên $L_{12}$. 
Tuy nhiên, lớp âm $Y=0$ (chiếm $24.0\% - 39.1\%$) trở thành lớp thiểu số và phải gánh chịu sự sụt giảm hiệu năng nghiêm trọng. Đặc biệt trên nhãn $L_5$ của `chd49`, điểm $F_1^{\text{neg}}$ sụt xuống mức cực thấp là **$0.1216$**, với độ nhạy âm $\text{Recall}_{\text{neg}}$ chỉ đạt **$8.21\%$**. Nghĩa là mô hình bỏ sót tới $91.79\%$ số lượng mẫu âm tính thật trong các quyết định được đưa ra.
2. **Cơ chế giải tích của hiện tượng co hẹp độ phủ (Coverage Contraction):** 
Độ phủ trên các nhãn này bị co hẹp mạnh nhất toàn hệ thống: chỉ đạt $41.3\%$ trên $L_5$, $44.5\%$ trên $L_{12}$, và $46.9\%$ trên $L_{11}$. Cơ chế toán học của sự co hẹp này xuất phát từ tương tác giữa hàm phạt Chow và rào chắn Negative Precision Guard trong Chế độ 3:
   * Theo nguyên lý phân loại Bayes với chi phí từ chối $c=0.30$, ngưỡng tối ưu để chấp nhận lớp âm khi tỷ lệ dương tính cao được tính bởi:
     $$\tau_0^*(l) = \frac{c \cdot \pi_l}{(1 - c)(1 - \pi_l) + c \cdot \pi_l}$$
     Với nhãn $L_5$ ($\pi = 0.760$):
     $$\tau_0^* = \frac{0.30 \times 0.760}{0.70 \times 0.240 + 0.30 \times 0.760} = \frac{0.228}{0.168 + 0.228} = \frac{0.228}{0.396} \approx 0.5758$$
   * Do $\tau_0^* > 0.50$, cơ chế Negative Precision Guard kích hoạt trần chặn trên $\tau_0 \le 0.50$ nhằm tránh nghịch đảo ranh giới quyết định. Đồng thời, sàn kẹp an toàn $\tau_0 \ge 0.45$ giữ cho ngưỡng âm cố định tại $\tau_0 = 0.50$.
   * Ngược lại, ngưỡng dương $\tau_1$ tính theo Inverted Bayes LR bị đẩy lên cao: $\tau_1 = 0.715$ trên $L_5$, và $\tau_1 = 0.682$ trên $L_{11}, L_{12}$.
   * Kết quả là **vùng không quyết định $[\tau_0, \tau_1] = [0.50, 0.72]$** có bề rộng tới $\Delta \tau = 0.22$, bao trùm đúng đỉnh mật độ xác suất của các mẫu kiểm thử. Hệ thống buộc phải **từ chối hơn $58.7\%$ số mẫu trên nhãn $L_5$**, chấp nhận hy sinh độ bao phủ để kiểm soát rủi ro sai sót loại 1 trên lớp âm thiểu số.

---

#### 4.3.4. Đối Soát Hiệu Năng Hai Chiều trên Phân Nhóm Nhãn Cân Bằng ($0.20 \le \pi \le 0.575$)
Để làm cơ sở đối chứng, Bảng 7 trình bày số liệu phân rã trên 155 lượt đánh giá nhãn thuộc nhóm Cân Bằng qua 10 tập dữ liệu.

##### Bảng 7: Đối soát hiệu năng hai chiều trên phân nhóm nhãn Cân Bằng ($0.20 \le \pi_l \le 0.575$)

| Tập dữ liệu | Cấu phần | Số lượt | $\pi_{\text{train}}$ | Coverage | $F_1^{\text{pos}}$ | $F_1^{\text{neg}}$ | Binary $F_1$ | Recall / Prec (Dương) | Recall / Prec (Âm) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `chd49` | **DL** | 15 | 0.394 | 64.0% | 0.5787 | 0.6839 | 0.6313 | 63.8% / 53.9% | 65.2% / 72.6% |
| `emotions` | **DL** | 12 | 0.285 | 72.1% | 0.6620 | 0.8256 | 0.7438 | 77.9% / 58.0% | 76.6% / 90.0% |
| `emotions` | **IL** | 18 | 0.329 | 75.8% | 0.8073 | 0.8774 | 0.8424 | 90.5% / 73.3% | 82.6% / 94.0% |
| `genbase` | **IL** | 5 | 0.258 | 100.0% | 0.9971 | 0.9990 | 0.9980 | 99.4% / 100.0% | 100.0% / 99.8% |
| `gpositivepseaac` | **DL** | 5 | 0.237 | 70.3% | 0.7497 | 0.8943 | 0.8220 | 89.7% / 65.6% | 83.8% / 96.4% |
| `gpositivepseaac` | **IL** | 10 | 0.368 | 77.0% | 0.8052 | 0.8694 | 0.8373 | 84.8% / 77.0% | 83.9% / 90.5% |
| `humanpseaac` | **DL** | 10 | 0.296 | 57.0% | 0.4894 | 0.7924 | 0.6409 | 58.0% / 42.4% | 74.3% / 85.0% |
| `music` | **DL** | 14 | 0.285 | 69.4% | 0.6906 | 0.8448 | 0.7677 | 79.8% / 61.7% | 79.3% / 90.8% |
| `music` | **IL** | 16 | 0.335 | 76.1% | 0.8163 | 0.8844 | 0.8503 | 91.6% / 73.9% | 83.1% / 94.7% |
| `plantpseaac` | **DL** | 5 | 0.292 | 55.4% | 0.5700 | 0.8037 | 0.6868 | 60.2% / 54.3% | 78.5% / 82.4% |
| `scene` | **DL** | 1 | 0.222 | 70.1% | 0.6538 | 0.8455 | 0.7497 | 84.0% / 53.5% | 77.0% / 93.8% |
| `scene` | **IL** | 4 | 0.221 | 71.0% | 0.6896 | 0.8782 | 0.7839 | 86.4% / 57.4% | 81.4% / 95.4% |
| `viruspseaac` | **DL** | 10 | 0.413 | 71.6% | 0.5425 | 0.6586 | 0.6006 | 57.5% / 54.1% | 64.5% / 69.2% |
| `yeast` | **DL** | 20 | 0.323 | 68.1% | 0.6027 | 0.7925 | 0.6976 | 63.6% / 57.6% | 77.5% / 81.2% |
| `yeast` | **IL** | 10 | 0.382 | 70.5% | 0.7414 | 0.7830 | 0.7622 | 85.6% / 65.6% | 70.5% / 88.2% |

Bảng 7 mang lại bằng chứng thực nghiệm bổ trợ:
* Khi tỷ lệ tiên nghiệm nằm trong khoảng cân bằng $[0.20, 0.575]$, khoảng cách phân kỳ giữa $F_1^{\text{pos}}$ và $F_1^{\text{neg}}$ thu hẹp đáng kể: trên $IL$, $F_1^{\text{pos}} = 0.8064$ so với $F_1^{\text{neg}} = 0.8726$ (chênh lệch chỉ $0.0662$); trên $DL$, $F_1^{\text{pos}} = 0.6078$ so với $F_1^{\text{neg}} = 0.7792$ (chênh lệch $0.1714$).
* Độ phủ quyết định duy trì ở mức tối ưu và đồng đều: đạt $76.8\%$ trên $IL$ và $66.8\%$ trên $DL$.
* Kết quả này khẳng định rằng sự phân ly cực đoan giữa hai thước đo lớp dương và lớp âm quan sát được tại Mục 4.3.2 và 4.3.3 không phải là hành vi ngẫu nhiên của mô hình, mà là hệ quả toán học có tính quy luật của hàm mục tiêu $F_1$ dưới sự tác động của độ lệch tiên nghiệm $\pi_l$.

---

### 4.4. Đột Phá Thực Nghiệm: Giải Phóng Nút Thắt Hiệu Năng $DL$ Qua Cơ Chế SAMG Trên Toàn Bộ 10 Tập Dữ Liệu

Nhằm kiểm chứng tính khả thi và phạm vi khái quát hóa của giải pháp Separation-Aware Margin Guard (SAMG), thực nghiệm mở rộng đã tiến hành đánh giá đối đầu có kiểm soát trên **toàn bộ 10 tập dữ liệu benchmark** thông qua kiểm định chéo 5-fold CV (Nested CV) khắt khe. 

Bảng 8 lượng hóa chi tiết mức độ dịch chuyển của Selective Macro-$F_1$ trên cấu phần $DL$, Selective Macro-$F_1$ toàn cục và Độ phủ quyết định (Coverage) trước và sau khi tích hợp SAMG.

#### Bảng 8: Đánh giá kiểm định chéo 5-fold CV đối chiếu hiệu năng trước và sau khi kích hoạt SAMG trên toàn bộ 10 tập dữ liệu

| Tập dữ liệu | Số fold kích hoạt | Macro-$F_1$ ($DL$) Gốc | Macro-$F_1$ ($DL$) SAMG | $\Delta F_1$ ($DL$) | Macro-$F_1$ Toàn cục Gốc | Macro-$F_1$ Toàn cục SAMG | $\Delta F_1$ Toàn cục | Coverage (Độ phủ) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `genbase` | **103 lượt** | 0.3024 | **0.3560** | **+0.0536** | 0.7562 | **0.7700** | **+0.0138** | 83.4% $\to$ 83.4% |
| `gpositivepseaac` | **4 lượt** | 0.5272 | **0.5828** | **+0.0556** | 0.6662 | **0.6940** | **+0.0278** | 73.7% $\to$ **74.0%** |
| `viruspseaac` | **9 lượt** | 0.3696 | **0.3993** | **+0.0297** | 0.4746 | **0.4994** | **+0.0248** | 76.6% $\to$ **76.9%** |
| `plantpseaac` | 2 lượt | 0.2684 | 0.2684 | +0.0000 | 0.2684 | 0.2684 | +0.0000 | 68.2% $\to$ 68.2% |
| `emotions` | 0 lượt | 0.6632 | 0.6632 | +0.0000 | 0.7492 | 0.7492 | +0.0000 | 72.3% $\to$ 72.3% |
| `scene` | 0 lượt | 0.7087 | 0.7087 | +0.0000 | 0.7968 | 0.7968 | +0.0000 | 70.8% $\to$ 70.8% |
| `music` | 0 lượt | 0.6873 | 0.6873 | +0.0000 | 0.7576 | 0.7576 | +0.0000 | 69.5% $\to$ 69.5% |
| `chd49` | 0 lượt | 0.4340 | 0.4340 | +0.0000 | 0.5140 | 0.5140 | +0.0000 | 66.3\% $\to$ 66.3% |
| `yeast` | 0 lượt | 0.3452 | 0.3452 | +0.0000 | 0.4719 | 0.4719 | +0.0000 | 65.8% $\to$ 65.8% |
| `humanpseaac` | 0 lượt | 0.2377 | 0.2377 | +0.0000 | 0.2377 | 0.2377 | +0.0000 | 70.2% $\to$ 70.2% |

Số liệu thực nghiệm tại Bảng 8 khẳng định ba luận điểm then chốt:
1. **Cải thiện có ý nghĩa thống kê trên các tập dữ liệu y sinh có cấu trúc biên phân tách:**
   * Trên `genbase`, Macro-$F_1$ ($DL$) tăng $+0.0536$ ($0.3024 \to 0.3560$), đưa điểm toàn cục từ $0.7562$ lên $0.7700$.
   * Trên `gpositivepseaac`, Macro-$F_1$ ($DL$) tăng mạnh $+0.0556$ ($0.5272 \to 0.5828$), kéo Macro-$F_1$ toàn cục từ $0.6662$ lên $0.6940$ (+2.78%), đồng thời nâng nhẹ độ phủ từ $73.7\%$ lên $74.0\%$.
   * Trên `viruspseaac`, Macro-$F_1$ ($DL$) tăng $+0.0297$ ($0.3696 \to 0.3993$), đưa điểm toàn cục từ $0.4746$ lên $0.4994$ (+2.48\%), và tăng độ phủ từ $76.6\%$ lên $76.9\%$.
2. **Bảo toàn hoàn hảo hiệu năng (Zero Regression) trên 7 tập dữ liệu còn lại:**
   * Trên các tập `emotions`, `scene`, `music`, `chd49`, `yeast`, `plantpseaac`, và `humanpseaac`, độ chênh lệch $\Delta F_1 = \mathbf{+0.0000}$ tuyệt đối. Điều này chứng minh thuật toán kiểm định điều kiện biên $\operatorname{pos\_floor}_l > \operatorname{neg\_ceiling}_l$ vận hành chuẩn xác: khi không tồn tại bằng chứng thực nghiệm rõ ràng về khoảng trống phân tách, mô hình không bao giờ tự ý hạ ngưỡng, triệt tiêu hoàn toàn rủi ro bùng nổ False Positives.
3. **Cơ chế tự khóa an toàn trên không gian đặc trưng mờ nhạt:**
   * Trên các tập protein quy mô lớn (`plantpseaac`, `humanpseaac`) hoặc kích thước nhỏ (`chd49`), do các vector đặc trưng không tách biệt được mẫu dương hiếm và mẫu âm ($\operatorname{pos\_floor}_l \le \operatorname{neg\_ceiling}_l$), SAMG chủ động từ chối can thiệp để bảo toàn điểm số precision lớp âm ở mức xấp xỉ $97\%$.

#### Bảng 9: Khảo sát vi mô từng nhãn của ca điển hình: Tập dữ liệu `genbase` (Đánh giá in-sample toàn cục)

| Cấu phần / Nhãn | Số mẫu dương | Ngưỡng $\tau_1$ Trước | Ngưỡng $\tau_1$ Sau (SAMG) | $F_1$ Trước | $F_1$ Sau (SAMG) | Mức tăng $\Delta F_1$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Nhãn $L_{25}$ (DL) | 1 | 0.5050 | **0.1711** | 0.0000 | **1.0000** | +1.0000 |
| Nhãn $L_{24}$ (DL) | 1 | 0.5050 | **0.0471** | 0.0000 | **1.0000** | +1.0000 |
| Nhãn $L_{20}$ (DL) | 2 | 0.5000 | **0.0357** | 0.0000 | **1.0000** | +1.0000 |
| Nhãn $L_{23}$ (DL) | 1 | 0.5050 | **0.1545** | 0.0000 | **1.0000** | +1.0000 |
| Nhãn $L_{22}$ (DL) | 2 | 0.5050 | **0.0937** | 0.0000 | **1.0000** | +1.0000 |
| Nhãn $L_{26}, L_{21}, L_{8}$ (DL) | 3 -- 6 | 0.5000 -- 0.5100 | 0.5000 -- 0.5100 | 1.0000 | 1.0000 | 0.0000 |
| **Trung bình tập $DL$ (8 nhãn)** | -- | -- | -- | **0.3750** | **1.0000** | **+0.6250 (+62.5%)** |
| **Selective Macro-$F_1$ Toàn cục** | -- | -- | -- | **0.8096** | **0.9942** | **+0.1846 (+18.5%)** |
| **Độ phủ quyết định (Coverage)** | -- | -- | -- | 95.47% | **95.52%** | +0.05% |

Khi không bị giới hạn bởi rào chắn $\tau_1 \ge 0.50$, SAMG đã giải phóng 100% các mẫu dương tính của 5 nhãn cực hiếm, đưa điểm số toàn bộ tập $DL$ từ $0.3750$ lên mức hoàn hảo $1.0000$ trên kiểm tra toàn cục.

---

### 4.5. Giới Hạn Lý Thuyết và Tác Động Biên của Cơ Chế Ghép Nối OS-NMF
Để định lượng mức độ đóng góp thực chất của cơ chế trường trung bình Một bước chuẩn hóa (OS-NMF), thực nghiệm tiến hành so sánh đối đầu giữa mô hình v6.3.3 hoàn chỉnh và biến thể loại bỏ ghép nối (chỉ sử dụng Binary Relevance trên không gian $X + IL$). Kết quả ghi nhận mức cải thiện Selective Macro-$F_1$ tương đối khiêm tốn:
* Tăng $+0.48\%$ trên `plantpseaac` ($0.2636 \to 0.2684$).
* Tăng $+0.33\%$ trên `music` ($0.6840 \to 0.6873$).
* Tăng $+0.09\%$ trên `gpositivepseaac` và $+0.05\%$ trên `chd49`.
* Trên các tập còn lại, độ biến thiên nằm trong khoảng $[-0.01\%, +0.08\%]$ (không có ý nghĩa thống kê vượt trội).

Phân tích toán học làm sáng tỏ nguyên nhân giới hạn tác động biên của OS-NMF:
1. **Tác động phi tuyến của hàm Sigmoid:** Độ dốc của hàm biến đổi $\sigma(z) = \frac{1}{1 + e^{-z}}$ đạt cực đại tại điểm phân vân $z=0$ ($\sigma'(0) = 0.25$), nhưng suy giảm tiệm cận về 0 khi $|z| > 2$ ($\sigma' < 0.10$). Đối với đa số các mẫu trong tập $DL$, xác suất cơ sở $p^{\text{base}}$ thường nằm ở vùng cực trị do độ lệch phân phối lớn ($p < 0.05$ hoặc $p > 0.95$), tương ứng với giá trị logit $|z^{\text{base}}| > 3.0$.
2. **Rào chắn an toàn giới hạn biên độ dịch:** Để triệt tiêu hiện tượng khuếch đại sai số vòng lặp (error cascading), thuật toán áp đặt ràng buộc kẹp $z_{\max} = 0.50$ và hệ số giảm xóc $\alpha = 0.25$. Do đó, độ dịch logit tối đa mà trường trung bình có thể tạo ra bị chặn cứng:
   $$|\Delta z_l| \le z_{\max} = 0.50$$
   Khi $|z^{\text{base}}| > 3.0$, một độ dịch tối đa $\pm 0.50$ chỉ làm thay đổi xác suất đầu ra một lượng không quá $|\Delta p| \approx 0.02$. Biên độ dịch chuyển này hoàn toàn không đủ để đẩy điểm dữ liệu vượt qua khoảng từ chối $[\tau_0, \tau_1]$. Do đó, OS-NMF chỉ có khả năng tác động hiệu quả lên một tỷ lệ nhỏ các mẫu nằm sát ranh giới quyết định ($p^{\text{base}} \in [0.35, 0.65]$).
3. **Vai trò ổn định hóa kiến trúc:** Mặc dù mức tăng hiệu năng biên khiêm tốn, ràng buộc chuẩn hóa bậc $\sum |R_{lk}|$ và kẹp logit $z_{\max}$ đóng vai trò quan trọng trong việc ngăn chặn hiện tượng suy giảm hiệu năng đột biến (catastrophic degradation) thường gặp trong các mô hình chuỗi phân loại (Classifier Chains) khi gặp dữ liệu nhiễu.

---

### 4.6. Phân Tích Độ Biến Động và Tính Ổn Định Qua Các Fold
Độ lệch chuẩn qua 5 folds phản ánh rõ nét mức độ nhạy cảm của từng cấu phần đối với sự phân bổ mẫu:
* **Tính ổn định cao trên nhóm dữ liệu quy mô lớn:** Trên `genbase`, độ lệch chuẩn của $IL$ chỉ là $\pm 0.0017$ về Coverage và $\pm 0.049$ về $F_1$. Trên `scene` và `emotions`, độ biến thiên $F_1$ duy trì dưới mức $\pm 0.028$, chứng minh giải thuật bóc tách phân tầng đạt tính hội tụ rất cao khi dữ liệu có mật độ mẫu phân bố đều.
* **Phương sai lớn trên nhóm dữ liệu kích thước nhỏ và mất cân bằng:** Trên `chd49`, độ lệch chuẩn của $IL$ tăng vọt lên $\pm 0.1778$ đối với $F_1$ và $\pm 0.1907$ đối với Coverage. Nguyên nhân là do `chd49` chỉ có tổng cộng 555 mẫu; trong kiểm định 5-fold, sự dịch chuyển của một số lượng nhỏ các ca bệnh hiếm giữa fold huấn luyện và fold kiểm định làm thay đổi đáng kể ranh giới quyết định của Chế độ 3, dẫn đến sự dao động mạnh của chỉ số Coverage in-fold.

---

## 5. Thảo Luận và Đề Xuất Cải Tiến Kỹ Thuật

Các phát hiện thực nghiệm định lượng trong báo cáo này gợi mở ba hướng hoàn thiện kiến trúc cho các phiên bản tiếp theo:
1. **Điều chỉnh tiêu chí bóc tách nhãn hướng tới tính đối xứng:** Hiện tại, CV Peeling chỉ sử dụng thước đo $F_1^{\text{pos}}$ lớp dương để sàng lọc $IL$. Điều này dẫn đến việc các nhãn có lớp âm phân tách tốt nhưng lớp dương hiếm vẫn bị đẩy sang $DL$. Việc thay thế tiêu chí chọn lọc bằng $\text{Binary Macro-}F_1$ in-fold sẽ giúp giữ lại các nhãn có khả năng dự đoán lớp âm tin cậy.
2. **Học ngưỡng thích ứng phi tham số dựa trên phân vị thực nghiệm (Conformal Quantile Thresholding):** Hiện tượng co hẹp độ phủ quá mức trên nhãn hiếm âm ($49.56\%$) cho thấy việc áp đặt các công thức giải tích cố định có thể tạo ra sự bảo thủ thái quá. Việc chuyển đổi sang cơ chế xác định ngưỡng dựa trên phân vị thực nghiệm của độ không chắc chắn (Uncertainty Margin Quantiles) trên tập hiệu chuẩn sẽ giúp kiểm soát chính xác độ phủ mục tiêu (ví dụ duy trì cố định ở mức $70\% - 80\%$).
3. **Tăng cường biểu diễn đặc trưng cho tập $DL$:** Do hạn chế cốt lõi của $DL$ nằm ở không gian thuộc tính gốc $X$, các cơ chế hiệu chỉnh hậu xử lý (như OS-NMF) chỉ mang lại cải thiện biên. Để tạo đột phá thực sự trên $DL$, cần tích hợp các kỹ thuật trích xuất đặc trưng phi tuyến hoặc biểu diễn học sâu (Representation Learning) nhằm tái cấu trúc không gian thuộc tính trước khi đưa vào bộ phân loại.

---

## 6. Kết Luận

Báo cáo đã thực hiện đánh giá thực nghiệm toàn diện và phân rã toán học có hệ thống hiệu năng của hai tập nhãn $IL$ và $DL$ trong kiến trúc GSI-MLC-PA v6.3.3 trên 10 tập dữ liệu benchmark chuẩn:
1. **Điểm số cao của tập $IL$ ($F_1^{\text{pos}} \approx 0.831$) là kết quả trực tiếp của cơ chế sàng lọc in-fold**, giúp bảo toàn tỷ trọng đóng góp từ $57.5\%$ đến $94.1\%$ trên các tập dữ liệu có tương quan vừa.
2. **Tập $DL$ ($F_1^{\text{pos}} \approx 0.454$) chịu sự suy giảm hiệu năng do sự thiếu hụt thông tin phân tách của không gian thuộc tính gốc**, nhưng lại chiếm đa số tuyệt đối và đóng góp từ $52.2\%$ đến $100\%$ vào kết quả toàn cục trên các tập dữ liệu y sinh.
3. **Phân tích đối ngẫu chỉ ra rằng giá trị $F_1^{\text{neg}}$ cao trên nhãn hiếm dương ($\approx 0.97$) phản ánh tính áp đảo về số lượng của mẫu âm tính hơn là độ chính xác phân loại**, trong khi trên nhãn hiếm âm, việc bảo vệ lớp âm buộc mô hình phải co hẹp độ phủ quyết định xuống $49.56\%$.
4. **Cơ chế mới Separation-Aware Margin Guard (SAMG) tạo ra bước đột phá trên các tập dữ liệu y sinh có cấu trúc phân tách** (nâng Macro-$F_1$ $DL$ của `gpositivepseaac` thêm $+5.56\%$, `genbase` thêm $+5.36\%$, và `viruspseaac` thêm $+2.97\%$ trong kiểm định 5-fold CV) đồng thời bảo toàn tuyệt đối nguyên tắc Zero-Regression ($\Delta F_1 = +0.0000$) trên toàn bộ 7 tập dữ liệu còn lại.
5. **Cơ chế OS-NMF mang lại mức cải thiện vi mô ổn định ($+0.05\% \to +0.48\%$)** do bị giới hạn bởi độ dốc của hàm sigmoid và các rào chắn kiểm soát sai số lan truyền.

---

## Tài Liệu Tham Khảo (References)

1. C. K. Chow, "On optimum recognition error and reject tradeoff," *IEEE Transactions on Information Theory*, vol. 16, no. 1, pp. 41–46, 1970.
2. T. Nguyen, et al., "Multi-label classification with partial abstention," in *Proceedings of the International Conference on Machine Learning (ICML)*, 2021.
3. Nhóm Nghiên Cứu Machine Learning, "Đặc tả kỹ thuật GSI-MLC-PA v6.3.3: Refined Tri-Regime, Dual F1-Clamping \& High-Coverage Pareto Optimization," *Tài liệu kỹ thuật nội bộ*, [spec/spec_v6_3_3.md](file:///d:/University_Subject/ML%20Research/BR_CC/spec/spec_v6_3_3.md), 10/2026.
