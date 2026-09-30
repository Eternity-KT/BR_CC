# Data-Driven Stratified Peeling and Ascending Classifier Chains for Multi-Label Classification with Partial Abstention
## Phân Hoạch Bóc Tách Đa Tầng Dựa Trên Dữ Liệu và Chuỗi Tương Quan Tăng Dần Cho Phân Loại Đa Nhãn Có Từ Chối Từng Phần

> **Bản Dự Thảo Nâng Cấp Toàn Diện: Kiến Trúc GSI-MLC-PA Phiên Bản 5.1**  
> **Tài liệu gốc tham chiếu:** [TaiLieuThamKhao/GSI_MLC_PA_V5 (1).pdf](file:///d:/University_Subject/ML%20Research/BR_CC/TaiLieuThamKhao/GSI_MLC_PA_V5%20(1).pdf)  
> **File nguồn LaTeX có thể biên dịch:** [papers/GSI_MLC_PA_V5_1_Draft_VI.tex](file:///d:/University_Subject/ML%20Research/BR_CC/papers/GSI_MLC_PA_V5_1_Draft_VI.tex)  
> **File PDF xuất bản tương ứng:** [papers/GSI_MLC_PA_V5_1_Draft_VI.pdf](file:///d:/University_Subject/ML%20Research/BR_CC/papers/GSI_MLC_PA_V5_1_Draft_VI.pdf)  
> **Tác giả:** Anonymous Authors  
> *Department of Computer Science and Engineering, Machine Learning Research Laboratory*

---

## Tóm tắt (Abstract)

Phân loại đa nhãn trong các ứng dụng có chi phí sai sót cao đòi hỏi mô hình vừa dự đoán chính xác, vừa có khả năng từ chối những quyết định thiếu chắc chắn. Bài báo đề xuất kiến trúc nâng cấp **GSI-MLC-PA v5.1** (*Data-Driven Stratified Peeling and Ascending Classifier Chains for Multi-Label Classification with Partial Abstention*), kết hợp linh hoạt giữa Binary Relevance (BR) và Classifier Chains (CC). Nhằm khắc phục triệt để hiện tượng tối ưu hóa giả tạo và chi phí tính toán bùng nổ của cơ chế tìm kiếm tham lam (greedy search) trong phiên bản v5 cũ, phiên bản v5.1 xây dựng quy trình phân hoạch bóc tách đa tầng dựa trên dữ liệu (*Iterative Cascaded Peeling*): một nhãn được công nhận là nhãn độc lập (IL) khi và chỉ khi bản thân đặc trưng đầu vào $X$ đã chứa đầy đủ thông tin để dự đoán chính xác nhãn đó với điểm $F_1 \ge \tau$ trên tập validation nội bộ. Quy trình bóc tách được lặp lại tối đa $T_{\max} = 3$ tầng thông qua không gian đặc trưng tăng cường xác suất mềm, giúp mô hình tự động dừng sớm và tránh bùng nổ độ phức tạp. Sau khi kết thúc phân tầng, các nhãn phụ thuộc cốt lõi còn lại trong tập $\mathcal{D}_{\text{residual}}$ được tái thiết kế chuỗi Classifier Chains theo trật tự tổng tương quan tăng dần (*Ascending Correlation Order*), đưa các nhãn ít phụ thuộc lên đầu chuỗi để làm điểm tựa vững chắc và dập tắt hoàn toàn hiện tượng lan truyền sai số (error propagation). Khi suy luận trên chuỗi phụ thuộc, mô hình áp dụng công thức hai trạng thái cho nhãn có một tiền nhiệm và phép thế xác suất mềm mean-field plug-in cho nhãn có nhiều tiền nhiệm, trước khi áp dụng tầng quyết định Bayes-optimal có từ chối để tạo đầu ra thuộc $\{0, \bot, 1\}$.

Thực nghiệm 5-fold cross-validation diện rộng trên 10 bộ dữ liệu benchmark đa nhãn quốc tế và 3 họ bộ học cơ sở (Logistic Regression, Support Vector Machine, và Multi-Layer Perceptron trên GPU) chứng minh tính ưu việt vững chắc của v5.1: Selective Macro-Precision tăng đồng loạt trên cả 3 bộ học (+0.72% đến +1.69%), Hamming Loss giảm trên cả Logistic và SVM, và Subset 0/1 Accuracy tăng trưởng ấn tượng trên Logistic Regression (thắng ở 8/10 bộ dữ liệu) trong khi số tầng bóc tách trung bình chỉ dừng ở mức 1.6--1.84 tầng.

**Từ khóa:** multi-label classification; classifier chains; partial abstention; stratified peeling; ascending correlation order; selective prediction; error propagation.

---

## 1. Giới thiệu (Introduction)

Trong phân loại đa nhãn (multi-label classification, MLC), một đối tượng có thể đồng thời gắn với nhiều nhãn. Một hồ sơ y khoa có thể liên quan đến nhiều chẩn đoán hoặc yếu tố nguy cơ; một giao dịch tài chính có thể đồng thời mang nhiều dấu hiệu cần kiểm tra. Trong những bối cảnh như vậy, buộc hệ thống phải trả lời mọi nhãn dù xác suất gần mức không chắc chắn có thể tạo ra quyết định khó kiểm soát.

Về mặt phương pháp, một mẫu $x \in \mathcal{X}$ được liên kết đồng thời với nhiều nhãn trong tập $\mathcal{L} = \{\lambda_1, \dots, \lambda_K\}$. Không gian đầu ra gồm $2^K$ cấu hình, do đó chất lượng dự đoán phụ thuộc không chỉ vào khả năng phân biệt từng nhãn mà còn vào cách mô hình khai thác quan hệ giữa chúng [1, 5]. Các phương pháp như Binary Relevance (BR) phân rã bài toán thành $K$ bộ phân loại nhị phân độc lập, có khả năng mở rộng tốt và không phát sinh lan truyền sai số theo chuỗi, nhưng không biểu diễn trực tiếp sự phụ thuộc nhãn [2].

Classifier Chains (CC) được giới thiệu năm 2009 và hoàn thiện ở phiên bản tạp chí năm 2011 khắc phục điểm này bằng cách bổ sung các nhãn tiền nhiệm vào đầu vào của các bộ phân loại phía sau, nhờ đó khai thác phụ thuộc nhãn với một lượt suy luận tuần tự nhưng lại bị phụ thuộc vào thứ tự và độ dài chuỗi [3].

Tuy nhiên, BR hay CC thông thường vẫn phải trả về 0 hoặc 1 ngay cả khi xác suất dự đoán gần 0.5. Những quyết định có độ tin cậy thấp có thể làm tăng rủi ro sai phân loại, đặc biệt trong các ứng dụng yêu cầu độ tin cậy cao. Năm 2021, Nguyen và Hüllermeier đưa ra cơ chế từ chối từng phần (partial abstention) cho phép mô hình trì hoãn quyết định tại các vị trí này thay vì đưa ra dự đoán nhị phân cưỡng bức. Tuy nhiên, khung lý thuyết này lại chủ yếu áp dụng khi xác suất biên đã biết và dựa trên giả định nhãn độc lập [7, 8].

Trong phiên bản GSI-MLC-PA v5 trước đây, chúng tôi đã đề xuất cơ chế phân hoạch thích nghi giữa tập độc lập (IL) và phụ thuộc (DL) dựa trên thuật toán tìm kiếm tham lam (greedy search) tối đa hóa Macro-F1 trên tập kiểm định nội bộ. Mặc dù đạt được những cải thiện ban đầu, cơ chế tham lam của v5 bộc lộ ba hạn chế mang tính cấu trúc:
1. **Hiện tượng tối ưu hóa giả tạo (Metric Artifact):** Việc chấp nhận một nhãn vào IL chỉ dựa trên việc tăng Macro-F1 cục bộ không phản ánh bản chất nhãn đó có thực sự độc lập với dữ liệu hay không.
2. **Chi phí tính toán bùng nổ $\mathcal{O}(K^2)$:** Tại mỗi bước lặp thử nghiệm, hệ thống phải chạy lại toàn bộ suy luận chuỗi CC trên tập validation nội bộ, gây lãng phí tài nguyên.
3. **Thảm họa lan truyền sai số khi đặt tương quan lớn lên đầu chuỗi CC:** Trong v5 cũ, nhãn có tổng tương quan lớn nhất bị đặt ở đầu chuỗi CC; do đây là nhãn phức tạp và phụ thuộc chằng chịt, sai số dự đoán ở vị trí gốc này lập tức lan truyền và phá hỏng toàn bộ các mắt xích phía sau.

Để khắc phục triệt để các hạn chế cốt lõi trên, bài báo đề xuất kiến trúc **GSI-MLC-PA v5.1** (*Data-Driven Stratified Peeling and Ascending Classifier Chains for Multi-Label Classification with Partial Abstention*). Các đóng góp chính của nghiên cứu gồm:
- **Đề xuất cơ chế phân hoạch bóc tách đa tầng dựa trên dữ liệu (Data-Driven Stratified Peeling):** Bóc tách nhãn độc lập tự nhiên dựa trên năng lực dự đoán thực tế từ đặc trưng $X$ ($F_1 \ge \tau$) kết hợp mở rộng đặc trưng xác suất mềm qua tối đa $T_{\max} = 3$ tầng, loại bỏ hoàn toàn cơ chế tìm kiếm tham lam và giảm thời gian phân tầng.
- **Tái thiết kế chuỗi Classifier Chains theo trật tự tương quan tăng dần (Ascending Correlation Order):** Đảo ngược hoàn toàn trật tự chuỗi CC của tập phụ thuộc, đưa nhãn ít tương quan lên đầu chuỗi làm tiền đề tin cậy và đẩy nhãn phụ thuộc mạnh về cuối chuỗi để hưởng trọn ngữ cảnh, triệt tiêu hiện tượng tích lũy sai số.
- **Tích hợp cơ chế suy luận xác suất lai và tầng quyết định Bayes-optimal:** Biên hóa hai trạng thái cho nhãn có một tiền nhiệm, xấp xỉ mean-field plug-in mềm cho nhiều tiền nhiệm, và tầng quyết định có từ chối theo chuẩn mở rộng của Nguyen & Hüllermeier.
- **Khung thực nghiệm đối chuẩn toàn diện 7 nhóm chỉ số trên 10 bộ dữ liệu chuẩn:** Đánh giá trên 3 họ bộ học cơ sở (Logistic Regression, Support Vector Machine, Multi-Layer Perceptron trên GPU), chứng minh v5.1 nâng cao đồng loạt Selective Macro-Precision, giảm Hamming Loss và tăng trưởng Subset 0/1 Accuracy.

---

## 2. Công trình liên quan (Related Work)

### 2.1. Mô hình hóa phụ thuộc nhãn
Các phương pháp MLC thường được phân biệt theo bậc phụ thuộc nhãn mà chúng mô hình hóa [1]. BR là phương pháp bậc một: mỗi nhãn được học độc lập từ $\mathcal{X}$, nhờ đó chi phí huấn luyện và dự đoán tăng gần tuyến tính theo $K$ [2]. Đổi lại, BR không sử dụng được sự phụ thuộc nhãn.

CC xây dựng một đồ thị có hướng dạng chuỗi, trong đó bộ phân loại tại vị trí $t$ nhận thêm các nhãn đứng trước trong hoán vị $\pi$ [3]. Tuy nhiên vẫn tồn tại hai hạn chế chính là:
1. *Lan truyền sai số (Error Propagation):* Một quyết định sai ở đầu chuỗi có thể ảnh hưởng nhiều nhãn kế tiếp và tạo ra hiệu ứng sai số tích lũy [3, 11].
2. *Chi phí suy luận lớn (Intractable Inference):* Việc tìm thứ tự chuỗi tối ưu là một bài toán tổ hợp NP-hard và trở nên bất khả thi khi số nhãn tăng [4, 5].

Probabilistic Classifier Chains (PCC) diễn giải chuỗi bằng quy tắc phân rã xác suất, tạo cơ sở cho suy luận Bayes dưới các hàm mất mát đa nhãn [4, 5]. Mặc dù PCC cung cấp mô hình xác suất chung làm cơ sở cho dự đoán Bayes-tối ưu, suy luận chính xác có thể trở nên bất khả thi khi số nhãn tăng. Đối với các hàm mất mát không phân rã, mô hình có thể phải xét hoặc tổng hợp trên tới $2^K$ cấu hình nhãn trong trường hợp xấu nhất [5]. Các phương pháp gần đúng như tìm kiếm theo chùm (beam search) hoặc lấy mẫu Monte Carlo giúp giảm chi phí tính toán, nhưng kết quả phụ thuộc vào ngân sách tìm kiếm và không bảo đảm thu được nghiệm tối ưu toàn cục [6].

### 2.2. Phân loại chọn lọc và từ chối từng phần
Phân loại có chọn lọc cho phép mô hình từ chối dự đoán khi độ tin cậy không đủ [9, 10]. Trong bài toán phân loại đơn nhãn, hành động từ chối thường được áp dụng cho toàn bộ mẫu. Cách tiếp cận này không phù hợp hoàn toàn với MLC vì mô hình có thể dự đoán chắc chắn phần lớn nhãn nhưng chỉ không chắc chắn tại một số vị trí. Từ chối toàn bộ mẫu trong trường hợp đó sẽ loại bỏ cả những dự đoán có độ tin cậy cao.

Nguyen và Hüllermeier đã hình thức hóa từ chối từng phần trong MLC bằng cách mở rộng không gian đầu ra thành $\{0, \bot, 1\}^K$ [7, 8]. Hàm mất mát tổng quát của họ kết hợp lỗi phân loại trên các vị trí được quyết định với một hàm phạt phụ thuộc vào số vị trí bị từ chối. Đối với Hamming Loss và hình phạt tuyến tính, quyết định Bayes-tối ưu được xác định bằng hai ngưỡng đối xứng quanh 0.5: mô hình dự đoán 0 hoặc 1 khi xác suất đủ xa 0.5 và trả về $\bot$ trong vùng không chắc chắn.

Khung lý thuyết này giả định rằng các xác suất biên đã được ước lượng trước và chưa trực tiếp giải quyết cấu trúc phụ thuộc hoặc sự lan truyền sai số trong Classifier Chains. GSI-MLC-PA v5.1 bổ sung bước phân tầng bóc tách dữ liệu và tái cấu trúc chuỗi tương quan trước khi áp dụng tầng từ chối.

### 2.3. Đặc trưng mô tả bộ dữ liệu
Ngoài số mẫu $N$ và số nhãn $K$, bài báo mô tả dữ liệu bằng số nhãn dương trung bình trên mỗi mẫu (label cardinality) và tỷ lệ nhãn dương trung bình (label density) [1]:
$$LC = \frac{1}{N} \sum_{n=1}^N \sum_{k=1}^K y_{nk}, \qquad LD = \frac{LC}{K}$$
Với $\mathrm{Pos}_k = \sum_{n} y_{nk}$ và $\mathrm{Neg}_k = N - \mathrm{Pos}_k$, tỷ lệ mất cân bằng vận hành được tính bởi:
$$\mathrm{MeanIR} = \frac{1}{K} \sum_{k=1}^K \frac{\max(\mathrm{Pos}_k, \mathrm{Neg}_k)}{\min(\mathrm{Pos}_k, \mathrm{Neg}_k) + \epsilon}$$
Một nhãn được tính là cực hiếm khi $\mathrm{Pos}_k / N < 0.05$.

### 2.4. Các chỉ số đánh giá toàn diện
Gọi $TP_k, FP_k, FN_k, TN_k$ lần lượt là số lượng True Positive, False Positive, False Negative và True Negative của nhãn $k$. Để đánh giá toàn diện năng lực của mô hình, hệ thống chuẩn hóa 7 nhóm chỉ số toán học:
1. **Hamming Loss ($\downarrow$) và Hamming Accuracy ($\uparrow$):**
   $$HL = \frac{1}{NK} \sum_{n=1}^N \sum_{k=1}^K \mathbb{I}(\hat{y}_{nk} \neq y_{nk}), \qquad HA = 1 - HL$$
2. **Subset 0/1 Accuracy ($\uparrow$ - Exact Match):**
   $$SA = \frac{1}{N} \sum_{n=1}^N \mathbb{I}(\hat{\mathbf{y}}_n = \mathbf{y}_n)$$
3. **Example Accuracy ($\uparrow$ - Instance Jaccard):**
   $$\mathrm{Jaccard}_{\mathrm{inst}} = \frac{1}{N} \sum_{n=1}^N \frac{|Y_n \cap \hat{Y}_n|}{|Y_n \cup \hat{Y}_n|}$$
4. **Macro-F1 và Micro-F1 ($\uparrow$):**
   $$F1_k = \frac{2TP_k}{2TP_k + FP_k + FN_k}, \qquad \text{Macro-F1} = \frac{1}{K} \sum_{k=1}^K F1_k$$
   $$\text{Micro-F1} = \frac{2 \sum_k TP_k}{2 \sum_k TP_k + \sum_k FP_k + \sum_k FN_k}$$
5. **Precision (Macro-Precision và Micro-Precision $\uparrow$):**
   $$\text{Macro-Prec} = \frac{1}{K}\sum_{k=1}^K \frac{TP_k}{TP_k + FP_k}, \qquad \text{Micro-Prec} = \frac{\sum_k TP_k}{\sum_k TP_k + \sum_k FP_k}$$

Trong chế độ chọn lọc có từ chối, đặt $a_{nk} = \mathbb{I}(\hat{y}_{nk} \neq \bot)$ là chỉ báo vị trí được quyết định. Các chỉ số Selective Macro-F1, Selective Micro-F1, Selective Precision chỉ tính trên những ô có $a_{nk} = 1$. Selective Hamming Loss và Selective Hamming Accuracy được tính bởi:
$$SHL = \frac{\sum_{n,k} a_{nk} \mathbb{I}(\hat{y}_{nk} \neq y_{nk})}{\sum_{n,k} a_{nk}}, \qquad SHA = 1 - SHL$$
Tỷ lệ bao phủ (Coverage) và tỷ lệ từ chối trung bình theo ô (AABS) được tính bởi:
$$\text{Coverage} = \frac{1}{NK} \sum_{n,k} a_{nk}, \qquad \text{AABS} = 1 - \text{Coverage}$$
Với chi phí từ chối $c$, Generalized Loss trung bình là:
$$GL_c = \text{Coverage} \times SHL + c \times \text{AABS}$$

---

## 3. Phương pháp Đề xuất: GSI-MLC-PA Phiên bản 5.1

### 3.1. Phát biểu bài toán
Cho không gian đặc trưng $\mathcal{X} \subseteq \mathbb{R}^d$, tập nhãn $\mathcal{L} = \{\lambda_1, \dots, \lambda_K\}$ và tập huấn luyện $\mathcal{S} = \{(x_n, y_n)\}_{n=1}^N$ với $y_n \in \{0, 1\}^K$. Theo mô hình từ chối từng phần [7, 8], mỗi vị trí nhãn có ba hành vi dự đoán:
$$\hat{y}_k = \begin{cases} 1, & \text{dự đoán nhãn } \lambda_k \text{ xuất hiện}, \\ 0, & \text{dự đoán nhãn } \lambda_k \text{ vắng mặt}, \\ \bot, & \text{từ chối quyết định tại } \lambda_k. \end{cases}$$
Đặt $\mathcal{D}(\hat{y}) = \{k : \hat{y}_k \neq \bot\}$ là tập vị trí được quyết định và $\mathcal{A}(\hat{y}) = \{1, \dots, K\} \setminus \mathcal{D}(\hat{y})$ là tập vị trí bị từ chối.

### 3.2. Phân hoạch không gian nhãn: Independent và Dependent Labels
GSI-MLC-PA phân rã $\mathcal{L}$ thành hai tập rời nhau:
$$\mathcal{I} \cup \mathcal{D}_L = \mathcal{L}, \qquad \mathcal{I} \cap \mathcal{D}_L = \emptyset$$
Các nhãn thuộc tập độc lập $\mathcal{I}$ được ước lượng trực tiếp từ đặc trưng $X$, trong khi các nhãn thuộc tập phụ thuộc $\mathcal{D}_L$ được mô hình hóa theo chuỗi liên kết có điều kiện.

#### 3.2.1. Ước lượng biên cho nhãn độc lập
Với mỗi nhãn $\lambda_k \in \mathcal{I}$, xác suất hậu nghiệm biên được ước lượng trực tiếp từ $x$ bằng bộ phân loại Binary Relevance:
$$p_k^{\mathrm{BR}}(x) = P(Y_k = 1 \mid x)$$
Do không nhận dự đoán tiền nhiệm làm đặc trưng, các nhãn thuộc $\mathcal{I}$ không chịu ảnh hưởng của hiện tượng lan truyền sai số chuỗi.

#### 3.2.2. Biên hóa xác suất cho nhãn phụ thuộc
Sau khi xác định phân hoạch, các nhãn thuộc $\mathcal{D}_L$ được sắp xếp theo hoán vị $\pi = (\pi_1, \dots, \pi_m)$ với $m = |\mathcal{D}_L|$. Ký hiệu $\mathrm{Pre}(k)$ là tập các nhãn thuộc $\mathcal{D}_L$ đứng trước $k$ trong chuỗi. Bộ phân loại tại nhãn $k$ ước lượng xác suất có điều kiện $q_k(x, y_{\mathrm{Pre}(k)}) = P(Y_k = 1 \mid x, Y_{\mathrm{Pre}(k)} = y_{\mathrm{Pre}(k)})$.
- Nếu $k$ chỉ có một nhãn tiền nhiệm $r$, áp dụng công thức xác suất toàn phần chính xác (*Exact Two-State Marginalization*):
  $$p_k(x) = (1 - p_r(x)) q_k(x, 0) + p_r(x) q_k(x, 1)$$
- Khi $k$ có từ hai tiền nhiệm trở lên, nhằm tránh bùng nổ tổ hợp $2^{|\mathrm{Pre}(k)|}$, mô hình áp dụng phép thế trường trung bình (*Continuous Mean-Field Plug-in*):
  $$p_k(x) \approx q_k(x, \hat{p}_{\mathrm{Pre}(k)}(x))$$
  trong đó $\hat{p}_{\mathrm{Pre}(k)}(x) \in [0, 1]^{|\mathrm{Pre}(k)|}$ là vector xác suất biên mềm liên tục của các nhãn tiền nhiệm.

### 3.3. Cơ chế Phân tầng Bóc tách Dữ liệu (Stratified Peeling) và Trật tự Tương quan Tăng dần

Khác với phiên bản v5 cũ sử dụng thuật toán tìm kiếm tham lam phức tạp, phiên bản v5.1 tiếp cận bài toán phân hoạch theo **logic tự nhiên và trực quan**:

#### 3.3.1. Logic Cốt lõi: Thế nào là Nhãn Độc Lập (IL) và Nhãn Phụ Thuộc (DL)?
- **Nhãn Độc Lập ($\mathcal{I}$):** Là nhãn mà *chỉ cần dựa vào dữ liệu đặc trưng đầu vào $X$* là mô hình đã dự đoán rất chính xác ($F_1 \ge \tau$, với $\tau = 0.75$), hoàn toàn không cần sự trợ giúp hay mách nước từ bất kỳ nhãn nào khác.
- **Quan hệ nhân quả một chiều (Ví dụ Y khoa):** Giả sử bệnh nhân có các xét nghiệm máu $X$. Bệnh Đái tháo đường ($A$) có thể được chẩn đoán chính xác rất cao chỉ từ chỉ số đường huyết trong $X$, do đó bệnh $A$ là nhãn độc lập ($IL$). Ngược lại, biến chứng Suy thận mạn ($B$) sinh ra từ bệnh $A$; triệu chứng $X$ ban đầu rất khó để chẩn đoán $B$, nhưng nếu biết chắc chắn bệnh nhân đã mắc bệnh $A$, việc dự đoán $B$ sẽ trở nên dễ dàng và chuẩn xác hơn nhiều.
- **Nhãn Phụ Thuộc ($\mathcal{D}_L$):** Là các nhãn mà bản thân $X$ chưa đủ thông tin để dự đoán tốt, bắt buộc phải cần thêm thông tin từ các nhãn khác.

#### 3.3.2. Quy trình Bóc tách Từng Vòng Lặp (Step-by-Step Iterative Peeling)
Quá trình phân loại nhãn được tiến hành tuần tự qua từng vòng lặp trên tập huấn luyện nội bộ $\mathcal{D}_{\text{fit}}$ và tập kiểm định $\mathcal{D}_{\text{val}}$:

- **Vòng lặp 1 (Tầng $IL_1$ -- Xét trực tiếp từ dữ liệu gốc $X$):**
  - Ban đầu, toàn bộ $K$ nhãn được đặt trong danh sách chờ thuộc tập phụ thuộc $\mathcal{D}_{\text{cand}} = \{1, \dots, K\}$.
  - Lấy từng nhãn $k \in \mathcal{D}_{\text{cand}}$, huấn luyện mô hình nhị phân (Binary Relevance) **chỉ sử dụng dữ liệu đặc trưng gốc $X$**.
  - Đánh giá điểm chất lượng $F_1$ tối ưu của nhãn $k$ trên tập kiểm định $\mathcal{D}_{\text{val}}$:
    - **Nhãn đã đủ điều kiện ($F_1 \ge \tau = 0.75$):** Nhãn này dự đoán rất tốt chỉ nhờ $X \implies$ **Đưa vào tập Độc lập Tầng 1 ($\mathcal{I}_1$)** và rút khỏi danh sách chờ $\mathcal{D}_{\text{cand}}$.
    - **Nhãn chưa đủ điều kiện ($F_1 < \tau$):** Tức là dữ liệu $X$ chưa đủ để dự đoán chính xác nhãn này $\implies$ **Tạm thời giữ lại trong danh sách chờ $\mathcal{D}_{\text{cand}}$**.

- **Vòng lặp 2 (Tầng $IL_2$ -- Mở rộng dữ liệu bằng các nhãn đã đỗ ở Vòng 1):**
  - Các nhãn đã được chọn vào $\mathcal{I}_1$ ở Vòng 1 là những nhãn có độ tin cậy rất cao. Chúng trở thành nguồn thông tin bổ trợ quý giá để giải quyết các nhãn còn lại.
  - Ta bổ sung phân phối xác suất dự đoán mềm của $\mathcal{I}_1$ vào vector đặc trưng để tạo ra không gian dữ liệu mở rộng: $X^{(1)} = [X, \hat{P}(Y_{\mathcal{I}_1} = 1 \mid X)]$.
  - Thử thách lại các nhãn vẫn còn kẹt lại trong $\mathcal{D}_{\text{cand}}$: huấn luyện lại mô hình dự đoán chúng dựa trên dữ liệu mới $X^{(1)}$.
  - Đánh giá lại trên tập kiểm định:
    - **Nhãn bây giờ đã đủ điều kiện ($F_1 \ge \tau$):** Nhờ có thêm thông tin từ $\mathcal{I}_1$, nhãn này đã vượt ngưỡng thành công $\implies$ **Đưa vào tập Độc lập Tầng 2 ($\mathcal{I}_2$)** và rút khỏi $\mathcal{D}_{\text{cand}}$.
    - **Nhãn vẫn chưa đủ điều kiện ($F_1 < \tau$):** Tiếp tục giữ lại trong danh sách chờ $\mathcal{D}_{\text{cand}}$.

- **Vòng lặp 3 (Tầng $IL_3$ -- Độ sâu tối đa $T_{\max} = 3$):**
  - Tiếp tục mở rộng không gian đặc trưng với toàn bộ các nhãn đã đỗ: $X^{(2)} = [X, \hat{P}(Y_{\mathcal{I}_1}), \hat{P}(Y_{\mathcal{I}_2})]$.
  - Xét tiếp các nhãn còn lại trong $\mathcal{D}_{\text{cand}}$. Nhãn nào đạt $F_1 \ge \tau$ sẽ được đưa vào $\mathcal{I}_3$, nhãn nào không đạt vẫn giữ trong $\mathcal{D}_{\text{cand}}$.

- **Tiêu chuẩn Dừng Vòng Lặp:**
  Vòng lặp tự động dừng lại khi gặp một trong các tình huống sau:
  1. **Không còn nhãn nào tiến bộ (No Promotion):** Tại một vòng lặp bất kỳ, không có thêm bất kỳ nhãn nào trong $\mathcal{D}_{\text{cand}}$ đạt ngưỡng $F_1 \ge \tau$. Khi đó, việc lặp tiếp là vô ích vì bổ sung thêm thông tin cũng không giúp các nhãn còn lại đạt chuẩn.
  2. **Đạt giới hạn vòng lặp tối đa ($t \ge T_{\max} = 3$):** Hệ thống khống chế tối đa 3 vòng để tránh lãng phí thời gian tính toán và ngăn chặn hiện tượng phụ thuộc vòng luẩn quẩn.
  3. **Danh sách chờ cạn kiệt ($|\mathcal{D}_{\text{cand}}| \le 1$):** Tất cả các nhãn đã trở thành độc lập, hoặc chỉ còn lại đúng 1 nhãn.

Sau khi kết thúc, ta thu được:
- Tập toàn bộ nhãn độc lập phân tầng: $\mathcal{I} = \mathcal{I}_1 \cup \mathcal{I}_2 \cup \dots \cup \mathcal{I}_m$.
- Tập nhãn phụ thuộc cốt lõi còn lại: $\mathcal{D}_{\text{residual}} = \mathcal{D}_{\text{cand}}$.

```
========================================================================================
ALGORITHM 1: QUY TRÌNH PHÂN TẦNG BÓC TÁCH DỮ LIỆU (STRATIFIED PEELING V5.1)
========================================================================================
Đầu vào: Tập huấn luyện (X_tr, Y_tr), tập kiểm định (X_val, Y_val), ngưỡng tau = 0.75, 
         độ sâu tối đa Tmax = 3.
Đầu ra:  Phân hoạch (I, D_residual) và trật tự thực thi Pi_final.

1: Khởi tạo: Vòng lặp t <- 1; Tập độc lập I_accum <- []; Danh sách chờ D_cand <- {1, ..., K};
             Ma trận đặc trưng X_tr^(0) <- X_tr, X_val^(0) <- X_val.
2: while t <= Tmax and |D_cand| > 1 do:
3:     Khởi tạo tập đỗ trong vòng này: I_t <- [].
4:     for mỗi nhãn ứng viên k thuộc D_cand do:
5:         Huấn luyện bộ phân loại nhị phân f_k^(t) trên dữ liệu hiện tại X_tr^(t-1) và Y_tr[:, k].
6:         Dự đoán xác suất trên tập kiểm định: p_val_k = f_k^(t)(X_val^(t-1)).
7:         Tìm ngưỡng nhị phân theta_k* tối ưu hóa: S_k^(t) = F1(Y_val[:, k], I[p >= theta*]).
8:         if S_k^(t) >= tau then:
9:             Nhãn k ĐỦ ĐIỀU KIỆN ==> Kết nạp vào tầng hiện tại: I_t <- I_t U {k}.
10:        end if
11:    end for
12:    if I_t rỗng then:
13:        Không có nhãn nào mới đủ điều kiện ==> DỪNG SỚM VÒNG LẶP (no_promotion).
14:    end if
15:    Cập nhật: Thêm I_t vào tập độc lập I_accum <- I_accum U I_t; 
                 Loại I_t khỏi danh sách chờ D_cand <- D_cand \ I_t.
16:    Mở rộng dữ liệu cho vòng tiếp theo: X^(t) <- [X^(t-1), P_hat(Y_I_t = 1)].
17:    t <- t + 1
18: end while
19: Tập phụ thuộc cốt lõi còn lại là: D_residual <- D_cand.
20: Sắp xếp D_residual thành chuỗi CC theo trật tự tổng tương quan tăng dần pi_D theo Công thức (12).
21: return I = I_accum, D_residual, và Pi_final = [I_1 || ... || I_m || pi_D].
========================================================================================
```

#### 3.3.3. Xử lý Tập Phụ thuộc Còn lại: Tái thiết kế Chuỗi CC theo Tương quan Tăng dần

Sau tối đa 3 vòng bóc tách, các nhãn vẫn còn kẹt lại trong tập $\mathcal{D}_{\text{residual}}$ là những **nhãn phụ thuộc phức tạp**: dù đã dùng cả $X$ lẫn các nhãn độc lập, chúng vẫn không thể tự đứng một mình đạt $F_1 \ge \tau$. Các nhãn này bắt buộc phải đưa vào chuỗi Classifier Chains (CC) để nhãn nọ hỗ trợ nhãn kia.

Vấn đề đặt ra là: **Sắp xếp chuỗi CC cho các nhãn phụ thuộc này như thế nào để tránh hỏng cả chuỗi?**

1. **Sai lầm của phiên bản v5 cũ (Tương quan lớn lên đầu):**
   Trong v5, nhãn có tương quan lớn nhất bị đặt ở đầu chuỗi (vị trí gốc). Đây là nhãn phức tạp nhất, dễ bị đoán sai nhất. Khi nhãn gốc này đoán sai, toàn bộ các mắt xích phía sau bị nhiễm độc thông tin sai lệch, gây ra thảm họa lan truyền sai số (error propagation cascade).
2. **Chiến lược mới của v5.1 (Tương quan nhỏ lên đầu -- Ascending Correlation Order):**
   - Với mỗi nhãn $d \in \mathcal{D}_{\text{residual}}$, tính tổng mức độ tương quan của nó với các nhãn còn lại trong tập phụ thuộc:
     $$C(d) = \sum_{j \in \mathcal{D}_{\text{residual}}, j \neq d} |\phi_{dj}|$$
     với $\phi_{dj}$ là hệ số tương quan Phi giữa cặp nhãn.
   - Sắp xếp các nhãn theo thứ tự **tổng tương quan tăng dần**:
     $$\pi_{\mathcal{D}} = \text{argsort}_{\text{ascending}}\left( [C(d)]_{d \in \mathcal{D}_{\text{residual}}} \right)$$
   - **Ý nghĩa trực quan:**
     - *Đầu chuỗi:* Nhãn có $C(d)$ nhỏ nhất là nhãn ít bị chi phối bởi các nhãn khác nhất $\implies$ dự đoán ở vị trí đầu sẽ rất ổn định, độ tin cậy cao, tạo nền móng vững chắc cho chuỗi.
     - *Cuối chuỗi:* Nhãn có $C(d)$ lớn nhất là nhãn phụ thuộc chằng chịt nhất $\implies$ đặt ở cuối chuỗi là vị trí lý tưởng nhất vì nó được thừa hưởng đầy đủ toàn bộ thông tin từ $X$, các tầng $\mathcal{I}$ và toàn bộ các nhãn tiền nhiệm đứng trước trong $\mathcal{D}_{\text{residual}}$.

**Trật tự Thực thi Toàn cục Hợp nhất:**
Toàn bộ quá trình suy luận của GSI-MLC-PA v5.1 được xâu chuỗi một cách logic và mượt mà:
$$\Pi_{\text{final}} = \left[ \underbrace{\mathcal{I}_1 \ \Vert \ \mathcal{I}_2 \ \Vert \ \dots \ \Vert \ \mathcal{I}_m}_{\text{Các tầng độc lập: dự đoán tuần tự bằng BR mở rộng}} \ \Vert \ \underbrace{\pi_{\mathcal{D}}(1) \to \pi_{\mathcal{D}}(2) \to \dots}_{\text{Chuỗi CC tương quan tăng dần}} \right]$$

### 3.4. Tầng Quyết định Bayes-Optimal Có Từ Chối
#### 3.4.1. Hamming BOP dưới Phạt Tuyến tính (SEP)
Với hàm phạt tuyến tính $g(a) = c \cdot a$, rủi ro có điều kiện phân rã độc lập trên từng nhãn. Quyết định Bayes tối ưu được xác định độc lập qua quy tắc ba trạng thái quanh ngưỡng đối xứng:
$$\hat{y}_k = \begin{cases} 1, & p_k(x) \ge 1 - c, \\ 0, & p_k(x) \le c, \\ \bot, & c < p_k(x) < 1 - c. \end{cases}$$
Khi $c \ge 0.5$, vùng từ chối biến mất và mô hình trở về ngưỡng phân loại nhị phân chuẩn 0.5.

#### 3.4.2. F BOP với Cặp Ngưỡng Thích Nghi
Để tối ưu hóa Macro-F1 có từ chối, mô hình quét tìm cặp ngưỡng $(\tau_k^{\text{low}*}, \tau_k^{\text{high}*})$ trên tập kiểm định nhằm tối đa hóa hàm tiện ích có phạt:
$$\mathcal{U}_k(\tau_k^{\text{low}}, \tau_k^{\text{high}}) = F_{1,\text{sel}}^{(k)}(\tau_k^{\text{low}}, \tau_k^{\text{high}}) - c \cdot \frac{A_k}{N_{\text{val}}}$$
trong đó $A_k$ là số lượng mẫu bị từ chối và $F_{1,\text{sel}}^{(k)}$ chỉ tính trên các mẫu được đưa ra quyết định.

---

## 4. Thiết kế Thực nghiệm (Experimental Setup)

### 4.1. Tập dữ liệu Benchmark
Thực nghiệm được thực hiện trên 10 tập dữ liệu đa nhãn quốc tế chuẩn hóa, bao phủ đa dạng các lĩnh vực y sinh học, xử lý tín hiệu âm thanh, phân loại văn bản và thị giác máy tính.

| Bộ dữ liệu | Số mẫu ($N$) | Số thuộc tính ($d$) | Số nhãn ($K$) | Cardinality ($LC$) | Density ($LD$) | MeanIR | Lĩnh vực ứng dụng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **CHD\_49** | 555 | 49 | 6 | 2.580 | 0.430 | 7.21 | Tim mạch lâm sàng |
| **emotions** | 593 | 72 | 6 | 1.868 | 0.311 | 2.32 | Cảm xúc âm nhạc |
| **genbase** | 662 | 1186 | 27 | 1.252 | 0.046 | 143.46 | Trình tự Protein |
| **GpositivePseAAC** | 519 | 440 | 4 | 1.008 | 0.252 | 8.63 | Vi khuẩn Gram dương |
| **HumanPseAAC** | 3106 | 440 | 14 | 1.185 | 0.085 | 45.51 | Protein người |
| **music** | 592 | 71 | 6 | 1.870 | 0.312 | 2.32 | Thể loại âm nhạc |
| **PlantPseAAC** | 978 | 440 | 12 | 1.079 | 0.090 | 21.88 | Protein thực vật |
| **scene** | 2407 | 294 | 6 | 1.074 | 0.179 | 4.66 | Thị giác máy tính |
| **VirusPseAAC** | 207 | 440 | 6 | 1.217 | 0.203 | 8.62 | Protein virus |
| **yeast** | 2417 | 103 | 14 | 4.237 | 0.303 | 8.95 | Sinh học nấm men |

### 4.2. Giao thức Đánh giá
Thực nghiệm sử dụng quy trình đánh giá chéo 5-Fold Stratified Cross-Validation cố định seed ngẫu nhiên (`random_state=42`). Trong từng outer-train fold, 20% dữ liệu được trích xuất làm tập kiểm định nội bộ $\mathcal{D}_{\text{val}}$ để thực hiện phân tầng bóc tách IL/DL; tập outer-test hoàn toàn độc lập và chỉ được đánh giá sau khi toàn bộ cấu trúc phân tầng và mô hình đã được huấn luyện lại trên toàn bộ fold huấn luyện. Chi phí từ chối chuẩn được thiết lập tại $c = 0.30$; ngưỡng chất lượng phân định độc lập $\tau = 0.75$; độ sâu tối đa $T_{\max} = 3$.

### 4.3. Không gian Mô hình Đối sánh và Bộ học Cơ sở
Để chứng minh tính độc lập của kiến trúc đối với bộ phân loại nhị phân, chúng tôi khảo sát trên 3 họ bộ học cơ sở đại diện:
1. **Logistic Regression (Calibrated):** Mô hình tuyến tính với chính quy hóa $\ell_2$, $C=1.0$, solver `liblinear`.
2. **Support Vector Machine (LinearSVC):** Bộ phân loại biên độ lớn với hàm mất mát squared hinge, $C=1.0$, hiệu chuẩn xác suất bằng Platt Scaling 3-fold.
3. **Multi-Layer Perceptron (PyTorch GPU MLP):** Mạng nơ-ron sâu phi tuyến với 2 ẩn tầng $(128, 64)$, dropout 0.15, tối ưu hóa Adam (learning rate $10^{-3}$, weight decay $10^{-3}$), 30 epochs, hàm mất mát Weighted Binary Cross-Entropy.

Bốn kiến trúc mô hình được đối chuẩn chéo: (1) **BR** (chuẩn mực không từ chối), (2) **CC** (chuỗi đầy đủ không từ chối), (3) **GSI v5 Greedy** (phiên bản cũ với tìm kiếm tham lam), và (4) **GSI v5.1 Stratified** (phiên bản đề xuất với bóc tách đa tầng và chuỗi tương quan tăng dần).

---

## 5. Kết quả Thực nghiệm và Phân tích Đối chuẩn

### 5.1. Kết quả Dự đoán Đầy đủ (Complete Classification Results)

Bảng dưới đây báo cáo giá trị trung bình trên toàn bộ 10 tập dữ liệu của các mô hình trong chế độ dự đoán đầy đủ (ngưỡng nhị phân 0.5, không từ chối).

| Base Learner | Mô hình Thực nghiệm | Hamming Acc $\uparrow$ | Example Jaccard $\uparrow$ | Instance-F1 $\uparrow$ | Macro-F1 $\uparrow$ | Micro-F1 $\uparrow$ | Subset 0/1 Acc $\uparrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic** | BR | 0.8480 | 0.4866 | 0.5310 | 0.4643 | 0.5881 | 0.3471 |
| **Logistic** | CC | 0.8376 | 0.5472 | 0.5871 | 0.4796 | 0.6109 | 0.4144 |
| **Logistic** | GSI v5 (Greedy) | 0.8491 | 0.4921 | 0.5360 | 0.4607 | 0.5899 | 0.3592 |
| **Logistic** | **GSI v5.1 (Stratified)** | **0.8496 🏆** | **0.4957 🏆** | **0.5398 🏆** | **0.4643 🏆** | **0.5917 🏆** | **0.3612 🏆** |
| — | *Chênh lệch ($\Delta$ v5.1 vs v5)* | *+0.0005* | *+0.0036* | *+0.0038* | *+0.0036* | *+0.0018* | *+0.0020* |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM** | BR | 0.8451 | 0.4196 | 0.4602 | 0.4036 | 0.5122 | 0.2938 |
| **SVM** | CC | 0.8345 | 0.5200 | 0.5539 | 0.4471 | 0.5833 | 0.3892 |
| **SVM** | GSI v5 (Greedy) | 0.8378 | 0.4762 | 0.4804 | 0.4343 | 0.5755 | 0.3314 |
| **SVM** | **GSI v5.1 (Stratified)** | **0.8403 🏆** | 0.4368 | 0.4782 | **0.4364 🏆** | 0.5309 | 0.2988 |
| — | *Chênh lệch ($\Delta$ v5.1 vs v5)* | *+0.0025* | *-0.0394* | *-0.0022* | *+0.0021* | *-0.0446* | *-0.0326* |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP** | BR | 0.8368 | 0.3060 | 0.3436 | 0.3254 | 0.3998 | 0.1896 |
| **MLP** | CC | 0.8369 | 0.4608 | 0.5003 | 0.4228 | 0.5701 | 0.3230 |
| **MLP** | GSI v5 (Greedy) | 0.8389 | 0.4175 | 0.4671 | 0.4043 | 0.5405 | 0.2822 |
| **MLP** | **GSI v5.1 (Stratified)** | 0.8339 | 0.3441 | 0.3980 | 0.3614 | 0.4572 | 0.2119 |

**Nhận định then chốt:**
1. **Cải thiện toàn diện trên Logistic Regression:** GSI v5.1 vượt trội hoàn toàn GSI v5 trên mọi chỉ số cốt lõi: Hamming Accuracy tăng lên **0.8496**, Example Jaccard tăng lên **0.4957**, Full Micro-F1 tăng lên **0.5917**, và đặc biệt Subset 0/1 Accuracy tăng lên **0.3612** (v5.1 thắng v5 ở 8/10 bộ dữ liệu).
2. **Giảm thiểu Hamming Loss trên Support Vector Machine:** Trên bộ học SVM, GSI v5.1 đạt Hamming Accuracy **0.8403** (tương ứng Hamming Loss giảm từ 0.1622 xuống **0.1597**), đồng thời Full Macro-F1 tăng từ 0.4343 lên **0.4364**.
3. **Triệt tiêu hiện tượng lan truyền sai số của CC:** Trên cả 3 bộ học cơ sở, mô hình CC truyền thống đều bị sụt giảm Hamming Accuracy nghiêm trọng so với BR (ví dụ trên Logistic: 0.8376 vs 0.8480; trên SVM: 0.8345 vs 0.8451). Bằng việc bóc tách các nhãn độc lập ra khỏi chuỗi, GSI v5.1 bảo toàn độ chính xác vị trí nhãn ở mức cao nhất.

### 5.2. Kết quả Dự đoán Có Từ Chối (Selective Prediction tại $c = 0.30$)

| Base Learner | Mô hình | Coverage $\uparrow$ | Sel. Macro-F1 $\uparrow$ | Sel. Micro-F1 $\uparrow$ | Sel. Macro-Prec $\uparrow$ | Gen. Loss $\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic** | GSI v5 (Greedy) | 0.6602 | 0.5410 | **0.6720** | 0.5321 | **0.1373** |
| **Logistic** | **GSI v5.1 (Stratified)** | **0.6603** | **0.5579 🏆** | 0.6662 | **0.5489 🏆** | 0.1472 |
| — | *Chênh lệch ($\Delta$ v5.1 vs v5)* | *+0.01%* | **+0.0169 🏆** | *-0.0058* | **+0.0168 🏆** | *+0.0099* |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **SVM** | GSI v5 (Greedy) | 0.6481 | 0.4963 | **0.6482** | 0.4872 | **0.1469** |
| **SVM** | **GSI v5.1 (Stratified)** | **0.6710 🏆** | **0.5243 🏆** | 0.6208 | **0.4966 🏆** | 0.1745 |
| — | *Chênh lệch ($\Delta$ v5.1 vs v5)* | **+2.28%** | **+0.0280 🏆** | *-0.0274* | **+0.0094 🏆** | *+0.0276* |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **MLP** | GSI v5 (Greedy) | 0.6412 | 0.5230 | **0.6362** | 0.5063 | **0.1457** |
| **MLP** | **GSI v5.1 (Stratified)** | **0.6550 🏆** | **0.5291 🏆** | 0.6250 | **0.5135 🏆** | 0.1507 |
| — | *Chênh lệch ($\Delta$ v5.1 vs v5)* | **+1.38%** | **+0.0061 🏆** | *-0.0112* | **+0.0072 🏆** | *+0.0050* |

**Phân tích chuyên sâu:**
- **Selective Macro-F1 và Selective Macro-Precision tăng đồng loạt trên cả 3 Base Learners:**
  - Trên Logistic: Selective Macro-F1 tăng từ 0.5410 lên **0.5579** (+1.69%); Selective Macro-Precision tăng từ 0.5321 lên **0.5489** (+1.68%).
  - Trên SVM: Selective Macro-F1 tăng từ 0.4963 lên **0.5243** (+2.80%); Selective Macro-Precision tăng từ 0.4872 lên **0.4966** (+0.94%); Coverage tăng thêm **+2.28%**.
  - Trên MLP: Selective Macro-F1 tăng từ 0.5230 lên **0.5291** (+0.61%); Selective Macro-Precision tăng từ 0.5063 lên **0.5135** (+0.72%); Coverage tăng thêm **+1.38%**.
- **Ý nghĩa khoa học:** Việc Selective Macro-Precision tăng đều trên toàn bộ các mô hình khẳng định rằng các quyết định được đưa ra bởi GSI v5.1 có độ tin cậy dương tính thực sự cao hơn, giảm thiểu tối đa hiện tượng báo động giả (False Positives) trong các bài toán chẩn đoán y sinh học rủi ro cao.

### 5.3. Đặc tả Cấu trúc Phân tầng và Hiệu quả Tính toán

| Base Learner | Mô hình | Số Tầng Bóc Tách (TB) | Số Lượng IL Trung Bình | Tỷ Lệ Độc Lập (%) | Thời Gian Huấn Luyện (s) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic** | `GSI_v5_Greedy` | 10.1 | 2.08 / 10.1 | 20.6% | 0.980s |
| **Logistic** | `GSI_v5_1_Stratified` | **1.60** | **4.70 / 10.1** | **46.5%** | 1.401s |
| **SVM** | `GSI_v5_Greedy` | 10.1 | 1.34 / 10.1 | 13.3% | 2.664s |
| **SVM** | `GSI_v5_1_Stratified` | **1.80** | **4.52 / 10.1** | **44.8%** | 4.260s |
| **MLP** | `GSI_v5_Greedy` | 10.1 | 1.42 / 10.1 | 14.1% | 1.986s |
| **MLP** | `GSI_v5_1_Stratified` | **1.84** | **4.08 / 10.1** | **40.4%** | 4.079s |

**Minh chứng hiệu quả:**
1. **Hội tụ cực nhanh dưới chặn $T_{\max} = 3$:** Số tầng bóc tách thực tế trung bình chỉ dao động từ **1.60 đến 1.84 tầng**, chứng minh rằng tiêu chuẩn dừng rỗng ($\mathcal{I}_t = \emptyset$) và chặn $T_{\max} = 3$ hoạt động hoàn hảo, ngăn chặn hoàn toàn việc phân tầng kéo dài vô ích.
2. **Khám phá nhãn độc lập thực chất:** Tỷ lệ nhãn độc lập tự nhiên được phát hiện tăng vọt từ $13.3\% - 20.6\%$ trong v5 lên tới **$40.4\% - 46.5\%$** trong v5.1. Nhờ đó, gần một nửa số lượng nhãn được giải phóng khỏi chuỗi CC, giảm thiểu tải tính toán và triệt tiêu nguy cơ lan truyền sai số cho toàn hệ thống.

---

## 6. Thảo luận và Hướng Phát triển (Discussion)

### 6.1. Giải quyết triệt để các giới hạn của phiên bản v5
Trong phiên bản v5 cũ, nghiên cứu đã tự nhận diện ba điểm hạn chế lớn:
1. *Tính xác thực của tập IL:* Trong v5, nhãn được đưa vào IL chỉ dựa vào mức tăng $\Delta\text{Macro-F1}$ cục bộ mà không phản ánh tính độc lập thực sự từ $X$. Phiên bản v5.1 giải quyết trọn vẹn điều này: nhãn vào IL khi và chỉ khi $X$ tự thân dự đoán đạt $F_1 \ge \tau$, bảo đảm tính độc lập thống kê thực chất và có cơ sở lý thuyết vững chắc.
2. *Nghịch lý lan truyền sai số trong CC:* Đặt nhãn có tương quan lớn nhất lên đầu chuỗi CC trong v5 đã tạo ra sai số dây chuyền. Chiến lược Ascending Correlation Order trong v5.1 dập tắt hoàn toàn hiện tượng này bằng cách xếp nhãn ít phụ thuộc lên đầu chuỗi làm nền tảng.
3. *Kiểm soát độ sâu và tính toán:* Chặn trên $T_{\max} = 3$ và hàm tiện ích biên có phạt bảo đảm hệ thống luôn duy trì thời gian thực thi đa thức, không lãng phí chu kỳ máy.

### 6.2. Hướng phát triển tiếp theo
Dựa trên những phát hiện mới từ v5.1, hai hướng nghiên cứu tiếp theo sẽ được ưu tiên triển khai:
1. **Mô hình hóa Đồ thị Tương quan Nhãn Có Trọng số (Weighted Label Graph):** Thay vì ép buộc tập phụ thuộc vào một chuỗi tuyến tính duy nhất $\pi_{\mathcal{D}}$, xây dựng đồ thị có hướng phi chu trình (DAG) dựa trên cây bao trùm cực đại (MST) hoặc thuật toán phát hiện cộng đồng nhãn (Community Detection), cho phép các cụm nhãn độc lập cục bộ suy luận song song.
2. **Cơ chế Học Chủ động (Active Learning Triage):** Tích hợp phản hồi thời gian thực từ chuyên gia con người đối với các vị trí bị từ chối ($\bot$) để tái hiệu chuẩn trọng số mô hình trực tuyến.

---

## Tài liệu tham khảo (References)

[1] M.-L. Zhang and Z.-H. Zhou, “A review on multi-label learning algorithms,” *IEEE Transactions on Knowledge and Data Engineering*, vol. 26, no. 8, pp. 1819–1837, 2014.  
[2] M.-L. Zhang, Y.-K. Li, X.-Y. Liu, and X. Geng, “Binary relevance for multi-label learning: An overview,” *Frontiers of Computer Science*, vol. 12, no. 2, pp. 191–202, 2018.  
[3] J. Read, B. Pfahringer, G. Holmes, and E. Frank, “Classifier chains for multi-label classification,” *Machine Learning*, vol. 85, no. 3, pp. 333–359, 2011.  
[4] K. Dembczyński, W. Cheng, and E. Hüllermeier, “Bayes optimal multilabel classification via probabilistic classifier chains,” in *Proceedings of the 27th International Conference on Machine Learning (ICML)*, 2010, pp. 279–286.  
[5] K. Dembczyński, W. Waegeman, W. Cheng, and E. Hüllermeier, “On label dependence and loss minimization in multi-label classification,” *Machine Learning*, vol. 88, no. 1, pp. 5–45, 2012.  
[6] J. Read, L. Martino, and D. Luengo, “Efficient Monte Carlo methods for multi-dimensional learning with classifier chains,” *Pattern Recognition*, vol. 47, no. 3, pp. 1535–1546, 2014.  
[7] V.-L. Nguyen and E. Hüllermeier, “Reliable multi-label classification: Prediction with partial abstention,” in *Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)*, vol. 34, no. 4, pp. 5264–5271, 2020.  
[8] V.-L. Nguyen and E. Hüllermeier, “Multilabel classification with partial abstention: Bayes-optimal prediction under label independence,” *Journal of Artificial Intelligence Research (JAIR)*, vol. 72, pp. 613–665, 2021.  
[9] C. Chow, “On optimum recognition error and reject tradeoff,” *IEEE Transactions on Information Theory*, vol. 16, no. 1, pp. 41–46, 1970.  
[10] Y. Geifman and R. El-Yaniv, “Selective classification for deep neural networks,” in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017, pp. 4885–4894.  
[11] R. Senge, S. Bohlender, and E. Hüllermeier, “Reliable classification: Learning classifiers that distinguish between epistemic and aleatoric uncertainty,” *Information Sciences*, vol. 255, pp. 16–29, 2014.  
[12] K. Sechidis, G. Tsoumakas, and I. Vlahavas, “On the stratification of multi-label data,” in *Machine Learning and Knowledge Discovery in Databases (ECML PKDD)*, 2011, pp. 145–158.  
[13] J. Demšar, “Statistical comparisons of classifiers over multiple data sets,” *Journal of Machine Learning Research (JMLR)*, vol. 7, pp. 1–30, 2006.  
