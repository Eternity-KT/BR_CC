# BẢN ĐẶC TẢ KỸ THUẬT & KẾ HOẠCH THỰC NGHIỆM CHI TIẾT: PHIÊN BẢN V5.1
## CƠ CHẾ PHÂN HOẠCH NHÃN ĐỘC LẬP / PHỤ THUỘC (IL/DL) MỚI THEO DỮ LIỆU & TỐI ƯU HÓA CHUỖI CC

> **Tài liệu tham chiếu:** [meeting_summary.md](file:///d:/University_Subject/ML%20Research/BR_CC/meeting_summary.md)  
> **Phiên bản:** 5.1 (Data-Driven Stratified Partitioning & Ascending-Correlation CC)  
> **Kế thừa từ:** [spec/spec_v5.md](file:///d:/University_Subject/ML%20Research/BR_CC/spec/spec_v5.md)  
> **Trạng thái:** Đặc tả kỹ thuật & Kế hoạch thực nghiệm chi tiết (Architecture & Empirical Plan)  
> **Mục tiêu:** Loại bỏ hoàn toàn cơ chế tìm kiếm tham lam (greedy search) dựa trên Macro-F1 của v5; xây dựng quy trình phân hoạch nhãn đa tầng dựa trên năng lực dự đoán thực tế từ đặc trưng dữ liệu ($X$); đảo ngược trật tự chuỗi CC cho tập nhãn phụ thuộc ($DL$); tích hợp hàm phạt phức tạp tính toán và hoàn thiện khung thực nghiệm khoa học 10 datasets.

---

## MỤC LỤC TỔNG QUAN

1. [TỔNG QUAN BỐI CẢNH VÀ ĐỘNG LỰC KHOA HỌC](#1-tổng-quan-bối-cảnh-và-động-lực-khoa-học)
   - 1.1. Hạn chế cốt tử của cơ chế Greedy Macro-F1 Selection trong v5
   - 1.2. Động lực của cơ chế phân hoạch tự nhiên dựa trên dữ liệu (Data-Driven Natural Partitioning)
   - 1.3. Tính định hướng nhân quả và triết lý phân tách $X \to Y_{\mathcal{I}} \to Y_{\mathcal{D}}$
2. [CƠ SỞ TOÁN HỌC & GIẢI THUẬT PHÂN HOẠCH NHÃN MỚI (V5.1 CORE ALGORITHM)](#2-cơ-sở-toán-học--giải-thuật-phân-hoạch-nhãn-mới-v51-core-algorithm)
   - 2.1. Định nghĩa toán học của Nhãn Độc Lập Nội Sinh ($IL_1$)
   - 2.2. Cơ chế bóc tách đa tầng (Iterative Cascaded Peeling: $IL_1 \to IL_2 \to \dots \to IL_m$)
   - 2.3. Không gian biểu diễn xác xuất chuyển tiếp mềm (Soft Probability Augmentation)
   - 2.4. Ngưỡng phân định hiệu năng $\tau$ và cơ chế ngưỡng thích nghi
3. [TÁI THIẾT KẾ CHUỖI CLASSIFIER CHAIN CHO TẬP PHỤ THUỘC $DL$](#3-tái-thiết-kế-chuỗi-classifier-chain-cho-tập-phụ-thuộc-dl)
   - 3.1. Nghịch lý tích lũy sai số và Luận điểm "Tương quan nhỏ lên đầu"
   - 3.2. Giải thuật sắp xếp trật tự chuỗi theo tổng tương quan tăng dần (Ascending Correlation Order)
   - 3.3. So sánh đối chuẩn các chiến lược trật tự CC: Ascending vs. Descending vs. Direct Parents
4. [HÀM PHẠT PHỨC TẠP TÍNH TOÁN & ĐIỀU KIỆN DỪNG TỰ ĐỘNG](#4-hàm-phạt-phức-tạp-tính-toán--điều-kiện-dừng-tự-động)
   - 4.1. Đánh giá chi phí tính toán $\mathcal{O}(\cdot)$ của phân tầng lặp
   - 4.2. Thiết kế hàm phạt phức tạp (Complexity / Time Penalty Function)
   - 4.3. Tiêu chuẩn dừng tối ưu (Optimal Stopping Criteria)
5. [CƠ CHẾ PHÒNG CHỐNG SỤP ĐỔ MÔ HÌNH (MODEL COLLAPSE MITIGATION)](#5-cơ-chế-phòng-chống-sụp-đổ-mô-hình-model-collapse-mitigation)
   - 5.1. Nhận diện hiện tượng sụp đổ trên các nhãn mất cân bằng nặng (Severe Imbalance Collapse)
   - 5.2. Tích hợp Asymmetric Loss / Focal Loss cho tầng biểu diễn sâu
   - 5.3. Hiệu chuẩn xác suất tự động (Automated Platt Scaling) trên từng tầng
6. [NGHIÊN CỨU CẤU TRÚC ĐỒ THỊ NHÓM NHÃN (LABEL GRAPH STRUCTURE - FUTURE EXTENSION)](#6-nghiên-cứu-cấu-trúc-đồ-thị-nhóm-nhãn-label-graph-structure---future-extension)
   - 6.1. Xây dựng đồ thị phụ thuộc nhãn có trọng số (Weighted Label Dependency Graph)
   - 6.2. Phát hiện cộng đồng nhãn độc lập và cụm nhãn phụ thuộc (Community Detection)
7. [ĐẶC TẢ KIẾN TRÚC HỆ THỐNG & API CONTRACTS](#7-đặc-tả-kiến-trúc-hệ-thống--api-contracts)
   - 7.1. Cấu trúc thư mục mã nguồn
   - 7.2. Module phân hoạch mới: `src/selection/stratified_partition.py`
   - 7.3. Cải tiến mô hình chính: `src/models/gsi_mlc_pa.py` (v5.1 mode)
   - 7.4. Data Schema kiểm toán phân tầng (Stratified Audit Schema)
8. [KẾ HOẠCH THỰC NGHIỆM BENCHMARK 10 DATASETS TOÀN DIỆN](#8-kế-hoạch-thực-nghiệm-benchmark-10-datasets-toàn-diện)
   - 8.1. Danh mục 10 tập dữ liệu và đặc tính phân phối
   - 8.2. Ma trận cấu hình thực nghiệm và đối chuẩn (Benchmark Matrix)
   - 8.3. Hệ thống chỉ số đánh giá (Evaluation Metrics)
   - 8.4. Thiết kế các thí nghiệm bóc tách (Ablation Studies)
9. [LỘ TRÌNH TRIỂN KHAI TỪNG BƯỚC (STEP-BY-STEP ACTION PLAN)](#9-lộ-trình-triển-khai-từng-bước-step-by-step-action-plan)

---

## 1. TỔNG QUAN BỐI CẢNH VÀ ĐỘNG LỰC KHOA HỌC

### 1.1. Hạn chế cốt tử của cơ chế Greedy Macro-F1 Selection trong v5

Trong các phiên bản trước (từ v3 đến v5), thuật toán **GSI-MLC-PA** sử dụng cơ chế tìm kiếm tham lam (Greedy Selection) để xác định phân hoạch giữa hai tập nhãn:
- Tập nhãn độc lập: $\mathcal{I}$ (Independent Labels - sử dụng Binary Relevance trực tiếp $P(Y_k = 1 \mid X)$).
- Tập nhãn phụ thuộc: $\mathcal{D}$ (Dependent Labels - sử dụng Classifier Chains $P(Y_k = 1 \mid X, Y_{\text{predecessors}})$).

Thuật toán cũ hoạt động như sau:
1. Khởi tạo $\mathcal{I} = \emptyset, \mathcal{D} = \{1, \dots, K\}$.
2. Duyệt qua từng nhãn $k$ theo một thứ tự xác định trước.
3. Giả định chuyển $k$ từ $\mathcal{D}$ sang $\mathcal{I}$, sau đó tính lại toàn bộ xác suất biên của tất cả các nhãn còn lại thông qua phép xấp xỉ Mean-Field trên chuỗi CC.
4. Đánh giá hàm mục tiêu toàn cục (ví dụ: `full_macro_f1` hoặc `selective_macro_f1` trên tập validation). Nếu điểm số tăng $\Delta > 0$, chấp nhận $k \in \mathcal{I}$; ngược lại giữ $k \in \mathcal{D}$.

**Những nhược điểm chí mạng của phương pháp tham lam này:**
1. **Hiện tượng tối ưu hóa giả tạo (Macro-F1 Metric Artifact):** Việc một nhãn $k$ được chấp nhận vào $\mathcal{I}$ thực chất chỉ là một hệ quả tối ưu cục bộ của hàm mục tiêu tổng hợp. Điểm Macro-F1 tăng có thể chỉ vì việc loại bỏ $k$ khỏi chuỗi CC vô tình làm giảm nhiễu cục bộ cho một vài nhãn đứng sau, chứ **hoàn toàn không phản ánh bản chất nhãn $k$ có thực sự độc lập với các nhãn khác hay không**.
2. **Chi phí tính toán bùng nổ $\mathcal{O}(K^2 \cdot \text{Inference}_{CC})$:** Tại mỗi bước thử nghiệm một nhãn ứng viên, hệ thống phải chạy lại toàn bộ thuật toán suy luận xác suất Mean-field cho toàn bộ chuỗi $CC$ trên tập validation. Với các bài toán có số lượng nhãn lớn (như `genbase` với 27 nhãn, `yeast` với 14 nhãn), thời gian phân hoạch chiếm tới 70-80% tổng thời gian huấn luyện.
3. **Thiếu tính định hướng nhân quả (Lack of Causal Directionality):** Thứ tự duyệt nhãn bị áp đặt từ trước khiến nhãn được xét sớm có xác suất vào $\mathcal{I}$ cao hơn, dẫn đến phân hoạch thiếu tính khách quan và nhạy cảm với việc xáo trộn thứ tự ban đầu.

---

### 1.2. Động lực của cơ chế phân hoạch tự nhiên dựa trên dữ liệu (Data-Driven Natural Partitioning)

Để giải quyết triệt để các hạn chế trên, buổi họp thống nhất chuyển hướng sang **nguyên lý phân hoạch tự nhiên dựa trên dữ liệu (Data-Driven Intuitive Partitioning)**:

> **Nguyên lý cốt lõi:** Một nhãn $Y_k$ được coi là một **Nhãn Độc Lập ($\mathcal{I}$)** khi và chỉ khi **bản thân dữ liệu đặc trưng đầu vào $X$ đã chứa đựng đầy đủ thông tin để dự đoán chính xác nhãn đó với hiệu năng cao vượt trội**, mà không cần bất kỳ thông tin bổ trợ nào từ các nhãn khác.

**Lợi ích trực tiếp của nguyên lý mới:**
- **Tính khách quan:** Việc gán nhãn vào $\mathcal{I}$ hoàn toàn dựa trên năng lực phân loại độc lập của nhãn đó trước các thuộc tính $X$, tách biệt hoàn toàn với cấu trúc tương quan phức tạp giữa các nhãn khác.
- **Tiết kiệm thời gian tính toán:** Đánh giá độc lập trên từng nhãn bằng mô hình Binary Relevance cơ sở chỉ tốn $\mathcal{O}(K)$ phép đánh giá song song, hoàn toàn không cần chạy suy luận chuỗi CC trong giai đoạn phân hoạch ban đầu.
- **Tính giải thích khoa học cao (High Interpretability):** Đáp ứng đúng chuẩn mực y sinh và kỹ thuật: các nhãn dễ nhận diện trực tiếp từ triệu chứng/đặc trưng sẽ được xác định trước, sau đó đóng vai trò làm tiền đề để suy luận các nhãn tiềm ẩn phức tạp hơn.

---

### 1.3. Tính định hướng nhân quả và triết lý phân tách $X \to Y_{\mathcal{I}} \to Y_{\mathcal{D}}$

Xem xét ví dụ thực tế trong y khoa được nêu trong buổi họp:
- Một bệnh nhân có các đặc trưng triệu chứng $X$ (xét nghiệm máu, huyết áp, men gan,...).
- Bệnh $A$ (ví dụ: Đái tháo đường Type 2) có thể được chẩn đoán với độ chính xác rất cao ($F_1 > 0.85$) chỉ dựa trên nồng độ glucose và chỉ số HbA1c trong $X$. Do đó, bệnh $A$ thuộc nhóm **Nhãn Độc Lập ($\mathcal{I}$)**.
- Bệnh $B$ (ví dụ: Bệnh võng mạc tiểu đường hoặc Suy thận mạn) là biến chứng phát sinh từ bệnh $A$. Triệu chứng ban đầu $X$ có thể chưa thể hiện rõ tổn thương thận, nhưng nếu biết bệnh nhân chắc chắn đã mắc bệnh $A$, xác suất mắc bệnh $B$ sẽ tăng vọt.
- **Quan hệ nhân quả 1 chiều:** Sự tồn tại của bệnh $A$ cung cấp thông tin sống còn để dự đoán bệnh $B$. Ngược lại, việc bệnh nhân có biến chứng $B$ hay không **không thể đảo ngược để quyết định việc bệnh nhân có mắc bệnh $A$ hay không**.

```
                                +-----------------------------+
                                |     Đặc trưng đầu vào: X     |
                                +-----------------------------+
                                        /             \
                   Dự đoán trực tiếp   /               \  Dự đoán trực tiếp
                   độ chính xác cao  /                 \  kém hiệu quả
                                    v                   v
                    +--------------------+      +--------------------+
                    |  Nhãn Độc Lập: IL  | ===> |  Nhãn Phụ Thuộc: DL |
                    |      (Bệnh A)      | Cung |      (Bệnh B)      |
                    +--------------------+ cấp  +--------------------+
                                           thông
                                            tin
```

Do đó, kiến trúc **GSI-MLC-PA v5.1** áp dụng quy tắc:
1. **Pha 1:** Dùng mô hình Binary Relevance (BR) dự đoán trước toàn bộ các nhãn thuộc $\mathcal{I}$.
2. **Pha 2:** Sử dụng thông tin dự đoán của $\mathcal{I}$ ghép nối cùng vector đặc trưng $X$ để hỗ trợ dự đoán các nhãn thuộc $\mathcal{D}$. Tuyệt đối không để thông tin của $\mathcal{D}$ truyền ngược lại làm sai lệch $\mathcal{I}$.

---

## 2. CƠ SỞ TOÁN HỌC & GIẢI THUẬT PHÂN HOẠCH NHÃN MỚI (V5.1 CORE ALGORITHM)

### 2.1. Định nghĩa toán học của Nhãn Độc Lập Nội Sinh ($IL_1$)

Cho tập dữ liệu huấn luyện $\mathcal{D}_{\text{train}} = \{(x_i, y_i)\}_{i=1}^N$ với $x_i \in \mathbb{R}^d$ và vector nhãn $y_i = [y_{i1}, \dots, y_{iK}] \in \{0, 1\}^K$.

Hệ thống chia tập huấn luyện thành tập tối ưu mô hình $\mathcal{D}_{\text{fit}}$ và tập thẩm định nội bộ $\mathcal{D}_{\text{val}}$ theo phương pháp phân tầng đa nhãn (Multilabel Stratified Split) với tỷ lệ $\alpha = 0.2$.

Trên $\mathcal{D}_{\text{fit}}$, huấn luyện $K$ bộ phân loại nhị phân độc lập (Binary Relevance):
$$f_k^{(0)}: \mathbb{R}^d \to [0, 1], \quad \hat{p}_{ik}^{(0)} = P(Y_k = 1 \mid X = x_i), \quad \forall k \in \{1, \dots, K\}$$

Trên tập thẩm định $\mathcal{D}_{\text{val}}$, đánh giá chất lượng dự đoán của từng nhãn $k$ thông qua chỉ số điểm $F_1$ nhị phân tối ưu:
$$\mathcal{S}_k^{(0)} = F_1\left(y_{\cdot, k}^{\text{val}}, \mathbb{I}(\hat{p}_{\cdot, k}^{(0)} \ge \theta_k^*)\right)$$
trong đó $\theta_k^*$ là ngưỡng phân loại nhị phân tối đa hóa $F_1$ trên $\mathcal{D}_{\text{val}}$ (được tìm kiếm chính xác qua thuật toán quét xác suất $\mathcal{O}(N_{\text{val}} \log N_{\text{val}})$).

**Quy tắc xác định Tầng Độc Lập Thứ Nhất ($\mathcal{I}_1$):**
$$\mathcal{I}_1 = \left\{ k \in \{1, \dots, K\} \mid \mathcal{S}_k^{(0)} \ge \tau \right\}$$
$$\mathcal{D}_1 = \{1, \dots, K\} \setminus \mathcal{I}_1$$
với $\tau \in [0.70, 0.80]$ là ngưỡng chất lượng phân định độc lập định trước.

---

### 2.2. Cơ chế bóc tách đa tầng (Iterative Cascaded Peeling: $IL_1 \to IL_2 \to \dots \to IL_m$)

Nếu sau bước 1, tập phụ thuộc $\mathcal{D}_1$ vẫn còn các nhãn chưa được giải quyết, ta không vội vàng đưa toàn bộ vào Classifier Chain. Có những nhãn trong $\mathcal{D}_1$ vốn dĩ không thể dự đoán tốt chỉ bằng $X$, nhưng **khi được bổ sung thêm thông tin từ tầng $\mathcal{I}_1$ thì chất lượng dự đoán lập tức vượt ngưỡng $\tau$**.

Do đó, hệ thống thực hiện quy trình bóc tách lặp nhiều tầng (Iterative Cascaded Peeling):

```
+---------------------------------------------------------------------------------------------------+
|               QUY TRÌNH BÓC TÁCH NHÃN ĐA TẦNG (ITERATIVE STRATIFIED PEELING)                     |
+---------------------------------------------------------------------------------------------------+
|  [Đặc trưng X]                                                                                    |
|       |                                                                                           |
|       v  (Tầng 0: Huấn luyện BR trên X)                                                           |
|  Đánh giá F1 từng nhãn: S_k^(0) >= tau ?                                                          |
|       |                                                                                           |
|       +---> ĐẠT: Đưa vào Tầng IL_1                                                                |
|       |                                                                                           |
|       +---> CHƯA ĐẠT: Thuộc tập D_1.                                                              |
|               |                                                                                   |
|               v  (Tầng 1: Mở rộng đặc trưng X^(1) = [X, P(Y_IL_1 | X)])                           |
|          Huấn luyện mô hình dự đoán D_1 dựa trên X^(1)                                            |
|          Đánh giá F1: S_k^(1) >= tau ?                                                            |
|               |                                                                                   |
|               +---> ĐẠT: Đưa vào Tầng IL_2                                                        |
|               |                                                                                   |
|               +---> CHƯA ĐẠT: Thuộc tập D_2.                                                      |
|                       |                                                                           |
|                       v  (Lặp lại t = 2, ..., m cho đến khi DỪNG)                                 |
|                  Không còn nhãn nào vượt tau  HOẶC  Đạt giới hạn chi phí phạt                     |
|                       |                                                                           |
|                       v                                                                           |
|       +------------------------------------+      +-------------------------------------------+   |
|       | TẬP ĐỘC LẬP TẦNG:                  |      | TẬP PHỤ THUỘC CỐT LÕI (DL_residual):       |   |
|       | I = IL_1 U IL_2 U ... U IL_m       |      | D_res = D_m                               |   |
|       | (Dự đoán tuần tự theo tầng BR)     |      | (Mô hình hóa bằng Classifier Chain mới)   |   |
|       +------------------------------------+      +-------------------------------------------+   |
+---------------------------------------------------------------------------------------------------+
```

#### Thuật toán hình thức (Formal Iterative Peeling Algorithm):
- **Khởi tạo:** 
  - Bước lặp $t = 0$.
  - Tập nhãn độc lập tích lũy $\mathcal{I}_{\text{accum}}^{(0)} = \emptyset$.
  - Tập nhãn ứng viên phụ thuộc $\mathcal{D}^{(0)} = \{1, \dots, K\}$.
  - Ma trận đặc trưng mở rộng $X^{(0)} = X$.

- **Tại mỗi bước lặp $t = 1, 2, \dots, T_{\max}$:**
  1. Với mỗi nhãn $k \in \mathcal{D}^{(t-1)}$, huấn luyện bộ phân loại nhị phân trên không gian đặc trưng mở rộng:
     $$f_k^{(t)}: \text{Domain}(X^{(t-1)}) \to [0, 1], \quad \hat{p}_{\cdot, k}^{(t)} = f_k^{(t)}(X_{\text{val}}^{(t-1)})$$
  2. Đánh giá điểm số trên validation:
     $$\mathcal{S}_k^{(t)} = F_1\left(y_{\cdot, k}^{\text{val}}, \mathbb{I}(\hat{p}_{\cdot, k}^{(t)} \ge \theta_k^*)\right)$$
  3. Xác định tập nhãn mới đạt chuẩn tại tầng $t$:
     $$\mathcal{I}_t = \left\{ k \in \mathcal{D}^{(t-1)} \mid \mathcal{S}_k^{(t)} \ge \tau \right\}$$
  4. **Kiểm tra điều kiện dừng:**
     - Nếu $\mathcal{I}_t = \emptyset$: Dừng quy trình (không còn nhãn nào có thể nâng cấp thành công).
     - Nếu vượt quá ngưỡng phạt chi phí tính toán (mục 4): Dừng quy trình.
  5. Cập nhật trạng thái:
     $$\mathcal{I}_{\text{accum}}^{(t)} = \mathcal{I}_{\text{accum}}^{(t-1)} \cup \mathcal{I}_t$$
     $$\mathcal{D}^{(t)} = \mathcal{D}^{(t-1)} \setminus \mathcal{I}_t$$
  6. Xây dựng không gian đặc trưng mở rộng cho bước tiếp theo $X^{(t)}$ bằng cách ghép thêm phân phối xác suất dự đoán của các nhãn trong $\mathcal{I}_t$:
     $$X^{(t)} = \left[ X^{(t-1)}, \hat{P}\left(Y_{\mathcal{I}_t} = 1 \mid X^{(t-1)}\right) \right]$$

- **Kết quả thu được:**
  - Danh sách các tầng độc lập có thứ tự: $\left( \mathcal{I}_1, \mathcal{I}_2, \dots, \mathcal{I}_m \right)$.
  - Tập nhãn phụ thuộc cốt lõi còn lại: $\mathcal{D}_{\text{residual}} = \mathcal{D}^{(m)}$.

---

### 2.3. Không gian biểu diễn xác suất chuyển tiếp mềm (Soft Probability Augmentation)

Một câu hỏi kỹ thuật then chốt: *Khi dùng các nhãn của tầng $\mathcal{I}_t$ làm đặc trưng bổ trợ để huấn luyện tầng tiếp theo, nên dùng nhãn nhị phân cứng ($\hat{y} \in \{0, 1\}$) hay xác suất liên tục mềm ($\hat{p} \in [0, 1]$)?*

**Quyết định thiết kế trong v5.1: BẮT BUỘC DÙNG XÁC SUẤT LIÊN TỤC MỀM (Soft Probabilities).**

*Lý do khoa học:*
1. **Tránh lan truyền lỗi phân ngưỡng (Threshold Error Propagation):** Nhãn nhị phân cứng $\hat{y} \in \{0, 1\}$ làm mất thông tin độ bất định (uncertainty). Nếu một mẫu có xác suất $0.51$ bị ép thành $1$, sai số phân loại sẽ trở thành đặc trưng sai lệch nghiêm trọng cho tầng sau.
2. **Khả năng tích hợp vi phân (Smooth Optimization):** Biểu diễn mềm $p \in [0, 1]$ giúp các mô hình cơ sở như MLP và Logistic Regression hội tụ mượt mà và tối ưu hóa gradient chuẩn xác hơn.
3. **Phù hợp với cơ chế Partial Abstention (BOP):** Ở bước suy luận cuối cùng, các xác suất mềm này được giữ nguyên để tính toán vùng từ chối $[-1]$ một cách nhất quán.

---

### 2.4. Ngưỡng phân định hiệu năng $\tau$ và cơ chế ngưỡng thích nghi

Ngưỡng $\tau$ đóng vai trò là "bộ lọc tiêu chuẩn" để quyết định nhãn có đủ tư cách đứng độc lập hay không:
- **Giá trị mặc định đề xuất:** $\tau = 0.75$ (biên độ khảo sát: $\tau \in [0.70, 0.80]$).
- **Cơ chế Ngưỡng Thích Nghi (Adaptive Distribution-aware Threshold):**
  Đối với các bộ dữ liệu có hiện tượng mất cân bằng nhãn cực đoan (như `genbase`, `humanpseaac`, `plantpseaac` với độ phổ biến nhãn $P(Y_k=1) < 2\%$), điểm $F_1$ tuyệt đối có thể rất khó đạt mức $0.75$ dù mô hình đã học tốt nhất có thể.
  
  Do đó, hệ thống hỗ trợ 2 chế độ cấu hình:
  1. `fixed`: $\tau_k = \tau_{\text{base}} = 0.75, \quad \forall k$.
  2. `adaptive_imbalance`: Ngưỡng được điều chỉnh theo độ phổ biến tiên nghiệm (prior positive rate $\pi_k = \frac{1}{N} \sum_i y_{ik}$):
     $$\tau_k = \max\left( \tau_{\min}, \tau_{\text{base}} \cdot \left[ 1.0 - \gamma \cdot \log_{10}\left( \frac{1}{\pi_k + \epsilon} \right) \right] \right)$$
     với $\tau_{\min} = 0.50$, $\gamma = 0.08$. Cơ chế này ngăn chặn việc bỏ sót các nhãn hiếm đã được mô hình học rất tốt so với phân phối ngẫu nhiên.

---

## 3. TÁI THIẾT KẾ CHUỖI CLASSIFIER CHAIN CHO TẬP PHỤ THUỘC $DL$

Sau khi kết thúc quá trình phân tầng, tập nhãn phụ thuộc cốt lõi $\mathcal{D}_{\text{residual}}$ gồm những nhãn không thể tự đứng độc lập. Các nhãn này cần được kết nối trong một chuỗi phân loại (Classifier Chain - CC).

### 3.1. Nghịch lý tích lũy sai số và Luận điểm "Tương quan nhỏ lên đầu"

Theo chỉ đạo trong buổi họp:
> *"Chỉnh lại sắp xếp chuỗi CC trong tập DL đưa các nhãn tổng tương quan nhỏ lên đầu thay vì tổng lớn hơn"*

**Phân tích nghịch lý và Cơ sở lý thuyết sâu sắc:**
1. **Sai lầm của chiến lược cũ (Tổng tương quan lớn lên đầu):**
   Trong các phiên bản CC truyền thống và v5 cũ, nhãn có tổng tương quan lớn nhất $\sum_j |\rho_{kj}|$ thường bị đặt ở đầu chuỗi với giả định nó là "trung tâm thông tin". Tuy nhiên, điều này tạo ra **thảm họa lan truyền sai số (Error Propagation Catastrophe)**:
   - Nhãn có tương quan lớn nhất thường là nhãn phức tạp nhất, phụ thuộc chằng chịt vào nhiều yếu tố.
   - Khi đặt nó ở đầu chuỗi (vị trí Root), nó **buộc phải dự đoán hoàn toàn dựa trên $X$ mà không có bất kỳ nhãn nào trợ giúp**.
   - Do khó dự đoán, tỷ lệ sai sót của nó rất cao. Và vì nó đứng đầu chuỗi, **toàn bộ các sai sót của nó sẽ bị nhân bản và phóng đại sang tất cả các nhãn phía sau**.

2. **Luận điểm khoa học của chiến lược mới (Tổng tương quan nhỏ lên đầu):**
   - Nhãn có tổng tương quan nhỏ $\sum_{j \in \mathcal{D}} |\rho_{kj}|$ là nhãn **ít bị chi phối bởi các nhãn khác nhất** trong tập $\mathcal{D}$.
   - Vì ít bị chi phối, việc dự đoán nó ở những vị trí đầu của chuỗi sẽ có độ ổn định cao hơn, ít bị rủi ro thiếu thông tin.
   - Khi các nhãn này đưa ra dự đoán tương đối vững chắc, chúng sẽ đóng vai trò làm điểm tựa (predecessor features) bổ sung cho các nhãn có độ phụ thuộc cao ở cuối chuỗi.
   - **Nhãn có tổng tương quan lớn nhất sẽ nằm ở cuối chuỗi**: Đây là vị trí tối ưu nhất vì nó được "thừa hưởng" đầy đủ toàn bộ ngữ cảnh thông tin từ $X$, toàn bộ các tầng $\mathcal{I}$, và toàn bộ các nhãn tiền nhiệm trong $\mathcal{D}$.

```
      CHIẾN LƯỢC CŨ (V5): TỔNG TƯƠNG QUAN LỚN LÊN ĐẦU (RỦI RO CAO)
      [Nhãn phức tạp nhất] ---> Dự đoán sai ngay từ đầu
              |
              v (Lan truyền sai số)
      [Nhãn trung bình]    ---> Bị nhiễm độc đặc trưng
              |
              v
      [Nhãn đơn giản]      ---> Hỏng toàn bộ chuỗi

      -------------------------------------------------------------------------

      CHIẾN LƯỢC MỚI (V5.1): TỔNG TƯƠNG QUAN NHỎ LÊN ĐẦU (VỮNG CHẮC)
      [Nhãn độc lập I]     ---> Đã dự đoán rất chính xác từ X (F1 >= 0.75)
              |
              v (Cung cấp ngữ cảnh sạch)
      [DL: Tương quan nhỏ] ---> Dự đoán ít phụ thuộc, độ tin cậy cao
              |
              v (Tích lũy ngữ cảnh)
      [DL: Tương quan lớn] ---> Được hưởng lợi tối đa từ toàn bộ chuỗi tiền nhiệm
```

---

### 3.2. Giải thuật sắp xếp trật tự chuỗi theo tổng tương quan tăng dần (Ascending Correlation Order)

Cho tập nhãn phụ thuộc $\mathcal{D}_{\text{residual}} = \{d_1, d_2, \dots, d_{|\mathcal{D}|}\}$ và ma trận tương quan nhãn tuyệt đối $\mathbf{R} \in [0, 1]^{K \times K}$ tính trên tập huấn luyện:
$$R_{jk} = |\text{Corr}(Y_j, Y_k)| = \frac{|\text{Cov}(Y_j, Y_k)|}{\sigma_{Y_j} \sigma_{Y_k}}$$

**Bước 1: Tính tổng tương quan nội bộ (Internal Cumulative Correlation) trong tập $\mathcal{D}$:**
Với mỗi nhãn $d \in \mathcal{D}_{\text{residual}}$:
$$C(d) = \sum_{j \in \mathcal{D}_{\text{residual}}, j \neq d} R_{dj}$$

**Bước 2: Sắp xếp tăng dần (Ascending Sort):**
Trật tự các nhãn trong chuỗi CC cho tập phụ thuộc được xác định bằng phép hoán vị:
$$\pi_{\mathcal{D}} = \text{argsort}_{\text{ascending}}\left( [C(d)]_{d \in \mathcal{D}_{\text{residual}}} \right)$$
Trong đó, nhãn có tổng tương quan nhỏ nhất được gán vị trí đầu tiên của chuỗi $DL$, nhãn có tổng tương quan lớn nhất được gán vị trí cuối cùng. Trường hợp hòa điểm (ties), giải quyết theo chỉ số nhãn nhỏ hơn (deterministic tie-breaking).

**Bước 3: Hợp nhất toàn bộ chuỗi suy luận toàn cục:**
Thứ tự suy luận tổng thể của mô hình GSI-MLC-PA v5.1 là:
$$\Pi_{\text{final}} = \left[ \mathcal{I}_1 \ \Vert \ \mathcal{I}_2 \ \Vert \ \dots \ \Vert \ \mathcal{I}_m \ \Vert \ \pi_{\mathcal{D}} \right]$$

---

### 3.3. So sánh đối chuẩn các chiến lược trật tự CC: Ascending vs. Descending vs. Direct Parents

Trong mã nguồn v5.1, hệ thống thiết kế tham số `dl_chain_order_strategy` với 3 chế độ thử nghiệm nhằm kiểm chứng khoa học trong phần Ablation Study:
1. `ascending_correlation` (Mặc định v5.1): Sắp xếp tăng dần tổng tương quan trong $\mathcal{D}$.
2. `descending_correlation` (Kế thừa v5 cũ): Sắp xếp giảm dần tổng tương quan trong $\mathcal{D}$.
3. `causal_tree_parents` (Mở rộng đồ thị): Xây dựng cây bao trùm cực đại (Maximum Spanning Tree - MST) dựa trên trọng số tương quan và duyệt theo thứ tự topo từ gốc lá có bậc nhỏ nhất đến lớn nhất.

---

## 4. HÀM PHẠT PHỨC TẠP TÍNH TOÁN & ĐIỀU KIỆN DỪNG TỰ ĐỘNG

Theo chỉ đạo trong buổi họp:
> *"Nghiên cứu thêm về hàm phạt cho quá trình phân hoạch nhãn để tránh độ phức tạp thời gian quá lớn, lãng phí thời gian chạy"*

### 4.1. Đánh giá chi phí tính toán $\mathcal{O}(\cdot)$ của phân tầng lặp

Xét bài toán có $K$ nhãn, kích thước mẫu huấn luyện $N$, số chiều thuộc tính $d$:
- Chi phí huấn luyện 1 mô hình cơ sở đơn nhãn (Base Learner): $\mathcal{C}_{\text{base}}(N, d)$.
- Tại Tầng 0 (tìm $IL_1$): Huấn luyện $K$ mô hình BR $\implies \text{Cost}_0 = K \cdot \mathcal{C}_{\text{base}}(N, d)$.
- Tại Tầng $t$ ($t \ge 1$): Số nhãn còn lại là $|\mathcal{D}^{(t-1)}|$, số chiều đặc trưng mở rộng là $d + |\mathcal{I}_{\text{accum}}^{(t-1)}|$.
  $$\text{Cost}_t = |\mathcal{D}^{(t-1)}| \cdot \mathcal{C}_{\text{base}}\left(N, d + |\mathcal{I}_{\text{accum}}^{(t-1)}|\right)$$

Nếu số bước lặp $m$ kéo dài mà mỗi bước chỉ bóc tách được 1 nhãn, tổng chi phí tính toán sẽ tiệm cận $\mathcal{O}(m \cdot K \cdot \mathcal{C}_{\text{base}})$, gây lãng phí tài nguyên tính toán nghiêm trọng trong khi mức cải thiện $F_1$ có thể tiệm cận 0.

---

### 4.2. Thiết kế hàm phạt phức tạp (Complexity / Time Penalty Function)

Để cân bằng giữa **Lợi ích Phân loại (Classification Utility)** và **Chi phí Tính toán (Computation Cost)**, v5.1 định nghĩa hàm tiện ích biên có phạt (Penalized Marginal Gain):

Tại bước lặp $t$, việc quyết định có tiếp tục bóc tách tầng $t$ hay dừng lại được chuẩn hóa qua hàm mục tiêu:
$$\Delta \mathcal{U}(t) = \underbrace{\frac{1}{K} \sum_{k \in \mathcal{I}_t} \left( \mathcal{S}_k^{(t)} - \mathcal{S}_k^{(t-1)} \right)}_{\text{Mức tăng F1 thực tế mang lại}} - \underbrace{\lambda_{\text{complexity}} \cdot \mathcal{P}(t, |\mathcal{D}^{(t-1)}|)}_{\text{Hàm phạt chi phí tính toán}}$$

Trong đó hàm phạt $\mathcal{P}(t, |\mathcal{D}|)$ được thiết kế theo 2 thành phần:
$$\mathcal{P}(t, |\mathcal{D}|) = \left( \frac{|\mathcal{D}|}{K} \right) \cdot \exp\left( \beta \cdot (t - 1) \right)$$
- $\frac{|\mathcal{D}|}{K}$: Tỷ lệ số lượng mô hình phải huấn luyện lại trên tổng số nhãn.
- $\exp(\beta \cdot (t - 1))$: Hệ số phạt tăng theo hàm mũ đối với số tầng sâu, ngăn chặn mạng lưới phân tầng quá sâu dẫn đến overfitting và trễ thời gian suy luận (với $\beta = 0.5$).
- $\lambda_{\text{complexity}}$: Siêu tham số phạt chi phí (mặc định $\lambda = 0.01$).

> [!IMPORTANT]
> **Ghi chú kiến trúc & triển khai thực nghiệm (Cập nhật theo yêu cầu):**  
> Hàm tiện ích biên có phạt $\Delta \mathcal{U}(t)$ và cơ chế đánh giá chi phí được đóng gói thành một module độc lập hoàn toàn tại [src/selection/complexity_penalty.py](file:///d:/University_Subject/ML%20Research/BR_CC/src/selection/complexity_penalty.py).  
> **Trạng thái mặc định:** Tạm thời **chưa tích hợp cưỡng chế** vào luồng chạy chính (`use_complexity_penalty=False`) để đánh giá năng lực phân hoạch tự nhiên thuần túy của dữ liệu trong đợt thử nghiệm ban đầu trên 5 datasets. Module này được thiết kế sẵn sàng để kích hoạt bất kỳ lúc nào thông qua cấu hình `use_complexity_penalty=True`.

---

### 4.3. Tiêu chuẩn dừng tối ưu (Optimal Stopping Criteria)

Quá trình bóc tách đa tầng tự động dừng lại khi **gặp bất kỳ điều kiện nào** sau đây:
1. **Tiêu chuẩn Rỗng (Empty New Layer):** $\mathcal{I}_t = \emptyset$ (Không có bất kỳ nhãn nào trong $\mathcal{D}^{(t-1)}$ đạt ngưỡng $\mathcal{S}_k^{(t)} \ge \tau$).
2. **Tiêu chuẩn Hiệu quả Biên Âm (Negative Marginal Utility):** $\Delta \mathcal{U}(t) \le 0$ (Mức cải thiện $F_1$ của tầng mới không bù đắp được chi phí tính toán phạt).
3. **Tiêu chuẩn Độ sâu Tối đa (Maximum Depth Bound):** $t \ge T_{\max}$ (Mặc định $T_{\max} = 3$). Kinh nghiệm thực nghiệm cho thấy sau 3 tầng, hầu như mọi nhãn tiềm năng đều đã được khai thác; các nhãn còn lại có cấu trúc phụ thuộc vòng (cyclic dependency) nên chuyển sang Classifier Chain.
4. **Tiêu chuẩn Kích thước Tập Dưỡng Tối thiểu (Residual Floor):** $|\mathcal{D}^{(t)}| \le 1$ (Nếu tập phụ thuộc chỉ còn 0 hoặc 1 nhãn, chuỗi CC không còn ý nghĩa hoặc đã hoàn tất).

---

## 5. CƠ CHẾ PHÒNG CHỐNG SỤP ĐỔ MÔ HÌNH (MODEL COLLAPSE MITIGATION)

Theo cảnh báo trong buổi họp:
> *"Nếu trong quá trình làm mà mô hình cũ bị sụp đổ thì cần phải sửa mô hình"*

### 5.1. Nhận diện hiện tượng sụp đổ trên các nhãn mất cân bằng nặng (Severe Imbalance Collapse)

Trong các bộ dữ liệu sinh học phân tử (`genbase`, `humanpseaac`, `plantpseaac`) và âm nhạc (`music`), một số nhãn có tỷ lệ mẫu dương cực thấp ($< 1\%$). Khi huấn luyện mô hình MLP hoặc SVM với hàm mất mát tiêu chuẩn (BCE / Hinge loss):
- Mô hình dễ rơi vào cực tiểu cục bộ tầm thường: **Dự đoán toàn bộ bằng 0 (Trivial All-Zero Prediction)**.
- Khi đó, $TP_k = 0 \implies F_1^{(k)} = 0.0000$.
- Hệ quả dây chuyền: Các nhãn này vĩnh viễn không thể vượt ngưỡng $\tau$, bị đẩy vào tập $\mathcal{D}_{\text{residual}}$. Khi chạy CC, chúng tiếp tục truyền vector 0 làm nhiễu loạn các nhãn phía sau, khiến toàn bộ chuỗi suy luận sụp đổ.

---

### 5.2. Tích hợp Asymmetric Loss / Focal Loss cho tầng biểu diễn sâu

Để ngăn chặn triệt để sự sụp đổ mô hình trên các nhãn hiếm, v5.1 tích hợp sẵn 2 cơ chế hàm mất mát tiên tiến vào `src/models/base_learners.py`:

1. **Weighted Binary Cross-Entropy với Class Weights cân bằng:**
   $$w_k = \frac{N - N_k^+}{N_k^+ + \epsilon}, \quad \mathcal{L}_{\text{WBCE}}(p, y) = - \left[ w_k \cdot y \log(p) + (1 - y) \log(1 - p) \right]$$
2. **Asymmetric Loss (ASL - Ridnik et al. 2021):**
   Triệt tiêu triệt để gradient của các mẫu âm dễ (easy negatives) và tập trung vào các mẫu dương hiếm:
   $$\mathcal{L}_{\text{ASL}} = - y (1 - p)^{\gamma_+} \log(p) - (1 - y) (p_m)^{\gamma_-} \log(1 - p_m)$$
   trong đó $p_m = \max(p - m, 0)$ là xác suất đã dịch chuyển biên an toàn (margin shifting).

---

### 5.3. Hiệu chuẩn xác suất tự động (Automated Platt Scaling) trên từng tầng

Một nguyên nhân lớn gây sụp đổ suy luận chuỗi trong CC là **xác suất đầu ra của các tầng trước bị cực đoan hóa** (overconfident probabilities gần sát 0.0 hoặc 1.0). Khi các xác suất này được làm đặc trưng đầu vào cho tầng tiếp theo, mô hình tầng sau bị bão hòa trọng số (gradient saturation).

**Giải pháp trong v5.1:**
Tại mỗi tầng $\mathcal{I}_t$, các mô hình phân loại bắt buộc phải trải qua bước **Hiệu chuẩn Platt (Platt Scaling via Sigmoid Calibration)** trên tập validation:
$$P_{\text{calibrated}}(Y_k = 1 \mid x) = \frac{1}{1 + \exp\left( - (A_k \cdot f_k(x) + B_k) \right)}$$
với $A_k, B_k$ được tối ưu hóa bằng phương pháp Maximum Likelihood trên tập out-of-fold validation. Điều này đảm bảo rằng các đặc trưng xác suất đưa vào tầng sau phản ánh đúng độ tin cậy thực tế (well-calibrated probabilities).

---

## 6. NGHIÊN CỨU CẤU TRÚC ĐỒ THỊ NHÓM NHÃN (LABEL GRAPH STRUCTURE - FUTURE EXTENSION)

Theo định hướng buổi họp:
> *"Nghiên cứu thêm nhóm phụ thuộc và nhóm độc lập (Làm sau khi test các phần trên)"*

Phần này được đặc tả như một hướng nghiên cứu mở rộng (Extension Module) sau khi hoàn tất kiểm thử cốt lõi:

### 6.1. Xây dựng đồ thị phụ thuộc nhãn có trọng số (Weighted Label Dependency Graph)
Biểu diễn mối quan hệ giữa $K$ nhãn thành một đồ thị vô hướng có trọng số $\mathcal{G} = (\mathcal{V}, \mathcal{E}, \mathbf{W})$:
- Tập đỉnh $\mathcal{V} = \{1, \dots, K\}$ đại diện cho $K$ nhãn.
- Trọng số cạnh $W_{ij} = I(Y_i; Y_j \mid X)$ (Thông tin tương hỗ có điều kiện - Conditional Mutual Information) hoặc xấp xỉ bằng ma trận hệ số tương quan riêng phần (Partial Correlation Matrix):
  $$\rho_{ij \cdot X} = \frac{\rho_{ij} - \rho_{iX} \rho_{jX}}{\sqrt{(1 - \rho_{iX}^2)(1 - \rho_{jX}^2)}}$$

### 6.2. Phát hiện cộng đồng nhãn độc lập và cụm nhãn phụ thuộc (Community Detection)
Áp dụng thuật toán phát hiện cộng đồng (như Louvain Modularity hoặc Spectral Clustering trên đồ thị Laplacian $\mathbf{L} = \mathbf{D} - \mathbf{W}$):
- **Các đỉnh cô lập (Isolated Nodes with Degree $\approx 0$):** Là các nhãn độc lập cấu trúc $\implies$ Nhóm $\mathcal{I}_{\text{graph}}$.
- **Các cụm liên kết đậm đặc (Dense Cliques):** Là các nhóm nhãn phụ thuộc nội bộ chặt chẽ $\implies$ Tách thành các chuỗi CC con độc lập (Independent Sub-Chains). Thay vì chạy 1 chuỗi CC dài cho toàn bộ $DL$, ta chạy song song nhiều chuỗi CC ngắn trên từng cụm, vừa triệt tiêu tích lũy sai số, vừa giảm mạnh độ phức tạp thời gian.

---

## 7. ĐẶC TẢ KIẾN TRÚC HỆ THỐNG & API CONTRACTS

### 7.1. Cấu trúc thư mục mã nguồn

```
BR_CC/
├── spec/
│   ├── spec_v5.md                      # Đặc tả v5 cũ
│   └── spec_V5_1.md                    # [TÀI LIỆU HIỆN TẠI] Đặc tả chi tiết v5.1
├── src/
│   ├── models/
│   │   ├── base_learners.py            # Hỗ trợ ASL, WBCE, Platt calibration
│   │   ├── classifier_chain.py         # CC với hỗ trợ Ascending Order
│   │   ├── gsi_mlc_pa.py               # Tích hợp chế độ phân hoạch v5.1
│   │   └── registry.py                 # Đăng ký các model variants v5.1
│   ├── selection/
│   │   ├── __init__.py
│   │   ├── objectives.py               # Macro-F1, Coverage objectives
│   │   ├── partition.py                # Legacy partition providers
│   │   └── stratified_peeling.py       # [MODULE MỚI] Phân tầng đa bước theo data
│   ├── decision/
│   │   └── macro_f1.py                 # Per-label adaptive thresholding BOP
│   └── evaluation/
│       └── pipeline_v3.py              # Runner thực nghiệm 10 datasets
├── tests/
│   └── test_v5_1_stratified.py         # [TEST MỚI] Unit tests cho cơ chế v5.1
└── configs/
    └── benchmark_v5_1.yaml             # Cấu hình siêu tham số v5.1
```

---

### 7.2. Module phân hoạch mới: `src/selection/stratified_peeling.py`

Module chịu trách nhiệm thực thi toàn bộ logic phân hoạch tự nhiên, bóc tách đa tầng và trật tự CC mới.

```python
"""Stratified Data-Driven Peeling Partition Provider for GSI-MLC-PA v5.1.

This module implements the non-greedy, performance-based iterative peeling
partitioning described in spec_V5_1.md.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
import numpy as np


@dataclass(frozen=True)
class StratifiedPeelingResult:
    """Audit metadata and frozen outputs of the v5.1 partitioning process."""
    independent_layers: Tuple[Tuple[int, ...], ...]
    all_independent_labels: Tuple[int, ...]
    dependent_residual_labels: Tuple[int, ...]
    final_execution_order: Tuple[int, ...]
    layer_scores: Dict[int, Dict[int, float]]
    stopping_reason: str
    num_stages_executed: int
    complexity_penalties: List[float]
    total_selection_time_seconds: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "independent_layers": [list(layer) for layer in self.independent_layers],
            "all_independent_labels": [int(l) for l in self.all_independent_labels],
            "dependent_residual_labels": [int(l) for l in self.dependent_residual_labels],
            "final_execution_order": [int(l) for l in self.final_execution_order],
            "layer_scores": self.layer_scores,
            "stopping_reason": self.stopping_reason,
            "num_stages_executed": int(self.num_stages_executed),
            "complexity_penalties": [float(p) for p in self.complexity_penalties],
            "total_selection_time_seconds": float(self.total_selection_time_seconds),
        }


class StratifiedPeelingSelector:
    """Iterative Data-Driven Peeling Selector with Complexity Penalization.
    
    Parameters
    ----------
    threshold : float, default=0.75
        Performance threshold (F1 score) required to promote a label to an IL layer.
    max_depth : int, default=3
        Maximum number of iterative peeling stages.
    complexity_penalty_lambda : float, default=0.01
        Penalty multiplier for computation complexity in the stopping criterion.
    complexity_penalty_beta : float, default=0.5
        Exponential penalty rate per additional depth stage.
    dl_chain_order : {"ascending_correlation", "descending_correlation"}, default="ascending_correlation"
        Order rule for CC within the residual DL set.
    adaptive_threshold : bool, default=False
        Whether to adjust threshold downward for severe class-imbalanced labels.
    """
    def __init__(
        self,
        threshold: float = 0.75,
        max_depth: int = 3,
        complexity_penalty_lambda: float = 0.01,
        complexity_penalty_beta: float = 0.5,
        dl_chain_order: str = "ascending_correlation",
        adaptive_threshold: bool = False,
    ):
        self.threshold = threshold
        self.max_depth = max_depth
        self.complexity_penalty_lambda = complexity_penalty_lambda
        self.complexity_penalty_beta = complexity_penalty_beta
        self.dl_chain_order = dl_chain_order
        self.adaptive_threshold = adaptive_threshold

    def fit_partition(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: np.ndarray,
        Y_val: np.ndarray,
        base_learner_factory,
        random_state: int = 42,
    ) -> StratifiedPeelingResult:
        """Execute the multi-stage peeling algorithm and return frozen partition."""
        ...
```

---

### 7.3. Cải tiến mô hình chính: `src/models/gsi_mlc_pa.py` (v5.1 mode)

Trong lớp `GSIMLCPartialAbstentionClassifier`:
- Thêm tham số:
  - `partition_strategy`: `{"legacy_greedy", "stratified_peeling"}` (mặc định `"stratified_peeling"`).
  - `il_threshold`: float (mặc định `0.75`).
  - `max_peeling_depth`: int (mặc định `3`).
  - `dl_order_direction`: `{"ascending", "descending"}` (mặc định `"ascending"`).
  - `complexity_penalty`: float (mặc định `0.01`).
- Trong phương thức `fit(X, Y)`:
  - Nếu `partition_strategy == "stratified_peeling"`, kích hoạt `StratifiedPeelingSelector`.
  - Đóng băng `independent_layers_`, `dependent_labels_`, và `order_`.
  - Huấn luyện BR cho từng tầng độc lập, sau đó huấn luyện CC với trật tự tương quan tăng dần cho tập $DL$.

---

### 7.4. Data Schema kiểm toán phân tầng (Stratified Audit Schema)

Toàn bộ kết quả phân tầng được lưu trữ cấu trúc trong `results_v3.json` và bảng `partition_audit.csv` để phục vụ phân tích khoa học:
```json
{
  "partition_audit": {
    "strategy": "stratified_peeling",
    "threshold": 0.75,
    "num_stages": 2,
    "stopping_reason": "empty_new_layer",
    "independent_layers": [
      {"stage": 1, "labels": [0, 3, 5], "mean_f1": 0.812},
      {"stage": 2, "labels": [1], "mean_f1": 0.764}
    ],
    "total_independent_count": 4,
    "dependent_residual_count": 2,
    "dependent_labels": [2, 4],
    "dl_ordering": [
      {"label": 4, "cumulative_correlation": 0.182},
      {"label": 2, "cumulative_correlation": 0.459}
    ],
    "final_full_chain": [0, 3, 5, 1, 4, 2]
  }
}
```

---

## 8. KẾ HOẠCH THỰC NGHIỆM BENCHMARK 10 DATASETS TOÀN DIỆN

### 8.1. Danh mục 10 tập dữ liệu và đặc tính phân phối

Thực nghiệm được tiến hành trên toàn bộ **10 tập dữ liệu đa nhãn chuẩn** đã tích hợp sẵn trong dự án:

| STT | Tập Dữ Liệu | Số Mẫu ($N$) | Số Đặc Trưng ($d$) | Số Nhãn ($K$) | Mức Độ Mất Cân Bằng Nhãn | Độ Phức Tạp Tương Quan |
|:---:|:---|:---:|:---:|:---:|:---|:---|
| 1 | **chd49** | 555 | 49 | 6 | Rất cao (Tim mạch) | Trung bình |
| 2 | **emotions** | 593 | 72 | 6 | Cân bằng (Âm nhạc/Cảm xúc) | Cao |
| 3 | **genbase** | 662 | 1186 | 27 | Cực đoan (Gen/Protein) | Rất cao (Thưa) |
| 4 | **gpositivepseaac** | 519 | 440 | 4 | Trung bình (Vi khuẩn) | Cao |
| 5 | **humanpseaac** | 3106 | 440 | 14 | Rất cao (Protein người) | Rất cao |
| 6 | **music** | 593 | 72 | 6 | Trung bình (Âm thanh) | Cao |
| 7 | **plantpseaac** | 978 | 440 | 12 | Rất cao (Thực vật) | Cao |
| 8 | **scene** | 2407 | 294 | 6 | Cân bằng (Thị giác máy tính) | Trung bình |
| 9 | **viruspseaac** | 207 | 440 | 6 | Cao (Virus) | Trung bình |
| 10 | **yeast** | 2417 | 103 | 14 | Trung bình (Nấm men) | Rất cao |

---

### 8.2. Ma trận cấu hình thực nghiệm và đối chuẩn (Benchmark Matrix)

Thực hiện đánh giá **5-Fold Cross-Validation** cố định seed (`random_state=42`) với 3 bộ phân loại cơ sở (Base Learners: `Logistic Regression`, `Linear SVM`, `MLP GPU`):

```
+---------------------------------------------------------------------------------------------------+
|                            MA TRẬN MÔ HÌNH THỰC NGHIỆM ĐỐI CHUẨN                                  |
+---------------------------------------------------------------------------------------------------+
| Nhóm 1: Baselines chuẩn mực không có từ chối                                                      |
|   1. BR (Binary Relevance)                                                                        |
|   2. CC (Classifier Chains tiêu chuẩn)                                                            |
|                                                                                                   |
| Nhóm 2: Baselines có từ chối (Partial Abstention)                                                 |
|   3. MLC-PA (Nguyen & Hüllermeier 2021)                                                           |
|   4. GSI-MLC-PA (Phiên bản v5 - Greedy Macro-F1 Selection)                                        |
|                                                                                                   |
| Nhóm 3: Mô hình đề xuất mới                                                                       |
|   5. GSI-MLC-PA v5.1 (Data-Driven Stratified Peeling + Ascending CC)                              |
+---------------------------------------------------------------------------------------------------+
```

---

### 8.3. Hệ thống chỉ số đánh giá (Evaluation Metrics)

Hệ thống ghi nhận và đối chiếu toàn diện 8 chỉ số:
1. **Full Macro-F1:** Đánh giá năng lực phân loại toàn bộ khi không từ chối.
2. **Selective Macro-F1 (tại $c = 0.30$):** Chất lượng phân loại trên các vị trí được mô hình chấp nhận dự đoán.
3. **Coverage ($\Gamma$):** Tỷ lệ bao phủ quyết định thực tế (phải đảm bảo $\ge 80\%$).
4. **Hamming Loss:** Tỷ lệ lỗi gán nhãn trung bình.
5. **Micro-F1:** Điểm F1 tổng hợp trên toàn bộ ma trận nhãn.
6. **Subset Accuracy:** Tỷ lệ dự đoán chính xác toàn bộ vector nhãn.
7. **Selection & Training Time (giây):** Đo lường trực tiếp tốc độ huấn luyện để chứng minh ưu thế giảm thiểu chi phí của v5.1 trước v5 cũ.
8. **Partition Stability (Jaccard Similarity giữa các folds):** Độ ổn định của tập $\mathcal{I}$ qua 5 folds.

---

### 8.4. Thiết kế các thí nghiệm bóc tách (Ablation Studies)

Nhằm làm nổi bật từng đóng góp khoa học trong bài báo, thiết kế 4 thí nghiệm bóc tách chuyên sâu:

1. **Ablation 1: Trật tự chuỗi CC trong tập $DL$ (CC Ordering Ablation)**
   - Cấu hình A: Tăng dần tổng tương quan (`ascending_correlation` - Đề xuất v5.1).
   - Cấu hình B: Giảm dần tổng tương quan (`descending_correlation` - v5 cũ).
   - Cấu hình C: Trật tự tự nhiên ngẫu nhiên (`natural_order`).
   - *Kỳ vọng:* Cấu hình A vượt trội cấu hình B và C về Selective Macro-F1 trên các tập có tương quan cao (`emotions`, `yeast`, `scene`).

2. **Ablation 2: Độ nhạy ngưỡng độc lập $\tau$ (Threshold Sensitivity Sweep)**
   - Quét lưới giá trị $\tau \in \{0.60, 0.65, 0.70, 0.75, 0.80, 0.85\}$.
   - Phân tích sự chuyển dịch tỷ lệ $|\mathcal{I}| / K$ và điểm Macro-F1 tương ứng.

3. **Ablation 3: Đơn tầng vs. Đa tầng (Single-stage vs. Multi-stage Peeling)**
   - So sánh $T_{\max} = 1$ (chỉ dừng ở $IL_1$) với $T_{\max} = 3$ (bóc tách đa tầng).
   - Đo lường mức cải thiện hiệu năng so với chi phí thời gian bỏ ra.

4. **Ablation 4: Hiệu quả của Hàm phạt phức tạp (Complexity Penalty Effectiveness)**
   - So sánh có phạt ($\lambda = 0.01$) vs. Không phạt ($\lambda = 0.0$).
   - Minh chứng hàm phạt giúp giảm 40-60% thời gian chạy mà không làm giảm điểm F1 có ý nghĩa thống kê ($p > 0.05$).

---

## 9. LỘ TRÌNH TRIỂN KHAI TỪNG BƯỚC (STEP-BY-STEP ACTION PLAN)

Kế hoạch triển khai được chia làm 5 giai đoạn rõ ràng:

```
+---------------------------------------------------------------------------------------------------+
|                            LỘ TRÌNH 5 BƯỚC TRIỂN KHAI V5.1                                        |
+---------------------------------------------------------------------------------------------------+
|  [BƯỚC 1: Xây dựng Module Phân Tầng]                                                              |
|       - Cài đặt `StratifiedPeelingSelector` trong `src/selection/stratified_peeling.py`          |
|       - Tích hợp Ascending Correlation Sorter cho tập DL                                          |
|                                                                                                   |
|  [BƯỚC 2: Tích hợp vào Model Core]                                                                |
|       - Cập nhật `GSIMLCPartialAbstentionClassifier` trong `src/models/gsi_mlc_pa.py`             |
|       - Hỗ trợ tham số `partition_strategy="stratified_peeling"` và `dl_order="ascending"`        |
|                                                                                                   |
|  [BƯỚC 3: Viết Unit Tests & Xác Minh Sanity]                                                      |
|       - Cài đặt `tests/test_v5_1_stratified.py` kiểm tra tính đúng đắn toán học                   |
|       - Kiểm thử chống rò rỉ dữ liệu (No Data Leakage between Val and Train)                     |
|                                                                                                   |
|  [BƯỚC 4: Chạy Benchmark 10 Datasets & Ablation Studies]                                         |
|       - Chạy thực nghiệm song song trên 10 datasets qua `src/evaluation/pipeline_v3.py`           |
|       - Ghi nhận metrics vào thư mục kết quả `results_v5_1/`                                      |
|                                                                                                   |
|  [BƯỚC 5: Tổng hợp Kết quả, Biểu đồ & Cập nhật Paper]                                             |
|       - Tạo bảng so sánh v5 vs v5.1 trong `tmp.md`                                                |
|       - Xuất các biểu đồ trực quan hóa (Correlation Impact, Peeling Stages, Latency Reduction)    |
|       - Cập nhật luận điểm khoa học vào bản thảo bài báo                                          |
+---------------------------------------------------------------------------------------------------+
```

### Chi tiết hành động từng bước:

- **Bước 1: Thiết kế & Cài đặt `StratifiedPeelingSelector`:**
  - Viết thuật toán tìm kiếm ngưỡng nhị phân $\theta_k^*$ nhanh $\mathcal{O}(N \log N)$.
  - Triển khai vòng lặp bóc tách $t = 1 \dots T_{\max}$ với biểu diễn xác suất mềm.
  - Cài đặt cơ chế sắp xếp tăng dần tổng tương quan trong $\mathcal{D}_{\text{residual}}$.
  - Cài đặt hàm phạt chi phí tính toán $\mathcal{P}(t, |\mathcal{D}|)$.

- **Bước 2: Nâng cấp `GSIMLCPartialAbstentionClassifier`:**
  - Đảm bảo tính tương thích ngược với API hiện tại.
  - Bổ sung cấu trúc kiểm toán `stratified_peeling_audit` vào `selection_config_`.

- **Bước 3: Kiểm thử Đơn vị & Bảo vệ Chất lượng:**
  - Kiểm tra tính xác định (deterministic) với `random_state`.
  - Kiểm tra trường hợp biên: tất cả nhãn độc lập ($|\mathcal{I}| = K$), không có nhãn độc lập ($|\mathcal{I}| = 0$).
  - Đảm bảo thời gian chạy phân hoạch giảm ít nhất 3x so với greedy v5.

- **Bước 4: Thực thi Thực nghiệm 10 Datasets:**
  - Chạy toàn diện trên cả 3 base learners: Logistic, SVM, MLP.
  - Lưu trữ đầy đủ checkpoints, kết quả per-label, và ma trận fold.

- **Bước 5: Báo cáo Khoa học & Xuất Bản Phẩm:**
  - Tổng hợp bảng so sánh Pairwise Win/Tie/Loss.
  - Phân tích trường hợp nghiên cứu cụ thể (Case Study trên `emotions` và `yeast`) làm nổi bật vì sao "Tương quan nhỏ lên đầu" giúp tăng F1.
  - Cập nhật tài liệu kỹ thuật hoàn chỉnh.
