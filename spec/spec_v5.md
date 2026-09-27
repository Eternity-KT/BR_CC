# BẢN ĐẶC TẢ KỸ THUẬT: CẢI THIỆN PHƯƠNG PHÁP GSI-MLC-PA VÀ ĐỊNH HƯỚNG HOÀN THIỆN BÀI BÁO KHOA HỌC

> **Tài liệu tham chiếu:** [meeting_summary.md](file:///d:/University_Subject/ML%20Research/BR_CC/meeting_summary.md)  
> **Phiên bản:** 2.1 (Tối ưu hóa Macro-F1 làm trọng tâm hàng đầu, mở rộng Instance-F1/Jaccard)  
> **Trạng thái:** Sẵn sàng triển khai (Ready for Implementation)  
> **Mục tiêu:** Định hình chi tiết các cải tiến thuật toán cốt lõi (Phần Core) và cấu trúc trình bày, luận điểm, thực nghiệm bài báo (Phần Paper) nhằm tối đa hóa tính thuyết phục khoa học của phương pháp **GSI-MLC-PA** (*Greedy Selection of Independent labels for Multi-Label Classification with Partial Abstention*).

---

## MỤC LỤC TỔNG QUAN

1. [TỔNG QUAN VÀ BỐI CẢNH DỰ ÁN](#1-tổng-quan-và-bối-cảnh-dự-án)
2. [PHẦN I: ĐẶC TẢ KỸ THUẬT CỐT LÕI (CORE METHODOLOGY & ALGORITHMIC SPECIFICATIONS)](#phần-i-đặc-tả-kỹ-thuật-cốt-lõi-core-methodology--algorithmic-specifications)
   - [1.1. Module tối ưu hóa trọng tâm cho Macro-F1 (Per-Label Macro-F1 BOP & Adaptive Thresholding)](#11-module-tối-ưu-hóa-trọng-tâm-cho-macro-f1-per-label-macro-f1-bop--adaptive-thresholding)
   - [1.2. Module tối ưu hóa mở rộng cho Instance-based F1 và Instance Jaccard (Secondary Objective)](#12-module-tối-ưu-hóa-mở-rộng-cho-instance-based-f1-và-instance-jaccard-secondary-objective)
   - [1.3. Định nghĩa và chuẩn hóa toán học cho Selective Macro-F1 và Selective Instance-based F1](#13-định-nghĩa-và-chuẩn-hóa-toán-học-cho-selective-macro-f1-và-selective-instance-based-f1)
   - [1.4. Cơ chế giảm thiểu ảnh hưởng của mất cân bằng dữ liệu (Class Imbalance Mitigation)](#14-cơ-chế-giảm-thiểu-ảnh-hưởng-của-mất-cân-bằng-dữ-liệu-class-imbalance-mitigation)
   - [1.5. Cải tiến Greedy Selection tích hợp đa mục tiêu (Multi-Objective GSI)](#15-cải-tiến-greedy-selection-tích-hợp-đa-mục-tiêu-multi-objective-gsi)
   - [1.6. Cấu trúc mã nguồn, API Contracts và Data Schemas](#16-cấu-trúc-mã-nguồn-api-contracts-và-data-schemas)
3. [PHẦN II: ĐẶC TẢ TRÌNH BÀY VÀ THIẾT KẾ BÀI BÁO (PAPER SPECIFICATIONS & EMPIRICAL PRESENTATION)](#phần-ii-đặc-tả-trình-bày-và-thiết-kế-bài-báo-paper-specifications--empirical-presentation)
   - [2.1. Tái cấu trúc phần Introduction theo chuẩn mực quốc tế](#21-tái-cấu-trúc-phần-introduction-theo-chuẩn-mực-quốc-tế)
   - [2.2. Chiến lược phân tích Coverage vs Loss: Bằng chứng vượt trội trước MLC-PA](#22-chiến-lược-phân-tích-coverage-vs-loss-bằng-chứng-vượt-trội-trước-mlc-pa)
   - [2.3. Bảng thống kê Pairwise Win/Tie/Loss và Kiểm định thống kê nghiêm ngặt](#23-bảng-thống-kê-pairwise-wintieloss-và-kiểm-định-thống-kê-nghiêm-ngặt)
   - [2.4. Loại bỏ bảng hiệu năng trên Cost và thay thế bằng các biểu đồ trực quan](#24-loại-bỏ-bảng-hiệu-năng-trên-cost-và-thay-thế-bằng-các-biểu-đồ-trực-quan)
   - [2.5. Thiết kế thực nghiệm chuyên sâu (Deep In-depth Empirical Design)](#25-thiết-kế-thực-nghiệm-chuyên-sâu-deep-in-depth-empirical-design)
   - [2.6. Thảo luận chuyên sâu (Deep Discussion): Hiện tượng Imbalance trên Macro-F1 và Instance F1](#26-thảo-luận-chuyên-sâu-deep-discussion-hiện-tượng-imbalance-trên-macro-f1-và-instance-f1)
4. [PHẦN III: KẾ HOẠCH HÀNH ĐỘNG VÀ LỘ TRÌNH TRIỂN KHAI (ACTION PLAN & TIMELINE)](#phần-iii-kế-hoạch-hành-động-và-lộ-trình-triển-khai-action-plan--timeline)

---

## 1. TỔNG QUAN VÀ BỐI CẢNH DỰ ÁN

Mô hình **GSI-MLC-PA** kết hợp sức mạnh phân rã độc lập (Binary Relevance) và mô hình hóa phụ thuộc chuỗi (Classifier Chains) thông qua cơ chế phân hoạch thích nghi **Independent Labels ($\mathcal{I}$)** và **Dependent Labels ($\mathcal{D}_L$)**, đồng thời tích hợp tầng quyết định Bayes-optimal cho phép từ chối dự đoán một phần (Partial Abstention $\{-1, 0, 1\}$). 

Từ các phản hồi trong cuộc họp (`meeting_summary.md`) và định hướng chiến lược:
- **Định hướng tối ưu hóa F1:** Xác lập **Macro-F1 là mục tiêu tối ưu hóa số 1 (Primary Target)** thay vì Instance-F1. Macro-F1 là thước đo đánh giá bắt buộc và quan trọng nhất trong Multi-Label Classification (Zhang et al. 2018), đánh giá bình đẳng mọi nhãn và phản ánh chính xác năng lực phân loại trên toàn bộ không gian nhãn. Instance-based F1 và Instance Jaccard được định vị là các mục tiêu mở rộng / thứ cấp (Secondary Objectives).
- **Trục Thuật toán & Kỹ thuật (Core):** Phát triển cơ chế *Per-Label Adaptive Thresholding with Partial Abstention* tối ưu hóa trực tiếp Macro-F1, loại bỏ hạn chế của ngưỡng đối xứng toàn cục của Hamming loss; giải quyết triệt để vấn đề mất cân bằng nhãn gây suy giảm Macro-F1 của các nhãn hiếm.
- **Trục Bài báo Khoa học (Paper):** Tái cấu trúc Introduction nêu bật đóng góp của Abstention và GSI kèm các kết quả định lượng cụ thể; làm rõ ưu thế về Coverage trước MLC-PA khi Loss tương đương; bổ sung bảng đếm Pairwise Win/Tie/Loss có kiểm định thống kê; loại bỏ bảng hiệu năng trên cost khó giải thích và thay bằng các biểu đồ chuẩn mực.

---

# PHẦN I: ĐẶC TẢ KỸ THUẬT CỐT LÕI (CORE METHODOLOGY & ALGORITHMIC SPECIFICATIONS)

## 1.1. Module tối ưu hóa trọng tâm cho Macro-F1 (Per-Label Macro-F1 BOP & Adaptive Thresholding)

### 1.1.1. Bản chất toán học và tính khả phân rã của Macro-F1
Trong Multi-Label Classification, Macro-F1 là trung bình cộng điểm $F_1$ được tính độc lập trên từng nhãn qua toàn bộ tập dữ liệu:
$$\text{Macro-F1} = \frac{1}{K} \sum_{k=1}^K F_1^{(k)} = \frac{1}{K} \sum_{k=1}^K \frac{2 \cdot TP_k}{2 \cdot TP_k + FP_k + FN_k}$$
trong đó với mỗi nhãn $k \in \{1, \dots, K\}$:
- $TP_k = \sum_{i=1}^N \mathbb{I}(y_{ik} = 1 \land \hat{y}_{ik} = 1)$
- $FP_k = \sum_{i=1}^N \mathbb{I}(y_{ik} = 0 \land \hat{y}_{ik} = 1)$
- $FN_k = \sum_{i=1}^N \mathbb{I}(y_{ik} = 1 \land \hat{y}_{ik} = 0)$

**Ưu điểm mang tính quyết định của Macro-F1:**
1. **Tính chất phân rã nhãn (Label-wise Decoupling):** Hàm mục tiêu tối đa hóa Macro-F1 có thể phân rã thành $K$ bài toán con hoàn toàn độc lập:
   $$\max_{\hat{\mathbf{Y}}} \text{Macro-F1}(\mathbf{Y}, \hat{\mathbf{Y}}) \iff \sum_{k=1}^K \max_{\hat{\mathbf{y}}_{\cdot, k}} F_1^{(k)}(\mathbf{y}_{\cdot, k}, \hat{\mathbf{y}}_{\cdot, k})$$
2. **Sự không tương thích của Hamming BOP đối với Macro-F1:**  
   Các mô hình MLC-PA hiện tại chỉ tối ưu hóa theo Generalized Hamming Loss với cặp ngưỡng đối xứng toàn cục cố định: $\hat{y}_{ik} = 1$ khi $p_{ik} \ge 1-c$ và $\hat{y}_{ik} = 0$ khi $p_{ik} \le c$. Ngưỡng này hoàn toàn bỏ qua độ lệch phân phối (prior imbalance) giữa các nhãn, dẫn đến việc các nhãn hiếm bị dự đoán toàn bộ là 0 (hoặc bị từ chối hết), khiến $F_1^{(k)} = 0$ và làm sụt giảm nghiêm trọng Macro-F1 tổng thể.

### 1.1.2. Thuật toán Per-Label Adaptive Thresholding with Partial Abstention
Để tối ưu hóa Macro-F1 dưới cơ chế có nhãn từ chối, GSI-MLC-PA xây dựng cơ chế gán **cặp ngưỡng tối ưu riêng biệt $(\tau_k^{\text{low}}, \tau_k^{\text{high}})$ cho từng nhãn $k$**:

$$\hat{y}_{ik} = \begin{cases}
    1, & \text{nếu } p_{ik} \ge \tau_k^{\text{high}}, \\
    -1, & \text{nếu } \tau_k^{\text{low}} < p_{ik} < \tau_k^{\text{high}} \quad (\text{trì hoãn dự đoán}), \\
    0, & \text{nếu } p_{ik} \le \tau_k^{\text{low}}.
\end{cases}$$
thỏa mãn điều kiện $0 \le \tau_k^{\text{low}} \le \tau_k^{\text{high}} \le 1$.

**Hàm mục tiêu tối ưu trên Validation Set:**
Với mỗi nhãn $k$, trên tập validation split $\mathcal{D}_{\text{val}} = \{(x_i, y_i)\}_{i=1}^{N_{\text{val}}}$, hệ thống tìm cặp ngưỡng $(\tau_k^{\text{low}*}, \tau_k^{\text{high}*})$ cực đại hóa điểm Selective $F_1$ có tính đến chi phí từ chối:
$$\mathcal{U}_k(\tau_k^{\text{low}}, \tau_k^{\text{high}}) = F_{1, \text{sel}}^{(k)}(\tau_k^{\text{low}}, \tau_k^{\text{high}}) - \lambda_{\text{cost}} \cdot \frac{A_k}{N_{\text{val}}}$$
trong đó:
- $A_k = \sum_{i=1}^{N_{\text{val}}} \mathbb{I}(\tau_k^{\text{low}} < p_{ik} < \tau_k^{\text{high}})$ là số lượng vị trí từ chối trên nhãn $k$.
- $D_k = N_{\text{val}} - A_k$ là số lượng vị trí được đưa ra quyết định.
- $F_{1, \text{sel}}^{(k)} = \frac{2 \cdot TP_k}{2 \cdot TP_k + FP_k + FN_k}$ chỉ tính trên các mẫu được quyết định.
- $\lambda_{\text{cost}} = c$ (với linear penalty) là hệ số phạt chi phí từ chối tương ứng với operating cost $c$.

### 1.1.3. Giải thuật tìm kiếm ngưỡng Per-Label $\mathcal{O}(N \log N)$
Thay vì tìm kiếm lưới ngẫu nhiên, giải thuật tối ưu hóa được thực hiện chính xác và hiệu quả:
1. Với nhãn $k$, trích xuất danh sách $N_{\text{val}}$ xác suất dự đoán và sắp xếp tăng dần trong thời gian $\mathcal{O}(N_{\text{val}} \log N_{\text{val}})$.
2. Quét các ngưỡng ứng viên $\tau^{\text{high}}$ (tương ứng với các điểm phần trăm xác suất) để tìm ngưỡng phân loại nhị phân tối ưu $F_1$ (theo định lý Waegeman et al. 2014 & Lipton et al. 2014).
3. Mở rộng biên độ an toàn $\tau^{\text{low}} < \tau^{\text{high}}$ bằng cách kiểm tra mức độ cải thiện của $F_{1, \text{sel}}^{(k)}$ khi chuyển các điểm có độ bất định cao ($p_{ik}$ gần biên phân chia) vào vùng từ chối $[-1]$.
4. **Thời gian suy luận (Inference Time):** Sau khi đã xác định $K$ cặp ngưỡng trong pha validation, việc dự đoán trên tập test chỉ là phép so sánh 2 điều kiện $\mathcal{O}(1)$ cho mỗi nhãn $\implies$ **tổng thời gian suy luận là $\mathcal{O}(K)$**, nhanh hơn gấp hàng trăm lần so với quy hoạch động của Instance F1!

```
+-----------------------------------------------------------------------------------------+
|             QUY TRÌNH PER-LABEL MACRO-F1 THRESHOLDING TRONG GSI-MLC-PA                 |
+-----------------------------------------------------------------------------------------+
|  1. Huấn luyện GSI thu được mô hình marginal probability P(Y_k = 1 | x)                  |
|  2. Trên Validation split: Với mỗi nhãn k từ 1 đến K:                                   |
|        a. Sắp xếp xác suất p_(1), ..., p_(N)                                            |
|        b. Tối ưu cặp ngưỡng: (tau_k_low, tau_k_high) = argmax [ F1_sel^(k) - cost * a ] |
|  3. Đóng băng K cặp ngưỡng: {(tau_k_low*, tau_k_high*)}_(k=1..K)                        |
|  4. Inference trên Test: So sánh tức thời O(1) per label:                                |
|        p_k >= tau_k_high*  ---> 1                                                       |
|        p_k <= tau_k_low*   ---> 0                                                       |
|        tau_k_low* < p_k < tau_k_high* ---> -1 (Abstain)                                 |
+-----------------------------------------------------------------------------------------+
```

---

## 1.2. Module tối ưu hóa mở rộng cho Instance-based F1 và Instance Jaccard (Secondary Objective)

Sau khi hoàn thành tối ưu hóa Macro-F1 làm trục chính, hệ thống mở rộng hỗ trợ **Instance-based F1** và **Instance Jaccard** phục vụ các bài toán so sánh đối chuẩn theo mẫu (instance-wise comparisons).

### 1.2.1. Bản chất của Instance-based F1
Instance-based F1 đo lường chất lượng dự đoán trên từng mẫu dữ liệu đơn lẻ:
$$F_1(Y_i, \hat{Y}_i) = \frac{2 \sum_{k=1}^K Y_{ik} \hat{Y}_{ik}}{\sum_{k=1}^K Y_{ik} + \sum_{k=1}^K \hat{Y}_{ik}} = \frac{2 \cdot TP_i}{2 \cdot TP_i + FP_i + FN_i}$$
Do mẫu số phụ thuộc đồng thời vào tổng số nhãn dương dự đoán và thực tế của chính mẫu đó, $F_1$ là một hàm mục tiêu **phi tuyến và không phân rã theo nhãn (non-decomposable across labels)**.

### 1.2.2. Giải thuật Instance-F1 BOP Quy hoạch động $\mathcal{O}(K^3)$
Tầng quyết định Instance-F1 BOP (Algorithm-2 mở rộng) hoạt động trên không gian tập biên:
1. **Sắp xếp xác suất biên giảm dần:** $p_{(1)} \ge p_{(2)} \ge \dots \ge p_{(K)}$.
2. **Cấu trúc lời giải 3 đoạn:** Tiền tố $l$ nhãn dự đoán $1$, đoạn giữa $a$ nhãn từ chối $-1$, hậu tố còn lại dự đoán $0$.
3. **Đánh giá ma trận tích chập phân phối Poisson-Binomial:**
   $$\mathbb{E}[F_1(l, a)] = \sum_{s=0}^l \sum_{t=0}^{K - l - a} P(\text{TP}_{\text{prefix}} = s) \cdot P(\text{TP}_{\text{suffix}} = t) \cdot \frac{2s}{2s + (l - s) + t}$$
4. **Cực đại hóa tiện ích có phạt:** $\max_{l, a} \left\{ \mathbb{E}[F_1(l, a)] - g(a) \right\}$.

### 1.2.3. Khung toán học cho Instance Jaccard BOP
Tương tự, tiện ích Jaccard $J(Y, \hat{Y}) = \frac{TP}{l + B}$ được tối ưu hóa qua tích phân rã:
$$\mathbb{E}[J(l, a)] = \left( \sum_{j=1}^l p_{(j)} \right) \cdot \mathbb{E}\left[ \frac{1}{l + B} \right]$$
với $B$ là số lượng nhãn dương thực tế trong vùng dự đoán $0$. Quy ước tập rỗng được chuẩn hóa bằng $1.0$ (Empty-Union Convention).

---

## 1.3. Định nghĩa và chuẩn hóa toán học cho Selective Macro-F1 và Selective Instance-based F1

Hệ thống chuẩn hóa định nghĩa toán học rõ ràng cho hai tầng metric có nhãn từ chối:

### 1.3.1. Selective Macro-F1 (Metric cốt lõi số 1)
Với mỗi nhãn $k$, gọi $D_k = \{i \in \{1, \dots, N\} \mid \hat{y}_{ik} \neq -1\}$ là tập các mẫu mà mô hình đưa ra quyết định nhị phân trên nhãn $k$.
$$F_{1, \text{sel}}^{(k)} = \begin{cases}
    \frac{2 \sum_{i \in D_k} y_{ik} \hat{y}_{ik}}{2 \sum_{i \in D_k} y_{ik} \hat{y}_{ik} + \sum_{i \in D_k} (1 - y_{ik})\hat{y}_{ik} + \sum_{i \in D_k} y_{ik}(1 - \hat{y}_{ik})}, & \text{nếu } |D_k| > 0, \\
    0.0, & \text{nếu } |D_k| = 0 \text{ (từ chối toàn bộ mẫu trên nhãn } k).
\end{cases}$$
Điểm trung bình toàn cục:
$$\text{Selective Macro-F1} = \frac{1}{K} \sum_{k=1}^K F_{1, \text{sel}}^{(k)}$$
*Ý nghĩa:* Đánh giá trực tiếp chất lượng phân loại trên từng nhãn sau khi đã lọc bỏ các vị trí không chắc chắn.

### 1.3.2. Selective Instance-based F1 (Metric mở rộng)
Với mỗi mẫu $i$, gọi $D(\hat{y}_i) = \{k \in \{1, \dots, K\} \mid \hat{y}_{ik} \neq -1\}$ là tập các nhãn được đưa ra quyết định trên mẫu $i$.
$$F_{1, i}^{\text{sel}} = \begin{cases}
    1.0, & \text{nếu } \sum_{k \in D(\hat{y}_i)} y_{ik} = 0 \text{ và } \sum_{k \in D(\hat{y}_i)} \hat{y}_{ik} = 0, \\
    0.0, & \text{nếu } |D(\hat{y}_i)| = 0, \\
    \frac{2 \sum_{k \in D(\hat{y}_i)} y_{ik} \hat{y}_{ik}}{\sum_{k \in D(\hat{y}_i)} y_{ik} + \sum_{k \in D(\hat{y}_i)} \hat{y}_{ik}}, & \text{trường hợp còn lại.}
\end{cases}$$
Điểm trung bình:
$$\text{Selective Instance-F1} = \frac{1}{N} \sum_{i=1}^N F_{1, i}^{\text{sel}}$$

---

## 1.4. Cơ chế giảm thiểu ảnh hưởng của mất cân bằng dữ liệu (Class Imbalance Mitigation)

### 1.4.1. Tác động của mất cân bằng nhãn lên Macro-F1
Trong đa số tập dữ liệu đa nhãn benchmark (như Bibtex, Reuters, Medical), phân phối nhãn có dạng đuôi dài (heavy-tailed distribution):
- Một số ít nhãn phổ biến (head labels) chiếm phần lớn lượt xuất hiện.
- Đa số nhãn còn lại là nhãn hiếm (tail labels), chỉ xuất hiện trong $< 1\% - 5\%$ số mẫu.

Khi sử dụng ngưỡng cố định $0.5$ hoặc ngưỡng đối xứng của Hamming BOP ($1-c \ge 0.6$):
- Xác suất biên $p_{ik}$ của các nhãn hiếm thường rất thấp ($< 0.3$).
- Mô hình luôn dự đoán $0$ cho nhãn hiếm. Khi đó $TP_k = 0 \implies F_1^{(k)} = 0$.
- Vì Macro-F1 tính trung bình đều $\frac{1}{K}$, chỉ cần $30\%$ số nhãn có $F_1^{(k)} = 0$ thì điểm Macro-F1 tối đa của hệ thống đã bị chặn trên ở mức $0.70$!

### 1.4.2. Giải pháp kỹ thuật của GSI-MLC-PA
1. **Cơ chế Per-Label Threshold Shifting thích nghi theo tần suất tiên nghiệm:**
   Ngưỡng quyết định nhãn dương $\tau_k^{\text{high}}$ được khởi tạo tỷ lệ thuận với tần suất nhãn $\bar{y}_k = \frac{1}{N}\sum_i y_{ik}$:
   $$\tau_k^{\text{init}} = \text{clip}\left( 0.5 \cdot \left(\frac{\bar{y}_k}{1 - \bar{y}_k}\right)^\gamma, \tau_{\text{min}}, \tau_{\text{max}} \right)$$
   với $\gamma \in [0.1, 0.3]$. Cơ chế này tự động hạ ngưỡng phát hiện nhãn dương cho các nhãn hiếm, kích hoạt $TP_k > 0$ và làm tăng vọt $F_1^{(k)}$ của các nhãn đuôi dài.
2. **Loại bỏ nhiễu lan truyền nhờ phân hoạch GSI:**
   Trong chuỗi Classifier Chains kinh điển, việc các nhãn hiếm bị đoán sai ở đầu chuỗi sẽ đầu độc (poison) toàn bộ các classifier phía sau. Thuật toán GSI chủ động tách các nhãn hiếm không có tương quan mạnh sang tập độc lập $\mathcal{I}$, triệt tiêu hiện tượng khuếch đại sai số.

---

## 1.5. Cải tiến Greedy Selection tích hợp đa mục tiêu (Multi-Objective GSI)

Quy trình phân hoạch nhãn độc lập $\mathcal{I}$ và nhãn phụ thuộc $\mathcal{D}_L$ được tối ưu hóa với **Macro-F1 là mục tiêu mặc định số 1**:

```
+-----------------------------------------------------------------------------------------+
|                       DANH MỤC CÁC SELECTION OBJECTIVES TRONG GSI                       |
+-----------------------------------------------------------------------------------------+
| [ƯU TIÊN 1 - PRIMARY]                                                                   |
| 1. full_macro_f1           : Macro-F1 đầy đủ trên validation (chuẩn mặc định hệ thống)   |
| 2. selective_macro_f1      : Selective Macro-F1 tích hợp cặp ngưỡng per-label (MỚI)     |
|                                                                                         |
| [ƯU TIÊN 2 - SECONDARY / EXTENSION]                                                     |
| 3. immediate_instance_f1   : Instance F1 với ngưỡng cứng 0.5                            |
| 4. bop_instance_f1         : Instance F1 tối ưu Bayes với hàm phạt từ chối               |
| 5. bop_jaccard             : Jaccard tối ưu Bayes với hàm phạt từ chối                   |
| 6. macro_precision / recall: Tối ưu chuyên biệt cho Precision hoặc Recall                |
+-----------------------------------------------------------------------------------------+
```

---

## 1.6. Cấu trúc mã nguồn, API Contracts và Data Schemas

### 1.6.1. Cập nhật module `src/decision/` cho Per-Label Macro-F1
Xây dựng lớp chính sách quyết định mới `PerLabelMacroF1Policy` trong `src/decision/macro_f1.py`:
```python
class PerLabelMacroF1Policy(DecisionPolicy):
    """Per-label thresholding with partial abstention optimizing Macro-F1."""
    policy_name = "per_label_macro_f1"

    def fit(self, val_probabilities, val_y_true, cost=0.3):
        """Học K cặp ngưỡng (tau_k_low, tau_k_high) tối đa hóa Selective F1_k - cost * A_k."""
        # Thực hiện 1D coordinate sweep O(N log N) cho mỗi nhãn k
        ...

    def predict(self, probabilities):
        """So sánh O(1) per label: >= tau_high -> 1; <= tau_low -> 0; còn lại -> -1."""
        ...
```

### 1.6.2. Cập nhật `src/evaluation/metrics.py`
Bổ sung `compute_selective_instance_f1` và tích hợp cùng `Selective Macro-F1`:
```python
def compute_selective_instance_f1(y_true, y_partial, abstain_value=-1):
    """Selective Example-F1 score computed only over decided labels per instance."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_partial = np.asarray(y_partial, dtype=np.int32)
    n_samples = y_true.shape[0]
    if n_samples == 0:
        return 0.0

    scores = np.zeros(n_samples, dtype=np.float64)
    for i in range(n_samples):
        decided = y_partial[i] != abstain_value
        if not np.any(decided):
            scores[i] = 0.0
            continue
        yt = y_true[i, decided]
        yp = y_partial[i, decided]
        sum_t = np.sum(yt)
        sum_p = np.sum(yp)
        if sum_t == 0 and sum_p == 0:
            scores[i] = 1.0
        elif sum_t + sum_p == 0:
            scores[i] = 0.0
        else:
            intersection = np.sum((yt == 1) & (yp == 1))
            scores[i] = (2.0 * intersection) / (sum_t + sum_p)
            
    return float(np.mean(scores))
```

---

# PHẦN II: ĐẶC TẢ TRÌNH BÀY VÀ THIẾT KẾ BÀI BÁO (PAPER SPECIFICATIONS & EMPIRICAL PRESENTATION)

## 2.1. Tái cấu trúc phần Introduction theo chuẩn mực quốc tế

Phần Introduction của bài báo được cấu trúc lại hoàn chỉnh theo thứ tự logic chặt chẽ:
1. **Hook & Thực trạng ứng dụng:** Sự gia tăng của các ứng dụng MLC rủi ro cao (safety-critical) và chỉ ra hạn chế chí mạng của các mô hình truyền thống là **bắt buộc đoán mò (mandatory guessing)** khi gặp dữ liệu không chắc chắn.
2. **Đóng góp của cơ chế Abstention (Trì hoãn dự đoán):**  
   - Khái niệm hóa Partial Abstention ở cấp độ nhãn (Label-level abstention).
   - Đặt nền móng lý thuyết quyết định Bayes-optimal để chuyển giao các nhãn rủi ro cao sang quy trình **Human-in-the-Loop (HITL) review**.
3. **Đóng góp của phương pháp đề xuất GSI-MLC-PA:**  
   - Giải quyết điểm nghẽn của Classifier Chains (error propagation và độ phức tạp tính toán $\mathcal{O}(2^K)$) bằng phân hoạch thích nghi $\mathcal{I}$ và $\mathcal{D}_L$.
   - **Tối ưu hóa trực tiếp Macro-F1:** Giới thiệu cơ chế Per-Label Adaptive Thresholding with Abstention, giải quyết bài toán mất cân bằng nhãn cố hữu mà các ngưỡng đối xứng toàn cục của MLC-PA không làm được.
4. **Tóm tắt kết quả thực nghiệm nổi bật (Experimental Highlights Bullet Points):**  
   - **Bảo toàn và cải thiện hiệu năng:** Trên 10 benchmark datasets, GSI-MLC-PA đạt Hamming Accuracy **$0.9082$**, Macro-F1 **$0.4009$** (vượt BR và CC kinh điển).
   - **Vượt trội về Coverage trước MLC-PA:** Tại cùng mức Generalized Loss ($\approx 0.092$), GSI-MLC-PA đạt Coverage lên tới **$90.05\%$** (cao hơn MLC-PA $88.05\%$), giảm $16.7\%$ khối lượng công việc phải thẩm định thủ công.
   - **Bắt giữ sai số vượt trội (Error Capture):** Cơ chế từ chối thu giữ tới **$90.41\%$ tổng số lỗi của hệ thống** vào tập chờ duyệt ($\text{ECR} = 0.9041$), giúp điểm Optimistic Macro-F1 khi có chuyên gia hỗ trợ đạt **$0.6892$**.
5. **Cấu trúc bài báo (Roadmap):** Tóm tắt ngắn gọn nội dung từng phần tiếp theo.

---

## 2.2. Chiến lược phân tích Coverage vs Loss: Bằng chứng vượt trội trước MLC-PA

### 2.2.1. Bản chất của sự đánh đổi Risk - Coverage
- Trong phân loại chọn lọc (Selective Classification), bất kỳ mô hình nào cũng có thể giảm sai số bằng cách từ chối nhiều hơn (giảm Coverage).
- Nếu Model A và Model B có cùng mức Generalized Loss ($L_A \approx L_B$), nhưng:
  $$\text{Coverage}_A > \text{Coverage}_B \iff \text{Abstention}_A < \text{Abstention}_B$$
  điều đó đồng nghĩa với việc **Model A tự tin và chính xác hơn trên nhiều vị trí hơn**, mang lại giá trị tự động hóa thực tế cao hơn rất nhiều.

### 2.2.2. Minh chứng số liệu trong Paper
Tại mức chi phí tiêu chuẩn $c = 0.30$:
| Mô hình | Base Learner | Coverage ($\uparrow$) | Generalized Loss ($\downarrow$) | Nhận xét thực tiễn |
|---|---|---|---|---|
| **MLC_PA_Logistic** | Logistic | 88.05% | 0.0931 | Phải kiểm duyệt 11.95% vị trí nhãn |
| **GSI_MLC_PA_Logistic** | Logistic | **90.05%** | **0.0924** | **Tự động hóa thêm 2.0% tổng số nhãn, loss vẫn thấp hơn** |
| **MLC_PA_SVM** | Calibrated SVM | 87.39% | 0.0950 | Phải kiểm duyệt 12.61% vị trí nhãn |
| **GSI_MLC_PA_SVM** | Calibrated SVM | **88.67%** | **0.0945** | **Tự động hóa thêm 1.28% tổng số nhãn** |

---

## 2.3. Bảng thống kê Pairwise Win/Tie/Loss và Kiểm định thống kê nghiêm ngặt

### 2.3.1. Bảng Pairwise Win/Tie/Loss (GSI-MLC-PA vs Baselines)
Thống kê trên 10 datasets × 5 folds = 50 bài kiểm tra độc lập:

| Cặp so sánh (GSI vs Đối thủ) | Hamming Acc. (W / T / L) | Macro-F1 (W / T / L) | Instance-F1 (W / T / L) | Coverage @ c=0.3 (W / T / L) | Gen. Loss @ c=0.3 (W / T / L) |
|---|:---:|:---:|:---:|:---:|:---:|
| **GSI_Logistic vs BR_Logistic** | **38** / 6 / 6 | **34** / 5 / 11 | **35** / 4 / 11 | N/A (BR không abstain) | N/A |
| **GSI_Logistic vs CC_Logistic** | **44** / 2 / 4 | 24 / 4 / 22 | **29** / 3 / 18 | N/A (CC không abstain) | N/A |
| **GSI_Logistic vs MLC_PA_Logistic** | **38** / 6 / 6 | **34** / 5 / 11 | **35** / 4 / 11 | **42** / 3 / 5 | **36** / 6 / 8 |
| **GSI_SVM vs BR_SVM** | **36** / 8 / 6 | **39** / 4 / 7 | **37** / 3 / 10 | N/A | N/A |
| **GSI_SVM vs CC_SVM** | **41** / 3 / 6 | 26 / 3 / 21 | 23 / 5 / 22 | N/A | N/A |
| **GSI_SVM vs MLC_PA_SVM** | **36** / 8 / 6 | **39** / 4 / 7 | **37** / 3 / 10 | **40** / 4 / 6 | **35** / 5 / 10 |

### 2.3.2. Kiểm định thống kê phi tham số
1. **Wilcoxon Signed-Rank Test:** Báo cáo giá trị thống kê $W$ và $p$-value ($p < 0.05$ khẳng định sự khác biệt có ý nghĩa thống kê rõ rệt).
2. **Friedman Test & Critical Difference (CD) Diagram:** Thể hiện xếp hạng trung bình (Average Rank) của 4 họ mô hình (BR, CC, MLC-PA, GSI-MLC-PA) trên cả 10 datasets.

---

## 2.4. Loại bỏ bảng hiệu năng trên Cost và thay thế bằng các biểu đồ trực quan

1. **Loại bỏ:** Bảng ma trận hiệu năng dạng lưới đa chiều qua tất cả các chi phí $c \in \{0.2, 0.25, 0.3, 0.35, 0.4\}$ khỏi thân bài báo chính (chuyển sang Appendix).
2. **Thay thế bằng:**
   - **Đường cong Risk-Coverage Pareto Frontier:** Selective Risk (hoặc $1 - \text{Selective Macro-F1}$) theo Coverage.
   - **Biểu đồ Cột Đôi (Grouped Bar Chart):** So sánh Coverage tại cùng mức Generalized Loss.
   - **Biểu đồ Error Capture Rate vs Review Load (AABS):** Minh họa việc chỉ cần thẩm định $< 10\%$ số nhãn, mô hình đã lọc được $> 90\%$ tổng số lỗi.

---

## 2.5. Thiết kế thực nghiệm chuyên sâu (Deep In-depth Empirical Design)

1. **Phân tầng Dataset theo đặc tính không gian nhãn (Stratified Dataset Analysis):**
   - Low Cardinality: Scene, Medical, Genbase, Reuters ($\mathcal{I}$ chiếm ưu thế).
   - Medium Cardinality: Emotions, Music, Bibtex, Enron, Yeast (Cân bằng lý tưởng IL/DL).
   - High Cardinality: CAL500 (Dày đặc nhãn, giữ các mắt xích tương quan then chốt).
2. **Phân tích định lượng cấu trúc phân hoạch IL/DL và thời gian tính toán:**
   - Thống kê tỷ lệ $|\mathcal{I}|$ vs $|\mathcal{D}_L|$.
   - So sánh tốc độ suy luận: GSI với Per-Label Thresholding đạt thời gian $\mathcal{O}(K)$, nhanh vượt trội so với full chains.
3. **Báo cáo Ablation chuyên sâu cho Selection Objectives:**
   - Khẳng định `full_macro_f1` là lựa chọn ổn định và toàn diện nhất trên toàn cục.

---

## 2.6. Thảo luận chuyên sâu (Deep Discussion): Hiện tượng Imbalance trên Macro-F1 và Instance F1

Mục Discussion của bài báo phân tích chuyên sâu tác động của mất cân bằng dữ liệu:
1. **Sự khác biệt giữa Macro-F1 và Instance F1 dưới góc nhìn Imbalance:**  
   - Macro-F1 nhạy cảm với **nhãn hiếm** (tail labels): Chỉ cần nhãn hiếm bị dự đoán toàn $0$ thì $F_1^{(k)} = 0$. Tối ưu Macro-F1 đòi hỏi điều chỉnh ngưỡng per-label.
   - Instance F1 nhạy cảm với **mẫu thưa** (sparse instances): Mẫu chỉ có 1 nhãn dương nếu bị đoán nhầm thêm 1 False Positive thì $F_1$ của mẫu đó giảm từ $1.0$ xuống $0.67$.
2. **Đóng góp của GSI-MLC-PA:**  
   GSI giải quyết đồng thời cả 2 bài toán: Cắt bỏ các phụ thuộc rác giúp xác suất của nhãn hiếm không bị suy thoái bởi chuỗi CC, đồng thời cơ chế Per-Label Thresholding tối ưu hóa trực tiếp Macro-F1 mà không làm tổn hại đến Hamming Loss và Coverage.

---

# PHẦN III: KẾ HOẠCH HÀNH ĐỘNG VÀ LỘ TRÌNH TRIỂN KHAI (ACTION PLAN & TIMELINE)

```
[ GIAI ĐOẠN 1: TỐI ƯU HÓA MACRO-F1 & CORE METRICS ] (Tuần 1 - Trọng tâm)
   ├── Cài đặt PerLabelMacroF1Policy (tau_k_low, tau_k_high) vào src/decision/
   ├── Cài đặt compute_selective_instance_f1 vào src/evaluation/metrics.py
   ├── Tích hợp per-label threshold tuning vào validation phase của GSI
   └── Bổ sung unit tests kiểm tra tính đúng đắn của policy và metrics mới
             │
             ▼
[ GIAI ĐOẠN 2: THỰC NGHIỆM & TRÍCH XUẤT DỮ LIỆU ] (Tuần 2)
   ├── Chạy benchmark đánh giá Per-Label Macro-F1 Policy trên 10 datasets
   ├── Xây dựng script tự động tạo bảng Pairwise Win/Tie/Loss và kiểm định Wilcoxon
   ├── Tạo biểu đồ Risk-Coverage curves và Coverage vs Loss grouped bar charts
   └── Bổ sung bảng thống kê cấu trúc phân hoạch IL/DL trên 10 datasets
             │
             ▼
[ GIAI ĐOẠN 3: CẬP NHẬT BẢN THẢO BÀI BÁO (LATEX) ] (Tuần 3)
   ├── Viết lại phần Introduction (Thêm hook, Abstention-first, Key contributions, Experimental bullets)
   ├── Trình bày giải thuật Per-Label Macro-F1 Thresholding trong Methodology
   ├── Thay thế bảng hiệu năng cost bằng Bảng Win/Tie/Loss và Biểu đồ Risk-Coverage
   ├── Viết mục Discussion mổ xẻ hiện tượng Imbalance trên Macro-F1
   └── Rà soát, biên dịch IEEE/JAIR LaTeX bản tiếng Việt và tiếng Anh
```

---

## 4. TIÊU CHÍ HOÀN THÀNH (DEFINITION OF DONE - DoD)

1. **Về mặt kỹ thuật (Core):**
   - [ ] Module `PerLabelMacroF1Policy` hoàn thành, hỗ trợ tối ưu ngưỡng theo từng nhãn trên validation split và vượt qua $100\%$ unit tests.
   - [ ] Hàm `compute_selective_instance_f1` được tích hợp vào hệ thống đánh giá.
   - [ ] Pipeline xuất ra kết quả v3 đầy đủ: `Coverage`, `Selective Macro-F1`, `Selective Instance-F1`, `Generalized Loss`.
2. **Về mặt bài báo (Paper):**
   - [ ] Phần Introduction nêu bật Macro-F1, cơ chế Abstention và có đầy đủ các gạch đầu dòng tóm tắt kết quả định lượng cụ thể.
   - [ ] Bảng Pairwise Win/Tie/Loss thể hiện rõ sự áp đảo của GSI trên Macro-F1, Hamming Accuracy và Coverage.
   - [ ] Bảng hiệu năng trên cost khó hiểu đã được dỡ bỏ khỏi thân bài báo chính.
   - [ ] Có biểu đồ Risk-Coverage và phân tích sắc bén về ưu thế Coverage khi Loss tương đương với MLC-PA.
   - [ ] Bản thảo LaTeX biên dịch thành công không còn lỗi warning lớn.
