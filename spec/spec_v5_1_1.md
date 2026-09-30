# BẢN ĐẶC TẢ KỸ THUẬT & KẾ HOẠCH NÂNG CẤP CHI TIẾT: PHIÊN BẢN V5.1.1
## TỐI ƯU HÓA CHUỖI CC THEO NGƯỠNG TƯƠNG QUAN, CHUẨN HÓA ĐẶC TRƯNG TẦNG ĐỘC LẬP, KIỂM TOÁN PHÂN TẦNG VÀ KHUNG THỰC NGHIỆM BÓC TÁCH (ABLATION STUDY)

> **Tài liệu tham chiếu:** [meeting_summary.md](file:///d:/University_Subject/ML%20Research/BR_CC/meeting_summary.md) | [spec/spec_V5_1.md](file:///d:/University_Subject/ML%20Research/BR_CC/spec/spec_V5_1.md) | [tmp.md](file:///d:/University_Subject/ML%20Research/BR_CC/tmp.md)  
> **Phiên bản:** 5.1.1 (Threshold-Filtered Sparse CC, Normalized Augmented Features, Fine-Grained Layer Audit & Layer-Wise Ablation Framework)  
> **Kế thừa từ:** [spec/spec_V5_1.md](file:///d:/University_Subject/ML%20Research/BR_CC/spec/spec_V5_1.md) (Data-Driven Stratified Peeling & Ascending Correlation)  
> **Mục tiêu:** Hiện thực hóa toàn bộ 5 yêu cầu cốt lõi (`**core**`) và 2 yêu cầu mở rộng (`**option**`) được ghi nhận trong `meeting_summary.md`:
> 1. Thiết lập khung thực nghiệm bóc tách chi tiết (Ablation Framework) đánh giá định lượng tác động độc lập của từng tầng nhãn ($IL_1, IL_2, IL_3, DL$) đến hiệu năng cuối cùng.
> 2. Đảm bảo toàn bộ tập nhãn độc lập $IL$ và dữ liệu gốc $X$ đóng vai trò làm không gian đặc trưng tĩnh (Static Context Features) cho chuỗi CC của tập $DL$.
> 3. Tích hợp cơ chế kiểm toán chi tiết số lượng và danh sách nhãn được bóc tách qua từng tầng xuất ra báo cáo và file kết quả.
> 4. Chuẩn hóa vector xác suất của nhãn $IL$ theo cùng thang đo phân phối của ma trận dữ liệu gốc $X$ trước khi đưa vào các tầng sau.
> 5. Tái thiết kế Classifier Chain cho tập $DL$: Thay vì chuỗi dày đặc (Dense CC) sắp xếp theo tổng tương quan, chuyển sang đồ thị phụ thuộc chọn lọc (Sparse Threshold CC) chỉ kết nối các nhãn có tương quan vượt ngưỡng ($\theta_{\text{corr}} \in [0.7, 0.8]$), triệt tiêu nhiễu từ các nhãn không liên quan.

---

## MỤC LỤC TỔNG QUAN

1. [BỐI CẢNH VÀ ĐỘNG LỰC NÂNG CẤP V5.1.1](#1-bối-cảnh-và-động-lực-nâng-cấp-v511)
   - 1.1. Phân tích 5 yêu cầu cốt lõi (`core`) từ cuộc họp
   - 1.2. Phân tích 2 yêu cầu mở rộng (`option`)
   - 1.3. Sơ đồ kiến trúc tổng thể GSI-MLC-PA v5.1.1
2. [MODULE 1: KHUNG THỰC NGHIỆM BÓC TÁCH ẢNH HƯỞNG TỪNG TẦNG NHÃN (LAYER-WISE ABLATION FRAMEWORK)](#2-module-1-khung-thực-nghiệm-bóc-tách-ảnh-hưởng-từng-tầng-nhãn-layer-wise-ablation-framework)
   - 2.1. Động lực khoa học & Mục tiêu bóc tách
   - 2.2. Ma trận 6 cấu hình bóc tách chi tiết (Ablation Regimes)
   - 2.3. Quy trình cô lập nhãn và hệ thống chỉ số đánh giá tương ứng
3. [MODULE 2: KHÔNG GIAN ĐẶC TRƯNG BỔ TRỢ CỐ ĐỊNH CHO TẬP PHỤ THUỘC DL](#3-module-2-không-gian-đặc-trưng-bổ-trợ-cố-định-cho-tập-phụ-thuộc-dl)
   - 3.1. Hiện trạng v5.1 và điểm cần hoàn thiện
   - 3.2. Cơ chế Static Context: Hợp nhất $X$ và toàn bộ $IL$ cho chuỗi $DL$
   - 3.3. Tách biệt kiến trúc 2 pha (Two-Phase Decoupled Execution)
4. [MODULE 3: CHUẨN HÓA ĐẶC TRƯNG XÁC SUẤT MỀM (FEATURE NORMALIZATION CONSISTENCY)](#4-module-3-chuẩn-hóa-đặc-trưng-xác-suất-mềm-feature-normalization-consistency)
   - 4.1. Vấn đề lệch phân phối đặc trưng (Distribution Mismatch) trong v5.1
   - 4.2. Giải pháp chuẩn hóa thích ứng theo Scaler của dữ liệu gốc $X$
   - 4.3. Các phương án chuẩn hóa toán học (StandardScaler, MaxAbsScaler, Logit Scaling)
5. [MODULE 4: TÁI THIẾT KẾ CHUỖI CLASSIFIER CHAIN THEO NGƯỠNG TƯƠNG QUAN (THRESHOLD-FILTERED SPARSE CC)](#5-module-4-tái-thiết-kế-chuỗi-classifier-chain-theo-ngưỡng-tương-quan-threshold-filtered-sparse-cc)
   - 5.1. Hạn chế của chuỗi dày đặc (Dense CC) trong v5.1
   - 5.2. Khung lý thuyết Đồ thị Phụ thuộc Thưa (Sparse Selective Dependency Graph)
   - 5.3. Tiêu chuẩn lọc ngưỡng tương quan $\theta_{\text{corr}} \in [0.7, 0.8]$
   - 5.4. Giải thuật lựa chọn tiền nhiệm (Active Predecessor Selection Algorithm)
   - 5.5. Cơ chế suy luận xác suất chọn lọc (Sparse Marginalization)
6. [MODULE 5: KIỂM TOÁN VÀ THEO DÕI CHI TIẾT SỐ LƯỢNG NHÃN PHÂN TẦNG (FINE-GRAINED LAYER AUDIT)](#6-module-5-kiểm-toán-và-theo-dõi-chi-tiết-số-lượng-nhãn-phân-tầng-fine-grained-layer-audit)
   - 6.1. Cấu trúc dữ liệu kiểm toán mở rộng (`StratifiedAuditRecord_v5_1_1`)
   - 6.2. Định dạng xuất file (`stage_audit.csv`, bảng tóm tắt Markdown, JSON checkpoints)
7. [MODULE MỞ RỘNG (OPTION): HIỆU NĂNG BR TRÊN IL & NGƯỠNG SUY GIẢM THÍCH ỨNG](#7-module-mở-rộng-option-hiệu-năng-br-trên-il--ngưỡng-suy-giảm-thích-ứng)
   - 7.1. Đánh giá hiệu năng chuyên biệt của Binary Relevance trên tập $IL$
   - 7.2. Cơ chế suy giảm ngưỡng bóc tách theo tầng ($\tau_1 > \tau_2 > \tau_3$)
8. [ĐẶC TẢ KIẾN TRÚC CODE & API CONTRACTS](#8-đặc-tả-kiến-trúc-code--api-contracts)
   - 8.1. Danh mục file cập nhật và cấu trúc module mới
   - 8.2. Chi tiết API Class & Method signatures
9. [KẾ HOẠCH THỰC NGHIỆM VÀ LỘ TRÌNH TRIỂN KHAI (ACTION PLAN)](#9-kế-hoạch-thực-nghiệm-và-lộ-trình-triển-khai-action-plan)
10. [KẾT LUẬN & ĐÓNG GÓP KHOA HỌC DỰ KIẾN CHO BÀI BÁO](#10-kết-luận--đóng-góp-khoa-học-dự-kiến-cho-bài-báo)

---

## 1. BỐI CẢNH VÀ ĐỘNG LỰC NÂNG CẤP V5.1.1

### 1.1. Phân tích 5 yêu cầu cốt lõi (`core`) từ cuộc họp

Trong phiên bản v5.1 ([spec/spec_V5_1.md](file:///d:/University_Subject/ML%20Research/BR_CC/spec/spec_V5_1.md)), hệ thống đã giải quyết thành công bài toán phân rã nhãn tự nhiên dựa trên dữ liệu thông qua cơ chế bóc tách đa tầng (*Stratified Peeling*) và đảo ngược trật tự chuỗi CC cho tập phụ thuộc theo tương quan tăng dần (*Ascending Correlation Order*). Kết quả thực nghiệm tại [tmp.md](file:///d:/University_Subject/ML%20Research/BR_CC/tmp.md) đã chứng minh Selective Macro-Precision tăng đồng loạt trên cả 3 base learners, Hamming Loss giảm và Subset 0/1 Accuracy tăng mạnh trên Logistic Regression.

Tuy nhiên, biên bản cuộc họp tại [meeting_summary.md](file:///d:/University_Subject/ML%20Research/BR_CC/meeting_summary.md) đã chỉ ra 5 vấn đề cốt lõi cần phải hoàn thiện để nâng cấp lên phiên bản **v5.1.1**:

1. **Bóc tách vai trò từng tầng nhãn (Core 1):** Cần một khung phân tích bóc tách (Ablation Study) có kiểm soát nghiêm ngặt để xác định chính xác tầng nhãn nào ($IL_1, IL_2, IL_3$ hay $DL$) tạo ra tác động lớn nhất đến các chỉ số cuối cùng (Macro-F1, Precision, Subset Accuracy, Hamming Loss).
2. **Không gian đặc trưng tĩnh cho CC của DL (Core 2):** Khi huấn luyện chuỗi CC cho tập nhãn phụ thuộc $DL$, toàn bộ các nhãn thuộc tập độc lập tích lũy $IL = \bigcup_{t} IL_t$ cùng dữ liệu gốc $X$ phải được cung cấp đồng thời làm không gian đặc trưng đầu vào nền tảng.
3. **Minh bạch hóa quá trình phân tầng (Core 3):** Báo cáo thực nghiệm cần hiển thị rõ ràng số lượng nhãn và danh sách nhãn cụ thể được tách ra ở từng tầng bóc tách ($|IL_1|, |IL_2|, |IL_3|, |DL|$), thay vì chỉ ghi nhận số tầng trung bình.
4. **Chuẩn hóa phân phối đặc trưng nhãn IL (Core 4):** Việc ghép trực tiếp vector xác suất mềm $\hat{P}(Y_{IL} = 1 \mid X) \in [0, 1]$ vào ma trận $X$ đã qua chuẩn hóa (ví dụ `MaxAbsScaler` trong $[-1, 1]$ hoặc `StandardScaler` với mean 0, variance 1) tạo ra sự lệch phân phối đặc trưng (distribution mismatch), gây bất lợi cho việc học trọng số của các bộ học tuyến tính (SVM, Logistic) và mạng nơ-ron (MLP). Do đó, các đặc trưng IL bổ sung bắt buộc phải được chuẩn hóa theo cùng quy chuẩn của dữ liệu $X$.
5. **Tái thiết kế chuỗi CC cho DL dựa trên ngưỡng tương quan (Core 5):** Trong v5.1, chuỗi CC của $DL$ là một chuỗi dày đặc (Dense Chain): mỗi nhãn $d_k$ nhận toàn bộ các nhãn $d_1, \dots, d_{k-1}$ đứng trước làm đầu vào. Điều này khiến mô hình bị "ô nhiễm đặc trưng" từ những nhãn đứng trước không hề có tương quan ngữ nghĩa. Cần chuyển sang mô hình đồ thị thưa (Sparse Dependency Chain): chỉ các nhãn tiền nhiệm có độ tương quan cặp vượt ngưỡng cao ($\theta_{\text{corr}} \in [0.7, 0.8]$) mới được đưa vào làm đặc trưng.

### 1.2. Phân tích 2 yêu cầu mở rộng (`option`)

Bên cạnh 5 yêu cầu cốt lõi, phiên bản v5.1.1 chuẩn bị sẵn nền tảng kỹ thuật cho 2 mở rộng:
- **Option 1:** Đánh giá độc lập hiệu năng của mô hình Binary Relevance trên tập nhãn $IL$ để kiểm chứng giả thuyết: "Các nhãn được chọn vào $IL$ thực sự có thể tự đứng vững với chất lượng phân loại cao mà không cần chuỗi CC".
- **Option 2:** Thử nghiệm cơ chế ngưỡng suy giảm thích ứng theo từng tầng bóc tách ($\tau_1 = 0.80 \to \tau_2 = 0.75 \to \tau_3 = 0.70$) để bóc tách triệt để các nhãn bán độc lập ở tầng sâu.

---

### 1.3. Sơ đồ kiến trúc tổng thể GSI-MLC-PA v5.1.1

```
+==================================================================================================+
|                        GSI-MLC-PA PHIÊN BẢN 5.1.1: ARCHITECTURE PIPELINE                         |
+==================================================================================================+

   [ Dữ liệu gốc X ] (Đã qua chuẩn hóa Scaler S_X: e.g. MaxAbsScaler / StandardScaler)
          │
          ▼
   +-----------------------------------------------------------------------------------------------+
   | PHA 1: PHÂN TẦNG BÓC TÁCH NHÃN ĐỘC LẬP (STRATIFIED PEELING WITH NORMALIZED AUGMENTATION)      |
   |                                                                                               |
   |  Tầng 1 (t=1): Huấn luyện BR trên X. Nhãn đạt F1 >= tau ==> Vào IL_1                          |
   |                Các nhãn còn lại ==> Danh sách chờ D_cand^(1).                                 |
   |                Kiểm toán: Ghi nhận |IL_1| và danh sách nhãn.                                  |
   |                                                                                               |
   |  Tầng 2 (t=2): Tính xác suất P_hat(Y_IL1).                                                    |
   |                [CHẾ ĐỘ MỚI v5.1.1]: CHUẨN HÓA P_hat(Y_IL1) THEO SCALER S_X!                    |
   |                Tạo ma trận mở rộng: X^(1) = [X, Normalize(P_hat(Y_IL1))].                     |
   |                Thử lại D_cand^(1). Nhãn đạt F1 >= tau ==> Vào IL_2.                           |
   |                Kiểm toán: Ghi nhận |IL_2| và danh sách nhãn.                                  |
   |                                                                                               |
   |  Tầng 3 (t=3): Lặp lại với X^(2) = [X, Normalize(P_hat(Y_IL1)), Normalize(P_hat(Y_IL2))].    |
   |                Nhãn đạt F1 >= tau ==> Vào IL_3. Còn lại ==> D_residual.                       |
   |                Kiểm toán: Ghi nhận |IL_3| và |D_residual|.                                    |
   +-----------------------------------------------------------------------------------------------+
          │
          │ Tập độc lập tích lũy: IL = IL_1 U IL_2 U IL_3
          │ Tập phụ thuộc cốt lõi: DL = D_residual
          ▼
   +-----------------------------------------------------------------------------------------------+
   | PHA 2: KHÔNG GIAN ĐẶC TRƯNG NỀN TẢNG CỐ ĐỊNH (STATIC CONTEXT EXPANSION FOR DL)                |
   |                                                                                               |
   |  Toàn bộ nhãn IL được dự đoán bằng BR mở rộng.                                                |
   |  Không gian đặc trưng nền tảng cung cấp cho toàn bộ các classifier trong DL:                  |
   |             X_context_DL = [ X, Normalize(P_hat(Y_IL = 1 | X)) ]                              |
   +-----------------------------------------------------------------------------------------------+
          │
          ▼
   +-----------------------------------------------------------------------------------------------+
   | PHA 3: TÁI THIẾT KẾ CHUỖI CC CHO DL DỰA TRÊN NGƯỠNG TƯƠNG QUAN (THRESHOLD-FILTERED SPARSE CC)|
   |                                                                                               |
   |  1. Tính ma trận tương quan cặp Phi-coefficient giữa các nhãn: R = [ |phi_ij| ].              |
   |  2. Đặt ngưỡng tương quan khắt khe: theta_corr in [0.7, 0.8].                                 |
   |  3. Với mỗi nhãn d_k in DL:                                                                   |
   |     Tìm tập phụ thuộc hoạt động (Active Predecessors):                                        |
   |         Parents(d_k) = { d_j in DL đứng trước d_k | |phi_{j, k}| >= theta_corr }              |
   |  4. Classifier cho d_k CHỈ nhận: [ X_context_DL, Y_{Parents(d_k)} ]                           |
   |     ==> Triệt tiêu hoàn toàn nhiễu từ các nhãn không tương quan!                              |
   +-----------------------------------------------------------------------------------------------+
          │
          ▼
   +-----------------------------------------------------------------------------------------------+
   | PHA 4: TẦNG QUYẾT ĐỊNH BAYES-OPTIMAL CÓ TỪ CHỐI (PARTIAL ABSTENTION BOP)                      |
   |                                                                                               |
   |  Xác suất biên hợp nhất p_k(x) được đưa qua bộ quyết định 3 trạng thái:                       |
   |           y_hat_k in { 0 (Phủ định), 1 (Khẳng định), bot (Từ chối / Abstain) }                |
   |  Đánh giá đầy đủ qua 7 nhóm metric và kiểm toán bóc tách từng tầng nhãn (Ablation).          |
   +===============================================================================================+
```

---

## 2. MODULE 1: KHUNG THỰC NGHIỆM BÓC TÁCH ẢNH HƯỞNG TỪNG TẦNG NHÃN (LAYER-WISE ABLATION FRAMEWORK)

### 2.1. Động lực khoa học & Mục tiêu bóc tách

Trong các hệ thống phân tầng đa bước, một câu hỏi khoa học cốt tử của các phản biện (peer-reviewers) là:
> *"Liệu việc phân tách thành nhiều tầng $IL_1, IL_2, IL_3$ có thực sự mang lại giá trị gia tăng, hay hiệu năng mô hình chủ yếu chỉ do tầng đầu tiên $IL_1$ hoặc chuỗi $DL$ quyết định?"*

Để trả lời thuyết phục câu hỏi này, phiên bản v5.1.1 thiết lập **Khung Thực Nghiệm Bóc Tách Đa Cấu Hình (Systematic Layer-Wise Ablation Framework)**. Khung này cho phép "bật/tắt" có chọn lọc từng tầng nhãn khi đánh giá, từ đó đo lường chính xác đóng góp cận biên (*Marginal Contribution*) của từng tập nhãn đối với các chỉ số toàn cục.

### 2.2. Ma trận 6 cấu hình bóc tách chi tiết (Ablation Regimes)

Dựa trên yêu cầu chi tiết tại mục `**core**` của [meeting_summary.md](file:///d:/University_Subject/ML%20Research/BR_CC/meeting_summary.md), hệ thống định nghĩa 6 chế độ đánh giá bóc tách:

| Mã Cấu Hình (Regime ID) | Tên Cấu Hình Nghiên Cứu | Tập Nhãn Được Đánh Giá ($\mathcal{Y}_{\text{eval}}$) | Tập Nhãn Bị Vô Hiệu Hóa ($\mathcal{Y}_{\text{masked}}$) | Ý Nghĩa Khoa Học & Mục Tiêu Phân Tích |
| :--- | :--- | :--- | :--- | :--- |
| **`ABL_1_ONLY_IL1`** | Độc lập Tầng 1 | $\mathcal{I}_1$ | $\mathcal{D}_L^{(1)}$ (tắt toàn bộ $DL$ và các tầng sau) | Đo lường năng lực phân loại độc lập thuần túy của các nhãn dễ nhất từ dữ liệu gốc $X$. |
| **`ABL_2_ONLY_IL2`** | Độc lập Tầng 2 Riêng Biệt | $\mathcal{I}_2$ | $\mathcal{I}_1 \cup \mathcal{D}_L^{(2)}$ | Kiểm chứng xem bản thân tầng $\mathcal{I}_2$ khi đứng một mình có đạt chất lượng cao hay không. |
| **`ABL_2_ACCUM_IL12`**| Độc lập Tích Lũy Tầng 1+2 | $\mathcal{I}_1 \cup \mathcal{I}_2$ | $\mathcal{D}_L^{(2)}$ | Đo lường sức mạnh cộng hưởng tích lũy sau vòng bóc tách thứ hai khi chưa cần chuỗi $DL$. |
| **`ABL_3_ONLY_IL3`** | Độc lập Tầng 3 Riêng Biệt | $\mathcal{I}_3$ | $\mathcal{I}_1 \cup \mathcal{I}_2 \cup \mathcal{D}_L^{(3)}$ | Đánh giá giá trị thực tế của các nhãn được vớt ở tầng bóc tách sâu nhất. |
| **`ABL_3_ALL_IL`** | Toàn Bộ Nhãn Độc Lập | $\mathcal{I} = \mathcal{I}_1 \cup \mathcal{I}_2 \cup \mathcal{I}_3$ | $\mathcal{D}_{\text{residual}}$ (tắt toàn bộ $DL$) | Đo lường hiệu năng giới hạn trên của toàn bộ hệ thống nhãn độc lập (BR mở rộng). |
| **`ABL_3_ONLY_DL`** | Riêng Tập Phụ Thuộc Cốt Lõi | $\mathcal{D}_{\text{residual}}$ | $\mathcal{I} = \mathcal{I}_1 \cup \mathcal{I}_2 \cup \mathcal{I}_3$ (tắt toàn bộ $IL$) | Đo lường mức độ khó, rủi ro sai số và hiệu năng thực tế của riêng chuỗi Classifier Chains. |

### 2.3. Quy trình toán học cô lập nhãn và đánh giá chỉ số

Cho ma trận nhãn thực tế $Y \in \{0, 1\}^{N \times K}$ và ma trận dự đoán $\hat{Y} \in \{0, \bot, 1\}^{N \times K}$.

Với mỗi cấu hình bóc tách $\text{Regime} = (\mathcal{Y}_{\text{eval}}, \mathcal{Y}_{\text{masked}})$:
1. **Lọc ma trận con:**
   $$Y^{(\text{eval})} = Y[:, \mathcal{Y}_{\text{eval}}], \qquad \hat{Y}^{(\text{eval})} = \hat{Y}[:, \mathcal{Y}_{\text{eval}}]$$
2. **Tính toán các chỉ số chuyên biệt:**
   - **Selective Macro-F1 trên tập con:**
     $$\text{Macro-F1}(\mathcal{Y}_{\text{eval}}) = \frac{1}{|\mathcal{Y}_{\text{eval}}|} \sum_{k \in \mathcal{Y}_{\text{eval}}} F_1(Y_{*, k}, \hat{Y}_{*, k} \mid \hat{Y}_{*, k} \neq \bot)$$
   - **Selective Macro-Precision:**
     $$\text{Macro-Prec}(\mathcal{Y}_{\text{eval}}) = \frac{1}{|\mathcal{Y}_{\text{eval}}|} \sum_{k \in \mathcal{Y}_{\text{eval}}} \text{Precision}(Y_{*, k}, \hat{Y}_{*, k} \mid \hat{Y}_{*, k} \neq \bot)$$
   - **Subset 0/1 Accuracy trên tập con:**
     $$\text{Subset-0/1}(\mathcal{Y}_{\text{eval}}) = \frac{1}{N} \sum_{i=1}^N \mathbb{I}\left( Y_{i, \mathcal{Y}_{\text{eval}}} = \hat{Y}_{i, \mathcal{Y}_{\text{eval}}} \right)$$
   - **Hamming Loss trên tập con:**
     $$\text{Hamming-Loss}(\mathcal{Y}_{\text{eval}}) = \frac{1}{N \cdot |\mathcal{Y}_{\text{eval}}|} \sum_{i=1}^N \sum_{k \in \mathcal{Y}_{\text{eval}}} \mathbb{I}(y_{ik} \neq \hat{y}_{ik})$$
   - **Coverage trên tập con:**
     $$\text{Coverage}(\mathcal{Y}_{\text{eval}}) = \frac{1}{N \cdot |\mathcal{Y}_{\text{eval}}|} \sum_{i=1}^N \sum_{k \in \mathcal{Y}_{\text{eval}}} \mathbb{I}(\hat{y}_{ik} \neq \bot)$$

Khung này được tự động kích hoạt sau khi huấn luyện xong mô hình chính, xuất kết quả vào file `results_v5_1_test/ablation_layer_analysis.csv`.

---

## 3. MODULE 2: KHÔNG GIAN ĐẶC TRƯNG BỔ TRỢ CỐ ĐỊNH CHO TẬP PHỤ THUỘC DL

### 3.1. Hiện trạng v5.1 và điểm cần hoàn thiện

Trong phiên bản v5.1, chuỗi Classifier Chains được khởi tạo dựa trên toàn bộ $K$ nhãn với trật tự thực thi $\Pi_{\text{final}} = [\mathcal{I}, \pi_{\mathcal{D}}]$. Mặc dù các nhãn $\mathcal{I}$ đứng đầu chuỗi, cách thiết kế này khiến `ClassifierChainClassifier` trong thư viện đối xử với $\mathcal{I}$ như các mắt xích CC thông thường (khi huấn luyện dùng ground truth $Y_{\mathcal{I}}$, khi suy luận nhận xác suất từ BR).

Theo yêu cầu `core` số 2 của [meeting_summary.md](file:///d:/University_Subject/ML%20Research/BR_CC/meeting_summary.md):
> *"Sử dụng toàn bộ IL cùng dữ liệu ban đầu làm đầu vào dự đoán cho CC của DL - cái này nếu chưa có thì làm, có rồi thì thôi"*

### 3.2. Cơ chế Static Context: Hợp nhất $X$ và toàn bộ $IL$ cho chuỗi $DL$

Ở phiên bản v5.1.1, cấu trúc được phân rã thành **Mô Hình 2 Pha Tách Rời (Two-Phase Decoupled Architecture)**:

```
[ Giai đoạn 1: BR Mở Rộng ] ──────────> Ước lượng xác suất độc lập: P_hat(Y_IL | X)
                                                     │
                                                     ▼ (Chuẩn hóa Normalizer)
[ Ma trận đặc trưng gốc X ] ──────────> [ Không gian ngữ cảnh tĩnh: X_context_DL ]
                                                     │
                                                     ▼
                                      [ Giai đoạn 2: Chuỗi CC của DL ]
                                      Mỗi classifier trong DL đều nhận
                                      X_context_DL làm đặc trưng cố định.
```

Định nghĩa toán học của ma trận đặc trưng ngữ cảnh tĩnh cho $DL$:
$$X_{\text{context\_DL}} = \left[ X \ \Vert \ \Phi_{\text{norm}}\left( \hat{P}(Y_{\mathcal{I}} = 1 \mid X) \right) \right] \in \mathbb{R}^{N \times (d + |\mathcal{I}|)}$$

Trong đó:
- $X \in \mathbb{R}^{N \times d}$ là ma trận đặc trưng gốc đã được chuẩn hóa.
- $\hat{P}(Y_{\mathcal{I}} = 1 \mid X) \in [0, 1]^{N \times |\mathcal{I}|}$ là vector xác suất mềm dự đoán từ các bộ phân loại Binary Relevance của các tầng độc lập tích lũy $\mathcal{I} = \mathcal{I}_1 \cup \mathcal{I}_2 \cup \mathcal{I}_3$.
- $\Phi_{\text{norm}}(\cdot)$ là hàm chuẩn hóa được định nghĩa tại Module 3.

### 3.3. Lợi ích khoa học vượt trội
1. **Triệt tiêu hiện tượng rò rỉ và lan truyền sai số:** Các nhãn trong $\mathcal{I}$ không phụ thuộc vào bất kỳ nhãn nào trong $DL$, do đó đóng vai trò là "mỏ neo ngữ cảnh" (context anchors) hoàn toàn ổn định và an toàn.
2. **Đồng nhất giữa Training và Inference:** Trong quá trình huấn luyện chuỗi $DL$, thay vì đưa nhãn cứng $Y_{\mathcal{I}}$ (dễ gây overfitting), mô hình sử dụng trực tiếp xác suất out-of-fold hoặc xác suất dự đoán mềm đã chuẩn hóa $\Phi_{\text{norm}}(\hat{P})$, giúp giảm thiểu tối đa hiện tượng sai lệch phân phối giữa lúc học và lúc suy luận thực tế (*train-test discrepancy*).

---

## 4. MODULE 3: CHUẨN HÓA ĐẶC TRƯNG XÁC SUẤT MỀM (FEATURE NORMALIZATION CONSISTENCY)

### 4.1. Vấn đề lệch phân phối đặc trưng (Distribution Mismatch) trong v5.1

Trong thực nghiệm học máy đa nhãn:
- Dữ liệu thuộc tính $X$ thường trải qua bước tiền xử lý chuẩn hóa bằng:
  - `MaxAbsScaler`: Đưa các giá trị về đoạn $[-1, 1]$ (giữ nguyên tính thưa của ma trận).
  - `StandardScaler`: Chuẩn hóa về phân phối chuẩn có $\mu = 0, \sigma^2 = 1$.
- Trong khi đó, vector dự đoán xác suất mềm $\hat{P}(Y_k = 1 \mid X)$ có miền giá trị luôn nằm trong đoạn $[0, 1]$ với kỳ vọng thường rất nhỏ (do nhãn trong MLC thường thưa, tỷ lệ dương tính $LD < 0.15 \implies \mathbb{E}[\hat{P}] \approx 0.05 - 0.20$).

**Hậu quả khi ghép nối trực tiếp không qua chuẩn hóa ($X_{\text{aug}} = [X, \hat{P}]$):**
1. **Đối với Support Vector Machine (LinearSVC) & Logistic Regression:** Hàm mục tiêu tối ưu có dạng $\min_w \frac{1}{2}\|w\|^2 + C \sum \mathcal{L}(w^T x)$. Hệ số phạt $L_2$ phạt đều trên tất cả các chiều của vector trọng số $w$. Khi các thuộc tính $X$ và thuộc tính $\hat{P}$ có phương sai và khoảng giá trị khác biệt nhau, bộ tối ưu sẽ bị thiên vị nặng nề, làm giảm tốc độ hội tụ và giảm độ chính xác của siêu phẳng phân cách.
2. **Đối với Multi-Layer Perceptron (MLP Neural Network):** Sự bất cân xứng về phân phối đầu vào dẫn đến hiện tượng trôi dạt gradient (*internal covariate shift*) ngay tại tầng tuyến tính đầu tiên, làm suy giảm hiệu năng huấn luyện trên GPU.

### 4.2. Giải pháp chuẩn hóa thích ứng theo Scaler của dữ liệu gốc $X$

Để thực hiện yêu cầu `core` số 4:
> *"Khi đưa các nhãn IL vào làm dữ liệu dự đoán tầng sau cần normalized theo chuẩn data ban đầu"*

GSI-MLC-PA v5.1.1 triển khai module **`AugmentationNormalizer`** với nguyên lý:
> **Nguyên lý chuẩn hóa nhất quán:** Các đặc trưng xác suất bổ sung phải được chuẩn hóa bằng chính phương pháp và thang đo thống kê đang áp dụng cho ma trận đặc trưng gốc $X$.

### 4.3. Các phương án chuẩn hóa toán học

Hệ thống hỗ trợ 3 chiến lược chuẩn hóa thông qua tham số cấu hình `aug_normalization_strategy`:

#### Phương án 1: Matching Base Scaler (Mặc định - Khuyến nghị)
Sử dụng một bản sao của bộ `Scaler` đã áp dụng cho $X$ để fit và transform trên ma trận xác suất $\hat{P}$:
- Nếu $X$ chuẩn hóa bằng `MaxAbsScaler`:
  $$\Phi_{\text{norm}}(\hat{P}) = \frac{\hat{P}}{\max(|\hat{P}| + \epsilon)} \in [0, 1]$$
- Nếu $X$ chuẩn hóa bằng `StandardScaler`:
  $$\Phi_{\text{norm}}(\hat{P}) = \frac{\hat{P} - \mu_{\hat{P}}}{\sigma_{\hat{P}} + \epsilon} \implies \mathbb{E}[\Phi] = 0, \ \text{Var}(\Phi) = 1$$

#### Phương án 2: Centered Zero-Mean MinMax Scaling (Chuyển đổi về khoảng $[-1, 1]$)
Đưa xác suất từ đoạn $[0, 1]$ về đoạn đối xứng $[-1, 1]$ để tương đồng hoàn hảo với `MaxAbsScaler` của $X$:
$$\Phi_{\text{centered}}(\hat{P}) = 2 \cdot \hat{P} - 1.0$$
*Ưu điểm:* Biến đổi tuyến tính bảo toàn tuyệt đối thứ tự xác suất; điểm không chắc chắn $0.5$ chuyển thành giá trị $0.0$ (tâm đối xứng), điểm chắc chắn 1 thành $+1.0$ và điểm chắc chắn 0 thành $-1.0$.

#### Phương án 3: Logit Transformation (Chuyển đổi sang không gian Log-Odds)
Chuyển đổi xác suất sang không gian log-odds vô hạn trước khi chuẩn hóa:
$$z = \log\left( \frac{\text{clip}(\hat{P}, \epsilon, 1-\epsilon)}{1 - \text{clip}(\hat{P}, \epsilon, 1-\epsilon)} \right), \qquad \Phi_{\text{logit}}(\hat{P}) = \text{StandardScaler}(z)$$
*Ưu điểm:* Đưa xác suất về phân phối tiệm cận Gauss đối xứng, cực kỳ thích hợp cho bộ phân loại tuyến tính Logistic và SVM.

---

## 5. MODULE 4: TÁI THIẾT KẾ CHUỖI CLASSIFIER CHAIN THEO NGƯỠNG TƯƠNG QUAN (THRESHOLD-FILTERED SPARSE CC)

### 5.1. Hạn chế của chuỗi dày đặc (Dense CC) trong v5.1

Trong phiên bản v5.1, sau khi bóc tách được tập phụ thuộc $\mathcal{D}_{\text{residual}} = \{d_1, d_2, \dots, d_m\}$, chuỗi CC được xây dựng bằng cách sắp xếp theo tổng tương quan tăng dần (*Ascending Correlation*).

Mặc dù trật tự tăng dần đã dập tắt hiện tượng lan truyền sai số từ nhãn gốc, **cấu trúc chuỗi vẫn là một chuỗi dày đặc (Dense Full Chain)**:
$$\text{Đặc trưng đầu vào của } d_k = [X_{\text{context}}, Y_{d_1}, Y_{d_2}, \dots, Y_{d_{k-1}}]$$

**Hạn chế cốt tử:**
- Nếu nhãn $d_k$ đứng ở vị trí thứ 10 trong chuỗi $DL$, nó buộc phải nhận toàn bộ 9 nhãn đứng trước làm đặc trưng, **ngay cả khi 8 trong số 9 nhãn đó hoàn toàn không có bất kỳ tương quan thực tế nào với $d_k$** (hệ số tương quan $|\phi_{j, k}| < 0.05$).
- Việc nhồi nhét các nhãn không tương quan vào đầu vào classifier gây ra hiện tượng **ô nhiễm không gian đặc trưng (Feature Dimensionality Explosion & Noise Pollution)**, làm giảm khả năng khái quát hóa và tăng nguy cơ overfitting.

### 5.2. Khung lý thuyết Đồ thị Phụ thuộc Thưa (Sparse Selective Dependency Graph)

Yêu cầu `core` số 5 của [meeting_summary.md](file:///d:/University_Subject/ML%20Research/BR_CC/meeting_summary.md) nêu rõ:
> *"Làm lại phần classifier chain trong DL: thay vì sắp xếp theo tổng độ tương quan thì dựa vào ma trận tương quan, tìm tập nhãn có tương quan cao vượt ngưỡng (0.7-0.8) để tính, sử dụng cả các nhãn IL làm features space"*

Để hiện thực hóa điều này, phiên bản v5.1.1 chuyển đổi kiến trúc Classifier Chain từ dạng **Chuỗi Đầy Đủ Cố Định (Full Sequential Chain)** sang dạng **Đồ Thị Hướng Có Điều Kiện Chọn Lọc (Selective Directed Acyclic Graph - SDAG)**:

```
[ Chuỗi Dày Đặc v5 cũ ]                  [ Chuỗi Chọn Lọc Theo Ngưỡng v5.1.1 Mới ]
   d_1 ───> d_2 ───> d_3 ───> d_4           d_1 ──────────> d_4 (Nếu |phi_{1,4}| >= 0.75)
    │        │        │        │            d_2 ──(không nối)─> d_4 (Nếu |phi_{2,4}| = 0.08)
    └────────┴────────┴────────┘            d_3 ──────────> d_4 (Nếu |phi_{3,4}| >= 0.81)
 (Mọi mắt xích đều nối với nhau,          (CHỈ nối các nhãn có tương quan thực sự mạnh,
      kể cả không tương quan)                  loại bỏ hoàn toàn cạnh nhiễu!)
```

### 5.3. Tiêu chuẩn lọc ngưỡng tương quan $\theta_{\text{corr}} \in [0.7, 0.8]$

Cho ma trận nhãn huấn luyện $Y \in \{0, 1\}^{N \times K}$. Hệ số tương quan Phi giữa cặp nhãn $(j, k)$ được tính bởi:
$$\phi_{jk} = \frac{p_{11} p_{00} - p_{10} p_{01}}{\sqrt{p_{1*} p_{0*} p_{*1} p_{*0}}}$$
với $p_{ab} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(y_{ij} = a \land y_{ik} = b)$.

Thiết lập ngưỡng tương quan lựa chọn: $\theta_{\text{corr}} = 0.75$ (biên độ cấu hình hỗ trợ: $0.70 \le \theta_{\text{corr}} \le 0.80$).

### 5.4. Giải thuật lựa chọn tiền nhiệm hoạt động (Active Predecessor Selection)

```
========================================================================================
ALGORITHM 2: XÂY DỰNG CHUỖI PHỤ THUỘC CHỌN LỌC THEO NGƯỠNG (THRESHOLD-FILTERED SPARSE CC)
========================================================================================
Đầu vào: Tập nhãn phụ thuộc DL = {d_1, ..., d_m}, trật tự pi_D,
         ma trận tương quan R = [|phi_ij|], ngưỡng theta_corr = 0.75,
         không gian ngữ cảnh nền tảng X_context_DL = [X, Normalize(P_hat(Y_IL))].
Đầu ra:  Tập các bộ phân loại phụ thuộc thưa {f_k}_(k in DL) và bản đồ cha mẹ Parents(d_k).

1: Khởi tạo bản đồ cha mẹ: Parents_map <- {}.
2: for mỗi vị trí t từ 1 đến m do:
3:     Nhãn đích hiện tại: target_label <- pi_D[t].
4:     Danh sách các nhãn tiền nhiệm trong chuỗi DL:
5:         Candidate_predecessors <- [ pi_D[1], ..., pi_D[t-1] ].
6:     
7:     Lọc các nhãn tiền nhiệm có tương quan cao vượt ngưỡng:
8:         Active_parents <- [ j in Candidate_predecessors sao cho R[j, target_label] >= theta_corr ].
9:     
10:    Parents_map[target_label] <- Active_parents.
11:    
12:    Xây dựng ma trận đặc trưng huấn luyện cho target_label:
13:    if Active_parents rỗng then:
14:        // Không có nhãn nào trong DL đủ tương quan mạnh ==> CHỈ dùng X_context_DL!
15:        X_train_target <- X_context_DL.
16:    else:
17:        // Ghép không gian ngữ cảnh với các nhãn cha mẹ thực sự có tương quan
18:        Y_parents <- Y_train[:, Active_parents].
19:        X_train_target <- [ X_context_DL, Y_parents ].
20:    end if
21:    
22:    Huấn luyện bộ phân loại nhị phân f_(target_label) trên (X_train_target, Y_train[:, target_label]).
23: end for
24: return {f_k}, Parents_map.
========================================================================================
```

### 5.5. Cơ chế suy luận xác suất chọn lọc (Sparse Marginalization)

Khi suy luận xác suất cho mẫu mới $x$:
1. Nếu $\text{Parents}(d_k) = \emptyset$:
   $$p_{d_k}(x) = f_{d_k}(x_{\text{context\_DL}})$$
2. Nếu $|\text{Parents}(d_k)| = 1$ với cha duy nhất là $r$:
   Áp dụng biên hóa chính xác 2 trạng thái (*Exact Two-State Marginalization*):
   $$p_{d_k}(x) = (1 - p_r(x)) f_{d_k}([x_{\text{context\_DL}}, 0]) + p_r(x) f_{d_k}([x_{\text{context\_DL}}, 1])$$
3. Nếu $|\text{Parents}(d_k)| \ge 2$:
   Áp dụng thế xác suất mềm trung bình (*Mean-Field Plug-in*):
   $$p_{d_k}(x) \approx f_{d_k}\left( \left[ x_{\text{context\_DL}}, [p_j(x)]_{j \in \text{Parents}(d_k)} \right] \right)$$

Do số lượng nhãn cha trong $\text{Parents}(d_k)$ bị khống chế chặt bởi ngưỡng $\theta_{\text{corr}} \ge 0.70$, số lượng tiền nhiệm thực tế thường rất nhỏ ($\le 2-3$ nhãn). Điều này giúp **thời gian suy luận nhanh hơn gấp nhiều lần** và loại bỏ hoàn toàn hiện tượng bùng nổ sai số của chuỗi CC dài.

---

## 6. MODULE 5: KIỂM TOÁN VÀ THEO DÕI CHI TIẾT SỐ LƯỢNG NHÃN PHÂN TẦNG (FINE-GRAINED LAYER AUDIT)

### 6.1. Cấu trúc dữ liệu kiểm toán mở rộng (`StratifiedAuditRecord_v5_1_1`)

Để đáp ứng yêu cầu `core` số 3:
> *"Cần biết rõ tách được bao nhiêu nhãn qua từng tầng - ghi thêm vào file kết quả chạy chỉ số về số lượng nhãn được tách ra sau mỗi tầng"*

Hệ thống bổ sung cấu trúc dữ liệu kiểm toán chuẩn hóa:

```python
@dataclass(frozen=True)
class StratifiedAuditRecord_v5_1_1:
    dataset: str
    base_learner: str
    fold: int
    n_total_labels: int                     # Tổng số nhãn K
    n_il_stage_1: int                       # Số lượng nhãn bóc tách ở Tầng 1
    labels_il_stage_1: Tuple[int, ...]      # Danh sách chỉ số nhãn Tầng 1
    n_il_stage_2: int                       # Số lượng nhãn bóc tách ở Tầng 2
    labels_il_stage_2: Tuple[int, ...]      # Danh sách chỉ số nhãn Tầng 2
    n_il_stage_3: int                       # Số lượng nhãn bóc tách ở Tầng 3
    labels_il_stage_3: Tuple[int, ...]      # Danh sách chỉ số nhãn Tầng 3
    n_total_il: int                         # Tổng số nhãn độc lập: |IL_1| + |IL_2| + |IL_3|
    pct_total_il: float                     # Tỷ lệ nhãn độc lập (%)
    n_residual_dl: int                      # Số lượng nhãn phụ thuộc cốt lõi |DL|
    pct_residual_dl: float                  # Tỷ lệ nhãn phụ thuộc (%)
    labels_residual_dl: Tuple[int, ...]     # Danh sách chỉ số nhãn DL
    sparse_active_connections: int          # Tổng số cạnh tương quan >= theta_corr trong DL
    stopping_reason: str                    # Lý do dừng bóc tách
    selection_time_seconds: float           # Thời gian thực thi phân tầng (s)
```

### 6.2. Định dạng xuất file báo cáo

Khi chạy thực nghiệm benchmark, hệ thống tự động xuất các bảng kiểm toán sau:
1. **File chi tiết từng fold:** `results_v5_1_test/stage_audit_details.csv`.
2. **File tổng hợp trung bình theo Dataset:** `results_v5_1_test/stage_audit_summary.csv`.
3. **Bảng hiển thị trực quan trong báo cáo Markdown:**

| Tập Dữ Liệu | Tổng K | $|IL_1|$ | $|IL_2|$ | $|IL_3|$ | Tổng $|IL|$ | Tỷ Lệ IL (%) | $|DL_{\text{res}}|$ | Số Cạnh Phụ Thuộc ($\ge \theta$) | Lý Do Dừng |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **emotions** | 6 | 2.4 | 0.8 | 0.0 | 3.2 | 53.3% | 2.8 | 1.4 | no_promotion |
| **scene** | 6 | 3.0 | 0.6 | 0.0 | 3.6 | 60.0% | 2.4 | 1.0 | no_promotion |
| **chd49** | 49 | 18.2 | 4.6 | 1.2 | 24.0 | 49.0% | 25.0 | 12.6 | max_depth |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

---

## 7. MODULE MỞ RỘNG (OPTION): HIỆU NĂNG BR TRÊN IL & NGƯỠNG SUY GIẢM THÍCH ỨNG

### 7.1. Đánh giá hiệu năng chuyên biệt của Binary Relevance trên tập $IL$ (Option 1)

Yêu cầu `option` số 1 trong `meeting_summary.md`:
> *"Xem các mô hình BR hiệu năng như nào trên các nhãn độc lập"*

**Thiết kế thí nghiệm:**
- Với mỗi tập dữ liệu, sau khi xác định được tập nhãn độc lập $\mathcal{I}$, trích xuất ma trận dự đoán của mô hình cơ sở Binary Relevance thuần túy trên tập $\mathcal{I}$.
- So sánh hiệu năng của mô hình BR độc lập này với:
  1. Hiệu năng của mô hình Classifier Chains (CC) khi chạy trên cùng tập nhãn $\mathcal{I}$.
  2. Hiệu năng của chính mô hình GSI v5.1.1.
- **Mục tiêu khoa học:** Chứng minh rằng trên tập nhãn $\mathcal{I}$, mô hình Binary Relevance đạt hiệu năng tương đương hoặc vượt trội Classifier Chains, khẳng định việc tách $\mathcal{I}$ ra khỏi chuỗi CC là hoàn toàn đúng đắn và không làm mất mát thông tin.

### 7.2. Cơ chế suy giảm ngưỡng bóc tách theo tầng (Decaying Threshold - Option 2)

Yêu cầu `option` số 2 trong `meeting_summary.md`:
> *"Thử giảm threshold theo các lần lặp - Làm sau khi chạy thử nghiệm bước trên"*

**Động lực:**
- Ở Tầng 1, yêu cầu khắt khe $\tau_1 = 0.80$ để chỉ những nhãn thực sự mạnh từ $X$ mới được vào $\mathcal{I}_1$.
- Ở Tầng 2, các nhãn còn lại đã được bổ sung thêm thông tin từ $\mathcal{I}_1$; lúc này có thể hạ nhẹ ngưỡng $\tau_2 = 0.75$.
- Ở Tầng 3, hạ tiếp xuống $\tau_3 = 0.70$ để thu nạp các nhãn bán độc lập, chỉ để lại những nhãn thực sự phức tạp nhất vào $DL$.

**Công thức suy giảm tuyến tính:**
$$\tau(t) = \tau_{\text{start}} - (t - 1) \times \Delta_{\tau}, \qquad \text{với } \tau_{\text{start}} = 0.80, \ \Delta_{\tau} = 0.05$$
Hệ thống cung cấp cờ `--use_decaying_threshold` trong script thực nghiệm để kích hoạt chế độ này khi cần so sánh đối chuẩn.

---

## 8. ĐẶC TẢ KIẾN TRÚC CODE & API CONTRACTS

### 8.1. Danh mục file cập nhật và cấu trúc module mới

```
d:/University_Subject/ML Research/BR_CC/
├── spec/
│   ├── spec_v5.md                      # Đặc tả v5 cũ (Greedy Selection)
│   ├── spec_V5_1.md                    # Đặc tả v5.1 (Stratified Peeling)
│   └── spec_v5_1_1.md                  # ĐẶC TẢ KỸ THUẬT V5.1.1 (FILE HIỆN TẠI)
├── src/
│   ├── selection/
│   │   ├── stratified_peeling.py       # Cập nhật: Feature Normalization & Fine-grained Audit
│   │   └── sparse_chain.py             # MỚI: Xây dựng đồ thị CC theo ngưỡng tương quan theta_corr
│   ├── models/
│   │   ├── gsi_mlc_pa.py               # Cập nhật: Tích hợp Sparse CC & Static Context Features
│   │   └── classifier_chain.py         # Cập nhật: SparseClassifierChainClassifier
│   └── evaluation/
│       └── layer_ablation.py           # MỚI: Khung thực nghiệm bóc tách Ablation (Module 1)
└── scripts/
    ├── run_v5_1_1_experiments.py       # Script chạy thực nghiệm chuẩn v5.1.1 trên 10 datasets
    └── run_layer_ablation_study.py     # Script chạy 6 cấu hình bóc tách tầng (Ablation Study)
```

### 8.2. Chi tiết API Class & Method signatures

#### 1. Cập nhật `StratifiedPeelingConfig` trong `src/selection/stratified_peeling.py`:
```python
@dataclass(frozen=True)
class StratifiedPeelingConfig:
    threshold: float = 0.75
    max_depth: int = 3
    dl_order_direction: str = "ascending"
    min_labels_residual: int = 1
    use_complexity_penalty: bool = False
    penalty_config: Optional[ComplexityPenaltyConfig] = None
    
    # BỔ SUNG CHO V5.1.1:
    aug_normalization: str = "matching"      # Options: ["matching", "centered", "logit", "none"]
    decaying_threshold: bool = False         # Option 2: Giảm ngưỡng theo tầng
    threshold_decay_step: float = 0.05       # Bước giảm delta tau
```

#### 2. Module mới: `SparseClassifierChainClassifier` trong `src/models/sparse_chain.py`:
```python
class SparseClassifierChainClassifier(BaseEstimator, ClassifierMixin):
    """Classifier Chain with correlation-threshold filtered active predecessor conditioning."""
    def __init__(
        self,
        base_estimator=None,
        order=None,
        correlation_matrix=None,
        correlation_threshold=0.75,
        random_state=42,
    ):
        self.base_estimator = base_estimator
        self.order = order
        self.correlation_matrix = correlation_matrix
        self.correlation_threshold = float(correlation_threshold)
        self.random_state = random_state
        self.classifiers_ = []
        self.active_parents_map_ = {}

    def fit(self, X_context, Y_dl):
        """Fit sparse CC where each classifier only conditions on parents with corr >= threshold."""
        ...
```

#### 3. Module mới: `LayerAblationEvaluator` trong `src/evaluation/layer_ablation.py`:
```python
class LayerAblationEvaluator:
    """Evaluates the 6 layer-wise ablation regimes defined in Module 1."""
    def __init__(self, model, cost=0.30):
        self.model = model
        self.cost = cost

    def evaluate_regimes(self, X_test, Y_test) -> Dict[str, Dict[str, float]]:
        """Run evaluation across all 6 ablation configurations and return metrics dict."""
        ...
```

---

## 9. KẾ HOẠCH THỰC NGHIỆM VÀ LỘ TRÌNH TRIỂN KHAI (ACTION PLAN)

Khung triển khai được chia làm 4 giai đoạn logic và tuần tự:

```
+========================================================================================+
|                        LỘ TRÌNH TRIỂN KHAI THỰC NGHIỆM V5.1.1                          |
+========================================================================================+

  [ Giai đoạn 1: Chuẩn Hóa Feature & Kiểm Toán Tầng (Core 3 & 4) ]
   1. Cập nhật `src/selection/stratified_peeling.py`:
      - Tích hợp hàm chuẩn hóa `AugmentationNormalizer` cho vector xác suất mềm P_hat(Y_IL).
      - Mở rộng cấu trúc trả về `StratifiedPeelingResult` với audit chi tiết từng tầng.
   2. Viết unit-test kiểm chứng tính bảo toàn thang đo của feature chuẩn hóa.

  [ Giai đoạn 2: Tái Thiết Kế Sparse CC Theo Ngưỡng Tương Quan (Core 2 & 5) ]
   1. Xây dựng module `src/models/sparse_chain.py`:
      - Lọc nhãn cha có |phi_ij| >= theta_corr in [0.70, 0.80].
      - Nhận không gian ngữ cảnh X_context_DL = [X, Normalize(P_hat(Y_IL))] làm đặc trưng cố định.
   2. Tích hợp vào mô hình chính `src/models/gsi_mlc_pa.py` (v5.1.1 mode).
   3. Kiểm thử trên dataset mẫu (`emotions`, `scene`) để xác nhận mã nguồn chạy mượt mà.

  [ Giai đoạn 3: Hiện Thực Hóa Khung Thực Nghiệm Bóc Tách (Core 1) ]
   1. Xây dựng module `src/evaluation/layer_ablation.py`.
   2. Hiện thực hóa 6 cấu hình:
      - ABL_1_ONLY_IL1
      - ABL_2_ONLY_IL2
      - ABL_2_ACCUM_IL12
      - ABL_3_ONLY_IL3
      - ABL_3_ALL_IL
      - ABL_3_ONLY_DL
   3. Viết script `scripts/run_layer_ablation_study.py` để tự động hóa quá trình chạy.

  [ Giai đoạn 4: Chạy Benchmark Toàn Diện 10 Datasets & Xuất Báo Cáo ]
   1. Chạy 5-Fold Cross-Validation trên toàn bộ 10 datasets với 3 base learners:
      - Logistic Regression
      - Linear SVM
      - MLP (GPU PyTorch)
   2. Xuất dữ liệu kiểm toán chi tiết ra `results_v5_1_test/stage_audit_summary.csv`.
   3. Xuất bảng so sánh đối chuẩn 7 nhóm metric và kết quả bóc tách ablation vào báo cáo tổng hợp.
+========================================================================================+
```

---

## 10. KẾT LUẬN & ĐÓNG GÓP KHOA HỌC DỰ KIẾN CHO BÀI BÁO

Phiên bản **GSI-MLC-PA v5.1.1** hoàn thiện trọn vẹn bức tranh phương pháp luận khoa học:
1. **Tính minh bạch và chặt chẽ của phân tầng:** Việc ghi nhận cụ thể số lượng nhãn và danh sách nhãn qua từng tầng giải tỏa hoàn toàn nghi ngờ về tính "hộp đen" của thuật toán.
2. **Loại bỏ nhiễu bằng cấu trúc thưa:** Chuyển đổi từ chuỗi CC dày đặc sang chuỗi chọn lọc theo ngưỡng tương quan cao ($\theta_{\text{corr}} \ge 0.70$) giúp giảm mạnh độ phức tạp tham số và ngăn chặn hiện tượng ô nhiễm đặc trưng.
3. **Chuẩn hóa nhất quán:** Đảm bảo độ ổn định số học và hội tụ tối ưu cho cả 3 họ mô hình (Logistic, SVM, Deep MLP).
4. **Bằng chứng bóc tách không thể bác bỏ (Irrefutable Ablation Evidence):** Khung thí nghiệm 6 cấu hình bóc tách sẽ cung cấp bằng chứng thực nghiệm rõ ràng nhất về sự cần thiết của từng tầng nhãn độc lập và chuỗi phụ thuộc, tạo nền tảng vững chắc để bài báo đạt chuẩn xuất bản tại các hội nghị/tạp chí hàng đầu (IEEE TKDE, JAIR, ESWA).
