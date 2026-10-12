# TÀI LIỆU ĐẶC TẢ KỸ THUẬT PHIÊN BẢN v6.3: BÓC TÁCH IL MỘT TẦNG VÀ ABLATION CƠ CHẾ SUY LUẬN DL BẰNG CC/PCC

**Tài liệu tham chiếu:** `comment.md`, `ablation.md`, `spec/spec_v6_2.md`, `src/models/gsi_v6_2.py`, `src/evaluation/metric_contract.py`, `TaiLieuThamKhao/PCC.pdf` (Mục 7.1.3)  
**Nhánh Git:** `v6`  
**Tên phiên bản:** `v6.3`  
**Thư mục kết quả:** `result_pa_v6_3/`  
**Ngày lập đặc tả:** 11/10/2026  
**Trạng thái:** Đã triển khai và hoàn tất full run ngày 12/10/2026; dataset, shared chain order và số ancestral samples đã được xác nhận

---

## 1. Bối Cảnh và Mục Tiêu Kỹ Thuật (Motivation & Objectives)

### 1.1. Kiến trúc đối chứng v6.2

GSI-MLC-PA v6.2 gồm hai pha học chính:

1. Bóc tách nhãn độc lập đa tầng bằng dự đoán ngoài mẫu (Out-of-Fold - OOF). Ở mỗi tầng, nhãn đạt ngưỡng Selective Macro-F1/F1 cấu hình được chuyển từ tập ứng viên sang tập nhãn độc lập $IL_i$; xác suất OOF của các nhãn đã chọn được bổ sung vào không gian đặc trưng cho tầng kế tiếp.
2. Với tập nhãn phụ thuộc còn lại $DL$, v6.2 tính tương quan Pearson tuyệt đối giữa các vector sai số BR, xây dựng tập cha cục bộ $DL_{temp}[l]$, rồi huấn luyện một BR có điều kiện cho từng nhãn $l \in DL$.

Hai thay đổi cần khảo sát ở v6.3 là số tầng bóc tách $IL$ và bộ suy luận dùng cho $DL$. Các thay đổi này phải được cô lập thành các cấu hình riêng để có thể quy kết chênh lệch kết quả cho đúng thành phần.

### 1.2. Mục tiêu nghiên cứu của v6.3

v6.3 trả lời hai câu hỏi nghiên cứu:

- **RQ1 - Độ sâu phân hoạch:** Chỉ chọn $IL$ tại tầng đầu tiên có tốt hơn bóc tách đa tầng của v6.2 hay không, khi giữ nguyên cơ chế suy luận $DL$ của v6.2?
- **RQ2 - Cơ chế suy luận $DL$:** Với phân hoạch một tầng được cố định, chất lượng thay đổi thế nào khi thay BR có điều kiện của v6.2 bằng Classifier Chain (CC) hoặc Probabilistic Classifier Chain (PCC), trên hai không gian đặc trưng $FS=X$ và $FS=[X,\hat P_{IL}]$?

Các khía cạnh đánh giá bắt buộc gồm:

- chất lượng dự đoán đầy đủ và dự đoán có từ chối;
- số lượng nhãn $IL$ theo từng tầng và số lượng nhãn $DL$;
- Macro-F1 và Coverage theo từng tầng $IL$ và trên $DL$;
- chi phí huấn luyện toàn mô hình, phần $IL$, phần $DL$;
- độ nhạy theo chi phí từ chối;
- so sánh công bằng trên cùng fold, seed, bộ học cơ sở và dữ liệu.

### 1.3. Phạm vi và nguyên tắc cô lập

- Không sửa hành vi của `GSIMLCPAv6_2Classifier`; v6.2 là đối chứng bất biến.
- Các mô hình `BR`, `CC`, `MLC_PA` và `GSI_v6_2` phải dùng kết quả đã có. Không chạy lại chỉ để bổ sung metric còn thiếu; ô thiếu được ghi `-` theo yêu cầu trong `ablation.md`.
- `BR` và `CC` là baseline dự đoán đầy đủ, không có cơ chế từ chối. Chúng vẫn xuất hiện trong bảng/hình selective và rejection-cost như **complete-metric references**: selective metric bằng đúng complete metric, `coverage=1`, `ABS=AABS=0`, generalized loss bằng Hamming loss và giá trị không đổi theo cost. Đây là đường tham chiếu, không phải kết quả áp dụng từ chối.
- Chỉ các cấu hình v6.3 mới hoặc cấu hình chịu ảnh hưởng trực tiếp bởi thay đổi cơ chế mới được chạy.
- `hamming_bop` là cơ chế quyết định được thực thi trong lần chạy này. Nhánh `f_measure_bop` chỉ được tạo khung thư mục và để trống, không sinh checkpoint, bảng hay hình kết quả.
- Không dùng tập test để chọn $IL$, thứ tự chuỗi, siêu tham số PCC, ngưỡng hoặc operating point.

---

## 2. Ma Trận Mô Hình và Thiết Kế Ablation

### 2.1. Các đối chứng kế thừa, không chạy lại

| Mã mô hình | Tên hiển thị | Vai trò | Nguồn kết quả |
|:---|:---|:---|:---|
| `BR` | Binary Relevance | Đối chứng không mô hình hóa phụ thuộc nhãn | Kế thừa kết quả dự án |
| `CC` | Classifier Chain | Đối chứng chuỗi toàn bộ nhãn | Kế thừa kết quả dự án |
| `MLC_PA` | MLC-PA | Đối chứng dự đoán có từ chối | Kế thừa kết quả dự án |
| `GSI_v6_2` | GSI-MLC-PA v6.2 | Đối chứng bóc tách đa tầng + BR phụ thuộc theo tương quan sai số | Kế thừa kết quả v6.2 |

Việc kế thừa chỉ hợp lệ khi checkpoint có cùng dataset fingerprint, outer-fold split, seed, bộ học cơ sở, tiền xử lý, metric contract và cost. Nếu không chứng minh được tính tương thích, hàng vẫn được nhập để báo cáo nhưng phải mang trạng thái `inherited_unmatched`; không được dùng hàng đó cho kiểm định cặp.

### 2.2. Đối chứng bắt buộc cho RQ1: chỉ thay đổi số tầng bóc tách

Để so sánh một tầng với đa tầng mà không làm lẫn ảnh hưởng của bộ suy luận $DL$, v6.3 phải có thêm cấu hình:

| Mã mô hình | Phân hoạch | $FS$ cho DL | Bộ suy luận DL | Mục đích |
|:---|:---|:---|:---|:---|
| `GSI_v6_3_1L_ResidualBR` | Chỉ $IL_1$ | $[X,\hat P_{IL_1}]$ | BR có điều kiện theo tương quan sai số giống v6.2 | So sánh trực tiếp với `GSI_v6_2` để trả lời RQ1 |

Cấu hình này không được bỏ qua. Nếu chỉ so sánh v6.2 với một cấu hình đồng thời đổi cả độ sâu phân hoạch và bộ suy luận $DL$, kết quả không thể xác định cải thiện đến từ thay đổi nào.

### 2.3. Bốn cấu hình ablation suy luận DL theo `ablation.md`

Tất cả bốn cấu hình dưới đây dùng cùng phân hoạch một tầng $IL_1/DL$ của `GSI_v6_3_1L_ResidualBR` trong từng outer fold. Phân hoạch được tính một lần, lưu checkpoint, sau đó tái sử dụng nguyên trạng.

| Ablation | Mã mô hình | Không gian đặc trưng nền của DL | Bộ suy luận DL |
|:---|:---|:---|:---|
| 1.1.1 | `GSI_v6_3_CC_X` | $FS=X$ | CC |
| 1.1.2 | `GSI_v6_3_PCC_X` | $FS=X$ | PCC |
| 1.2.1 | `GSI_v6_3_CC_X_IL` | $FS=[X,\hat P_{IL_1}]$ | CC |
| 1.2.2 | `GSI_v6_3_PCC_X_IL` | $FS=[X,\hat P_{IL_1}]$ | PCC |

Ma trận này tạo hai phép so sánh trực tiếp:

- CC so với PCC khi giữ cố định không gian đặc trưng;
- $X$ so với $[X,\hat P_{IL_1}]$ khi giữ cố định bộ suy luận.

### 2.4. Ma trận so sánh cuối cùng

Mỗi bộ học cơ sở được chọn phải có tối đa chín dòng mô hình:

1. `BR`;
2. `CC`;
3. `MLC_PA`;
4. `GSI_v6_2`;
5. `GSI_v6_3_1L_ResidualBR`;
6. `GSI_v6_3_CC_X`;
7. `GSI_v6_3_PCC_X`;
8. `GSI_v6_3_CC_X_IL`;
9. `GSI_v6_3_PCC_X_IL`.

Mô hình thứ năm là đối chứng bổ sung cần thiết cho mục tiêu một tầng/đa tầng. Tám mô hình còn lại bao gồm đầy đủ danh sách bắt buộc trong `ablation.md`.

---

## 3. Nền Tảng Toán Học (Mathematical Foundations)

### 3.1. Bóc tách $IL$ một tầng

Trong một outer-training fold, với dữ liệu $(X,Y)$ gồm $N$ mẫu và $K$ nhãn, mỗi nhãn $l$ được đánh giá bằng xác suất OOF từ $q=5$ inner folds:

$$
\hat p_l^{\mathrm{OOF}} = \operatorname{OOFProb}(X,y_l;q).
$$

Áp dụng chính sách `hamming_bop` tại chi phí $c$ để tạo dự đoán một phần $\hat y_l^{\mathrm{partial}}$. Điểm dùng để quyết định thăng hạng giữ nguyên contract của v6.2:

$$
s_l = \operatorname{SelectiveF1}(y_l,\hat y_l^{\mathrm{partial}}).
$$

Phân hoạch một tầng được xác định bởi:

$$
IL_1 = \{l \mid s_l \ge \tau_{F1}\},
\qquad
DL = \mathcal L \setminus IL_1.
$$

Sau khi xác định $IL_1$, thuật toán dừng bóc tách. Không tạo $IL_2,IL_3,\ldots$ và không đánh giá lại nhãn $DL$ trên $[X,\hat P_{IL_1}^{\mathrm{OOF}}]$.

Quy tắc singleton của v6.2 không được tự động chuyển nhãn $DL$ cuối cùng sang một tầng $IL$ mới, vì hành vi này sẽ vi phạm điều kiện “chỉ chọn IL ở tầng 1”. Nếu $|DL|=1$, nhãn đó vẫn thuộc $DL$ và được bộ suy luận DL xử lý theo cấu hình đang chạy.

### 3.2. Hai không gian đặc trưng cho $DL$

Hai chế độ được định nghĩa tách biệt:

$$
FS_X = X,
$$

$$
FS_{X+IL} = [X,\operatorname{Normalize}(\hat P_{IL_1}^{\mathrm{OOF}})].
$$

Ở thời điểm suy diễn, $\hat P_{IL_1}^{\mathrm{OOF}}$ được thay bằng xác suất của các mô hình $IL_1$ đã refit trên toàn bộ outer-training fold:

$$
FS_{X+IL}^{test} = [X^{test},\operatorname{Normalize}(\hat P_{IL_1}^{test})].
$$

Không được dùng ground truth của $IL_1$ làm đặc trưng cho $DL$ ở test. Chiến lược chuẩn hóa phải giống v6.2 và được lưu trong checkpoint.

### 3.3. Classifier Chain cho tập $DL$

Với thứ tự nhãn $\pi=(\pi_1,\ldots,\pi_m)$ trên $DL$, CC huấn luyện:

$$
f_i:\left(FS,y_{\pi_1},\ldots,y_{\pi_{i-1}}\right)\mapsto y_{\pi_i}.
$$

Trong huấn luyện, các nhãn tiền nhiệm lấy từ ground truth của outer-training fold. Trong suy diễn, triển khai v6.3 dùng cơ chế greedy tương thích `ClassifierChainClassifier`: dự đoán nhị phân của các tiền nhiệm được ghép vào đặc trưng cho bước sau; xác suất của từng bộ phân loại được lưu để tính metric và BOP.

CC và PCC phải dùng cùng một thứ tự $\pi$ trong cùng fold để tránh biến thứ tự chuỗi thành một yếu tố ablation thứ ba.

Qua audit, `src/models/gsi_v6_2.py` xây đồ thị cha cục bộ từ tương quan sai số và không tạo một thứ tự chuỗi toàn cục. Vì vậy, mô tả “dùng tương quan để chọn thứ tự” trong `ablation.md` chưa đủ để suy ra $\pi$ cho CC/PCC; chiến lược order phải được xác nhận và khóa trong cấu hình trước full run.

### 3.4. Probabilistic Classifier Chain cho tập $DL$

Trong mục này, **PCC** là *Probabilistic Classifier Chain*, không phải Pearson Correlation Coefficient dùng trong v6.2.

Theo Mục 7.1.3 của `TaiLieuThamKhao/PCC.pdf`, PCC phân rã phân phối xác suất đồng thời có điều kiện bằng quy tắc tích:

$$
P(\mathbf y\mid FS)=\prod_{i=1}^{m}P\left(y_{\pi_i}\mid FS,y_{\pi_1},\ldots,y_{\pi_{i-1}}\right).
$$

Mỗi mắt xích học một hàm xác suất:

$$
f_i:\left(FS,y_{\pi_1},\ldots,y_{\pi_{i-1}}\right)
\mapsto
P\left(y_{\pi_i}=1\mid FS,y_{\pi_1},\ldots,y_{\pi_{i-1}}\right).
$$

Đối với v6.3, PCC phải tạo xác suất biên để dùng chung lớp metric và `hamming_bop`. Với mỗi mẫu $x$, lấy $S$ mẫu tổ tiên (*ancestral samples*) $\mathbf y^{(1)},\ldots,\mathbf y^{(S)}$ từ chuỗi, sau đó ước lượng:

$$
\hat p_j(x)=\frac{1}{S}\sum_{s=1}^{S}\mathbb{1}\left[y_j^{(s)}=1\right].
$$

Dự đoán đầy đủ dùng ngưỡng $0.5$ trên $\hat p_j$. Dự đoán có từ chối đưa cùng ma trận xác suất này vào `hamming_bop` theo từng chi phí $c$.

Việc dùng lấy mẫu tổ tiên để ước lượng xác suất biên là lựa chọn tích hợp của v6.3; nó không đồng nghĩa với kích hoạt thuật toán GFM hay `f_measure_bop`. Đã xác nhận $S=1{,}000$ ancestral samples cho mỗi mẫu kiểm tra. CC và PCC dùng chung thứ tự DL theo tổng tương quan tăng dần do selector tạo trong từng fold. Số mẫu $S$, seed lấy mẫu và thứ tự chuỗi phải được lưu trong checkpoint.

### 3.5. Chính sách từ chối `hamming_bop`

Với penalty tuyến tính, `linear_boundary="minimum_loss"` và xác suất biên $p_{ij}$, luật quyết định theo từng vị trí là:

$$
\hat y_{ij}=\begin{cases}
\mathbb{1}[p_{ij}\ge 0.5], & \min(p_{ij},1-p_{ij})\le c,\\
-1, & \min(p_{ij},1-p_{ij})>c.
\end{cases}
$$

Trong đó $-1$ biểu diễn từ chối. Lưới chi phí kế thừa cấu hình thực nghiệm hiện hành:

$$
\mathcal C=\{0.20,0.25,0.30,0.35,0.40,0.50\},
$$

và chi phí báo cáo chính là $c=0.30$. Hai giá trị này phải được khóa trong file cấu hình; thay đổi lưới chi phí tạo một config hash mới.

### 3.6. Metric đầy đủ và metric chọn lọc

Các metric đầy đủ dùng dự đoán nhị phân không từ chối:

- Hamming Accuracy;
- Subset Accuracy;
- Micro-F1;
- Macro-F1;
- Instance-F1.

Instance-F1 kế thừa contract hiện có của dự án:

$$
\operatorname{InstanceF1}=\frac{1}{N}\sum_{i=1}^{N}
\frac{2TP_i}{2TP_i+FP_i+FN_i},
$$

trong đó trường hợp tập nhãn thật và tập nhãn dự đoán của cùng một mẫu đều rỗng nhận điểm $1$.

Đối với dự đoán chọn lọc, mọi vị trí từ chối bị loại khỏi tập vị trí được chấm metric. Các metric bắt buộc gồm:

- Selective Hamming Accuracy;
- Selective Subset Accuracy;
- Selective Micro-F1;
- Selective Macro-F1;
- Selective Instance-F1;
- Generalized Loss;
- Coverage, ABS và AABS.

Selective Macro-F1, Selective Micro-F1, Selective Instance-F1, Generalized Loss, Coverage, ABS và AABS phải kế thừa implementation/edge cases trong `src/evaluation/metrics.py` và metric contract của dự án. Không được thay `Selective Instance-F1` bằng `Optimistic Instance-F1`.

`Selective Subset Accuracy` chưa có trong metric contract hiện tại. Quy ước tạm thời của v6.3 là: với mỗi mẫu có ít nhất một vị trí được quyết định, điểm bằng $1$ khi tất cả vị trí được quyết định đều đúng; mẫu bị từ chối toàn bộ bị loại khỏi mẫu số. Nếu không có mẫu nào có quyết định, kết quả là `NaN`. `[CẦN XÁC NHẬN: selective_subset_accuracy]`.

Generalized Loss tuyến tính được tính bởi:

$$
\mathcal L_{gen}=\frac{1}{NK}\sum_{i=1}^{N}
\left(e_i+c\,a_i\right),
$$

trong đó $e_i$ là số quyết định sai và $a_i$ là số nhãn bị từ chối của mẫu $i$.

---

## 4. Thuật Toán Chi Tiết và Mã Giả (Detailed Pseudocode)

```text
====================================================================================================
Thuật toán: GSI-MLC-PA v6.3 - Single-Layer IL + Configurable DL Inference
====================================================================================================
ĐẦU VÀO:
  X, Y                    : outer-training data
  base_learner            : logistic | svm_calibrated | mlp
  tau_f1                  : ngưỡng chọn IL_1, mặc định kế thừa v6.2 (= 0.75)
  inner_folds             : 5
  dl_feature_space        : x | x_il
  dl_inference            : residual_br | cc | pcc
  dl_chain_order          : correlation_ascending, dùng chung cho CC và PCC
  pcc_num_samples         : 1000 ancestral samples cho mỗi mẫu kiểm tra
  cost_grid               : [0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
  random_state            : 42

ĐẦU RA:
  model_v6_3
  IL_1, DL
  OOF partition audit
  fold predictions/probabilities
  layer metrics and timing audit

PHA 1 - PHÂN HOẠCH MỘT TẦNG:
1:  IL_1 <- empty
2:  DL <- {0, 1, ..., K-1}
3:  FOR EACH label l IN DL DO
4:      p_oof[l] <- FiveFoldOOFProbability(X, Y[:, l], base_learner)
5:      y_partial_oof[l] <- HammingBOP(p_oof[l], report_cost)
6:      score[l] <- SelectiveF1(Y[:, l], y_partial_oof[l])
7:      IF score[l] >= tau_f1 THEN
8:          IL_1 <- IL_1 UNION {l}
9:      END IF
10: END FOR
11: DL <- ALL_LABELS MINUS IL_1
12: DỪNG BÓC TÁCH; KHÔNG TẠO IL_2

PHA 2 - HUẤN LUYỆN IL_1:
13: Train one BR model for each l in IL_1 using X
14: Reuse p_oof[IL_1] as leakage-free train augmentation
15: At inference, predict p_test[IL_1] from the refitted BR models

PHA 3 - CHỌN KHÔNG GIAN ĐẶC TRƯNG DL:
16: IF dl_feature_space == "x" THEN
17:     FS_train <- X
18:     FS_test  <- X_test
19: ELSE IF dl_feature_space == "x_il" THEN
20:     FS_train <- [X, Normalize(p_oof[IL_1])]
21:     FS_test  <- [X_test, Normalize(p_test[IL_1])]
22: END IF

PHA 4 - HUẤN LUYỆN VÀ SUY LUẬN DL:
23: IF dl_inference == "residual_br" THEN
24:     Execute exactly the v6.2 residual-correlation conditional-BR procedure on DL
25: ELSE IF dl_inference == "cc" THEN
26:     Fit one Classifier Chain on Y[:, DL] with fixed order pi
27:     Predict greedily and return per-label probabilities
28: ELSE IF dl_inference == "pcc" THEN
29:     Fit probabilistic chain factors on Y[:, DL] with the same order pi
30:     Draw S ancestral samples per test instance using a deterministic fold seed
31:     Estimate marginal p_test[DL] by sample frequency
32: END IF

PHA 5 - KẾT HỢP ĐẦU RA VÀ ĐÁNH GIÁ:
33: Assemble p_test[IL_1] and p_test[DL] in original label order
34: y_full <- (p_test >= 0.5)
35: FOR EACH c IN cost_grid DO
36:     y_partial[c] <- HammingBOP(p_test, c)
37:     Compute full, selective, calibration and layer/group metrics
38: END FOR
39: Save checkpoint, predictions, provenance, timings and fold-complete marker atomically
====================================================================================================
```

### 4.1. Quy tắc biên

- **$IL_1=\emptyset$:** $FS_{X+IL}$ thoái hóa có kiểm soát thành $X$; vẫn giữ mã cấu hình `x_il` và ghi `effective_feature_space=x` để audit.
- **$DL=\emptyset$:** không khởi tạo CC/PCC; xác suất cuối chỉ đến từ BR trên $IL_1$.
- **$|DL|=1$:** CC/PCC có chuỗi một phần tử và tương đương một bộ phân loại xác suất đơn; nhãn không được chuyển sang $IL$.
- **Nhãn chỉ có một lớp trong train fold:** dùng constant classifier, ghi cờ `single_class_train=true`; PCC vẫn phải trả xác suất 0 hoặc 1 hợp lệ.
- **PCC tái lập:** seed lấy mẫu phải là hàm xác định của `random_state`, fold, dataset và label-order hash.
- **Resume:** fold chỉ được coi là hoàn tất khi checkpoint và predictions đều tồn tại, hash khớp và marker `status=complete` đã được ghi nguyên tử.

---

## 5. Giao Thức Thực Nghiệm (Experimental Protocol)

### 5.1. Dữ liệu và bộ học cơ sở

- Các bộ học cơ sở hiện có: `logistic`, `svm_calibrated`, `mlp`.
- Runner cho phép chọn một hoặc nhiều dataset. Giá trị mặc định phải lấy từ file cấu hình v6.3, không hard-code trong model.
- Danh sách chính thức đã được khóa trong `configs/v6_3_run.json` và gồm đúng 10 dataset của v6.2: `emotions`, `scene`, `chd49`, `music`, `gpositivepseaac`, `genbase`, `humanpseaac`, `plantpseaac`, `viruspseaac`, `yeast`.
- MeanIR dùng để sắp xếp hình phải được tính chỉ từ outer-training data của từng fold khi phục vụ phân tích fold; thứ tự hiển thị toàn cục dùng MeanIR của toàn bộ dataset metadata đã đóng băng, không lấy từ test prediction.

### 5.2. Chia fold và kiểm soát rò rỉ

1. Dùng cùng outer 5-fold multilabel-stratified split cho tất cả mô hình có so sánh cặp.
2. Mọi chuẩn hóa đặc trưng được fit trong outer-training fold.
3. Phân hoạch $IL_1/DL$, xác suất tăng cường $IL_1$, thứ tự chuỗi và mọi lựa chọn siêu tham số chỉ dùng outer-training data.
4. OOF bên trong outer-training fold dùng 5 folds hoặc số folds nhỏ hơn khi nhãn hiếm không cho phép; số folds hiệu lực phải được audit.
5. Test fold chỉ được dùng một lần để tính kết quả cuối.
6. Cùng seed và cùng split hash là điều kiện bắt buộc để thực hiện kiểm định thống kê cặp.

### 5.3. Đơn vị tổng hợp

- Kết quả chi tiết được lưu theo từng outer fold.
- Hình so sánh dùng trung bình năm outer folds; error bar dùng sample standard deviation (`ddof=1`).
- Không trung bình trực tiếp các mean của dataset để tuyên bố mô hình tốt hơn nếu kích thước/độ khó khác nhau; báo cáo phải hiển thị kết quả theo dataset trước khi nêu trung bình toàn cục.
- Với metric thiếu ở kết quả kế thừa, hiển thị `-`; khi tổng hợp, `-` là missing chứ không phải 0.

### 5.4. Phép so sánh chính

- **RQ1:** `GSI_v6_3_1L_ResidualBR - GSI_v6_2` trên từng dataset, learner, fold.
- **RQ2a:** `GSI_v6_3_PCC_X - GSI_v6_3_CC_X`.
- **RQ2b:** `GSI_v6_3_PCC_X_IL - GSI_v6_3_CC_X_IL`.
- **Ảnh hưởng IL context cho CC:** `GSI_v6_3_CC_X_IL - GSI_v6_3_CC_X`.
- **Ảnh hưởng IL context cho PCC:** `GSI_v6_3_PCC_X_IL - GSI_v6_3_PCC_X`.

Nếu bổ sung kiểm định thống kê, phải chốt trước statistical unit, test, alpha và multiplicity correction; không dùng từ “có ý nghĩa thống kê” trước khi các mục này được cấu hình.

---

## 6. Cấu Trúc Mã Nguồn Dự Kiến (Software Architecture)

```text
BR_CC/
├── configs/
│   └── v6_3_run.json
├── docs/
│   └── progress/
│       ├── phase_v6_3_00.md
│       ├── phase_v6_3_01.md
│       └── ...
├── scripts/
│   ├── run_v6_3_ablation.py
│   └── generate_v6_3_figures.py
├── src/
│   ├── evaluation/
│   │   ├── layer_metrics.py
│   │   └── result_store_v6.py
│   ├── models/
│   │   ├── probabilistic_classifier_chain.py
│   │   └── gsi_v6_3.py
│   └── selection/
│       └── single_stage_peeling.py
├── tests/
│   ├── test_v6_3_single_stage.py
│   ├── test_v6_3_pcc.py
│   ├── test_v6_3_result_store.py
│   └── test_v6_3_metrics.py
└── result_pa_v6_3/
```

### 6.1. `src/selection/single_stage_peeling.py`

- Tái sử dụng `evaluate_label_5fold_cv` và chuẩn hóa OOF của v6.2.
- Trả `IL_1`, `DL`, OOF probabilities, per-label scores, effective inner-fold count và stopping reason `single_stage_completed`.
- Không sửa `CVStratifiedPeelingSelector` để tránh làm thay đổi các phiên bản cũ.

### 6.2. `src/models/probabilistic_classifier_chain.py`

API tối thiểu:

```python
class ProbabilisticClassifierChainClassifier:
    def __init__(
        self,
        base_estimator,
        order,
        n_samples,
        random_state=42,
    ): ...

    def fit(self, X, Y): ...
    def sample(self, X, n_samples=None, random_state=None): ...
    def predict_proba(self, X): ...
    def predict(self, X, threshold=0.5): ...
```

`predict_proba` phải trả xác suất biên theo thứ tự cột ban đầu, không phải xác suất điều kiện cục bộ tại một đường greedy.

### 6.3. `src/models/gsi_v6_3.py`

Lớp `GSIMLCPAv6_3Classifier` có các tham số chính:

```python
GSIMLCPAv6_3Classifier(
    base_learner="logistic",
    stratified_threshold=0.75,
    cv_folds=5,
    max_peeling_depth=1,
    dl_feature_space="x_il",       # "x" | "x_il"
    dl_inference="residual_br",    # "residual_br" | "cc" | "pcc"
    dl_chain_order="correlation_ascending",
    pcc_num_samples=1000,
    cost=0.30,
    abstain_value=-1,
    random_state=42,
)
```

`max_peeling_depth` được cố định bằng `1` trong v6.3; truyền giá trị khác phải phát sinh `ValueError` để tránh chạy sai tên phiên bản.

### 6.4. Runner và result store

`scripts/run_v6_3_ablation.py` chịu trách nhiệm:

- đọc cấu hình đóng băng;
- kiểm tra config hash/dataset fingerprint/split hash;
- nhập kết quả kế thừa;
- chỉ lập lịch các mô hình mới hoặc bị ảnh hưởng;
- resume theo fold;
- lưu checkpoint/predictions nguyên tử;
- tổng hợp bảng, hình và thư mục `all`.

`src/evaluation/result_store_v6.py` chịu trách nhiệm duy nhất cho quy ước đường dẫn, schema CSV, provenance và thao tác kế thừa. Model không tự ghi file.

---

## 7. Quy Tắc Phiên Bản và Kế Thừa Kết Quả

### 7.1. Đánh số lần chạy

- Lần chạy cấu hình gốc, chưa đổi cơ chế, thuộc `result_pa_v6_1`.
- Mỗi lần thay đổi cơ chế của một hoặc nhiều mô hình, tăng hậu tố phiên bản thêm một.
- Đặc tả này tạo `result_pa_v6_3` và không ghi đè `result_pa_v6_1`, `result_pa_v6_2` hoặc các thư mục `results_v6_2` hiện có.

### 7.2. Chỉ chạy lại mô hình bị ảnh hưởng

| Nhóm | Hành động ở v6.3 |
|:---|:---|
| `BR`, `CC`, `MLC_PA`, `GSI_v6_2` | Kế thừa, không fit lại |
| `GSI_v6_3_1L_ResidualBR` | Chạy mới |
| Bốn biến thể CC/PCC | Chạy mới |

Khi kế thừa, dữ liệu phải được ghi vào cây kết quả mới kèm:

- `source_version`;
- `source_path`;
- `source_config_hash`;
- `source_split_hash`;
- `provenance_status` (`computed`, `inherited_matched`, `inherited_unmatched`, `missing`);
- checksum của artifact nguồn.

Không được giả lập metric còn thiếu từ bảng tổng hợp khác. Nếu predictions cũ không tồn tại hoặc không chứng minh được cùng fold, metric thiếu giữ `-`.

`comment.md` tham chiếu folder `results_pa_v6_2`, nhưng workspace hiện có `results_v6_2/` và không có `results_pa_v6_2/` hay `result_pa_v6_2/`. Audit triển khai đã khóa nguồn kế thừa tại `results_v6_2/v6_2_detailed_folds.csv`; mỗi hàng kế thừa vẫn phải mang provenance và chỉ được dùng cho kiểm định cặp khi split/config tương thích.

---

## 8. Cấu Trúc Thư Mục Kết Quả `result_pa_v6_3`

### 8.1. Cây thư mục bắt buộc

```text
result_pa_v6_3/
├── logistic/
│   ├── hamming_bop/
│   │   ├── checkpoints/
│   │   │   ├── BR/
│   │   │   │   ├── <dataset>.json
│   │   │   │   └── predictions/<dataset>.npz
│   │   │   ├── CC/
│   │   │   ├── MLC_PA/
│   │   │   ├── GSI_v6_2/
│   │   │   ├── GSI_v6_3_1L_ResidualBR/
│   │   │   ├── GSI_v6_3_CC_X/
│   │   │   ├── GSI_v6_3_PCC_X/
│   │   │   ├── GSI_v6_3_CC_X_IL/
│   │   │   └── GSI_v6_3_PCC_X_IL/
│   │   ├── figures/
│   │   └── tables/
│   └── f_measure_bop/
│       ├── checkpoints/
│       ├── figures/
│       └── tables/
├── svm/
│   ├── hamming_bop/{checkpoints,figures,tables}/
│   └── f_measure_bop/{checkpoints,figures,tables}/
├── mlp/
│   ├── hamming_bop/{checkpoints,figures,tables}/
│   └── f_measure_bop/{checkpoints,figures,tables}/
└── all/
    ├── hamming_bop/{checkpoints,figures,tables}/
    └── f_measure_bop/{checkpoints,figures,tables}/
```

Ba folder learner chỉ được tạo khi learner tương ứng có trong cấu hình. Cây trên minh họa cấu hình đầy đủ; trong `checkpoints` chỉ tạo folder cho các mô hình được chọn. Folder `all` luôn tồn tại để chứa tổng hợp liên learner.

### 8.2. Quy ước checkpoint và predictions

Mỗi `checkpoints/<model>/<dataset>.json` chứa:

- cấu hình đầy đủ và hash;
- fingerprint dữ liệu và split hash;
- trạng thái từng fold;
- danh sách $IL_1$, $DL$, điểm phân hoạch và thứ tự chuỗi theo fold;
- timing train/inference, timing $IL$, timing $DL$;
- PCC sampling metadata nếu có;
- metric theo fold;
- provenance nếu là kết quả kế thừa;
- danh sách đường dẫn predictions cùng checksum.

Mỗi `predictions/<dataset>.npz` chứa tối thiểu:

- `test_indices`;
- `y_true`;
- `probabilities`;
- `y_full`;
- `y_partial_c_<cost>` cho từng cost;
- `fold_ids`;
- `il_labels_by_fold`, `dl_labels_by_fold` hoặc tham chiếu đến checkpoint nếu mảng có độ dài biến đổi.

Trong folder `all`, checkpoint JSON là manifest tổng hợp đường dẫn của các learner; prediction NPZ có các key phân tách theo learner. Không huấn luyện mô hình tại tầng `all`.

### 8.3. Nhánh `f_measure_bop`

Trong v6.3, ba thư mục `checkpoints`, `figures`, `tables` dưới `f_measure_bop` được tạo để cố định schema nhưng không chứa artifact kết quả. Không sao chép kết quả `hamming_bop` sang nhánh này.

---

## 9. Đặc Tả Bảng Kết Quả (CSV Schemas)

Tất cả file CSV trong folder learner chứa đủ các mô hình được chọn, không chia tiếp theo model. Mỗi bảng có các cột định danh chung:

`version, base_learner, decision_policy, model, dataset, mean_ir, fold, aggregation, source_version, provenance_status, config_hash, split_hash`.

`fold` nhận `1..5`; các hàng tổng hợp dùng `aggregation=mean|std` và để `fold=-`. Giá trị metric thiếu ở artifact kế thừa được xuất dưới dạng chuỗi `-` theo yêu cầu. Loader phân tích phải chuyển `-` thành missing, không chuyển thành 0.

### 9.1. `complete_metric.csv`

Các cột metric bắt buộc:

- `hamming_accuracy`;
- `subset_accuracy`;
- `micro_f1`;
- `macro_f1`;
- `instance_f1`.

### 9.2. `selective_metric.csv`

Thêm cột `cost`. Các cột metric bắt buộc:

- `selective_hamming_accuracy`;
- `selective_subset_accuracy`;
- `selective_micro_f1`;
- `selective_macro_f1`;
- `selective_instance_f1`;
- `generalized_loss`;
- `coverage`;
- `abs`;
- `aabs`.

Đối với `BR` và `CC`, mỗi complete fold được lặp trên toàn bộ `cost_grid` chỉ để tạo đường tham chiếu không từ chối: Hamming/Subset/Micro-F1/Macro-F1 lấy đúng từ `complete_metric.csv`; `instance_f1` tiếp tục là `-` nếu artifact nguồn không có; `generalized_loss = 1 - hamming_accuracy`; `coverage=1`; `abs=aabs=0`. Trường `decision_policy` phải là `none_complete_reference` để không diễn giải nhầm là đã chạy BOP.

### 9.3. `calibration_metric.csv`

Các cột metric bắt buộc:

- `brier_score`;
- `ece`;
- `log_loss`;
- `ece_bin_count`.

Calibration được tính trên xác suất trước khi BOP ra quyết định. ECE phải kế thừa binning convention hiện có; không thay đổi số bin giữa các mô hình trong cùng lần chạy.

### 9.4. `model_performance.csv`

Một hàng tổng hợp trên mỗi model/dataset, kèm độ lệch chuẩn qua folds khi có:

- `fit_time_sec_mean`, `fit_time_sec_std`;
- `inference_time_sec_mean`, `inference_time_sec_std`;
- `il_time_sec_mean`, `il_time_sec_std`;
- `dl_time_sec_mean`, `dl_time_sec_std`;
- `il_count_mean`, `il_count_std`;
- `dl_count_mean`, `dl_count_std`;
- `dl_parent_count_mean`, `dl_parent_count_std`;
- `dl_parent_map_by_fold` (JSON string dạng `fold -> label -> parents`; với `GSI_v6_2`/ResidualBR là cha theo đồ thị cục bộ, với CC/PCC là toàn bộ tiền nhiệm đứng trước nhãn trong chuỗi);
- `chain_order_by_fold` (JSON string, dùng cho CC/PCC);
- `pcc_num_samples`;
- `computed_fold_count`, `inherited_fold_count`, `missing_fold_count`.

### 9.5. `layer_performance.csv`

File này đáp ứng yêu cầu lưu số nhãn, Macro-F1 và Coverage qua các tầng $IL$ và $DL$. Mỗi hàng có grain `model × dataset × fold × layer`:

- `layer_type` (`IL` hoặc `DL`);
- `layer_index` (`1..m` cho v6.2; luôn `1` cho IL của v6.3; `0` cho DL);
- `label_indices` (JSON string);
- `label_count`;
- `cumulative_label_count`;
- `full_macro_f1` trên đúng nhóm nhãn của hàng;
- `selective_macro_f1` trên đúng nhóm nhãn của hàng;
- `coverage` trên đúng nhóm nhãn của hàng;
- `cumulative_full_macro_f1`;
- `cumulative_selective_macro_f1`;
- `cumulative_coverage`;
- `status` (`ok`, `empty_group`, `missing_inherited_metric`).

Các hàng `aggregation=mean|std` cung cấp trực tiếp số lượng $IL$ trung bình ở mỗi tầng, số lượng $DL$ trung bình và các metric trung bình năm folds. Nhóm rỗng không được gán điểm 0; dùng `-` ở CSV xuất cho người đọc và `status=empty_group`.

### 9.6. Bảng tổng hợp trong folder `all`

Folder `all/hamming_bop/tables` chứa cùng năm file trên, thêm cột `base_learner`. Đây là phép nối có kiểm tra schema từ `logistic`, `svm`, `mlp`, không phải phép trung bình các learner trừ khi hàng có `aggregation=learner_mean` được ghi rõ.

---

## 10. Đặc Tả Hình Ảnh (Figures)

### 10.1. Hình so sánh metric đầy đủ

Trong mỗi folder `figures`, tạo năm PNG:

- `complete_hamming_accuracy.png`;
- `complete_subset_accuracy.png`;
- `complete_micro_f1.png`;
- `complete_macro_f1.png`;
- `complete_instance_f1.png`.

Mỗi hình so sánh trung bình năm folds của các mô hình theo từng dataset, có error bar độ lệch chuẩn.

### 10.2. Hình so sánh metric chọn lọc

Tại `report_cost=0.30`, tạo sáu PNG:

- `selective_hamming_accuracy.png`;
- `selective_subset_accuracy.png`;
- `selective_micro_f1.png`;
- `selective_macro_f1.png`;
- `selective_instance_f1.png`;
- `selective_generalized_loss.png`.

Các vị trí bị từ chối bị loại khỏi metric chọn lọc theo contract ở Mục 3.6; Generalized Loss vẫn tính cả lỗi được quyết định và penalty từ chối.

### 10.3. Hình theo chi phí từ chối

Tạo năm đường cong:

- `rejection_cost_generalized_loss.png`;
- `rejection_cost_hamming_accuracy.png`;
- `rejection_cost_micro_f1.png`;
- `rejection_cost_macro_f1.png`;
- `rejection_cost_instance_f1.png`.

Trục hoành là cost, trục tung là metric trung bình trên các dataset/folds thuộc đúng scope của folder. Hình phải ghi rõ số dataset thực dùng sau khi loại missing. BR/CC không có cơ chế từ chối nhưng vẫn xuất hiện làm complete-metric references trong sáu hình selective và năm hình rejection-cost. Trên rejection-cost, chúng là đường ngang nét đứt, lấy hoàn toàn từ complete metrics và ghi rõ `coverage=1`; không được mô tả là kết quả BOP.

### 10.4. Sắp xếp dataset và phong cách

- Dataset trên mọi hình so sánh được sắp xếp theo MeanIR tăng dần; khi MeanIR bằng nhau, dùng tên dataset tăng dần để kết quả tái lập.
- Folder `all` dùng bố cục nhiều panel, mỗi panel tương ứng một learner; không gộp xác suất hoặc dự đoán giữa learner.
- Giá trị thiếu hiển thị `N/A` hoặc để trống có chú giải, không vẽ thành 0.
- Folder `Hinhmau/` hiện tồn tại nhưng không chứa file. Vì vậy lần chạy này dùng `TaiLieuThamKhao/Mau_Bieu_Do.jpg` làm tham chiếu trình bày và ghi rõ lựa chọn trong config.

---

## 11. Nhật Ký Tiến Độ Theo Phase

Mỗi phase phải tạo hoặc cập nhật một file `docs/progress/phase_v6_3_XX.md`. Không đợi đến cuối dự án mới viết lại theo trí nhớ.

Mẫu bắt buộc:

```markdown
# Progress Phase v6.3.XX — <Tên phase>

Ngày cập nhật: YYYY-MM-DD
Trạng thái: in_progress | completed | blocked

## Công việc đã làm được

## File đã tạo hoặc thay đổi

## Lệnh/test đã chạy và kết quả

## Thông tin cần kiểm chứng
- Mỗi giả định, công thức hoặc lựa chọn chưa được người dùng/tài liệu chốt.

## Công việc bị block
- Nêu công việc, nguyên nhân, bằng chứng đã kiểm tra và điều kiện để mở block.

## Artifact/kết quả được tạo

## Công việc cho phase sau
```

Ví dụ, nếu triển khai một công thức Selective Subset Accuracy chưa được xác nhận, phase log phải ghi chính xác công thức tạm dùng, file triển khai và ảnh hưởng tới bảng/hình.

---

## 12. Lộ Trình Triển Khai (Implementation Roadmap)

| Phase | Công việc | Sản phẩm | Tiêu chí nghiệm thu |
|:---|:---|:---|:---|
| `v6.3.00` | Audit nguồn kết quả, đóng băng config, xác nhận dataset/PCC/order/metric | `configs/v6_3_run.json`, phase log | Hash cấu hình ổn định; không còn tham số bắt buộc chưa xác nhận trước full run |
| `v6.3.01` | Cài single-stage peeling và đối chứng ResidualBR | selector, `gsi_v6_3.py`, tests | Không sinh $IL_2$; singleton vẫn ở DL; test leakage pass |
| `v6.3.02` | Cài CC/PCC cho DL và hai chế độ feature space | PCC module, model variants, tests | Xác suất đúng shape/range; sampling tái lập; cùng partition/order |
| `v6.3.03` | Cài metric bổ sung, layer audit, result store và kế thừa | evaluation modules, result-store tests | CSV đúng schema; missing=`-`; provenance đầy đủ; resume an toàn |
| `v6.3.04` | Smoke test trên dataset nhỏ và cả ba learner | checkpoints/predictions thử | Tất cả variant hoàn tất ít nhất 1 fold; không lỗi edge case |
| `v6.3.05` | Chạy full outer 5-fold chỉ cho mô hình mới | `result_pa_v6_3` | Đủ fold theo config; baseline không bị chạy lại |
| `v6.3.06` | Tổng hợp bảng, hình và báo cáo ablation | CSV/PNG/final phase log | Dataset đúng thứ tự MeanIR; hình/bảng khớp; missing minh bạch |

---

## 13. Bộ Kiểm Thử Bắt Buộc

### 13.1. Phân hoạch một tầng

- Không tạo tầng thứ hai dù nhãn DL có thể đạt ngưỡng sau khi thêm $IL_1$.
- $IL_1 \cap DL=\emptyset$ và $IL_1 \cup DL=\mathcal L$.
- Singleton DL không bị thăng hạng.
- OOF probabilities không chứa dự đoán in-sample.
- Phân hoạch tái lập với cùng seed.

### 13.2. CC/PCC

- CC và PCC dùng đúng cùng order trong một fold.
- PCC sampling khớp phân phối đã biết trên dữ liệu tổng hợp nhỏ.
- `predict_proba` của PCC nằm trong $[0,1]$, đúng shape và ổn định với seed.
- Các trường hợp $|DL|=0$ và $|DL|=1$ hoạt động.
- `x` không chứa cột IL; `x_il` dùng OOF IL khi train và predicted IL khi test.

### 13.3. Metric

- Kiểm tra tay complete/ selective Hamming Accuracy, Subset Accuracy, Micro-F1, Macro-F1 và Instance-F1.
- All-abstain, no-abstain, both-empty label set, label không có quyết định và group rỗng.
- `AABS = 1 - Coverage` trong sai số số học cho phép.
- Generalized Loss khớp contract tuyến tính.
- Metric thiếu kế thừa không bị chuyển thành 0.

### 13.4. Result store

- Tạo đúng cây learner/policy/model/dataset.
- `f_measure_bop` không chứa artifact kết quả.
- Resume không chạy lại fold hoàn tất.
- Hash không khớp buộc tạo artifact mới hoặc báo lỗi, không ghi đè âm thầm.
- Folder `all` truy vết được về artifact learner nguồn.

---

## 14. Tiêu Chí Hoàn Thành (Definition of Done)

v6.3 chỉ được coi là hoàn tất khi:

1. Có đối chứng `GSI_v6_3_1L_ResidualBR` để trả lời RQ1 mà không trộn thay đổi DL.
2. Bốn ablation 1.1.1, 1.1.2, 1.2.1, 1.2.2 dùng cùng partition một tầng và cùng chain order trong từng fold.
3. BR, CC, MLC-PA và GSI v6.2 được kế thừa, không chạy lại; mọi metric thiếu là `-`.
4. Kết quả được lưu đúng `result_pa_v6_3/{logistic,svm,mlp,all}/{hamming_bop,f_measure_bop}`.
5. Mỗi model/dataset có checkpoint JSON và prediction artifact, hoặc có provenance kế thừa/missing rõ ràng.
6. Năm bảng CSV bắt buộc tồn tại và qua kiểm tra schema.
7. Đã lưu số lượng $IL$ theo tầng, số lượng $DL$, Macro-F1 và Coverage theo tầng/nhóm.
8. Các hình hoàn chỉnh, chọn lọc và rejection-cost được tạo, dataset sắp theo MeanIR tăng dần.
9. Mọi phase có progress log gồm việc hoàn tất, thông tin cần kiểm chứng, blocker và phase sau.
10. Unit/integration tests pass; full run không có fold thiếu ngoài các mục được đánh dấu `missing` có lý do.
11. Báo cáo kết luận nêu ngoại lệ theo dataset/learner và không tuyên bố quan hệ nhân quả hoặc ý nghĩa thống kê ngoài bằng chứng.

---

## 15. Thông Tin Cần Kiểm Chứng Trước Khi Full Run

| ID | Nội dung cần xác nhận | Giá trị tạm thời trong đặc tả | Ảnh hưởng nếu thay đổi |
|:---|:---|:---|:---|
| `V63-C01` | Danh sách dataset chính thức | **Đã xác nhận:** 10 dataset v6.2 trong `configs/v6_3_run.json` | Đã khóa |
| `V63-C02` | Thứ tự chuỗi DL cho CC/PCC | **Đã xác nhận dùng chung:** khóa `correlation_ascending` trong từng fold | Đã khóa |
| `V63-C03` | Số ancestral samples của PCC | **Đã xác nhận:** 1.000 mẫu cho mỗi mẫu kiểm tra | Đã khóa |
| `V63-C04` | Quy ước Selective Subset Accuracy | Loại mẫu all-abstain khỏi mẫu số | Giá trị bảng/hình selective subset |
| `V63-C05` | Hình mẫu chính thức | Khóa vận hành: `TaiLieuThamKhao/Mau_Bieu_Do.jpg` vì `Hinhmau/` đang rỗng | Có thể đổi ở lần xuất hình sau |
| `V63-C06` | PCC chỉ cấp marginal probability cho `hamming_bop`, chưa dùng GFM | Khóa vận hành theo đặc tả | `f_measure_bop` vẫn để trống |
| `V63-C07` | Ngưỡng $\tau_{F1}$ và metric chọn IL chính xác | Khóa vận hành theo v6.2: `0.75`, Selective-F1 | Đã ghi config |
| `V63-C08` | Folder nguồn của kết quả v6.2 được kế thừa | Audit đã khóa `results_v6_2/v6_2_detailed_folds.csv` | Kiểm tra provenance khi nhập |
| `V63-C09` | Phân hoạch nền cho bốn ablation CC/PCC | Khóa vận hành theo đặc tả: phân hoạch một tầng chung | Thiết kế nhân tố sạch |

Không còn blocker cấu hình cho việc triển khai. `V63-C04` vẫn là quy ước metric do đặc tả đặt ra và phải tiếp tục được ghi trong phase log như một thông tin cần người dùng kiểm chứng sau lần chạy; nó không chặn benchmark theo yêu cầu hiện tại.

---

## 16. Sổ Bằng Chứng và Giả Định (Evidence Ledger)

| ID | Nội dung | Loại | Nguồn | Trạng thái |
|:---|:---|:---|:---|:---|
| `E1` | v6.2 dùng bóc tách đa tầng và BR phụ thuộc theo tương quan sai số | Mô tả dự án đã quan sát | `spec/spec_v6_2.md`, `src/models/gsi_v6_2.py` | Đã hỗ trợ |
| `E2` | v6.3 chỉ chọn $IL$ ở tầng 1 | Yêu cầu người dùng | `ablation.md` | Đã khóa trong thiết kế |
| `E3` | Bốn cấu hình CC/PCC trên $X$ và $X+IL$ | Yêu cầu người dùng | `ablation.md` | Đã khóa trong ma trận |
| `E4` | BR, CC, MLC-PA, GSI v6.2 không chạy lại; thiếu metric ghi `-` | Yêu cầu người dùng | `ablation.md` | Đã khóa trong result store |
| `E5` | PCC phân rã $P(\mathbf y\mid x)$ theo product rule và có thể ancestral sample | Nguồn tài liệu | `TaiLieuThamKhao/PCC.pdf`, Mục 7.1.3 | Đã hỗ trợ |
| `E6` | Đối chứng 1-layer + ResidualBR là cần thiết để cô lập RQ1 | Suy luận thiết kế thực nghiệm | Phát sinh từ RQ1 và nguyên tắc one-factor ablation | Đề xuất bắt buộc |
| `E7` | Quy ước Selective Subset Accuracy | Đề xuất mới | Chưa có contract trong dự án | `[CẦN XÁC NHẬN]` |
| `E8` | Cấu trúc version/result, bảng, hình và phase log | Yêu cầu người dùng + thiết kế kỹ thuật | `comment.md` và convention hiện có | Đã đặc tả; một số tên/giá trị cần xác nhận |
| `E9` | BR và CC không có cơ chế từ chối nhưng vẫn hiện trong selective/rejection dưới dạng complete-metric references | Yêu cầu người dùng | Xác nhận hiệu chỉnh ngày 12/10/2026 | Đã khóa contract: coverage 1, không đổi theo cost |
