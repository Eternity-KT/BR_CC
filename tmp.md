Searched for "DL"
Viewed gsi_mlc_pa.py:265-340
Viewed gsi_mlc_pa.py:150-265
Viewed classifier_chain.py:1-100
Viewed classifier_chain.py:101-160
Viewed classifier_chain.py:160-200
Viewed gsi_mlc_pa.py:520-650
Viewed gsi_mlc_pa.py:650-790
Viewed gsi_mlc_pa.py:340-450

Trong phiên bản **v3** (thực thi tại [src/models/gsi_mlc_pa.py](file:///d:/University_Subject/ML%20Research/BR_CC/src/models/gsi_mlc_pa.py)), cơ chế tính xác suất cho các nhãn phụ thuộc (**DL - Dependent Labels**) được thực hiện tuần tự qua chuỗi phân loại (Classifier Chain - CC) kết hợp với kỹ thuật **Biên duyên hóa xấp xỉ (Marginalization)**.

Dưới đây là chi tiết toàn bộ quy trình toán học và cài đặt code:

---

### 1. Thứ tự nhãn trong chuỗi (`self.order_`)
Trước khi tính xác suất, tập nhãn được sắp xếp thông qua hàm `_correlation_order` ([src/models/gsi_mlc_pa.py#L359-L393](file:///d:/University_Subject/ML%20Research/BR_CC/src/models/gsi_mlc_pa.py#L359-L393)):
1. **Các nhãn IL (Independent Labels)**: Luôn được xếp ở **đầu chuỗi** làm các "gốc" (roots).
2. **Các nhãn DL (Dependent Labels)**: Được nối tiếp vào chuỗi theo thứ tự có **độ tương quan tuyệt đối (Pearson/Phi correlation) lớn nhất** với các nhãn đã được xếp trước đó trong chuỗi.

Do đó, với mỗi nhãn $j \in \text{DL}$ ở vị trí `position` trong chuỗi, tập nhãn tiền nhiệm của nó là:
$$\text{pred}(j) = \{\pi_1, \pi_2, \dots, \pi_{\text{position}-1}\}$$
*(Tập tiền nhiệm này có thể gồm các nhãn IL hoặc các nhãn DL đứng trước $j$)*.

---

### 2. Dữ liệu huấn luyện của bộ phân loại CC cho từng nhãn DL
Trong quá trình `fit` của `ClassifierChainClassifier` ([src/models/classifier_chain.py#L101-L117](file:///d:/University_Subject/ML%20Research/BR_CC/src/models/classifier_chain.py#L101-L117)):
- Mỗi bộ phân loại nhị phân $h_j$ tương ứng với nhãn $j$ được huấn luyện với vector đặc trưng mở rộng bằng **nhãn thực tế (Ground-Truth Labels)**:
  $$X_{\text{train}}^{(j)} = [X, \quad Y_{\text{true}, \text{pred}(j)}]$$
- Mục tiêu: $h_j$ học phân phối điều kiện $P(Y_j = 1 \mid X, Y_{\text{pred}(j)})$.

---

### 3. Cơ chế suy diễn xác suất khi Inference (`_configured_probabilities`)
Tại thời điểm suy diễn ([src/models/gsi_mlc_pa.py#L266-L341](file:///d:/University_Subject/ML%20Research/BR_CC/src/models/gsi_mlc_pa.py#L266-L341)):
- **Đối với nhãn $i \in \text{IL}$**: Xác suất được gán trực tiếp từ mô hình **Binary Relevance (BR)** độc lập:
  $$\hat{P}(Y_i = 1 \mid X) = P_{\text{BR}}(Y_i = 1 \mid X)$$
- **Đối với nhãn $j \in \text{DL}$**: Cần tính xác suất biên duyên $\hat{P}(Y_j = 1 \mid X)$ dựa trên các nhãn tiền nhiệm $\text{pred}(j)$. Xác suất của các nhãn tiền nhiệm này đã được tính toán ở các bước trước:
  $$p_{\text{pred}} = \hat{P}(Y_{\text{pred}(j)} = 1 \mid X)$$

Hệ thống xử lý nhãn DL theo **3 trường hợp** dựa vào số lượng tiền nhiệm $m = |\text{pred}(j)|$:

#### Trường hợp 1: $m = 0$ (Root DL - nếu DL đứng ở vị trí đầu chuỗi)
Bộ phân loại $h_j$ không phụ thuộc nhãn nào:
$$\hat{P}(Y_j = 1 \mid X) = h_j(X)$$

#### Trường hợp 2: $m = 1$ (Chỉ có duy nhất 1 nhãn tiền nhiệm)
Mô hình thực hiện **Biên duyên hóa chính xác (Exact Marginalization)** theo công thức xác suất toàn phần:
$$\hat{P}(Y_j = 1 \mid X) = \sum_{y_{\text{parent}} \in \{0, 1\}} P(Y_j = 1 \mid X, Y_{\text{parent}} = y_{\text{parent}}) \cdot \hat{P}(Y_{\text{parent}} = y_{\text{parent}} \mid X)$$
$$\hat{P}(Y_j = 1 \mid X) = (1 - p_{\text{parent}}) \cdot h_j([X, 0]) + p_{\text{parent}} \cdot h_j([X, 1])$$

*Code thực thi ([src/models/gsi_mlc_pa.py#L318-L331](file:///d:/University_Subject/ML%20Research/BR_CC/src/models/gsi_mlc_pa.py#L318-L331)):*
```python
if len(predecessors) == 1:
    zeros = np.zeros((X.shape[0], 1), dtype=np.float32)
    ones = np.ones((X.shape[0], 1), dtype=np.float32)
    probability_given_zero = _positive_probability(classifier, np.hstack((X, zeros)))
    probability_given_one = _positive_probability(classifier, np.hstack((X, ones)))
    parent_probability = predecessor_probabilities[:, 0]
    probabilities[:, label_index] = (
        (1.0 - parent_probability) * probability_given_zero
        + parent_probability * probability_given_one
    )
```

#### Trường hợp 3: $m \ge 2$ (Có từ 2 nhãn tiền nhiệm trở lên)
Nếu tính chính xác, số lượng trường hợp cần duyệt là $2^m$ tổ hợp nhị phân (với $m=10$, cần $1024$ lần forward pass cho mỗi mẫu — không khả thi).

Do đó, v3 sử dụng **Xấp xỉ trường trung bình (Mean-Field Approximation / Soft-label feature input)**:
- Thay vì lấy tổng qua tất cả các cấu hình nhị phân rời rạc $\{0, 1\}^m$, các đặc trưng nhãn tiền nhiệm được thay thế bằng chính **giá trị kỳ vọng (xác suất mềm)** của chúng:
  $$\tilde{X}_{\text{mean\_field}} = \left[X, \quad \hat{P}(Y_{\pi_1}=1 \mid X), \quad \hat{P}(Y_{\pi_2}=1 \mid X), \quad \dots, \quad \hat{P}(Y_{\pi_{m}}=1 \mid X)\right]$$
- Sau đó đưa trực tiếp vector này vào bộ phân loại $h_j$:
  $$\hat{P}(Y_j = 1 \mid X) \approx h_j(\tilde{X}_{\text{mean\_field}})$$

*Code thực thi ([src/models/gsi_mlc_pa.py#L333-L340](file:///d:/University_Subject/ML%20Research/BR_CC/src/models/gsi_mlc_pa.py#L333-L340)):*
```python
else:
    # Mean-field marginalization: q(Y_parents) is represented by
    # its factorized means, avoiding an exponential 2**m sum.
    mean_field_features = np.hstack(
        (X, predecessor_probabilities.astype(np.float32))
    )
    probabilities[:, label_index] = _positive_probability(
        classifier, mean_field_features
    )
```

---

### 4. Đánh giá kỹ thuật liên quan đến ghi chú của bạn
Trong [meeting_summary.md](file:///d:/University_Subject/ML%20Research/BR_CC/meeting_summary.md), bạn có ghi chú:
> *"Nghiên cứu lại phần sắp xếp nhãn IL và DL, cần rõ ràng phần này để cải thiện performance"*
> *"Tách các nhãn IL ra riêng để đánh giá bằng BR, các nhãn thuộc DL sử dụng CC"*

Cơ chế hiện tại của v3 đang có 2 vấn đề lớn có thể ảnh hưởng đến performance của DL:
1. **Covariate Shift (Lệch phân phối đầu vào giữa Train và Test)**:
   - Khi huấn luyện CC, $h_j$ chỉ nhìn thấy các nhãn nhị phân rời rạc $Y \in \{0, 1\}$.
   - Khi suy diễn (inference), $h_j$ lại nhận các xác suất thực liên tục $p \in [0.0, 1.0]$ (soft probabilities). Đối với các mô hình tuyến tính hoặc cây quyết định, điều này dễ làm lệch điểm số đầu ra (logits) dẫn đến xác suất ước lượng kém chuẩn xác (miscalibration).
2. **Tiền nhiệm của DL bao gồm cả IL và DL khác**:
   - Hiện tại, chuỗi CC đang nối dài: $IL_1 \to IL_2 \to \dots \to DL_1 \to DL_2 \dots$
   - Điều này đồng nghĩa với việc các nhãn $DL$ đang phải phụ thuộc vào một chuỗi rất dài gồm toàn bộ các nhãn $IL$ phía trước. Nếu muốn "tách các nhãn IL ra riêng, các nhãn DL sử dụng CC", một hướng cải tiến tiềm năng là: **Mô hình CC chỉ huấn luyện và liên kết riêng giữa các nhãn thuộc tập DL**, hoặc sử dụng cấu trúc cây phụ thuộc cục bộ (Parent Tree) thay vì một chuỗi dài nối tiếp qua tất cả các nhãn IL.