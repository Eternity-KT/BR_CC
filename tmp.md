Ran command: `python -c "
import sys
sys.path.insert(0, r'd:\University_Subject\ML Research\BR_CC')
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from src.data.loader import load_dataset
from src.models.gsi_v6_3_3 import GSIMLCPAv6_3_3Classifier

X, Y, _, _ = load_dataset('genbase')
scaler = MaxAbsScaler()
X = scaler.fit_transform(X)

model = GSIMLCPAv6_3_3Classifier(random_state=42)
model.fit(X, Y)

dl_labels = model.dependent_labels_
P = model.predict_proba(X)

print('=== DISTRIBUTION OF P(Y=1) ON GENBASE DL LABELS ===')
for lbl in dl_labels[:5]:
    y_true = Y[:, lbl]
    p_pos = P[:, lbl]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    pos_probs = p_pos[y_true == 1]
    neg_probs = p_pos[y_true == 0]
    
    print(f'Label {lbl}: {n_pos} positive, {n_neg} negative samples')
    print(f'  Positive samples P(Y=1): {np.round(pos_probs, 4)}')
    print(f'  Negative samples max P(Y=1): {np.max(neg_probs):.4f}, 99th percentile: {np.percentile(neg_probs, 99):.4f}, mean: {np.mean(neg_probs):.4f}')
"`
Used tool: manage_task

Trực giác của bạn **hoàn toàn chính xác về mặt bản chất toán học và xếp hạng (Ranking / Anomaly Detection)**, và trên thực tế đây chính là một trong những nền tảng sâu xa để thiết kế các hệ thống nhận diện mẫu hiếm.

Tuy nhiên, giữa **"nguyên lý xếp hạng đúng"** và **"áp dụng được ở pha Inference thực tế"** có những rào cản chí mạng. 

Dưới đây là phân tích chi tiết: tại sao trực giác của bạn đúng, số liệu thực tế trên `genbase` nói lên điều gì, và vì sao trong thực tế không thể áp dụng nó một cách ngây thơ.

---

### 1. Trực giác của bạn đúng ở điểm nào? (Bản chất Ranking & Anomaly)

Trong Machine Learning trên dữ liệu mất cân bằng cực đoan ($\pi \approx 0.001 - 0.005$):
- Do bị áp đảo bởi 99.9% mẫu 0, bộ phân loại (Logistic/GBDT/SVM) sẽ bị thiên vị nặng (calibration bias) và ép xác suất $P(Y=1)$ của toàn bộ mẫu xuống rất thấp (thường không bao giờ chạm tới ngưỡng mặc định $0.5$).
- **Tuy nhiên, thứ tự xếp hạng (Rank-order) giữa các mẫu thường vẫn được bảo tồn rất tốt** (thể hiện qua chỉ số AUC-ROC thường đạt $> 0.95 - 0.99$).
- Nói cách khác: $P(Y=0)$ của các mẫu nhãn 1 thực sự sẽ **nằm ở đuôi thấp nhất** của phân bố $P(Y=0)$ (tương đương $P(Y=1)$ cao nhất).

#### Minh chứng số liệu thực tế từ chính mô hình chạy trên `genbase` DL:
*(Trích xuất phân bố xác suất thực tế trên tập genbase, $N=662$ mẫu)*:
- **Label 25** (chỉ có duy nhất **1 mẫu positive**, 661 mẫu negative):
  - Mẫu positive duy nhất có: $P(Y=1) = \mathbf{0.3235}$ (tức $P(Y=0) \approx 0.6765$).
  - 661 mẫu negative còn lại: $\max P(Y=1) = \mathbf{0.0227}$ (tức $P(Y=0) \ge 0.9773$, trung bình $P(Y=1) = 0.0062$).
  - $\rightarrow$ Đúng như bạn dự đoán: Mẫu positive có $P(Y=0)$ thấp hơn hẳn 661 mẫu còn lại ($0.67$ so với $\ge 0.98$).
- **Label 20** (có **2 mẫu positive**, 660 mẫu negative):
  - 2 mẫu positive có $P(Y=1) = [0.0475, 0.0889]$.
  - 660 mẫu negative có $\max P(Y=1) = \mathbf{0.0265}$.
  - $\rightarrow$ Cả 2 mẫu positive thực sự đứng ở vị trí Top-1 và Top-2 xác suất cao nhất toàn tập dữ liệu!

---

### 2. Vậy tại sao KHÔNG THỂ dùng cách này một cách trực tiếp ở pha Test?

Có **3 cái bẫy kỹ thuật chí mạng** khi triển khai ý tưởng này vào thực tế:

#### Bẫy 1: Ở pha Inference (Test), ta hoàn toàn KHÔNG BIẾT có bao nhiêu mẫu nhãn 1 ($K$ là ẩn số)
- Khi bạn nói: *"thì 4 nhãn 1 còn lại sẽ có $P(Y=0)$ thấp hơn hẳn..."* — bạn đang **giả định ngầm rằng mình đã biết trước tập dữ liệu có đúng 4 mẫu nhãn 1**.
- Nhưng khi đưa mô hình vào triển khai (inference):
  - Nếu test trên 1 mẫu đơn lẻ ($x_{\text{test}}$ online): Bạn không có tập hợp mẫu nào để so sánh "thấp hơn hẳn" là thấp hơn ai.
  - Nếu test trên một batch mới gồm 100 mẫu: Bạn **không biết** batch đó có $0$ mẫu, $2$ mẫu, hay $10$ mẫu nhãn 1.
  - Nếu batch test đó tình cờ **hoàn toàn là nhãn 0** (rất phổ biến với nhãn hiếm), nhưng vì thuật toán cố tìm "top những mẫu có $P(Y=0)$ thấp nhất" để gán nhãn 1, bạn sẽ tạo ra hàng loạt **False Positives** $\rightarrow$ Precision và $F_1$ sụp đổ hoàn toàn về $0$.

#### Bẫy 2: Hiện tượng "Hard Negatives" ở phần đuôi phân bố (Noise Overlap)
Trong 1.000 mẫu âm tính, không phải mẫu nào cũng dễ phân biệt. Chỉ cần có $2 - 3$ mẫu âm tính khó (nhiễu đặc trưng, hoặc nằm mấp mé ranh giới phân tách):
- Mô hình sẽ phân vân với các mẫu này, khiến $P(Y=0)$ của chúng cũng tụt xuống $0.70$ hoặc $0.60$ (như Label 26 ở trên: có mẫu âm tính có $P(Y=1) = 0.275$).
- Vì số lượng mẫu âm tính gấp hàng trăm lần mẫu dương tính, nên **số lượng hard negatives ở phần đuôi rất dễ vượt trội số lượng positive thật**. Khi đó, đuôi của phân bố bị nhiễm bẩn (tail contamination), không thể tách bạch được đâu là positive thật, đâu là hard negative.

#### Bẫy 3: Đánh đổi gắt gao giữa Precision và Recall
Nếu chọn một ngưỡng cố định thật thấp cho $P(Y=0)$ (ví dụ $P(Y=0) \le 0.90$, tức $P(Y=1) \ge 0.10$):
- Bạn sẽ bắt được mẫu dương tính (Recall cao).
- Nhưng bạn sẽ kéo theo một số lượng nhỏ mẫu âm tính bị phân loại nhầm thành dương tính (False Positives). Với $\pi \approx 0.003$, chỉ cần $0.5\%$ mẫu âm tính bị đoán nhầm là dương tính, số False Positives đã lớn hơn cả True Positives $\rightarrow$ **Precision $< 50\%$ ngay lập tức**.

---

### 3. GSI-MLC-PA đã hiện thực hóa trực giác này như thế nào?

Kiến trúc **GSI-MLC-PA** của chúng ta thực chất **đã khai thác trọn vẹn trực giác này**, nhưng giải quyết nó bằng cơ chế toán học chặt chẽ thay vì đoán mò:

```
P(Y=0) cực cao (>= 0.95)       P(Y=0) bị tụt (0.30 - 0.90)       P(Y=0) cực thấp (<= 0.30)
[==========================]   [-------------------------]   [=======================]
  KHẲNG ĐỊNH NHÃN 0 CHẮC CHẮN        VÙNG TỪ CHỐI (REJECTION)        KHẲNG ĐỊNH NHÃN 1
  (99.9% Negative Coverage)      (Chứa 4 mẫu 1 + Hard Negatives)   (Chỉ khi tín hiệu rất rõ)
```

1. **Negative Verification (Regime 1):** Tận dụng việc mô hình nhận diện nhãn 0 cực tốt để thiết lập ngưỡng an toàn cao ($P(Y=0) \ge 0.95$). Mọi mẫu vượt qua ngưỡng này được gán ngay $\hat{y}=0$, giải phóng và làm sạch 99% không gian dữ liệu.
2. **Selective Abstention (Vùng từ chối $[\tau_0, \tau_1]$):** Với các mẫu có $P(Y=0)$ thấp hơn hẳn (rơi vào khoảng $0.30 - 0.90$ nơi mà các nhãn 1 và hard negatives đang giằng co):
   - Thay vì vội vàng đoán liều là nhãn 1 (sẽ dính bẫy False Positive như phân tích ở trên),
   - Mô hình **từ chối đưa ra quyết định tự động**, giữ lại nhúm mẫu nghi ngờ này để thẩm định chuyên sâu (hoặc dùng mô hình chuỗi Classifier Chains / human-in-the-loop).

### Tóm lại:
Ý tưởng **"lần theo đuôi xác suất thấp của nhãn 0 để tìm nhãn 1"** là hoàn toàn có căn cứ khoa học (trong văn hiến gọi là *Anomaly Detection* hoặc *Outlier Retrieval*). Nhưng để biến nó thành một bộ phân loại chính xác, bắt buộc phải có **cơ chế từ chối (Chow's rejection)** để kiểm soát các mẫu âm tính gây nhiễu ở phần đuôi, thay vì gán nhãn 1 một cách cơ học.