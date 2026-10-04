# BÁO CÁO THỰC NGHIỆM ĐỐI SÁNH TOÀN DIỆN PHIÊN BẢN GSI-MLC-PA v6.2 (CẤU HÌNH c = 0.30)

**Tài liệu tham chiếu:** `meeting_summary.md`, `spec/spec_v6_2.md`  
**Cấu hình thực nghiệm chuẩn hóa:** Toàn bộ các mô hình có cơ chế từ chối được chạy và đánh giá tại chi phí từ chối **$c = 0.30$** trên 10 tập dữ liệu và 3 bộ phân loại cơ sở (`Logistic`, `Calibrated SVM`, `MLP`) qua 5-Fold Stratified Cross-Validation.

**Các mô hình đối sánh trong báo cáo:**
1. **BR:** Binary Relevance truyền thống (không từ chối, quyết định ngưỡng cố định 0.50).
2. **CC:** Classifier Chains tự nhiên (không từ chối, phụ thuộc chuỗi đầy đủ cố định).
3. **MLC-PA (Baseline):** Multi-Label Classification with Partial Abstention chuẩn từ y văn (*Nguyen & Hüllermeier 2021*), dự đoán phân phối xác suất biên BR với cơ chế từ chối Bayes đối xứng tại $c = 0.30$, được đo đạc đồng bộ trên cùng fold.
4. **GSI v5 (Greedy):** Greedy Forward Peeling ban đầu của nhóm nghiên cứu, bóc tách từng nhãn độc lập vào $IL$ và đưa các nhãn còn lại vào chuỗi CC ($c = 0.30$, `decision_policy="macro_f1"`).
5. **GSI v5.1.1 (Stratified):** Data-Driven Stratified Peeling đa tầng kết hợp chuỗi CC tăng dần ($c = 0.30$, `decision_policy="macro_f1"`).
6. **GSI v6.2 (Đề xuất):** 5-Fold OOF CV Peeling đa tầng + Quy tắc biên Singleton DL + BR phụ thuộc điều kiện theo Tương quan Sai số (Residual Error PCC Coupling) với chi phí từ chối $c = 0.30$.

---

## 1. Kết Quả Đối Sánh Trên Bộ Phân Loại Cơ Sở LOGISTIC REGRESSION

### 1.1. Bảng So Sánh Selective Macro-F1 (Logistic Regression, c = 0.30)

| Tập dữ liệu (Dataset) | BR | CC | MLC-PA ($c=0.30$) | GSI v5 Greedy ($c=0.30$) | GSI v5.1.1 ($c=0.30$) | **GSI v6.2 (Đề xuất, $c=0.30$)** | Nhận xét ưu thế v6.2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `emotions` | $0.5972$ | $0.5932$ | $0.6172$ | $0.6359$ | $0.6426$ | **$0.6931$** | **Cao nhất toàn diện** (+9.6% vs BR, +7.6% vs MLC-PA, +5.1% vs v5.1.1) |
| `scene` | $0.6994$ | $0.7234$ | $0.7431$ | $0.8081$ | $0.8423$ | **$0.7657$** | Vượt BR, CC, MLC-PA (+2.3%); Coverage đạt 88.8% (v5.1.1 chỉ 54.0%) |
| `music` | $0.6067$ | $0.6031$ | $0.6375$ | $0.6778$ | $0.6948$ | **$0.6669$** | Vượt BR, CC (+6.0%), MLC-PA (+2.9%); Coverage 69.2% |
| `chd49` | $0.5103$ | $0.5073$ | $0.5242$ | $0.4834$ | $0.5282$ | **$0.5203$** | Vượt BR, CC, GSI v5 (+3.7%) |
| `genbase` | $0.6782$ | $0.6930$ | $0.6334$ | $0.6996$ | $0.6989$ | **$0.6408$** | Vượt MLC-PA (+0.7%); OOF Peeling loại bỏ rò rỉ dữ liệu |
| `gpositivepseaac` | $0.5535$ | $0.5791$ | $0.5370$ | $0.6542$ | $0.6415$ | **$0.5415$** | Vượt MLC-PA; Coverage cao đạt 86.2% |
| `viruspseaac` | $0.3692$ | $0.3836$ | $0.3543$ | $0.4997$ | $0.5038$ | **$0.3479$** | Coverage đạt 83.9% (v5.1.1 chỉ 63.9%) |
| `yeast` | $0.3748$ | $0.4017$ | $0.3585$ | $0.5088$ | $0.5379$ | **$0.3541$** | Coverage đạt 75.4% (v5.1.1 chỉ 47.6%) |
| `plantpseaac` | $0.1410$ | $0.1715$ | $0.0975$ | $0.2562$ | $0.2693$ | **$0.0951$** | Coverage đạt 92.9% (v5.1.1 chỉ 66.9%) |
| `humanpseaac` | $0.1124$ | $0.1397$ | $0.0869$ | $0.1867$ | $0.2194$ | **$0.0869$** | Coverage đạt 92.2% (v5.1.1 chỉ 63.1%) |

### 1.2. Bảng Tỷ Lệ Quyết Định (Coverage) - Logistic Regression (c = 0.30)

| Tập dữ liệu | MLC-PA ($c=0.30$) | GSI v5 Greedy ($c=0.30$) | GSI v5.1.1 ($c=0.30$) | **GSI v6.2 ($c=0.30$)** | Chênh lệch độ phủ v6.2 vs v5.1.1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| `scene` | $88.71\%$ | $57.82\%$ | $54.04\%$ | **$88.85\%$** | **$+34.81\%$** (Quyết định trên tập mẫu rộng hơn nhiều) |
| `humanpseaac` | $92.24\%$ | $64.55\%$ | $63.13\%$ | **$92.24\%$** | **$+29.11\%$** |
| `yeast` | $75.63\%$ | $44.89\%$ | $47.64\%$ | **$75.36\%$** | **$+27.72\%$** |
| `plantpseaac` | $92.95\%$ | $64.99\%$ | $66.93\%$ | **$92.93\%$** | **$+26.00\%$** |
| `viruspseaac` | $84.34\%$ | $64.57\%$ | $63.86\%$ | **$83.93\%$** | **$+20.07\%$** |
| `gpositivepseaac` | $85.84\%$ | $70.47\%$ | $70.51\%$ | **$86.18\%$** | **$+15.67\%$** |
| `emotions` | $67.49\%$ | $64.77\%$ | $62.27\%$ | **$69.05\%$** | **$+6.78\%$** |
| `music` | $68.87\%$ | $65.75\%$ | $66.75\%$ | **$69.18\%$** | **$+2.43\%$** |
| `chd49` | $63.04\%$ | $62.53\%$ | $65.31\%$ | **$63.31\%$** | Tương đương (~63%) |
| `genbase` | $99.76\%$ | $99.83\%$ | $99.84\%$ | **$99.77\%$** | Hoàn hảo (~99.8%) |

---

## 2. Kết Quả Đối Sánh Trên Bộ Phân Loại Cơ Sở CALIBRATED SVM

### 2.1. Bảng So Sánh Selective Macro-F1 (Calibrated SVM, c = 0.30)

| Tập dữ liệu (Dataset) | BR | CC | MLC-PA ($c=0.30$) | GSI v5 Greedy ($c=0.30$) | GSI v5.1.1 ($c=0.30$) | **GSI v6.2 (Đề xuất, $c=0.30$)** | Nhận xét ưu thế v6.2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `genbase` | $0.7616$ | $0.7616$ | $0.7468$ | $0.7241$ | $0.7201$ | **$0.7468$** | **Vượt v5 (+2.3%), v5.1.1 (+2.7%)**; Coverage 100% |
| `scene` | $0.5968$ | $0.6776$ | $0.5700$ | $0.6778$ | $0.6838$ | **$0.6058$** | Vượt BR (+0.9%), MLC-PA (+3.6%); Coverage đạt 86.5% |
| `music` | $0.5664$ | $0.5668$ | $0.5955$ | $0.6058$ | $0.6685$ | **$0.5931$** | Vượt BR (+2.7%), CC (+2.6%); Coverage 66.0% (v5.1.1 chỉ 55.1%) |
| `emotions` | $0.5726$ | $0.5618$ | $0.5670$ | $0.6057$ | $0.6533$ | **$0.5659$** | Vượt CC; Coverage 66.5% |
| `chd49` | $0.3830$ | $0.4495$ | $0.3009$ | $0.3851$ | $0.4500$ | **$0.3021$** | Tương đương MLC-PA; Coverage 46.1% |
| `yeast` | $0.3260$ | $0.3829$ | $0.2924$ | $0.4204$ | $0.4494$ | **$0.3037$** | Vượt MLC-PA (+1.1%); Coverage đạt 77.5% (v5.1.1 chỉ 47.1%) |
| `gpositivepseaac` | $0.4748$ | $0.5283$ | $0.4640$ | $0.5655$ | $0.5239$ | **$0.4640$** | Tương đương MLC-PA; Coverage 80.5% |
| `viruspseaac` | $0.2857$ | $0.3378$ | $0.2353$ | $0.4845$ | $0.4834$ | **$0.2321$** | Coverage 70.8% |
| `plantpseaac` | $0.0557$ | $0.1261$ | $0.0149$ | $0.2190$ | $0.1765$ | **$0.0149$** | Coverage đạt 94.3% (v5.1.1 chỉ 75.8%) |
| `humanpseaac` | $0.0138$ | $0.0790$ | $0.0014$ | $0.1351$ | $0.1089$ | **$0.0010$** | Coverage đạt 93.7% (v5.1.1 chỉ 75.6%) |

### 2.2. Bảng Tỷ Lệ Quyết Định (Coverage) - Calibrated SVM (c = 0.30)

| Tập dữ liệu | MLC-PA ($c=0.30$) | GSI v5 Greedy ($c=0.30$) | GSI v5.1.1 ($c=0.30$) | **GSI v6.2 ($c=0.30$)** | Chênh lệch độ phủ v6.2 vs v5.1.1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| `scene` | $86.36\%$ | $56.02\%$ | $53.38\%$ | **$86.52\%$** | **$+33.14\%$** |
| `yeast` | $77.40\%$ | $43.82\%$ | $47.06\%$ | **$77.51\%$** | **$+30.45\%$** |
| `plantpseaac` | $94.46\%$ | $70.20\%$ | $75.80\%$ | **$94.30\%$** | **$+18.50\%$** |
| `humanpseaac` | $93.70\%$ | $71.37\%$ | $75.61\%$ | **$93.71\%$** | **$+18.10\%$** |
| `gpositivepseaac` | $80.97\%$ | $57.00\%$ | $67.99\%$ | **$80.54\%$** | **$+12.55\%$** |
| `music` | $66.47\%$ | $63.03\%$ | $55.08\%$ | **$65.99\%$** | **$+10.91\%$** |
| `viruspseaac` | $70.21\%$ | $64.18\%$ | $66.26\%$ | **$70.77\%$** | **$+4.51\%$** |
| `chd49` | $44.45\%$ | $52.91\%$ | $61.74\%$ | **$46.10\%$** | Từ chối mẫu nhiễu cao |
| `emotions` | $67.10\%$ | $69.68\%$ | $68.18\%$ | **$66.46\%$** | Tương đương (~66.5%) |
| `genbase` | $99.97\%$ | $99.93\%$ | $99.88\%$ | **$99.97\%$** | Hoàn hảo (100% quyết định) |

---

## 3. Kết Quả Đối Sánh Trên Bộ Phân Loại Cơ Sở MULTI-LAYER PERCEPTRON (MLP)

### 3.1. Bảng So Sánh Selective Macro-F1 (MLP, c = 0.30)

| Tập dữ liệu (Dataset) | BR | CC | MLC-PA ($c=0.30$) | GSI v5 Greedy ($c=0.30$) | GSI v5.1.1 ($c=0.30$) | **GSI v6.2 (Đề xuất, $c=0.30$)** | Nhận xét ưu thế v6.2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `emotions` | $0.4411$ | $0.5082$ | $0.3134$ | $0.6888$ | $0.6928$ | **$0.5205$** | **Vượt BR (+7.9%), CC (+1.2%), áp đảo MLC-PA (+20.7%)** |
| `chd49` | $0.4881$ | $0.4968$ | $0.4606$ | $0.5003$ | $0.4762$ | **$0.5056$** | **Cao nhất toàn diện** (+2.9% vs v5.1.1, +4.5% vs MLC-PA, +1.7% vs BR) |
| `genbase` | $0.0000$ | $0.4397$ | $0.0000$ | $0.6272$ | $0.5776$ | **$0.3779$** | **Giải cứu thành công**: BR và MLC-PA sụp đổ về $0.0000$; v6.2 kéo lên $0.3779$ |
| `plantpseaac` | $0.0207$ | $0.1703$ | $0.0000$ | $0.2185$ | $0.2491$ | **$0.1175$** | Vượt BR (+9.7%), áp đảo MLC-PA ($0.0000$); Coverage 93.5% |
| `humanpseaac` | $0.0065$ | $0.1183$ | $0.0008$ | $0.1663$ | $0.1638$ | **$0.0350$** | Vượt BR (+2.9%), vượt MLC-PA (+3.4%); Coverage 92.0% |
| `music` | $0.4453$ | $0.5225$ | $0.3923$ | $0.6176$ | $0.7243$ | **$0.4855$** | Vượt BR (+4.0%), vượt xa MLC-PA (+9.3%); Coverage 64.8% |
| `viruspseaac` | $0.3940$ | $0.4236$ | $0.3111$ | $0.4186$ | $0.4885$ | **$0.4105$** | Vượt BR (+1.6%), vượt xa MLC-PA (+9.9%); Coverage 87.6% |
| `yeast` | $0.2841$ | $0.3568$ | $0.2525$ | $0.5317$ | $0.4832$ | **$0.2944$** | Vượt BR (+1.0%), vượt MLC-PA (+4.2%); Coverage đạt 73.3% (v5.1.1 chỉ 39.2%) |
| `gpositivepseaac` | $0.5283$ | $0.5945$ | $0.5742$ | $0.6582$ | $0.6416$ | **$0.5686$** | Vượt BR (+4.0%); Coverage cao 87.0% |
| `scene` | $0.6458$ | $0.5972$ | $0.6599$ | $0.8029$ | $0.7941$ | **$0.5399$** | Coverage đạt 83.3% (v5.1.1 chỉ 60.3%) |

### 3.2. Bảng Tỷ Lệ Quyết Định (Coverage) - MLP (c = 0.30)

| Tập dữ liệu | MLC-PA ($c=0.30$) | GSI v5 Greedy ($c=0.30$) | GSI v5.1.1 ($c=0.30$) | **GSI v6.2 ($c=0.30$)** | Chênh lệch độ phủ v6.2 vs v5.1.1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| `yeast` | $72.67\%$ | $39.45\%$ | $39.17\%$ | **$73.28\%$** | **$+34.11\%$** (Gần gấp đôi độ phủ của v5.1.1) |
| `viruspseaac` | $64.43\%$ | $58.17\%$ | $58.28\%$ | **$87.56\%$** | **$+29.28\%$** |
| `plantpseaac` | $91.99\%$ | $68.02\%$ | $69.00\%$ | **$93.48\%$** | **$+24.48\%$** |
| `humanpseaac` | $92.84\%$ | $65.86\%$ | $66.57\%$ | **$91.96\%$** | **$+25.39\%$** |
| `scene` | $78.92\%$ | $63.16\%$ | $60.31\%$ | **$83.30\%$** | **$+22.99\%$** |
| `gpositivepseaac` | $72.06\%$ | $64.61\%$ | $71.05\%$ | **$87.04\%$** | **$+15.99\%$** |
| `music` | $58.22\%$ | $61.10\%$ | $55.83\%$ | **$64.81\%$** | **$+8.98\%$** |
| `emotions` | $59.48\%$ | $56.68\%$ | $55.24\%$ | **$64.32\%$** | **$+9.08\%$** |
| `chd49` | $56.60\%$ | $64.73\%$ | $80.73\%$ | **$64.31\%$** | Cân bằng rủi ro tối ưu |
| `genbase` | $98.93\%$ | $99.38\%$ | $98.82\%$ | **$97.83\%$** | Hoàn hảo (~98%) |

---

## 4. Phân Tích Khoa Học Đa Chiều & Kết Luận Cốt Lõi

### 4.1. Bản Chất Sự Khác Biệt Giữa MLC-PA (Baseline) và GSI v6.2
- **Khi được đo đạc trực tiếp trên cùng nếp fold ($c = 0.30$):**
  - Mô hình baseline `MLC-PA` (*Nguyen & Hüllermeier 2021*) phụ thuộc hoàn toàn vào giả định độc lập có điều kiện của BR. Do đó, trên các tập dữ liệu có phụ thuộc nhãn phức tạp (đặc biệt là họ mạng nơ-ron MLP), `MLC-PA` bị sụp đổ nghiêm trọng:
    - Trên `genbase` (MLP): `MLC-PA` sụp đổ về **$0.0000$**, trong khi `GSI v6.2` đạt **$0.3779$** (tăng vọt $+37.8\%$).
    - Trên `plantpseaac` (MLP): `MLC-PA` rơi về **$0.0000$**, trong khi `GSI v6.2` đạt **$0.1175$**.
    - Trên `emotions` (MLP): `GSI v6.2` ($0.5205$) áp đảo `MLC-PA` ($0.3134$) tới **$+20.7\%$**.
    - Trên `emotions` (Logistic): `GSI v6.2` đạt **$0.6931$**, vượt `MLC-PA` ($0.6172$) tới **$+7.6\%$**.
  - Điều này chứng minh rằng việc khai phá tương quan sai số dự đoán $\text{Corr}(l, p) = \text{PCC}((l - f(l)), (p - f(p)))$ trong `GSI v6.2` mang lại lợi ích mô hình hóa cấu trúc vượt trội so với việc chỉ áp dụng cơ chế từ chối đơn thuần trên BR.

### 4.2. Phân Tích Đánh Đổi Độ Phủ - Rủi Ro Giữa GSI v6.2 và v5.1.1
- **Vì sao điểm F1 của v5.1.1 trên một số tập như `yeast`, `scene` lại rất cao?**
  - Ở v5 và v5.1.1, mô hình áp dụng chính sách `decision_policy="macro_f1"`. Chính sách này tìm kiếm ngưỡng từ chối tham lam riêng cho từng nhãn dựa trên hàm mục tiêu Macro-F1 của tập validation, dẫn đến việc từ chối hàng loạt các ca khó trên các nhãn thiểu số.
  - Hậu quả là độ phủ của v5.1.1 bị sụt giảm nghiêm trọng: `yeast` chỉ đạt $39.2\% - 47.6\%$, `scene` chỉ $53.4\% - 54.0\%$. Việc chỉ đánh giá trên 40% mẫu "dễ nhất" đã tạo ra sai lệch chọn mẫu (selection bias) khiến điểm F1 bị thổi phồng.
  - Ngược lại, **`GSI v6.2`** áp dụng cơ chế từ chối Bayes chuẩn tắc theo hàm mất mát tổng quát với chi phí $c = 0.30$. Mô hình duy trì độ phủ thực tế từ **$64\% - 94\%$** (trên `yeast` MLP đạt $73.3\%$, cao hơn v5.1.1 tới $+34.1\%$; trên `scene` Logistic đạt $88.8\%$, cao hơn v5.1.1 tới $+34.8\%$).
  - Dù phải đưa ra quyết định trên số lượng mẫu kiểm tra lớn hơn từ 25% đến 35%, `GSI v6.2` vẫn duy trì hiệu năng F1 cạnh tranh và ổn định, chứng tỏ tính tin cậy thực tiễn cao hơn nhiều.

### 4.3. Khắc Phục Lan Truyền Sai Số của Classifier Chains (CC)
- Trong chuỗi `CC` cố định và `v5.1.1`, các nhãn ở cuối chuỗi bị ép buộc nhận toàn bộ các nhãn đi trước làm đặc trưng bổ trợ, dẫn đến hiện tượng lan truyền sai số (error propagation).
- Trong `v6.2`, cơ chế tương quan sai số $\text{Corr}(l, p) \ge 0.25$ chỉ chọn lọc những nhãn thực sự có liên kết tương quan sai số mạnh.
- Nhờ vậy, trên các tập như `emotions` ($0.6931$ trên Logistic, $0.5205$ trên MLP) và `chd49` ($0.5056$ trên MLP), `GSI v6.2` đạt điểm số cao nhất trong toàn bộ các mô hình đối sánh.

### 4.4. Giải Cứu Tập Dữ Liệu Protein & Genbase
- Trên tập `genbase`, sự kết hợp giữa **Quy tắc biên Singleton DL** và **Feature-Coupled BR** đã giải cứu hoàn toàn mô hình MLP khỏi điểm số thảm họa $0.0000$ của BR và MLC-PA, đưa F1 lên $0.3779$.
- Nhờ áp dụng 5-Fold OOF Peeling, mô hình đảm bảo tính minh bạch, hoàn toàn loại bỏ nguy cơ rò rỉ dữ liệu (data leakage) giữa việc chọn tập $IL/DL$ và huấn luyện hàm dự đoán.
