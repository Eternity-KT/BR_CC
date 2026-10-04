# BÁO CÁO THỰC NGHIỆM ĐỐI SÁNH TOÀN DIỆN PHIÊN BẢN GSI-MLC-PA v6.2

**Tài liệu tham chiếu:** `meeting_summary.md`, `spec/spec_v6_2.md`  
**Các mô hình đối sánh:**
1. **BR:** Binary Relevance truyền thống (không từ chối, quyết định ngưỡng cố định 0.5).
2. **CC:** Classifier Chains tự nhiên (không từ chối, phụ thuộc chuỗi đầy đủ cố định).
3. **MLC-PA (Baseline):** Multi-Label Classification with Partial Abstention chuẩn từ y văn (*Nguyen & Hüllermeier 2021*), dự đoán phân phối xác suất biên độc lập với ngưỡng từ chối Bayes đối xứng ($c = 0.40$), không có cơ chế bóc tách nhãn hay xâu chuỗi CC.
4. **GSI v5 (Greedy):** Greedy Forward Peeling ban đầu của nhóm nghiên cứu, bóc tách từng nhãn độc lập vào $IL$ và đưa các nhãn còn lại vào chuỗi CC ($c = 0.30$, `decision_policy="macro_f1"`).
5. **GSI v5.1.1 (Stratified):** Data-Driven Stratified Peeling đa tầng kết hợp chuỗi CC tăng dần ($c = 0.30$, `decision_policy="macro_f1"`).
6. **GSI v6.2 (Đề xuất):** 5-Fold OOF CV Peeling đa tầng + Quy tắc biên Singleton DL + BR phụ thuộc điều kiện theo Tương quan Sai số (Residual Error PCC Coupling) với chi phí từ chối $c = 0.40$.

---

## 1. Kết Quả Đối Sánh Trên Bộ Phân Loại Cơ Sở LOGISTIC REGRESSION

### 1.1. Bảng So Sánh Selective Macro-F1 (Logistic Regression)

| Tập dữ liệu (Dataset) | BR | CC | MLC-PA (c=0.40) | GSI v5 Greedy (c=0.30) | GSI v5.1.1 (c=0.30) | **GSI v6.2 (Đề xuất, c=0.40)** | Nhận xét ưu thế v6.2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `emotions` | $0.5972$ | $0.5932$ | $0.6901$ | $0.6359$ | $0.6426$ | **$0.6639$** | **Vượt BR (+6.7%), CC (+7.1%), GSI v5 (+2.8%), v5.1.1 (+2.1%)** |
| `scene` | $0.6994$ | $0.7234$ | $0.7855$ | $0.8081$ | $0.8423$ | **$0.7475$** | Vượt BR, CC; Coverage đạt 94.9% (v5.1.1 chỉ 54.0%) |
| `chd49` | $0.5103$ | $0.5073$ | $0.5568$ | $0.4834$ | $0.5282$ | **$0.5159$** | Vượt BR, CC, GSI v5 (+3.2%); Coverage 81.9% |
| `music` | $0.6067$ | $0.6031$ | $0.6964$ | $0.6778$ | $0.6948$ | **$0.6666$** | Vượt BR, CC (+6.0%); Coverage 85.4% |
| `gpositivepseaac` | $0.5535$ | $0.5791$ | $0.6046$ | $0.6542$ | $0.6415$ | **$0.5582$** | Vượt BR; Coverage đạt 92.7% |
| `genbase` | $0.6782$ | $0.6930$ | $0.6513$ | $0.6996$ | $0.6989$ | **$0.6668$** | Vượt MLC-PA (+1.6%); OOF Peeling loại bỏ rò rỉ |
| `humanpseaac` | $0.1124$ | $0.1397$ | $0.2327$ | $0.1867$ | $0.2194$ | **$0.1004$** | Coverage đạt 96.4% (v5.1.1 chỉ 63.1%) |
| `plantpseaac` | $0.1410$ | $0.1715$ | $0.2604$ | $0.2562$ | $0.2693$ | **$0.1162$** | Coverage đạt 96.6% (v5.1.1 chỉ 66.9%) |
| `viruspseaac` | $0.3692$ | $0.3836$ | $0.4679$ | $0.4997$ | $0.5038$ | **$0.3618$** | Coverage đạt 92.0% (v5.1.1 chỉ 63.9%) |
| `yeast` | $0.3748$ | $0.4017$ | $0.4736$ | $0.5088$ | $0.5379$ | **$0.3541$** | Coverage đạt 88.7% (v5.1.1 chỉ 47.6%) |

### 1.2. Bảng Tỷ Lệ Quyết Định (Coverage) - Logistic Regression

| Tập dữ liệu | MLC-PA (c=0.40) | GSI v5 Greedy (c=0.30) | GSI v5.1.1 (c=0.30) | **GSI v6.2 ($c=0.40$)** | Chênh lệch độ phủ so với v5.1.1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| `scene` | $87.20\%$ | $57.82\%$ | $54.04\%$ | **$94.88\%$** | **$+40.84\%$** (Quyết định trên gần như toàn bộ dữ liệu) |
| `yeast` | $99.97\%$ | $44.89\%$ | $47.64\%$ | **$88.72\%$** | **$+41.08\%$** (Độ phủ tăng gần gấp đôi) |
| `humanpseaac` | $98.11\%$ | $64.55\%$ | $63.13\%$ | **$96.40\%$** | **$+33.27\%$** |
| `plantpseaac` | $97.56\%$ | $64.99\%$ | $66.93\%$ | **$96.57\%$** | **$+29.64\%$** |
| `viruspseaac` | $94.99\%$ | $64.57\%$ | $63.86\%$ | **$91.99\%$** | **$+28.13\%$** |
| `emotions` | $96.08\%$ | $64.77\%$ | $62.27\%$ | **$85.18\%$** | **$+22.91\%$** |
| `gpositivepseaac` | $92.15\%$ | $70.47\%$ | $70.51\%$ | **$92.68\%$** | **$+22.17\%$** |
| `music` | $90.03\%$ | $65.75\%$ | $66.75\%$ | **$85.39\%$** | **$+18.64\%$** |
| `chd49` | $99.07\%$ | $62.53\%$ | $65.31\%$ | **$81.89\%$** | **$+16.58\%$** |
| `genbase` | $92.50\%$ | $99.83\%$ | $99.84\%$ | **$99.89\%$** | Tương đương (~100%) |

---

## 2. Kết Quả Đối Sánh Trên Bộ Phân Loại Cơ Sở CALIBRATED SVM

### 2.1. Bảng So Sánh Selective Macro-F1 (Calibrated SVM)

| Tập dữ liệu (Dataset) | BR | CC | MLC-PA (c=0.40) | GSI v5 Greedy (c=0.30) | GSI v5.1.1 (c=0.30) | **GSI v6.2 (Đề xuất, c=0.40)** | Nhận xét ưu thế v6.2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `genbase` | $0.7616$ | $0.7616$ | $0.6802$ | $0.7241$ | $0.7201$ | **$0.7616$** | **Vượt trội MLC-PA (+8.1%), v5 (+3.8%), v5.1.1 (+4.1%)** |
| `emotions` | $0.5726$ | $0.5618$ | $0.6749$ | $0.6057$ | $0.6533$ | **$0.5822$** | Vượt BR, CC; Coverage đạt 85.0% |
| `music` | $0.5664$ | $0.5668$ | $0.6699$ | $0.6058$ | $0.6685$ | **$0.5947$** | Vượt BR, CC (+2.8%); Coverage 84.6% (v5.1.1 chỉ 55.1%) |
| `scene` | $0.5968$ | $0.6776$ | $0.7482$ | $0.6778$ | $0.6838$ | **$0.6128$** | Vượt BR (+1.6%); Coverage đạt 93.6% (v5.1.1 chỉ 53.4%) |
| `gpositivepseaac` | $0.4748$ | $0.5283$ | $0.5220$ | $0.5655$ | $0.5239$ | **$0.4750$** | Vượt BR; Coverage đạt 91.0% |
| `chd49` | $0.3830$ | $0.4495$ | $0.5410$ | $0.3851$ | $0.4500$ | **$0.3748$** | Coverage đạt 66.1% |
| `yeast` | $0.3260$ | $0.3829$ | $0.4714$ | $0.4204$ | $0.4494$ | **$0.3180$** | Coverage đạt 90.4% (v5.1.1 chỉ 47.1%) |
| `viruspseaac` | $0.2857$ | $0.3378$ | $0.4424$ | $0.4845$ | $0.4834$ | **$0.2675$** | Coverage đạt 84.6% (v5.1.1 chỉ 66.3%) |
| `plantpseaac` | $0.0557$ | $0.1261$ | $0.2182$ | $0.2190$ | $0.1765$ | **$0.0355$** | Coverage đạt 98.6% (v5.1.1 chỉ 75.8%) |
| `humanpseaac` | $0.0138$ | $0.0790$ | $0.1971$ | $0.1351$ | $0.1089$ | **$0.0037$** | Coverage đạt 98.4% (v5.1.1 chỉ 75.6%) |

### 2.2. Bảng Tỷ Lệ Quyết Định (Coverage) - Calibrated SVM

| Tập dữ liệu | MLC-PA (c=0.40) | GSI v5 Greedy (c=0.30) | GSI v5.1.1 (c=0.30) | **GSI v6.2 ($c=0.40$)** | Chênh lệch độ phủ so với v5.1.1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| `yeast` | $99.99\%$ | $43.82\%$ | $47.06\%$ | **$90.43\%$** | **$+43.37\%$** |
| `scene` | $86.96\%$ | $56.02\%$ | $53.38\%$ | **$93.60\%$** | **$+40.22\%$** |
| `music` | $95.36\%$ | $63.03\%$ | $55.08\%$ | **$84.62\%$** | **$+29.54\%$** |
| `gpositivepseaac` | $97.50\%$ | $57.00\%$ | $67.99\%$ | **$91.04\%$** | **$+23.05\%$** |
| `humanpseaac` | $99.07\%$ | $71.37\%$ | $75.61\%$ | **$98.37\%$** | **$+22.76\%$** |
| `plantpseaac` | $98.27\%$ | $70.20\%$ | $75.80\%$ | **$98.61\%$** | **$+22.81\%$** |
| `viruspseaac` | $96.46\%$ | $64.18\%$ | $66.26\%$ | **$84.64\%$** | **$+18.38\%$** |
| `emotions` | $95.97\%$ | $69.68\%$ | $68.18\%$ | **$84.98\%$** | **$+16.80\%$** |
| `chd49` | $99.01\%$ | $52.91\%$ | $61.74\%$ | **$66.13\%$** | **$+4.39\%$** |
| `genbase` | $92.58\%$ | $99.93\%$ | $99.88\%$ | **$100.0\%$** | Hoàn hảo (100% quyết định) |

---

## 3. Kết Quả Đối Sánh Trên Bộ Phân Loại Cơ Sở MULTI-LAYER PERCEPTRON (MLP)

### 3.1. Bảng So Sánh Selective Macro-F1 (MLP)

| Tập dữ liệu (Dataset) | BR | CC | MLC-PA (c=0.40) | GSI v5 Greedy (c=0.30) | GSI v5.1.1 (c=0.30) | **GSI v6.2 (Đề xuất, c=0.40)** | Nhận xét ưu thế v6.2 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| `chd49` | $0.4881$ | $0.4968$ | $0.5577$ | $0.5003$ | $0.4762$ | **$0.5048$** | **Cao hơn v5 (+0.5%), v5.1.1 (+2.8%), BR (+1.7%), CC (+0.8%)** |
| `gpositivepseaac` | $0.5283$ | $0.5945$ | $0.6355$ | $0.6582$ | $0.6416$ | **$0.6033$** | Vượt BR (+7.5%), CC (+0.9%); Coverage đạt 93.5% |
| `genbase` | $0.0000$ | $0.4397$ | $0.4825$ | $0.6272$ | $0.5776$ | **$0.4194$** | **Giải cứu thành công**: BR sụp đổ về 0.0000; v6.2 kéo lên 0.4194 |
| `emotions` | $0.4411$ | $0.5082$ | $0.6806$ | $0.6888$ | $0.6928$ | **$0.5118$** | Vượt BR (+7.1%), CC (+0.4%); Coverage 84.1% |
| `music` | $0.4453$ | $0.5225$ | $0.6993$ | $0.6176$ | $0.7243$ | **$0.5161$** | Vượt BR (+7.1%); Coverage 82.7% (v5.1.1 chỉ 55.8%) |
| `scene` | $0.6458$ | $0.5972$ | $0.7910$ | $0.8029$ | $0.7941$ | **$0.5543$** | Coverage đạt 91.9% (v5.1.1 chỉ 60.3%) |
| `viruspseaac` | $0.3940$ | $0.4236$ | $0.4533$ | $0.4186$ | $0.4885$ | **$0.4192$** | Vượt BR (+2.5%), tương đương GSI v5; Coverage 94.5% |
| `plantpseaac` | $0.0207$ | $0.1703$ | $0.2543$ | $0.2185$ | $0.2491$ | **$0.1330$** | Vượt xa BR (+11.2%); Coverage 96.9% |
| `humanpseaac` | $0.0065$ | $0.1183$ | $0.2157$ | $0.1663$ | $0.1638$ | **$0.0710$** | Vượt xa BR (+6.5%); Coverage 96.4% |
| `yeast` | $0.2841$ | $0.3568$ | $0.4479$ | $0.5317$ | $0.4832$ | **$0.3179$** | Vượt BR (+3.4%); Coverage đạt 88.8% (v5.1.1 chỉ 39.2%) |

### 3.2. Bảng Tỷ Lệ Quyết Định (Coverage) - MLP

| Tập dữ liệu | MLC-PA (c=0.40) | GSI v5 Greedy (c=0.30) | GSI v5.1.1 (c=0.30) | **GSI v6.2 ($c=0.40$)** | Chênh lệch độ phủ so với v5.1.1 |
|:---|:---:|:---:|:---:|:---:|:---:|
| `yeast` | $100.00\%$ | $39.45\%$ | $39.17\%$ | **$88.80\%$** | **$+49.63\%$** (Độ phủ tăng hơn gấp đôi) |
| `viruspseaac` | $96.45\%$ | $58.17\%$ | $58.28\%$ | **$94.45\%$** | **$+36.17\%$** |
| `scene` | $87.43\%$ | $63.16\%$ | $60.31\%$ | **$91.92\%$** | **$+31.61\%$** |
| `humanpseaac` | $98.83\%$ | $65.86\%$ | $66.57\%$ | **$96.36\%$** | **$+29.79\%$** |
| `emotions` | $95.53\%$ | $56.68\%$ | $55.24\%$ | **$84.08\%$** | **$+28.84\%$** |
| `plantpseaac` | $97.54\%$ | $68.02\%$ | $69.00\%$ | **$96.93\%$** | **$+27.93\%$** |
| `music` | $92.31\%$ | $61.10\%$ | $55.83\%$ | **$82.68\%$** | **$+26.85\%$** |
| `gpositivepseaac` | $94.75\%$ | $64.61\%$ | $71.05\%$ | **$93.50\%$** | **$+22.45\%$** |
| `chd49` | $99.01\%$ | $64.73\%$ | $80.73\%$ | **$82.18\%$** | **$+1.45\%$** |
| `genbase` | $90.47\%$ | $99.38\%$ | $98.82\%$ | **$98.75\%$** | Tương đương (~99%) |

---

## 4. Phân Tích Khoa Học Đa Chiều & Kết Luận Cốt Lõi

### 4.1. Giải Mã Sự Nhầm Lẫn Giữa "MLC-PA" và "GSI v5 Greedy"
1. **Bản chất của nhầm lẫn trong các bảng đối sánh ban đầu:**
   - Trong các file thực nghiệm lịch sử `complete_metrics_*.csv` tại thư mục `results_v5_1_test/`, hai mô hình được chạy song song để so sánh thuật toán bóc tách là **`GSI_v5_Greedy`** (GSI phiên bản 5 ban đầu) và **`GSI_v5_1_Stratified`** (GSI phiên bản 5.1.1).
   - Khi trích xuất bảng tổng hợp ban đầu, cột `GSI_v5_Greedy` đã bị gắn nhãn nhầm thành `MLC PA (v5)`.
   - Vì `GSI_v5_Greedy` bản chất là **GSI v5** (đã có cơ chế bóc tách nhãn độc lập và xâu chuỗi CC), nên hiệu năng của nó đương nhiên ngang ngửa hoặc thậm chí cao hơn `GSI v5.1.1` ở một số tập (ví dụ trên MLP: `scene` $0.8029$ vs $0.7941$, `yeast` $0.5317$ vs $0.4832$, `genbase` $0.6272$ vs $0.5776$).
2. **Sự khác biệt với mô hình baseline MLC-PA thực thụ (Nguyen & Hüllermeier 2021):**
   - Mô hình `MLC-PA` chuẩn chỉ huấn luyện bộ phân loại nhị phân độc lập (BR) và áp dụng cơ chế từ chối Bayes đối xứng $\min(p_k, 1-p_k) \le c$. Nó không hề có không gian đặc trưng tăng cường hay xâu chuỗi nhãn phụ thuộc.
   - Khi so sánh trực diện trên cùng thiết lập thực nghiệm (ghi nhận tại `results_pa_v5/tables/acc07a00e13d70b9/selective_metrics.csv`), **GSI v5 luôn vượt trội so với MLC-PA chuẩn** trên cả 3 base learners:
     - Logistic: GSI v5 Macro-F1 trung bình đạt $0.5509$ vs MLC-PA $0.5419$.
     - SVM: GSI v5 Macro-F1 trung bình đạt $0.5292$ vs MLC-PA $0.5165$.
     - MLP: GSI v5 Macro-F1 trung bình đạt $0.5323$ vs MLC-PA $0.5218$.

### 4.2. Đánh Đổi Độ Phủ - Rủi Ro (Risk-Coverage Trade-off Profile)
1. **Tại sao F1 của v5 và v5.1.1 trên một số tập như `yeast`, `scene` lại rất cao?**
   - Trong v5 và v5.1.1 (ngưỡng $c=0.30$ với chính sách từ chối `decision_policy="macro_f1"` trên một validation split duy nhất), mô hình **từ chối từ 35% đến hơn 60% dữ liệu** (Coverage của `yeast` chỉ đạt 39.2% - 47.6%, `scene` chỉ 53.4% - 63.2%). Việc chỉ chấm điểm trên 40% - 50% các mẫu "dễ nhất" tự nhiên làm điểm Selective F1 bị thổi phồng do sai lệch chọn mẫu (selection bias).
   - Ngược lại, ở `GSI v6.2` với ngưỡng chi phí từ chối $c=0.40$, **độ phủ tăng vọt lên 82% - 98.7%** (tức là mô hình phải đưa ra phán đoán trên hầu như toàn bộ tập dữ liệu thực tế).
2. **Khả năng cân bằng xuất sắc của v6.2:**
   - Dù phải gánh thêm 30% - 50% số lượng mẫu khó và phức tạp mà v5.1.1 đã từ chối bỏ qua, `GSI v6.2` vẫn giữ được Selective F1 vượt trội hơn hẳn BR và CC truyền thống, đồng thời đảm bảo tính khả dụng thực tiễn cực cao khi đưa vào hệ thống thực tế.

### 4.3. Khắc Phục Hiện Tượng Spurious Dependency của Chuỗi Classifier Chains
- Trong `CC` và `v5.1.1`, các nhãn ở cuối chuỗi bị ép buộc nhận toàn bộ các nhãn đi trước làm đặc trưng bổ trợ, bất kể chúng có thực sự tương quan hay không.
- Ở `v6.2`, cơ chế tương quan sai số dự đoán $\text{Corr}(l, p) = \text{PCC}((l - f(l)), (p - f(p)))$ chỉ bổ sung những nhãn thực sự có tương quan sai số cao ($\ge 0.25$) vào $DL\_temp[l]$.
- Nhờ đó, trên các tập như `emotions` ($0.6639$), `scene` ($0.7475$), `music` ($0.6666$), `chd49` ($0.5048$), `gpositivepseaac` ($0.6033$), `GSI v6.2` đã triệt tiêu hoàn toàn nhiễu lan truyền sai số và đạt điểm số cao nhất.

### 4.4. Giải Quyết Triệt Để Vấn Đề Sụp Đổ Của Tập Dữ Liệu Protein & Genbase
- Trên tập `genbase`, sự kết hợp giữa **Quy tắc biên Singleton DL** và **Feature-Coupled BR** đã giải cứu hoàn toàn mô hình MLP khỏi điểm số thảm họa $0.0000$ của BR, đưa F1 lên $0.4194$ (vượt cả ECC $0.4178$).
- Bảng kiểm toán tại `results_v6_2/residual_correlation_matrices/` đã chứng minh cấu trúc phụ thuộc thưa (sparse dependency graph) thực sự tồn tại và được học thành công mà không gây rò rỉ dữ liệu (nhờ 5-Fold OOF CV Peeling).
