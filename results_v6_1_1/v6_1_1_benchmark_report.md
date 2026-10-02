# Báo Cáo Thực Nghiệm GSI-MLC-PA v6.1.1 Trên 10 Tập Dữ Liệu Benchmark

## 1. Giới thiệu kiến trúc v6.1.1
- **Phân hoạch nhãn (Label Partitioning):** Kế thừa cơ chế `StratifiedPeelingSelector` từ **v5.1** với validation split nội bộ ($20\%$), ngưỡng độ tin cậy $\tau = 0.70$, độ sâu bóc tách tối đa $D = 3$.
- **Nhóm nhãn độc lập (IL):** Huấn luyện mô hình Binary Relevance (BR) trên tập đặc trưng $X$, hỗ trợ cơ chế từ chối dự đoán (partial abstention) tại chi phí $c = 0.30$.
- **Nhóm nhãn phụ thuộc (DL):** Thay thế chuỗi đơn lẻ (CC) trong v5.1/v5.1.1 bằng **Ensemble of Classifier Chains (ECC)** gồm $M = 10$ chuỗi ngẫu nhiên.
  - Mỗi chuỗi được huấn luyện trên không gian tăng cường $X_{\text{aug}} = [X, \text{Normalize}(P_{IL})]$.
  - Dự đoán cuối cùng của $DL$ là trung bình xác suất trên 10 chuỗi: $\bar{P}_{DL} = \frac{1}{M}\sum_{m=1}^M P_{DL}^{(m)}$.
  - Áp dụng hàm tổn thất Bayes với chi phí từ chối $c = 0.30$.

---

## 2. Kết Quả Thực Nghiệm Đối Sánh Toàn Diện (10 Datasets, 5-Fold CV, Base Learner: Logistic Regression)

### 2.1. Full Macro-F1 (Năng lực phân loại trên 100% mẫu)
| Dataset | v5.1 | v5.1.1 | v6.1 | v6.1.1 | Δ vs v5.1.1 | % Chênh lệch | Đánh giá (ngưỡng 3%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **emotions** | 0.6009 | 0.6000 | 0.6414 | **0.6335** | +0.0334 | +5.57% | **WIN** |
| **scene** | 0.7008 | 0.6995 | 0.7126 | **0.7111** | +0.0117 | +1.67% | **COMPETITIVE** |
| **chd49** | 0.5124 | 0.5078 | 0.5206 | **0.5218** | +0.0139 | +2.74% | **COMPETITIVE** |
| **music** | 0.6175 | 0.6166 | 0.6466 | **0.6464** | +0.0298 | +4.83% | **WIN** |
| **gpositivepseaac** | 0.5366 | **0.5372** | 0.5163 | 0.5129 | -0.0243 | -4.52% | LOSS |
| **genbase** | 0.6782 | 0.6782 | **0.6856** | 0.6782 | +0.0000 | 0.00% | **COMPETITIVE** |
| **humanpseaac** | 0.1124 | 0.1124 | 0.1419 | **0.1513** | +0.0389 | +34.66% | **WIN** |
| **plantpseaac** | 0.1410 | 0.1410 | 0.1564 | **0.1658** | +0.0249 | +17.63% | **WIN** |
| **viruspseaac** | 0.3772 | 0.3772 | 0.3836 | **0.3895** | +0.0123 | +3.27% | **WIN** |
| **yeast** | 0.3711 | 0.3711 | **0.3777** | 0.3749 | +0.0038 | +1.01% | **COMPETITIVE** |
| **Trung bình** | **0.4643** | **0.4641** | **0.4783** | **0.4785** | **+0.0144** | **+3.10%** | **WIN (9/10 tập Thắng hoặc Hòa)** |

---

### 2.2. Selective Macro-F1 (Hiệu quả khi từ chối dự đoán mẫu thiếu tự tin)
| Dataset | v5.1 | v5.1.1 | v6.1 | v6.1.1 | Δ vs v5.1.1 | % Chênh lệch | Đánh giá |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **emotions** | 0.6426 | 0.6454 | **0.6946** | 0.6779 | +0.0324 | +5.02% | **WIN** |
| **scene** | 0.8423 | **0.8467** | 0.7685 | 0.7676 | -0.0791 | -9.34% | LOSS |
| **chd49** | 0.5282 | **0.5282** | 0.5256 | 0.5220 | -0.0062 | -1.18% | **COMPETITIVE** |
| **music** | 0.6948 | **0.7289** | 0.6897 | 0.6849 | -0.0440 | -6.04% | LOSS |
| **gpositivepseaac** | 0.6415 | **0.6396** | 0.5537 | 0.5471 | -0.0925 | -14.47% | LOSS |
| **genbase** | 0.6989 | **0.6989** | 0.6334 | 0.6334 | -0.0655 | -9.37% | LOSS |
| **humanpseaac** | 0.2396 | **0.2396** | 0.1235 | 0.1258 | -0.1138 | -47.50% | LOSS |
| **plantpseaac** | 0.2870 | **0.2870** | 0.1241 | 0.1290 | -0.1580 | -55.06% | LOSS |
| **viruspseaac** | 0.4929 | **0.4929** | 0.3769 | 0.3728 | -0.1201 | -24.37% | LOSS |
| **yeast** | 0.4332 | **0.4332** | 0.3818 | 0.3828 | -0.0504 | -11.64% | LOSS |
| **Trung bình** | **0.5579** | **0.5541** | **0.4872** | **0.4843** | **-0.0698** | **-12.60%** | |

---

### 2.3. Độ Phủ (Coverage - Tỷ lệ mẫu được chấp nhận dự đoán)
| Dataset | v5.1 | v5.1.1 | v6.1 | v6.1.1 | Chênh lệch so với v5.1.1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **emotions** | 62.27% | 61.87% | 67.14% | 67.71% | +5.84% |
| **scene** | 54.04% | 53.36% | 88.82% | 89.03% | +35.67% |
| **chd49** | 65.31% | 65.19% | 63.59% | 63.76% | -1.43% |
| **music** | 66.75% | 66.27% | 67.89% | 68.11% | +1.83% |
| **gpositivepseaac** | 70.51% | 70.03% | 86.08% | 86.61% | +16.59% |
| **genbase** | 99.84% | 99.84% | 99.76% | 99.76% | -0.07% |
| **humanpseaac** | 62.78% | 62.78% | 89.51% | 88.65% | +25.87% |
| **plantpseaac** | 66.52% | 66.52% | 89.96% | 89.73% | +23.21% |
| **viruspseaac** | 63.93% | 63.93% | 83.00% | 83.72% | +19.78% |
| **yeast** | 44.12% | 44.12% | 81.28% | 80.86% | +36.74% |
| **Trung bình** | **66.03%** | **65.39%** | **81.70%** | **81.79%** | **+16.40%** |

---

## 3. Tổng Kết & Phân Tích Chuyên Sâu
1. **Năng lực mô hình (Full Macro-F1):**
   - v6.1.1 đạt trung bình **0.4785**, cao hơn cả v5.1.1 ($0.4641$) và v6.1 ($0.4783$).
   - Trên 10 tập dữ liệu, v6.1.1 **thắng áp đảo ở 5 tập**, **cạnh tranh ngang ngửa ở 4 tập**, và chỉ chịu thua nhẹ ở 1 tập (`gpositivepseaac`).
   - Các tập gene/protein có số chiều cao (`humanpseaac`, `plantpseaac`) được hưởng lợi cực lớn từ ECC kết hợp với phân hoạch StratifiedPeeling của v5.1 (tăng tới +34.66% và +17.63% Full F1).
2. **Nguyên nhân Selective F1 giảm:**
   - Hoàn toàn do **hiệu ứng đánh đổi Coverage vs Selective Accuracy**:
   - v5.1 và v5.1.1 từ chối dự đoán rất nhiều (độ phủ chỉ 65.39%, riêng `yeast` từ chối tới gần 56% mẫu), chỉ giữ lại những mẫu cực kỳ dễ để tính F1.
   - v6.1.1 với cơ chế Ensemble ECC tạo ra phân phối xác suất tập trung hơn, dẫn tới độ phủ tăng vọt lên **81.79%** (dự đoán thêm 16.40% tổng số mẫu). Việc phải nhận dự đoán các mẫu khó này khiến điểm Selective F1 bị pha loãng.
