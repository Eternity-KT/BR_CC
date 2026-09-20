# Bảng Kết Quả Đánh Giá Trên 5 Tập Dữ Liệu Mới: CHD49 và 4 Tập *PseAAC

> **Mô hình thực nghiệm:**
> 1. **MLC-PA:** Multi-Label Classification with Partial Abstention (Nguyen & Hüllermeier, 2021).
> 2. **GSI-MLC-PA:** Group-Sensitive Information Multi-Label Classification with Partial Abstention.
> **5 Tập dữ liệu mới:** `chd49`, `viruspseaac`, `gpositivepseaac`, `plantpseaac`, `humanpseaac` (Đánh giá qua 5-Fold Cross-Validation).

---
### Định nghĩa các chỉ số:
- **Coverage (Độ bao phủ $\Gamma$):** Tỷ lệ phần trăm các vị trí nhãn được chấp nhận dự đoán (không từ chối).
- **Selective Macro-F1:** Macro-F1 tính trên các vị trí nhãn được chấp nhận.
- **Optimistic Macro-F1:** Macro-F1 giả định chuyên gia con người can thiệp gán đúng các nhãn bị từ chối.
- **Full Macro-F1 ($c = 0.50$):** Macro-F1 khi mô hình dự đoán toàn bộ (Coverage = 100%).

---

## 1. Lớp Phân Loại Cơ Sở: Logistic

### 1.1. So sánh Tổng quan theo Mức Chi phí Từ chối ($c$) (Trung bình 5 Tập Dữ Liệu Mới)

| Chi phí $c$ | MLC_PA_Logistic Coverage | MLC_PA_Logistic Sel. F1 | GSI_MLC_PA_Logistic Coverage | GSI_MLC_PA_Logistic Sel. F1 | $\Delta$ Coverage | $\Delta$ Sel. F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$c = 0.20$** | 0.7399 | 0.3179 | **0.7565** | **0.3064** | +1.65% | -0.0115 |
| **$c = 0.25$** | 0.7921 | 0.3190 | **0.8057** | **0.3115** | +1.36% | -0.0076 |
| **$c = 0.30$** | 0.8368 | 0.3200 | **0.8501** | **0.3155** | +1.33% | -0.0045 |
| **$c = 0.35$** | 0.8773 | 0.3226 | **0.8865** | **0.3155** | +0.92% | -0.0071 |
| **$c = 0.40$** | 0.9206 | 0.3225 | **0.9269** | **0.3129** | +0.63% | -0.0096 |
| **Full ($c = 0.50$)** | 1.0000 | 0.3373 | **1.0000** | **0.3252** | +0.00% | -0.0121 |


### 1.2. Chi tiết 5 Tập Dữ Liệu Mới tại Ngưỡng Chi Phí Chuẩn $c = 0.30$

| Tập dữ liệu | MLC-PA Coverage | MLC-PA Sel. F1 | GSI Coverage | GSI Sel. F1 | $\Delta$ Coverage | $\Delta$ Sel. F1 | Mô hình tốt hơn |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CHD49** | 0.6304 | 0.5242 | **0.6691** | **0.5160** | +3.87% | -0.0082 | Tương đương |
| **VIRUSPSEAAC** | 0.8434 | 0.3543 | **0.8459** | **0.3595** | +0.25% | +0.0052 | **GSI 🏆** (+Cov) |
| **GPOSITIVEPSEAAC** | 0.8584 | 0.5370 | **0.8695** | **0.5366** | +1.11% | -0.0004 | **GSI 🏆** (+Cov) |
| **PLANTPSEAAC** | 0.9295 | 0.0975 | **0.9367** | **0.0891** | +0.71% | -0.0085 | Tương đương |
| **HUMANPSEAAC** | 0.9224 | 0.0869 | **0.9293** | **0.0763** | +0.68% | -0.0105 | Tương đương |
| **TRUNG BÌNH** | **0.8368** | **0.3200** | **0.8501** | **0.3155** | **+1.33%** | **-0.0045** | Tương đương |

## 2. Lớp Phân Loại Cơ Sở: SVM

### 2.1. So sánh Tổng quan theo Mức Chi phí Từ chối ($c$) (Trung bình 5 Tập Dữ Liệu Mới)

| Chi phí $c$ | MLC_PA_SVM Coverage | MLC_PA_SVM Sel. F1 | GSI_MLC_PA_SVM Coverage | GSI_MLC_PA_SVM Sel. F1 | $\Delta$ Coverage | $\Delta$ Sel. F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$c = 0.20$** | 0.6365 | 0.1380 | **0.6537** | **0.1837** | +1.72% | +0.0457 |
| **$c = 0.25$** | 0.7133 | 0.1934 | **0.7291** | **0.2142** | +1.58% | +0.0208 |
| **$c = 0.30$** | 0.7676 | 0.2033 | **0.7821** | **0.2202** | +1.45% | +0.0169 |
| **$c = 0.35$** | 0.8170 | 0.2036 | **0.8291** | **0.2253** | +1.21% | +0.0217 |
| **$c = 0.40$** | 0.8756 | 0.2208 | **0.8795** | **0.2524** | +0.39% | +0.0316 |
| **Full ($c = 0.50$)** | 1.0000 | 0.2426 | **1.0000** | **0.2791** | +0.00% | +0.0365 |


### 2.2. Chi tiết 5 Tập Dữ Liệu Mới tại Ngưỡng Chi Phí Chuẩn $c = 0.30$

| Tập dữ liệu | MLC-PA Coverage | MLC-PA Sel. F1 | GSI Coverage | GSI Sel. F1 | $\Delta$ Coverage | $\Delta$ Sel. F1 | Mô hình tốt hơn |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CHD49** | 0.4445 | 0.3009 | **0.4852** | **0.3025** | +4.06% | +0.0016 | **GSI 🏆** (+Cov) |
| **VIRUSPSEAAC** | 0.7021 | 0.2353 | **0.7149** | **0.2076** | +1.28% | -0.0277 | Tương đương |
| **GPOSITIVEPSEAAC** | 0.8097 | 0.4640 | **0.8416** | **0.5611** | +3.18% | +0.0971 | **GSI 🏆** (+Cov) |
| **PLANTPSEAAC** | 0.9446 | 0.0149 | **0.9373** | **0.0244** | -0.73% | +0.0095 | **GSI 🏆** (+F1) |
| **HUMANPSEAAC** | 0.9370 | 0.0014 | **0.9317** | **0.0056** | -0.53% | +0.0042 | Tương đương |
| **TRUNG BÌNH** | **0.7676** | **0.2033** | **0.7821** | **0.2202** | **+1.45%** | **+0.0169** | **GSI 🏆** |

## 3. Lớp Phân Loại Cơ Sở: MLP

### 3.1. So sánh Tổng quan theo Mức Chi phí Từ chối ($c$) (Trung bình 5 Tập Dữ Liệu Mới)

| Chi phí $c$ | MLC_PA_MLP Coverage | MLC_PA_MLP Sel. F1 | GSI_MLC_PA_MLP Coverage | GSI_MLC_PA_MLP Sel. F1 | $\Delta$ Coverage | $\Delta$ Sel. F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **$c = 0.20$** | 0.7119 | 0.2606 | **0.7371** | **0.3131** | +2.53% | +0.0525 |
| **$c = 0.25$** | 0.7659 | 0.2920 | **0.7919** | **0.3288** | +2.60% | +0.0368 |
| **$c = 0.30$** | 0.8164 | 0.2948 | **0.8404** | **0.3301** | +2.40% | +0.0353 |
| **$c = 0.35$** | 0.8687 | 0.2984 | **0.8859** | **0.3385** | +1.73% | +0.0402 |
| **$c = 0.40$** | 0.9131 | 0.3105 | **0.9254** | **0.3460** | +1.23% | +0.0355 |
| **Full ($c = 0.50$)** | 1.0000 | 0.3204 | **1.0000** | **0.3512** | +0.00% | +0.0308 |


### 3.2. Chi tiết 5 Tập Dữ Liệu Mới tại Ngưỡng Chi Phí Chuẩn $c = 0.30$

| Tập dữ liệu | MLC-PA Coverage | MLC-PA Sel. F1 | GSI Coverage | GSI Sel. F1 | $\Delta$ Coverage | $\Delta$ Sel. F1 | Mô hình tốt hơn |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **CHD49** | 0.5757 | 0.4912 | **0.6356** | **0.4929** | +6.00% | +0.0016 | **GSI 🏆** (+Cov) |
| **VIRUSPSEAAC** | 0.8015 | 0.3508 | **0.8352** | **0.4271** | +3.37% | +0.0763 | **GSI 🏆** (+Cov) |
| **GPOSITIVEPSEAAC** | 0.8589 | 0.5395 | **0.8714** | **0.5460** | +1.25% | +0.0065 | **GSI 🏆** (+Cov) |
| **PLANTPSEAAC** | 0.9326 | 0.0550 | **0.9373** | **0.1204** | +0.47% | +0.0655 | **GSI 🏆** (+Cov) |
| **HUMANPSEAAC** | 0.9134 | 0.0374 | **0.9226** | **0.0638** | +0.93% | +0.0265 | **GSI 🏆** (+Cov) |
| **TRUNG BÌNH** | **0.8164** | **0.2948** | **0.8404** | **0.3301** | **+2.40%** | **+0.0353** | **GSI 🏆** |

## 4. Ma Trận So Sánh Toàn Diện: Coverage & Selective Macro-F1 tại $c = 0.30$

| Tập dữ liệu | MLC-PA Logistic | GSI Logistic | MLC-PA SVM | GSI SVM | MLC-PA MLP | GSI MLP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **CHD49** | 0.630 / 0.524 | **0.669 / 0.516** | 0.445 / 0.301 | **0.485 / 0.303** | 0.576 / 0.491 | **0.636 / 0.493** |
| **VIRUSPSEAAC** | 0.843 / 0.354 | **0.846 / 0.359** | 0.702 / 0.235 | **0.715 / 0.208** | 0.802 / 0.351 | **0.835 / 0.427** |
| **GPOSITIVEPSEAAC** | 0.858 / 0.537 | **0.870 / 0.537** | 0.810 / 0.464 | **0.842 / 0.561** | 0.859 / 0.540 | **0.871 / 0.546** |
| **PLANTPSEAAC** | 0.930 / 0.098 | **0.937 / 0.089** | 0.945 / 0.015 | **0.937 / 0.024** | 0.933 / 0.055 | **0.937 / 0.120** |
| **HUMANPSEAAC** | 0.922 / 0.087 | **0.929 / 0.076** | 0.937 / 0.001 | **0.932 / 0.006** | 0.913 / 0.037 | **0.923 / 0.064** |
| **TRUNG BÌNH** | 0.8368 / 0.3200 | **0.8501 / 0.3155** | 0.7676 / 0.2033 | **0.7821 / 0.2202** | 0.8164 / 0.2948 | **0.8404 / 0.3301** |

