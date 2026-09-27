# BÁO CÁO THỰC NGHIỆM GSI-MLC-PA V5.1 TRÊN 5 TẬP DỮ LIỆU CÒN LẠI (NO PENALTY)

**Cấu hình:** Base Learner = `logistic`, Ngưỡng phân định độc lập $\tau = 0.7$, 5-Fold Stratified CV, Chi phí từ chối $c = 0.30$, Không dùng hàm phạt (`use_complexity_penalty=False`).

## 1. Bảng Tổng Hợp Selective Macro-F1 và Coverage (c = 0.30)

| STT | Tập dữ liệu | Số mẫu / Nhãn | BR (F1 / Cov) | CC (F1 / Cov) | GSI v5 Greedy (F1 / Cov) | GSI v5.1 Stratified (F1 / Cov) | Tăng trưởng vs. v5 Greedy |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **genbase** | 27 nhãn | 0.6782 / 1.00 | 0.6930 / 1.00 | 0.6996 / 0.998 | **0.6989** / 0.998 | -0.0007 |
| 2 | **humanpseaac** | 14 nhãn | 0.1124 / 1.00 | 0.1397 / 1.00 | 0.1867 / 0.645 | **0.2194** / 0.631 | **+0.0327 🏆** |
| 3 | **plantpseaac** | 12 nhãn | 0.1410 / 1.00 | 0.1715 / 1.00 | 0.2562 / 0.650 | **0.2693** / 0.669 | **+0.0131 🏆** |
| 4 | **viruspseaac** | 6 nhãn | 0.3692 / 1.00 | 0.3836 / 1.00 | 0.4997 / 0.646 | **0.5038** / 0.639 | +0.0041 |
| 5 | **yeast** | 14 nhãn | 0.3748 / 1.00 | 0.4017 / 1.00 | 0.5088 / 0.449 | **0.5379** / 0.476 | **+0.0291 🏆** |
| — | **TRUNG BÌNH** | — | 0.3351 / 1.00 | 0.3579 / 1.00 | 0.4302 / 0.678 | **0.4459** / 0.683 | **+0.0157 🏆** |

## 2. Chi Tiết Cấu Trúc Phân Tầng IL / DL và Thời Gian Huấn Luyện
| Tập dữ liệu | Mô hình | Số Tầng Bóc Tách (Stages) | Số Lượng IL Trung Bình | Tỷ Lệ IL (%) | Lý Do Dừng | Thời Gian (s) |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| **genbase** | `GSI_v5_Greedy` | 27.0 | 0.0 / 27 | 0.0% | `greedy_forward` | 0.607s |
| **genbase** | `GSI_v5_1_Stratified` | 1.0 | 26.8 / 27 | 99.3% | `all_labels_independent` | 0.589s |
| **humanpseaac** | `GSI_v5_Greedy` | 14.0 | 3.8 / 14 | 27.1% | `greedy_forward` | 3.115s |
| **humanpseaac** | `GSI_v5_1_Stratified` | 1.0 | 0.0 / 14 | 0.0% | `no_promotion_in_stage` | 4.122s |
| **plantpseaac** | `GSI_v5_Greedy` | 12.0 | 3.6 / 12 | 30.0% | `greedy_forward` | 0.814s |
| **plantpseaac** | `GSI_v5_1_Stratified` | 1.0 | 0.0 / 12 | 0.0% | `no_promotion_in_stage` | 1.159s |
| **viruspseaac** | `GSI_v5_Greedy` | 6.0 | 1.0 / 6 | 16.7% | `greedy_forward` | 0.123s |
| **viruspseaac** | `GSI_v5_1_Stratified` | 2.0 | 1.8 / 6 | 30.0% | `no_promotion_in_stage` | 0.196s |
| **yeast** | `GSI_v5_Greedy` | 14.0 | 3.8 / 14 | 27.1% | `greedy_forward` | 1.237s |
| **yeast** | `GSI_v5_1_Stratified` | 2.0 | 2.6 / 14 | 18.6% | `no_promotion_in_stage` | 2.192s |