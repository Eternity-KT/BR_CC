# BÁO CÁO THỰC NGHIỆM TOÀN DIỆN GSI-MLC-PA V5.1 TRÊN TOÀN BỘ 10 TẬP DỮ LIỆU (NO PENALTY)

**Cấu hình:** Base Learner = `logistic`, Ngưỡng phân định độc lập $\tau = 0.7$, 5-Fold Stratified CV, Chi phí từ chối $c = 0.30$, Không dùng hàm phạt (`use_complexity_penalty=False`).

## 1. Bảng Tổng Hợp Selective Macro-F1 và Coverage (c = 0.30)

| STT | Tập dữ liệu | Số mẫu / Nhãn | BR (F1 / Cov) | CC (F1 / Cov) | GSI v5 Greedy (F1 / Cov) | GSI v5.1 Stratified (F1 / Cov) | Tăng trưởng vs. v5 Greedy |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **emotions** | 6 nhãn | 0.5972 / 1.00 | 0.5932 / 1.00 | 0.6359 / 0.648 | **0.6426** / 0.623 | +0.0067 |
| 2 | **scene** | 6 nhãn | 0.6994 / 1.00 | 0.7234 / 1.00 | 0.8081 / 0.578 | **0.8423** / 0.540 | **+0.0342 🏆** |
| 3 | **chd49** | 6 nhãn | 0.5103 / 1.00 | 0.5073 / 1.00 | 0.4834 / 0.625 | **0.5282** / 0.653 | **+0.0448 🏆** |
| 4 | **music** | 6 nhãn | 0.6067 / 1.00 | 0.6031 / 1.00 | 0.6778 / 0.657 | **0.6948** / 0.667 | **+0.0170 🏆** |
| 5 | **gpositivepseaac** | 4 nhãn | 0.5535 / 1.00 | 0.5791 / 1.00 | 0.6542 / 0.705 | **0.6415** / 0.705 | -0.0127 |
| 6 | **genbase** | 27 nhãn | 0.6782 / 1.00 | 0.6930 / 1.00 | 0.6996 / 0.998 | **0.6989** / 0.998 | -0.0007 |
| 7 | **humanpseaac** | 14 nhãn | 0.1124 / 1.00 | 0.1397 / 1.00 | 0.1867 / 0.645 | **0.2194** / 0.631 | **+0.0327 🏆** |
| 8 | **plantpseaac** | 12 nhãn | 0.1410 / 1.00 | 0.1715 / 1.00 | 0.2562 / 0.650 | **0.2693** / 0.669 | **+0.0131 🏆** |
| 9 | **viruspseaac** | 6 nhãn | 0.3692 / 1.00 | 0.3836 / 1.00 | 0.4997 / 0.646 | **0.5038** / 0.639 | +0.0041 |
| 10 | **yeast** | 14 nhãn | 0.3748 / 1.00 | 0.4017 / 1.00 | 0.5088 / 0.449 | **0.5379** / 0.476 | **+0.0291 🏆** |
| — | **TRUNG BÌNH** | — | 0.4643 / 1.00 | 0.4796 / 1.00 | 0.5410 / 0.660 | **0.5579** / 0.660 | **+0.0168 🏆** |

## 2. Chi Tiết Cấu Trúc Phân Tầng IL / DL và Thời Gian Huấn Luyện
| Tập dữ liệu | Mô hình | Số Tầng Bóc Tách (Stages) | Số Lượng IL Trung Bình | Tỷ Lệ IL (%) | Lý Do Dừng | Thời Gian (s) |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| **emotions** | `GSI_v5_Greedy` | 6.0 | 2.6 / 6 | 43.3% | `greedy_forward` | 0.183s |
| **emotions** | `GSI_v5_1_Stratified` | 1.8 | 3.6 / 6 | 60.0% | `no_promotion_in_stage` | 0.281s |
| **scene** | `GSI_v5_Greedy` | 6.0 | 1.4 / 6 | 23.3% | `greedy_forward` | 2.997s |
| **scene** | `GSI_v5_1_Stratified` | 2.2 | 3.6 / 6 | 60.0% | `no_promotion_in_stage` | 4.387s |
| **chd49** | `GSI_v5_Greedy` | 6.0 | 2.0 / 6 | 33.3% | `greedy_forward` | 0.230s |
| **chd49** | `GSI_v5_1_Stratified` | 2.0 | 2.0 / 6 | 33.3% | `no_promotion_in_stage` | 0.315s |
| **music** | `GSI_v5_Greedy` | 6.0 | 2.0 / 6 | 33.3% | `greedy_forward` | 0.219s |
| **music** | `GSI_v5_1_Stratified` | 1.8 | 3.6 / 6 | 60.0% | `no_promotion_in_stage` | 0.310s |
| **gpositivepseaac** | `GSI_v5_Greedy` | 4.0 | 0.6 / 4 | 15.0% | `greedy_forward` | 0.279s |
| **gpositivepseaac** | `GSI_v5_1_Stratified` | 1.2 | 3.0 / 4 | 75.0% | `all_labels_independent` | 0.454s |
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