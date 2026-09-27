# BÁO CÁO THỰC NGHIỆM ĐỐI CHUẨN GSI-MLC-PA V5.1 (NO PENALTY VS. WITH PENALTY) TRÊN 5 TẬP DỮ LIỆU

**Cấu hình:** Base Learner = `logistic`, Ngưỡng phân định độc lập $\tau = 0.7$, 5-Fold Stratified CV, Phạt chi phí tính toán $\lambda = 0.01$.

## 1. Bảng Tổng Hợp Kết Quả Selective Macro-F1 và Coverage (c = 0.30)

| Tập dữ liệu | BR (F1/Cov) | CC (F1/Cov) | GSI v5 Greedy | GSI v5.1 (No Penalty) | GSI v5.1 (With Penalty) | Thời gian NoPen vs WithPen |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **emotions** | 0.5972 / 1.00 | 0.5932 / 1.00 | 0.6359 / 0.648 | 0.6426 / 0.623 | **0.6426** / 0.623 | 0.28s vs **0.28s** |
| **scene** | 0.6994 / 1.00 | 0.7234 / 1.00 | 0.8081 / 0.578 | 0.8423 / 0.540 | **0.8441** / 0.541 | 4.39s vs **4.22s** |
| **chd49** | 0.5103 / 1.00 | 0.5073 / 1.00 | 0.4834 / 0.625 | 0.5282 / 0.653 | **0.5282** / 0.653 | 0.32s vs **0.31s** |
| **music** | 0.6067 / 1.00 | 0.6031 / 1.00 | 0.6778 / 0.657 | 0.6948 / 0.667 | **0.6948** / 0.667 | 0.31s vs **0.28s** |
| **gpositivepseaac** | 0.5535 / 1.00 | 0.5791 / 1.00 | 0.6542 / 0.705 | 0.6415 / 0.705 | **0.6415** / 0.705 | 0.45s vs **0.41s** |
| **TRUNG BÌNH** | 0.5934 / 1.00 | 0.6012 / 1.00 | 0.6519 / 0.643 | 0.6699 / 0.638 | **0.6702** / 0.638 | 1.15s vs **1.10s** |

## 2. Chi Tiết Ảnh Hưởng Của Hàm Phạt Phức Tạp (Complexity Penalty Diagnostics)
| Tập dữ liệu | Model | Số Tầng Bóc Tách (Stages) | Số Lượng IL | Lý Do Dừng | Thời Gian (s) |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **emotions** | `GSI_v5_Greedy` | 6.0 | 2.6/6 | `greedy_forward` | 0.183s |
| **emotions** | `GSI_v5_1_NoPenalty` | 1.8 | 3.6/6 | `no_promotion_in_stage` | 0.281s |
| **emotions** | `GSI_v5_1_WithPenalty` | 1.8 | 3.6/6 | `no_promotion_in_stage` | 0.279s |
| **scene** | `GSI_v5_Greedy` | 6.0 | 1.4/6 | `greedy_forward` | 2.997s |
| **scene** | `GSI_v5_1_NoPenalty` | 2.2 | 3.6/6 | `no_promotion_in_stage` | 4.387s |
| **scene** | `GSI_v5_1_WithPenalty` | 2.0 | 3.4/6 | `no_promotion_in_stage` | 4.225s |
| **chd49** | `GSI_v5_Greedy` | 6.0 | 2.0/6 | `greedy_forward` | 0.230s |
| **chd49** | `GSI_v5_1_NoPenalty` | 2.0 | 2.0/6 | `no_promotion_in_stage` | 0.315s |
| **chd49** | `GSI_v5_1_WithPenalty` | 2.0 | 2.0/6 | `no_promotion_in_stage` | 0.314s |
| **music** | `GSI_v5_Greedy` | 6.0 | 2.0/6 | `greedy_forward` | 0.219s |
| **music** | `GSI_v5_1_NoPenalty` | 1.8 | 3.6/6 | `no_promotion_in_stage` | 0.310s |
| **music** | `GSI_v5_1_WithPenalty` | 1.8 | 3.6/6 | `no_promotion_in_stage` | 0.276s |
| **gpositivepseaac** | `GSI_v5_Greedy` | 4.0 | 0.6/4 | `greedy_forward` | 0.279s |
| **gpositivepseaac** | `GSI_v5_1_NoPenalty` | 1.2 | 3.0/4 | `all_labels_independent` | 0.454s |
| **gpositivepseaac** | `GSI_v5_1_WithPenalty` | 1.2 | 3.0/4 | `all_labels_independent` | 0.405s |