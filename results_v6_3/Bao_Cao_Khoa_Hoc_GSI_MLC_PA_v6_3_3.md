# BÁO CÁO KHOA HỌC: GSI-MLC-PA v6.3.3 (ADAPTIVE TRI-REGIME INFERENCE)
**Đánh giá thực nghiệm toàn diện trên 10 tập dữ liệu benchmark với 3 bộ phân loại cơ sở (Logistic Regression, Linear SVM, MLP)**

*(Đối chuẩn trực tiếp giữa 6 mô hình: BR, CC, MLC-PA, GSI v6.2.1, GSI v6.3.2 và GSI v6.3.3 Đề Xuất)*

---
## 1. BẢNG TỔNG HỢP TOÀN CỤC (GRAND BENCHMARK SUMMARY: 30 CONFIGS)
| Mô hình | Selective Macro-F1 (↑) | Độ phủ Coverage (%) | Tỷ số F1 / Coverage (↑) | Selective Micro-F1 (↑) | Subset Acc (0/1) (↑) | Hamming Loss (↓) | Sel. Hamming (↓) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| BR | 0.3978 | 100.0% | 0.3978 | 0.5000 | 0.2768 | 0.1567 | 0.1567 |
| CC | 0.4498 | 100.0% | 0.4498 | 0.5850 | **0.3716** | 0.1623 | 0.1623 |
| MLC-PA | 0.3781 | 78.2% | 0.4835 | 0.5088 | 0.2768 | 0.1567 | **0.0976** |
| GSI v6.2.1 | 0.4140 | 80.3% | 0.5154 | 0.5702 | 0.3112 | **0.1557** | 0.1023 |
| GSI v6.3.2 | 0.5317 | 73.3% | 0.7252 | 0.6480 | 0.3425 | 0.1594 | 0.1515 |
| **GSI v6.3.3 (Đề Xuất)** | **0.5347** | 72.8% | **0.7340** | **0.6592** | 0.3403 | 0.1599 | 0.1385 |

---
## 2. PHÂN TÍCH THEO TỪNG BỘ HỌC CƠ SỞ (BASE LEARNERS)
| Bộ học cơ sở | Mô hình | Selective Macro-F1 (↑) | Coverage (%) | Subset Accuracy (↑) | Hamming Loss (↓) | Sel. Hamming (↓) | Selective Micro-F1 (↑) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | BR | 0.4643 | 100.0% | 0.3471 | 0.1520 | 0.1520 | 0.5881 |
| **Logistic Regression** | CC | 0.4796 | 100.0% | **0.4129** | 0.1604 | 0.1604 | 0.6103 |
| **Logistic Regression** | MLC-PA | 0.4590 | 81.9% | 0.3471 | 0.1520 | 0.1033 | 0.6096 |
| **Logistic Regression** | GSI v6.2.1 | 0.4709 | 82.1% | 0.3537 | **0.1503** | **0.1016** | 0.6152 |
| **Logistic Regression** | GSI v6.3.2 | 0.5593 | 73.7% | 0.3646 | 0.1554 | 0.1459 | 0.6677 |
| **Logistic Regression** | **GSI v6.3.3 (Đề Xuất)** | **0.5660** | 73.3% | 0.3663 | 0.1554 | 0.1354 | **0.6787** |
| **Linear SVM (Platt)** | BR | 0.4036 | 100.0% | 0.2938 | 0.1549 | 0.1549 | 0.5122 |
| **Linear SVM (Platt)** | CC | 0.4472 | 100.0% | **0.3826** | 0.1635 | 0.1635 | 0.5785 |
| **Linear SVM (Platt)** | MLC-PA | 0.3788 | 78.1% | 0.2938 | 0.1549 | **0.0946** | 0.5382 |
| **Linear SVM (Platt)** | GSI v6.2.1 | 0.3838 | 78.2% | 0.2976 | **0.1541** | 0.0953 | 0.5405 |
| **Linear SVM (Platt)** | GSI v6.3.2 | 0.5120 | 73.2% | 0.3379 | 0.1566 | 0.1516 | 0.6252 |
| **Linear SVM (Platt)** | **GSI v6.3.3 (Đề Xuất)** | **0.5123** | 72.6% | 0.3335 | 0.1576 | 0.1351 | **0.6396** |
| **MLP (Neural Net)** | BR | 0.3254 | 100.0% | 0.1896 | 0.1632 | 0.1632 | 0.3998 |
| **MLP (Neural Net)** | CC | 0.4228 | 100.0% | 0.3194 | 0.1631 | 0.1631 | 0.5663 |
| **MLP (Neural Net)** | MLC-PA | 0.2965 | 74.6% | 0.1896 | 0.1632 | **0.0950** | 0.3787 |
| **MLP (Neural Net)** | GSI v6.2.1 | 0.3871 | 80.7% | 0.2824 | **0.1626** | 0.1099 | 0.5550 |
| **MLP (Neural Net)** | GSI v6.3.2 | 0.5239 | 73.1% | **0.3251** | 0.1663 | 0.1570 | 0.6510 |
| **MLP (Neural Net)** | **GSI v6.3.3 (Đề Xuất)** | **0.5258** | 72.6% | 0.3213 | 0.1667 | 0.1452 | **0.6592** |

---
## 3. ĐỐI SÁNH SELECTIVE MACRO-F1 CHI TIẾT TRÊN 30 CẤU HÌNH
| Tập dữ liệu | Bộ học cơ sở | BR | CC | MLC-PA | GSI v6.2.1 | GSI v6.3.2 | GSI v6.3.3 (Đề Xuất) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `emotions` | Logistic Regression | 0.5972 | 0.5932 | 0.6172 | 0.6883 | **0.7499** | 0.7447 |
| `emotions` | Linear SVM (Platt) | 0.5726 | 0.5618 | 0.5670 | 0.5675 | **0.7454** | 0.7402 |
| `emotions` | MLP (Neural Net) | 0.4411 | 0.5082 | 0.3134 | 0.5257 | **0.7020** | 0.6946 |
| `scene` | Logistic Regression | 0.6994 | 0.7234 | 0.7431 | 0.7642 | **0.7968** | **0.7968** |
| `scene` | Linear SVM (Platt) | 0.5968 | 0.6776 | 0.5700 | 0.6077 | 0.7838 | **0.7839** |
| `scene` | MLP (Neural Net) | 0.6458 | 0.5972 | 0.6599 | 0.5399 | **0.7317** | **0.7317** |
| `yeast` | Logistic Regression | 0.3748 | 0.4017 | 0.3585 | 0.3562 | 0.4284 | **0.4638** |
| `yeast` | Linear SVM (Platt) | 0.3260 | 0.3829 | 0.2924 | 0.3059 | 0.4043 | **0.4382** |
| `yeast` | MLP (Neural Net) | 0.2841 | 0.3568 | 0.2525 | 0.2944 | 0.3792 | **0.4365** |
| `plantpseaac` | Logistic Regression | 0.1410 | 0.1715 | 0.0975 | 0.0951 | **0.2685** | 0.2676 |
| `plantpseaac` | Linear SVM (Platt) | 0.0557 | 0.1261 | 0.0149 | 0.0149 | **0.1857** | 0.1828 |
| `plantpseaac` | MLP (Neural Net) | 0.0207 | 0.1703 | 0.0000 | 0.1175 | **0.2560** | 0.2553 |
| `humanpseaac` | Logistic Regression | 0.1124 | 0.1397 | 0.0869 | 0.0869 | **0.2380** | 0.2374 |
| `humanpseaac` | Linear SVM (Platt) | 0.0138 | 0.0790 | 0.0014 | 0.0010 | **0.1662** | 0.1646 |
| `humanpseaac` | MLP (Neural Net) | 0.0065 | 0.1183 | 0.0008 | 0.0350 | **0.2110** | 0.2107 |
| `chd49` | Logistic Regression | 0.5103 | 0.5073 | **0.5242** | 0.5203 | 0.4672 | 0.5078 |
| `chd49` | Linear SVM (Platt) | 0.3830 | **0.4495** | 0.3009 | 0.3021 | 0.3613 | 0.3762 |
| `chd49` | MLP (Neural Net) | 0.4881 | 0.4968 | 0.4606 | **0.5056** | 0.4638 | 0.4788 |
| `music` | Logistic Regression | 0.6067 | 0.6031 | 0.6375 | 0.6679 | **0.7617** | 0.7515 |
| `music` | Linear SVM (Platt) | 0.5664 | 0.5668 | 0.5955 | 0.5931 | **0.7533** | 0.7489 |
| `music` | MLP (Neural Net) | 0.4453 | 0.5225 | 0.3923 | 0.4950 | **0.6954** | 0.6863 |
| `gpositivepseaac` | Logistic Regression | 0.5535 | 0.5791 | 0.5370 | 0.5415 | 0.6677 | **0.6706** |
| `gpositivepseaac` | Linear SVM (Platt) | 0.4748 | 0.5283 | 0.4640 | 0.4672 | 0.5824 | **0.5829** |
| `gpositivepseaac` | MLP (Neural Net) | 0.5283 | 0.5945 | 0.5742 | 0.5686 | **0.6864** | 0.6819 |
| `genbase` | Logistic Regression | 0.6782 | 0.6930 | 0.6334 | 0.6408 | **0.7562** | **0.7562** |
| `genbase` | Linear SVM (Platt) | 0.7616 | 0.7616 | 0.7468 | 0.7468 | **0.7672** | **0.7672** |
| `genbase` | MLP (Neural Net) | 0.0000 | 0.4397 | 0.0000 | 0.3791 | **0.6239** | **0.6239** |
| `viruspseaac` | Logistic Regression | 0.3692 | 0.3836 | 0.3543 | 0.3479 | 0.4584 | **0.4635** |
| `viruspseaac` | Linear SVM (Platt) | 0.2857 | 0.3378 | 0.2353 | 0.2321 | **0.3705** | 0.3374 |
| `viruspseaac` | MLP (Neural Net) | 0.3940 | 0.4236 | 0.3111 | 0.4105 | **0.4898** | 0.4586 |

---
## 4. HỆ THỐNG BIỂU ĐỒ METRIC ĐỐI SO SÁNH TRỰC QUAN
- **Hình 1 (Selective Macro-F1):** `figures/fig1_macro_f1_comparison.png`
- **Hình 2 (Đồ thị Pareto Coverage vs F1):** `figures/fig2_f1_vs_coverage_pareto.png`
- **Hình 3 (Subset 0/1 Accuracy):** `figures/fig3_subset_accuracy_comparison.png`
- **Hình 4 (Hamming Loss):** `figures/fig4_hamming_loss_comparison.png`
- **Hình 5 (Chuyên sâu CHD49 & VirusPseAAC):** `figures/fig5_focus_chd49_viruspseaac.png`
- **Hình 6 (Biểu đồ Radar đa mục tiêu):** `figures/fig6_radar_5criteria.png`