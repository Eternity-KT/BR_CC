# BÁO CÁO KHẢO SÁT: ĐỘ MẤT CÂN BẰNG CỦA CÁC NHÃN TRONG TẬP DL (DEPENDENT LABELS) Ở PHIÊN BẢN v6.2.1
**Tài liệu tham chiếu:** `meeting_summary.md` (Mục **core** & **option**), `spec/spec_v6_2.md`
**Phiên bản khảo sát:** GSI-MLC-PA v6.2.1 (`GSI_v6_2_Decay` - Hạ ngưỡng phân tầng IL: Tầng 1 $\tau=0.75$, Tầng 2 $\tau=0.70$, Tầng 3 $\tau=0.65$)
**Bộ phân loại cơ sở:** Logistic Regression, Calibrated SVM, Multi-Layer Perceptron (MLP)

---
## 1. Tổng Quan Về Tập DL & Tỷ Lệ Nhãn Mất Cân Bằng (Toàn Bộ 10 Tập Dữ Liệu)
Bảng dưới đây thống kê số lượng nhãn bị kẹt lại trong tập phụ thuộc $DL$ so với tập độc lập $IL$, cùng các chỉ số mất cân bằng (Imbalance Ratio - IR, tần suất nhãn hiếm < 5%):

| Tập Dữ Liệu | Tổng Nhãn $K$ | Nhãn $IL$ | Nhãn $DL$ | Tỷ lệ $DL$ (%) | Mean IR ($DL$) | Median IR ($DL$) | Tần suất TB ($DL$) | Số nhãn hiếm (<5%) trong $DL$ | Mean IR ($IL$) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **emotions** | 6 | 3 | 3 | 50.0% | 2.51 | 2.53 | 28.50% | 0/3 | 2.13 |
| **scene** | 6 | 4 | 2 | 33.3% | 4.05 | 4.05 | 20.02% | 0/2 | 4.97 |
| **chd49** | 6 | 2 | 4 | 66.7% | 9.64 | 1.89 | 30.27% | 1/4 | 2.37 |
| **music** | 6 | 3 | 3 | 50.0% | 2.51 | 2.54 | 28.49% | 0/3 | 2.12 |
| **gpositivepseaac** | 4 | 2 | 2 | 50.0% | 15.53 | 15.53 | 13.58% | 1/2 | 1.74 |
| **genbase** | 27 | 18 | 9 | 33.3% | 372.91 | 330.00 | 0.39% | 9/9 | 28.73 |
| **humanpseaac** | 14 | 0 | 14 | 100.0% | 45.51 | 28.30 | 8.47% | 7/14 | 0 nhãn IL |
| **plantpseaac** | 12 | 0 | 12 | 100.0% | 21.88 | 20.05 | 8.99% | 6/12 | 0 nhãn IL |
| **viruspseaac** | 6 | 1 | 5 | 83.3% | 5.36 | 5.27 | 23.57% | 0/5 | 24.88 |
| **yeast** | 14 | 2 | 12 | 85.7% | 9.95 | 3.54 | 22.85% | 1/12 | 2.97 |

---
## 2. Phân Tích Chi Tiết 3 Mô Hình Cơ Sở (Logistic, SVM, MLP)
Sự khác biệt về phân tách $IL$ và $DL$ giữa các bộ phân loại cơ sở:

| Base Learner | Dataset | $n_{IL}$ | $n_{DL}$ | % DL | Mean IR ($DL$) | Mean IR ($IL$) | Số nhãn < 5% trong DL |
|:---|:---|---:|---:|---:|---:|---:|---:|
| Logistic | emotions | 3 | 3 | 50.0% | 2.51 | 2.13 | 0 |
| Logistic | scene | 4 | 2 | 33.3% | 4.05 | 4.97 | 0 |
| Logistic | chd49 | 2 | 4 | 66.7% | 9.64 | 2.37 | 1 |
| Logistic | music | 3 | 3 | 50.0% | 2.51 | 2.12 | 0 |
| Logistic | gpositivepseaac | 2 | 2 | 50.0% | 15.53 | 1.74 | 1 |
| Logistic | genbase | 18 | 9 | 33.3% | 372.91 | 28.73 | 9 |
| Logistic | humanpseaac | 0 | 14 | 100.0% | 45.51 | N/A | 7 |
| Logistic | plantpseaac | 0 | 12 | 100.0% | 21.88 | N/A | 6 |
| Logistic | viruspseaac | 1 | 5 | 83.3% | 5.36 | 24.88 | 0 |
| Logistic | yeast | 2 | 12 | 85.7% | 9.95 | 2.97 | 1 |
| SVM | emotions | 3 | 3 | 50.0% | 2.51 | 2.13 | 0 |
| SVM | scene | 4 | 2 | 33.3% | 4.05 | 4.97 | 0 |
| SVM | chd49 | 2 | 4 | 66.7% | 9.64 | 2.37 | 1 |
| SVM | music | 3 | 3 | 50.0% | 2.51 | 2.12 | 0 |
| SVM | gpositivepseaac | 1 | 3 | 75.0% | 11.01 | 1.50 | 1 |
| SVM | genbase | 22 | 5 | 18.5% | 528.60 | 55.93 | 5 |
| SVM | humanpseaac | 0 | 14 | 100.0% | 45.51 | N/A | 7 |
| SVM | plantpseaac | 0 | 12 | 100.0% | 21.88 | N/A | 6 |
| SVM | viruspseaac | 1 | 5 | 83.3% | 5.36 | 24.88 | 0 |
| SVM | yeast | 2 | 12 | 85.7% | 9.95 | 2.97 | 1 |
| MLP | emotions | 1 | 5 | 83.3% | 2.18 | 3.01 | 0 |
| MLP | scene | 2 | 4 | 66.7% | 4.45 | 5.09 | 0 |
| MLP | chd49 | 2 | 4 | 66.7% | 9.64 | 2.37 | 1 |
| MLP | music | 0 | 6 | 100.0% | 2.32 | N/A | 0 |
| MLP | gpositivepseaac | 2 | 2 | 50.0% | 15.53 | 1.74 | 1 |
| MLP | genbase | 12 | 15 | 55.6% | 247.05 | 13.96 | 14 |
| MLP | humanpseaac | 0 | 14 | 100.0% | 45.51 | N/A | 7 |
| MLP | plantpseaac | 0 | 12 | 100.0% | 21.88 | N/A | 6 |
| MLP | viruspseaac | 1 | 5 | 83.3% | 5.36 | 24.88 | 0 |
| MLP | yeast | 2 | 12 | 85.7% | 9.95 | 2.97 | 1 |

---
## 3. Khảo Sát Chi Tiết Từng Nhãn Trong Tập $DL$ và Hiệu Năng BR F1
### 3.1. Nhóm Bị Kẹt 100% Trong DL: `humanpseaac` & `plantpseaac`
Hai tập dữ liệu này có đặc điểm chung: **Không có bất kỳ nhãn nào lọt được vào $IL$ ($n_{IL} = 0, n_{DL} = K$)** trên cả 3 mô hình cơ sở.

#### Chi tiết tập nhãn `humanpseaac` ($N = 14$ nhãn):
| Index | Tên Nhãn | Trạng thái v6.2.1 | Số mẫu Positive | Tần suất (%) | Imbalance Ratio (IR) | BR F1 Score |
|---:|:---|:---:|---:|---:|---:|---:|
| 10 | `Nucleus` | **DL** | 1021 | 32.87% | 2.04 | 0.4800 |
| 1 | `Cytoplasm` | **DL** | 817 | 26.30% | 2.80 | 0.2201 |
| 5 | `Extracell` | **DL** | 385 | 12.40% | 7.07 | 0.3270 |
| 9 | `Mitochondrion` | **DL** | 364 | 11.72% | 7.53 | 0.3550 |
| 12 | `Plasma_membrane` | **DL** | 354 | 11.40% | 7.77 | 0.3034 |
| 4 | `Endoplasmic_reticulum` | **DL** | 229 | 7.37% | 12.56 | 0.1574 |
| 6 | `Golgi_apparatus` | **DL** | 161 | 5.18% | 18.29 | 0.0412 |
| 2 | `Cytoskeleton` | **DL** | 79 | 2.54% | 38.32 | 0.0225 |
| 7 | `Lysosome` | **DL** | 77 | 2.48% | 39.34 | 0.0435 |
| 0 | `Centriole` | **DL** | 77 | 2.48% | 39.34 | 0.0220 |
| 11 | `Peroxisome` | **DL** | 47 | 1.51% | 65.09 | 0.0000 |
| 3 | `Endosome` | **DL** | 24 | 0.77% | 128.42 | 0.0000 |
| 8 | `Microsome` | **DL** | 24 | 0.77% | 128.42 | 0.0000 |
| 13 | `Synapse` | **DL** | 22 | 0.71% | 140.18 | 0.0000 |

#### Chi tiết tập nhãn `plantpseaac` ($N = 12$ nhãn):
| Index | Tên Nhãn | Trạng thái v6.2.1 | Số mẫu Positive | Tần suất (%) | Imbalance Ratio (IR) | BR F1 Score |
|---:|:---|:---:|---:|---:|---:|---:|
| 2 | `Chloroplast` | **DL** | 286 | 29.24% | 2.42 | 0.4500 |
| 3 | `Cytoplasm` | **DL** | 182 | 18.61% | 4.37 | 0.2206 |
| 8 | `Nucleus` | **DL** | 152 | 15.54% | 5.43 | 0.4833 |
| 7 | `Mitochondrion` | **DL** | 150 | 15.34% | 5.52 | 0.1834 |
| 0 | `Cell_membrane` | **DL** | 56 | 5.73% | 16.46 | 0.1220 |
| 11 | `Vacuole` | **DL** | 52 | 5.32% | 17.81 | 0.1111 |
| 4 | `Endoplasmic_reticulum` | **DL** | 42 | 4.29% | 22.29 | 0.0800 |
| 10 | `Plastid` | **DL** | 39 | 3.99% | 24.08 | 0.4444 |
| 1 | `Cell_wall` | **DL** | 32 | 3.27% | 29.56 | 0.1860 |
| 5 | `Extracell` | **DL** | 22 | 2.25% | 43.45 | 0.0000 |
| 6 | `Golgi_apparatus` | **DL** | 21 | 2.15% | 45.57 | 0.0000 |
| 9 | `Peroxisome` | **DL** | 21 | 2.15% | 45.57 | 0.0000 |

### 3.2. Nhóm Mất Cân Bằng Cực Đoan: `genbase`
`genbase` có 27 nhãn, trong đó các nhãn phân bố thành 2 cực rõ rệt:

| Index | Tên Nhãn | Trạng thái v6.2.1 | Số mẫu Positive | Tần suất (%) | Imbalance Ratio (IR) | BR F1 Score |
|---:|:---|:---:|---:|---:|---:|---:|
| 4 | `PDOC00791` | **IL** | 171 | 25.83% | 2.87 | 0.9971 |
| 0 | `PDOC00154` | **IL** | 79 | 11.93% | 7.38 | 1.0000 |
| 1 | `PDOC00343` | **IL** | 76 | 11.48% | 7.71 | 1.0000 |
| 9 | `PDOC00670` | **IL** | 66 | 9.97% | 9.03 | 0.9924 |
| 2 | `PDOC00271` | **IL** | 62 | 9.37% | 9.68 | 1.0000 |
| 7 | `PDOC00224` | **IL** | 51 | 7.70% | 11.98 | 1.0000 |
| 3 | `PDOC00064` | **IL** | 49 | 7.40% | 12.51 | 0.9684 |
| 17 | `PDOC00662` | **IL** | 41 | 6.19% | 15.15 | 1.0000 |
| 12 | `PDOC00561` | **IL** | 36 | 5.44% | 17.39 | 0.9859 |
| 10 | `PDOC50002` | **IL** | 33 | 4.98% | 19.06 | 0.9851 |
| 6 | `PDOC50007` | **IL** | 31 | 4.68% | 20.35 | 0.8727 |
| 11 | `PDOC50106` | **IL** | 29 | 4.38% | 21.83 | 1.0000 |
| 5 | `PDOC00380` | **IL** | 23 | 3.47% | 27.78 | 1.0000 |
| 16 | `PDOC50156` | **IL** | 17 | 2.57% | 37.94 | 0.9697 |
| 14 | `PDOC50003` | **IL** | 14 | 2.11% | 46.29 | 0.8800 |
| 13 | `PDOC50017` | **IL** | 14 | 2.11% | 46.29 | 0.8800 |
| 18 | `PDOC00018` | **IL** | 9 | 1.36% | 72.56 | 0.8750 |
| 8 | `PDOC00100` | **DL** | 6 | 0.91% | 109.33 | 0.5000 |
| 19 | `PDOC50001` | **IL** | 5 | 0.76% | 131.40 | 0.8889 |
| 15 | `PDOC50006` | **DL** | 4 | 0.60% | 164.50 | 0.0000 |
| 21 | `PDOC00750` | **DL** | 3 | 0.45% | 219.67 | 0.0000 |
| 26 | `PDOC00030` | **DL** | 3 | 0.45% | 219.67 | 0.0000 |
| 20 | `PDOC00014` | **DL** | 2 | 0.30% | 330.00 | 0.0000 |
| 22 | `PDOC50196` | **DL** | 2 | 0.30% | 330.00 | 0.0000 |
| 23 | `PDOC50199` | **DL** | 1 | 0.15% | 661.00 | 0.0000 |
| 24 | `PDOC00660` | **DL** | 1 | 0.15% | 661.00 | 0.0000 |
| 25 | `PDOC00653` | **DL** | 1 | 0.15% | 661.00 | 0.0000 |

### 3.3. Các Tập Dữ Liệu Còn Lại: `yeast`, `chd49`, `gpositivepseaac`, `viruspseaac`, `scene`, `emotions`, `music`

#### Tập `yeast`:
| Index | Tên Nhãn | Trạng thái v6.2.1 | Số mẫu Positive | Tần suất (%) | Imbalance Ratio (IR) | BR F1 Score |
|---:|:---|:---:|---:|---:|---:|---:|
| 11 | `Class12` | **IL** | 1816 | 75.13% | 3.02 | 0.8532 |
| 12 | `Class13` | **IL** | 1799 | 74.43% | 2.91 | 0.8469 |
| 1 | `Class2` | **DL** | 1038 | 42.95% | 1.33 | 0.5150 |
| 2 | `Class3` | **DL** | 983 | 40.67% | 1.46 | 0.6586 |
| 3 | `Class4` | **DL** | 862 | 35.66% | 1.80 | 0.5999 |
| 0 | `Class1` | **DL** | 762 | 31.53% | 2.17 | 0.5737 |
| 4 | `Class5` | **DL** | 722 | 29.87% | 2.35 | 0.4870 |
| 5 | `Class6` | **DL** | 597 | 24.70% | 3.05 | 0.2560 |
| 7 | `Class8` | **DL** | 480 | 19.86% | 4.04 | 0.0121 |
| 6 | `Class7` | **DL** | 428 | 17.71% | 4.65 | 0.0491 |
| 10 | `Class11` | **DL** | 289 | 11.96% | 7.36 | 0.0205 |
| 9 | `Class10` | **DL** | 253 | 10.47% | 8.55 | 0.0079 |
| 8 | `Class9` | **DL** | 178 | 7.36% | 12.58 | 0.0000 |
| 13 | `Class14` | **DL** | 34 | 1.41% | 70.09 | 0.0000 |

#### Tập `chd49`:
| Index | Tên Nhãn | Trạng thái v6.2.1 | Số mẫu Positive | Tần suất (%) | Imbalance Ratio (IR) | BR F1 Score |
|---:|:---|:---:|---:|---:|---:|---:|
| 5 | `L6` | **IL** | 422 | 76.04% | 3.17 | 0.8514 |
| 0 | `L1` | **IL** | 338 | 60.90% | 1.56 | 0.7203 |
| 4 | `L5` | **DL** | 268 | 48.29% | 1.07 | 0.5465 |
| 2 | `L3` | **DL** | 214 | 38.56% | 1.59 | 0.4329 |
| 1 | `L2` | **DL** | 174 | 31.35% | 2.19 | 0.4365 |
| 3 | `L4` | **DL** | 16 | 2.88% | 33.69 | 0.0833 |

#### Tập `gpositivepseaac`:
| Index | Tên Nhãn | Trạng thái v6.2.1 | Số mẫu Positive | Tần suất (%) | Imbalance Ratio (IR) | BR F1 Score |
|---:|:---|:---:|---:|---:|---:|---:|
| 2 | `Cytoplasm` | **IL** | 208 | 40.08% | 1.50 | 0.7711 |
| 0 | `Cell_membrane` | **IL** | 174 | 33.53% | 1.98 | 0.6810 |
| 3 | `Extracell` | **DL** | 123 | 23.70% | 3.22 | 0.5860 |
| 1 | `Cell_wall` | **DL** | 18 | 3.47% | 27.83 | 0.3077 |

#### Tập `viruspseaac`:
| Index | Tên Nhãn | Trạng thái v6.2.1 | Số mẫu Positive | Tần suất (%) | Imbalance Ratio (IR) | BR F1 Score |
|---:|:---|:---:|---:|---:|---:|---:|
| 3 | `Host_cytoplasm` | **DL** | 87 | 42.03% | 1.38 | 0.3774 |
| 4 | `Host_nucleus` | **DL** | 84 | 40.58% | 1.46 | 0.5132 |
| 1 | `Host_cell_membrane` | **DL** | 33 | 15.94% | 5.27 | 0.2182 |
| 2 | `Host_endoplasm_reticulum` | **DL** | 20 | 9.66% | 9.35 | 0.1481 |
| 5 | `Secreted` | **DL** | 20 | 9.66% | 9.35 | 0.0800 |
| 0 | `Viral_capsid` | **IL** | 8 | 3.86% | 24.88 | 0.9412 |

#### Tập `scene`:
| Index | Tên Nhãn | Trạng thái v6.2.1 | Số mẫu Positive | Tần suất (%) | Imbalance Ratio (IR) | BR F1 Score |
|---:|:---|:---:|---:|---:|---:|---:|
| 4 | `mountain` | **DL** | 533 | 22.14% | 3.52 | 0.5447 |
| 3 | `field` | **IL** | 433 | 17.99% | 4.56 | 0.8182 |
| 5 | `urban` | **DL** | 431 | 17.91% | 4.58 | 0.5929 |
| 0 | `beach` | **IL** | 427 | 17.74% | 4.64 | 0.6838 |
| 2 | `foliage` | **IL** | 397 | 16.49% | 5.06 | 0.7239 |
| 1 | `sunset` | **IL** | 364 | 15.12% | 5.61 | 0.8639 |

#### Tập `emotions`:
| Index | Tên Nhãn | Trạng thái v6.2.1 | Số mẫu Positive | Tần suất (%) | Imbalance Ratio (IR) | BR F1 Score |
|---:|:---|:---:|---:|---:|---:|---:|
| 2 | `relaxing-calm` | **IL** | 264 | 44.52% | 1.25 | 0.7110 |
| 5 | `angry-aggresive` | **IL** | 189 | 31.87% | 2.14 | 0.6793 |
| 0 | `amazed-suprised` | **DL** | 173 | 29.17% | 2.43 | 0.6070 |
| 4 | `sad-lonely` | **DL** | 168 | 28.33% | 2.53 | 0.6132 |
| 1 | `happy-pleased` | **DL** | 166 | 27.99% | 2.57 | 0.3065 |
| 3 | `quiet-still` | **IL** | 148 | 24.96% | 3.01 | 0.7887 |

#### Tập `music`:
| Index | Tên Nhãn | Trạng thái v6.2.1 | Số mẫu Positive | Tần suất (%) | Imbalance Ratio (IR) | BR F1 Score |
|---:|:---|:---:|---:|---:|---:|---:|
| 2 | `relaxing-clam` | **IL** | 264 | 44.59% | 1.24 | 0.7280 |
| 5 | `angry-aggresive` | **IL** | 189 | 31.93% | 2.13 | 0.7248 |
| 0 | `amazed-suprised` | **DL** | 173 | 29.22% | 2.42 | 0.5987 |
| 4 | `sad-lonely` | **DL** | 167 | 28.21% | 2.54 | 0.6014 |
| 1 | `happy-pleased` | **DL** | 166 | 28.04% | 2.57 | 0.1881 |
| 3 | `quiet-still` | **IL** | 148 | 25.00% | 3.00 | 0.8043 |

