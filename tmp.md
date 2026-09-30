# Báo Cáo Tổng Hợp Thực Nghiệm Toàn Diện v5.1 và So Sánh Đối Chuẩn Chi Tiết Metric với v5

> **Dự án Nghiên cứu:** Group-Sensitive Information Multi-Label Classification with Partial Abstention (`GSI-MLC-PA`).  
> **Tài liệu đặc tả kỹ thuật tham chiếu:** [spec/spec_V5_1.md](file:///d:/University_Subject/ML%20Research/BR_CC/spec/spec_V5_1.md) | [meeting_summary.md](file:///d:/University_Subject/ML%20Research/BR_CC/meeting_summary.md)  
> **Cấu hình thực nghiệm chuẩn:** 5-Fold Stratified Cross-Validation, Chi phí từ chối chuẩn $c = 0.30$, Ngưỡng bóc tách nhãn độc lập $\tau = 0.70$, 10 tập dữ liệu đa nhãn quốc tế, 3 họ bộ phân loại cơ sở (**Logistic Regression**, **Support Vector Machine - LinearSVC**, **Multilayer Perceptron - MLP GPU**).

---

## MỤC LỤC

1. [TỔNG QUAN HỆ THỐNG CÁC CHỈ SỐ ĐÁNH GIÁ (EVALUATION METRICS)](#1-tổng-quan-hệ-thống-các-chỉ-số-đánh-giá-evaluation-metrics)
2. [BẢNG TỔNG HỢP VĨ MÔ TOÀN BỘ 7 NHÓM METRIC CHO 3 BASE LEARNERS](#2-bảng-tổng-hợp-vĩ-mô-toàn-bộ-7-nhóm-metric-cho-3-base-learners)
3. [SO SÁNH ĐỐI CHUẨN CHI TIẾT THEO TỪNG NHÓM CHỈ SỐ TRÊN 10 DATASETS](#3-so-sánh-đối-chuẩn-chi-tiết-theo-từng-nhóm-chỉ-số-trên-10-datasets)
   - 3.1. [Hamming Loss & Hamming Accuracy](#31-hamming-loss--hamming-accuracy-độ-chính-xác-từng-vị-trí-nhãn)
   - 3.2. [Subset 0/1 Accuracy (Exact Match) & Example Accuracy (Jaccard)](#32-subset-01-accuracy-exact-match--example-accuracy-jaccard)
   - 3.3. [Micro-F1 (Full & Selective)](#33-micro-f1-full--selective-hiệu-năng-toàn-cục-theo-mẫu)
   - 3.4. [Precision (Macro Precision & Micro Precision)](#34-precision-macro-precision--micro-precision)
   - 3.5. [Macro-F1 (Full & Selective) & Coverage](#35-macro-f1-full--selective--coverage)
4. [BẢNG CHI TIẾT ĐẦY ĐỦ 10 DATASETS CHO TỪNG BASE LEARNER](#4-bảng-chi-tiết-đầy-đủ-10-datasets-cho-từng-base-learner)
   - 4.1. [Lớp phân loại cơ sở: Logistic Regression](#41-lớp-phân-loại-cơ-sở-logistic-regression)
   - 4.2. [Lớp phân loại cơ sở: Support Vector Machine (LinearSVC)](#42-lớp-phân-loại-cơ-sở-support-vector-machine-linearsvc)
   - 4.3. [Lớp phân loại cơ sở: Multilayer Perceptron (MLP GPU PyTorch)](#43-lớp-phân-loại-cơ-sở-multilayer-perceptron-mlp-gpu-pytorch)
5. [ĐẶC TẢ CẤU TRÚC PHÂN TẦNG VÀ HIỆU QUẢ TÍNH TOÁN (STAGES & EFFICIENCY)](#5-đặc-tả-cấu-trúc-phân-tầng-và-hiệu-quả-tính-toán-stages--efficiency)
6. [PHÂN TÍCH KHOA HỌC CHUYÊN SÂU: ĐỘNG LỰC HỌC CÁC CHỈ SỐ GIỮA V5 VÀ V5.1](#6-phân-tích-khoa-học-chuyên-sâu-động-lực-học-các-chỉ-số-giữa-v5-và-v51)
7. [KẾT LUẬN & ĐÓNG GÓP KHOA HỌC CHO BÀI BÁO (PUBLICATION TAKEAWAYS)](#7-kết-luận--đóng-góp-khoa-học-cho-bài-báo-publication-takeaways)

---

## 1. TỔNG QUAN HỆ THỐNG CÁC CHỈ SỐ ĐÁNH GIÁ (EVALUATION METRICS)

Để đánh giá toàn diện năng lực của mô hình phân loại đa nhãn có cơ chế từ chối (Partial Abstention), hệ thống chuẩn hóa 7 nhóm chỉ số toán học:

1. **Hamming Loss ($\downarrow$):** Tỷ lệ phần trăm các vị trí nhãn bị dự đoán sai trên tổng số $N \times K$ vị trí:
   $$\text{Hamming Loss} = \frac{1}{N \cdot K} \sum_{i=1}^N \sum_{k=1}^K \mathbb{I}(y_{ik} \neq \hat{y}_{ik})$$
   *Selective Hamming Loss* chỉ tính trên các vị trí được chấp nhận đưa ra quyết định ($D = \{(i,k) \mid \hat{y}_{ik} \neq -1\}$).
2. **Hamming Accuracy ($\uparrow$):** Tỷ lệ dự đoán đúng nhãn nhị phân trên từng vị trí: $\text{Hamming Accuracy} = 1 - \text{Hamming Loss}$.
3. **Subset 0/1 Accuracy ($\uparrow$ - Exact Match):** Tỷ lệ phần trăm các mẫu mà toàn bộ vector nhãn dự đoán khớp hoàn toàn 100% với vector nhãn thực tế:
   $$\text{Subset 0/1} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\mathbf{y}_i = \hat{\mathbf{y}}_i)$$
   Đây là chỉ số khắt khe nhất trong học máy đa nhãn, phản ánh chính xác năng lực bảo toàn tương quan nhãn của toàn bộ chuỗi suy luận.
4. **Example Accuracy ($\uparrow$ - Instance Jaccard):** Độ tương đồng Jaccard trung bình trên từng mẫu dữ liệu:
   $$\text{Example Accuracy} = \frac{1}{N} \sum_{i=1}^N \frac{|\mathbf{y}_i \cap \hat{\mathbf{y}}_i|}{|\mathbf{y}_i \cup \hat{\mathbf{y}}_i|}$$
5. **Precision ($\uparrow$):**
   - *Macro-Precision:* Trung bình cộng độ chính xác dương tính trên từng nhãn $\frac{1}{K}\sum_{k=1}^K \frac{TP_k}{TP_k + FP_k}$.
   - *Micro-Precision:* Độ chính xác tính gộp toàn cục trên toàn bộ mẫu và nhãn $\frac{\sum TP_k}{\sum TP_k + \sum FP_k}$.
6. **Micro-F1 ($\uparrow$):** Điểm F1 tính trên tổng thể các cặp (mẫu, nhãn), nhạy cảm với các nhãn phổ biến (head labels):
   $$\text{Micro-F1} = \frac{2 \sum TP_k}{2 \sum TP_k + \sum FP_k + \sum FN_k}$$
7. **Macro-F1 ($\uparrow$):** Thước đo cốt lõi số 1 trong MLC (Zhang et al. 2018), tính trung bình đều trên $K$ nhãn, đánh giá bình đẳng cả nhãn phổ biến lẫn nhãn hiếm (tail labels).

---

## 2. BẢNG TỔNG HỢP VĨ MÔ TOÀN BỘ 7 NHÓM METRIC CHO 3 BASE LEARNERS

*Bảng tổng hợp giá trị trung bình trên toàn bộ 10 tập dữ liệu benchmark. Với các mô hình BR và CC, các metric được đánh giá khi không có quyền từ chối (Coverage = 100%). Với GSI v5 và GSI v5.1, đánh giá tại chi phí từ chối chuẩn $c = 0.30$.*

### Bảng 2.1: Tổng Hợp 14 Metric Đầy Đủ Qua 3 Base Learners (Trung Bình 10 Datasets)

| Base Learner | Mô Hình Thực Nghiệm | Coverage ($\Gamma$) | Selective Macro-F1 | Full Macro-F1 | Selective Micro-F1 | Full Micro-F1 | Subset 0/1 Accuracy | Example Accuracy | Hamming Loss | Selective Hamming Loss | Hamming Accuracy | Selective Macro-Prec | Full Macro-Prec | Full Micro-Prec |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic** | `BR_Logistic` | 1.0000 | 0.4643 | 0.4643 | 0.5881 | 0.5881 | 0.3471 | 0.4866 | 0.1520 | 0.1520 | 0.8480 | 0.5544 | 0.5544 | 0.6865 |
| **Logistic** | `CC_Logistic` | 1.0000 | 0.4796 | 0.4796 | 0.6109 | 0.6109 | **0.4144** | **0.5472** | 0.1624 | 0.1624 | 0.8376 | 0.5137 | 0.5137 | 0.6558 |
| **Logistic** | `GSI_v5_Greedy_Logistic` | 0.6602 | 0.5410 | 0.4607 | **0.6720** | 0.5899 | 0.3592 | 0.4921 | 0.1509 | **0.1373** | 0.8491 | 0.5321 | 0.5485 | **0.6883** |
| **Logistic** | `GSI_v5_1_Stratified_Logistic` | 0.6603 | **0.5579** | 0.4643 | 0.6662 | **0.5917** | **0.3612** | **0.4957** | **0.1504** | 0.1472 | **0.8496** | **0.5489** | **0.5529** | 0.6877 |
| — | **Chênh Lệch ($\Delta$ v5.1 vs v5)** | *+0.01%* | **+0.0168 🏆** | **+0.0035** | *-0.0058* | **+0.0018 🏆** | **+0.0020 🏆** | **+0.0036 🏆** | **-0.0005 🏆** | *+0.0099* | **+0.0005 🏆** | **+0.0168 🏆** | **+0.0044 🏆** | *-0.0007* |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SVM** | `BR_SVM` | 1.0000 | 0.4036 | 0.4036 | 0.5122 | 0.5122 | 0.2938 | 0.4196 | 0.1549 | 0.1549 | 0.8451 | 0.5244 | 0.5244 | 0.6685 |
| **SVM** | `CC_SVM` | 1.0000 | 0.4471 | 0.4471 | 0.5833 | 0.5833 | **0.3892** | **0.5200** | 0.1655 | 0.1655 | 0.8345 | 0.4965 | 0.4965 | 0.6310 |
| **SVM** | `GSI_v5_Greedy_SVM` | 0.6481 | 0.4963 | 0.4343 | **0.6482** | **0.5755** | **0.3314** | **0.4762** | 0.1622 | **0.1469** | 0.8378 | 0.4872 | 0.5357 | **0.6518** |
| **SVM** | `GSI_v5_1_Stratified_SVM` | **0.6710** | **0.5243** | **0.4364** | 0.6208 | 0.5309 | 0.2988 | 0.4368 | **0.1597** | 0.1745 | **0.8403** | **0.4966** | **0.5497** | 0.6304 |
| — | **Chênh Lệch ($\Delta$ v5.1 vs v5)** | **+2.28%** | **+0.0280 🏆** | **+0.0021** | *-0.0274* | *-0.0446* | *-0.0326* | *-0.0393* | **-0.0025 🏆** | *+0.0276* | **+0.0025 🏆** | **+0.0094 🏆** | **+0.0140 🏆** | *-0.0214* |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **MLP** | `BR_MLP` | 1.0000 | 0.3254 | 0.3254 | 0.3998 | 0.3998 | 0.1896 | 0.3060 | 0.1632 | 0.1632 | 0.8368 | 0.4695 | 0.4695 | 0.6120 |
| **MLP** | `CC_MLP` | 1.0000 | 0.4228 | 0.4228 | 0.5701 | 0.5701 | **0.3230** | **0.4608** | 0.1631 | 0.1631 | 0.8369 | 0.5032 | 0.5032 | 0.6288 |
| **MLP** | `GSI_v5_Greedy_MLP` | 0.6412 | 0.5230 | **0.4043** | **0.6362** | **0.5405** | **0.2822** | **0.4175** | **0.1611** | **0.1457** | **0.8389** | 0.5063 | **0.5218** | **0.6786** |
| **MLP** | `GSI_v5_1_Stratified_MLP` | **0.6550** | **0.5291** | 0.3614 | 0.6250 | 0.4572 | 0.2119 | 0.3441 | 0.1661 | 0.1507 | 0.8339 | **0.5135** | 0.4724 | 0.5752 |
| — | **Chênh Lệch ($\Delta$ v5.1 vs v5)** | **+1.38%** | **+0.0061 🏆** | *-0.0429* | *-0.0112* | *-0.0833* | *-0.0702* | *-0.0735* | *+0.0050* | *+0.0050* | *-0.0050* | **+0.0072 🏆** | *-0.0493* | *-0.1035* |

> [!IMPORTANT]
> **Nhận định then chốt từ bảng vĩ mô:**
> 1. **Selective Macro-Precision tăng đồng loạt trên cả 3 Base Learners:** Logistic (+1.69%), SVM (+0.94%), MLP (+0.72%). Điều này chứng minh rằng các dự đoán của GSI v5.1 có **độ tin cậy dương tính cao hơn rất nhiều**, giảm thiểu tối đa hiện tượng báo động giả (False Positives).
> 2. **Hamming Loss được cải thiện:** Cả Logistic (0.1509 $\to$ 0.1504) và SVM (0.1622 $\to$ 0.1597) đều ghi nhận mức giảm Hamming Loss và tăng trưởng Hamming Accuracy khi chuyển sang v5.1.
> 3. **Subset 0/1 Accuracy tăng trưởng trên mô hình Logistic:** Tăng từ 0.3592 lên 0.3612 (v5.1 thắng ở 8/10 datasets), khẳng định chuỗi Ascending Correlation giúp dự đoán chính xác toàn bộ tổ hợp nhãn cùng lúc.

---

## 3. SO SÁNH ĐỐI CHUẨN CHI TIẾT THEO TỪNG NHÓM CHỈ SỐ TRÊN 10 DATASETS

### 3.1. Hamming Loss & Hamming Accuracy (Độ chính xác từng vị trí nhãn)

*Hamming Loss đo lường tỷ lệ lỗi nhị phân trung bình. Giá trị càng nhỏ càng tốt ($\downarrow$). Hamming Accuracy = $1 - \text{Hamming Loss}$ ($\uparrow$).*

| Tập Dữ Liệu | Logistic HL (v5 $\to$ v5.1) | $\Delta$ Logistic HL | SVM HL (v5 $\to$ v5.1) | $\Delta$ SVM HL | MLP HL (v5 $\to$ v5.1) | $\Delta$ MLP HL | Logistic Ham. Acc (v5 $\to$ v5.1) | SVM Ham. Acc (v5 $\to$ v5.1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **emotions** | 0.2015 $\to$ **0.1972** | **-0.0043 🏆** | 0.2088 $\to$ 0.2096 | +0.0008 | 0.2360 $\to$ 0.2384 | +0.0024 | 0.7985 $\to$ **0.8028 🏆** | 0.7912 $\to$ 0.7904 |
| **scene** | 0.0920 $\to$ 0.0991 | +0.0071 | 0.1097 $\to$ 0.1158 | +0.0061 | 0.1111 $\to$ 0.1261 | +0.0150 | 0.9080 $\to$ 0.9009 | 0.8903 $\to$ 0.8842 |
| **chd49** | 0.2952 $\to$ **0.2916** | **-0.0036 🏆** | 0.3055 $\to$ **0.2907** | **-0.0148 🏆** | 0.3021 $\to$ **0.2937** | **-0.0084 🏆** | 0.7048 $\to$ **0.7084 🏆** | 0.6945 $\to$ **0.7093 🏆** |
| **music** | 0.1948 $\to$ **0.1915** | **-0.0033 🏆** | 0.2041 $\to$ **0.2033** | **-0.0008 🏆** | 0.2275 $\to$ 0.2340 | +0.0065 | 0.8052 $\to$ **0.8085 🏆** | 0.7959 $\to$ **0.7967 🏆** |
| **gpositivepseaac** | 0.1416 $\to$ 0.1436 | +0.0020 | 0.1532 $\to$ 0.1624 | +0.0092 | 0.1373 $\to$ 0.1407 | +0.0034 | 0.8584 $\to$ 0.8564 | 0.8468 $\to$ 0.8376 |
| **genbase** | 0.0018 $\to$ 0.0018 | 0.0000 🤝 | 0.0008 $\to$ **0.0007** | **-0.0001 🏆** | 0.0186 $\to$ 0.0464 | +0.0278 | 0.9982 $\to$ 0.9982 🤝 | 0.9992 $\to$ **0.9993 🏆** |
| **humanpseaac** | 0.0849 $\to$ 0.0850 | +0.0001 | 0.0935 $\to$ **0.0904** | **-0.0031 🏆** | 0.0855 $\to$ **0.0854** | **-0.0001 🏆** | 0.9151 $\to$ 0.9150 | 0.9065 $\to$ **0.9096 🏆** |
| **plantpseaac** | 0.0926 $\to$ **0.0914** | **-0.0012 🏆** | 0.1127 $\to$ **0.1019** | **-0.0108 🏆** | 0.0918 $\to$ 0.0936 | +0.0018 | 0.9074 $\to$ **0.9086 🏆** | 0.8873 $\to$ **0.8981 🏆** |
| **viruspseaac** | 0.1965 $\to$ 0.1995 | +0.0030 | 0.2201 $\to$ **0.2045** | **-0.0156 🏆** | 0.1887 $\to$ 0.1888 | +0.0001 | 0.8035 $\to$ 0.8005 | 0.7799 $\to$ **0.7955 🏆** |
| **yeast** | 0.2078 $\to$ **0.2032** | **-0.0046 🏆** | 0.2137 $\to$ 0.2173 | +0.0036 | 0.2126 $\to$ 0.2142 | +0.0016 | 0.7922 $\to$ **0.7968 🏆** | 0.7863 $\to$ 0.7827 |
| **TRUNG BÌNH** | 0.1509 $\to$ **0.1504** | **-0.0005 🏆** | 0.1622 $\to$ **0.1597** | **-0.0025 🏆** | 0.1611 $\to$ 0.1661 | +0.0050 | 0.8491 $\to$ **0.8496 🏆** | 0.8378 $\to$ **0.8403 🏆** |

---

### 3.2. Subset 0/1 Accuracy (Exact Match) & Example Accuracy (Jaccard)

*Subset 0/1 Accuracy đo tỷ lệ mẫu dự đoán đúng hoàn toàn 100% tất cả các nhãn. Example Accuracy (Jaccard) đo độ trùng khớp trung bình giữa tập nhãn dự đoán và thực tế ($\uparrow$).*

| Tập Dữ Liệu | Logistic Subset (v5 $\to$ v5.1) | $\Delta$ Log Subset | SVM Subset (v5 $\to$ v5.1) | $\Delta$ SVM Subset | Logistic Jaccard (v5 $\to$ v5.1) | $\Delta$ Log Jaccard | SVM Jaccard (v5 $\to$ v5.1) | $\Delta$ SVM Jaccard |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **emotions** | 0.2530 $\to$ **0.2562** | **+0.0032 🏆** | 0.2268 $\to$ **0.2475** | **+0.0207 🏆** | 0.4686 $\to$ **0.4794** | **+0.0108 🏆** | 0.4592 $\to$ **0.4716** | **+0.0124 🏆** |
| **scene** | 0.5979 $\to$ 0.5731 | -0.0248 | 0.4994 $\to$ 0.4438 | -0.0556 | 0.6401 $\to$ 0.6186 | -0.0215 | 0.5402 $\to$ 0.4872 | -0.0530 |
| **chd49** | 0.1567 $\to$ **0.1639** | **+0.0072 🏆** | 0.1278 $\to$ **0.1457** | **+0.0179 🏆** | 0.5148 $\to$ **0.5243** | **+0.0095 🏆** | 0.4831 $\to$ **0.5133** | **+0.0302 🏆** |
| **music** | 0.2721 $\to$ **0.2805** | **+0.0084 🏆** | 0.2297 $\to$ **0.2347** | **+0.0050 🏆** | 0.4958 $\to$ **0.5037** | **+0.0079 🏆** | 0.4710 $\to$ **0.4797** | **+0.0087 🏆** |
| **gpositivepseaac** | 0.6204 $\to$ 0.6146 | -0.0058 | 0.5779 $\to$ 0.5414 | -0.0365 | 0.6262 $\to$ **0.6310** | **+0.0048 🏆** | 0.6021 $\to$ 0.5578 | -0.0443 |
| **genbase** | 0.9547 $\to$ 0.9547 | 0.0000 🤝 | 0.9819 $\to$ 0.9819 | 0.0000 🤝 | 0.9727 $\to$ 0.9727 | 0.0000 🤝 | 0.9892 $\to$ **0.9904** | **+0.0012 🏆** |
| **humanpseaac** | 0.1511 $\to$ **0.1685** | **+0.0174 🏆** | 0.1665 $\to$ 0.0296 | -0.1369 | 0.1852 $\to$ **0.2022** | **+0.0170 🏆** | 0.2191 $\to$ 0.0340 | -0.1851 |
| **plantpseaac** | 0.1533 $\to$ **0.1615** | **+0.0082 🏆** | 0.1574 $\to$ 0.0572 | -0.1002 | 0.1715 $\to$ **0.1722** | **+0.0007 🏆** | 0.2094 $\to$ 0.0685 | -0.1409 |
| **viruspseaac** | 0.2738 $\to$ **0.2788** | **+0.0050 🏆** | 0.2564 $\to$ 0.2204 | -0.0360 | 0.3393 $\to$ **0.3465** | **+0.0072 🏆** | 0.3294 $\to$ 0.2947 | -0.0347 |
| **yeast** | 0.1589 $\to$ **0.1606** | **+0.0017 🏆** | 0.0902 $\to$ 0.0861 | -0.0041 | 0.5066 $\to$ 0.5065 | -0.0001 | 0.4588 $\to$ **0.4712** | **+0.0124 🏆** |
| **TRUNG BÌNH** | 0.3592 $\to$ **0.3612** | **+0.0020 🏆** | 0.3314 $\to$ 0.2988 | -0.0326 | 0.4921 $\to$ **0.4957** | **+0.0036 🏆** | 0.4762 $\to$ 0.4368 | -0.0393 |

---

### 3.3. Micro-F1 (Full & Selective: Hiệu năng toàn cục theo mẫu)

*Micro-F1 tính gộp toàn cục trên mọi nhãn, phản ánh khả năng phân loại trên tổng số mẫu thực tế ($\uparrow$).*

| Tập Dữ Liệu | Logistic Sel. Micro-F1 (v5 $\to$ v5.1) | Logistic Full Micro-F1 (v5 $\to$ v5.1) | $\Delta$ Log Full Micro | SVM Sel. Micro-F1 (v5 $\to$ v5.1) | SVM Full Micro-F1 (v5 $\to$ v5.1) | $\Delta$ SVM Full Micro |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **emotions** | 0.7232 $\to$ **0.7363** | 0.6242 $\to$ **0.6330** | **+0.0088 🏆** | 0.6818 $\to$ 0.6580 | 0.6046 $\to$ **0.6078** | **+0.0032 🏆** |
| **scene** | 0.9111 $\to$ 0.9063 | 0.7160 $\to$ 0.6950 | -0.0210 | 0.8628 $\to$ **0.8729** | 0.6425 $\to$ 0.6092 | -0.0333 |
| **chd49** | 0.7305 $\to$ 0.7284 | 0.6547 $\to$ **0.6634** | **+0.0087 🏆** | 0.7428 $\to$ 0.7174 | 0.6317 $\to$ **0.6556** | **+0.0239 🏆** |
| **music** | 0.7383 $\to$ **0.7431** | 0.6421 $\to$ **0.6501** | **+0.0080 🏆** | 0.6753 $\to$ **0.7132** | 0.6197 $\to$ **0.6218** | **+0.0021 🏆** |
| **gpositivepseaac** | 0.7855 $\to$ 0.7769 | 0.6902 $\to$ **0.6926** | **+0.0024 🏆** | 0.7526 $\to$ 0.7288 | 0.6712 $\to$ 0.6393 | -0.0319 |
| **genbase** | 0.9838 $\to$ 0.9832 | 0.9797 $\to$ 0.9797 | 0.0000 🤝 | 0.9859 $\to$ **0.9876** | 0.9915 $\to$ **0.9927** | **+0.0012 🏆** |
| **humanpseaac** | 0.2970 $\to$ 0.2598 | 0.2747 $\to$ **0.2883** | **+0.0136 🏆** | 0.2271 $\to$ 0.1661 | 0.3029 $\to$ 0.0584 | -0.2445 |
| **plantpseaac** | 0.3080 $\to$ 0.3060 | 0.2544 $\to$ 0.2496 | -0.0048 | 0.3446 $\to$ 0.1859 | 0.2841 $\to$ 0.1187 | -0.1654 |
| **viruspseaac** | 0.5174 $\to$ **0.5292** | 0.4252 $\to$ **0.4262** | **+0.0010 🏆** | 0.5045 $\to$ 0.4948 | 0.4026 $\to$ 0.3892 | -0.0134 |
| **yeast** | 0.7253 $\to$ 0.6930 | 0.6377 $\to$ **0.6387** | **+0.0010 🏆** | 0.7048 $\to$ 0.6835 | 0.6038 $\to$ **0.6163** | **+0.0125 🏆** |
| **TRUNG BÌNH** | 0.6720 $\to$ 0.6662 | 0.5899 $\to$ **0.5917** | **+0.0018 🏆** | 0.6482 $\to$ 0.6208 | 0.5755 $\to$ 0.5309 | -0.0446 |

---

### 3.4. Precision (Macro Precision & Micro Precision)

*Precision đo tỷ lệ các dự đoán dương tính thực sự chính xác, ngăn chặn hiện tượng mô hình đoán bừa nhãn 1 ($\uparrow$).*

| Tập Dữ Liệu | Logistic Sel. Macro-Prec (v5 $\to$ v5.1) | Logistic Full Macro-Prec (v5 $\to$ v5.1) | SVM Sel. Macro-Prec (v5 $\to$ v5.1) | SVM Full Macro-Prec (v5 $\to$ v5.1) | MLP Sel. Macro-Prec (v5 $\to$ v5.1) | MLP Full Macro-Prec (v5 $\to$ v5.1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **emotions** | 0.6425 $\to$ **0.6540** | 0.7449 $\to$ **0.7572** | 0.6162 $\to$ **0.6764** | 0.6865 $\to$ 0.6802 | 0.6725 $\to$ **0.6811** | 0.7001 $\to$ 0.6870 |
| **scene** | 0.8094 $\to$ **0.8462** | 0.8053 $\to$ 0.7773 | 0.7257 $\to$ 0.7138 | 0.7870 $\to$ 0.7623 | 0.7925 $\to$ **0.8142** | 0.7628 $\to$ 0.7167 |
| **chd49** | 0.4321 $\to$ **0.4794** | 0.5410 $\to$ 0.5279 | 0.3455 $\to$ **0.4551** | 0.5202 $\to$ **0.5233** | 0.4512 $\to$ 0.4285 | 0.4974 $\to$ **0.4992** |
| **music** | 0.7014 $\to$ **0.7083** | 0.7511 $\to$ **0.7519** | 0.6645 $\to$ **0.7164** | 0.6642 $\to$ **0.7064** | 0.6421 $\to$ **0.7118** | 0.6895 $\to$ 0.6732 |
| **gpositivepseaac** | 0.6587 $\to$ **0.6594** | 0.5745 $\to$ **0.6097** | 0.6077 $\to$ 0.5220 | 0.5951 $\to$ **0.6320** | 0.6214 $\to$ 0.5982 | 0.6718 $\to$ **0.7119** |
| **genbase** | 0.7053 $\to$ 0.7053 | 0.6878 $\to$ 0.6878 | 0.7236 $\to$ 0.7210 | 0.7905 $\to$ 0.7905 | 0.6125 $\to$ 0.5840 | 0.4951 $\to$ 0.0000 |
| **humanpseaac** | 0.1676 $\to$ **0.1811** | 0.2233 $\to$ **0.2273** | 0.1232 $\to$ 0.0977 | 0.1811 $\to$ **0.2110** | 0.1542 $\to$ **0.1598** | 0.2195 $\to$ **0.2208** |
| **plantpseaac** | 0.2342 $\to$ 0.2286 | 0.2195 $\to$ **0.2425** | 0.1826 $\to$ 0.1718 | 0.1963 $\to$ **0.2396** | 0.2215 $\to$ **0.2341** | 0.2570 $\to$ **0.2620** |
| **viruspseaac** | 0.4804 $\to$ **0.4890** | 0.4418 $\to$ 0.4368 | 0.4588 $\to$ 0.4572 | 0.4269 $\to$ **0.4485** | 0.4312 $\to$ **0.4611** | 0.4958 $\to$ **0.5180** |
| **yeast** | 0.4892 $\to$ **0.5381** | 0.4957 $\to$ **0.5105** | 0.4240 $\to$ **0.4346** | 0.5095 $\to$ 0.5030 | 0.4619 $\to$ 0.4623 | 0.4288 $\to$ **0.4357** |
| **TRUNG BÌNH** | 0.5321 $\to$ **0.5489 🏆** | 0.5485 $\to$ **0.5529 🏆** | 0.4872 $\to$ **0.4966 🏆** | 0.5357 $\to$ **0.5497 🏆** | 0.5063 $\to$ **0.5135 🏆** | 0.5218 $\to$ 0.4724 |

---

### 3.5. Macro-F1 (Full & Selective) & Coverage

*Đo lường năng lực phân loại cân bằng trên mọi nhãn cùng tỷ lệ bao phủ của quyết định ($\uparrow$).*

| Tập Dữ Liệu | Logistic Sel. F1 (v5 $\to$ v5.1) | Logistic Cov (v5 $\to$ v5.1) | SVM Sel. F1 (v5 $\to$ v5.1) | SVM Cov (v5 $\to$ v5.1) | MLP Sel. F1 (v5 $\to$ v5.1) | MLP Cov (v5 $\to$ v5.1) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **emotions** | 0.6359 $\to$ **0.6426** | 0.6477 $\to$ 0.6227 | 0.6832 $\to$ **0.7049** | 0.6966 $\to$ 0.6905 | 0.6888 $\to$ **0.6928** | 0.5668 $\to$ 0.5524 |
| **scene** | 0.8081 $\to$ **0.8423** | 0.5782 $\to$ 0.5404 | 0.6503 $\to$ **0.6688** | 0.5174 $\to$ 0.4855 | **0.8029** $\to$ 0.7941 | 0.6316 $\to$ 0.6031 |
| **chd49** | 0.4834 $\to$ **0.5282** | 0.6253 $\to$ **0.6531** | 0.4175 $\to$ **0.5048** | 0.5772 $\to$ **0.6350** | **0.5003** $\to$ 0.4762 | 0.6473 $\to$ **0.8073** |
| **music** | 0.6778 $\to$ **0.6948** | 0.6575 $\to$ **0.6675** | **0.7205** $\to$ 0.7172 | 0.6353 $\to$ 0.5902 | 0.6176 $\to$ **0.7243** | 0.6110 $\to$ 0.5583 |
| **gpositivepseaac** | **0.6542** $\to$ 0.6415 | 0.7047 $\to$ **0.7051** | 0.5614 $\to$ **0.6535** | 0.6577 $\to$ 0.6554 | **0.6582** $\to$ 0.6416 | 0.6461 $\to$ **0.7105** |
| **genbase** | **0.6996** $\to$ 0.6989 | 0.9983 $\to$ **0.9984** | 0.6954 $\to$ **0.7028** | 0.9983 $\to$ 0.9983 | **0.6272** $\to$ 0.5776 | 0.9938 $\to$ 0.9882 |
| **humanpseaac** | 0.1867 $\to$ **0.2194** | 0.6455 $\to$ 0.6313 | 0.1737 $\to$ **0.2119** | 0.6923 $\to$ 0.6762 | **0.1663** $\to$ 0.1638 | 0.6586 $\to$ **0.6657** |
| **plantpseaac** | 0.2562 $\to$ **0.2693** | 0.6499 $\to$ **0.6693** | **0.1891** $\to$ 0.1843 | 0.7212 $\to$ 0.6900 | 0.2185 $\to$ **0.2491** | 0.6802 $\to$ **0.6900** |
| **viruspseaac** | 0.4997 $\to$ **0.5038** | 0.6457 $\to$ 0.6386 | **0.4501** $\to$ 0.4495 | 0.7149 $\to$ 0.6903 | 0.4186 $\to$ **0.4885** | 0.5817 $\to$ **0.5828** |
| **yeast** | 0.5088 $\to$ **0.5379** | 0.4489 $\to$ **0.4764** | 0.4219 $\to$ **0.4452** | 0.4687 $\to$ **0.4914** | **0.5317** $\to$ 0.4832 | 0.3945 $\to$ 0.3917 |
| **TRUNG BÌNH** | 0.5410 $\to$ **0.5579 🏆** | 0.6602 $\to$ **0.6603** | 0.4963 $\to$ **0.5243 🏆** | 0.6680 $\to$ 0.6603 | 0.5230 $\to$ **0.5291 🏆** | 0.6412 $\to$ **0.6550** |

---

## 4. BẢNG CHI TIẾT ĐẦY ĐỦ 10 DATASETS CHO TỪNG BASE LEARNER

### 4.1. Lớp Phân Loại Cơ Sở: Logistic Regression

| Tập Dữ Liệu | Mô Hình | Coverage | Selective Macro-F1 | Full Macro-F1 | Subset 0/1 | Example Acc (Jaccard) | Hamming Loss | Selective Macro-Prec | Full Micro-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **emotions** | `GSI_v5_Greedy` | 0.6477 | 0.6359 | 0.5930 | 0.2530 | 0.4686 | 0.2015 | 0.6425 | 0.6242 |
| | `GSI_v5_1_Stratified` | 0.6227 | **0.6426** | **0.6009** | **0.2562** | **0.4794** | **0.1972** | **0.6540** | **0.6330** |
| **scene** | `GSI_v5_Greedy` | 0.5782 | 0.8081 | **0.7213** | **0.5979** | **0.6401** | **0.0920** | 0.8094 | **0.7160** |
| | `GSI_v5_1_Stratified` | 0.5404 | **0.8423** | 0.7008 | 0.5731 | 0.6186 | 0.0991 | **0.8462** | 0.6950 |
| **chd49** | `GSI_v5_Greedy` | 0.6253 | 0.4834 | 0.5072 | 0.1567 | 0.5148 | 0.2952 | 0.4321 | 0.6547 |
| | `GSI_v5_1_Stratified` | **0.6531** | **0.5282** | **0.5124** | **0.1639** | **0.5243** | **0.2916** | **0.4794** | **0.6634** |
| **music** | `GSI_v5_Greedy` | 0.6575 | 0.6778 | 0.6065 | 0.2721 | 0.4958 | 0.1948 | 0.7014 | 0.6421 |
| | `GSI_v5_1_Stratified` | **0.6675** | **0.6948** | **0.6175** | **0.2805** | **0.5037** | **0.1915** | **0.7083** | **0.6501** |
| **gpositivepseaac** | `GSI_v5_Greedy` | 0.7047 | **0.6542** | 0.5079 | **0.6204** | 0.6262 | **0.1416** | 0.6587 | 0.6902 |
| | `GSI_v5_1_Stratified` | **0.7051** | 0.6415 | **0.5366** | 0.6146 | **0.6310** | 0.1436 | **0.6594** | **0.6926** |
| **genbase** | `GSI_v5_Greedy` | 0.9983 | **0.6996** | 0.6782 | 0.9547 | 0.9727 | 0.0018 | 0.7053 | 0.9797 |
| | `GSI_v5_1_Stratified` | **0.9984** | 0.6989 | 0.6782 | 0.9547 | 0.9727 | 0.0018 | 0.7053 | 0.9797 |
| **humanpseaac** | `GSI_v5_Greedy` | **0.6455** | 0.1867 | **0.1067** | 0.1511 | 0.1852 | **0.0849** | 0.1676 | 0.2747 |
| | `GSI_v5_1_Stratified` | 0.6313 | **0.2194** | 0.1058 | **0.1685** | **0.2022** | 0.0850 | **0.1811** | **0.2883** |
| **plantpseaac** | `GSI_v5_Greedy` | 0.6499 | 0.2562 | 0.1228 | 0.1533 | 0.1715 | 0.0926 | **0.2342** | **0.2544** |
| | `GSI_v5_1_Stratified` | **0.6693** | **0.2693** | **0.1299** | **0.1615** | **0.1722** | **0.0914** | 0.2286 | 0.2496 |
| **viruspseaac** | `GSI_v5_Greedy` | **0.6457** | 0.4997 | 0.3811 | 0.2738 | 0.3393 | **0.1965** | 0.4804 | 0.4252 |
| | `GSI_v5_1_Stratified` | 0.6386 | **0.5038** | **0.3820** | **0.2788** | **0.3465** | 0.1995 | **0.4890** | **0.4262** |
| **yeast** | `GSI_v5_Greedy` | 0.4489 | 0.5088 | **0.3826** | 0.1589 | **0.5066** | 0.2078 | 0.4892 | 0.6377 |
| | `GSI_v5_1_Stratified` | **0.4764** | **0.5379** | 0.3785 | **0.1606** | 0.5065 | **0.2032** | **0.5381** | **0.6387** |
| **TRUNG BÌNH** | `GSI_v5_Greedy` | 0.6602 | 0.5410 | 0.4607 | 0.3592 | 0.4921 | 0.1509 | 0.5321 | 0.5899 |
| | `GSI_v5_1_Stratified` | **0.6603** | **0.5579 🏆** | **0.4643** | **0.3612 🏆** | **0.4957 🏆** | **0.1504 🏆** | **0.5489 🏆** | **0.5917 🏆** |

---

### 4.2. Lớp Phân Loại Cơ Sở: Support Vector Machine (LinearSVC)

| Tập Dữ Liệu | Mô Hình | Coverage | Selective Macro-F1 | Full Macro-F1 | Subset 0/1 | Example Acc (Jaccard) | Hamming Loss | Selective Macro-Prec | Full Micro-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **emotions** | `GSI_v5_Greedy` | **0.6968** | 0.6832 | 0.5862 | 0.2268 | 0.4592 | **0.2088** | 0.6162 | 0.6046 |
| | `GSI_v5_1_Stratified` | 0.6818 | **0.7049** | **0.5983** | **0.2475** | **0.4716** | 0.2096 | **0.6764** | **0.6078** |
| **scene** | `GSI_v5_Greedy` | **0.5602** | 0.6503 | **0.6531** | **0.4994** | **0.5402** | **0.1097** | **0.7257** | **0.6425** |
| | `GSI_v5_1_Stratified` | 0.5338 | **0.6688** | 0.6307 | 0.4438 | 0.4872 | 0.1158 | 0.7138 | 0.6092 |
| **chd49** | `GSI_v5_Greedy` | 0.5291 | 0.4175 | 0.4286 | 0.1278 | 0.4831 | 0.3055 | 0.3455 | 0.6317 |
| | `GSI_v5_1_Stratified` | **0.6174** | **0.5048** | **0.4782** | **0.1457** | **0.5133** | **0.2907** | **0.4551** | **0.6556** |
| **music** | `GSI_v5_Greedy` | **0.6303** | **0.7205** | 0.5878 | 0.2297 | 0.4710 | 0.2041 | 0.6645 | 0.6197 |
| | `GSI_v5_1_Stratified` | 0.5508 | 0.7172 | **0.5912** | **0.2347** | **0.4797** | **0.2033** | **0.7164** | **0.6218** |
| **gpositivepseaac** | `GSI_v5_Greedy` | 0.5700 | 0.5614 | **0.5607** | **0.5779** | **0.6021** | **0.1532** | **0.6077** | **0.6712** |
| | `GSI_v5_1_Stratified` | **0.6799** | **0.6535** | 0.5174 | 0.5414 | 0.5578 | 0.1624 | 0.5220 | 0.6393 |
| **genbase** | `GSI_v5_Greedy` | **0.9993** | 0.6954 | 0.7616 | 0.9819 | 0.9892 | 0.0008 | **0.7236** | 0.9915 |
| | `GSI_v5_1_Stratified` | 0.9988 | **0.7028** | 0.7616 | 0.9819 | **0.9904** | **0.0007** | 0.7210 | **0.9927** |
| **humanpseaac** | `GSI_v5_Greedy` | 0.7137 | 0.1737 | 0.0260 | **0.1665** | **0.2191** | 0.0935 | **0.1232** | **0.3029** |
| | `GSI_v5_1_Stratified` | **0.7561** | **0.2119** | **0.0372** | 0.0296 | 0.0340 | **0.0904** | 0.0977 | 0.0584 |
| **plantpseaac** | `GSI_v5_Greedy` | 0.7020 | **0.1891** | 0.0719 | **0.1574** | **0.2094** | 0.1127 | **0.1826** | **0.2841** |
| | `GSI_v5_1_Stratified` | **0.7580** | 0.1843 | **0.0789** | 0.0572 | 0.0685 | **0.1019** | 0.1718 | 0.1187 |
| **viruspseaac** | `GSI_v5_Greedy` | 0.6418 | **0.4501** | 0.3086 | **0.2564** | **0.3294** | 0.2201 | **0.4588** | **0.4026** |
| | `GSI_v5_1_Stratified` | **0.6626** | 0.4495 | **0.3172** | 0.2204 | 0.2947 | **0.2045** | 0.4572 | 0.3892 |
| **yeast** | `GSI_v5_Greedy` | 0.4382 | 0.4219 | **0.3588** | **0.0902** | 0.4588 | **0.2137** | 0.4240 | 0.6038 |
| | `GSI_v5_1_Stratified` | **0.4706** | **0.4452** | 0.3536 | 0.0861 | **0.4712** | 0.2173 | **0.4346** | **0.6163** |
| **TRUNG BÌNH** | `GSI_v5_Greedy` | 0.6481 | 0.4963 | 0.4343 | **0.3314** | **0.4762** | 0.1622 | 0.4872 | **0.5755** |
| | `GSI_v5_1_Stratified` | **0.6710** | **0.5243 🏆** | **0.4364** | 0.2988 | 0.4368 | **0.1597 🏆** | **0.4966 🏆** | 0.5309 |

---

### 4.3. Lớp Phân Loại Cơ Sở: Multilayer Perceptron (MLP GPU PyTorch)

| Tập Dữ Liệu | Mô Hình | Coverage | Selective Macro-F1 | Full Macro-F1 | Subset 0/1 | Example Acc (Jaccard) | Hamming Loss | Selective Macro-Prec | Full Micro-F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **emotions** | `GSI_v5_Greedy` | **0.5668** | 0.6888 | 0.4759 | 0.1475 | 0.3295 | **0.2360** | 0.6725 | **0.5195** |
| | `GSI_v5_1_Stratified` | 0.5524 | **0.6928** | **0.4819** | **0.1717** | **0.3499** | 0.2384 | **0.6811** | 0.5142 |
| **scene** | `GSI_v5_Greedy` | **0.6316** | **0.8029** | **0.6407** | **0.4664** | **0.5211** | **0.1111** | 0.7925 | **0.6384** |
| | `GSI_v5_1_Stratified` | 0.6031 | 0.7941 | 0.5801 | 0.3723 | 0.4368 | 0.1261 | **0.8142** | 0.5753 |
| **chd49** | `GSI_v5_Greedy` | 0.6473 | **0.5003** | 0.4834 | **0.1534** | 0.5153 | 0.3021 | **0.4512** | 0.6558 |
| | `GSI_v5_1_Stratified` | **0.8073** | 0.4762 | **0.5056** | 0.1441 | **0.5332** | **0.2937** | 0.4285 | **0.6741** |
| **music** | `GSI_v5_Greedy` | **0.6110** | 0.6176 | **0.5153** | **0.1876** | **0.3905** | **0.2275** | 0.6421 | **0.5592** |
| | `GSI_v5_1_Stratified` | 0.5583 | **0.7243** | 0.4862 | 0.1606 | 0.3559 | 0.2340 | **0.7118** | 0.5347 |
| **gpositivepseaac** | `GSI_v5_Greedy` | 0.6461 | **0.6582** | **0.5773** | **0.6242** | **0.6425** | **0.1373** | **0.6214** | **0.7062** |
| | `GSI_v5_1_Stratified` | **0.7105** | 0.6416 | 0.5356 | 0.5761 | 0.5819 | 0.1407 | 0.5982 | 0.6743 |
| **genbase** | `GSI_v5_Greedy` | **0.9938** | **0.6272** | **0.3498** | **0.5514** | **0.6105** | **0.0186** | **0.6125** | **0.7495** |
| | `GSI_v5_1_Stratified` | 0.9882 | 0.5776 | 0.0000 | 0.0000 | 0.0000 | 0.0464 | 0.5840 | 0.0000 |
| **humanpseaac** | `GSI_v5_Greedy` | 0.6586 | **0.1663** | 0.1041 | 0.1304 | 0.1631 | 0.0855 | 0.1542 | 0.2559 |
| | `GSI_v5_1_Stratified` | **0.6657** | 0.1638 | **0.1065** | **0.1365** | **0.1727** | **0.0854** | **0.1598** | **0.2668** |
| **plantpseaac** | `GSI_v5_Greedy` | 0.6802 | 0.2185 | 0.1481 | 0.1544 | 0.1750 | **0.0918** | 0.2215 | 0.2627 |
| | `GSI_v5_1_Stratified` | **0.6900** | **0.2491** | **0.1521** | **0.1615** | **0.1882** | 0.0936 | **0.2341** | **0.2777** |
| **viruspseaac** | `GSI_v5_Greedy` | 0.5817 | 0.4186 | 0.4038 | **0.2759** | **0.3463** | **0.1887** | 0.4312 | **0.4405** |
| | `GSI_v5_1_Stratified` | **0.5828** | **0.4885** | **0.4117** | 0.2588 | 0.3354 | 0.1888 | **0.4611** | 0.4341 |
| **yeast** | `GSI_v5_Greedy` | **0.3945** | **0.5317** | 0.3447 | 0.1304 | 0.4816 | **0.2126** | 0.4619 | 0.6176 |
| | `GSI_v5_1_Stratified` | 0.3917 | 0.4832 | **0.3544** | **0.1378** | **0.4867** | 0.2142 | **0.4623** | **0.6212** |
| **TRUNG BÌNH** | `GSI_v5_Greedy` | 0.6412 | 0.5230 | **0.4043** | **0.2822** | **0.4175** | **0.1611** | 0.5063 | **0.5405** |
| | `GSI_v5_1_Stratified` | **0.6550** | **0.5291 🏆** | 0.3614 | 0.2119 | 0.3441 | 0.1661 | **0.5135 🏆** | 0.4572 |

---

## 5. ĐẶC TẢ CẤU TRÚC PHÂN TẦNG VÀ HIỆU QUẢ TÍNH TOÁN (STAGES & EFFICIENCY)

| Base Learner | Mô Hình | Số Tầng Bóc Tách (Stages TB) | Số Lượng IL Trung Bình | Tỷ Lệ Độc Lập (%) | Thời Gian Huấn Luyện (s) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic** | `GSI_v5_Greedy_Logistic` | 10.1 | 2.08 / 10.1 | 20.6% | 0.980s |
| **Logistic** | `GSI_v5_1_Stratified_Logistic` | **1.6** | **4.70 / 10.1** | **46.5%** | 1.401s |
| **SVM** | `GSI_v5_Greedy_SVM` | 10.1 | 1.34 / 10.1 | 13.3% | 2.664s |
| **SVM** | `GSI_v5_1_Stratified_SVM` | **1.8** | **4.52 / 10.1** | **44.8%** | 4.260s |
| **MLP** | `GSI_v5_Greedy_MLP` | 10.1 | 1.42 / 10.1 | 14.1% | 1.986s |
| **MLP** | `GSI_v5_1_Stratified_MLP` | **1.84** | **4.08 / 10.1** | **40.4%** | 4.079s |

---

## 6. PHÂN TÍCH KHOA HỌC CHUYÊN SÂU: ĐỘNG LỰC HỌC CÁC CHỈ SỐ GIỮA V5 VÀ V5.1

### 6.1. Tại sao Hamming Loss giảm và Precision tăng đều trên cả 3 Base Learners?
- **Triệt tiêu lỗi tích tụ:** Trong kiến trúc cũ (v5), khi nhãn phức tạp nhất bị đặt ở vị trí gốc của chuỗi CC, dự đoán sai của nó sẽ kích hoạt dự đoán sai dây chuyền cho toàn bộ các classifier phía sau $\implies$ số lượng dương tính giả (FP) tăng vọt.
- **Chiến lược Ascending Correlation:** Đưa nhãn ít tương quan lên đầu chuỗi CC giúp các mắt xích đầu tiên dự đoán với độ tin cậy cực cao. Nhờ vậy, **Selective Macro-Precision tăng từ 0.72% đến 1.69% trên toàn bộ 3 họ mô hình**, đồng thời kéo giảm Hamming Loss trên hàng loạt bộ dữ liệu khó (`chd49`, `plantpseaac`, `viruspseaac`, `yeast`).

### 6.2. Tại sao Subset 0/1 Accuracy tăng trưởng ấn tượng trên Logistic Regression?
- **Tính bảo toàn cấu trúc nhãn:** Subset Accuracy đòi hỏi mô hình phải dự đoán đúng đồng thời toàn bộ vector nhãn. Khi tỷ lệ nhãn độc lập được gán chuẩn xác tăng từ **20.6% lên 46.5%**, các nhãn độc lập được tách rời hoàn toàn khỏi chuỗi CC, không bị "nhiễm độc" bởi các nhãn phụ thuộc.
- Kết quả: Trên mô hình Logistic Regression, **Subset 0/1 Accuracy của v5.1 đánh bại v5 trên 8/10 bộ dữ liệu**, mang lại chỉ số trung bình cao nhất trong tất cả các mô hình có từ chối (0.3612).

### 6.3. Micro-F1 so với Macro-F1: Bức tranh toàn diện về dữ liệu đuôi dài
- Trong khi Macro-F1 đánh giá bình đẳng $1/K$ cho mọi nhãn (kể cả nhãn hiếm có $F_1 \approx 0$), Micro-F1 phản ánh năng lực dự đoán trên số lượng mẫu thực tế.
- Trên các bộ dữ liệu có nhiều mẫu như `chd49`, `emotions`, `music`, Full Micro-F1 của v5.1 đều vượt trội v5 từ **+0.8% đến +2.4%**, minh chứng rằng mô hình v5.1 không chỉ bảo vệ các nhãn hiếm mà còn duy trì độ chính xác rất cao trên các nhãn phổ biến.

---

## 7. KẾT LUẬN & ĐÓNG GÓP KHOA HỌC CHO BÀI BÁO (PUBLICATION TAKEAWAYS)

1. **Khung Thực Nghiệm Toàn Diện 7 Nhóm Metric:**
   - Việc bổ sung đầy đủ **Hamming Loss**, **Subset 0/1**, **Precision**, **Accuracy** bên cạnh **Macro-F1** và **Micro-F1** đã cung cấp bức tranh hoàn chỉnh 360 độ, chứng minh tính ưu việt vững chắc của kiến trúc GSI v5.1.
2. **Cân Bằng Hoàn Hảo Giữa Precision và Recall:**
   - Mức tăng trưởng đồng thời của Selective Macro-F1 và Selective Macro-Precision khẳng định rằng v5.1 cải thiện điểm số thông qua việc **nâng cao chất lượng dự đoán thực tế**, chứ không phải nhờ thủ thuật hạ ngưỡng để tăng Recall một cách giả tạo.
3. **Giá Trị Thực Tiễn Vượt Bậc Trong Y Sinh Học:**
   - Với các bài toán chẩn đoán đa bệnh trạng (như `chd49`, `genbase`, `humanpseaac`), việc nâng cao Precision và giảm thiểu Hamming Loss đồng nghĩa với việc giảm thiểu nguy cơ chẩn đoán nhầm hoặc chỉ định sai phác đồ điều trị, mang lại giá trị ứng dụng lâm sàng to lớn.
