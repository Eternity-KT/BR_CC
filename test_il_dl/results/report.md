# Báo cáo thực nghiệm: Chained GSI vs. Bipartite GSI (DL chỉ phụ thuộc IL)

- **Mô hình cơ sở (Base Learner):** `mlp`
- **Chiến lược kiểm định:** 5-Fold Multilabel Stratified Cross-Validation
- **Hai mô hình đối chiếu:**
  1. `GSI_Chained` (Mô hình gốc): DL phụ thuộc cả IL và các nhãn DL trước nó trong chuỗi CC.
  2. `GSI_Bipartite` (Mô hình mới): DL CHỈ phụ thuộc vào tập IL, độc lập với các nhãn DL khác.

---

## 1. Kết quả tổng hợp tại Dự đoán Đầy đủ (Full, Ngưỡng 0.5, Coverage = 100%)

| Tập dữ liệu | Mô hình | Complete Macro-F1 (Mean ± Std) | Số nhãn IL trung bình | Số nhãn DL trung bình | Thời gian Train (s) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `emotions` | **GSI_Bipartite** | 0.5536 ± 0.0312 | 6.0 | 0.0 | 0.18s |
| `emotions` | **GSI_Chained** | 0.5229 ± 0.0802 | 4.2 | 1.8 | 1.11s |
| `enron` | **GSI_Bipartite** | 0.1513 ± 0.0061 | 32.8 | 20.2 | 15.77s |
| `enron` | **GSI_Chained** | 0.1917 ± 0.0165 | 8.6 | 44.4 | 3.96s |
| `music` | **GSI_Bipartite** | 0.6215 ± 0.0526 | 6.0 | 0.0 | 0.15s |
| `music` | **GSI_Chained** | 0.6282 ± 0.0251 | 2.6 | 3.4 | 0.48s |
| `scene` | **GSI_Bipartite** | 0.6230 ± 0.0108 | 6.0 | 0.0 | 0.61s |
| `scene` | **GSI_Chained** | 0.6504 ± 0.0138 | 3.2 | 2.8 | 0.62s |
| `yeast` | **GSI_Bipartite** | 0.4152 ± 0.0178 | 13.4 | 0.6 | 0.89s |
| `yeast` | **GSI_Chained** | 0.3977 ± 0.0134 | 5.0 | 9.0 | 1.19s |

---

## 2. Kết quả tại Chi phí từ chối chuẩn (c = 0.3)

| Tập dữ liệu | Mô hình | Selective Macro-F1 (Mean ± Std) | Coverage (Mean ± Std) | Thời gian Suy diễn (s) |
| :--- | :--- | :---: | :---: | :---: |
| `emotions` | **GSI_Bipartite** | 0.4571 ± 0.1039 | 0.1355 ± 0.0339 | 0.0005s |
| `emotions` | **GSI_Chained** | 0.3578 ± 0.1825 | 0.1120 ± 0.0581 | 0.0014s |
| `enron` | **GSI_Bipartite** | 0.0912 ± 0.0156 | 0.0077 ± 0.0029 | 0.0091s |
| `enron` | **GSI_Chained** | 0.1874 ± 0.0129 | 0.7051 ± 0.0408 | 0.0237s |
| `music` | **GSI_Bipartite** | 0.3747 ± 0.1682 | 0.0321 ± 0.0154 | 0.0007s |
| `music` | **GSI_Chained** | 0.4714 ± 0.2391 | 0.0589 ± 0.0114 | 0.0018s |
| `scene` | **GSI_Bipartite** | 0.8305 ± 0.1004 | 0.1263 ± 0.0218 | 0.0009s |
| `scene` | **GSI_Chained** | 0.8070 ± 0.0787 | 0.3593 ± 0.0378 | 0.0030s |
| `yeast` | **GSI_Bipartite** | 0.0143 ± 0.0319 | 0.0001 ± 0.0003 | 0.0014s |
| `yeast` | **GSI_Chained** | 0.0948 ± 0.0607 | 0.0221 ± 0.0136 | 0.0051s |

---

## 3. Bảng so sánh trực tiếp (Head-to-Head Comparison: Bipartite vs. Chained)

| Tập dữ liệu | Metric | GSI_Chained | GSI_Bipartite | Chênh lệch (Δ Bipartite - Chained) |
| :--- | :--- | :---: | :---: | :---: |
| `emotions` | Complete Macro-F1 | 0.5229 | 0.5536 | **+0.0306** |
| `emotions` | Selective Macro-F1 (c=0.3) | 0.3578 | 0.4571 | **+0.0993** |
| `emotions` | Coverage (c=0.3) | 0.1120 | 0.1355 | **+0.0235** |
| `enron` | Complete Macro-F1 | 0.1917 | 0.1513 | **-0.0404** |
| `enron` | Selective Macro-F1 (c=0.3) | 0.1874 | 0.0912 | **-0.0961** |
| `enron` | Coverage (c=0.3) | 0.7051 | 0.0077 | **-0.6974** |
| `music` | Complete Macro-F1 | 0.6282 | 0.6215 | **-0.0067** |
| `music` | Selective Macro-F1 (c=0.3) | 0.4714 | 0.3747 | **-0.0967** |
| `music` | Coverage (c=0.3) | 0.0589 | 0.0321 | **-0.0268** |
| `scene` | Complete Macro-F1 | 0.6504 | 0.6230 | **-0.0275** |
| `scene` | Selective Macro-F1 (c=0.3) | 0.8070 | 0.8305 | **+0.0235** |
| `scene` | Coverage (c=0.3) | 0.3593 | 0.1263 | **-0.2330** |
| `yeast` | Complete Macro-F1 | 0.3977 | 0.4152 | **+0.0175** |
| `yeast` | Selective Macro-F1 (c=0.3) | 0.0948 | 0.0143 | **-0.0805** |
| `yeast` | Coverage (c=0.3) | 0.0221 | 0.0001 | **-0.0220** |

---

## 4. Chi tiết qua các mức chi phí từ chối c ∈ [0.2, 0.4]

| Tập dữ liệu | Chi phí c | Mô hình | Selective Macro-F1 | Coverage |
| :--- | :---: | :--- | :---: | :---: |
| `emotions` | 0.2 | GSI_Bipartite | 0.2785 ± 0.0785 | 0.0273 ± 0.0036 |
| `emotions` | 0.2 | GSI_Chained | 0.1834 ± 0.0789 | 0.0198 ± 0.0108 |
| `emotions` | 0.25 | GSI_Bipartite | 0.4010 ± 0.1268 | 0.0663 ± 0.0149 |
| `emotions` | 0.25 | GSI_Chained | 0.3299 ± 0.1493 | 0.0525 ± 0.0294 |
| `emotions` | 0.3 | GSI_Bipartite | 0.4571 ± 0.1039 | 0.1355 ± 0.0339 |
| `emotions` | 0.3 | GSI_Chained | 0.3578 ± 0.1825 | 0.1120 ± 0.0581 |
| `emotions` | 0.35 | GSI_Bipartite | 0.6113 ± 0.0408 | 0.2562 ± 0.0563 |
| `emotions` | 0.35 | GSI_Chained | 0.5331 ± 0.1077 | 0.2238 ± 0.0934 |
| `emotions` | 0.4 | GSI_Bipartite | 0.6133 ± 0.0195 | 0.4450 ± 0.0414 |
| `emotions` | 0.4 | GSI_Chained | 0.5579 ± 0.1241 | 0.4173 ± 0.1018 |
| `enron` | 0.2 | GSI_Bipartite | 0.0208 ± 0.0144 | 0.0009 ± 0.0008 |
| `enron` | 0.2 | GSI_Chained | 0.1594 ± 0.0129 | 0.5791 ± 0.0330 |
| `enron` | 0.25 | GSI_Bipartite | 0.0540 ± 0.0146 | 0.0030 ± 0.0014 |
| `enron` | 0.25 | GSI_Chained | 0.1740 ± 0.0258 | 0.6483 ± 0.0350 |
| `enron` | 0.3 | GSI_Bipartite | 0.0912 ± 0.0156 | 0.0077 ± 0.0029 |
| `enron` | 0.3 | GSI_Chained | 0.1874 ± 0.0129 | 0.7051 ± 0.0408 |
| `enron` | 0.35 | GSI_Bipartite | 0.1716 ± 0.0378 | 0.0240 ± 0.0078 |
| `enron` | 0.35 | GSI_Chained | 0.1930 ± 0.0183 | 0.7540 ± 0.0439 |
| `enron` | 0.4 | GSI_Bipartite | 0.2154 ± 0.0104 | 0.0931 ± 0.0234 |
| `enron` | 0.4 | GSI_Chained | 0.2003 ± 0.0145 | 0.8101 ± 0.0471 |
| `music` | 0.2 | GSI_Bipartite | 0.0333 ± 0.0745 | 0.0011 ± 0.0018 |
| `music` | 0.2 | GSI_Chained | 0.0333 ± 0.0745 | 0.0017 ± 0.0018 |
| `music` | 0.25 | GSI_Bipartite | 0.1889 ± 0.0843 | 0.0087 ± 0.0051 |
| `music` | 0.25 | GSI_Chained | 0.2222 ± 0.1712 | 0.0166 ± 0.0040 |
| `music` | 0.3 | GSI_Bipartite | 0.3747 ± 0.1682 | 0.0321 ± 0.0154 |
| `music` | 0.3 | GSI_Chained | 0.4714 ± 0.2391 | 0.0589 ± 0.0114 |
| `music` | 0.35 | GSI_Bipartite | 0.6126 ± 0.1055 | 0.1028 ± 0.0278 |
| `music` | 0.35 | GSI_Chained | 0.5748 ± 0.1585 | 0.1488 ± 0.0280 |
| `music` | 0.4 | GSI_Bipartite | 0.6811 ± 0.1132 | 0.2571 ± 0.0291 |
| `music` | 0.4 | GSI_Chained | 0.6963 ± 0.0696 | 0.3090 ± 0.0402 |
| `scene` | 0.2 | GSI_Bipartite | 0.6493 ± 0.1109 | 0.0251 ± 0.0086 |
| `scene` | 0.2 | GSI_Chained | 0.7019 ± 0.0947 | 0.2009 ± 0.0418 |
| `scene` | 0.25 | GSI_Bipartite | 0.7695 ± 0.0691 | 0.0599 ± 0.0155 |
| `scene` | 0.25 | GSI_Chained | 0.8142 ± 0.1037 | 0.2741 ± 0.0398 |
| `scene` | 0.3 | GSI_Bipartite | 0.8305 ± 0.1004 | 0.1263 ± 0.0218 |
| `scene` | 0.3 | GSI_Chained | 0.8070 ± 0.0787 | 0.3593 ± 0.0378 |
| `scene` | 0.35 | GSI_Bipartite | 0.8573 ± 0.0690 | 0.2466 ± 0.0423 |
| `scene` | 0.35 | GSI_Chained | 0.8100 ± 0.0676 | 0.4699 ± 0.0243 |
| `scene` | 0.4 | GSI_Bipartite | 0.8337 ± 0.0185 | 0.4549 ± 0.0609 |
| `scene` | 0.4 | GSI_Chained | 0.8044 ± 0.0152 | 0.6297 ± 0.0256 |
| `yeast` | 0.2 | GSI_Bipartite | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| `yeast` | 0.2 | GSI_Chained | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| `yeast` | 0.25 | GSI_Bipartite | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 |
| `yeast` | 0.25 | GSI_Chained | 0.0284 ± 0.0389 | 0.0015 ± 0.0024 |
| `yeast` | 0.3 | GSI_Bipartite | 0.0143 ± 0.0319 | 0.0001 ± 0.0003 |
| `yeast` | 0.3 | GSI_Chained | 0.0948 ± 0.0607 | 0.0221 ± 0.0136 |
| `yeast` | 0.35 | GSI_Bipartite | 0.1128 ± 0.0791 | 0.0053 ± 0.0064 |
| `yeast` | 0.35 | GSI_Chained | 0.1308 ± 0.0047 | 0.0843 ± 0.0328 |
| `yeast` | 0.4 | GSI_Bipartite | 0.3147 ± 0.0432 | 0.0510 ± 0.0470 |
| `yeast` | 0.4 | GSI_Chained | 0.2909 ± 0.0691 | 0.2064 ± 0.0218 |

---

## 5. Phân tích & Nhận định khoa học cho bài báo (Key Scientific Findings)

1. **Tác động của việc triệt tiêu liên kết chuỗi giữa các nhãn DL:**
   - Khi nhãn DL chỉ phụ thuộc vào nhãn IL (cấu trúc đồ thị phân đôi Bipartite DAG), sai số dự đoán không còn bị tích tụ hay khuếch đại dọc theo chuỗi như Classifier Chains.
   - Nhược điểm: Mô hình không tận dụng được tương quan nội bộ (intra-DL correlation) giữa các nhãn phụ thuộc.

2. **Phân hoạch nhãn IL và DL:**
   - Số lượng nhãn được chọn vào tập IL phản ánh mức độ tự tin của mô hình vào khả năng dự đoán biên độc lập.
   - Việc quan sát số lượng |IL| và |DL| giúp lý giải vì sao trên một số bộ dữ liệu, mô hình Bipartite cho kết quả vượt trội hoặc tương đương Chained GSI.

3. **Tốc độ huấn luyện và suy diễn:**
   - Bipartite GSI cho phép tính toán song song các nhãn DL (sau khi có xác suất IL), giảm thời gian suy diễn đáng kể so với chuỗi tuần tự.