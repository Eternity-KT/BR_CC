# BÁO CÁO THỰC NGHIỆM ĐỐI SÁNH GSI-MLC-PA v6.2 (RESIDUAL ERROR CORRELATION)

**Tài liệu tham chiếu:** `meeting_summary.md`, `spec/spec_v6_2.md`  
**Cấu hình thực nghiệm:** Ngưỡng bóc tách $\tau = 0.75$, Ngưỡng tương quan $\tau_{\text{corr}} = 0.25$, Chi phí từ chối $c = 0.4$, 5-Fold Cross Validation.  

## 1. Giới Thiệu Cốt Lõi
Phiên bản **v6.2** triển khai thuật toán được chỉ đạo trong `meeting_summary.md`:
1. **Bóc tách tầng độc lập (IL):** Vòng lặp `Do...While(1)` kết hợp quy tắc biên Singleton DL.
2. **Xử lý tập DL bằng tương quan sai số:** Đo lường $PCC(l - f(l), p - f(p))$ để ghép đặc trưng có điều kiện $FS[l] = FS(X, IL, DL\_temp[l])$.
3. **Mô hình BR tinh chỉnh:** Huấn luyện bộ phân loại nhị phân BR riêng biệt cho từng nhãn $DL$ trên không gian đặc trưng điều kiện tương ứng.

## 2. Bảng Tổng Hợp Kết Quả Selective Macro-F1

|                                 |         BR |        CC |       ECC |     GSI_v6 |   GSI_v6_2 |
|:--------------------------------|-----------:|----------:|----------:|-----------:|-----------:|
| ('Logistic', 'chd49')           | 0.510348   | 0.507319  | 0.530877  | 0.503641   | 0.51594    |
| ('Logistic', 'emotions')        | 0.597189   | 0.59319   | 0.64158   | 0.60229    | 0.663866   |
| ('Logistic', 'genbase')         | 0.678226   | 0.69304   | 0.679707  | 0.659356   | 0.666764   |
| ('Logistic', 'gpositivepseaac') | 0.553548   | 0.579118  | 0.55602   | 0.527232   | 0.558221   |
| ('Logistic', 'humanpseaac')     | 0.112385   | 0.139667  | 0.149863  | 0.0851953  | 0.10038    |
| ('Logistic', 'music')           | 0.606741   | 0.603077  | 0.647998  | 0.626629   | 0.666622   |
| ('Logistic', 'plantpseaac')     | 0.140987   | 0.17151   | 0.161883  | 0.0935581  | 0.116211   |
| ('Logistic', 'scene')           | 0.699389   | 0.723445  | 0.738403  | 0.728016   | 0.747522   |
| ('Logistic', 'viruspseaac')     | 0.369238   | 0.383646  | 0.38229   | 0.343852   | 0.361816   |
| ('Logistic', 'yeast')           | 0.374751   | 0.401731  | 0.383934  | 0.370623   | 0.354051   |
| ('MLP', 'chd49')                | 0.488108   | 0.496762  | 0.498926  | 0.49735    | 0.50478    |
| ('MLP', 'emotions')             | 0.44109    | 0.508187  | 0.514449  | 0.479662   | 0.511751   |
| ('MLP', 'genbase')              | 0          | 0.439697  | 0.417758  | 0.0476543  | 0.41943    |
| ('MLP', 'gpositivepseaac')      | 0.528277   | 0.594524  | 0.58505   | 0.596163   | 0.603305   |
| ('MLP', 'humanpseaac')          | 0.00652423 | 0.118289  | 0.119462  | 0.0850878  | 0.0710395  |
| ('MLP', 'music')                | 0.445304   | 0.522513  | 0.531775  | 0.516312   | 0.516076   |
| ('MLP', 'plantpseaac')          | 0.0207409  | 0.170331  | 0.16246   | 0.136179   | 0.132986   |
| ('MLP', 'scene')                | 0.645814   | 0.597156  | 0.594713  | 0.586793   | 0.554329   |
| ('MLP', 'viruspseaac')          | 0.39401    | 0.423633  | 0.428024  | 0.420122   | 0.419211   |
| ('MLP', 'yeast')                | 0.284138   | 0.356776  | 0.360626  | 0.344071   | 0.31791    |
| ('SVM', 'chd49')                | 0.383011   | 0.449528  | 0.47431   | 0.409496   | 0.374772   |
| ('SVM', 'emotions')             | 0.572632   | 0.561798  | 0.626566  | 0.577091   | 0.582231   |
| ('SVM', 'genbase')              | 0.761645   | 0.761645  | 0.761645  | 0.754238   | 0.761645   |
| ('SVM', 'gpositivepseaac')      | 0.474824   | 0.528268  | 0.545693  | 0.560411   | 0.475043   |
| ('SVM', 'humanpseaac')          | 0.0138332  | 0.0790161 | 0.0676892 | 0.00590429 | 0.00372862 |
| ('SVM', 'music')                | 0.566405   | 0.566809  | 0.621761  | 0.60639    | 0.59467    |
| ('SVM', 'plantpseaac')          | 0.0556537  | 0.1261    | 0.12965   | 0.0524311  | 0.0354757  |
| ('SVM', 'scene')                | 0.596766   | 0.677649  | 0.723074  | 0.663237   | 0.612804   |
| ('SVM', 'viruspseaac')          | 0.28573    | 0.337838  | 0.338741  | 0.280844   | 0.26752    |
| ('SVM', 'yeast')                | 0.325997   | 0.382879  | 0.352417  | 0.349939   | 0.318036   |


## 3. Bảng Tổng Hợp Coverage (Tỷ Lệ Quyết Định)
|                                 |   BR |   CC |   ECC |   GSI_v6 |   GSI_v6_2 |
|:--------------------------------|-----:|-----:|------:|---------:|-----------:|
| ('Logistic', 'chd49')           |    1 |    1 |     1 | 0.848071 |   0.818944 |
| ('Logistic', 'emotions')        |    1 |    1 |     1 | 0.866107 |   0.851792 |
| ('Logistic', 'genbase')         |    1 |    1 |     1 | 0.998936 |   0.998936 |
| ('Logistic', 'gpositivepseaac') |    1 |    1 |     1 | 0.928729 |   0.926788 |
| ('Logistic', 'humanpseaac')     |    1 |    1 |     1 | 0.967866 |   0.963966 |
| ('Logistic', 'music')           |    1 |    1 |     1 | 0.871079 |   0.853923 |
| ('Logistic', 'plantpseaac')     |    1 |    1 |     1 | 0.970617 |   0.965674 |
| ('Logistic', 'scene')           |    1 |    1 |     1 | 0.954669 |   0.948774 |
| ('Logistic', 'viruspseaac')     |    1 |    1 |     1 | 0.925423 |   0.919875 |
| ('Logistic', 'yeast')           |    1 |    1 |     1 | 0.908492 |   0.887184 |
| ('MLP', 'chd49')                |    1 |    1 |     1 | 0.83331  |   0.821758 |
| ('MLP', 'emotions')             |    1 |    1 |     1 | 0.843361 |   0.840809 |
| ('MLP', 'genbase')              |    1 |    1 |     1 | 0.997709 |   0.987523 |
| ('MLP', 'gpositivepseaac')      |    1 |    1 |     1 | 0.890641 |   0.93497  |
| ('MLP', 'humanpseaac')          |    1 |    1 |     1 | 0.963552 |   0.963551 |
| ('MLP', 'music')                |    1 |    1 |     1 | 0.841773 |   0.826819 |
| ('MLP', 'plantpseaac')          |    1 |    1 |     1 | 0.968234 |   0.969254 |
| ('MLP', 'scene')                |    1 |    1 |     1 | 0.919105 |   0.919185 |
| ('MLP', 'viruspseaac')          |    1 |    1 |     1 | 0.944903 |   0.944455 |
| ('MLP', 'yeast')                |    1 |    1 |     1 | 0.89762  |   0.88805  |
| ('SVM', 'chd49')                |    1 |    1 |     1 | 0.676941 |   0.66134  |
| ('SVM', 'emotions')             |    1 |    1 |     1 | 0.847516 |   0.849813 |
| ('SVM', 'genbase')              |    1 |    1 |     1 | 0.999944 |   1        |
| ('SVM', 'gpositivepseaac')      |    1 |    1 |     1 | 0.92008  |   0.910409 |
| ('SVM', 'humanpseaac')          |    1 |    1 |     1 | 0.979773 |   0.983726 |
| ('SVM', 'music')                |    1 |    1 |     1 | 0.84737  |   0.846216 |
| ('SVM', 'plantpseaac')          |    1 |    1 |     1 | 0.980918 |   0.98611  |
| ('SVM', 'scene')                |    1 |    1 |     1 | 0.949783 |   0.936004 |
| ('SVM', 'viruspseaac')          |    1 |    1 |     1 | 0.848216 |   0.846414 |
| ('SVM', 'yeast')                |    1 |    1 |     1 | 0.916557 |   0.904262 |


## 4. Ma Trận Tương Quan Sai Số Chi Tiết
Toàn bộ ma trận Pearson Correlation và đồ thị phụ thuộc cục bộ $DL\_temp[l]$ được lưu trữ tại `results_v6_2/residual_correlation_matrices/`.
