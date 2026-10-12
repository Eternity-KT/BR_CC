# Báo cáo ablation GSI-MLC-PA v6.3

Ngày tạo: 2026-10-12  
Cấu hình: 10 dataset, 3 base learner, outer 5-fold; PCC dùng 1.000 ancestral samples cho mỗi test x; CC/PCC dùng chung chain order trong từng fold.

## 1. Trạng thái thực nghiệm

- Fold mô hình mới có complete metric: 750 / 750.
- Fold mô hình mới có selective metric tại cost=0.3: 750 / 750.
- BR, CC, MLC_PA và GSI_v6_2 được kế thừa từ v6.2, không chạy lại.
- BR và CC không có cơ chế từ chối nhưng vẫn được giữ làm đường tham chiếu: selective metrics của chúng lấy từ complete metrics, coverage=1 và không đổi theo cost.
- Các kết quả kế thừa mang `inherited_unmatched`: artifact nguồn không có prediction/split hash để chứng minh ghép cặp fold; vì vậy không dùng chúng cho kết luận paired.

## 2. Trung bình các fold và dataset của mô hình mới

| base_learner | model | macro_f1 | micro_f1 | hamming_accuracy | selective_macro_f1 | generalized_loss | coverage | brier_score | ece |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| logistic | GSI_v6_3_1L_ResidualBR | 0.4711 | 0.5945 | 0.8497 | 0.4692 | 0.1319 | 0.8203 | 0.1067 | 0.0376 |
| logistic | GSI_v6_3_CC_X | 0.4744 | 0.6108 | 0.8408 | 0.4769 | 0.139 | 0.8245 | 0.1133 | 0.0499 |
| logistic | GSI_v6_3_CC_X_IL | 0.4813 | 0.6171 | 0.8427 | 0.4844 | 0.1374 | 0.8289 | 0.1124 | 0.0499 |
| logistic | GSI_v6_3_PCC_X | 0.4631 | 0.5882 | 0.8488 | 0.453 | 0.1323 | 0.8106 | 0.1072 | 0.0364 |
| logistic | GSI_v6_3_PCC_X_IL | 0.4695 | 0.5941 | 0.8501 | 0.464 | 0.1309 | 0.8126 | 0.1063 | 0.0355 |
| svm_calibrated | GSI_v6_3_1L_ResidualBR | 0.4076 | 0.5153 | 0.8452 | 0.3844 | 0.1339 | 0.7813 | 0.1078 | 0.0278 |
| svm_calibrated | GSI_v6_3_CC_X | 0.4292 | 0.5791 | 0.8337 | 0.4236 | 0.1435 | 0.7953 | 0.1175 | 0.0438 |
| svm_calibrated | GSI_v6_3_CC_X_IL | 0.4329 | 0.5823 | 0.8344 | 0.4257 | 0.1428 | 0.7956 | 0.117 | 0.0435 |
| svm_calibrated | GSI_v6_3_PCC_X | 0.419 | 0.5424 | 0.8438 | 0.3937 | 0.135 | 0.777 | 0.1089 | 0.0287 |
| svm_calibrated | GSI_v6_3_PCC_X_IL | 0.4239 | 0.5461 | 0.8447 | 0.3971 | 0.1344 | 0.7772 | 0.1083 | 0.0292 |
| mlp | GSI_v6_3_1L_ResidualBR | 0.401 | 0.5385 | 0.8367 | 0.3875 | 0.1407 | 0.8078 | 0.1133 | 0.0401 |
| mlp | GSI_v6_3_CC_X | 0.4068 | 0.5503 | 0.8365 | 0.3987 | 0.1418 | 0.8092 | 0.1144 | 0.043 |
| mlp | GSI_v6_3_CC_X_IL | 0.414 | 0.5566 | 0.8368 | 0.4039 | 0.1419 | 0.8114 | 0.1142 | 0.0437 |
| mlp | GSI_v6_3_PCC_X | 0.4007 | 0.5407 | 0.8366 | 0.3877 | 0.1411 | 0.8115 | 0.1141 | 0.0429 |
| mlp | GSI_v6_3_PCC_X_IL | 0.407 | 0.5449 | 0.8366 | 0.3936 | 0.1411 | 0.8116 | 0.1139 | 0.0434 |

## 3. Đối chiếu mô tả với các baseline kế thừa

Bảng này hỗ trợ đọc RQ1 ở mức mô tả tổng hợp. Các hàng v6.2 là `inherited_unmatched`, vì vậy không được diễn giải như so sánh paired hay bằng chứng ý nghĩa thống kê. Dấu `-` phản ánh metric không có trong artifact nguồn.

| base_learner | model | macro_f1 | micro_f1 | hamming_accuracy | selective_macro_f1 | generalized_loss | coverage |
| --- | --- | --- | --- | --- | --- | --- | --- |
| logistic | BR | 0.4643 | 0.5881 | 0.848 | 0.4643 | 0.152 | 1.0 |
| logistic | CC | 0.4796 | 0.6103 | 0.8396 | 0.4796 | 0.1604 | 1.0 |
| logistic | GSI_v6_2 | 0.4711 | 0.5945 | 0.8499 | 0.4712 | - | 0.8208 |
| logistic | GSI_v6_3_1L_ResidualBR | 0.4711 | 0.5945 | 0.8497 | 0.4692 | 0.1319 | 0.8203 |
| logistic | GSI_v6_3_CC_X | 0.4744 | 0.6108 | 0.8408 | 0.4769 | 0.139 | 0.8245 |
| logistic | GSI_v6_3_CC_X_IL | 0.4813 | 0.6171 | 0.8427 | 0.4844 | 0.1374 | 0.8289 |
| logistic | GSI_v6_3_PCC_X | 0.4631 | 0.5882 | 0.8488 | 0.453 | 0.1323 | 0.8106 |
| logistic | GSI_v6_3_PCC_X_IL | 0.4695 | 0.5941 | 0.8501 | 0.464 | 0.1309 | 0.8126 |
| logistic | MLC_PA | 0.4643 | 0.5881 | 0.848 | 0.459 | - | 0.8189 |
| svm_calibrated | BR | 0.4036 | 0.5122 | 0.8451 | 0.4036 | 0.1549 | 1.0 |
| svm_calibrated | CC | 0.4472 | 0.5785 | 0.8365 | 0.4472 | 0.1635 | 1.0 |
| svm_calibrated | GSI_v6_2 | 0.4102 | 0.517 | 0.846 | 0.3829 | - | 0.7819 |
| svm_calibrated | GSI_v6_3_1L_ResidualBR | 0.4076 | 0.5153 | 0.8452 | 0.3844 | 0.1339 | 0.7813 |
| svm_calibrated | GSI_v6_3_CC_X | 0.4292 | 0.5791 | 0.8337 | 0.4236 | 0.1435 | 0.7953 |
| svm_calibrated | GSI_v6_3_CC_X_IL | 0.4329 | 0.5823 | 0.8344 | 0.4257 | 0.1428 | 0.7956 |
| svm_calibrated | GSI_v6_3_PCC_X | 0.419 | 0.5424 | 0.8438 | 0.3937 | 0.135 | 0.777 |
| svm_calibrated | GSI_v6_3_PCC_X_IL | 0.4239 | 0.5461 | 0.8447 | 0.3971 | 0.1344 | 0.7772 |
| svm_calibrated | MLC_PA | 0.4036 | 0.5122 | 0.8451 | 0.3788 | - | 0.7811 |
| mlp | BR | 0.3254 | 0.3998 | 0.8368 | 0.3254 | 0.1632 | 1.0 |
| mlp | CC | 0.4228 | 0.5663 | 0.8369 | 0.4228 | 0.1631 | 1.0 |
| mlp | GSI_v6_2 | 0.4017 | 0.5419 | 0.8378 | 0.3855 | - | 0.8079 |
| mlp | GSI_v6_3_1L_ResidualBR | 0.401 | 0.5385 | 0.8367 | 0.3875 | 0.1407 | 0.8078 |
| mlp | GSI_v6_3_CC_X | 0.4068 | 0.5503 | 0.8365 | 0.3987 | 0.1418 | 0.8092 |
| mlp | GSI_v6_3_CC_X_IL | 0.414 | 0.5566 | 0.8368 | 0.4039 | 0.1419 | 0.8114 |
| mlp | GSI_v6_3_PCC_X | 0.4007 | 0.5407 | 0.8366 | 0.3877 | 0.1411 | 0.8115 |
| mlp | GSI_v6_3_PCC_X_IL | 0.407 | 0.5449 | 0.8366 | 0.3936 | 0.1411 | 0.8116 |
| mlp | MLC_PA | 0.3254 | 0.3998 | 0.8368 | 0.2965 | - | 0.7462 |

## 4. Chênh lệch ghép cặp so với đối chứng single-stage ResidualBR

Delta dương tốt hơn đối với F1/coverage; delta âm tốt hơn đối với generalized loss. Đây là mô tả thực nghiệm, không phải tuyên bố ý nghĩa thống kê hay quan hệ nhân quả.

| model | paired_folds | delta_macro_f1 | delta_selective_macro_f1 | delta_generalized_loss | delta_coverage |
| --- | --- | --- | --- | --- | --- |
| GSI_v6_3_CC_X | 150 | 0.0102 | 0.0194 | 0.0059 | 0.0065 |
| GSI_v6_3_PCC_X | 150 | 0.001 | -0.0023 | 0.0007 | -0.0034 |
| GSI_v6_3_CC_X_IL | 150 | 0.0162 | 0.0243 | 0.0052 | 0.0088 |
| GSI_v6_3_PCC_X_IL | 150 | 0.0069 | 0.0045 | -0.0001 | -0.0027 |

## 5. So sánh trực tiếp PCC–CC và X_IL–X

| comparison | paired_folds | delta_macro_f1 | delta_selective_macro_f1 | delta_generalized_loss |
| --- | --- | --- | --- | --- |
| PCC_X − CC_X | 150 | -0.0092 | -0.0216 | -0.0052 |
| PCC_X_IL − CC_X_IL | 150 | -0.0093 | -0.0197 | -0.0053 |
| CC_X_IL − CC_X | 150 | 0.0059 | 0.0049 | -0.0007 |
| PCC_X_IL − PCC_X | 150 | 0.0059 | 0.0068 | -0.0007 |

## 6. Mô hình có Selective Macro-F1 cao nhất theo dataset/learner

| base_learner | dataset | model | selective_macro_f1 |
| --- | --- | --- | --- |
| logistic | chd49 | GSI_v6_3_CC_X | 0.5335 |
| logistic | emotions | GSI_v6_3_1L_ResidualBR | 0.6829 |
| logistic | genbase | GSI_v6_3_1L_ResidualBR | 0.6458 |
| logistic | gpositivepseaac | GSI_v6_3_CC_X_IL | 0.5589 |
| logistic | humanpseaac | GSI_v6_3_CC_X_IL | 0.1232 |
| logistic | music | GSI_v6_3_CC_X_IL | 0.6711 |
| logistic | plantpseaac | GSI_v6_3_CC_X | 0.1221 |
| logistic | scene | GSI_v6_3_CC_X_IL | 0.7628 |
| logistic | viruspseaac | GSI_v6_3_CC_X_IL | 0.374 |
| logistic | yeast | GSI_v6_3_CC_X | 0.3843 |
| mlp | chd49 | GSI_v6_3_CC_X_IL | 0.4947 |
| mlp | emotions | GSI_v6_3_CC_X_IL | 0.542 |
| mlp | genbase | GSI_v6_3_1L_ResidualBR | 0.3698 |
| mlp | gpositivepseaac | GSI_v6_3_PCC_X | 0.5781 |
| mlp | humanpseaac | GSI_v6_3_CC_X | 0.0753 |
| mlp | music | GSI_v6_3_CC_X_IL | 0.5457 |
| mlp | plantpseaac | GSI_v6_3_CC_X | 0.127 |
| mlp | scene | GSI_v6_3_CC_X_IL | 0.5822 |
| mlp | viruspseaac | GSI_v6_3_CC_X | 0.4377 |
| mlp | yeast | GSI_v6_3_CC_X_IL | 0.3451 |
| svm_calibrated | chd49 | GSI_v6_3_CC_X | 0.3485 |
| svm_calibrated | emotions | GSI_v6_3_CC_X | 0.5961 |
| svm_calibrated | genbase | GSI_v6_3_CC_X | 0.7468 |
| svm_calibrated | gpositivepseaac | GSI_v6_3_CC_X_IL | 0.5243 |
| svm_calibrated | humanpseaac | GSI_v6_3_CC_X | 0.0594 |
| svm_calibrated | music | GSI_v6_3_CC_X_IL | 0.6088 |
| svm_calibrated | plantpseaac | GSI_v6_3_CC_X_IL | 0.0793 |
| svm_calibrated | scene | GSI_v6_3_CC_X_IL | 0.6772 |
| svm_calibrated | viruspseaac | GSI_v6_3_CC_X_IL | 0.2922 |
| svm_calibrated | yeast | GSI_v6_3_CC_X | 0.3399 |

## 7. Quy mô IL_1

| base_learner | model | mean | std | min | max |
| --- | --- | --- | --- | --- | --- |
| logistic | GSI_v6_3_1L_ResidualBR | 3.28 | 5.0872 | 0.0 | 20.0 |
| logistic | GSI_v6_3_CC_X | 3.28 | 5.0872 | 0.0 | 20.0 |
| logistic | GSI_v6_3_CC_X_IL | 3.28 | 5.0872 | 0.0 | 20.0 |
| logistic | GSI_v6_3_PCC_X | 3.28 | 5.0872 | 0.0 | 20.0 |
| logistic | GSI_v6_3_PCC_X_IL | 3.28 | 5.0872 | 0.0 | 20.0 |
| svm_calibrated | GSI_v6_3_1L_ResidualBR | 3.16 | 5.5378 | 0.0 | 20.0 |
| svm_calibrated | GSI_v6_3_CC_X | 3.16 | 5.5378 | 0.0 | 20.0 |
| svm_calibrated | GSI_v6_3_CC_X_IL | 3.16 | 5.5378 | 0.0 | 20.0 |
| svm_calibrated | GSI_v6_3_PCC_X | 3.16 | 5.5378 | 0.0 | 20.0 |
| svm_calibrated | GSI_v6_3_PCC_X_IL | 3.16 | 5.5378 | 0.0 | 20.0 |
| mlp | GSI_v6_3_1L_ResidualBR | 1.78 | 1.8438 | 0.0 | 9.0 |
| mlp | GSI_v6_3_CC_X | 1.78 | 1.8438 | 0.0 | 9.0 |
| mlp | GSI_v6_3_CC_X_IL | 1.78 | 1.8438 | 0.0 | 9.0 |
| mlp | GSI_v6_3_PCC_X | 1.78 | 1.8438 | 0.0 | 9.0 |
| mlp | GSI_v6_3_PCC_X_IL | 1.78 | 1.8438 | 0.0 | 9.0 |

## 8. Tính minh bạch của dữ liệu kế thừa

- `complete_metric.csv`: 1920 ô metric/metadata kế thừa bị thiếu và được xuất `-`.
- `selective_metric.csv`: 8400 ô metric/metadata kế thừa bị thiếu và được xuất `-`.
- `calibration_metric.csv`: 4440 ô metric/metadata kế thừa bị thiếu và được xuất `-`.
- `layer_performance.csv`: 17280 ô metric/metadata kế thừa bị thiếu và được xuất `-`.

## 9. Giới hạn diễn giải

- Selective Subset Accuracy dùng quy ước tạm của đặc tả: loại mẫu all-abstain khỏi mẫu số.
- Không thực hiện kiểm định ý nghĩa thống kê trong báo cáo này.
- So sánh paired chỉ dùng năm mô hình v6.3 mới vì chúng dùng cùng outer split và cùng partition/order trong fold.
- Metric thiếu của v6.2 không được nội suy hoặc thay bằng 0.
