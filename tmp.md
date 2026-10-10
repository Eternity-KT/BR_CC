
                                  [Ước lượng Tiên nghiệm In-Fold pi_l]
                                                  │
                                                  ▼
                         [Chỉ số Cân bằng Nhãn: beta_l = 1 - 2*|pi_l - 0.5|]
                                                  │
         ┌────────────────────────────────────────┼────────────────────────────────────────┐
         │                                        │                                        │
         ▼                                        ▼                                        ▼
 [CHẾ ĐỘ 1: LỆCH ÂM CỰC ĐOAN]             [CHẾ ĐỘ 2: CÂN BẰNG / NHẸ]              [CHẾ ĐỘ 3: LỆCH DƯƠNG CỰC ĐOAN]
   (beta_l < 0.40 & pi_l < 0.20)            (beta_l >= 0.40 <=> 0.2 <= pi <= 0.8)   (beta_l < 0.40 & pi_l > 0.80)
   - Balanced-Root Platt Scaling            - TẮT Tail-Calibrator                   - Inverted Platt Scaling
   - Bayes LR Thresholds (tau_0, tau_1)     - Khôi phục Chow đối xứng:              - Inverted Bayes LR Thresholds
   - Precision Guard: tau_1 >= 0.50           tau_0 = c, tau_1 = 1 - c              - Neg-Precision Guard: tau_0 <= 0.50
         │                                        │                                        │
         └────────────────────────────────────────┼────────────────────────────────────────┘
                                                  ▼
                       [Hàm Nội Suy Trơn Ngưỡng (Smooth Boundary Blending)]
                                                  │
                                                  ▼
                        [Chuẩn Hóa Thang Đo Mean-Field (OS-NMF Alignment)]
                                                  │
                                                  ▼
                           [Ràng Buộc Độ Phủ Tối Thiểu (Coverage Guard)]
                                                  │
                                                  ▼
                              [Dự Đoán Cuối Cùng Y_pred in {-1, 0, 1}]
