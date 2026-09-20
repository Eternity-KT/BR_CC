# Đặc trưng của 10 bộ dữ liệu đa nhãn

## Kết quả tính lại

Bảng dưới đây được tính trực tiếp trên toàn bộ các tệp ARFF hiện có trong thư mục `data/`, thông qua cấu hình và quy tắc tách nhãn của `src/data/loader.py`. Ảnh tham chiếu chỉ có 9 dòng dữ liệu; bảng này bổ sung **Genbase**, là bộ dữ liệu thứ 10 trong cấu hình benchmark của dự án.

  | Bộ dữ liệu | N | K | LC | LD | MeanIR | Nhãn hiếm | Rare (%) |
  |---|---:|---:|---:|---:|---:|---:|---:|
  | Emotions | 593 | 6 | 1.8685 | 0.3114 | 2.32 | 0/6 | 0.0 |
  | Music | 592 | 6 | 1.8699 | 0.3117 | 2.32 | 0/6 | 0.0 |
  | Scene | 2,407 | 6 | 1.0740 | 0.1790 | 4.66 | 0/6 | 0.0 |
  | Yeast | 2,417 | 14 | 4.2371 | 0.3026 | 8.95 | 1/14 | 7.1 |
  | Genbase | 662 | 27 | 1.2523 | 0.0464 | 143.46 | 18/27 | 66.7 |
  | Medical | 978 | 45 | 1.2454 | 0.0277 | 328.07 | 38/45 | 84.4 |
  | Enron | 1,702 | 53 | 3.3784 | 0.0637 | 136.87 | 38/53 | 71.7 |
  | CAL500 | 502 | 174 | 26.0438 | 0.1497 | 22.34 | 67/174 | 38.5 |
  | Bibtex | 7,395 | 159 | 2.4019 | 0.0151 | 87.70 | 156/159 | 98.1 |
  | Reuters-K500 | 6,000 | 103 | 1.4622 | 0.0142 | 885.89 | 96/103 | 93.2 |

## Định nghĩa các chỉ số

Ký hiệu $Y \in \{0,1\}^{N \times K}$ là ma trận nhãn, $y_{ik}=1$ khi mẫu $i$ mang nhãn $k$. Với mỗi nhãn $k$:

- $n_k^+ = \sum_{i=1}^{N} y_{ik}$: số mẫu dương.
- $n_k^- = N-n_k^+$: số mẫu âm.

Các chỉ số trong bảng được tính như sau:

- **N**: số mẫu dữ liệu.
- **K**: số nhãn.
- **Label Cardinality (LC)**: số nhãn dương trung bình trên mỗi mẫu.

  \[
  LC=\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K}y_{ik}
  \]

- **Label Density (LD)**: tỷ lệ nhãn dương trung bình trên toàn bộ không gian nhãn.

  \[
  LD=\frac{LC}{K}=\frac{1}{NK}\sum_{i=1}^{N}\sum_{k=1}^{K}y_{ik}
  \]

- **Imbalance Ratio của nhãn $k$**: tỷ lệ giữa lớp đa số và lớp thiểu số trong bài toán nhị phân tương ứng với nhãn đó. Mẫu số được chặn ở 1 để chỉ số vẫn hữu hạn khi một lớp không có quan sát.

  \[
  IR_k=\frac{\max(n_k^+,n_k^-)}{\max\left(1,\min(n_k^+,n_k^-)\right)}
  \]

- **MeanIR**: trung bình $IR_k$ trên $K$ nhãn.

  \[
  MeanIR=\frac{1}{K}\sum_{k=1}^{K}IR_k
  \]

- **Rare (%)**: phần trăm nhãn có tỷ lệ mẫu dương nhỏ hơn 5%.

  \[
  Rare(\%)=\frac{100}{K}\sum_{k=1}^{K}\mathbf{1}\left(\frac{n_k^+}{N}<0.05\right)
  \]

Cột **Nhãn hiếm** trình bày cả tử số và mẫu số dùng để tính `Rare (%)`, giúp kiểm tra trực tiếp tỷ lệ đã làm tròn.

## Kiểm tra chất lượng dữ liệu

- Cả 10 ma trận nhãn đều có đúng số hàng `N`, số cột `K` và chỉ chứa giá trị nhị phân `0/1`.
- Không có mẫu nào hoàn toàn không mang nhãn.
- Reuters-K500 có 4 nhãn với số mẫu dương bằng 0: `CCAT.C18.C18`, `ECAT.E13.E13`, `MCAT.MCAT` và `MCAT.M14.M143`. Bốn nhãn này được tính là nhãn hiếm và làm MeanIR của Reuters-K500 tăng mạnh. Quy tắc chặn mẫu số ở 1 cho ra MeanIR hữu hạn `885.89`, phù hợp với cách tính thể hiện trong ảnh tham chiếu.
- Các giá trị được tính từ phiên bản dữ liệu cục bộ của dự án nên có thể khác nhẹ so với bảng trong ảnh. Khác biệt đáng chú ý là MeanIR của Scene (`4.66`) và Yeast (`8.95`); các chỉ số còn lại nhất quán với dữ liệu đang được dùng để chạy benchmark.

## Mã tính toán tối giản

```python
import numpy as np
from src.data.loader import DATASET_CONFIG, load_dataset

for dataset_name in DATASET_CONFIG:
    _, y, _, _ = load_dataset(dataset_name)
    n, k = y.shape

    positive = y.sum(axis=0).astype(float)
    negative = n - positive
    minority = np.minimum(positive, negative)
    majority = np.maximum(positive, negative)

    lc = y.sum(axis=1).mean()
    ld = y.mean()
    mean_ir = (majority / np.maximum(minority, 1.0)).mean()
    rare_count = (positive / n < 0.05).sum()
    rare_pct = rare_count / k * 100
```

## Nguồn dữ liệu

| Bộ dữ liệu | Tệp được phân tích |
|---|---|
| Emotions | `data/emotions/emotions.arff` |
| Music | `data/Music.arff` |
| Scene | `data/Scene.arff` |
| Yeast | `data/Yeast.arff` |
| Genbase | `data/genbase/genbase.arff` |
| Medical | `data/medical/medical.arff` |
| Enron | `data/enron/enron.arff` |
| CAL500 | `data/CAL500.arff` |
| Bibtex | `data/bibtex/bibtex.arff` |
| Reuters-K500 | `data/REUTERS-K500-EX2.arff` |
