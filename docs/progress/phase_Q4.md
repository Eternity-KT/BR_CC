# Handoff Phase Q4 — F1-BOP hoàn chỉnh

**Trạng thái:** Hoàn thành ngày 2026-08-31.

## Objective đã hoàn thành

- Cài `FbetaBOPPolicy`, mặc định `beta=1`, cho cả complete mode (`allow_abstention=False`) và partial mode.
- Cài Algorithm 2 của Nguyen–Hüllermeier bằng count-distribution dynamic programming dưới giả định conditional label independence (CLI).
- Hỗ trợ cả SEP `g(a)=a*c` và PAR `g(a)=a*K*c/(K+a)` mà không thay core model.
- Khóa deterministic tie: expected utility, rồi nhiều quyết định hơn, rồi stable original label index.
- Ghi metadata/config phân biệt exact CLI BOP với “BOP under CLI approximation” khi chỉ có dependent marginal probabilities.
- Exhaustive validate toàn bộ action space complete/partial tới `K=6`, gồm fixture xác định và ngẫu nhiên.
- Thêm counterexample chứng minh F1-BOP không phải threshold `0.5`/Hamming action.
- Đăng ký F-beta/F1 policy trong registry, nhưng chưa nối vào objective chọn IL/DL.

## Files đã sửa/tạo

| File | Thay đổi Q4 |
|---|---|
| `src/decision/count_distribution.py` | Prefix/suffix Poisson-binomial count distributions dùng DP `O(K²)` |
| `src/decision/fbeta.py` | Complete/partial F-beta BOP, expected utility, config và CLI metadata |
| `src/decision/__init__.py` | Đăng ký canonical `fbeta` cùng alias `fbeta_bop`, `f1`, `f1_bop` |
| `tests/test_decision_fbeta.py` | 9 test Q4, exhaustive oracle, counterexample, edge, registry và runtime smoke |
| `specify.md` | Đánh dấu Q4 hoàn thành |

Q4 không sửa `main.py`, GSI selection hoặc probability estimator. Các thay đổi Q1–Q3 vẫn được giữ nguyên trong worktree.

## Quyết định kỹ thuật và giả định

1. Sau stable sort giảm dần theo marginal probability, mỗi candidate partial có dạng ba vùng:

   ```text
   [predict 1] [abstain] [predict 0]
   ```

   Đây là boundary-set structure của generalized monotonic accuracy measures trong paper.
2. Với `l > 0`, DP dùng phân phối số positive trong prefix và recurrence `S` của Algorithm 2 để đánh giá mọi biên phải trong `O(K³)` time; prefix count cache dùng `O(K²)` memory.
3. Metric contract của repo quy ước F-beta bằng `1` khi truth và prediction trên decided set đều rỗng. Vì vậy implementation bổ sung các negative-only candidate (`l=0`) mà dạng Algorithm 2 thường bỏ khi định nghĩa zero-positive F-measure bằng `0`. Exhaustive oracle xác nhận phần mở rộng này là tối ưu theo đúng contract của repo.
4. `allow_abstention=False` chỉ cho candidate không có `-1`; output complete không phụ thuộc cost/penalty vì `g(0)=0`.
5. Generalized expected utility là:

   ```text
   E[F_beta(Y_D, Yhat_D) | x] - g(|A|)
   ```

   `predict_with_utility` trả cả action và utility; `expected_utility` trả utility theo instance.
6. Với marginal probabilities thật sự thỏa CLI, đây là Bayes-optimal action theo objective trên. Với dependent marginals của GSI/CC, config ghi rõ `BOP under CLI approximation`; không tuyên bố exact BOP của joint distribution phụ thuộc.
7. Tie được xét với tolerance `1e-12`; ưu tiên nhiều quyết định hơn, sau đó action lớn hơn tại original label index nhỏ hơn. Probability sort dùng stable sort để các nhãn đồng xác suất reproducible.
8. Q4 chỉ cung cấp module/factory. Việc inject F1 policy vào GSI partition objective thuộc Q6, nên default Hamming và output Q3 không đổi.

## Exhaustive oracle

Oracle enumerate:

- mọi truth vector trong `{0,1}^K` và xác suất joint CLI tương ứng;
- mọi complete action trong `{0,1}^K` khi `allow_abstention=False`;
- mọi partial action trong `{0,-1,1}^K` khi `allow_abstention=True`;
- utility đúng empty-set convention và SEP/PAR penalty.

Các case cố định phủ `K=1..6`, `beta ∈ {0.5,1,2}`, complete/partial và linear/concave. Ngoài ra có 20 fixture random deterministic đến `K=5`.

Counterexample complete F1:

```text
p = (0.4, 0.4)
threshold 0.5 action = (0, 0), expected F1 = 0.36
F1-BOP action        = (1, 1), expected F1 = 0.48
```

## Test/lệnh đã chạy + kết quả

```text
python -m unittest tests.test_decision_fbeta -v
```

- PASS 9/9 Q4 tests.
- DP utility bằng direct truth enumeration và exhaustive action optimum đến 11 chữ số thập phân.

```text
python -m unittest discover -s tests -v
```

- PASS 44/44 test Q1–Q4.

```text
python tests_unit.py
```

- PASS toàn bộ 15 nhóm regression/unit/integration cũ; Hamming SEP/PAR/GSI và schema v2 không regression.

Runtime smoke một instance, partial PAR, máy phát triển hiện tại:

| K | Thời gian |
|---:|---:|
| 6 | 0.667 ms |
| 20 | 3.985 ms |
| 40 | 15.966 ms |
| 80 | 62.073 ms |

Các số này chỉ là smoke benchmark, không phải kết quả nghiên cứu chính thức; cần benchmark theo batch/dataset thật trước khi báo cáo runtime.

```text
python -m compileall -q main.py src tests
git diff --check
```

- PASS syntax/import/whitespace; Git chỉ cảnh báo LF/CRLF của môi trường Windows.

## Output/cache được tạo

- Không tạo output/cache thật; benchmark dùng probability array trong bộ nhớ.
- Test schema v3 vẫn chỉ dùng temporary directories và tự xóa.
- Không sửa `results/`, `results_pa/` hoặc tạo `results_pa_v3/`.

## Việc chưa hoàn thành hoặc blocker

- Chưa cài Jaccard-BOP; đây là Q5.
- Chưa nối F1-BOP vào GSI selection objective; đây là Q6 sau khi Jaccard parity hoàn tất.
- Chưa benchmark batch/dataset thật hay tối ưu vector hóa qua nhiều instance; chỉ có runtime smoke một instance.
- Không có blocker kỹ thuật cho Q5.

## Git status và thay đổi ngoài phạm vi

- Nhánh hiện tại: `Khanh_1`, tracking `origin/Khanh_1` trước các thay đổi Q1–Q4.
- `note.txt` vẫn ở trạng thái deleted có từ trước, không thuộc các phase này và không được chỉnh sửa.
- Các thay đổi Q1–Q4 chưa commit/push tại thời điểm viết handoff.

## Entry point/lệnh đầu tiên cho Phase Q5

```powershell
python -m unittest discover -s tests -v
```

Sau baseline, đọc Algorithm 3 trong paper và cài `src/decision/jaccard.py`, tái sử dụng `count_distribution.py`. Giữ cùng CLI metadata, empty-union convention, SEP/PAR và tie rule; exhaustive validate `{0,-1,1}^K`, `K <= 6`, rồi rà registry/config parity với Hamming và F-beta. Không nối objective vào GSI trước Q6.
