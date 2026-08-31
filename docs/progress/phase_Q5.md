# Handoff Phase Q5 — Jaccard-BOP hoàn chỉnh

**Trạng thái:** Hoàn thành ngày 2026-08-31.

## Objective đã hoàn thành

- Cài `JaccardBOPPolicy` theo Algorithm 3 của Nguyen–Hüllermeier cho complete và partial prediction.
- Tái sử dụng count-distribution DP của Q4 và tách boundary-set mechanics dùng chung cho F-beta/Jaccard.
- Hỗ trợ SEP/PAR penalty, empty-union convention và deterministic tie-breaking thống nhất với Q4.
- Đăng ký Jaccard trong policy registry với alias `jaccard_bop`, `iou` và `intersection_over_union`.
- Exhaustive validate mọi truth/action tới `K=6`, gồm complete/partial và linear/concave.
- Rà API, JSON config và scientific config hash parity cho Hamming, F-beta và Jaccard.
- Chạy runtime/memory smoke ở `K=100` mà không nối policy vào GSI selection.

## Files đã sửa/tạo

| File | Thay đổi Q5 |
|---|---|
| `src/decision/boundary.py` | Base nội bộ dùng chung cho action construction, SEP/PAR, tie, batch API và utility |
| `src/decision/fbeta.py` | Delegate phần mechanics dùng chung sang `BoundarySetBOPPolicy`; công thức Algorithm 2 giữ nguyên |
| `src/decision/jaccard.py` | Complete/partial Algorithm-3 Jaccard BOP và metadata CLI |
| `src/decision/__init__.py` | Export/đăng ký Jaccard policy và aliases |
| `tests/test_decision_jaccard.py` | 7 test exhaustive, edge, parity, hash và runtime/memory Q5 |
| `specify.md` | Đánh dấu Q5 hoàn thành |

Q5 chỉ thay decision modules/tests/documentation; không sửa BR, CC, GSI probability generation, partition selection, evaluator hay output schema.

## Quyết định kỹ thuật và giả định

1. Candidate vẫn có boundary form sau stable descending sort:

   ```text
   [predict 1] [abstain] [predict 0]
   ```

2. Với `l > 0`, gọi `B` là số positive truth trong decided-negative suffix. Jaccard của candidate rút gọn thành:

   ```text
   Jaccard = TP / (l + B)
   E[Jaccard] = E[TP] * E[1 / (l + B)]
   ```

   Phép tách kỳ vọng dùng giả định CLI giữa positive prefix và negative suffix. Algorithm-3 recurrence cập nhật kỳ vọng nghịch đảo khi thêm từng suffix label.
3. Prefix Poisson-binomial distribution từ `count_distribution.py` tính `E[TP]`; cùng cache có thể tiếp tục tái sử dụng ở phase sau.
4. Repo quy ước empty union bằng `1`. Vì vậy, tương tự Q4, implementation thêm negative-only candidates (`l=0`), có expected score bằng xác suất mọi decided-negative truth đều bằng `0`. Exhaustive oracle xác nhận extension này tối ưu đúng contract.
5. `allow_abstention=False` chỉ xét complete action và bất biến theo cost/penalty do `g(0)=0`.
6. Generalized utility là:

   ```text
   E[Jaccard(Y_D, Yhat_D) | x] - g(|A|)
   ```

7. CLI metadata ghi rõ:
   - exact BOP chỉ khi marginal probabilities thỏa conditional label independence;
   - với dependent marginals của GSI/CC, đây là `BOP under CLI approximation`, không phải exact joint-dependent BOP.
8. `BoundarySetBOPPolicy` chỉ là base trong decision layer. Refactor F-beta sang base này không đổi public API, config hay Algorithm-2 output; toàn bộ exhaustive Q4 vẫn pass.
9. Tie tolerance là `1e-12`, sau đó ưu tiên nhiều decided labels hơn và stable original label index như Q4.

## Exhaustive oracle và counterexample

Oracle enumerate:

- mọi truth trong `{0,1}^K` theo joint probability CLI;
- mọi complete action trong `{0,1}^K`;
- mọi partial action trong `{0,-1,1}^K`;
- empty-union score và SEP/PAR penalty đúng contract.

Fixed fixtures phủ `K=1..6`. Thêm 20 random fixtures deterministic đến `K=5` cho cả complete/partial và linear/concave.

Counterexample complete Jaccard:

```text
p = (0.4, 0.4)
threshold 0.5 action = (0, 0), expected Jaccard = 0.36
Jaccard-BOP action   = (1, 1), expected Jaccard = 0.40
```

## Test/lệnh đã chạy + kết quả

```text
python -m unittest tests.test_decision_jaccard -v
```

- PASS 7/7 test Q5.
- DP utility bằng direct truth enumeration và exhaustive action optimum đến 11 chữ số thập phân.
- Registry tạo được đúng ba family; JSON config serializable và ba config tạo ba scientific hash khác nhau.

```text
python -m unittest discover -s tests -v
```

- PASS 51/51 test Q1–Q5.
- Toàn bộ 9 test F-beta Q4 vẫn pass sau refactor boundary base.

```text
python tests_unit.py
```

- PASS toàn bộ 15 nhóm regression/unit/integration cũ; không regression Hamming, model API, GSI hoặc schema v2.

Runtime smoke một instance, partial PAR, máy phát triển hiện tại:

| K | Thời gian |
|---:|---:|
| 6 | 0.650 ms |
| 20 | 3.673 ms |
| 40 | 13.884 ms |
| 80 | 57.174 ms |
| 100 | 89.271 ms |

`tracemalloc` peak tại `K=100` là khoảng `0.061 MiB`. Đây chỉ là smoke benchmark cho implementation và không phải số liệu runtime/memory chính thức của báo cáo.

```text
python -m compileall -q main.py src tests
git diff --check
```

- PASS syntax/import/whitespace; Git chỉ cảnh báo LF/CRLF của môi trường Windows.

## Output/cache được tạo

- Không tạo output/cache thật; benchmark chỉ dùng arrays trong bộ nhớ.
- Integration tests chỉ dùng temporary directories và tự xóa.
- Không sửa `results/`, `results_pa/` hoặc tạo `results_pa_v3/`.

## Việc chưa hoàn thành hoặc blocker

- Chưa nối F1/Jaccard policies vào GSI partition objective; đây là Q6.
- Chưa thêm selection objective registry hoặc partition provider; đây là Q6–Q7.
- Chưa benchmark policy trên dataset/batch thật; Q5 chỉ có single-instance smoke.
- Không có blocker kỹ thuật cho Q6.

## Git status và thay đổi ngoài phạm vi

- Nhánh hiện tại: `Khanh_1`, tracking `origin/Khanh_1` trước các thay đổi Q1–Q5.
- `note.txt` vẫn ở trạng thái deleted có từ trước, không thuộc các phase này và không được chỉnh sửa.
- Các thay đổi Q1–Q5 chưa commit/push tại thời điểm viết handoff.

## Entry point/lệnh đầu tiên cho Phase Q6

```powershell
python -m unittest discover -s tests -v
```

Sau baseline, tạo `src/selection/objectives.py` và injection point tối thiểu cho `selection_objective`/`decision_policy` trong GSI. Giữ default `full_macro_f1` bitwise tương thích; mọi candidate partition phải sinh probability matrix trước rồi mới qua policy/objective. Thêm `immediate_instance_f1`, `bop_instance_f1`, `bop_jaccard`, F0.5/F2 nhưng chưa triển khai ablation partition modes của Q7.
