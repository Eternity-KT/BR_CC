# Handoff Phase Q3 — Decision-policy interface và Hamming regression

**Trạng thái:** Hoàn thành ngày 2026-08-31.

## Objective đã hoàn thành

- Tạo package `src/decision` với interface chung, registry tối thiểu và `HammingBOPPolicy`.
- Di chuyển công thức quyết định SEP/PAR ra khỏi core xác suất; `MLCPartialAbstentionClassifier` và `GSIMLCPartialAbstentionClassifier` delegate sang policy mới.
- Giữ nguyên API `predict`, `predict_from_proba` và `decision_mask` đang có.
- Cho evaluator schema v3 tạo Hamming policy qua registry thay vì gọi trực tiếp decision rule của classifier.
- Ghi config policy đầy đủ vào fold cache hash/run manifest và model metadata của schema v3.
- Khóa regression bitwise trên fixture ngưỡng, fixture ngẫu nhiên và toàn bộ test cũ.

## Files đã sửa/tạo

| File | Thay đổi Q3 |
|---|---|
| `src/decision/base.py` | `DecisionPolicy`, probability/cost/penalty validation và penalty helper dùng chung |
| `src/decision/registry.py` | Registry/factory có canonical name và alias xác định |
| `src/decision/hamming.py` | Hamming SEP/PAR, expected utility, config serialization và tie rule |
| `src/decision/__init__.py` | Public API và đăng ký `hamming`/`hamming_bop`/`sep`/`par` |
| `src/models/mlc_pa.py` | Delegate SEP/PAR sang `HammingBOPPolicy`; không đổi probability estimator |
| `src/models/gsi_mlc_pa.py` | Delegate final BOP sang `HammingBOPPolicy`; không đổi IL/DL hay marginalization |
| `main.py` | Evaluator v3 lấy Hamming policy từ registry và lưu metadata |
| `src/evaluation/pipeline_v3.py` | Policy config tham gia pair/run config hash và manifest |
| `tests/test_decision_hamming.py` | 5 test Q3 cho registry, boundary, parity, evaluator và cache config |
| `specify.md` | Đánh dấu Q3 hoàn thành |

Các thay đổi và handoff Q1–Q2 vẫn chưa commit, được giữ nguyên trong worktree.

## Quyết định kỹ thuật và giả định

1. `HammingBOPPolicy` là decision-time module thuần, chỉ nhận marginal probabilities; không fit và không thay core BR/CC/GSI.
2. PAR giữ chính xác `np.argsort(..., kind="stable")`, `np.isclose` và tie rule cũ: chọn phương án có nhiều nhãn được quyết định hơn.
3. Có hai `linear_boundary` để bảo toàn tương thích số học:
   - MLC-PA cũ dùng `minimum_loss`: `min(p, 1-p) <= c`;
   - GSI cũ dùng `symmetric_thresholds`: `p <= c` hoặc `p >= 1-c`.
   Hai biểu thức tương đương về toán nhưng có thể khác tại biên floating-point như `p=0.70, c=0.30`; mode được ghi trong config/hash để không che giấu khác biệt.
4. `expected_utility` của Hamming trả âm của expected generalized-Hamming loss theo từng instance; giá trị lớn hơn tốt hơn.
5. Schema v2 vẫn gọi external model API như trước. Chỉ evaluator v3 được chuyển sang registry ở Q3.
6. Chưa thêm CLI chọn policy vì F1/Jaccard chưa được nối vào evaluation/selection; injection cho GSI thuộc Q6.

## Test/lệnh đã chạy + kết quả

```text
python -m unittest tests.test_decision_hamming -v
```

- PASS 5/5 test Q3.
- Hamming policy và model delegate bằng bitwise với implementation pre-Q3 trên SEP, PAR, nhiều cost, biên xác định và probability ngoài `[0,1]` sau clipping.
- GSI giữ riêng direct-threshold boundary behavior.
- Evaluator v3 pass cả khi method `classifier.predict_from_proba` bị cố ý làm lỗi, chứng minh output partial đi qua registry.

```text
python -m unittest discover -s tests -v
```

- PASS 35/35 test Q1–Q3.

```text
python tests_unit.py
```

- PASS toàn bộ 15 nhóm regression/unit/integration cũ, gồm nhóm 7, 8 và 11 cho SEP/PAR/GSI boundary.

```text
python -m compileall -q main.py src tests
git diff --check
```

- PASS syntax/import/whitespace; Git chỉ cảnh báo LF/CRLF của môi trường Windows.

## Output/cache được tạo

- Không tạo output hay cache thật trong workspace.
- Test schema v3 chỉ dùng temporary directories và tự xóa.
- Không sửa `results/`, `results_pa/` hoặc tạo `results_pa_v3/`.

## Việc chưa hoàn thành hoặc blocker

- Chưa cài F-beta policy; đây là Q4.
- Chưa cài Jaccard policy; đây là Q5.
- Chưa nối policy khác Hamming vào objective chọn IL/DL; đây là Q6.
- Không có blocker kỹ thuật cho Q4.

## Git status và thay đổi ngoài phạm vi

- Nhánh hiện tại: `Khanh_1`, tracking `origin/Khanh_1` trước các thay đổi Q1–Q3.
- `note.txt` vẫn ở trạng thái deleted có từ trước, không thuộc Q1–Q3 và không được chỉnh sửa.
- Các thay đổi Q1–Q3 chưa commit/push tại thời điểm viết handoff.

## Entry point/lệnh đầu tiên cho Phase Q4

```powershell
python -m unittest discover -s tests -v
```

Sau baseline, cài `src/decision/fbeta.py` theo Algorithm 2 của Nguyen–Hüllermeier dưới giả định conditional label independence. Dùng exhaustive enumeration trên mọi truth/action với `K <= 6` làm oracle; phải kiểm tra cả `allow_abstention=False` và partial mode, quy ước empty-set, SEP/PAR penalty, tie ưu tiên nhiều quyết định rồi stable label index. Không nối F1 policy vào GSI selection trong Q4.
