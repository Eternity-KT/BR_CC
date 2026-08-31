# Handoff Phase Q1 — Complete, partial, optimistic và group metrics

Ngày hoàn thành: 2026-08-31

Phase tiếp theo: Q2 — Pipeline integration, output schema v3 và resumability

## Objective đã hoàn thành

- Cài complete metrics theo contract v3: Macro/Micro-F1, Hamming Accuracy/Loss, Subset Accuracy, Instance-F1/Jaccard và Macro Precision/Recall.
- Cài selective metrics, generalized loss cho linear/concave penalty, coverage, ABS/AABS, risk-at-coverage và AURC khi có acceptance confidence.
- Cài rejected counterfactual Macro-F1, rejected error rate, error capture và các raw count bắt buộc.
- Cài optimistic/oracle completion, năm optimistic metrics và Oracle Gain trên cùng mẫu số Full.
- Cài per-label support/prevalence, full/selective quality, coverage và rejected diagnostics.
- Cài named-group metrics cho IL/DL cùng quy ước group rỗng.
- Hoàn thiện facade v3 có scope rõ ràng, critical-label hook và lookup alias `Example-F1` mà không tạo cột trùng.
- Giữ nguyên `src/evaluation/__init__.py`, `metrics.py`, cache schema v2, runner và toàn bộ result hiện có.

## Files đã sửa/tạo

| File | Thay đổi Q1 |
|---|---|
| `src/evaluation/metric_utils.py` | Validation ma trận, confusion counts, positive PRF và instance set-score dùng chung |
| `src/evaluation/complete_metrics.py` | Implementation complete metrics |
| `src/evaluation/abstention_metrics.py` | Implementation Selective/Rejected/Optimistic/Diagnostics |
| `src/evaluation/group_metrics.py` | Implementation per-label và named-group metrics |
| `src/evaluation/metric_facade.py` | Structured bundle, alias và critical-label hook |
| `src/evaluation/metric_contract.py` | Ghi rõ AURC là `NaN` nếu thiếu acceptance confidence |
| `tests/test_metrics_v3.py` | 15 reference/edge-case tests cho Q1 |
| `tests/test_metric_contract.py` | Đổi mô tả từ skeleton Q0 sang migration boundary Q2 |
| `specify.md` | Đánh dấu Q0 và Q1 đã hoàn thành |

## Quyết định kỹ thuật và giả định

1. Facade Q1 trả bundle có các key: `Full`, `Selective`, `Rejected`, `Optimistic`, `Diagnostics`, `Per Label`, `Groups`, `Critical Labels` và contract version.
2. `Full` và `Selective` dùng `AliasAwareMetricDict`: lookup tên cũ hoạt động nhưng khi iterate/serialize chỉ có canonical names. `Example-F1` trỏ tới `Instance-F1`; không có cột số liệu thứ hai.
3. Khi facade không nhận `y_partial`, `y_full` được xem như no-abstention policy với cost `0`; nếu có `y_partial` thì cost phải được truyền tường minh.
4. Generalized Loss giữ đúng hai penalty đã có: `c*a` cho linear và `c*K*a/(K+a)` cho concave, chuẩn hóa bởi `N*K`.
5. `Risk at Coverage` dùng lỗi trên decided positions của operating point hiện tại. AURC dùng lỗi của `y_full` sau khi rank mọi label-position theo acceptance confidence giảm dần; tie giữ row-major order ổn định. Không có confidence thì AURC là `NaN`.
6. Rejected counterfactual Macro-F1 tính positive-class F1 trên rejected positions từng nhãn; nhãn không có rejected position là `NaN` và bị loại khỏi macro aggregation. Nếu không nhãn nào có rejected position, aggregate là `NaN`.
7. Optimistic completion chỉ thay vị trí abstain bằng ground truth; quyết định sai đã chấp nhận vẫn giữ nguyên.
8. Critical-label hook trả `Status = N/A` nếu chưa có domain config, không tự suy ra nhãn quan trọng từ prevalence/test set.
9. Named groups có thể tổng quát hơn IL/DL. Group rỗng trả count `0`, mọi group metric khác là `NaN`; index sai, trùng hoặc ngoài khoảng bị từ chối.
10. Với `K=1` toàn âm, contract positive-label Macro-F1 trả `0`. `sklearn` có thể suy luận input `1x1` là binary-class và cho kết quả khác khi `average="macro"`; reference test sklearn vì thế dùng multilabel indicator `K>=2` và có test riêng cho quy ước này.

## API shape chuyển cho Q2

```python
bundle = compute_metric_bundle(
    y_true,
    y_full,
    y_partial=y_partial,
    cost=cost,
    penalty="linear",
    acceptance_confidence=confidence,
    label_names=label_names,
    label_groups={"IL": il_indices, "DL": dl_indices},
    critical_labels=critical_labels,
)

bundle["Full"]
bundle["Selective"]
bundle["Rejected"]
bundle["Optimistic"]
bundle["Diagnostics"]
bundle["Per Label"]
bundle["Groups"]
bundle["Critical Labels"]
```

Facade này import trực tiếp từ `src.evaluation.metric_facade`; chưa export qua `src.evaluation` để tránh migration nửa chừng.

## Test/lệnh đã chạy + kết quả

```text
python -m unittest discover -s tests -v
```

- PASS 20/20 tests: 5 contract/migration-boundary và 15 Q1 metric tests.
- Bao phủ perfect, hand-computed, random sklearn reference, invalid input, all-abstain, no-abstain, no-error, zero denominator, zero support, oracle completion, linear/concave loss, AURC, empty group, per-label, alias và critical-label hook.

```text
python tests_unit.py
```

- PASS toàn bộ 15 nhóm regression/unit/integration cũ.
- Xác nhận production schema-v2 metrics, cache, model decision rules, GSI validation/refit và plots không bị ảnh hưởng.

```text
python -m compileall -q src\evaluation tests
git diff --check
```

- PASS syntax/import; kiểm tra diff không có whitespace error. Cảnh báo LF/CRLF của Git không làm thay đổi nội dung.

## Output/cache được tạo

- Không chạy dataset benchmark và không tạo/sửa result, cache hoặc model artifact.
- Chỉ tạo source/test/documentation cho metric layer v3.

## Việc chưa hoàn thành hoặc blocker

- Metric bundle v3 chưa được gọi từ `main._evaluate_model`; production vẫn xuất schema v2. Đây là phạm vi Q2.
- Chưa có output directory/schema v3, config hash, fold checkpoint hoặc migration guard. Đây là phạm vi Q2.
- Runner hiện chưa cung cấp acceptance confidence, label names, IL/DL indices và critical-label config cho facade; Q2 phải map các nguồn này, cho phép `AURC = NaN` khi confidence chưa khả dụng.
- Chưa định nghĩa JSON representation chuẩn cho `NaN`; Q2 phải chọn `null` hoặc metadata validity thay vì dựa vào JSON `NaN` không chuẩn.
- Không có blocker kỹ thuật để bắt đầu Q2.

## Git status và thay đổi ngoài phạm vi

- Nhánh hiện tại: `Khanh_1`, tracking `origin/Khanh_1` trước các thay đổi Q1.
- `note.txt` vẫn ở trạng thái deleted có từ trước, không thuộc Q1 và không được chỉnh sửa.
- Các thay đổi Q1 chưa commit/push tại thời điểm viết handoff này.

## Entry point/lệnh đầu tiên cho Phase Q2

```powershell
python -m unittest discover -s tests -v
```

Sau baseline, audit `main.py::_evaluate_model` và `src/evaluation/cache.py`. Tạo schema-v3 serializer/checkpoint tách biệt trước, sau đó mới nối `compute_metric_bundle`; không ghi đè `results_pa/` hoặc cache schema v2.
