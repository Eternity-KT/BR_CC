# Handoff Phase Q0 — Audit và metric contract

Ngày hoàn thành: 2026-08-30

Phase tiếp theo: Q1 — Complete, partial, optimistic và group metrics

## Objective đã hoàn thành

- Chạy lại baseline test trước khi sửa production code.
- Audit source metric, cache schema và bốn result file hiện có.
- Chuyển note cuộc họp thành `meeting_summary.md`, tách quyết định đã chốt và câu hỏi cần xác nhận với giảng viên.
- Khóa contract schema v3 gồm tên chuẩn, scope, công thức, mẫu số, hướng đánh giá, alias và quy ước biên.
- Tạo skeleton module/facade cùng deterministic fixtures và contract tests.
- Không nối skeleton vào runner, không thay `src/evaluation/metrics.py`, cache hoặc output production.

## Files đã sửa/tạo

| File | Trạng thái Q0 |
|---|---|
| `meeting_summary.md` | Tóm tắt cuộc họp và các điểm cần xác nhận |
| `src/evaluation/metric_contract.py` | Contract schema v3, không tính metric |
| `src/evaluation/complete_metrics.py` | Skeleton API complete metrics cho Q1 |
| `src/evaluation/abstention_metrics.py` | Skeleton API selective/rejected/optimistic cho Q1 |
| `src/evaluation/group_metrics.py` | Skeleton API per-label và IL/DL group cho Q1 |
| `src/evaluation/metric_facade.py` | Facade v3 chưa nối production |
| `tests/fixtures/metric_cases.py` | Complete, partial, all-abstain, zero-support và empty-group fixtures |
| `tests/test_metric_contract.py` | Test contract, signature và ranh giới migration |
| `tests/__init__.py`, `tests/fixtures/__init__.py` | Package marker cho test mới |

`specify.md` đã được chia thành Q0–Q15 ở công việc trước Q0; mỗi phase được thiết kế cho tối đa một quota model 5 giờ.

## Audit baseline

### Runtime

- Python `3.13.14`
- NumPy `2.3.4`
- pandas `2.3.3`
- scikit-learn `1.7.2`

### Source và cache

- Production facade hiện vẫn là `src/evaluation/__init__.py` trỏ tới `metrics.py`.
- `src/evaluation/cache.py` dùng `CACHE_SCHEMA_VERSION = 2`.
- Cache v2 cố ý không persist `Macro Precision` và `Macro Recall` qua `UNPERSISTED_METRICS`; Q2 phải bỏ/migrate quy tắc này khi schema v3 được nối.
- Production `compute_all_metrics` vẫn trả đúng năm field v2: `Macro-F1`, `Micro-F1`, `Hamming Loss`, `Subset Accuracy`, `Example-F1`.

### Result files

| File | Audit |
|---|---|
| `results/tables/raw_results.json` | Legacy dataset-first; 139,331 bytes; có đủ 10 dataset |
| `results_pa/tables/raw_results.json` | Legacy dataset-first; 15,084,346 bytes; có đủ 10 dataset |
| `results_pa/tables/MLC_PA.json` | Schema 2; 227,118 bytes; Logistic; có 9/10 dataset, thiếu `genbase` |
| `results_pa/tables/GSI_MLC_PA.json` | Schema 2; 14,726,585 bytes; BR-MLP và CC-MLP; objective `complete_macro_f1`; có đủ 10 dataset |

Không file result/cache nào được sửa hoặc tạo trong Q0.

## Quyết định kỹ thuật và giả định

1. Core BR/CC/GSI và xác suất hiện tại được giữ nguyên; metric v3 là module mới.
2. Input metric hợp lệ phải là ma trận `N x K` không rỗng; `N == 0` hoặc `K == 0` sẽ là `ValueError` ở Q1.
3. Positive-class Macro-F1 là trung bình F1 từng nhãn, không phải harmonic mean của Macro Precision/Recall; `zero_division=0`.
4. Instance-F1/Jaccard của một instance có true set và predicted set đều rỗng bằng `1.0`.
5. All-abstain: Selective Hamming Accuracy và Risk at Coverage là `NaN`; Selective Micro-F1 bằng `0`; mỗi nhãn không có quyết định đóng góp `0` vào Selective Macro-F1.
6. Group rỗng: `Group Label Count = 0`, các group metric còn lại là `NaN`.
7. Rejected metric có mẫu số bằng `0` trả `NaN` và luôn xuất raw counts.
8. AURC dùng ranking label-position theo acceptance confidence; tính mean risk trên mọi prefix `m=1..N*K`; tie giữ thứ tự row-major ổn định.
9. Optimistic metric là oracle upper bound, không phải automatic score; Oracle Gain chỉ lấy hiệu với Full metric cùng mẫu số.
10. Alias đổi tên và derived compatibility field được tách riêng để tránh coi `Selective Hamming Loss` là alias số học trực tiếp của accuracy.

## Test/lệnh đã chạy + kết quả

```text
python tests_unit.py
```

- PASS toàn bộ 15 nhóm unit/integration hiện có.
- Kết thúc bằng `ALL UNIT TESTS PASSED SUCCESSFULLY!`.

```text
python -m unittest tests.test_metric_contract -v
```

- PASS 5/5 contract tests.
- Xác nhận tên/definition nhất quán, fixtures hợp lệ, skeleton signatures ổn định và production facade chưa đổi.

```text
python -m compileall -q src\evaluation tests
git diff --check
```

- PASS, không có lỗi syntax/import hoặc whitespace; Git chỉ cảnh báo line-ending LF/CRLF của `specify.md`.

## Output/cache được tạo

- Chỉ tạo source skeleton, test fixtures và tài liệu Q0 nêu trên.
- Không chạy benchmark dataset, không tạo model artifact, cache hay result schema v3.

## Việc chưa hoàn thành hoặc blocker

- Các hàm trong bốn module v3 đang chủ động ném `NotImplementedError`; đây là exit state của Q0, Q1 sẽ cài đặt.
- Chưa thay production facade và chưa migrate cache/schema; để dành cho Q2 sau khi unit tests Q1 đầy đủ.
- Còn phải xác nhận với giảng viên ý nghĩa “instant base F1”, “cả hai tập”, critical labels và domain cost; Q1 có thể triển khai metric tổng quát mà chưa cần các câu trả lời này.
- Không có blocker kỹ thuật để bắt đầu Q1.

## Git status và thay đổi ngoài phạm vi

- `note.txt` đang ở trạng thái deleted từ trước và không thuộc Q0; không khôi phục hoặc chỉnh sửa.
- `specify.md` là thay đổi lập phase từ yêu cầu trước; Q0 không đảo lại.
- Tất cả file Q0 khác là file mới, chưa commit.

## Entry point/lệnh đầu tiên cho Phase Q1

```powershell
python -m unittest tests.test_metric_contract -v
```

Sau baseline, bắt đầu ở `src/evaluation/complete_metrics.py`: cài validation dùng chung và complete metrics cho fixture `complete_metric_case()`, rồi mới triển khai partial/oracle/group để giữ phạm vi kiểm thử nhỏ.
