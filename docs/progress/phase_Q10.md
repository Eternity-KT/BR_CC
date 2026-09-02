# Handoff Phase Q10 — Deployment metrics, label policy và visualization

**Trạng thái:** Hoàn thành ngày 2026-09-02.

## Objective đã hoàn thành

- Tạo deployment layer độc lập, không sửa công thức huấn luyện BR/CC/GSI.
- Xuất risk-at-coverage, AURC, review load theo label-position/case, rejected error rate, error capture, random-rejection expectation/lift và optimistic gain.
- Tạo/validate policy theo dataset từ `configs/label_policy.json`; critical labels chỉ đến từ config/global CLI, không suy từ prevalence hoặc outer test.
- Tính critical-label Recall/F1/Coverage, critical error capture và optimistic critical F1; khi thiếu policy trả `N/A` có lý do.
- Tính cost-sensitive utility cho reviewer accuracy mặc định `0.80/0.90/0.95/1.00` hoặc grid khai báo trong policy.
- Cài ba rule chọn operating point: minimum generalized loss, maximum utility với coverage constraint, và maximum coverage với risk constraint.
- Bộ chọn cost fit model/scaler trên inner-train và chỉ score inner-validation; outer test không xuất hiện trong API selection.
- Cache/config hash v3 gồm label-policy content hash và toàn bộ operating-point settings. Đường dẫn nguồn policy chỉ nằm trong manifest, không làm hash phụ thuộc máy.
- Thêm `c=0.5` vào default schema-v3 như no-abstention sanity point; schema-v2 giữ nguyên default cũ.
- Xuất deployment/critical CSV và đủ bảy figure ở C8; plot tạo placeholder hợp lệ nếu cache partial, model/record thiếu hoặc metric là `NaN`.

## Files đã sửa/tạo

| File | Thay đổi Q10 |
|---|---|
| `src/evaluation/deployment.py` | Deployment metrics, reviewer scenarios và leakage-safe operating-point rules |
| `src/evaluation/label_policy.py` | Loader, validation và content hash policy theo dataset |
| `configs/label_policy.json` | Schema v1 rỗng; chờ domain owner khai báo label/cost thật |
| `src/evaluation/pipeline_v3.py` | Policy/hash, inner selection, deployment summaries và figure export |
| `src/evaluation/export_v3.py` | `deployment_metrics.csv`, `critical_label_deployment.csv`, giữ plot artifacts trong manifest |
| `src/evaluation/__init__.py` | Public deployment APIs, không làm mất exports cũ |
| `src/visualization/deployment.py` | Bảy figure robust cho deployment/IL-DL/objective/calibration |
| `src/visualization/__init__.py` | Export visualization entry point |
| `main.py` | Evaluator integration, v3 cost `0.5`, label-policy/operating-point CLI |
| `tests/test_deployment.py` | 9 deterministic tests cho formula, policy, leakage, CSV/plot |
| `specify.md` | Đánh dấu Q10 hoàn thành |

## Deployment contract

Mỗi operating point giữ các mẫu số tách biệt:

```text
Immediate/Full       : toàn bộ N*K vị trí, không abstain
Selective Risk       : lỗi / số vị trí đã quyết định
Review Load Position : số vị trí abstain / (N*K) = AABS
Review Load Case     : số case có ít nhất một abstain / N = ABS
Error Capture Rate   : lỗi Full nằm trong tập review / tổng lỗi Full
Optimistic gain      : metric oracle-completed - metric Full
```

`Random-Rejection Expected Capture = AABS`; `Error Capture Lift` so error capture thực với baseline này. Optimistic là upper bound reviewer đúng 100%, không được so trực tiếp với selective metric vì khác mẫu số.

Utility được chuẩn hóa theo `N*K` và gồm:

```text
decided FP/FN cost + review_count * review_cost
                    + expected reviewer FP/FN cost at accuracy h
Cost-Sensitive Utility = - total expected cost / (N*K)
```

Các label không khai báo weight/FP/FN cost dùng `1.0`. Không có `review_cost` hoặc policy thì utility/reviewer scenarios là `N/A`, không tự điền giả định domain.

## Operating-point leakage contract

```text
outer-train
  -> deterministic inner split
  -> scaler.fit(inner-train)
  -> selector.fit(inner-train)
  -> cost candidates score(inner-validation)
  -> freeze selected cost/audit
outer-test
  -> chỉ dùng để báo cáo tất cả operating points sau khi model final fit outer-train
```

`select_operating_point` từ chối mọi `data_scope` khác `inner_validation`. Selection audit ghi inner sizes/seed, candidate records và `Outer Test Access = false`. Nếu constraint không có candidate khả thi, status là `Infeasible`; không fallback bằng cách nhìn outer test.

## Output schema Q10

CSV mới:

- `deployment_metrics.csv`: operating point và reviewer scenarios, có cột `Scope` rõ.
- `critical_label_deployment.csv`: chỉ critical-label deployment status/metrics.

Figure được lưu dưới `figures/<config_hash>/`:

- `risk_coverage.png`;
- `optimistic_gain_vs_review_load.png`;
- `error_capture_vs_review_load.png`;
- `per_label_critical_metrics.png`;
- `il_dl_ablation.png`;
- `objective_comparison.png`;
- `calibration_reliability.png`.

Complete metrics tiếp tục ở `complete_metrics.csv`; deployment/Selective không được nối vào complete table. Dataset được giữ trong series key để plot nhiều dataset không trộn trung bình ngầm.

## Test/lệnh đã chạy + kết quả

```text
python -m unittest tests.test_deployment tests.test_pipeline_v3 -v
```

- PASS 19/19 targeted Q10 + schema-v3 integration tests.

```text
python -m unittest discover -s tests -v
```

- PASS 93/93 tests Q1–Q10.

```text
python tests_unit.py
```

- PASS toàn bộ 15 nhóm legacy regression/unit/integration.

```text
python -m compileall -q main.py src tests
python main.py --help
git diff --check
```

- PASS syntax/import, CLI exposure và whitespace check; chỉ có cảnh báo LF/CRLF của Git trên Windows.

Fixture tính tay xác nhận `Coverage=0.5`, `Selective Risk=0`, `Generalized Loss=0.125`, `Error Capture Rate=1`, lift `2`, optimistic Hamming gain `0.5` và AURC `0.20833...`. Policy fixture xác nhận reviewer accuracy `0.8` cho expected cost `3.3`/utility `-0.825`, reviewer hoàn hảo cho utility `-0.125`.

## Output/cache được tạo

- Unit/integration tests dùng temporary directories và tự dọn; sinh/đọc JSON, 9 CSV artifacts và 7 PNG figures.
- Không sửa `results/`, `results_pa/` hoặc các benchmark output đã có.
- Không chạy `emotions` 12-ID benchmark trong Q10; đây là exit riêng của Q11.

## Việc chưa hoàn thành hoặc blocker

- `configs/label_policy.json` cố ý chưa có critical label/cost thật. Cần domain owner điền trước khi claim hiệu quả thực tế; hiện critical deployment là `N/A` đúng contract.
- Chưa chạy system smoke `emotions`, 2 folds, đủ 12 IDs và audit artifact tự động; thuộc Q11.
- Chưa chạy full benchmark/ablation grid; thuộc Q12.
- Không có blocker kỹ thuật cho Q11.

## Git status và thay đổi ngoài phạm vi

- Q10 bắt đầu từ commit Q9 `148bc0d` trên nhánh `Khanh_1` và được gom thành một phase commit riêng sau handoff này.
- `note.txt` vẫn ở trạng thái deleted có từ trước Q10, không thuộc phase và không được stage.
- Local branch đang đi trước `origin/Khanh_1`; chỉ push khi người dùng yêu cầu.

## Entry point/lệnh đầu tiên cho Phase Q11

```powershell
python -m unittest discover -s tests -v
python tests_unit.py
python main.py --datasets emotions --models BR_Logistic BR_MLP CC_Logistic CC_MLP MLC_PA_Logistic MLC_PA_MLP GSI_MLC_PA_Logistic GSI_MLC_PA_MLP BR_SVM CC_SVM MLC_PA_SVM GSI_MLC_PA_SVM --n_splits 2 --abstention_costs 0.3 0.5 --report_cost 0.3 --result_schema 3 --output_dir results_pa_v3_smoke
```

Sau smoke, chạy audit script cho manifest/config hashes, completed folds, strict JSON, CSV headers/rows và 7 figures; cập nhật README và freeze full-run config trước khi đánh dấu Q11 hoàn thành.
