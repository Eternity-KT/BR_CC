# Handoff Phase Q7 — Partition modes và IL/DL ablation infrastructure

**Trạng thái:** Hoàn thành ngày 2026-09-02.

## Objective đã hoàn thành

- Tạo partition-provider module độc lập cho `learned`, `all_il`, `all_dl`, `fixed` và `random_matched`.
- Thêm mode `learned_no_correlation_order` và final-order controls `correlation`, `selection`, `natural` để tách hiệu ứng partition khỏi reorder.
- Giữ nguyên probability inference và decision-policy layer của GSI; provider chỉ tạo/freeze IL/DL partition.
- `random_matched` dùng seed xác định và giữ đúng số IL của learned reference theo từng fold.
- Nối partition config vào model factory, CLI, schema-v2 compatibility settings và schema-v3 scientific config hash.
- Xuất group metrics IL/DL, partition size, pairwise IL Jaccard stability, selection time và probability-inference time.
- Thêm `partition_audit.csv` cùng fold/model metadata đủ partition, order và timing.
- Chạy smoke ablation thật trên `emotions`, 2 folds, GSI-MLP, cùng natural final order cho sáu modes.

## Files đã sửa/tạo

| File | Thay đổi Q7 |
|---|---|
| `src/selection/partition.py` | Partition registry/provider, validation, deterministic random-matched và provenance |
| `src/selection/__init__.py` | Export partition API |
| `src/models/gsi_mlc_pa.py` | Partition injection, learned reference, order control và selection timing |
| `src/evaluation/pipeline_v3.py` | Partition config hash, fold audit, stability và timing summary |
| `src/evaluation/export_v3.py` | Xuất `partition_audit.csv` |
| `main.py` | Factory/cache/CLI, evaluator metadata và inference timing |
| `tests/test_partition_modes.py` | 8 test Q7 cho modes, invariants, audit và export |
| `specify.md` | Đánh dấu Q7 hoàn thành |

## Partition và order contract

| Mode | Cách tạo IL | Cần learned reference | Order mặc định |
|---|---|---:|---|
| `learned` | Greedy inner-validation selection của Q6 | Có | `correlation` |
| `learned_no_correlation_order` | Cùng partition với `learned` | Có | ép về `selection` |
| `all_il` | Mọi label thuộc IL | Không | theo `--gsi_final_order` |
| `all_dl` | Mọi label thuộc DL | Không | theo `--gsi_final_order` |
| `fixed` | Danh sách zero-based do config cung cấp | Không | theo `--gsi_final_order` |
| `random_matched` | Sample không hoàn lại, cùng số IL với learned | Có | theo `--gsi_final_order` |

Các invariant:

1. IL và DL rời nhau, hợp bằng toàn bộ `0..K-1`; empty IL và empty DL đều hợp lệ.
2. `fixed` validate kiểu số nguyên, duplicate và out-of-range; danh sách rỗng biểu diễn all-DL có chủ đích.
3. `random_matched` dùng `numpy.random.default_rng(random_state)` và lưu `reference_independent_count`.
4. Explicit legacy `order` permutation vẫn có precedence. Nếu không có explicit order, `final_order` chọn correlation/selection/natural.
5. `learned_no_correlation_order` luôn giữ selection order, ngay cả khi requested strategy là correlation; requested/effective strategy đều được audit.
6. Static/random partitions vẫn được chấm qua selection-objective API của Q6 trên inner validation để validation score có cùng semantics.
7. Correlation chỉ được tính sau khi partition đã freeze; outer test không tham gia partition, order hay objective.

## Output/audit contract

Schema-v3 `Model Metadata` và `Partition Audit` lưu:

- partition mode, IL/DL labels và counts;
- learned-reference labels/count khi mode cần reference;
- selection, correlation và final order;
- requested/effective final-order strategy;
- selection seconds và probability-inference seconds;
- pairwise IL Jaccard stability giữa các fold;
- group metrics IL/DL ở cả full và từng partial operating cost.

`partition_audit.csv` có một dòng cho mỗi model/dataset/fold. Schema-v2 mặc định vẫn tương thích cache Q6; Q7 keys chỉ được thêm vào v2 settings khi dùng cấu hình partition/order không mặc định.

## CLI

```text
--gsi_partition_mode {learned,learned_no_correlation_order,all_il,all_dl,fixed,random_matched}
--gsi_fixed_independent_labels [INDEX ...]
--gsi_final_order {correlation,selection,natural}
```

Ví dụ fixed partition không confound reorder:

```powershell
python main.py --datasets emotions --models GSI_MLC_PA --result_schema 3 `
  --gsi_partition_mode fixed --gsi_fixed_independent_labels 0 1 2 `
  --gsi_final_order natural --output_dir results_pa_v3_fixed
```

## Test/lệnh đã chạy + kết quả

```text
python -m unittest tests.test_partition_modes -v
```

- PASS 8/8 test Q7.
- Bao phủ provider registry, empty groups, invalid fixed labels, deterministic random seed, matched IL count, sáu modes qua cùng fit/evaluation API, no-correlation reorder, same-order invariants, stability summary và CSV export.

```text
python -m unittest discover -s tests -v
```

- PASS 70/70 test Q1–Q7.

```text
python tests_unit.py
```

- PASS toàn bộ 15 nhóm legacy regression/unit/integration.

```text
python -m compileall -q main.py src tests
python main.py --help
git diff --check
```

- PASS syntax/import, CLI exposure và whitespace check; Git chỉ cảnh báo LF/CRLF của môi trường Windows.

## Smoke ablation `emotions`

Cấu hình chung:

```text
dataset=emotions, folds=2, model=GSI_MLC_PA, base=MLP,
cost=0.3, final_order=natural, random_state=42, schema=3
```

| Mode | Folds | Mean Full Macro-F1 | Mean IL | IL stability | Mean selection s | Mean inference s |
|---|---:|---:|---:|---:|---:|---:|
| `learned` | 2/2 | 0.643144 | 3.5 | 0.400 | 4.3759 | 0.00210 |
| `learned_no_correlation_order` | 2/2 | 0.643144 | 3.5 | 0.400 | 1.0332 | 0.00182 |
| `all_il` | 2/2 | 0.643605 | 6.0 | 1.000 | 1.0297 | 0.00087 |
| `all_dl` | 2/2 | 0.621085 | 0.0 | 1.000 | 1.0241 | 0.00340 |
| `fixed` (`0,1,2`) | 2/2 | 0.634437 | 3.0 | 1.000 | 1.0119 | 0.00170 |
| `random_matched` | 2/2 | 0.628521 | 3.5 | 0.400 | 1.0501 | 0.00192 |

Audit script xác nhận:

```text
AUDIT_PASS learned_counts=2,5 random_counts=2,5 final_order=0,1,2,3,4,5
```

Timing chỉ là smoke trên máy phát triển, chịu ảnh hưởng warm-up/run order và không dùng làm claim hiệu năng trong báo cáo. Smoke output nằm ngoài repo tại thư mục tạm `C:\Users\ADMIN\AppData\Local\Temp\BR_CC_Q7_smoke_0d4b7fbf108a4904a61d4989f6b68e36`; không đụng `results/` hoặc `results_pa/`.

## Việc chưa hoàn thành hoặc blocker

- Chưa chạy random-matched 30 seeds hay full 10-dataset ablation; đây thuộc Q12.
- GSI vẫn dùng BR-MLP + CC-MLP cố định; matched Logistic/MLP registry là Q8.
- Chưa có calibrated SVM, deployment metrics hoặc final statistical analysis; lần lượt là Q9, Q10 và Q13.
- Không có blocker kỹ thuật cho Q8.

## Git status và thay đổi ngoài phạm vi

- Q7 bắt đầu từ commit `e83e409` trên nhánh `Khanh_1` và được gom trong một phase commit riêng sau handoff này.
- `note.txt` vẫn ở trạng thái deleted có từ trước Q7, không thuộc phase và không được stage.
- Không commit generated smoke cache/output.

## Entry point/lệnh đầu tiên cho Phase Q8

```powershell
python -m unittest discover -s tests -v
```

Sau baseline, tạo shared base-learner registry/factory và `configs/experiment.json`. Đăng ký đúng tám matched IDs cho BR, CC, MLC-PA và GSI-MLC-PA trên Logistic/MLP, giữ aliases cũ nhưng loại silent MLP fallback. Trước khi đổi pipeline, viết invariant test để BR và MLC-PA dùng cùng base có cùng full marginal probabilities/predictions trên tiny data.
