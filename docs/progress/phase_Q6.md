# Handoff Phase Q6 — Objective injection cho GSI

**Trạng thái:** Hoàn thành ngày 2026-08-31.

## Objective đã hoàn thành

- Tạo selection-objective registry độc lập với core GSI, gồm `full_macro_f1`, `immediate_instance_f1`, `bop_instance_f1`, `bop_jaccard`, `macro_precision`, `macro_recall`, `f_beta_0_5` và `f_beta_2`.
- Inject objective vào bước chọn IL/DL partition: mỗi candidate sinh xác suất trên inner validation, đi qua decision policy tương ứng rồi mới được chấm objective thực tế.
- Giữ `full_macro_f1` làm mặc định và tái tạo đúng partition/score của regression fixture trước Q6.
- Tách policy dùng để chấm selection candidate khỏi policy dùng cho dự đoán cuối, để có thể đổi objective mà không làm thay đổi ngầm cấu hình inference.
- Lưu selection history/config đủ objective, policy, beta, cost, penalty và inner-split seed để audit và tái lập.
- Thêm leakage spy test chứng minh selection objective chỉ nhận inner-validation target, không nhận outer-test target.
- Nối cấu hình mới vào CLI, model factory, schema-v2 cache key và schema-v3 run/fold metadata.
- Không thêm partition mode/ablation của Q7 và không thay thuật toán học xác suất cốt lõi của GSI.

## Files đã sửa/tạo

| File | Thay đổi Q6 |
|---|---|
| `src/selection/objectives.py` | Objective registry, policy mapping, scoring và diagnostics |
| `src/selection/__init__.py` | Public exports cho selection layer |
| `src/decision/configuration.py` | Adapter tạo Hamming/F-beta/Jaccard policy từ một cấu hình thống nhất |
| `src/decision/__init__.py` | Export decision-policy adapter |
| `src/models/gsi_mlc_pa.py` | Injection point, candidate evaluation, history/config và final decision delegation |
| `src/evaluation/pipeline_v3.py` | Truyền cấu hình Q6, hash run config và lưu fold metadata |
| `main.py` | CLI, factory, cache settings và output metadata |
| `tests/test_selection_objectives.py` | 11 test registry, regression, smoke, policy, cache và leakage |
| `specify.md` | Đánh dấu Q6 hoàn thành |

## Quyết định kỹ thuật và compatibility contract

1. `selection_objective` quyết định cách chấm candidate partition trên inner validation; `decision_policy` quyết định action cuối sau khi mô hình đã fit lại trên toàn bộ outer-train. Hai cấu hình được cache riêng để tránh nhập nhằng.
2. Mapping objective sang selection policy:

   | Objective | Policy chấm candidate | Abstention |
   |---|---|---|
   | `full_macro_f1` | Hamming threshold 0.5 | Không |
   | `macro_precision`, `macro_recall` | Hamming threshold 0.5 | Không |
   | `immediate_instance_f1` | F-beta, beta 1 | Không |
   | `bop_instance_f1` | F-beta, beta 1 | Có |
   | `bop_jaccard` | Jaccard | Có |
   | `f_beta_0_5`, `f_beta_2` | F-beta tương ứng | Có |

3. Objective dùng score thực tế trên inner-validation labels sau decision policy. Với partial action, generalized score là utility trên tập label đã quyết định trừ SEP/PAR abstention penalty; diagnostics lưu thêm coverage, predicted-positive rate, số abstention trung bình và penalty trung bình.
4. Default `full_macro_f1` vẫn dùng `(proba >= 0.5)` và macro-F1 như implementation trước Q6. Alias legacy `complete_macro_f1` được canonicalize nội bộ; default schema-v2 cache setting cũ được giữ nguyên để không vô hiệu hóa cache hiện hữu.
5. `cost` chỉ ảnh hưởng các selection objective có abstention; complete objectives không đổi theo cost.
6. Candidate flow giữ nguyên split discipline:

   ```text
   outer-train -> inner-train + inner-validation
               -> fit candidate trên inner-train
               -> probabilities trên inner-validation
               -> selection policy -> objective score
               -> chọn/freeze partition
               -> refit trên toàn outer-train
               -> predict outer-test bằng final decision policy
   ```

7. Selection history giữ các field cũ và bổ sung `selection_objective`, `selection_policy`, `beta`, `policy_config`, `objective_diagnostics`, `configured_cost`, `penalty`, `inner_split_seed` và `final_decision_policy`.
8. Các decision formula vẫn nằm trong `src/decision`; GSI selector chỉ gọi adapter/objective registry, không chứa bản sao công thức Hamming, F-beta hay Jaccard.

## Cấu hình CLI

Ví dụ đổi selection objective và final decision policy:

```powershell
python main.py --models GSI_MLC_PA --gsi_selection_objective bop_jaccard --gsi_decision_policy jaccard --gsi_penalty concave
```

Các option mới:

```text
--gsi_selection_objective {full_macro_f1,immediate_instance_f1,bop_instance_f1,bop_jaccard,macro_precision,macro_recall,f_beta_0_5,f_beta_2}
--gsi_decision_policy {hamming,fbeta,jaccard}
--gsi_beta FLOAT
--gsi_penalty {linear,concave}
```

## Test/lệnh đã chạy + kết quả

```text
python -m unittest tests.test_selection_objectives -v
```

- PASS 11/11 test Q6.
- Default regression giữ đúng learned partition `IL={0}`, `DL={1,2}` và 4 candidate history records.
- Tất cả objective tạo đúng selection policy/history metadata.
- Spy test trên full fit chỉ quan sát inner-validation target shape `(10, 3)`; thay outer prediction input không đổi selection history.

```text
python -m unittest discover -s tests -v
```

- PASS 62/62 test Q1–Q6.

```text
python tests_unit.py
```

- PASS toàn bộ 15 nhóm legacy regression/unit/integration.

```text
python -m compileall -q main.py src tests
python main.py --help
git diff --check
```

- PASS syntax/import và CLI exposure.
- `git diff --check` không có lỗi whitespace; Git chỉ cảnh báo LF sẽ được chuyển thành CRLF trên một số file trong môi trường Windows.

## Output/cache được tạo

- Không chạy benchmark dataset thật và không tạo `results_pa_v3/`.
- Tests dùng dữ liệu tổng hợp/in-memory hoặc temporary directories và tự dọn dẹp.
- Không sửa output nghiên cứu hiện hữu.

## Việc chưa hoàn thành hoặc blocker

- Chưa có `partition_provider` cho `all_il`, `all_dl`, `fixed`, `random_matched` hoặc learned variants; đây là Q7.
- Chưa xuất group metrics IL/DL, partition stability hay selection/inference timing; đây là Q7.
- Chưa chạy smoke ablation trên `emotions`; đây là Q7.
- Không có blocker kỹ thuật cho Q7.

## Git status và thay đổi ngoài phạm vi

- Nhánh hiện tại: `Khanh_1`; `HEAD` và `origin/Khanh_1` đang cùng ở commit `51e30a0` trước Q6.
- Thay đổi Q6 hiện chưa commit/push.
- `note.txt` vẫn ở trạng thái deleted có từ trước Q6, không thuộc phase này và không được chỉnh sửa.

## Entry point/lệnh đầu tiên cho Phase Q7

```powershell
python -m unittest discover -s tests -v
```

Sau baseline, tạo module `src/selection/partition.py` (hoặc package tương đương) để cung cấp cùng một interface cho `learned`, `all_il`, `all_dl`, `fixed` và `random_matched`. Giữ probability model/decision policy của Q6 nguyên vẹn; partition provider chỉ chịu trách nhiệm tạo/freeze IL-DL partition và order. Thêm invariant tests cho empty groups, fixed partition, deterministic random seed và same-order trước khi chạy smoke ablation nhỏ trên `emotions`.
