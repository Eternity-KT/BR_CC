"""Generate a concise evidence-backed Markdown report for v6.3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]

NEW_MODELS = [
    "GSI_v6_3_1L_ResidualBR",
    "GSI_v6_3_CC_X",
    "GSI_v6_3_PCC_X",
    "GSI_v6_3_CC_X_IL",
    "GSI_v6_3_PCC_X_IL",
]

def _read(path):
    return pd.read_csv(path, na_values=["-"], keep_default_na=True)


def _format_table(frame, digits=4):
    display = frame.copy()
    numeric = display.select_dtypes(include=[np.number]).columns
    display[numeric] = display[numeric].round(digits)
    # Keep the report generator self-contained.  ``DataFrame.to_markdown``
    # requires the optional ``tabulate`` package, which is not part of the
    # project's runtime environment.
    columns = [str(column) for column in display.columns]

    def cell(value):
        if pd.isna(value):
            rendered = "-"
        else:
            rendered = str(value)
        return rendered.replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(cell(column) for column in columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(cell(value) for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def generate(config_path: Path):
    with config_path.open("r", encoding="utf-8") as stream:
        config = json.load(stream)
    settings = config["settings"]
    root = WORKSPACE_ROOT / settings["output_dir"]
    tables = root / "all" / "hamming_bop" / "tables"
    complete = _read(tables / "complete_metric.csv")
    selective = _read(tables / "selective_metric.csv")
    calibration = _read(tables / "calibration_metric.csv")
    performance = _read(tables / "model_performance.csv")
    layers = _read(tables / "layer_performance.csv")

    complete_fold = complete[
        (complete["aggregation"] == "fold") & complete["model"].isin(NEW_MODELS)
    ].copy()
    selective["cost"] = pd.to_numeric(selective["cost"], errors="coerce")
    selective_fold = selective[
        (selective["aggregation"] == "fold")
        & selective["model"].isin(NEW_MODELS)
        & np.isclose(selective["cost"], settings["report_cost"], equal_nan=False)
    ].copy()
    calibration_fold = calibration[
        (calibration["aggregation"] == "fold")
        & calibration["model"].isin(NEW_MODELS)
    ].copy()

    all_complete_fold = complete[complete["aggregation"] == "fold"].copy()
    all_selective_fold = selective[
        (selective["aggregation"] == "fold")
        & np.isclose(selective["cost"], settings["report_cost"], equal_nan=False)
    ].copy()
    descriptive_summary = (
        all_complete_fold.groupby(["base_learner", "model"], sort=False)[
            ["macro_f1", "micro_f1", "hamming_accuracy"]
        ]
        .mean()
        .join(
            all_selective_fold.groupby(["base_learner", "model"], sort=False)[
                ["selective_macro_f1", "generalized_loss", "coverage"]
            ].mean()
        )
        .reset_index()
    )

    summary = (
        complete_fold.groupby(["base_learner", "model"], sort=False)[
            ["macro_f1", "micro_f1", "hamming_accuracy"]
        ]
        .mean()
        .join(
            selective_fold.groupby(["base_learner", "model"], sort=False)[
                ["selective_macro_f1", "generalized_loss", "coverage"]
            ].mean()
        )
        .join(
            calibration_fold.groupby(["base_learner", "model"], sort=False)[
                ["brier_score", "ece"]
            ].mean()
        )
        .reset_index()
    )

    pair_keys = ["base_learner", "dataset", "fold"]
    combined = complete_fold[pair_keys + ["model", "macro_f1"]].merge(
        selective_fold[
            pair_keys + ["model", "selective_macro_f1", "generalized_loss", "coverage"]
        ],
        on=pair_keys + ["model"],
        how="inner",
    )
    reference = combined[combined["model"] == "GSI_v6_3_1L_ResidualBR"].drop(
        columns="model"
    )
    deltas = []
    for model in NEW_MODELS[1:]:
        candidate = combined[combined["model"] == model].drop(columns="model")
        paired = candidate.merge(reference, on=pair_keys, suffixes=("", "_reference"))
        deltas.append(
            {
                "model": model,
                "paired_folds": len(paired),
                "delta_macro_f1": (paired["macro_f1"] - paired["macro_f1_reference"]).mean(),
                "delta_selective_macro_f1": (
                    paired["selective_macro_f1"]
                    - paired["selective_macro_f1_reference"]
                ).mean(),
                "delta_generalized_loss": (
                    paired["generalized_loss"] - paired["generalized_loss_reference"]
                ).mean(),
                "delta_coverage": (paired["coverage"] - paired["coverage_reference"]).mean(),
            }
        )
    delta_frame = pd.DataFrame(deltas)

    comparisons = [
        ("PCC_X − CC_X", "GSI_v6_3_PCC_X", "GSI_v6_3_CC_X"),
        ("PCC_X_IL − CC_X_IL", "GSI_v6_3_PCC_X_IL", "GSI_v6_3_CC_X_IL"),
        ("CC_X_IL − CC_X", "GSI_v6_3_CC_X_IL", "GSI_v6_3_CC_X"),
        ("PCC_X_IL − PCC_X", "GSI_v6_3_PCC_X_IL", "GSI_v6_3_PCC_X"),
    ]
    direct_rows = []
    for name, left_model, right_model in comparisons:
        left = combined[combined["model"] == left_model].drop(columns="model")
        right = combined[combined["model"] == right_model].drop(columns="model")
        paired = left.merge(right, on=pair_keys, suffixes=("_left", "_right"))
        direct_rows.append(
            {
                "comparison": name,
                "paired_folds": len(paired),
                "delta_macro_f1": (
                    paired["macro_f1_left"] - paired["macro_f1_right"]
                ).mean(),
                "delta_selective_macro_f1": (
                    paired["selective_macro_f1_left"]
                    - paired["selective_macro_f1_right"]
                ).mean(),
                "delta_generalized_loss": (
                    paired["generalized_loss_left"]
                    - paired["generalized_loss_right"]
                ).mean(),
            }
        )
    direct_frame = pd.DataFrame(direct_rows)

    per_dataset = (
        selective_fold.groupby(["base_learner", "dataset", "model"], sort=False)[
            "selective_macro_f1"
        ]
        .mean()
        .reset_index()
    )
    best = (
        per_dataset.sort_values("selective_macro_f1", ascending=False)
        .groupby(["base_learner", "dataset"], sort=False)
        .head(1)
        .sort_values(["base_learner", "dataset"])
    )

    layer_fold = layers[
        (layers["aggregation"] == "fold")
        & layers["model"].isin(NEW_MODELS)
        & (layers["layer_type"] == "IL")
    ].copy()
    il_summary = (
        layer_fold.groupby(["base_learner", "model"], sort=False)["label_count"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    inherited_missing = {}
    for filename, frame in (
        ("complete_metric.csv", complete),
        ("selective_metric.csv", selective),
        ("calibration_metric.csv", calibration),
        ("layer_performance.csv", layers),
    ):
        inherited = frame[
            frame["provenance_status"].astype(str).str.startswith("inherited")
        ]
        inherited_missing[filename] = int(inherited.isna().sum().sum())

    lines = [
        "# Báo cáo ablation GSI-MLC-PA v6.3",
        "",
        "Ngày tạo: 2026-10-12  ",
        "Cấu hình: 10 dataset, 3 base learner, outer 5-fold; PCC dùng 1.000 ancestral samples cho mỗi test x; CC/PCC dùng chung chain order trong từng fold.",
        "",
        "## 1. Trạng thái thực nghiệm",
        "",
        f"- Fold mô hình mới có complete metric: {len(complete_fold)} / 750.",
        f"- Fold mô hình mới có selective metric tại cost={settings['report_cost']}: {len(selective_fold)} / 750.",
        "- BR, CC, MLC_PA và GSI_v6_2 được kế thừa từ v6.2, không chạy lại.",
        "- BR và CC không có cơ chế từ chối nhưng vẫn được giữ làm đường tham chiếu: selective metrics của chúng lấy từ complete metrics, coverage=1 và không đổi theo cost.",
        "- Các kết quả kế thừa mang `inherited_unmatched`: artifact nguồn không có prediction/split hash để chứng minh ghép cặp fold; vì vậy không dùng chúng cho kết luận paired.",
        "",
        "## 2. Trung bình các fold và dataset của mô hình mới",
        "",
        _format_table(summary),
        "",
        "## 3. Đối chiếu mô tả với các baseline kế thừa",
        "",
        "Bảng này hỗ trợ đọc RQ1 ở mức mô tả tổng hợp. Các hàng v6.2 là `inherited_unmatched`, vì vậy không được diễn giải như so sánh paired hay bằng chứng ý nghĩa thống kê. Dấu `-` phản ánh metric không có trong artifact nguồn.",
        "",
        _format_table(descriptive_summary),
        "",
        "## 4. Chênh lệch ghép cặp so với đối chứng single-stage ResidualBR",
        "",
        "Delta dương tốt hơn đối với F1/coverage; delta âm tốt hơn đối với generalized loss. Đây là mô tả thực nghiệm, không phải tuyên bố ý nghĩa thống kê hay quan hệ nhân quả.",
        "",
        _format_table(delta_frame),
        "",
        "## 5. So sánh trực tiếp PCC–CC và X_IL–X",
        "",
        _format_table(direct_frame),
        "",
        "## 6. Mô hình có Selective Macro-F1 cao nhất theo dataset/learner",
        "",
        _format_table(best),
        "",
        "## 7. Quy mô IL_1",
        "",
        _format_table(il_summary),
        "",
        "## 8. Tính minh bạch của dữ liệu kế thừa",
        "",
    ]
    lines.extend(
        f"- `{filename}`: {count} ô metric/metadata kế thừa bị thiếu và được xuất `-`."
        for filename, count in inherited_missing.items()
    )
    lines.extend(
        [
            "",
            "## 9. Giới hạn diễn giải",
            "",
            "- Selective Subset Accuracy dùng quy ước tạm của đặc tả: loại mẫu all-abstain khỏi mẫu số.",
            "- Không thực hiện kiểm định ý nghĩa thống kê trong báo cáo này.",
            "- So sánh paired chỉ dùng năm mô hình v6.3 mới vì chúng dùng cùng outer split và cùng partition/order trong fold.",
            "- Metric thiếu của v6.2 không được nội suy hoặc thay bằng 0.",
            "",
        ]
    )
    report = root / "v6_3_ablation_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated report: {report}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=WORKSPACE_ROOT / "configs" / "v6_3_run.json",
    )
    args = parser.parse_args()
    generate(args.config)


if __name__ == "__main__":
    main()
