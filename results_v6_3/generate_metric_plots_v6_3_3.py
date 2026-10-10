"""
Publication-Quality Metric Plot Generator for GSI-MLC-PA v6.3.3.
Generates comprehensive visual comparisons across ALL 6 TARGET MODELS:
  1. BR (Binary Relevance)
  2. CC (Classifier Chains)
  3. MLC-PA (Nguyen & Hullermeier, 2021)
  4. GSI v6.2.1 (Decaying Layered Peeling + Symmetric Chow)
  5. GSI v6.3.2 (OS-NMF Mean-Field Coupling)
  6. GSI v6.3.3 (Proposed: Adaptive Tri-Regime + Smooth Blending + Negative Guard)

Outputs 6 high-resolution figures (PDF and PNG at 300 DPI) into results_v6_3/figures/:
  - fig1_macro_f1_comparison.png: Selective Macro-F1 across 10 datasets
  - fig2_f1_vs_coverage_pareto.png: Pareto trade-off curve (Coverage vs Macro-F1)
  - fig3_subset_accuracy_comparison.png: Exact match (Subset 0/1 Accuracy)
  - fig4_hamming_loss_comparison.png: Hamming loss (bit-error rate)
  - fig5_focus_chd49_viruspseaac.png: In-depth focus on CHD49 and VirusPseAAC recovery
  - fig6_radar_5criteria.png: Multi-criteria radar chart (F1, Cov, SubAcc, 1-HL, Micro-F1)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Setup paths
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = WORKSPACE_ROOT / "results_v6_3"
FIGS_DIR = RESULTS_DIR / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

DECAY_CSV = WORKSPACE_ROOT / "results_v6_2" / "decay_study" / "decay_detailed_folds.csv"
V632_LOG_CSV = RESULTS_DIR / "v6_3_2_all10ds_detailed_folds.csv"
V632_SVM_MLP_CSV = RESULTS_DIR / "v6_3_2_svm_mlp_detailed_folds.csv"
V633_LOG_CSV = RESULTS_DIR / "v6_3_3_all10ds_detailed_folds.csv"
V633_SVM_MLP_CSV = RESULTS_DIR / "v6_3_3_svm_mlp_detailed_folds.csv"

DATASETS = [
    "emotions",
    "scene",
    "yeast",
    "plantpseaac",
    "humanpseaac",
    "chd49",
    "music",
    "gpositivepseaac",
    "genbase",
    "viruspseaac",
]

DATASET_LABELS = {
    "emotions": "Emotions",
    "scene": "Scene",
    "yeast": "Yeast",
    "plantpseaac": "PlantPseAAC",
    "humanpseaac": "HumanPseAAC",
    "chd49": "CHD49",
    "music": "Music",
    "gpositivepseaac": "Gpositive",
    "genbase": "Genbase",
    "viruspseaac": "VirusPseAAC",
}

MODELS_6 = ["BR", "CC", "MLC_PA", "GSI_v6_2", "GSI_v6_3_2", "GSI_v6_3_3"]
MODEL_NAMES = {
    "BR": "BR",
    "CC": "CC",
    "MLC_PA": "MLC-PA",
    "GSI_v6_2": "GSI v6.2.1",
    "GSI_v6_3_2": "GSI v6.3.2",
    "GSI_v6_3_3": "GSI v6.3.3 (Đề Xuất)",
}

MODEL_COLORS = {
    "BR": "#7F8C8D",         # Gray
    "CC": "#2980B9",         # Blue
    "MLC_PA": "#E67E22",     # Orange
    "GSI_v6_2": "#8E44AD",   # Purple
    "GSI_v6_3_2": "#16A085", # Teal
    "GSI_v6_3_3": "#C0392B", # Crimson Red
}

def load_data():
    df_decay = pd.read_csv(DECAY_CSV)
    df_632_log = pd.read_csv(V632_LOG_CSV)
    df_633_log = pd.read_csv(V633_LOG_CSV)

    records = []
    # 1. BR, CC, MLC_PA, GSI_v6_2 from decay study
    for _, r in df_decay.iterrows():
        m = r["model"]
        if m == "GSI_v6_2_Decay":
            m = "GSI_v6_2"
        if m in ["BR", "CC", "MLC_PA", "GSI_v6_2"]:
            records.append({
                "dataset": r["dataset"],
                "learner": r["learner"],
                "model": m,
                "fold": r["fold"],
                "sel_f1": r.get("Selective_Macro_F1", r.get("Full_Macro_F1")),
                "coverage": r.get("Coverage", 1.0),
                "subset_acc": r.get("Subset_Accuracy", 0.0),
                "hamming_loss": r.get("Hamming_Loss", 0.0),
                "sel_micro_f1": r.get("Selective_Micro_F1", r.get("Full_Micro_F1", 0.0)),
            })

    # 2. v6.3.2 Logistic
    for _, r in df_632_log.iterrows():
        if r["model"] == "GSI_v6_3_2":
            records.append({
                "dataset": r["dataset"],
                "learner": "Logistic",
                "model": "GSI_v6_3_2",
                "fold": r["fold"],
                "sel_f1": r["Selective_Macro_F1"],
                "coverage": r["Coverage"],
                "subset_acc": r["Subset_Accuracy"],
                "hamming_loss": r["Hamming_Loss"],
                "sel_micro_f1": r["Selective_Micro_F1"],
            })

    # 3. v6.3.2 SVM & MLP
    if V632_SVM_MLP_CSV.exists():
        df_632_svm_mlp = pd.read_csv(V632_SVM_MLP_CSV)
        for _, r in df_632_svm_mlp.iterrows():
            if r["model"] == "GSI_v6_3_2":
                records.append({
                    "dataset": r["dataset"],
                    "learner": r["learner"],
                    "model": "GSI_v6_3_2",
                    "fold": r["fold"],
                    "sel_f1": r["Selective_Macro_F1"],
                    "coverage": r["Coverage"],
                    "subset_acc": r["Subset_Accuracy"],
                    "hamming_loss": r["Hamming_Loss"],
                    "sel_micro_f1": r["Selective_Micro_F1"],
                })

    # 4. v6.3.3 Logistic
    for _, r in df_633_log.iterrows():
        if r["Model"] == "GSI_v6_3_3":
            records.append({
                "dataset": r["Dataset"],
                "learner": "Logistic",
                "model": "GSI_v6_3_3",
                "fold": r["Fold"],
                "sel_f1": r["Selective_Macro_F1"],
                "coverage": r["Coverage"],
                "subset_acc": r["Subset_Accuracy"],
                "hamming_loss": r["Hamming_Loss"],
                "sel_micro_f1": r["Selective_Micro_F1"],
            })

    # 5. v6.3.3 SVM & MLP
    if V633_SVM_MLP_CSV.exists():
        df_svm_mlp = pd.read_csv(V633_SVM_MLP_CSV)
        for _, r in df_svm_mlp.iterrows():
            if r["model"] == "GSI_v6_3_3":
                records.append({
                    "dataset": r["dataset"],
                    "learner": r["learner"],
                    "model": "GSI_v6_3_3",
                    "fold": r["fold"],
                    "sel_f1": r["Selective_Macro_F1"],
                    "coverage": r["Coverage"],
                    "subset_acc": r["Subset_Accuracy"],
                    "hamming_loss": r["Hamming_Loss"],
                    "sel_micro_f1": r["Selective_Micro_F1"],
                })

    return pd.DataFrame(records)

def plot_fig1_macro_f1(df):
    """Figure 1: Selective Macro-F1 across 10 datasets (Primary: Logistic Learner)."""
    df_log = df[df["learner"] == "Logistic"]
    agg = df_log.groupby(["dataset", "model"])["sel_f1"].mean().unstack()[MODELS_6].reindex(DATASETS)

    x = np.arange(len(DATASETS))
    width = 0.13
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)

    for i, m in enumerate(MODELS_6):
        offset = (i - 2.5) * width
        bars = ax.bar(x + offset, agg[m], width, label=MODEL_NAMES[m], color=MODEL_COLORS[m], alpha=0.9, edgecolor="black", linewidth=0.5)
        if m == "GSI_v6_3_3":
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.012, f"{h:.2f}", ha='center', va='bottom', fontsize=7, fontweight='bold', color=MODEL_COLORS[m])

    ax.set_ylabel("Selective Macro-F1", fontsize=12, fontweight="bold")
    ax.set_title("Đối Sánh Selective Macro-F1 Trên 10 Tập Dữ Liệu Benchmark (Logistic Regression Base Learner)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=10, fontweight="bold", rotation=20)
    ax.set_ylim(0, 0.90)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9.5)

    plt.tight_layout()
    fig.savefig(FIGS_DIR / "fig1_macro_f1_comparison.png", dpi=300)
    fig.savefig(FIGS_DIR / "fig1_macro_f1_comparison.pdf")
    plt.close(fig)
    print("Saved Figure 1: Macro-F1 comparison")

def plot_fig2_pareto(df):
    """Figure 2: Pareto Trade-off Curve (Coverage vs Selective Macro-F1)."""
    df_log = df[df["learner"] == "Logistic"]
    agg = df_log.groupby(["model"]).agg(
        f1_mean=("sel_f1", "mean"),
        cov_mean=("coverage", "mean"),
    ).loc[MODELS_6]

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=300)

    for m in MODELS_6:
        x_val = agg.loc[m, "cov_mean"] * 100
        y_val = agg.loc[m, "f1_mean"]
        ax.scatter(x_val, y_val, s=220, color=MODEL_COLORS[m], edgecolors="black", linewidth=1.5, zorder=5, label=MODEL_NAMES[m])
        offset_y = 10 if m in ["GSI_v6_3_3", "CC", "GSI_v6_2"] else -25
        ax.annotate(f"{MODEL_NAMES[m]}\n(Cov={x_val:.1f}%, F1={y_val:.4f})", (x_val, y_val),
                    textcoords="offset points", xytext=(0, offset_y),
                    ha="center", fontsize=8.5, fontweight="bold", color=MODEL_COLORS[m])

    # Connect Pareto frontier
    ax.plot([agg.loc["GSI_v6_3_3", "cov_mean"]*100, agg.loc["CC", "cov_mean"]*100],
            [agg.loc["GSI_v6_3_3", "f1_mean"], agg.loc["CC", "f1_mean"]],
            linestyle="--", color="gray", alpha=0.6, label="Biên Pareto Hiệu Quả")

    ax.set_xlabel("Độ Bao Phủ Quyết Định - Coverage (%)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Selective Macro-F1", fontsize=11, fontweight="bold")
    ax.set_title("Đồ Thị Đánh Đổi Pareto Giữa Độ Phủ Quyết Định và Selective Macro-F1", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(65, 105)
    ax.set_ylim(0.35, 0.62)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", framealpha=0.9, fontsize=9)

    plt.tight_layout()
    fig.savefig(FIGS_DIR / "fig2_f1_vs_coverage_pareto.png", dpi=300)
    fig.savefig(FIGS_DIR / "fig2_f1_vs_coverage_pareto.pdf")
    plt.close(fig)
    print("Saved Figure 2: Pareto Trade-off")

def plot_fig3_subset_accuracy(df):
    """Figure 3: Subset 0/1 Accuracy across 10 datasets."""
    df_log = df[df["learner"] == "Logistic"]
    agg = df_log.groupby(["dataset", "model"])["subset_acc"].mean().unstack()[MODELS_6].reindex(DATASETS)

    x = np.arange(len(DATASETS))
    width = 0.13
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)

    for i, m in enumerate(MODELS_6):
        offset = (i - 2.5) * width
        bars = ax.bar(x + offset, agg[m], width, label=MODEL_NAMES[m], color=MODEL_COLORS[m], alpha=0.9, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Subset 0/1 Accuracy (Exact Match)", fontsize=12, fontweight="bold")
    ax.set_title("Đối Sánh Độ Chính Xác Toàn Khối (Subset 0/1 Accuracy) Trên 10 Tập Dữ Liệu", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=10, fontweight="bold", rotation=20)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9.5)

    plt.tight_layout()
    fig.savefig(FIGS_DIR / "fig3_subset_accuracy_comparison.png", dpi=300)
    fig.savefig(FIGS_DIR / "fig3_subset_accuracy_comparison.pdf")
    plt.close(fig)
    print("Saved Figure 3: Subset Accuracy comparison")

def plot_fig4_hamming_loss(df):
    """Figure 4: Hamming Loss across 10 datasets (Lower is better)."""
    df_log = df[df["learner"] == "Logistic"]
    agg = df_log.groupby(["dataset", "model"])["hamming_loss"].mean().unstack()[MODELS_6].reindex(DATASETS)

    x = np.arange(len(DATASETS))
    width = 0.13
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)

    for i, m in enumerate(MODELS_6):
        offset = (i - 2.5) * width
        bars = ax.bar(x + offset, agg[m], width, label=MODEL_NAMES[m], color=MODEL_COLORS[m], alpha=0.9, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Hamming Loss (↓ Càng thấp càng tốt)", fontsize=12, fontweight="bold")
    ax.set_title("Đối Sánh Hamming Loss Trên 10 Tập Dữ Liệu (Sai Số Bit Từng Nhãn)", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS], fontsize=10, fontweight="bold", rotation=20)
    ax.set_ylim(0, 0.40)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9.5)

    plt.tight_layout()
    fig.savefig(FIGS_DIR / "fig4_hamming_loss_comparison.png", dpi=300)
    fig.savefig(FIGS_DIR / "fig4_hamming_loss_comparison.pdf")
    plt.close(fig)
    print("Saved Figure 4: Hamming Loss comparison")

def plot_fig5_focus_chd49_virus(df):
    """Figure 5: Special In-Depth Focus on CHD49 and VirusPseAAC (Highlighting v6.3.2 vs v6.3.3 recovery)."""
    df_log = df[df["learner"] == "Logistic"]
    sub = df_log[df_log["dataset"].isin(["chd49", "viruspseaac"])]
    agg = sub.groupby(["dataset", "model"]).agg(
        f1=("sel_f1", "mean"),
        cov=("coverage", "mean"),
        subacc=("subset_acc", "mean")
    ).unstack(level=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    for idx, ds in enumerate(["chd49", "viruspseaac"]):
        ax = axes[idx]
        ds_name = DATASET_LABELS[ds]
        x = np.arange(len(MODELS_6))
        f1_vals = [agg.loc[ds, ("f1", m)] for m in MODELS_6]
        sub_vals = [agg.loc[ds, ("subacc", m)] for m in MODELS_6]

        width = 0.35
        b1 = ax.bar(x - width/2, f1_vals, width, label="Selective Macro-F1", color=[MODEL_COLORS[m] for m in MODELS_6], alpha=0.9, edgecolor="black")
        b2 = ax.bar(x + width/2, sub_vals, width, label="Subset Accuracy", color=[MODEL_COLORS[m] for m in MODELS_6], hatch="//", alpha=0.6, edgecolor="black")

        for bar in b1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.01, f"{h:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

        ax.set_title(f"Tập Dữ Liệu: {ds_name.upper()}\n({ 'Mẫu nhỏ N=207, d=440' if ds == 'viruspseaac' else 'Dày đặc, nhãn dương chiếm đa số (L5: 76%, L0: 61%)' })", fontsize=10.5, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_NAMES[m] for m in MODELS_6], fontsize=8, fontweight="bold", rotation=25)
        ax.set_ylim(0, 0.65)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        # Custom legend patches
        p1 = mpatches.Patch(facecolor="gray", label="Selective Macro-F1", edgecolor="black")
        p2 = mpatches.Patch(facecolor="gray", hatch="//", alpha=0.6, label="Subset Accuracy", edgecolor="black")
        ax.legend(handles=[p1, p2], loc="upper right", fontsize=8.5)

    plt.suptitle("Đánh Giá Chuyên Sâu Hiện Tượng Phục Hồi Hiệu Năng: v6.3.3 Đảo Ngược Sự Tụt Dốc Của v6.3.2", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGS_DIR / "fig5_focus_chd49_viruspseaac.png", dpi=300)
    fig.savefig(FIGS_DIR / "fig5_focus_chd49_viruspseaac.pdf")
    plt.close(fig)
    print("Saved Figure 5: Focus CHD49 and VirusPseAAC")

def plot_fig6_radar(df):
    """Figure 6: Radar Chart Comparing 6 Models on 5 Key Global Objectives."""
    df_log = df[df["learner"] == "Logistic"]
    agg = df_log.groupby("model").agg(
        f1=("sel_f1", "mean"),
        cov=("coverage", "mean"),
        subacc=("subset_acc", "mean"),
        hl_inv=("hamming_loss", lambda x: 1.0 - np.mean(x)),
        micro=("sel_micro_f1", "mean"),
    ).loc[MODELS_6]

    categories = [
        "Selective Macro-F1",
        "Coverage (%)",
        "Subset 0/1 Acc",
        "1 - Hamming Loss",
        "Selective Micro-F1"
    ]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True), dpi=300)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    plt.xticks(angles[:-1], categories, fontsize=10, fontweight="bold")
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
    plt.ylim(0, 1.05)

    for m in MODELS_6:
        values = [
            agg.loc[m, "f1"],
            agg.loc[m, "cov"],
            agg.loc[m, "subacc"],
            agg.loc[m, "hl_inv"],
            agg.loc[m, "micro"],
        ]
        values += values[:1]
        ax.plot(angles, values, linewidth=2.2 if m == 'GSI_v6_3_3' else 1.5,
                linestyle='solid' if m == 'GSI_v6_3_3' else ('dashdot' if m == 'GSI_v6_3_2' else '--'),
                label=MODEL_NAMES[m], color=MODEL_COLORS[m])
        ax.fill(angles, values, color=MODEL_COLORS[m], alpha=0.15 if m == 'GSI_v6_3_3' else 0.04)

    plt.title("Biểu Đồ Radar Đối So Sánh Đa Mục Tiêu Giữa 6 Hệ Thống Mô Hình", size=12, fontweight="bold", y=1.08)
    plt.legend(loc="upper right", bbox_to_anchor=(0.1, 0.1), fontsize=8.5, framealpha=0.9)

    plt.tight_layout()
    fig.savefig(FIGS_DIR / "fig6_radar_5criteria.png", dpi=300)
    fig.savefig(FIGS_DIR / "fig6_radar_5criteria.pdf")
    plt.close(fig)
    print("Saved Figure 6: Radar Chart")

def main():
    print("Loading data for metric plots...")
    df = load_data()
    print(f"Loaded {len(df)} records. Models: {df['model'].unique()}, Learners: {df['learner'].unique()}")

    plot_fig1_macro_f1(df)
    plot_fig2_pareto(df)
    plot_fig3_subset_accuracy(df)
    plot_fig4_hamming_loss(df)
    plot_fig5_focus_chd49_virus(df)
    plot_fig6_radar(df)
    print("\nALL 6 PUBLICATION FIGURES (WITH GSI v6.3.2) SUCCESSFULLY GENERATED IN:", FIGS_DIR)

if __name__ == "__main__":
    main()
