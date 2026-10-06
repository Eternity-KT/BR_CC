import sys
from pathlib import Path
import numpy as np
import pandas as pd
from ast import literal_eval

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data.loader import load_dataset

DATASETS = [
    "emotions",
    "scene",
    "chd49",
    "music",
    "gpositivepseaac",
    "genbase",
    "humanpseaac",
    "plantpseaac",
    "viruspseaac",
    "yeast",
]

BASE_LEARNERS = ["Logistic", "SVM", "MLP"]

def main():
    decay_breakdown_path = WORKSPACE_ROOT / "results_v6_2" / "decay_study" / "decay_peeling_breakdown.csv"
    if not decay_breakdown_path.exists():
        print(f"File not found: {decay_breakdown_path}")
        return

    df_breakdown = pd.read_csv(decay_breakdown_path)

    # Dictionary to cache dataset matrices
    ds_cache = {}
    for ds_name in DATASETS:
        X, Y, f_names, l_names = load_dataset(ds_name)
        if l_names is None or len(l_names) != Y.shape[1]:
            l_names = [f"L_{j}" for j in range(Y.shape[1])]
        ds_cache[ds_name] = {
            "X": X,
            "Y": Y,
            "label_names": l_names,
            "N": Y.shape[0],
            "K": Y.shape[1]
        }

    results = []
    per_label_records = []

    for _, row in df_breakdown.iterrows():
        learner = row["learner"]
        ds_name = row["dataset"]
        decay_dl = literal_eval(row["decay_dl"]) if isinstance(row["decay_dl"], str) else row["decay_dl"]
        decay_layers = literal_eval(row["decay_layers"]) if isinstance(row["decay_layers"], str) else row["decay_layers"]
        
        il_labels = []
        for layer in decay_layers:
            il_labels.extend(layer)
        il_labels = sorted(list(set(il_labels)))
        dl_labels = sorted(list(set(decay_dl)))

        ds_info = ds_cache[ds_name]
        Y = ds_info["Y"]
        N = ds_info["N"]
        K = ds_info["K"]
        l_names = ds_info["label_names"]

        # Compute per-label stats
        pos_counts = np.sum(Y == 1, axis=0)
        neg_counts = N - pos_counts
        freqs = pos_counts / N
        irs = []
        for k in range(K):
            pos = pos_counts[k]
            neg = neg_counts[k]
            ir = float(max(pos, neg) / max(1, min(pos, neg)))
            irs.append(ir)

        # DL stats
        dl_pos = [pos_counts[k] for k in dl_labels] if dl_labels else []
        dl_freq = [freqs[k] for k in dl_labels] if dl_labels else []
        dl_ir = [irs[k] for k in dl_labels] if dl_labels else []

        # IL stats
        il_pos = [pos_counts[k] for k in il_labels] if il_labels else []
        il_freq = [freqs[k] for k in il_labels] if il_labels else []
        il_ir = [irs[k] for k in il_labels] if il_labels else []

        # All stats
        all_ir = irs

        results.append({
            "learner": learner,
            "dataset": ds_name,
            "N": N,
            "K": K,
            "n_il": len(il_labels),
            "n_dl": len(dl_labels),
            "pct_dl": len(dl_labels) / K * 100,
            # Imbalance of DL
            "dl_mean_ir": np.mean(dl_ir) if dl_ir else np.nan,
            "dl_median_ir": np.median(dl_ir) if dl_ir else np.nan,
            "dl_min_ir": np.min(dl_ir) if dl_ir else np.nan,
            "dl_max_ir": np.max(dl_ir) if dl_ir else np.nan,
            "dl_mean_freq_pct": np.mean(dl_freq) * 100 if dl_freq else np.nan,
            "dl_rare_lt_5pct": sum(f < 0.05 for f in dl_freq) if dl_freq else 0,
            "dl_rare_lt_10pct": sum(f < 0.10 for f in dl_freq) if dl_freq else 0,
            # Imbalance of IL
            "il_mean_ir": np.mean(il_ir) if il_ir else np.nan,
            "il_median_ir": np.median(il_ir) if il_ir else np.nan,
            "il_min_ir": np.min(il_ir) if il_ir else np.nan,
            "il_max_ir": np.max(il_ir) if il_ir else np.nan,
            "il_mean_freq_pct": np.mean(il_freq) * 100 if il_freq else np.nan,
            # Overall
            "all_mean_ir": np.mean(all_ir),
            "all_max_ir": np.max(all_ir),
        })

        # Record individual DL labels
        for k in dl_labels:
            per_label_records.append({
                "learner": learner,
                "dataset": ds_name,
                "label_idx": k,
                "label_name": l_names[k],
                "status": "DL",
                "pos_count": int(pos_counts[k]),
                "neg_count": int(neg_counts[k]),
                "pos_freq_pct": float(freqs[k] * 100),
                "ir": float(irs[k])
            })
        for k in il_labels:
            per_label_records.append({
                "learner": learner,
                "dataset": ds_name,
                "label_idx": k,
                "label_name": l_names[k],
                "status": "IL",
                "pos_count": int(pos_counts[k]),
                "neg_count": int(neg_counts[k]),
                "pos_freq_pct": float(freqs[k] * 100),
                "ir": float(irs[k])
            })

    df_res = pd.DataFrame(results)
    df_labels = pd.DataFrame(per_label_records)

    out_dir = WORKSPACE_ROOT / "results_v6_2" / "decay_study"
    df_res.to_csv(out_dir / "dl_imbalance_summary_v6_2_1.csv", index=False)
    df_labels.to_csv(out_dir / "dl_per_label_imbalance_v6_2_1.csv", index=False)
    print("Exported summary and per-label imbalance successfully.")

if __name__ == "__main__":
    main()
