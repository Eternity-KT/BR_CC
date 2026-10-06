import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from ast import literal_eval

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data.loader import load_dataset
from src.evaluation.cv import get_multilabel_cv

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

def main():
    decay_breakdown_path = WORKSPACE_ROOT / "results_v6_2" / "decay_study" / "decay_peeling_breakdown.csv"
    df_breakdown = pd.read_csv(decay_breakdown_path)

    log_rows = df_breakdown[df_breakdown["learner"] == "Logistic"]

    records = []

    for _, row in log_rows.iterrows():
        ds_name = row["dataset"]
        decay_dl = set(literal_eval(row["decay_dl"]))
        decay_layers = literal_eval(row["decay_layers"])
        il_set = set()
        for lyr in decay_layers:
            il_set.update(lyr)

        X, Y, f_names, l_names = load_dataset(ds_name)
        N, K = Y.shape
        if l_names is None or len(l_names) != K:
            l_names = [f"L_{j}" for j in range(K)]

        # 5-fold CV to get BR OOF F1 per label
        cv = get_multilabel_cv(n_splits=5, random_state=42)
        oof_preds = np.zeros_like(Y)

        for train_idx, val_idx in cv.split(X, Y):
            X_tr, X_val = X[train_idx], X[val_idx]
            Y_tr, Y_val = Y[train_idx], Y[val_idx]

            for j in range(K):
                y_tr_j = Y_tr[:, j]
                if len(np.unique(y_tr_j)) <= 1:
                    oof_preds[val_idx, j] = int(y_tr_j[0]) if len(y_tr_j) > 0 else 0
                else:
                    clf = LogisticRegression(max_iter=1000, random_state=42)
                    clf.fit(X_tr, y_tr_j)
                    oof_preds[val_idx, j] = clf.predict(X_val)

        # Compute per-label BR F1
        for j in range(K):
            pos_c = int(np.sum(Y[:, j] == 1))
            neg_c = N - pos_c
            freq = pos_c / N
            ir = float(max(pos_c, neg_c) / max(1, min(pos_c, neg_c)))
            f1 = float(f1_score(Y[:, j], oof_preds[:, j], zero_division=0))
            status = "DL" if j in decay_dl else "IL"

            records.append({
                "dataset": ds_name,
                "label_idx": j,
                "label_name": l_names[j],
                "status": status,
                "pos_count": pos_c,
                "neg_count": neg_c,
                "pos_freq_pct": freq * 100,
                "ir": ir,
                "br_f1": f1
            })

    df_out = pd.DataFrame(records)
    out_path = WORKSPACE_ROOT / "results_v6_2" / "decay_study" / "br_f1_vs_imbalance_v6_2_1.csv"
    df_out.to_csv(out_path, index=False)
    print("Exported BR F1 vs Imbalance successfully.")

if __name__ == "__main__":
    main()
