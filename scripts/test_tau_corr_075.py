import sys
from pathlib import Path
import numpy as np

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.data.loader import load_dataset
from src.models.gsi_v6_2 import GSIMLCPAv6_2Classifier
from src.evaluation.cv import get_multilabel_cv
from src.evaluation.metrics import compute_selective_macro_f1

datasets = ["emotions", "scene", "chd49", "music", "yeast"]

for ds in datasets:
    X, Y, _, _ = load_dataset(ds)
    cv = get_multilabel_cv(n_splits=5, random_state=42)
    
    f1_025, f1_075 = [], []
    cov_025, cov_075 = [], []
    edges_025, edges_075 = [], []

    for train_idx, val_idx in cv.split(X, Y):
        X_tr, X_val = X[train_idx], X[val_idx]
        Y_tr, Y_val = Y[train_idx], Y[val_idx]
        
        # tau_corr = 0.25 (v6.2.1 default)
        m_025 = GSIMLCPAv6_2Classifier(
            base_learner="logistic",
            stratified_threshold=0.75,
            residual_corr_threshold=0.25,
            decaying_threshold=True,
            cost=0.30,
            random_state=42
        )
        m_025.fit(X_tr, Y_tr)
        p_025 = m_025.predict(X_val)
        f1_025.append(compute_selective_macro_f1(Y_val, p_025, abstain_value=-1))
        cov_025.append(float(np.mean(p_025 != -1)))
        n_edges_025 = sum(len(v) for v in m_025.dependency_graph_.values())
        edges_025.append(n_edges_025)

        # tau_corr = 0.75
        m_075 = GSIMLCPAv6_2Classifier(
            base_learner="logistic",
            stratified_threshold=0.75,
            residual_corr_threshold=0.75,
            decaying_threshold=True,
            cost=0.30,
            random_state=42
        )
        m_075.fit(X_tr, Y_tr)
        p_075 = m_075.predict(X_val)
        f1_075.append(compute_selective_macro_f1(Y_val, p_075, abstain_value=-1))
        cov_075.append(float(np.mean(p_075 != -1)))
        n_edges_075 = sum(len(v) for v in m_075.dependency_graph_.values())
        edges_075.append(n_edges_075)

    print(
        f"{ds:10s} | "
        f"tau=0.25: F1={np.mean(f1_025):.4f}, Cov={np.mean(cov_025)*100:.1f}%, Cạnh DL={np.mean(edges_025):.1f} | "
        f"tau=0.75: F1={np.mean(f1_075):.4f}, Cov={np.mean(cov_075)*100:.1f}%, Cạnh DL={np.mean(edges_075):.1f} | "
        f"Δ F1={np.mean(f1_075) - np.mean(f1_025):+.4f}"
    )
