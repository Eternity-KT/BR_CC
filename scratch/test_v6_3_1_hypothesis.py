import sys
from pathlib import Path
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import pandas as pd
from src.data.loader import load_dataset
from src.models.gsi_v6_2 import GSIMLCPAv6_2Classifier
from src.models.gsi_v6_3 import GSIMLCPAv6_3Classifier
from src.evaluation.cv import get_multilabel_cv
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_selective_macro_f1,
    compute_selective_micro_f1,
)
from sklearn.preprocessing import MaxAbsScaler

def test_on_dataset(dataset_name='plantpseaac'):
    X, Y, _, _ = load_dataset(dataset_name)
    if hasattr(X, 'toarray'): X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.int32)

    cv = get_multilabel_cv(n_splits=5, random_state=42)
    train_idx, test_idx = next(cv.split(X, Y))
    scaler = MaxAbsScaler()
    X_tr = scaler.fit_transform(X[train_idx])
    X_te = scaler.transform(X[test_idx])
    Y_tr, Y_te = Y[train_idx], Y[test_idx]

    # Baseline v6.2
    m62 = GSIMLCPAv6_2Classifier(base_learner='logistic', decaying_threshold=True, random_state=42)
    m62.fit(X_tr, Y_tr)
    p62_sel = m62.predict(X_te, cost=0.30)
    p62_full = m62.predict_full(X_te)
    m62_full = compute_all_metrics(Y_te, p62_full)
    m62_sel_f1 = compute_selective_macro_f1(Y_te, p62_sel)
    m62_cov = np.mean(p62_sel != -1)

    # v6.3 current
    m63 = GSIMLCPAv6_3Classifier(base_learner='logistic', decaying_threshold=True, use_prior_adaptive=True, calibrate_tail=True, random_state=42)
    m63.fit(X_tr, Y_tr)
    p63_sel = m63.predict(X_te, cost=0.30)
    p63_full = m63.predict_full(X_te)
    m63_full = compute_all_metrics(Y_te, p63_full)
    m63_sel_f1 = compute_selective_macro_f1(Y_te, p63_sel)
    m63_cov = np.mean(p63_sel != -1)

    print(f"\n=================== DATASET: {dataset_name} ===================")
    print(f"v6.2.1: Sel-F1={m62_sel_f1:.4f} | Cov={m62_cov*100:.1f}% | Full-F1={m62_full['Macro-F1']:.4f} | SubsetAcc={m62_full['Subset Accuracy']:.4f} | HL={m62_full['Hamming Loss']:.4f}")
    print(f"v6.3.0: Sel-F1={m63_sel_f1:.4f} | Cov={m63_cov*100:.1f}% | Full-F1={m63_full['Macro-F1']:.4f} | SubsetAcc={m63_full['Subset Accuracy']:.4f} | HL={m63_full['Hamming Loss']:.4f}")

if __name__ == '__main__':
    for ds in ['emotions', 'scene', 'plantpseaac', 'humanpseaac']:
        test_on_dataset(ds)
