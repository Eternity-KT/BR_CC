import sys
from pathlib import Path
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MaxAbsScaler
from src.data.loader import load_dataset
from src.evaluation.cv import get_multilabel_cv
from src.evaluation.metrics import (
    compute_all_metrics,
    compute_selective_macro_f1,
)

# Custom Tail Calibrator test variants
class VariantCalibrator:
    def __init__(self, mode="unweighted"):
        self.mode = mode # 'unweighted', 'sqrt_weighted', 'standard_weighted'
        self.model_ = None

    def fit(self, probs, y):
        p = np.clip(np.asarray(probs, dtype=np.float64).ravel(), 1e-5, 1.0 - 1e-5)
        labels = np.asarray(y, dtype=np.int32).ravel()
        if len(np.unique(labels)) <= 1:
            return self
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        
        n_pos = np.sum(labels == 1)
        n_neg = np.sum(labels == 0)
        pos_weight = float(n_neg) / float(max(n_pos, 1))

        if self.mode == "unweighted":
            sw = None
        elif self.mode == "sqrt_weighted":
            sw = np.where(labels == 1, np.sqrt(pos_weight), 1.0)
        elif self.mode == "standard_weighted":
            sw = np.where(labels == 1, pos_weight, 1.0)
        
        lr = LogisticRegression(C=1.0, solver="liblinear", max_iter=1000, random_state=42)
        lr.fit(logits, labels, sample_weight=sw)
        if lr.coef_[0, 0] > 0:
            self.model_ = lr
        return self

    def predict_proba(self, probs):
        p = np.clip(np.asarray(probs, dtype=np.float64).ravel(), 1e-5, 1.0 - 1e-5)
        if self.model_ is None:
            return p
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        return self.model_.predict_proba(logits)[:, 1].astype(np.float32)

def custom_predict(probs, priors, cost=0.30, gamma_min=0.70, precision_guard=True):
    N, K = probs.shape
    preds = np.full((N, K), -1, dtype=np.int32)
    
    for j in range(K):
        pj = float(priors[j])
        p_col = probs[:, j]
        
        # Bayes Likelihood Ratio base thresholds
        # Odds formulation
        num1 = (1.0 - cost) * pj
        den1 = cost * (1.0 - pj) + num1
        tau1_base = num1 / max(den1, 1e-12)

        num0 = cost * pj
        den0 = (1.0 - cost) * (1.0 - pj) + num0
        tau0_base = num0 / max(den0, 1e-12)

        if precision_guard:
            # tau1 must not fall below 0.50 (prevent false positive explosion)
            tau1 = max(0.50, min(0.95, tau1_base))
            tau0 = min(0.45, max(0.01, tau0_base))
        else:
            tau1 = max(0.51, min(0.95, tau1_base))
            tau0 = min(0.49, max(0.01, tau0_base))

        # Coverage guard: if coverage < gamma_min, expand decision region safely
        decided = (p_col <= tau0) | (p_col >= tau1)
        cov = np.mean(decided)
        if cov < gamma_min:
            # Safely expand tau0 towards median without lowering tau1 below 0.50
            for _ in range(50):
                tau0 += 0.01
                if precision_guard:
                    if tau1 > 0.55:
                        tau1 -= 0.005
                else:
                    tau1 -= 0.01
                if tau0 >= tau1:
                    break
                cov = np.mean((p_col <= tau0) | (p_col >= tau1))
                if cov >= gamma_min:
                    break

        preds[p_col >= tau1, j] = 1
        preds[p_col <= tau0, j] = 0

    return preds

def evaluate_variant(dataset_name, mode, precision_guard):
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
    K = Y.shape[1]
    priors = [np.mean(Y_tr[:, j]) for j in range(K)]

    # Fit base models and predict raw probas
    from src.models.gsi_v6_2 import GSIMLCPAv6_2Classifier
    m = GSIMLCPAv6_2Classifier(base_learner='logistic', decaying_threshold=True, random_state=42)
    m.fit(X_tr, Y_tr)
    raw_probs = m.predict_proba(X_te)

    # Calibrate with variant
    cal_probs = np.zeros_like(raw_probs)
    for j in range(K):
        # We simulate calibration
        cal = VariantCalibrator(mode=mode)
        # In reality OOF is used; here train probas for fast proof-of-concept
        tr_p = m.predict_proba(X_tr)[:, j]
        cal.fit(tr_p, Y_tr[:, j])
        cal_probs[:, j] = cal.predict_proba(raw_probs[:, j])

    sel_preds = custom_predict(cal_probs, priors, cost=0.30, gamma_min=0.70, precision_guard=precision_guard)
    full_preds = (cal_probs >= 0.5).astype(np.int32)

    full_m = compute_all_metrics(Y_te, full_preds)
    sel_f1 = compute_selective_macro_f1(Y_te, sel_preds)
    cov = np.mean(sel_preds != -1)
    return sel_f1, cov, full_m['Macro-F1'], full_m['Subset Accuracy'], full_m['Hamming Loss']

print('Running tests on 4 datasets...')
for ds in ['emotions', 'scene', 'plantpseaac', 'humanpseaac']:
    print(f'\n--- {ds} ---')
    # Original v6.3: standard_weighted, no precision guard
    f1_orig, cov_orig, ff1_orig, sa_orig, hl_orig = evaluate_variant(ds, 'standard_weighted', False)
    print(f'v6.3 (Orig): Sel-F1={f1_orig:.4f} | Cov={cov_orig*100:.1f}% | Full-F1={ff1_orig:.4f} | SA={sa_orig:.4f} | HL={hl_orig:.4f}')

    # Variant 1: unweighted Platt + precision guard
    f1_v1, cov_v1, ff1_v1, sa_v1, hl_v1 = evaluate_variant(ds, 'unweighted', True)
    print(f'v6.3.1 (Unweighted+Guard): Sel-F1={f1_v1:.4f} | Cov={cov_v1*100:.1f}% | Full-F1={ff1_v1:.4f} | SA={sa_v1:.4f} | HL={hl_v1:.4f}')

    # Variant 2: sqrt_weighted + precision guard
    f1_v2, cov_v2, ff1_v2, sa_v2, hl_v2 = evaluate_variant(ds, 'sqrt_weighted', True)
    print(f'v6.3.1 (SqrtWeighted+Guard): Sel-F1={f1_v2:.4f} | Cov={cov_v2*100:.1f}% | Full-F1={ff1_v2:.4f} | SA={sa_v2:.4f} | HL={hl_v2:.4f}')
