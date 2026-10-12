import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src.data.loader import load_dataset
from src.models.gsi_v6_3_3 import GSIMLCPAv6_3_3Classifier
from src.evaluation.cv import get_multilabel_cv
from src.evaluation.metrics import compute_selective_macro_f1
from sklearn.preprocessing import MaxAbsScaler

cv = get_multilabel_cv(n_splits=5, random_state=42)

for ds_name in ['viruspseaac', 'chd49', 'emotions', 'yeast']:
    X, Y, _, _ = load_dataset(ds_name)
    print(f"\n==================== DATASET: {ds_name} ====================")
    for learner in ['logistic', 'svm_calibrated', 'mlp']:
        # Compare:
        # Config 1: Baseline v6.3.3 (cost=0.30, gamma_min=0.70, tau_balance=0.75)
        # Config 2: High-Coverage (tau_balance=0.85, tau_1_cap=0.55, tau_0_floor=0.45, gamma_min=0.75)
        # Config 3: Max-Coverage (tau_balance=0.85, cost=0.40, gamma_min=0.80)
        
        f1_c1, cov_c1 = [], []
        f1_c2, cov_c2 = [], []
        f1_c3, cov_c3 = [], []
        
        for fold, (tr, te) in enumerate(cv.split(X, Y)):
            X_tr, X_te = X[tr], X[te]
            Y_tr, Y_te = Y[tr], Y[te]
            scaler = MaxAbsScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            
            # C1: v6.3.3 baseline
            m1 = GSIMLCPAv6_3_3Classifier(base_learner=learner, tau_balance=0.75, cost=0.30, gamma_min=0.70, random_state=42+fold).fit(X_tr, Y_tr)
            pred1 = m1.predict(X_te)
            f1_c1.append(compute_selective_macro_f1(Y_te, pred1))
            cov_c1.append(np.mean(pred1 != -1))
            
            # C2: Hybrid Clamped (tau_balance=0.85, gamma_min=0.75, band tightened)
            m2 = GSIMLCPAv6_3_3Classifier(base_learner=learner, tau_balance=0.85, cost=0.30, gamma_min=0.75, random_state=42+fold).fit(X_tr, Y_tr)
            P2 = m2.predict_proba(X_te)
            pred2 = np.full_like(P2, -1, dtype=np.int32)
            for j in range(Y.shape[1]):
                t0 = m2.adaptive_thresholds_[j]['tau_0']
                t1 = m2.adaptive_thresholds_[j]['tau_1']
                prior_j = m2.adaptive_thresholds_[j]['prior']
                if prior_j <= 0.50:
                    t1 = min(t1, 0.55)
                    t0 = max(t0, 0.40)
                    if t1 - t0 < 0.02: t0 = max(0.005, t1 - 0.02)
                else:
                    t0 = max(t0, 0.45)
                    t1 = min(t1, 0.60)
                    if t1 - t0 < 0.02: t1 = min(0.995, t0 + 0.02)
                pred2[P2[:, j] <= t0, j] = 0
                pred2[P2[:, j] >= t1, j] = 1
            f1_c2.append(compute_selective_macro_f1(Y_te, pred2))
            cov_c2.append(np.mean(pred2 != -1))
            
            # C3: Max-Coverage with gamma_min=0.80 and cost=0.40
            m3 = GSIMLCPAv6_3_3Classifier(base_learner=learner, tau_balance=0.85, cost=0.40, gamma_min=0.80, random_state=42+fold).fit(X_tr, Y_tr)
            P3 = m3.predict_proba(X_te)
            pred3 = np.full_like(P3, -1, dtype=np.int32)
            for j in range(Y.shape[1]):
                t0 = m3.adaptive_thresholds_[j]['tau_0']
                t1 = m3.adaptive_thresholds_[j]['tau_1']
                prior_j = m3.adaptive_thresholds_[j]['prior']
                if prior_j <= 0.50:
                    t1 = min(t1, 0.55)
                    if t1 - t0 < 0.02: t0 = max(0.005, t1 - 0.02)
                else:
                    t0 = max(t0, 0.45)
                    if t1 - t0 < 0.02: t1 = min(0.995, t0 + 0.02)
                pred3[P3[:, j] <= t0, j] = 0
                pred3[P3[:, j] >= t1, j] = 1
            f1_c3.append(compute_selective_macro_f1(Y_te, pred3))
            cov_c3.append(np.mean(pred3 != -1))
            
        print(f"[{learner:14s}]")
        print(f"  v6.3.3 gốc:         Cov = {np.mean(cov_c1):.1%} | F1 = {np.mean(f1_c1):.4f}")
        print(f"  C2 (Kẹp dải [40,55]):Cov = {np.mean(cov_c2):.1%} | F1 = {np.mean(f1_c2):.4f}")
        print(f"  C3 (Max Cov gamma=80):Cov = {np.mean(cov_c3):.1%} | F1 = {np.mean(f1_c3):.4f}")
