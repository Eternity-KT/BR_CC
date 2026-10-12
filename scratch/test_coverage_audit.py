import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from src.data.loader import load_dataset
from src.models.gsi_v6_3_2 import GSIMLCPAv6_3_2Classifier
from src.models.gsi_v6_3_3 import GSIMLCPAv6_3_3Classifier
from src.evaluation.cv import get_multilabel_cv
from src.evaluation.metrics import compute_selective_macro_f1
from sklearn.preprocessing import MaxAbsScaler

# Test proposed modifications:
# 1. tau_balance = 0.85
# 2. tau_balance = 0.90
# 3. Hybrid: tau_balance = 0.85 + tau_1 cap <= 0.55 on pi <= 0.50

DATASETS = ["chd49", "viruspseaac", "emotions", "music", "yeast", "gpositivepseaac"]
cv = get_multilabel_cv(n_splits=5, random_state=42)

results = []

for ds_name in DATASETS:
    X, Y, _, _ = load_dataset(ds_name)
    for learner in ["logistic", "svm_calibrated", "mlp"]:
        f1_v632, cov_v632 = [], []
        f1_v633, cov_v633 = [], []
        f1_optA, cov_optA = [], []
        f1_opt_hyb, cov_opt_hyb = [], []
        
        for fold, (tr, te) in enumerate(cv.split(X, Y)):
            X_tr, X_te = X[tr], X[te]
            Y_tr, Y_te = Y[tr], Y[te]
            scaler = MaxAbsScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            
            # v6.3.2
            m2 = GSIMLCPAv6_3_2Classifier(base_learner=learner, random_state=42 + fold).fit(X_tr, Y_tr)
            pred2 = m2.predict(X_te)
            f1_v632.append(compute_selective_macro_f1(Y_te, pred2))
            cov_v632.append(np.mean(pred2 != -1))
            
            # v6.3.3 default (tau_balance = 0.75)
            m3 = GSIMLCPAv6_3_3Classifier(base_learner=learner, tau_balance=0.75, random_state=42 + fold).fit(X_tr, Y_tr)
            pred3 = m3.predict(X_te)
            f1_v633.append(compute_selective_macro_f1(Y_te, pred3))
            cov_v633.append(np.mean(pred3 != -1))
            
            # Option A: tau_balance = 0.85
            mA = GSIMLCPAv6_3_3Classifier(base_learner=learner, tau_balance=0.85, random_state=42 + fold).fit(X_tr, Y_tr)
            predA = mA.predict(X_te)
            f1_optA.append(compute_selective_macro_f1(Y_te, predA))
            cov_optA.append(np.mean(predA != -1))
            
            # Option Hybrid: tau_balance = 0.85 + cap tau_1 <= 0.55 on pi <= 0.50
            P_hyb = mA.predict_proba(X_te)
            pred_hyb = np.full_like(P_hyb, -1, dtype=np.int32)
            for j in range(Y.shape[1]):
                t0 = mA.adaptive_thresholds_[j]['tau_0']
                t1 = mA.adaptive_thresholds_[j]['tau_1']
                prior_j = mA.adaptive_thresholds_[j]['prior']
                if prior_j <= 0.50:
                    t1 = min(t1, 0.55)
                    if t1 - t0 < 0.02:
                        t0 = max(0.005, t1 - 0.02)
                else:
                    t0 = max(t0, 0.45)
                    if t1 - t0 < 0.02:
                        t1 = min(0.995, t0 + 0.02)
                pred_hyb[P_hyb[:, j] <= t0, j] = 0
                pred_hyb[P_hyb[:, j] >= t1, j] = 1
            f1_opt_hyb.append(compute_selective_macro_f1(Y_te, pred_hyb))
            cov_opt_hyb.append(np.mean(pred_hyb != -1))
            
        results.append({
            "Dataset": ds_name,
            "Learner": learner,
            "v632_F1": np.mean(f1_v632), "v632_Cov": np.mean(cov_v632),
            "v633_F1": np.mean(f1_v633), "v633_Cov": np.mean(cov_v633),
            "OptA_F1": np.mean(f1_optA), "OptA_Cov": np.mean(cov_optA),
            "Hyb_F1": np.mean(f1_opt_hyb), "Hyb_Cov": np.mean(cov_opt_hyb),
        })
        print(f"[{ds_name:12s}|{learner:14s}] Cov: v632={np.mean(cov_v632):.1%}, v633={np.mean(cov_v633):.1%}, OptA={np.mean(cov_optA):.1%}, Hyb={np.mean(cov_opt_hyb):.1%} | F1 Hyb={np.mean(f1_opt_hyb):.4f}")

df_res = pd.DataFrame(results)
print("\n" + "="*80)
print("FINAL SUMMARY COMPARISON:")
print(df_res.to_string(index=False))
