import sys
from pathlib import Path
WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))

import numpy as np
import pandas as pd
from src.data.loader import load_dataset
from src.models.gsi_v6_3_3 import GSIMLCPAv6_3_3Classifier
from src.evaluation.cv import get_multilabel_cv
from src.evaluation.metrics import compute_selective_macro_f1
from sklearn.preprocessing import MaxAbsScaler

datasets = ['emotions', 'scene', 'yeast', 'plantpseaac', 'humanpseaac', 'chd49', 'music', 'gpositivepseaac', 'genbase', 'viruspseaac']
cv = get_multilabel_cv(n_splits=5, random_state=42)

print("=== EVALUATING GSI v6.3.3 WITH COST = 0.30 vs COST = 0.35 ===", flush=True)

results = []
for ds in datasets:
    X, Y, _, _ = load_dataset(ds)
    f1s_30, covs_30 = [], []
    f1s_35, covs_35 = [], []
    for fold, (tr, te) in enumerate(cv.split(X, Y)):
        X_tr, X_te = X[tr], X[te]
        Y_tr, Y_te = Y[tr], Y[te]
        scaler = MaxAbsScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        
        # Cost = 0.30 (Strict benchmark standard)
        m30 = GSIMLCPAv6_3_3Classifier(cost=0.30, gamma_min=0.70, tau_balance=0.85, random_state=42+fold).fit(X_tr, Y_tr)
        pred30 = m30.predict(X_te)
        f1s_30.append(compute_selective_macro_f1(Y_te, pred30))
        covs_30.append(np.mean(pred30 != -1))

        # Cost = 0.35 (Current run)
        m35 = GSIMLCPAv6_3_3Classifier(cost=0.35, gamma_min=0.78, tau_balance=0.85, random_state=42+fold).fit(X_tr, Y_tr)
        pred35 = m35.predict(X_te)
        f1s_35.append(compute_selective_macro_f1(Y_te, pred35))
        covs_35.append(np.mean(pred35 != -1))

    f1_30_m, cov_30_m = np.mean(f1s_30), np.mean(covs_30)
    f1_35_m, cov_35_m = np.mean(f1s_35), np.mean(covs_35)
    print(f"{ds:16s} | c=0.30: F1={f1_30_m:.4f}, Cov={cov_30_m*100:.1f}% | c=0.35: F1={f1_35_m:.4f}, Cov={cov_35_m*100:.1f}%", flush=True)
    results.append({
        "dataset": ds,
        "f1_30": f1_30_m, "cov_30": cov_30_m,
        "f1_35": f1_35_m, "cov_35": cov_35_m
    })

df = pd.DataFrame(results)
print("--------------------------------------------------------------------------------", flush=True)
print(f"AVERAGE (10 DS)  | c=0.30: F1={df['f1_30'].mean():.4f}, Cov={df['cov_30'].mean()*100:.1f}% | c=0.35: F1={df['f1_35'].mean():.4f}, Cov={df['cov_35'].mean()*100:.1f}%", flush=True)
