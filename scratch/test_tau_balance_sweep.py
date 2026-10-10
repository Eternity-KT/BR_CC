import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_dataset
from src.models.gsi_v6_3_2 import GSIMLCPAv6_3_2Classifier
from src.models.gsi_v6_3_3 import GSIMLCPAv6_3_3Classifier
from src.evaluation.cv import get_multilabel_cv
from src.evaluation.metrics import compute_selective_macro_f1
from sklearn.preprocessing import MaxAbsScaler
import numpy as np

for ds_name in ['chd49', 'emotions', 'music', 'viruspseaac']:
    X, Y, _, _ = load_dataset(ds_name)
    cv = get_multilabel_cv(n_splits=5, random_state=42)
    f1_v2 = []
    f1_v3 = []
    
    for fold_idx, (tr, te) in enumerate(cv.split(X, Y)):
        X_tr, X_te = X[tr], X[te]
        Y_tr, Y_te = Y[tr], Y[te]
        scaler = MaxAbsScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        
        m2 = GSIMLCPAv6_3_2Classifier(base_learner='logistic', random_state=42 + fold_idx)
        m2.fit(X_tr, Y_tr)
        f1_v2.append(compute_selective_macro_f1(Y_te, m2.predict(X_te)))
        
        # Test v6.3.3 with tau_balance = 0.75
        m3 = GSIMLCPAv6_3_3Classifier(
            base_learner='logistic',
            tau_balance=0.75,
            blend_kappa=20.0,
            random_state=42 + fold_idx
        )
        m3.fit(X_tr, Y_tr)
        f1_v3.append(compute_selective_macro_f1(Y_te, m3.predict(X_te)))
        
    diff = np.mean(f1_v3) - np.mean(f1_v2)
    sign = "+" if diff >= 0 else ""
    print(f'{ds_name:15s} | v6.3.2: {np.mean(f1_v2):.4f} | v6.3.3: {np.mean(f1_v3):.4f} ({sign}{diff:.4f})')
