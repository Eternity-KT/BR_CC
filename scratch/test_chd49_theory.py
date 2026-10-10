import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_dataset
from src.models.gsi_v6_3_2 import GSIMLCPAv6_3_2Classifier
from src.evaluation.cv import get_multilabel_cv
from src.evaluation.metrics import compute_selective_macro_f1
from sklearn.preprocessing import MaxAbsScaler
import numpy as np

X, Y, _, lc = load_dataset('chd49')
cv = get_multilabel_cv(n_splits=5, random_state=42)

for fold_idx, (tr, te) in enumerate(cv.split(X, Y)):
    X_tr, X_te = X[tr], X[te]
    Y_tr, Y_te = Y[tr], Y[te]
    scaler = MaxAbsScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    
    m = GSIMLCPAv6_3_2Classifier(base_learner='logistic', random_state=42 + fold_idx)
    m.fit(X_tr, Y_tr)
    preds = m.predict(X_te)
    print(f'Fold {fold_idx + 1}: Sel-F1 = {compute_selective_macro_f1(Y_te, preds):.4f}')
    if fold_idx == 1: # fold 2 was 0.3608
        print('Fold 2 thresholds:')
        for j in range(6):
            th = m.adaptive_thresholds_[j]
            print(f'  L{j} ({lc[j]}): prior={th["prior"]:.4f}, tau_0={th["tau_0"]:.4f}, tau_1={th["tau_1"]:.4f}')
        break
