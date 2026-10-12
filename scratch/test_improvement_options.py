import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.data.loader import load_dataset
from src.models.gsi_v6_3_3 import GSIMLCPAv6_3_3Classifier
from src.models.gsi_v6_3_2 import GSIMLCPAv6_3_2Classifier
from src.calibration.adaptive_calibrator import MultiLabelAdaptiveCalibrator
from src.calibration.tail_calibrator import TailCalibrator
from src.evaluation.cv import get_multilabel_cv
from src.evaluation.metrics import compute_selective_macro_f1
from sklearn.preprocessing import MaxAbsScaler

# Test proposed modifications on viruspseaac, chd49, emotions
cv = get_multilabel_cv(n_splits=5, random_state=42)

for ds_name in ['viruspseaac', 'emotions', 'chd49']:
    X, Y, _, _ = load_dataset(ds_name)
    print(f"\n==================== DATASET: {ds_name} ====================")
    for learner in ['logistic', 'svm_calibrated', 'mlp']:
        f1_v632, f1_v633_base, f1_v633_opt1, f1_v633_opt2, f1_v633_opt_both = [], [], [], [], []
        
        for fold, (tr, te) in enumerate(cv.split(X, Y)):
            X_tr, X_te = X[tr], X[te]
            Y_tr, Y_te = Y[tr], Y[te]
            scaler = MaxAbsScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            
            # 1. v6.3.2 baseline
            m2 = GSIMLCPAv6_3_2Classifier(base_learner=learner, random_state=42 + fold).fit(X_tr, Y_tr)
            f1_v632.append(compute_selective_macro_f1(Y_te, m2.predict(X_te)))
            
            # 2. v6.3.3 baseline
            m3 = GSIMLCPAv6_3_3Classifier(base_learner=learner, random_state=42 + fold).fit(X_tr, Y_tr)
            f1_v633_base.append(compute_selective_macro_f1(Y_te, m3.predict(X_te)))
            
            # 3. Option 1: Cap tau_1 <= 0.55 for labels with prior <= 0.50
            P3 = m3.predict_proba(X_te)
            preds_opt1 = np.full_like(P3, -1, dtype=np.int32)
            for j in range(Y.shape[1]):
                t0 = m3.adaptive_thresholds_[j]['tau_0']
                t1 = m3.adaptive_thresholds_[j]['tau_1']
                prior_j = m3.adaptive_thresholds_[j]['prior']
                if prior_j <= 0.50:
                    t1 = min(t1, 0.55)
                    # ensure separation
                    if t1 - t0 < 0.02:
                        t0 = max(0.005, t1 - 0.02)
                preds_opt1[P3[:, j] <= t0, j] = 0
                preds_opt1[P3[:, j] >= t1, j] = 1
            f1_v633_opt1.append(compute_selective_macro_f1(Y_te, preds_opt1))
            
        print(f"[{learner:14s}] v6.3.2: {np.mean(f1_v632):.4f} | v6.3.3: {np.mean(f1_v633_base):.4f} | Opt1 (Cap tau_1<=0.55): {np.mean(f1_v633_opt1):.4f}")
