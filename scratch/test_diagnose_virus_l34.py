import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.data.loader import load_dataset
from src.models.gsi_v6_3_2 import GSIMLCPAv6_3_2Classifier
from src.models.gsi_v6_3_3 import GSIMLCPAv6_3_3Classifier
from src.evaluation.cv import get_multilabel_cv
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import f1_score, precision_score, recall_score

X, Y, _, _ = load_dataset('viruspseaac')
cv = get_multilabel_cv(n_splits=5, random_state=42)

for learner_key, display_name in [('logistic', 'LOGISTIC'), ('svm_calibrated', 'SVM'), ('mlp', 'MLP')]:
    print(f'\n==================================== {display_name} ====================================')
    for target_label in [3, 4]:
        f1_2_list, f1_3_list = [], []
        prec_2_list, prec_3_list = [], []
        rec_2_list, rec_3_list = [], []
        cov_2_list, cov_3_list = [], []
        t0_2_list, t0_3_list = [], []
        t1_2_list, t1_3_list = [], []
        
        for fold_idx, (tr, te) in enumerate(cv.split(X, Y)):
            X_tr, X_te = X[tr], X[te]
            Y_tr, Y_te = Y[tr], Y[te]
            scaler = MaxAbsScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)
            
            m2 = GSIMLCPAv6_3_2Classifier(base_learner=learner_key, random_state=42 + fold_idx)
            m2.fit(X_tr, Y_tr)
            pred2 = m2.predict(X_te)[:, target_label]
            p2 = m2.predict_proba(X_te)[:, target_label]
            
            m3 = GSIMLCPAv6_3_3Classifier(base_learner=learner_key, random_state=42 + fold_idx)
            m3.fit(X_tr, Y_tr)
            pred3 = m3.predict(X_te)[:, target_label]
            p3 = m3.predict_proba(X_te)[:, target_label]
            
            y_true = Y_te[:, target_label]
            
            # v6.3.2
            mask2 = pred2 != -1
            cov2 = np.mean(mask2)
            f1_2 = f1_score(y_true[mask2], pred2[mask2], zero_division=0) if np.sum(mask2) > 0 else 0
            p_2 = precision_score(y_true[mask2], pred2[mask2], zero_division=0) if np.sum(mask2) > 0 else 0
            r_2 = recall_score(y_true[mask2], pred2[mask2], zero_division=0) if np.sum(mask2) > 0 else 0
            
            # v6.3.3
            mask3 = pred3 != -1
            cov3 = np.mean(mask3)
            f1_3 = f1_score(y_true[mask3], pred3[mask3], zero_division=0) if np.sum(mask3) > 0 else 0
            p_3 = precision_score(y_true[mask3], pred3[mask3], zero_division=0) if np.sum(mask3) > 0 else 0
            r_3 = recall_score(y_true[mask3], pred3[mask3], zero_division=0) if np.sum(mask3) > 0 else 0
            
            f1_2_list.append(f1_2)
            f1_3_list.append(f1_3)
            prec_2_list.append(p_2)
            prec_3_list.append(p_3)
            rec_2_list.append(r_2)
            rec_3_list.append(r_3)
            cov_2_list.append(cov2)
            cov_3_list.append(cov3)
            
            t0_2_list.append(m2.adaptive_thresholds_[target_label]['tau_0'])
            t1_2_list.append(m2.adaptive_thresholds_[target_label]['tau_1'])
            t0_3_list.append(m3.adaptive_thresholds_[target_label]['tau_0'])
            t1_3_list.append(m3.adaptive_thresholds_[target_label]['tau_1'])
            
        print(f'-- Label {target_label} (Prior = {np.mean(Y[:, target_label]):.3f}) --')
        print(f'  Thresholds: v6.3.2 [t0={np.mean(t0_2_list):.3f}, t1={np.mean(t1_2_list):.3f}] vs v6.3.3 [t0={np.mean(t0_3_list):.3f}, t1={np.mean(t1_3_list):.3f}]')
        print(f'  Coverage:   v6.3.2 = {np.mean(cov_2_list):.3f} vs v6.3.3 = {np.mean(cov_3_list):.3f} (diff: {np.mean(cov_3_list)-np.mean(cov_2_list):+.3f})')
        print(f'  Precision:  v6.3.2 = {np.mean(prec_2_list):.3f} vs v6.3.3 = {np.mean(prec_3_list):.3f} (diff: {np.mean(prec_3_list)-np.mean(prec_2_list):+.3f})')
        print(f'  Recall:     v6.3.2 = {np.mean(rec_2_list):.3f} vs v6.3.3 = {np.mean(rec_3_list):.3f} (diff: {np.mean(rec_3_list)-np.mean(rec_2_list):+.3f})')
        print(f'  F1 Score:   v6.3.2 = {np.mean(f1_2_list):.3f} vs v6.3.3 = {np.mean(f1_3_list):.3f} (diff: {np.mean(f1_3_list)-np.mean(f1_2_list):+.3f})')
