import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from src.data.loader import load_dataset
from src.models.gsi_v6_3_2 import GSIMLCPAv6_3_2Classifier
from src.models.gsi_v6_3_3 import GSIMLCPAv6_3_3Classifier
from src.evaluation.cv import get_multilabel_cv
from src.evaluation.metrics import compute_selective_macro_f1
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import f1_score

X, Y, _, _ = load_dataset('viruspseaac')
cv = get_multilabel_cv(n_splits=5, random_state=42)

for learner in ['logistic', 'svm', 'mlp']:
    print(f'==================================== {learner.upper()} ====================================')
    f1s_2, f1s_3 = [], []
    covs_2, covs_3 = [], []
    label_f1s_2 = [[] for _ in range(6)]
    label_f1s_3 = [[] for _ in range(6)]
    label_covs_2 = [[] for _ in range(6)]
    label_covs_3 = [[] for _ in range(6)]
    
    for fold_idx, (tr, te) in enumerate(cv.split(X, Y)):
        X_tr, X_te = X[tr], X[te]
        Y_tr, Y_te = Y[tr], Y[te]
        scaler = MaxAbsScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        
        m2 = GSIMLCPAv6_3_2Classifier(base_learner=learner, random_state=42 + fold_idx)
        m2.fit(X_tr, Y_tr)
        pred2 = m2.predict(X_te)
        f1_2 = compute_selective_macro_f1(Y_te, pred2)
        cov_2 = np.mean(pred2 != -1)
        f1s_2.append(f1_2)
        covs_2.append(cov_2)
        
        m3 = GSIMLCPAv6_3_3Classifier(base_learner=learner, tau_balance=0.75, blend_kappa=20.0, random_state=42 + fold_idx)
        m3.fit(X_tr, Y_tr)
        pred3 = m3.predict(X_te)
        f1_3 = compute_selective_macro_f1(Y_te, pred3)
        cov_3 = np.mean(pred3 != -1)
        f1s_3.append(f1_3)
        covs_3.append(cov_3)
        
        for j in range(6):
            m_2 = pred2[:, j] != -1
            score_2 = f1_score(Y_te[m_2, j], pred2[m_2, j], zero_division=0) if np.sum(m_2) > 0 else 0
            label_f1s_2[j].append(score_2)
            label_covs_2[j].append(np.mean(m_2))
            
            m_3 = pred3[:, j] != -1
            score_3 = f1_score(Y_te[m_3, j], pred3[m_3, j], zero_division=0) if np.sum(m_3) > 0 else 0
            label_f1s_3[j].append(score_3)
            label_covs_3[j].append(np.mean(m_3))
            
    print(f'Overall Macro-F1: v6.3.2={np.mean(f1s_2):.4f} | v6.3.3={np.mean(f1s_3):.4f} (diff: {np.mean(f1s_3)-np.mean(f1s_2):+.4f})')
    print(f'Overall Coverage: v6.3.2={np.mean(covs_2):.4f} | v6.3.3={np.mean(covs_3):.4f} (diff: {np.mean(covs_3)-np.mean(covs_2):+.4f})')
    print('Per-label breakdown (F1 | Coverage):')
    for j in range(6):
        mean_f1_2 = np.mean(label_f1s_2[j])
        mean_f1_3 = np.mean(label_f1s_3[j])
        mean_cov_2 = np.mean(label_covs_2[j])
        mean_cov_3 = np.mean(label_covs_3[j])
        print(f'  L{j}: F1 v632={mean_f1_2:.4f} -> v633={mean_f1_3:.4f} ({mean_f1_3-mean_f1_2:+.4f}) | Cov v632={mean_cov_2:.4f} -> v633={mean_cov_3:.4f} ({mean_cov_3-mean_cov_2:+.4f})')
