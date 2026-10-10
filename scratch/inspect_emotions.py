import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loader import load_dataset
from src.models.gsi_v6_3_2 import GSIMLCPAv6_3_2Classifier
from src.models.gsi_v6_3_3 import GSIMLCPAv6_3_3Classifier
from sklearn.preprocessing import MaxAbsScaler
import numpy as np

X, Y, fc, lc = load_dataset('emotions')
X = MaxAbsScaler().fit_transform(X)

m2 = GSIMLCPAv6_3_2Classifier(base_learner='logistic', random_state=42)
m2.fit(X, Y)

m3 = GSIMLCPAv6_3_3Classifier(base_learner='logistic', random_state=42)
m3.fit(X, Y)

print('=== Emotions Comparison ===')
for j in range(Y.shape[1]):
    prior = float(np.mean(Y[:, j]))
    th2 = m2.adaptive_thresholds_[j]
    th3 = m3.adaptive_thresholds_[j]
    t0_2, t1_2 = th2['tau_0'], th2['tau_1']
    t0_3, t1_3 = th3['tau_0'], th3['tau_1']
    cov2 = th2['empirical_coverage']
    cov3 = th3['empirical_coverage']
    bw3 = th3['blend_weight']
    reg3 = th3['regime']
    print(f'L{j} ({lc[j]}): prior={prior:.4f}')
    print(f'   v6.3.2: tau_0={t0_2:.4f}, tau_1={t1_2:.4f}, cov={cov2:.4f}')
    print(f'   v6.3.3: tau_0={t0_3:.4f}, tau_1={t1_3:.4f}, cov={cov3:.4f}, blend_w={bw3:.4f}, regime={reg3}')

p2 = m2.predict_proba(X)
p3 = m3.predict_proba(X)
print('\nMean probability diff (v6.3.3 - v6.3.2):', np.mean(np.abs(p3 - p2)))
