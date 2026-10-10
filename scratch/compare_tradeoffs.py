import pandas as pd
import numpy as np

# Load datasets
v5_df = pd.read_csv('results_v5_1_test/benchmark_3_base_learners_summary.csv')
v5_strat = v5_df[v5_df['Model'].str.contains('Stratified')].copy()
v5_clean = v5_strat[['Dataset', 'Base_Learner', 'Selective_Macro_F1', 'Coverage', 'Full_Macro_F1']].copy()
v5_clean.columns = ['dataset', 'learner', 'sel_f1_v5', 'cov_v5', 'full_f1_v5']

decay_df = pd.read_csv('results_v6_2/decay_study/decay_summary.csv')
v62_decay = decay_df[decay_df['model'] == 'GSI_v6_2_Decay'].copy()
v62_clean = v62_decay[['dataset', 'learner', 'Selective_Macro_F1_mean', 'Coverage_mean', 'Full_Macro_F1_mean']].copy()
v62_clean.columns = ['dataset', 'learner', 'sel_f1_v621', 'cov_v621', 'full_f1_v621']

v63_df = pd.read_csv('results_v6_3/v6_3_all10ds_summary.csv')
v63_sub = v63_df[v63_df['model'] == 'GSI_v6_3'].copy()
v63_clean = v63_sub[['dataset', 'learner', 'Selective_Macro_F1_mean', 'Coverage_mean', 'Full_Macro_F1_mean']].copy()
v63_clean.columns = ['dataset', 'learner', 'sel_f1_v63', 'cov_v63', 'full_f1_v63']

m = pd.merge(pd.merge(v5_clean, v62_clean, on=['dataset', 'learner']), v63_clean, on=['dataset', 'learner'])

m['ratio_v5'] = m['sel_f1_v5'] / m['cov_v5']
m['ratio_v621'] = m['sel_f1_v621'] / m['cov_v621']
m['ratio_v63'] = m['sel_f1_v63'] / m['cov_v63']

m['eff_v5'] = m['sel_f1_v5'] * m['cov_v5']
m['eff_v621'] = m['sel_f1_v621'] * m['cov_v621']
m['eff_v63'] = m['sel_f1_v63'] * m['cov_v63']

# Wins/Losses
print("--- HEAD TO HEAD COMPARISONS (out of 30 configs) ---")
print("v6.2.1 vs v5.1.1 on Coverage: v6.2.1 wins in", (m['cov_v621'] > m['cov_v5']).sum(), "/ 30")
print("v6.2.1 vs v5.1.1 on Full Macro-F1: v6.2.1 wins in", (m['full_f1_v621'] > m['full_f1_v5']).sum(), "/ 30")
print("v6.2.1 vs v5.1.1 on Sel Macro-F1: v6.2.1 wins in", (m['sel_f1_v621'] > m['sel_f1_v5']).sum(), "/ 30")
print("v6.2.1 vs v5.1.1 on Ratio (F1/Cov): v6.2.1 wins in", (m['ratio_v621'] > m['ratio_v5']).sum(), "/ 30")
print("v6.2.1 vs v5.1.1 on Effective F1 (F1*Cov): v6.2.1 wins in", (m['eff_v621'] > m['eff_v5']).sum(), "/ 30")

print("\nv6.3 vs v6.2.1 on Sel Macro-F1: v6.3 wins in", (m['sel_f1_v63'] > m['sel_f1_v621']).sum(), "/ 30")
print("v6.3 vs v6.2.1 on Full Macro-F1: v6.3 wins in", (m['full_f1_v63'] > m['full_f1_v621']).sum(), "/ 30")
print("v6.3 vs v6.2.1 on Coverage: v6.3 wins in", (m['cov_v63'] > m['cov_v621']).sum(), "/ 30")
print("v6.3 vs v6.2.1 on Ratio (F1/Cov): v6.3 wins in", (m['ratio_v63'] > m['ratio_v621']).sum(), "/ 30")
print("v6.3 vs v6.2.1 on Effective F1 (F1*Cov): v6.3 wins in", (m['eff_v63'] > m['eff_v621']).sum(), "/ 30")

print("\nv6.3 vs v5.1.1 on Effective F1 (F1*Cov): v6.3 wins in", (m['eff_v63'] > m['eff_v5']).sum(), "/ 30")
print("v6.3 vs v5.1.1 on Full Macro-F1: v6.3 wins in", (m['full_f1_v63'] > m['full_f1_v5']).sum(), "/ 30")
print("v6.3 vs v5.1.1 on Coverage: v6.3 wins in", (m['cov_v63'] > m['cov_v5']).sum(), "/ 30")

# Save merged table to csv for reference
m.to_csv('scratch/merged_comparison.csv', index=False)
print("\nSaved scratch/merged_comparison.csv")
