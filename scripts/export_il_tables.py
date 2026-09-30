import pandas as pd

df = pd.read_csv('results_v5_1_test/Test_DL_results/table_il_layers_breakdown.csv')
valid = df.dropna(subset=['selective_macro_f1'])

# 1. Macro Table
m = valid.groupby(['Base_Learner', 'Regime'])[['num_labels', 'coverage', 'selective_macro_f1', 'selective_macro_precision', 'subset_01_accuracy', 'hamming_loss_sel']].mean().reset_index()

print("=== LATEX MACRO TABLE ===")
for _, r in m.iterrows():
    b_name = r['Base_Learner'].upper()
    reg_label = {
        'ABL_1_ONLY_IL1': '$IL_1$ (Tầng 1)',
        'ABL_2_ACCUM_IL12': '$IL_{1+2}$ (Tích lũy 1+2)',
        'ABL_3_ALL_IL': '$IL_{1+2+3}$ (Toàn bộ $IL$)'
    }[r['Regime']]
    print(f"\\textbf{{{b_name}}} & {reg_label} & {r['num_labels']:.2f} & {r['coverage']*100:.1f}\\% & {r['selective_macro_f1']:.4f} & {r['selective_macro_precision']:.4f} & {r['subset_01_accuracy']:.4f} & {r['hamming_loss_sel']:.4f} \\\\")

print("\n=== LATEX PER DATASET TABLE ===")
datasets = ['emotions', 'scene', 'chd49', 'music', 'gpositivepseaac', 'genbase', 'humanpseaac', 'plantpseaac', 'viruspseaac', 'yeast']
for d in datasets:
    for b in ['logistic', 'svm', 'mlp']:
        sub = df[(df['Dataset']==d) & (df['Base_Learner']==b)]
        il1 = sub[sub['Regime']=='ABL_1_ONLY_IL1']
        il12 = sub[sub['Regime']=='ABL_2_ACCUM_IL12']
        il123 = sub[sub['Regime']=='ABL_3_ALL_IL']
        if len(il1) > 0 and pd.notna(il1['selective_macro_f1'].values[0]):
            k1, f1_1 = il1['num_labels'].values[0], il1['selective_macro_f1'].values[0]
            k2, f1_2 = il12['num_labels'].values[0], il12['selective_macro_f1'].values[0]
            k3, f1_3 = il123['num_labels'].values[0], il123['selective_macro_f1'].values[0]
            print(f"\\texttt{{{d}}} & \\texttt{{{b}}} & {k1:.1f} & {f1_1:.4f} & {k2:.1f} & {f1_2:.4f} & {k3:.1f} & {f1_3:.4f} \\\\")
        else:
            print(f"\\texttt{{{d}}} & \\texttt{{{b}}} & 0.0 & -- & 0.0 & -- & 0.0 & -- \\\\")
