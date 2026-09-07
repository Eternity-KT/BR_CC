"""Smoke test for BipartiteGSIPartialAbstentionClassifier."""

import numpy as np
from src.data.loader import load_dataset
from test_il_dl.bipartite_gsi import BipartiteGSIPartialAbstentionClassifier
from src.evaluation.metrics import compute_selective_macro_f1

def smoke_test():
    X, Y, _, _ = load_dataset("emotions")
    X_train, Y_train = X[:400], Y[:400]
    X_test, Y_test = X[400:], Y[400:]

    model = BipartiteGSIPartialAbstentionClassifier(
        cost=0.3,
        base_learner="mlp",
        random_state=42,
    )
    print("Fitting BipartiteGSIPartialAbstentionClassifier on emotions...")
    model.fit(X_train, Y_train)

    print(f"Fit complete in {model.total_fit_time_:.2f}s (selection: {model.selection_time_:.2f}s)")
    print(f"Independent labels ({len(model.independent_labels_)}): {model.independent_labels_}")
    print(f"Dependent labels ({len(model.dependent_labels_)}): {model.dependent_labels_}")

    probs = model.predict_proba(X_test)
    assert probs.shape == Y_test.shape, f"Invalid proba shape: {probs.shape}"
    assert np.all((probs >= 0.0) & (probs <= 1.0)), "Probabilities outside [0, 1]"

    # Test full prediction
    pred_full = model.predict_full_from_proba(probs)
    assert pred_full.shape == Y_test.shape

    # Test rejection prediction at cost 0.3
    pred_partial = model.predict_from_proba(probs, cost=0.3)
    assert pred_partial.shape == Y_test.shape
    assert set(np.unique(pred_partial)).issubset({-1, 0, 1})

    coverage = np.mean(pred_partial != -1)
    sel_macro_f1 = compute_selective_macro_f1(Y_test, pred_partial)
    print(f"Full predictions shape: {pred_full.shape}")
    print(f"Coverage at cost 0.3: {coverage:.4f}")
    print(f"Selective Macro-F1 at cost 0.3: {sel_macro_f1:.4f}")
    print("Smoke test passed successfully!")

if __name__ == "__main__":
    smoke_test()
