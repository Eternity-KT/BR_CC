import numpy as np
from src.models import (
    BinaryRelevanceClassifier,
    BinaryRelevanceLogisticRegression,
    BinaryRelevanceMLP,
    ClassifierChainClassifier
)

def run_tests():
    # 1. Synthetic data
    np.random.seed(42)
    X = np.random.randn(60, 12)
    Y = np.random.randint(0, 2, size=(60, 5))
    # Make label 4 degenerate (all 0)
    Y[:, 4] = 0

    print("1. Testing BinaryRelevanceLogisticRegression...")
    clf_lr = BinaryRelevanceLogisticRegression()
    clf_lr.fit(X, Y)
    preds_lr = clf_lr.predict(X)
    probs_lr = clf_lr.predict_proba(X)
    scores_lr = clf_lr.decision_function(X)

    assert preds_lr.shape == (60, 5), f"Shape mismatch: {preds_lr.shape}"
    assert probs_lr.shape == (60, 5), f"Probs shape mismatch: {probs_lr.shape}"
    assert scores_lr.shape == (60, 5), f"Scores shape mismatch: {scores_lr.shape}"
    assert np.all((probs_lr >= 0.0) & (probs_lr <= 1.0)), "Probs out of [0, 1] range"
    assert np.all(preds_lr[:, 4] == 0), "Degenerate label not 0"
    assert np.all(probs_lr[:, 4] == 0.0), "Degenerate label prob not 0.0"
    print("   -> Passed BinaryRelevanceLogisticRegression!")

    print("2. Testing BinaryRelevanceClassifier with base_estimator='logistic'...")
    clf_br_str = BinaryRelevanceClassifier(base_estimator="logistic")
    clf_br_str.fit(X, Y)
    preds_br = clf_br_str.predict(X)
    probs_br = clf_br_str.predict_proba(X)
    assert preds_br.shape == (60, 5)
    assert probs_br.shape == (60, 5)
    assert np.all((probs_br >= 0.0) & (probs_br <= 1.0))
    print("   -> Passed BinaryRelevanceClassifier(base_estimator='logistic')!")

    print("3. Testing BinaryRelevanceClassifier with base_estimator='svm' (LinearSVC)...")
    clf_br_svm = BinaryRelevanceClassifier(base_estimator="svm")
    clf_br_svm.fit(X, Y)
    preds_svm = clf_br_svm.predict(X)
    probs_svm = clf_br_svm.predict_proba(X)
    assert preds_svm.shape == (60, 5)
    assert probs_svm.shape == (60, 5)
    assert np.all((probs_svm >= 0.0) & (probs_svm <= 1.0))
    print("   -> Passed BinaryRelevanceClassifier(base_estimator='svm')!")

    print("4. Testing BinaryRelevanceMLP...")
    clf_br_mlp = BinaryRelevanceMLP(hidden_layer_sizes=(32,), max_iter=100)
    clf_br_mlp.fit(X, Y)
    preds_mlp = clf_br_mlp.predict(X)
    probs_mlp = clf_br_mlp.predict_proba(X)
    assert preds_mlp.shape == (60, 5)
    assert probs_mlp.shape == (60, 5)
    assert np.all((probs_mlp >= 0.0) & (probs_mlp <= 1.0))
    assert np.all(preds_mlp[:, 4] == 0)
    print("   -> Passed BinaryRelevanceMLP!")

    print("5. Testing ClassifierChain with base_estimator='logistic'...")
    clf_cc_lr = ClassifierChainClassifier(base_estimator="logistic")
    clf_cc_lr.fit(X, Y)
    preds_cc = clf_cc_lr.predict(X)
    probs_cc = clf_cc_lr.predict_proba(X)
    assert preds_cc.shape == (60, 5)
    assert probs_cc.shape == (60, 5)
    assert np.all((probs_cc >= 0.0) & (probs_cc <= 1.0))
    print("   -> Passed ClassifierChainClassifier(base_estimator='logistic')!")

    print("6. Testing ClassifierChain with base_estimator='mlp'...")
    clf_cc_mlp = ClassifierChainClassifier(base_estimator="mlp")
    clf_cc_mlp.fit(X, Y)
    preds_cc_mlp = clf_cc_mlp.predict(X)
    probs_cc_mlp = clf_cc_mlp.predict_proba(X)
    assert preds_cc_mlp.shape == (60, 5)
    assert probs_cc_mlp.shape == (60, 5)
    assert np.all((probs_cc_mlp >= 0.0) & (probs_cc_mlp <= 1.0))
    print("   -> Passed ClassifierChainClassifier(base_estimator='mlp')!")

    print("\nALL UNIT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_tests()

