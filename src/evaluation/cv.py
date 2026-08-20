"""
Cross-Validation Strategy for Multi-Label Classification
Uses 5-Fold Multilabel Stratified Cross Validation.
"""

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import KFold


def get_multilabel_cv(n_splits=5, random_state=42, shuffle=True):
    """
    Get a 5-fold cross-validation splitter suitable for multi-label data.

    Parameters:
        n_splits (int, default=5): Number of folds.
        random_state (int, default=42): Random seed for reproducibility.
        shuffle (bool, default=True): Whether to shuffle data before splitting.

    Returns:
        cv splitter object with .split(X, Y) generator.
    """
    try:
        return MultilabelStratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
    except Exception:
        # Fallback to standard KFold if iterative stratification encounters issues
        return KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
