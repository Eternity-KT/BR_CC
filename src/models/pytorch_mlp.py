"""
GPU-Accelerated Multi-Label Multi-Layer Perceptron (ML-MLP) & Fast Chain MLP in PyTorch.

Features:
- Single Multi-Output Neural Network for predicting all q labels simultaneously on GPU (0.2s - 1s training time).
- BCEWithLogitsLoss with automatic positive class imbalance weighting (pos_weight).
- BatchNorm, ReLU, Dropout, and AdamW optimizer with Cosine Annealing.
- Scikit-learn estimator compatible API (fit, predict, predict_proba, decision_function).
- CUDA GPU acceleration (RTX 4060) with CPU fallback.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression


def get_default_device():
    """Return 'cuda' if GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"


class MultiLabelMLPClassifier(BaseEstimator, ClassifierMixin):
    """
    Multi-Label Multi-Layer Perceptron Classifier running on GPU via PyTorch.
    Predicts all q labels simultaneously in a single forward/backward pass.

    Parameters:
        hidden_layer_sizes (tuple, default=(128, 64)):
            Neurons in hidden layers.
        lr (float, default=3e-3):
            Learning rate for AdamW optimizer.
        weight_decay (float, default=1e-3):
            L2 regularization strength.
        epochs (int, default=150):
            Number of training epochs.
        dropout (float, default=0.15):
            Dropout probability for regularization.
        pos_weight_clamp (float, default=15.0):
            Maximum pos_weight clamp value for class imbalance weighting.
        unbias_pos_weight (bool, default=False):
            If True and calibration is None, subtract log(pos_weight) from logits.
        calibration (str, default='sigmoid'):
            Probability calibration method: 'sigmoid' (Platt scaling) or None.
        device (str, default=None):
            'cuda', 'cpu', or None (auto-detects).
        random_state (int, default=42):
            Seed for reproducibility.
    """
    def __init__(
        self,
        hidden_layer_sizes=(128, 64),
        lr=1e-3,
        weight_decay=1e-3,
        epochs=30,
        dropout=0.15,
        pos_weight_clamp=15.0,
        unbias_pos_weight=False,
        calibration="sigmoid",
        device=None,
        random_state=42,
        **kwargs
    ):
        if "max_iter" in kwargs:
            epochs = kwargs.pop("max_iter")
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.dropout = dropout
        self.pos_weight_clamp = pos_weight_clamp
        self.unbias_pos_weight = unbias_pos_weight
        self.calibration = calibration
        self.device = device
        self.random_state = random_state
        self.extra_kwargs = kwargs
        self.model_ = None
        self.device_ = None
        self.n_labels_ = 0
        self.pos_weight_ = None
        self.calibrators_ = {}

    def _get_device(self):
        if self.device is not None:
            return torch.device(self.device)
        return torch.device(get_default_device())

    def _build_network(self, in_features, out_features):
        layers = []
        prev_dim = in_features

        for h_dim in self.hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU())
            if self.dropout > 0.0:
                layers.append(nn.Dropout(self.dropout))
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, out_features))
        return nn.Sequential(*layers)

    def fit(self, X, Y):
        """
        Fit the Multi-Output MLP model for all q labels simultaneously.
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

        self.device_ = self._get_device()
        X_arr = np.asarray(X, dtype=np.float32)
        Y_arr = np.asarray(Y, dtype=np.float32)
        if Y_arr.ndim == 1:
            Y_arr = Y_arr.reshape(-1, 1)

        n_samples, in_features = X_arr.shape
        self.n_labels_ = Y_arr.shape[1]

        # Detect constant (degenerate) labels in training set
        self.constant_labels_ = {}
        for j in range(self.n_labels_):
            col = Y_arr[:, j]
            if np.all(col == 0):
                self.constant_labels_[j] = 0
            elif np.all(col == 1):
                self.constant_labels_[j] = 1

        # Build network
        self.model_ = self._build_network(in_features, self.n_labels_).to(self.device_)

        # Transfer full batch to GPU directly (0 data-loader latency)
        X_tensor = torch.as_tensor(X_arr, device=self.device_)
        Y_tensor = torch.as_tensor(Y_arr, device=self.device_)

        # Class imbalance weighting
        pos_counts = Y_tensor.sum(dim=0).clamp(min=1.0)
        neg_counts = (float(n_samples) - pos_counts).clamp(min=1.0)
        pos_weight = (neg_counts / pos_counts).clamp(min=1.0, max=self.pos_weight_clamp)
        self.pos_weight_ = pos_weight.detach()

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        self.model_.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            logits = self.model_(X_tensor)
            loss = criterion(logits, Y_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Platt scaling (calibration="sigmoid")
        self.calibrators_ = {}
        if self.calibration == "sigmoid":
            self.model_.eval()
            with torch.no_grad():
                raw_logits = self.model_(X_tensor).cpu().numpy()

            for j in range(self.n_labels_):
                if j in self.constant_labels_:
                    continue
                col = Y_arr[:, j]
                pos_c = int(np.sum(col == 1))
                neg_c = n_samples - pos_c
                if min(pos_c, neg_c) < 2:
                    # Smoothed Laplace prior fallback
                    p_prior = (pos_c + 1.0) / (n_samples + 2.0)
                    b_prior = float(np.log(p_prior / (1.0 - p_prior)))
                    self.calibrators_[j] = (0.0, b_prior)
                else:
                    lr_cal = LogisticRegression(
                        C=1.0,
                        solver="liblinear",
                        random_state=self.random_state if self.random_state is not None else 42,
                    )
                    lr_cal.fit(raw_logits[:, j:j+1], col)
                    a_j = float(lr_cal.coef_[0, 0])
                    b_j = float(lr_cal.intercept_[0])
                    if a_j <= 0.0:
                        a_j = 1.0
                    self.calibrators_[j] = (a_j, b_j)

        return self

    def decision_function(self, X):
        """Return output logits (calibrated if calibration='sigmoid')."""
        self.model_.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        X_tensor = torch.as_tensor(X_arr, device=self.device_)

        with torch.no_grad():
            logits = self.model_(X_tensor)
            if self.unbias_pos_weight and not self.calibration and hasattr(self, "pos_weight_") and self.pos_weight_ is not None:
                logits = logits - torch.log(self.pos_weight_)
            scores = logits.cpu().numpy()

        if self.calibration == "sigmoid" and hasattr(self, "calibrators_"):
            for j, (a_j, b_j) in self.calibrators_.items():
                scores[:, j] = a_j * scores[:, j] + b_j

        if hasattr(self, 'constant_labels_') and self.constant_labels_:
            for j, val in self.constant_labels_.items():
                scores[:, j] = 50.0 if val == 1 else -50.0

        return scores

    def predict_proba(self, X):
        """Return calibrated predicted probabilities for all labels."""
        scores = self.decision_function(X)
        probs = 1.0 / (1.0 + np.exp(-np.clip(scores, -50.0, 50.0)))

        if hasattr(self, 'constant_labels_') and self.constant_labels_:
            for j, val in self.constant_labels_.items():
                probs[:, j] = float(val)

        return probs

    def predict(self, X):
        """Predict binary indicators {0, 1} for all labels."""
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(np.int32)


class FastPyTorchBinaryMLP(BaseEstimator, ClassifierMixin):
    """
    Fast GPU Binary MLP Classifier for Classifier Chains.
    Executes full-batch GPU training without Python DataLoader overhead.
    """
    def __init__(
        self,
        hidden_layer_sizes=(64,),
        lr=1e-3,
        weight_decay=1e-3,
        epochs=30,
        unbias_pos_weight=False,
        calibration="sigmoid",
        device=None,
        random_state=42,
        **kwargs
    ):
        if "max_iter" in kwargs:
            epochs = kwargs.pop("max_iter")
        self.hidden_layer_sizes = hidden_layer_sizes
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.unbias_pos_weight = unbias_pos_weight
        self.calibration = calibration
        self.device = device
        self.random_state = random_state
        self.extra_kwargs = kwargs
        self.model_ = None
        self.device_ = None
        self.pos_weight_ = None
        self.calibrator_ = None
        self.constant_label_ = None

    def _get_device(self):
        if self.device is not None:
            return torch.device(self.device)
        return torch.device(get_default_device())

    def fit(self, X, y):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

        self.device_ = self._get_device()
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        n_samples, in_features = X_arr.shape
        unique_y = np.unique(y_arr)
        if len(unique_y) <= 1:
            self.constant_label_ = int(unique_y[0]) if len(unique_y) == 1 else 0
            return self
        self.constant_label_ = None

        h_dim = self.hidden_layer_sizes[0] if len(self.hidden_layer_sizes) > 0 else 64

        self.model_ = nn.Sequential(
            nn.Linear(in_features, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1)
        ).to(self.device_)

        X_tensor = torch.as_tensor(X_arr, device=self.device_)
        y_tensor = torch.as_tensor(y_arr, device=self.device_)

        pos_count = float(y_tensor.sum().item())
        neg_count = float(n_samples) - pos_count
        pos_weight = torch.tensor([min(max(neg_count / max(pos_count, 1.0), 1.0), 10.0)], device=self.device_)
        self.pos_weight_ = pos_weight.detach()

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.model_.train()
        for _ in range(self.epochs):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(self.model_(X_tensor), y_tensor)
            loss.backward()
            optimizer.step()

        # Calibration
        self.calibrator_ = None
        if self.calibration == "sigmoid":
            self.model_.eval()
            with torch.no_grad():
                raw_logits = self.model_(X_tensor).squeeze(1).cpu().numpy().reshape(-1, 1)
            pos_c = int(pos_count)
            neg_c = int(neg_count)
            if min(pos_c, neg_c) < 2:
                p_prior = (pos_c + 1.0) / (n_samples + 2.0)
                b_prior = float(np.log(p_prior / (1.0 - p_prior)))
                self.calibrator_ = (0.0, b_prior)
            else:
                lr_cal = LogisticRegression(
                    C=1.0,
                    solver="liblinear",
                    random_state=self.random_state if self.random_state is not None else 42,
                )
                lr_cal.fit(raw_logits, y_arr.ravel())
                a = float(lr_cal.coef_[0, 0])
                b = float(lr_cal.intercept_[0])
                if a <= 0.0:
                    a = 1.0
                self.calibrator_ = (a, b)

        return self

    def decision_function(self, X):
        if self.constant_label_ is not None:
            return np.full(len(X), 50.0 if self.constant_label_ == 1 else -50.0, dtype=np.float32)

        self.model_.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        X_tensor = torch.as_tensor(X_arr, device=self.device_)
        with torch.no_grad():
            scores = self.model_(X_tensor).squeeze(1).cpu().numpy()
            if self.unbias_pos_weight and not self.calibration and hasattr(self, "pos_weight_") and self.pos_weight_ is not None:
                scores = scores - float(torch.log(self.pos_weight_).item())

        if self.calibration == "sigmoid" and hasattr(self, "calibrator_") and self.calibrator_ is not None:
            a, b = self.calibrator_
            scores = a * scores + b

        return scores

    def predict_proba(self, X):
        if self.constant_label_ is not None:
            p1 = np.full(len(X), float(self.constant_label_), dtype=np.float32)
            p0 = 1.0 - p1
            return np.column_stack([p0, p1])
        scores = self.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(-np.clip(scores, -50.0, 50.0)))
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        return (self.decision_function(X) >= 0.0).astype(np.int32)
