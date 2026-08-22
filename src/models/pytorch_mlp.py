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
            Maximum positive class imbalance weight clamp.
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
        self.device = device
        self.random_state = random_state
        self.extra_kwargs = kwargs
        self.model_ = None
        self.device_ = None
        self.n_labels_ = 0

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

        return self

    def decision_function(self, X):
        """Return raw output logits."""
        self.model_.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        X_tensor = torch.as_tensor(X_arr, device=self.device_)

        with torch.no_grad():
            logits = self.model_(X_tensor)
            scores = logits.cpu().numpy()

        if hasattr(self, 'constant_labels_') and self.constant_labels_:
            for j, val in self.constant_labels_.items():
                scores[:, j] = 50.0 if val == 1 else -50.0

        return scores

    def predict_proba(self, X):
        """Return sigmoid predicted probabilities for all labels."""
        self.model_.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        X_tensor = torch.as_tensor(X_arr, device=self.device_)

        with torch.no_grad():
            logits = self.model_(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()

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
        self.device = device
        self.random_state = random_state
        self.extra_kwargs = kwargs
        self.model_ = None
        self.device_ = None

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

        return self

    def decision_function(self, X):
        self.model_.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        X_tensor = torch.as_tensor(X_arr, device=self.device_)
        with torch.no_grad():
            scores = self.model_(X_tensor).squeeze(1).cpu().numpy()
        return scores

    def predict_proba(self, X):
        self.model_.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        X_tensor = torch.as_tensor(X_arr, device=self.device_)
        with torch.no_grad():
            p1 = torch.sigmoid(self.model_(X_tensor).squeeze(1)).cpu().numpy()
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X):
        self.model_.eval()
        X_arr = np.asarray(X, dtype=np.float32)
        X_tensor = torch.as_tensor(X_arr, device=self.device_)
        with torch.no_grad():
            preds = (torch.sigmoid(self.model_(X_tensor).squeeze(1)) >= 0.5).int().cpu().numpy()
        return preds
