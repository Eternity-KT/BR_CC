"""
Multi-label Classification Models: Binary Relevance and Classifier Chains.
"""

from .binary_relevance import (
    BinaryRelevanceClassifier,
    BinaryRelevanceLogisticRegression,
    BinaryRelevanceMLP
)
from .classifier_chain import ClassifierChainClassifier
from .mlc_pa import MLCPartialAbstentionClassifier, MLCPAClassifier
from .gsi_mlc_pa import GSIMLCPartialAbstentionClassifier, GSIMLCPAClassifier
try:
    from .pytorch_mlp import (
        MultiLabelMLPClassifier,
        FastPyTorchBinaryMLP,
        get_default_device
    )
except ImportError:  # PyTorch is an optional acceleration dependency.
    MultiLabelMLPClassifier = None
    FastPyTorchBinaryMLP = None

    def get_default_device():
        return "cpu"

__all__ = [
    "BinaryRelevanceClassifier",
    "BinaryRelevanceLogisticRegression",
    "BinaryRelevanceMLP",
    "ClassifierChainClassifier",
    "MLCPartialAbstentionClassifier",
    "MLCPAClassifier",
    "GSIMLCPartialAbstentionClassifier",
    "GSIMLCPAClassifier",
    "MultiLabelMLPClassifier",
    "FastPyTorchBinaryMLP"
]
