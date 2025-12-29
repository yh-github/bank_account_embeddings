"""Hierarchical account embeddings for transaction data.

This package provides a hierarchical neural network architecture for learning
account-level embeddings from banking transaction sequences.

Architecture:
    TransactionEncoder -> DayEncoder -> AccountEncoder

Usage:
    from hierarchical.models import AccountEncoder, DayEncoder, TransactionEncoder
    from hierarchical.data import HierarchicalDataset, collate_hierarchical
    from hierarchical.training import contrastive_loss
"""

import warnings

# Suppress PyTorch NestedTensor prototype user warning
# This appears when using padded masks in TransformerEncoderLayer on certain PyTorch versions.
# We accept the internal optimization and silence the warning.
warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors.*", category=UserWarning
)

from .models import AccountEncoder, DayEncoder, TransactionEncoder
from .data import HierarchicalDataset, collate_hierarchical, CategoricalVocabulary

__all__ = [
    "AccountEncoder",
    "DayEncoder",
    "TransactionEncoder",
    "HierarchicalDataset",
    "collate_hierarchical",
    "CategoricalVocabulary",
]
