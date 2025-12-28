"""Tests for vocabulary consistency between preprocessing and model loading."""

import os
import tempfile
import unittest

import torch
import numpy as np

from hierarchical.models.transaction import TransactionEncoder
from hierarchical.models.day import DayEncoder
from hierarchical.models.account import AccountEncoder
from hierarchical.data.dataset import collate_hierarchical
from hierarchical.training.utils import recursive_to_device


class TestVocabConsistency(unittest.TestCase):
    """Tests for vocabulary size consistency."""

    def test_vocab_size_mismatch_causes_error(self):
        """Model with smaller vocab should fail on larger vocab IDs."""
        # Create model with small vocab
        txn = TransactionEncoder(
            num_categories_group=10,
            num_categories_sub=10,
            num_counter_parties=100,  # Small vocab!
            embedding_dim=32,
            use_counter_party=True,
            use_balance=False
        )
        day = DayEncoder(txn, hidden_dim=32, num_layers=1, num_heads=2)
        model = AccountEncoder(day, hidden_dim=32, num_layers=1, num_heads=2)
        model.eval()
        
        # Create batch with IDs exceeding vocab size
        def make_stream(B, D, T):
            return {
                "cat_group": torch.randint(0, 10, (B, D, T)),
                "cat_sub": torch.randint(0, 10, (B, D, T)),
                "cat_cp": torch.randint(100, 200, (B, D, T)),  # IDs > vocab size!
                "amounts": torch.randn(B, D, T),
                "dates": torch.zeros(B, D, T, 4),
                "balance": None,
                "mask": torch.ones(B, D, T, dtype=torch.bool),
                "has_data": torch.ones(B, D, dtype=torch.bool),
            }
        
        batch = {
            "pos": make_stream(2, 5, 3),
            "neg": make_stream(2, 5, 3),
            "meta": {
                "day_mask": torch.ones(2, 5, dtype=torch.bool),
                "day_month": torch.randint(0, 12, (2, 5)),
                "day_weekend": torch.randint(0, 2, (2, 5)),
            },
        }
        
        # This should raise IndexError
        with self.assertRaises(IndexError):
            with torch.no_grad():
                model(batch)

    def test_vocab_sizes_from_checkpoint(self):
        """Verify that vocab sizes can be correctly inferred from checkpoint."""
        # Create model with known vocab sizes
        VOCAB_SIZES = {
            'cat_grp': 36,
            'cat_sub': 152,
            'cat_cp': 10000
        }
        
        txn = TransactionEncoder(
            num_categories_group=VOCAB_SIZES['cat_grp'],
            num_categories_sub=VOCAB_SIZES['cat_sub'],
            num_counter_parties=VOCAB_SIZES['cat_cp'],
            embedding_dim=64,
            use_counter_party=True,
            use_balance=True
        )
        day = DayEncoder(txn, hidden_dim=64, num_layers=2, num_heads=4)
        model = AccountEncoder(day, hidden_dim=64, num_layers=2, num_heads=4)
        
        state_dict = model.state_dict()
        
        # Infer vocab sizes from state_dict (like eval script does)
        prefix = 'day_encoder.txn_encoder.'
        inferred_cat_grp = state_dict[f'{prefix}cat_group_emb.weight'].shape[0]
        inferred_cat_sub = state_dict[f'{prefix}cat_sub_emb.weight'].shape[0]
        inferred_cat_cp = state_dict[f'{prefix}counter_party_emb.weight'].shape[0]
        
        self.assertEqual(inferred_cat_grp, VOCAB_SIZES['cat_grp'])
        self.assertEqual(inferred_cat_sub, VOCAB_SIZES['cat_sub'])
        self.assertEqual(inferred_cat_cp, VOCAB_SIZES['cat_cp'])


if __name__ == "__main__":
    unittest.main()
