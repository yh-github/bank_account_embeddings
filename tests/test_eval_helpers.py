"""
Tests for evaluation helper functions.

Tests the utility functions used in the evaluation pipeline including
tensor slicing to cutoff dates and batch embedding.
"""
import unittest
import torch
from hierarchical.evaluation.evaluate import slice_to_cutoff, embed_batch
from hierarchical.models.transaction import TransactionEncoder
from hierarchical.models.day import DayEncoder
from hierarchical.models.account import AccountEncoder


class TestSliceToCutoff(unittest.TestCase):
    """Test the slice_to_cutoff function for tensor truncation."""
    
    def test_slice_to_cutoff_basic(self):
        """Test basic slicing to a cutoff day."""
        item = {
            'account_id': 'test_123',
            'day_dates': [1, 3, 5, 7, 9],
            'days': [{'pos': f'day_{i}'} for i in range(5)]
        }
        
        result = slice_to_cutoff(item, cutoff_epoch_day=5)
        
        self.assertEqual(len(result['days']), 3)
        self.assertEqual(result['day_dates'], [1, 3, 5])
        self.assertEqual(result['n_days'], 3)
    
    def test_slice_to_cutoff_before_first_day(self):
        """Test slicing when cutoff is before first day."""
        item = {
            'account_id': 'test_123',
            'day_dates': [5, 7, 9],
            'days': [{'pos': f'day_{i}'} for i in range(3)]
        }
        
        result = slice_to_cutoff(item, cutoff_epoch_day=3)
        
        # Returns minimal data (first day)
        self.assertEqual(len(result['days']), 1)
        self.assertEqual(result['day_dates'], [5])
    
    def test_slice_to_cutoff_after_last_day(self):
        """Test slicing when cutoff is after last day."""
        item = {
            'account_id': 'test_123',
            'day_dates': [1, 3, 5],
            'days': [{'pos': f'day_{i}'} for i in range(3)]
        }
        
        result = slice_to_cutoff(item, cutoff_epoch_day=10)
        
        self.assertEqual(len(result['days']), 3)
        self.assertEqual(result['day_dates'], [1, 3, 5])
    
    def test_slice_to_cutoff_exact_match(self):
        """Test slicing when cutoff exactly matches last day."""
        item = {
            'account_id': 'test_123',
            'day_dates': [1, 3, 5, 7],
            'days': [{'pos': f'day_{i}'} for i in range(4)]
        }
        
        result = slice_to_cutoff(item, cutoff_epoch_day=7)
        
        self.assertEqual(len(result['days']), 4)
        self.assertEqual(result['day_dates'], [1, 3, 5, 7])


class TestEmbedBatch(unittest.TestCase):
    """Test the embed_batch function for model inference."""
    
    def setUp(self):
        """Create a simple model for testing."""
        self.device = 'cpu'
        
        txn_enc = TransactionEncoder(10, 10, 10, embedding_dim=32, use_balance=False)
        day_enc = DayEncoder(txn_enc, hidden_dim=32, num_layers=1)
        self.model = AccountEncoder(day_enc, hidden_dim=32, num_layers=1)
        self.model.eval()
    
    def _create_day(self, n_txns=4):
        """Helper to create a day dict."""
        return {
            'pos': {
                'cat_group': torch.randint(0, 10, (1, n_txns)),
                'cat_sub': torch.randint(0, 10, (1, n_txns)),
                'cat_cp': torch.randint(0, 10, (1, n_txns)),
                'amounts': torch.randn(1, n_txns),
                'dates': torch.zeros(1, n_txns, 4),
                'mask': torch.ones(1, n_txns).bool(),
                'has_data': torch.tensor([True]),
                'n_txns': n_txns
            },
            'neg': None,
            'meta': {'is_weekend': 0, 'month': 1},
            'day_offset': 0
        }
    
    def test_embed_batch_basic(self):
        """Test basic batch embedding."""
        import numpy as np
        
        batch_items = []
        for i in range(3):
            days = [self._create_day() for _ in range(5)]
            batch_items.append({
                'days': days,
                'day_dates': list(range(5)),
                'account_id': f'acc_{i}'
            })
        
        embeddings = embed_batch(self.model, batch_items, self.device)
        
        self.assertIsInstance(embeddings, np.ndarray)
        self.assertEqual(embeddings.shape, (3, 32))
        self.assertFalse(np.isnan(embeddings).any())
    
    def test_embed_batch_single_item(self):
        """Test embedding a single item."""
        import numpy as np
        
        days = [self._create_day(n_txns=2) for _ in range(3)]
        batch_items = [{
            'days': days,
            'day_dates': list(range(3)),
            'account_id': 'test_acc'
        }]
        
        embeddings = embed_batch(self.model, batch_items, self.device)
        
        self.assertEqual(embeddings.shape, (1, 32))
        self.assertFalse(np.isnan(embeddings).any())
    
    def test_embed_batch_varying_lengths(self):
        """Test batch with varying number of days."""
        import numpy as np
        
        # Account 1: 3 days
        batch_items = [{
            'days': [self._create_day(n_txns=2) for _ in range(3)],
            'day_dates': list(range(3)),
            'account_id': 'acc_1'
        }, {
            'days': [self._create_day(n_txns=3) for _ in range(7)],
            'day_dates': list(range(7)),
            'account_id': 'acc_2'
        }]
        
        embeddings = embed_batch(self.model, batch_items, self.device)
        
        self.assertEqual(embeddings.shape, (2, 32))
        self.assertFalse(np.isnan(embeddings).any())


if __name__ == '__main__':
    unittest.main()
