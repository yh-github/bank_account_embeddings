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
        # Create sample data
        item = {
            'account_id': 'test_123',
            'day_dates': [1, 3, 5, 7, 9],  # Epoch days
            'days': [{'pos': f'day_{i}'} for i in range(5)]
        }
        
        # Slice to day 5 (should include days 1, 3, 5)
        result = slice_to_cutoff(item, cutoff_day=5)
        
        self.assertEqual(len(result['days']), 3)
        self.assertEqual(result['day_dates'], [1, 3, 5])
        self.assertEqual(result['days'][2]['pos'], 'day_2')
        self.assertEqual(result['n_days'], 3)
    
    def test_slice_to_cutoff_before_first_day(self):
        """Test slicing when cutoff is before first day - should return empty."""
        item = {
            'day_dates': [5, 7, 9],
            'days': [{'pos': f'day_{i}'} for i in range(3)]
        }
        
        result = slice_to_cutoff(item, cutoff_day=3)
        
        self.assertEqual(len(result['days']), 0)
        self.assertEqual(result['day_dates'], [])
    
    def test_slice_to_cutoff_after_last_day(self):
        """Test slicing when cutoff is after last day - should return all."""
        item = {
            'account_id': 'test_123',
            'day_dates': [1, 3, 5],
            'days': [{'pos': f'day_{i}'} for i in range(3)]
        }
        
        result = slice_to_cutoff(item, cutoff_day=10)
        
        self.assertEqual(len(result['days']), 3)
        self.assertEqual(result['day_dates'], [1, 3, 5])
        self.assertEqual(result['n_days'], 3)
    
    def test_slice_to_cutoff_exact_match(self):
        """Test slicing when cutoff exactly matches last day."""
        item = {
            'account_id': 'test_123',
            'day_dates': [1, 3, 5, 7],
            'days': [{'pos': f'day_{i}'} for i in range(4)]
        }
        
        result = slice_to_cutoff(item, cutoff_day=7)
        
        self.assertEqual(len(result['days']), 4)
        self.assertEqual(result['day_dates'], [1, 3, 5, 7])
        self.assertEqual(result['n_days'], 4)


class TestEmbedBatch(unittest.TestCase):
    """Test the embed_batch function for model inference."""
    
    def setUp(self):
        """Create a simple model for testing."""
        self.device = 'cpu'
        
        # Create minimal model
        txn_enc = TransactionEncoder(10, 10, 10, embedding_dim=32, use_balance=False)
        day_enc = DayEncoder(txn_enc, hidden_dim=32, num_layers=1)
        self.model = AccountEncoder(day_enc, hidden_dim=32, num_layers=1)
        self.model.eval()
    
    def test_embed_batch_basic(self):
        """Test basic batch embedding."""
        # Create batch of items (list of dicts with 'days')
        batch_items = []
        for i in range(3):  # Batch size 3
            days = []
            for d in range(5):  # 5 days per account
                pos = {
                    'cat_group': torch.randint(0, 10, (1, 4)),
                    'cat_sub': torch.randint(0, 10, (1, 4)),
                    'cat_cp': torch.randint(0, 10, (1, 4)),
                    'amounts': torch.randn(1, 4),
                    'dates': torch.zeros(1, 4, 2),
                    'mask': torch.ones(1, 4).bool(),
                    'has_data': torch.tensor([True]),
                    'n_txns': 4
                }
                days.append({'pos': pos, 'neg': None})
            
            batch_items.append({'days': days})
        
        # Get embeddings
        embeddings = embed_batch(self.model, batch_items, self.device)
        
        # Verify output
        self.assertEqual(embeddings.shape, (3, 32))  # Batch size 3, hidden_dim 32
        self.assertFalse(torch.isnan(embeddings).any())
        self.assertTrue(embeddings.requires_grad == False)  # Should be in eval mode
    
    def test_embed_batch_single_item(self):
        """Test embedding a single item."""
        import numpy as np
        
        # Single item
        days = []
        for d in range(3):
            pos = {
                'cat_group': torch.randint(0, 10, (1, 2)),
                'cat_sub': torch.randint(0, 10, (1, 2)),
                'cat_cp': torch.randint(0, 10, (1, 2)),
                'amounts': torch.randn(1, 2),
                'dates': torch.zeros(1, 2, 2),
                'mask': torch.ones(1, 2).bool(),
                'has_data': torch.tensor([True]),
                'n_txns': 2
            }
            days.append({'pos': pos, 'neg': None})
        
        batch_items = [{'days': days}]
        
        embeddings = embed_batch(self.model, batch_items, self.device)
        
        self.assertEqual(embeddings.shape, (1, 32))
        self.assertFalse(np.isnan(embeddings).any())
    
    def test_embed_batch_varying_lengths(self):
        """Test batch with varying number of days per account."""
        import numpy as np
        
        batch_items = []
        
        # Account 1: 3 days
        days = []
        for d in range(3):
            pos = {
                'cat_group': torch.randint(0, 10, (1, 2)),
                'cat_sub': torch.randint(0, 10, (1, 2)),
                'cat_cp': torch.randint(0, 10, (1, 2)),
                'amounts': torch.randn(1, 2),
                'dates': torch.zeros(1, 2, 2),
                'mask': torch.ones(1, 2).bool(),
                'has_data': torch.tensor([True]),
                'n_txns': 2
            }
            days.append({'pos': pos, 'neg': None})
        batch_items.append({'days': days})
        
        # Account 2: 7 days
        days = []
        for d in range(7):
            pos = {
                'cat_group': torch.randint(0, 10, (1, 3)),
                'cat_sub': torch.randint(0, 10, (1, 3)),
                'cat_cp': torch.randint(0, 10, (1, 3)),
                'amounts': torch.randn(1, 3),
                'dates': torch.zeros(1, 3, 2),
                'mask': torch.ones(1, 3).bool(),
                'has_data': torch.tensor([True]),
                'n_txns': 3
            }
            days.append({'pos': pos, 'neg': None})
        batch_items.append({'days': days})
        
        # Get embeddings
        embeddings = embed_batch(self.model, batch_items, self.device)
        
        # Verify collation handled varying lengths
        self.assertEqual(embeddings.shape, (2, 32))
        self.assertFalse(np.isnan(embeddings).any())


if __name__ == '__main__':
    unittest.main()
