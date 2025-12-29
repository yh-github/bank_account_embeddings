"""
Tests for TransactionEncoder amount binning functionality.

Tests the amount discretization logic that bins transaction amounts
into quantiles for embedding.
"""
import unittest
import torch
from hierarchical.models.transaction import TransactionEncoder


class TestAmountBinning(unittest.TestCase):
    """Test amount binning/discretization in TransactionEncoder."""
    
    def test_amount_bins_creation(self):
        """Test that amount bins are created correctly."""
        # Create encoder with amount binning enabled
        encoder = TransactionEncoder(
            num_cat_group=10,
            num_cat_sub=10,
            num_counter_party=10,
            embedding_dim=32,
            use_amount_bins=True,
            num_amount_bins=5
        )
        
        # Verify bins were created
        self.assertTrue(hasattr(encoder, 'amount_bins'))
        self.assertIsNotNone(encoder.amount_bins)
        self.assertEqual(len(encoder.amount_bins), 5 + 1)  # num_bins + 1 for boundaries
        
        # Verify amount embedding layer was created
        self.assertTrue(hasattr(encoder, 'amount_emb'))
        self.assertEqual(encoder.amount_emb.num_embeddings, 5)
    
    def test_amount_bins_default_boundaries(self):
        """Test default amount bin boundaries."""
        encoder = TransactionEncoder(
            num_cat_group=10,
            num_cat_sub=10,
            num_counter_party=10,
            embedding_dim=32,
            use_amount_bins=True,
            num_amount_bins=4
        )
        
        # Default bins should be log-spaced
        bins = encoder.amount_bins
        self.assertEqual(len(bins), 5)  # 4 bins + 1 boundary
        
        # Bins should be increasing
        for i in range(len(bins) - 1):
            self.assertLess(bins[i], bins[i + 1])
    
    def test_forward_with_amount_bins(self):
        """Test forward pass uses amount binning."""
        encoder = TransactionEncoder(
            num_cat_group=10,
            num_cat_sub=10,
            num_counter_party=10,
            embedding_dim=32,
            use_amount_bins=True,
            num_amount_bins=3,
            use_balance=False
        )
        encoder.eval()
        
        # Create dummy input
        batch_size, seq_len = 2, 5
        cat_group = torch.randint(0, 10, (batch_size, seq_len))
        cat_sub = torch.randint(0, 10, (batch_size, seq_len))
        cat_cp = torch.randint(0, 10, (batch_size, seq_len))
        
        # Amounts spanning different ranges
        amounts = torch.tensor([
            [10.0, 50.0, 100.0, 500.0, 1000.0],
            [5.0, 25.0, 75.0, 250.0, 750.0]
        ])
        
        dates = torch.randn(batch_size, seq_len, 2)
        mask = torch.ones(batch_size, seq_len).bool()
        
        # Forward pass
        with torch.no_grad():
            output = encoder(cat_group, cat_sub, cat_cp, amounts, dates, None, mask)
        
        # Verify output shape
        self.assertEqual(output.shape, (batch_size, seq_len, 32))
        self.assertFalse(torch.isnan(output).any())
    
    def test_amount_binning_disabled(self):
        """Test that amount binning can be disabled."""
        encoder = TransactionEncoder(
            num_cat_group=10,
            num_cat_sub=10,
            num_counter_party=10,
            embedding_dim=32,
            use_amount_bins=False
        )
        
        # Verify bins not created
        self.assertFalse(hasattr(encoder, 'amount_bins'))
        self.assertFalse(hasattr(encoder, 'amount_emb'))
        
        # Should have amount projection instead
        self.assertTrue(hasattr(encoder, 'amount_proj'))
    
    def test_different_num_bins(self):
        """Test creating encoders with different numbers of bins."""
        for num_bins in [3, 5, 10]:
            encoder = TransactionEncoder(
                num_cat_group=10,
                num_cat_sub=10,
                num_counter_party=10,
                embedding_dim=32,
                use_amount_bins=True,
                num_amount_bins=num_bins
            )
            
            self.assertEqual(len(encoder.amount_bins), num_bins + 1)
            self.assertEqual(encoder.amount_emb.num_embeddings, num_bins)


if __name__ == '__main__':
    unittest.main()
