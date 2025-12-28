import sys
from unittest.mock import MagicMock, patch
import unittest
import torch
import torch.nn as nn
import tempfile
import os

class TestVariableDims(unittest.TestCase):

    def setUp(self):
        # Mock dependencies using patch.dict on sys.modules
        self.modules_patcher = patch.dict(sys.modules, {
            "xgboost": MagicMock(),
            "sklearn": MagicMock(),
            "sklearn.linear_model": MagicMock(),
            "sklearn.metrics": MagicMock(),
            "sklearn.model_selection": MagicMock(),
            "cleanlab": MagicMock(),
            "cleanlab.classification": MagicMock(),
            "pulearn": MagicMock()
        })
        self.modules_patcher.start()
        
        # Import inside setup to use mocks
        from hierarchical.evaluation.evaluate import load_model
        # Reloading might be needed if it was already imported, but for now this is fine 
        # as unittest runs fresh or we force reload if needed.
        # Ideally we don't import at top level at all.
        self.load_model = load_model

    def tearDown(self):
        self.modules_patcher.stop()
        
    def test_pyramid_architecture(self):
        """Test instantiation and forward pass with increasing dimensions (32->64->128)."""
        from hierarchical.models.transaction import TransactionEncoder
        from hierarchical.models.day import DayEncoder
        from hierarchical.models.account import AccountEncoder
        txn_dim = 32
        day_dim = 64
        acc_dim = 128
        
        # 1. Transaction Encoder
        txn_enc = TransactionEncoder(
            num_categories_group=10,
            num_categories_sub=10,
            num_counter_parties=10,
            embedding_dim=txn_dim
        )
        self.assertEqual(txn_enc.embedding_dim, txn_dim)
        
        # 2. Day Encoder (Input 32 -> Hidden 64)
        day_enc = DayEncoder(
            txn_enc,
            hidden_dim=day_dim,
            num_layers=1
        )
        self.assertIsNotNone(day_enc.input_proj)
        self.assertEqual(day_enc.hidden_dim, day_dim)
        
        # 3. Account Encoder (Input 64 -> Hidden 128)
        model = AccountEncoder(
            day_enc,
            hidden_dim=acc_dim,
            num_layers=1
        )
        self.assertIsNotNone(model.input_proj)
        self.assertEqual(model.hidden_dim, acc_dim)
        
        # 4. Dummy Forward Pass
        # Batch=2, Days=5, Txns=10
        # Pos Only
        B = 2
        D = 5 
        T = 10
        
        # Create dummy batch (simplified)
        # AccountEncoder expects [B, D, T] for input tensors
        
        pos = {
            'cat_group': torch.randint(0, 10, (B, D, T)),
            'cat_sub': torch.randint(0, 10, (B, D, T)),
            'cat_cp': torch.randint(0, 10, (B, D, T)),
            'amounts': torch.randn(B, D, T),
            'dates': torch.zeros(B, D, T, 2), # simplified
            'mask': torch.ones(B, D, T),
            'has_data': torch.ones(B, D) # [B, D]
        }
        
        neg = pos.copy() # simplified
        
        meta = {
            'day_mask': torch.ones(B, D).bool(),
            'day_month': torch.zeros(B, D).long(),
            'day_weekend': torch.zeros(B, D).long()
        }
        
        batch = {'pos': pos, 'neg': neg, 'meta': meta}
        
        output = model(batch)
        self.assertEqual(output.shape, (B, acc_dim))

    def test_load_variable_dims(self):
        from hierarchical.models.transaction import TransactionEncoder
        from hierarchical.models.day import DayEncoder
        from hierarchical.models.account import AccountEncoder
        """Test that load_model correctly detects dimensions."""
        txn_dim = 32
        day_dim = 64
        acc_dim = 128
        
        # Create Model
        txn_enc = TransactionEncoder(10, 10, 10, embedding_dim=txn_dim)
        day_enc = DayEncoder(txn_enc, hidden_dim=day_dim)
        model = AccountEncoder(day_enc, hidden_dim=acc_dim)
        
        # Save State Dict
        sd = model.state_dict()
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            torch.save(sd, tmp.name)
            tmp_path = tmp.name
            
        try:
            # Load with evaluate.load_model
            loaded_model = self.load_model(tmp_path, device='cpu')
            
            # Verify Dimensions
            self.assertEqual(loaded_model.hidden_dim, acc_dim)
            self.assertEqual(loaded_model.day_encoder.hidden_dim, day_dim)
            self.assertEqual(loaded_model.day_encoder.txn_encoder.embedding_dim, txn_dim)
            
            # Verify Projections exist
            self.assertIsNotNone(loaded_model.input_proj)
            self.assertIsNotNone(loaded_model.day_encoder.input_proj)
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

if __name__ == '__main__':
    unittest.main()
