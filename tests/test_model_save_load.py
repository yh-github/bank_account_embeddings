"""Tests for model save/load/inference cycle.

This test suite verifies that:
1. Models can be saved and loaded correctly
2. Loaded models produce identical outputs to original models  
3. No NaN values appear in model outputs after loading
"""

import os
import tempfile
import unittest

import torch
import numpy as np

from hierarchical.models.transaction import TransactionEncoder
from hierarchical.models.day import DayEncoder
from hierarchical.models.account import AccountEncoder


class TestModelSaveLoad(unittest.TestCase):
    """Tests for model save/load cycle."""

    def _create_model(self, hidden_dim=64, num_layers=2) -> AccountEncoder:
        """Create a full AccountEncoder model."""
        txn_enc = TransactionEncoder(
            num_categories_group=20,
            num_categories_sub=50,
            num_counter_parties=100,
            embedding_dim=hidden_dim,
            use_counter_party=True,
            use_balance=True
        )
        day_enc = DayEncoder(
            txn_encoder=txn_enc,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=4
        )
        model = AccountEncoder(
            day_encoder=day_enc,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=4
        )
        return model

    def _make_batch_data(self, B: int, D: int, T: int) -> dict:
        """Create batch data dict matching collate output."""
        def make_stream(B: int, D: int, T: int) -> dict:
            return {
                "cat_group": torch.randint(0, 20, (B, D, T)),
                "cat_sub": torch.randint(0, 50, (B, D, T)),
                "cat_cp": torch.randint(0, 100, (B, D, T)),
                "amounts": torch.randn(B, D, T),
                "dates": torch.zeros(B, D, T, 4),
                "balance": torch.randn(B, D, T, 7),  # Include balance features
                "mask": torch.ones(B, D, T, dtype=torch.bool),
                "has_data": torch.ones(B, D, dtype=torch.bool),
            }

        return {
            "pos": make_stream(B, D, T),
            "neg": make_stream(B, D, T),
            "meta": {
                "day_mask": torch.ones(B, D, dtype=torch.bool),
                "day_month": torch.randint(0, 12, (B, D)),
                "day_weekend": torch.randint(0, 2, (B, D)),
            },
        }

    def test_save_load_state_dict(self):
        """Model should produce identical output after save/load cycle."""
        model = self._create_model()
        model.eval()
        
        # Create test input
        batch = self._make_batch_data(B=2, D=5, T=3)
        
        # Run inference before save
        with torch.no_grad():
            out_before = model(batch)
        
        # Save
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            save_path = f.name
            torch.save(model.state_dict(), save_path)
        
        try:
            # Create new model with same architecture
            model2 = self._create_model()
            model2.load_state_dict(torch.load(save_path, weights_only=True))
            model2.eval()
            
            # Run inference after load
            with torch.no_grad():
                out_after = model2(batch)
            
            # Verify outputs match
            self.assertTrue(torch.allclose(out_before, out_after, atol=1e-6))
            
        finally:
            os.unlink(save_path)

    def test_loaded_model_no_nan(self):
        """Loaded model should not produce NaN values."""
        model = self._create_model()
        model.eval()
        
        # Save
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            save_path = f.name
            torch.save(model.state_dict(), save_path)
        
        try:
            # Load into new model
            model2 = self._create_model()
            model2.load_state_dict(torch.load(save_path, weights_only=True))
            model2.eval()
            
            # Run inference on multiple batches
            for _ in range(5):
                batch = self._make_batch_data(B=4, D=10, T=5)
                with torch.no_grad():
                    out = model2(batch)
                
                self.assertFalse(
                    torch.isnan(out).any(),
                    f"NaN detected in model output! Shape: {out.shape}"
                )
                
        finally:
            os.unlink(save_path)

    def test_inference_no_nan(self):
        """Fresh model inference should not produce NaN values."""
        model = self._create_model()
        model.eval()
        
        batch = self._make_batch_data(B=4, D=10, T=5)
        
        with torch.no_grad():
            out = model(batch)
        
        self.assertFalse(
            torch.isnan(out).any(),
            f"NaN detected in fresh model output! Shape: {out.shape}"
        )

    def test_variable_dimensions_save_load(self):
        """Save/load should work for different hidden dimensions."""
        for hidden_dim in [32, 64, 128, 256]:
            for num_layers in [1, 2, 4]:
                with self.subTest(hidden_dim=hidden_dim, num_layers=num_layers):
                    model = self._create_model(hidden_dim=hidden_dim, num_layers=num_layers)
                    model.eval()
                    
                    batch = self._make_batch_data(B=2, D=5, T=3)
                    
                    with torch.no_grad():
                        out_before = model(batch)
                    
                    # Save and load
                    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
                        save_path = f.name
                        torch.save(model.state_dict(), save_path)
                    
                    try:
                        model2 = self._create_model(hidden_dim=hidden_dim, num_layers=num_layers)
                        model2.load_state_dict(torch.load(save_path, weights_only=True))
                        model2.eval()
                        
                        with torch.no_grad():
                            out_after = model2(batch)
                        
                        self.assertTrue(torch.allclose(out_before, out_after, atol=1e-6))
                        self.assertFalse(torch.isnan(out_after).any())
                        
                    finally:
                        os.unlink(save_path)


class TestModelCheckpointWithHyperparameters(unittest.TestCase):
    """Tests for saving/loading model with hyperparameters."""

    def _create_model_with_config(self, config: dict) -> AccountEncoder:
        """Create model from config dict."""
        txn_enc = TransactionEncoder(
            num_categories_group=config['num_categories_group'],
            num_categories_sub=config['num_categories_sub'],
            num_counter_parties=config['num_counter_parties'],
            embedding_dim=config['hidden_dim'],
            use_counter_party=config.get('use_counter_party', True),
            use_balance=config.get('use_balance', True)
        )
        day_enc = DayEncoder(
            txn_encoder=txn_enc,
            hidden_dim=config['hidden_dim'],
            num_layers=config['day_num_layers'],
            num_heads=config['num_heads']
        )
        model = AccountEncoder(
            day_encoder=day_enc,
            hidden_dim=config['hidden_dim'],
            num_layers=config['account_num_layers'],
            num_heads=config['num_heads']
        )
        return model

    def test_save_load_with_config(self):
        """Model should be loadable using saved config."""
        config = {
            'num_categories_group': 36,
            'num_categories_sub': 152,
            'num_counter_parties': 10000,
            'hidden_dim': 256,
            'day_num_layers': 2,
            'account_num_layers': 4,
            'num_heads': 8,
            'use_counter_party': True,
            'use_balance': True
        }
        
        model = self._create_model_with_config(config)
        model.eval()
        
        # Create checkpoint with config
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': config
        }
        
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            save_path = f.name
            torch.save(checkpoint, save_path)
        
        try:
            # Load checkpoint
            loaded = torch.load(save_path, weights_only=False)
            loaded_config = loaded['config']
            
            # Recreate model from config
            model2 = self._create_model_with_config(loaded_config)
            model2.load_state_dict(loaded['model_state_dict'])
            model2.eval()
            
            # Verify no NaN
            def make_stream(B, D, T, config):
                return {
                    "cat_group": torch.randint(0, config['num_categories_group'], (B, D, T)),
                    "cat_sub": torch.randint(0, config['num_categories_sub'], (B, D, T)),
                    "cat_cp": torch.randint(0, config['num_counter_parties'], (B, D, T)),
                    "amounts": torch.randn(B, D, T),
                    "dates": torch.zeros(B, D, T, 4),
                    "balance": torch.randn(B, D, T, 7),
                    "mask": torch.ones(B, D, T, dtype=torch.bool),
                    "has_data": torch.ones(B, D, dtype=torch.bool),
                }
            
            batch = {
                "pos": make_stream(2, 5, 3, config),
                "neg": make_stream(2, 5, 3, config),
                "meta": {
                    "day_mask": torch.ones(2, 5, dtype=torch.bool),
                    "day_month": torch.randint(0, 12, (2, 5)),
                    "day_weekend": torch.randint(0, 2, (2, 5)),
                },
            }
            
            with torch.no_grad():
                out = model2(batch)
            
            self.assertFalse(torch.isnan(out).any())
            self.assertEqual(out.shape, (2, config['hidden_dim']))
            
        finally:
            os.unlink(save_path)


if __name__ == "__main__":
    unittest.main()
