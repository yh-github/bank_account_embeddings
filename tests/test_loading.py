"""Tests for model loading in evaluation module."""

import unittest
import torch
import os
import tempfile
import shutil

from hierarchical.models.transaction import TransactionEncoder
from hierarchical.models.day import DayEncoder
from hierarchical.models.account import AccountEncoder


class TestModelLoading(unittest.TestCase):
    """Tests for model loading with correct architecture."""
    
    def setUp(self):
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = self.tmp_dir_obj.name
        self.ckpt_path = os.path.join(self.tmp_dir, 'checkpoint.pth')
        
    def tearDown(self):
        self.tmp_dir_obj.cleanup()

    def _create_model(self, hidden_dim=32, day_layers=2, acc_layers=4):
        """Create a full AccountEncoder model."""
        txn_enc = TransactionEncoder(
            num_categories_group=10,
            num_categories_sub=10,
            num_counter_parties=10,
            embedding_dim=hidden_dim,
            use_counter_party=True,
            use_balance=True
        )
        day_enc = DayEncoder(
            txn_encoder=txn_enc,
            hidden_dim=hidden_dim,
            num_layers=day_layers,
            num_heads=4
        )
        model = AccountEncoder(
            day_encoder=day_enc,
            hidden_dim=hidden_dim,
            num_layers=acc_layers,
            num_heads=4
        )
        return model

    def test_day_encoder_uses_layers_module_list(self):
        """DayEncoder should use self.layers (ModuleList) not self.transformer."""
        model = self._create_model(day_layers=2)
        day_enc = model.day_encoder
        
        # Verify it uses self.layers
        self.assertTrue(hasattr(day_enc, 'layers'), "DayEncoder should have 'layers' attribute")
        self.assertEqual(len(day_enc.layers), 2, "DayEncoder should have 2 layers")
        
        # Verify it doesn't have self.transformer
        self.assertFalse(hasattr(day_enc, 'transformer'), "DayEncoder should NOT have 'transformer' attribute")

    def test_account_encoder_uses_transformer(self):
        """AccountEncoder should use self.transformer (TransformerEncoder)."""
        model = self._create_model(acc_layers=4)
        
        # Verify it uses self.transformer
        self.assertTrue(hasattr(model, 'transformer'), "AccountEncoder should have 'transformer' attribute")
        
    def test_save_load_preserves_architecture(self):
        """Save/load should preserve model architecture."""
        model = self._create_model(hidden_dim=32, day_layers=2, acc_layers=3)
        
        # Save with new checkpoint format
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': {
                'num_categories_group': 10,
                'num_categories_sub': 10,
                'num_counter_parties': 10,
                'txn_dim': 32,
                'day_dim': 32,
                'account_dim': 32,
                'hidden_dim': 32,
                'num_layers': 3,
                'day_num_layers': 2,
                'num_heads': 4,
                'use_balance': True,
                'use_counter_party': True,
            }
        }
        torch.save(checkpoint, self.ckpt_path)
        
        # Load
        loaded = torch.load(self.ckpt_path, weights_only=False)
        config = loaded['config']
        
        # Verify config
        self.assertEqual(config['num_layers'], 3)
        self.assertEqual(config['day_num_layers'], 2)
        
        # Create model from config and load state
        model2 = self._create_model(
            hidden_dim=config['hidden_dim'],
            day_layers=config['day_num_layers'],
            acc_layers=config['num_layers']
        )
        model2.load_state_dict(loaded['model_state_dict'])
        
        # Verify architecture
        self.assertEqual(len(model2.day_encoder.layers), 2)

    def test_legacy_checkpoint_detection(self):
        """Legacy checkpoints (just state_dict) should be handled."""
        model = self._create_model()
        
        # Save just state_dict (legacy format)
        torch.save(model.state_dict(), self.ckpt_path)
        
        # Load
        loaded = torch.load(self.ckpt_path, weights_only=False)
        
        # Should be able to detect it's legacy format
        has_config = isinstance(loaded, dict) and 'config' in loaded
        self.assertFalse(has_config, "Legacy checkpoint should not have 'config'")

    def test_state_dict_key_structure(self):
        """Verify state_dict keys match expected architecture."""
        model = self._create_model(day_layers=2, acc_layers=3)
        sd = model.state_dict()
        
        # Check for DayEncoder.layers keys (ModuleList)
        day_layer_keys = [k for k in sd.keys() if 'day_encoder.layers.' in k]
        self.assertGreater(len(day_layer_keys), 0, "Should have day_encoder.layers keys")
        
        # Check for AccountEncoder.transformer keys (TransformerEncoder)
        acc_transformer_keys = [k for k in sd.keys() if k.startswith('transformer.layers.')]
        self.assertGreater(len(acc_transformer_keys), 0, "Should have transformer.layers keys")
        
        # Should NOT have day_encoder.transformer keys
        old_day_keys = [k for k in sd.keys() if 'day_encoder.transformer.' in k]
        self.assertEqual(len(old_day_keys), 0, "Should NOT have day_encoder.transformer keys (old architecture)")


if __name__ == '__main__':
    unittest.main()
