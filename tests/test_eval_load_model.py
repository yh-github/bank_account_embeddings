"""Tests for evaluation script's model loading logic.

This tests the load_model function from unified_eval_optimized.py
to ensure it correctly reconstructs models from checkpoints.
"""

import os
import tempfile
import unittest

import torch
import numpy as np

from hierarchical.models.transaction import TransactionEncoder
from hierarchical.models.day import DayEncoder
from hierarchical.models.account import AccountEncoder


class TestEvalLoadModel(unittest.TestCase):
    """Tests that mimic unified_eval_optimized.py's load_model function."""

    def _create_and_save_model(self, hidden_dim=256, num_layers=4) -> str:
        """Create a model, save it, and return the path."""
        txn_enc = TransactionEncoder(
            num_categories_group=36,
            num_categories_sub=152,
            num_counter_parties=10000,
            embedding_dim=hidden_dim,
            use_counter_party=True,
            use_balance=True
        )
        day_enc = DayEncoder(
            txn_encoder=txn_enc,
            hidden_dim=hidden_dim,
            num_layers=2,  # DayEncoder uses fewer layers
            num_heads=4
        )
        model = AccountEncoder(
            day_encoder=day_enc,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=4
        )
        
        # Save
        f = tempfile.NamedTemporaryFile(suffix='.pth', delete=False)
        save_path = f.name
        f.close()
        torch.save(model.state_dict(), save_path)
        return save_path

    def _load_model_like_eval(self, checkpoint_path: str, hidden_dim: int = 256) -> AccountEncoder:
        """
        Mimic the load_model function from unified_eval_optimized.py
        This is where the bug likely is!
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint if isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint else checkpoint.get('model_state_dict', checkpoint)
        
        # Infer vocab sizes from checkpoint (like eval script does)
        prefix = 'day_encoder.txn_encoder.'
        cat_grp_size = state_dict.get(f'{prefix}cat_group_emb.weight', torch.zeros(100, 1)).shape[0]
        cat_sub_size = state_dict.get(f'{prefix}cat_sub_emb.weight', torch.zeros(100, 1)).shape[0]
        cat_cp_size = state_dict.get(f'{prefix}counter_party_emb.weight', torch.zeros(100, 1)).shape[0]
        
        use_balance = f'{prefix}balance_proj.weight' in state_dict
        use_cp = f'{prefix}counter_party_emb.weight' in state_dict
        
        # Detect number of layers from checkpoint keys
        max_layer_idx = 0
        for key in state_dict.keys():
            if 'transformer.layers.' in key:
                try:
                    parts = key.split('transformer.layers.')[1].split('.')
                    layer_idx = int(parts[0])
                    if layer_idx > max_layer_idx:
                        max_layer_idx = layer_idx
                except (ValueError, IndexError):
                    pass
        
        detected_num_layers = max_layer_idx + 1
        
        # Build model like eval script does
        txn_encoder = TransactionEncoder(
            num_categories_group=cat_grp_size,
            num_categories_sub=cat_sub_size,
            num_counter_parties=cat_cp_size,
            embedding_dim=hidden_dim,
            use_balance=use_balance,
            use_counter_party=use_cp,
        )
        
        # THE BUG IS LIKELY HERE: eval script uses hardcoded num_layers=2 for DayEncoder
        day_encoder = DayEncoder(
            txn_encoder=txn_encoder,
            hidden_dim=hidden_dim,
            num_layers=2,  # Hardcoded!
            num_heads=4
        )
        
        # And detected_num_layers for AccountEncoder
        model = AccountEncoder(
            day_encoder=day_encoder,
            hidden_dim=hidden_dim,
            num_layers=detected_num_layers,
            num_heads=4
        )
        
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _make_batch_data(self, B: int, D: int, T: int, vocab_sizes: dict) -> dict:
        """Create batch data dict."""
        def make_stream(B: int, D: int, T: int) -> dict:
            return {
                "cat_group": torch.randint(0, vocab_sizes['cat_grp'], (B, D, T)),
                "cat_sub": torch.randint(0, vocab_sizes['cat_sub'], (B, D, T)),
                "cat_cp": torch.randint(0, vocab_sizes['cat_cp'], (B, D, T)),
                "amounts": torch.randn(B, D, T),
                "dates": torch.zeros(B, D, T, 4),
                "balance": torch.randn(B, D, T, 7),
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

    def test_eval_load_model_no_nan(self):
        """Model loaded via eval's method should not produce NaN."""
        checkpoint_path = self._create_and_save_model(hidden_dim=256, num_layers=4)
        
        try:
            model = self._load_model_like_eval(checkpoint_path, hidden_dim=256)
            
            vocab_sizes = {'cat_grp': 36, 'cat_sub': 152, 'cat_cp': 10000}
            batch = self._make_batch_data(B=4, D=10, T=5, vocab_sizes=vocab_sizes)
            
            with torch.no_grad():
                out = model(batch)
            
            self.assertFalse(
                torch.isnan(out).any(),
                f"NaN detected! This reproduces the eval load bug."
            )
            
        finally:
            os.unlink(checkpoint_path)

    def test_layer_key_mismatch_detection(self):
        """Test that we can detect DayEncoder vs AccountEncoder layer keys."""
        checkpoint_path = self._create_and_save_model(hidden_dim=64, num_layers=4)
        
        try:
            state_dict = torch.load(checkpoint_path, weights_only=True)
            
            # Check what keys exist
            day_layer_keys = [k for k in state_dict.keys() if 'day_encoder.layers.' in k]
            account_layer_keys = [k for k in state_dict.keys() if 'transformer.layers.' in k and 'day_encoder' not in k]
            
            print(f"\nDayEncoder layer keys: {len(day_layer_keys)}")
            print(f"AccountEncoder layer keys: {len(account_layer_keys)}")
            
            # Show sample keys
            for k in list(state_dict.keys())[:10]:
                print(f"  {k}")
            
            # The issue: DayEncoder uses self.layers (ModuleList), not self.transformer
            # So keys are like: day_encoder.layers.0.self_attn...
            # But AccountEncoder uses self.transformer (TransformerEncoder)
            # So keys are like: transformer.layers.0.self_attn...
            
        finally:
            os.unlink(checkpoint_path)


if __name__ == "__main__":
    unittest.main()
