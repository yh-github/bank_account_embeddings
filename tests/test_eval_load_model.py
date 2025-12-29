"""Tests for evaluation script's model loading logic."""

import os
import tempfile
import unittest
import torch

# Import actual production code
from hierarchical.evaluation.evaluate import load_model
from hierarchical.models.transaction import TransactionEncoder
from hierarchical.models.day import DayEncoder
from hierarchical.models.account import AccountEncoder


class TestEvalLoadModel(unittest.TestCase):
    """Tests verify evaluating model loading using ACTUAL evaluate.py logic."""

    def _create_and_save_model(self, hidden_dim=256, num_layers=4) -> str:
        """Create a model, save it, and return the path."""
        txn_enc = TransactionEncoder(
            num_categories_group=36,
            num_categories_sub=152,
            num_counter_parties=10000,
            embedding_dim=hidden_dim,
            use_counter_party=True,
            use_balance=True,
        )
        day_enc = DayEncoder(
            txn_encoder=txn_enc,
            hidden_dim=hidden_dim,
            num_layers=2,  # DayEncoder uses fewer layers
            num_heads=4,
        )
        model = AccountEncoder(
            day_encoder=day_enc,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=4,
        )

        # Save (Simulate legacy checkpoint without 'config')
        f = tempfile.NamedTemporaryFile(suffix=".pth", delete=False)
        save_path = f.name
        f.close()
        torch.save(model.state_dict(), save_path)
        return save_path

    def _make_batch_data(self, B: int, D: int, T: int, vocab_sizes: dict) -> dict:
        """Create batch data dict."""

        def make_stream(B: int, D: int, T: int) -> dict:
            return {
                "cat_group": torch.randint(0, vocab_sizes["cat_grp"], (B, D, T)),
                "cat_sub": torch.randint(0, vocab_sizes["cat_sub"], (B, D, T)),
                "cat_cp": torch.randint(0, vocab_sizes["cat_cp"], (B, D, T)),
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

    def test_load_legacy_model(self):
        """Test loading a legacy (state_dict only) checkpoint via real load_model."""
        hidden_dim = 64
        checkpoint_path = self._create_and_save_model(
            hidden_dim=hidden_dim, num_layers=2
        )

        try:
            # THIS CALLS REAL CODE
            model = load_model(checkpoint_path, device="cpu", hidden_dim=hidden_dim)

            # Verify structure
            self.assertIsInstance(model, AccountEncoder)
            self.assertEqual(model.hidden_dim, hidden_dim)
            self.assertEqual(model.day_encoder.hidden_dim, hidden_dim)
            # Check inference
            vocab_sizes = {"cat_grp": 36, "cat_sub": 152, "cat_cp": 10000}
            batch = self._make_batch_data(B=2, D=5, T=10, vocab_sizes=vocab_sizes)
            with torch.no_grad():
                out = model(batch)
            self.assertEqual(out.shape, (2, hidden_dim))
            self.assertFalse(torch.isnan(out).any())

        finally:
            os.unlink(checkpoint_path)


if __name__ == "__main__":
    unittest.main()
