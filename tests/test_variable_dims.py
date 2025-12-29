"""
Test for variable dimension model loading.

Tests that the load_model function correctly handles models with
different dimensions for transaction, day, and account encoders.
"""

import tempfile
import torch
import unittest

from hierarchical.models.transaction import TransactionEncoder
from hierarchical.models.day import DayEncoder
from hierarchical.models.account import AccountEncoder
from hierarchical.evaluation.evaluate import load_model


class TestVariableDims(unittest.TestCase):
    """Test loading models with variable dimensions."""

    @staticmethod
    def load_model(ckpt_path, device="cpu"):
        """Wrapper for load_model function."""

        return load_model(
            ckpt_path, device, hidden_dim=128, model_type="hier", args=None
        )

    def test_load_variable_dims(self):
        from hierarchical.models.transaction import TransactionEncoder
        from hierarchical.models.day import DayEncoder
        from hierarchical.models.account import AccountEncoder

        """Test that load_model correctly detects dimensions."""
        txn_dim = 32
        day_dim = 64
        acc_dim = 128

        # Create Model (use_balance=False to simplify test)
        txn_enc = TransactionEncoder(
            10, 10, 10, embedding_dim=txn_dim, use_balance=False
        )
        day_enc = DayEncoder(txn_enc, hidden_dim=day_dim)
        model = AccountEncoder(day_enc, hidden_dim=acc_dim)

        # Save with embedded config (new format) so load_model can detect dimensions
        config = {
            "num_categories_group": 10,
            "num_categories_sub": 10,
            "num_counter_parties": 10,
            "txn_dim": txn_dim,
            "day_dim": day_dim,
            "account_dim": acc_dim,
            "num_layers": 2,  # AccountEncoder actual
            "day_num_layers": 1,  # DayEncoder actual
            "num_heads": 4,
            "use_balance": False,  # Must match model creation
            "use_counter_party": True,
        }

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            torch.save(
                {"model_state_dict": model.state_dict(), "config": config}, tmp.name
            )
            tmp_path = tmp.name

        try:
            # Load model
            loaded_model = self.load_model(tmp_path, device="cpu")

            # Verify model loaded correctly
            self.assertIsInstance(loaded_model, AccountEncoder)

            # Verify dimensions
            # The loaded model should have the same architecture
            self.assertEqual(loaded_model.hidden_dim, acc_dim)

            # Test inference (should not produce NaN)
            # Create dummy batch
            dummy_batch = {
                "pos": {
                    "cat_group": torch.randint(0, 10, (1, 5, 3)),
                    "cat_sub": torch.randint(0, 10, (1, 5, 3)),
                    "cat_cp": torch.randint(0, 10, (1, 5, 3)),
                    "amounts": torch.randn(1, 5, 3),
                    "dates": torch.zeros(1, 5, 3, 4),
                    "mask": torch.ones(1, 5, 3).bool(),
                    "has_data": torch.ones(1, 5).bool(),
                },
                "neg": {
                    "cat_group": torch.randint(0, 10, (1, 5, 2)),
                    "cat_sub": torch.randint(0, 10, (1, 5, 2)),
                    "cat_cp": torch.randint(0, 10, (1, 5, 2)),
                    "amounts": torch.randn(1, 5, 2),
                    "dates": torch.zeros(1, 5, 2, 4),
                    "mask": torch.ones(1, 5, 2).bool(),
                    "has_data": torch.ones(1, 5).bool(),
                },
                "meta": {
                    "day_mask": torch.ones(1, 5).bool(),
                    "day_dates": torch.arange(5).unsqueeze(0),
                    "day_month": torch.ones(1, 5).long(),
                    "day_weekend": torch.zeros(1, 5).long(),
                },
            }

            loaded_model.eval()
            with torch.no_grad():
                output = loaded_model(dummy_batch)

            self.assertFalse(torch.isnan(output).any())
            self.assertEqual(output.shape, (1, acc_dim))

        finally:
            import os

            os.unlink(tmp_path)

    def test_pyramid_architecture(self):
        """Test that pyramid architecture (growing dims) loads correctly."""
        txn_dim = 32
        day_dim = 64
        acc_dim = 128

        # Create pyramid model
        txn_enc = TransactionEncoder(
            10, 10, 10, embedding_dim=txn_dim, use_balance=False
        )
        day_enc = DayEncoder(txn_enc, hidden_dim=day_dim)
        model = AccountEncoder(day_enc, hidden_dim=acc_dim)

        config = {
            "num_categories_group": 10,
            "num_categories_sub": 10,
            "num_counter_parties": 10,
            "txn_dim": txn_dim,
            "day_dim": day_dim,
            "account_dim": acc_dim,
            "num_layers": 2,
            "day_num_layers": 1,
            "num_heads": 4,
            "use_balance": False,
            "use_counter_party": True,
        }

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            torch.save(
                {"model_state_dict": model.state_dict(), "config": config}, tmp.name
            )
            tmp_path = tmp.name

        try:
            loaded_model = self.load_model(tmp_path, device="cpu")

            # Verify pyramid structure preserved
            self.assertEqual(
                loaded_model.day_encoder.txn_encoder.embedding_dim, txn_dim
            )
            self.assertEqual(loaded_model.day_encoder.hidden_dim, day_dim)
            self.assertEqual(loaded_model.hidden_dim, acc_dim)

        finally:
            import os

            os.unlink(tmp_path)


if __name__ == "__main__":
    unittest.main()
