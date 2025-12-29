"""
Simple smoke tests for critical integration points.

These tests verify that key components work together correctly without
requiring full end-to-end pipeline execution.
"""
import os
import tempfile
import unittest

import torch
import pandas as pd

from hierarchical.models.account import AccountEncoder
from hierarchical.models.day import DayEncoder
from hierarchical.models.transaction import TransactionEncoder
from hierarchical.data.vocab import CategoricalVocabulary
from hierarchical.data.dataset import HierarchicalDataset, collate_hierarchical


class TestModelPersistence(unittest.TestCase):
    """Test that models can be saved, loaded, and produce consistent outputs."""
    
    def test_model_save_load_inference(self):
        """Smoke test: Save model, load it, verify inference works."""
        # Create a small model
        txn_enc = TransactionEncoder(
            num_categories_group=10,
            num_categories_sub=20,
            num_counter_parties=30,
            embedding_dim=32,
        )
        day_enc = DayEncoder(txn_enc, hidden_dim=32, num_heads=2, num_layers=1)
        model = AccountEncoder(day_enc, hidden_dim=32, num_layers=1, num_heads=2)
        
        # Create dummy input
        batch = self._create_dummy_batch()
        
        # Get original output
        model.eval()
        with torch.no_grad():
            output_before = model(batch)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "model.pth")
            
            # Save with config (matching real usage)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": {
                    "num_categories_group": 10,
                    "num_categories_sub": 20,
                    "num_counter_parties": 30,
                    "txn_dim": 32,
                    "day_dim": 32,
                    "account_dim": 32,
                    "num_layers": 1,
                    "day_num_layers": 1,
                    "num_heads": 2,
                    "use_balance": False,
                    "use_counter_party": True,
                },
            }
            torch.save(checkpoint, save_path)
            
            # Load model
            loaded_checkpoint = torch.load(save_path, weights_only=False)
            
            # Recreate model from config
            config = loaded_checkpoint["config"]
            txn_enc_new = TransactionEncoder(
                num_categories_group=config["num_categories_group"],
                num_categories_sub=config["num_categories_sub"],
                num_counter_parties=config["num_counter_parties"],
                embedding_dim=config["txn_dim"],
                use_balance=config["use_balance"],
                use_counter_party=config["use_counter_party"],
            )
            day_enc_new = DayEncoder(
                txn_enc_new,
                hidden_dim=config["day_dim"],
                num_heads=config["num_heads"],
                num_layers=config["day_num_layers"],
            )
            model_new = AccountEncoder(
                day_enc_new,
                hidden_dim=config["account_dim"],
                num_layers=config["num_layers"],
                num_heads=config["num_heads"],
            )
            
            # Load weights
            model_new.load_state_dict(loaded_checkpoint["model_state_dict"])
            model_new.eval()
            
            # Get output from loaded model
            with torch.no_grad():
                output_after = model_new(batch)
            
            # Verify outputs match
            self.assertEqual(output_before.shape, output_after.shape)
            torch.testing.assert_close(output_before, output_after)
            
            # Verify no NaNs
            self.assertFalse(torch.isnan(output_after).any())
    
    def _create_dummy_batch(self):
        """Create a minimal valid batch."""
        return {
            "pos": {
                "cat_group": torch.randint(0, 10, (2, 3, 5)),
                "cat_sub": torch.randint(0, 20, (2, 3, 5)),
                "cat_cp": torch.randint(0, 30, (2, 3, 5)),
                "amounts": torch.randn(2, 3, 5),
                "dates": torch.zeros(2, 3, 5, 4),
                "mask": torch.ones(2, 3, 5).bool(),
                "has_data": torch.ones(2, 3).bool(),
            },
            "neg": {
                "cat_group": torch.randint(0, 10, (2, 3, 4)),
                "cat_sub": torch.randint(0, 20, (2, 3, 4)),
                "cat_cp": torch.randint(0, 30, (2, 3, 4)),
                "amounts": torch.randn(2, 3, 4),
                "dates": torch.zeros(2, 3, 4, 4),
                "mask": torch.ones(2, 3, 4).bool(),
                "has_data": torch.ones(2, 3).bool(),
            },
            "meta": {
                "day_mask": torch.ones(2, 3).bool(),
                "day_dates": torch.arange(3).unsqueeze(0).expand(2, -1),
                "day_month": torch.ones(2, 3).long(),
                "day_weekend": torch.zeros(2, 3).long(),
            },
        }


class TestDataFlow(unittest.TestCase):
    """Test that data flows correctly through dataset and collation."""
    
    def test_vocab_encode_decode(self):
        """Smoke test: Vocabulary encoding/decoding roundtrip."""
        vocab = CategoricalVocabulary()
        
        # Build vocab from tokens
        tokens = ["cat_a", "cat_b", "cat_c", "cat_a", "cat_b"]
        vocab.fit(tokens)
        
        # Encode
        encoded = [vocab.encode(t) for t in ["cat_a", "cat_b", "cat_c", "unknown"]]
        
        # Verify structure
        self.assertEqual(len(vocab), 5)  # PAD, UNK, cat_a, cat_b, cat_c
        self.assertEqual(encoded[0], encoded[0])  # cat_a consistent
        self.assertEqual(encoded[3], 1)  # unknown -> UNK
        
        # Decode
        decoded = [vocab.decode(e) for e in encoded[:3]]
        self.assertEqual(decoded, ["cat_a", "cat_b", "cat_c"])
    
    def test_dataset_creates_valid_batches(self):
        """Smoke test: Dataset produces valid, collatable data."""
        # Create minimal transaction data
        df = pd.DataFrame({
            "accountId": ["acc_1"] * 30,
            "date": pd.date_range("2023-01-01", periods=30),
            "amount": [(-1)**i * (10 + i) for i in range(30)],
            "direction": ["Credit" if i % 2 == 0 else "Debit" for i in range(30)],
            "personeticsCategoryGroupId": [f"GRP{i%3}" for i in range(30)],
            "personeticsSubCategoryId": [f"SUB{i%5}" for i in range(30)],
            "deviceId": [f"dev{i%2}" for i in range(30)],
        })
        
        # Build vocabs
        cat_group_vocab = CategoricalVocabulary()
        cat_group_vocab.fit(df["personeticsCategoryGroupId"].unique())
        
        cat_sub_vocab = CategoricalVocabulary()
        cat_sub_vocab.fit(df["personeticsSubCategoryId"].unique())
        
        cp_vocab = CategoricalVocabulary()
        cp_vocab.fit(df["deviceId"].unique())
        
        # Create dataset
        dataset = HierarchicalDataset(
            df,
            cat_group_vocab,
            cat_sub_vocab,
            cp_vocab,
            account_ids=["acc_1"],
            max_days=30,
            min_days=10,
            max_txns_per_day=10,
        )
        
        # Get item
        self.assertEqual(len(dataset), 1)
        item = dataset[0]
        
        # Verify structure
        self.assertIn("account_id", item)
        self.assertIn("days", item)
        self.assertIsInstance(item["days"], list)
        self.assertGreater(len(item["days"]), 0)
        
        # Test collation
        batch = collate_hierarchical([item])
        
        # Verify batch structure
        self.assertIn("pos", batch)
        self.assertIn("neg", batch)
        self.assertIn("meta", batch)
        
        # Verify tensors
        self.assertIsInstance(batch["pos"]["cat_group"], torch.Tensor)
        self.assertIsInstance(batch["pos"]["amounts"], torch.Tensor)
        self.assertEqual(batch["pos"]["cat_group"].dim(), 3)  # [batch, days, txns]


class TestModelInference(unittest.TestCase):
    """Test that models can perform inference on real-ish data."""
    
    def test_end_to_end_inference(self):
        """Smoke test: Create model, feed real-ish data, get output."""
        # Create vocabs
        cat_group_vocab = CategoricalVocabulary()
        cat_group_vocab.fit(["GRP1", "GRP2", "GRP3"])
        
        cat_sub_vocab = CategoricalVocabulary()
        cat_sub_vocab.fit(["SUB1", "SUB2", "SUB3", "SUB4", "SUB5"])
        
        cp_vocab = CategoricalVocabulary()
        cp_vocab.fit(["dev1", "dev2"])
        
        # Create data
        df = pd.DataFrame({
            "accountId": ["acc_1"] * 50,
            "date": pd.date_range("2023-01-01", periods=50),
            "amount": [(-1)**i * (10 + i*2) for i in range(50)],
            "direction": ["Credit" if i % 2 == 0 else "Debit" for i in range(50)],
            "personeticsCategoryGroupId": [f"GRP{(i%3)+1}" for i in range(50)],
            "personeticsSubCategoryId": [f"SUB{(i%5)+1}" for i in range(50)],
            "deviceId": [f"dev{(i%2)+1}" for i in range(50)],
        })
        
        # Create dataset
        dataset = HierarchicalDataset(
            df, cat_group_vocab, cat_sub_vocab, cp_vocab,
            account_ids=["acc_1"], max_days=50, min_days=10, max_txns_per_day=5
        )
        
        # Create model
        txn_enc = TransactionEncoder(
            num_categories_group=len(cat_group_vocab),
            num_categories_sub=len(cat_sub_vocab),
            num_counter_parties=len(cp_vocab),
            embedding_dim=32,
        )
        day_enc = DayEncoder(txn_enc, hidden_dim=32, num_heads=2, num_layers=1)
        model = AccountEncoder(day_enc, hidden_dim=32, num_layers=1, num_heads=2)
        model.eval()
        
        # Get batch
        batch = collate_hierarchical([dataset[0]])
        
        # Inference
        with torch.no_grad():
            output = model(batch)
        
        # Verify output
        self.assertEqual(output.shape, (1, 32))
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
        # Verify embedding is normalized (roughly)
        norm = torch.norm(output, dim=-1)
        self.assertGreater(norm.item(), 0.1)  # Not degenerate


if __name__ == "__main__":
    unittest.main()
