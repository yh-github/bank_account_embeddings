"""End-to-end system tests for the training pipeline."""

import unittest
import pandas as pd
import tempfile
import os
import json
import torch
from types import SimpleNamespace
from hierarchical.training.pretrain import train


class TestE2ETraining(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.test_dir = self.tmp_dir_obj.name

        # Create dummy transaction data
        self.txn_path = os.path.join(self.test_dir, "txn.csv")
        self.acc_path = os.path.join(self.test_dir, "acc.csv")

        # Generate sufficient history (>5 days) for 3 accounts
        data = []
        for acc in ["A1", "A2", "A3"]:
            for day in range(1, 8):  # 7 days
                data.append(
                    {
                        "accountId": acc,
                        "id": f"{acc}_t{day}",
                        "date": f"2023-01-0{day}",
                        "amount": 100.0 if day % 2 == 0 else -50.0,
                        "direction": "C" if day % 2 == 0 else "D",
                        "categoryGroupId": "C1",
                        "categoryId": "S1",
                        "counterParty": "CP1",
                    }
                )
        pd.DataFrame(data).to_csv(self.txn_path, index=False)

        # Accounts
        pd.DataFrame(
            {
                "accountId": ["A1", "A2", "A3"],
                "availableBalance": [1000, 2000, 500],
                "balanceDateTime": ["2023-01-05", "2023-01-05", "2023-01-05"],
            }
        ).to_csv(self.acc_path, index=False)

        # Config
        self.config_path = os.path.join(self.test_dir, "config.json")
        config = [
            {"name": "TestBank", "txn_file": self.txn_path, "acc_file": self.acc_path}
        ]
        with open(self.config_path, "w") as f:
            json.dump(config, f)

        self.output_dir = os.path.join(self.test_dir, "output")

    def tearDown(self):
        self.tmp_dir_obj.cleanup()

    def test_training_loop(self):
        # Set seeds for reproducibility
        torch.manual_seed(42)
        import random

        random.seed(42)
        import numpy as np

        np.random.seed(42)

        # Mock Arguments
        args = SimpleNamespace(
            output_dir=self.output_dir,
            bank_config=self.config_path,
            data_file=None,
            account_file=None,
            preprocessed_file=None,
            cutoff_date=None,
            augment=False,
            epochs=2,
            batch_size=2,
            hidden_dim=16,
            num_heads=2,
            num_layers=1,
            dropout=0.0,
            txn_dim=16,
            day_dim=16,
            account_dim=16,
            use_amount_binning=False,
            num_amount_bins=10,
            lr=1e-4,  # Lower LR for stability
            num_workers=0,  # Important for tests
            use_balance=True,
            use_counter_party=True,
            no_counter_party=False,
            use_amp=False,
            max_days=10,
            max_txns_per_day=5,
            vocab_dir=None,
            gradient_checkpointing=False,
        )

        # Run training
        # This should create vocabularies and a model checkpoint
        train(args)

        # Verification
        if not os.path.exists(os.path.join(self.output_dir, "model_best.pth")):
            print(f"Contents of {self.output_dir}: {os.listdir(self.output_dir)}")
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "model_best.pth")),
            "Model checkpoint not found",
        )
        self.assertTrue(
            os.path.exists(
                os.path.join(self.output_dir, "vocabularies", "cat_group_vocab.pkl")
            ),
            "Vocab not found",
        )
        self.assertTrue(
            os.path.exists(os.path.join(self.output_dir, "train_ids.npy")),
            "Train split not saved",
        )


if __name__ == "__main__":
    unittest.main()
