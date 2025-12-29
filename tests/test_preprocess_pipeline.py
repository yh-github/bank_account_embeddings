import unittest
import os
import sys
import tempfile
import pandas as pd
import torch
from unittest.mock import patch

from hierarchical.data import preprocess_tensors


class TestPreprocessPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.test_dir = self.tmp_dir_obj.name

        self.txn_file = os.path.join(self.test_dir, "transactions.csv")
        self.acc_file = os.path.join(self.test_dir, "accounts.csv")
        self.vocab_dir = os.path.join(self.test_dir, "vocabularies")
        self.output_dir = os.path.join(self.test_dir, "output")
        os.makedirs(self.vocab_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        # Create Dummy Data
        df_txn = pd.DataFrame(
            {
                "id": ["tx_1", "tx_2"],
                "accountId": ["acc_1", "acc_1"],
                "date": ["2022-01-01", "2022-01-02"],
                "amount": [-10.0, 100.0],
                "direction": ["Debit", "Credit"],
                "personeticsCategoryGroupId": ["GRP1", "GRP1"],
                "personeticsSubCategoryId": ["SUB1", "SUB1"],
                "deviceId": ["dev_1", "dev_1"],
                "user_name": ["u1", "u1"],
            }
        )
        df_txn.to_csv(self.txn_file, index=False)

        df_acc = pd.DataFrame(
            {
                "id": ["acc_1"],
                "type": ["Checking"],
                "availableBalance": [1000.0],
                "balanceDateTime": ["2022-01-01T00:00:00"],
            }
        )
        df_acc.to_csv(self.acc_file, index=False)

    def tearDown(self):
        self.tmp_dir_obj.cleanup()

    def test_train_mode_pipeline(self):
        # 1. Build Vocabs (implicitly via code path)
        # 2. Process

        test_args = [
            "preprocess_tensors.py",
            "--mode",
            "train",
            "--data_file",
            self.txn_file,
            "--account_file",
            self.acc_file,
            "--output_dir",
            self.output_dir,
            "--vocab_dir",
            self.vocab_dir,
            "--min_days",
            "1",  # Ensure our 2 days pass
        ]

        with patch.object(sys, "argv", test_args):
            preprocess_tensors.main()

        # Verify Output
        expected_tensor_file = os.path.join(self.output_dir, "pretrain_tensors.pt")
        self.assertTrue(os.path.exists(expected_tensor_file))

        # Check Tensors
        data = torch.load(expected_tensor_file, weights_only=False)  # List of dicts
        self.assertEqual(len(data), 1)  # 1 account
        item = data[0]
        self.assertEqual(item["account_id"], "acc_1")
        self.assertEqual(len(item["days"]), 2)

    def test_vocab_reuse(self):
        # Create headers only to simulate usage
        # Run once to build vocabs
        self.test_train_mode_pipeline()

        # Run again using pre-built vocabs
        test_args = [
            "preprocess_tensors.py",
            "--mode",
            "train",
            "--data_file",
            self.txn_file,
            "--account_file",
            self.acc_file,
            "--output_dir",
            self.output_dir,
            "--vocab_dir",
            self.vocab_dir,  # Should load from here
            "--min_days",
            "1",
        ]

        with patch.object(sys, "argv", test_args):
            preprocess_tensors.main()

        self.assertTrue(
            os.path.exists(os.path.join(self.vocab_dir, "cat_group_vocab.pkl"))
        )


if __name__ == "__main__":
    unittest.main()
