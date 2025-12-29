"""Component tests for data loader."""

import unittest
import pandas as pd
import tempfile
import os
from hierarchical.data.loader import load_joint_bank_data


class TestLoader(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.test_dir = self.tmp_dir.name

        # Create dummy data for Bank A
        self.txn_a_path = os.path.join(self.test_dir, "txn_a.csv")
        pd.DataFrame(
            {
                "accountId": ["1", "2"],
                "id": ["tx1", "tx2"],
                "date": ["2023-01-01", "2023-01-02"],
                "amount": [100.0, -50.0],
                "direction": ["C", "D"],
            }
        ).to_csv(self.txn_a_path, index=False)

        self.acc_a_path = os.path.join(self.test_dir, "acc_a.csv")
        pd.DataFrame(
            {
                "accountId": ["1", "2"],
                "availableBalance": [1000, 2000],
                "balanceDateTime": ["2023-01-03", "2023-01-03"],
            }
        ).to_csv(self.acc_a_path, index=False)

        # Create dummy data for Bank B (same local IDs to test prefixing)
        self.txn_b_path = os.path.join(self.test_dir, "txn_b.csv")
        pd.DataFrame(
            {
                "accountId": ["1", "3"],  # ID '1' overlaps locally
                "id": ["tx1", "tx3"],  # ID 'tx1' overlaps locally
                "date": ["2023-01-01", "2023-01-04"],
                "amount": [200.0, -20.0],
                "direction": ["C", "D"],
            }
        ).to_csv(self.txn_b_path, index=False)

        self.acc_b_path = os.path.join(self.test_dir, "acc_b.csv")
        pd.DataFrame(
            {
                "accountId": ["1", "3"],
                "availableBalance": [500, 3000],
                "balanceDateTime": ["2023-01-05", "2023-01-05"],
            }
        ).to_csv(self.acc_b_path, index=False)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_joint_load_prefixing(self):
        configs = [
            {"name": "BankA", "txn_file": self.txn_a_path, "acc_file": self.acc_a_path},
            {"name": "BankB", "txn_file": self.txn_b_path, "acc_file": self.acc_b_path},
        ]

        merged_txns, merged_accs = load_joint_bank_data(configs)

        # Check lengths
        self.assertEqual(len(merged_txns), 4)  # 2 from A + 2 from B
        self.assertEqual(len(merged_accs), 4)  # 2 from A + 2 from B

        # Check Account ID Prefixing
        unique_accs = merged_accs["accountId"].unique()
        self.assertIn("BankA_1", unique_accs)
        self.assertIn("BankB_1", unique_accs)
        self.assertNotIn("1", unique_accs)  # Original IDs should be gone

        # Verify transaction linkage
        txns_bank_a = merged_txns[merged_txns["accountId"] == "BankA_1"]
        self.assertEqual(len(txns_bank_a), 1)
        self.assertEqual(txns_bank_a.iloc[0]["amount"], 100.0)

    def test_cutoff_date(self):
        configs = [
            {
                "name": "BankA",
                "txn_file": self.txn_a_path,
                "acc_file": self.acc_a_path,
                "cutoff_date": "2023-01-02",
            }
        ]
        # Only txns strictly BEFORE cutoff should remain
        # 2023-01-01 is < 2023-01-02.
        # 2023-01-02 is NOT < 2023-01-02 (it is equal).

        merged_txns, _ = load_joint_bank_data(configs)

        self.assertEqual(len(merged_txns), 1)
        self.assertEqual(merged_txns.iloc[0]["date"], pd.Timestamp("2023-01-01"))


if __name__ == "__main__":
    unittest.main()
