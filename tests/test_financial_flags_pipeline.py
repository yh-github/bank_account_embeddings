import unittest
import os
import yaml
import tempfile
import pandas as pd
from contextlib import redirect_stdout
import io

from hierarchical.data import financial_flags


class TestFinancialFlagsPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.test_dir = self.tmp_dir_obj.name

        # Paths
        self.txn_file = os.path.join(self.test_dir, "transactions.csv")
        self.acc_file = os.path.join(self.test_dir, "accounts.csv")
        self.pat_file = os.path.join(self.test_dir, "patterns.csv")
        self.config_file = os.path.join(self.test_dir, "config.yaml")
        self.output_dir = os.path.join(self.test_dir, "output")

        # 1. Create Data
        # Transactions
        # User U1: Has 'Savings' txns starting late (Emerging)
        txns = []
        # Early history (non-savings)
        for d in range(1, 10):
            txns.append(
                {
                    "id": f"tx_{d}",
                    "accountId": "acc_1",
                    "user_name": "user_1",
                    "date": f"2022-01-{d:02d}",
                    "amount": -10,
                    "direction": "D",
                    "personeticsSubCategoryId": "OTHER",
                    "patternId": "",
                }
            )
        # Late history (Savings appear in June)
        for d in range(1, 4):
            txns.append(
                {
                    "id": f"tx_new_{d}",
                    "accountId": "acc_1",
                    "user_name": "user_1",
                    "date": f"2022-06-{d:02d}",
                    "amount": -100,
                    "direction": "D",
                    "personeticsSubCategoryId": "SAVINGS_1",
                    "patternId": "pat_1",
                }
            )

        df_txn = pd.DataFrame(txns)
        df_txn.to_csv(self.txn_file, index=False)

        # Accounts
        df_acc = pd.DataFrame({"id": ["acc_1"], "type": ["Checking"]})
        df_acc.to_csv(self.acc_file, index=False)

        # Patterns
        df_pat = pd.DataFrame(
            {
                "patternId": ["pat_1"],
                "patternTypeBusiness": ["DateOfMonth"],
                "deviceId": ["dev_1"],
            }
        )
        df_pat.to_csv(self.pat_file, index=False)

        # 2. Config
        config = {
            "paths": {
                "transaction_file": self.txn_file,
                "account_file": self.acc_file,
                "recurring_pattern_file": self.pat_file,
                "output_dir": self.output_dir,
            },
            "parameters": {
                "primacy_period_months": 4,
                "gap_threshold_days": 20,
                "min_transactions": 2,  # Lower for test
            },
            "filters": {"account_type": "Checking", "direction": "D"},
            "categories": {
                "SAVINGS_1": {
                    "name": "Savings Test",
                    "group": "SAVINGS_GROUP",
                    "group_name": "Savings Group",
                },
                "OTHER": {"name": "Other", "group": "OTHER_GROUP"},
            },
            "target_category_groups": ["SAVINGS_GROUP"],
            "skipped_payment_filters": {"pattern_types": ["DateOfMonth"]},
        }
        with open(self.config_file, "w") as f:
            yaml.dump(config, f)

    def tearDown(self):
        self.tmp_dir_obj.cleanup()

    def test_main_execution(self):
        # Run main
        # Capture stdout to reduce noise
        f = io.StringIO()
        with redirect_stdout(f):
            financial_flags.main(self.config_file)

        # Verify Output
        emerging_path = os.path.join(self.output_dir, "emerging_flags.csv")
        skipped_path = os.path.join(self.output_dir, "skipped_payment_flags.csv")

        # Emerging: account start Jan, Savings start June -> >4 months -> Should Flag
        self.assertTrue(os.path.exists(emerging_path))
        df_em = pd.read_csv(emerging_path)
        self.assertEqual(len(df_em), 1)
        self.assertEqual(df_em.iloc[0]["flag_name"], "EMERGING_SAVINGS_TEST")

        # Skipped Payment: Not testing logic here (covered in unit test), just pipeline integration
        # (My data might not trigger skipped payment unless I engineer gaps, but file should typically not be saved if empty?)
        # Logic says: "skipped_payment_flags.csv (0 rows - not saved)"
        # So check if created only if expected.
        # But main runs without error.


if __name__ == "__main__":
    unittest.main()
