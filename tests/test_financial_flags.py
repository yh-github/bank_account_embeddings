import unittest
import pandas as pd
from datetime import datetime
from hierarchical.data.financial_flags import (
    generate_flag_name,
    detect_emerging,
    detect_skipped_payment,
)


class TestFinancialFlags(unittest.TestCase):
    def test_generate_flag_name(self):
        self.assertEqual(
            generate_flag_name("Buy Now Pay Later", "EMERGING"),
            "EMERGING_BUY_NOW_PAY_LATER",
        )
        self.assertEqual(
            generate_flag_name("Savings & Investment", "EMERGING"),
            "EMERGING_SAVINGS_INVESTMENT",
        )
        self.assertEqual(
            generate_flag_name("  Bad   Spacing  ", "EMERGING"), "EMERGING_BAD_SPACING"
        )

    def test_detect_emerging(self):
        # Setup Data
        # Account 1: Start 2023-01-01. Cat 100 appears 2023-06-01 (5 months later) -> FLAG
        # Account 2: Start 2023-01-01. Cat 100 appears 2023-02-01 (1 month later) -> NO FLAG
        data = {
            "accountId": ["A1", "A1", "A2", "A2"],
            "user_name": ["U1", "U1", "U2", "U2"],
            "date": [
                datetime(2023, 1, 1),
                datetime(2023, 6, 1),
                datetime(2023, 1, 1),
                datetime(2023, 2, 1),
            ],
            "personeticsSubCategoryId": [1, 100, 1, 100],
            "amount": [10, 10, 10, 10],
        }
        df = pd.DataFrame(data)

        categories = {100: {"name": "Target Cat", "group": "G1"}}
        target_subcategories = [100]
        primacy_months = 4

        flags = detect_emerging(df, target_subcategories, categories, primacy_months)

        self.assertEqual(len(flags), 1)
        self.assertEqual(flags.iloc[0]["accountId"], "A1")
        self.assertEqual(flags.iloc[0]["flag_name"], "EMERGING_TARGET_CAT")
        self.assertAlmostEqual(flags.iloc[0]["months_since_start"], 5.0, delta=0.2)

    def test_detect_emerging_empty(self):
        df = pd.DataFrame(columns=["accountId", "date", "personeticsSubCategoryId"])
        flags = detect_emerging(df, [100], {}, 4)
        self.assertTrue(flags.empty)

    def test_detect_skipped_payment(self):
        # Setup Data
        # Pattern 1: Regular payments (Jan, Feb, Mar) -> No Skip
        # Pattern 2: Skipped payment (Jan, Mar) -> Gap > 30 days -> Skip
        dates_reg = [datetime(2023, 1, 1), datetime(2023, 2, 1), datetime(2023, 3, 1)]
        dates_skip = [
            datetime(2023, 1, 1),
            datetime(2023, 3, 1),
            datetime(2023, 4, 1),
            datetime(2023, 5, 1),
        ]

        data = []
        # Regular
        for d in dates_reg:
            data.append(
                {"user_name": "U1", "accountId": "A1", "date": d, "patternId": "P1"}
            )
        # Skip
        for d in dates_skip:
            data.append(
                {"user_name": "U2", "accountId": "A2", "date": d, "patternId": "P2"}
            )

        df = pd.DataFrame(data)

        recurring_patterns = pd.DataFrame(
            {
                "patternId": ["P1", "P2"],
                "patternTypeBusiness": ["DateOfMonth", "DateOfMonth"],
                "deviceId": ["D1", "D2"],
            }
        )

        pattern_types = ["DateOfMonth"]
        gap_threshold = 45  # Days

        flags = detect_skipped_payment(
            df, recurring_patterns, pattern_types, gap_threshold, min_transactions=3
        )

        self.assertEqual(len(flags), 1)
        self.assertEqual(flags.iloc[0]["patternId"], "P2")
        self.assertEqual(flags.iloc[0]["gap_days"], 59)  # Jan 1 to Mar 1

    def test_skipped_payment_exclude_bimonthly(self):
        # 3 transactions, gaps > 50 days -> Should be excluded
        dates = [datetime(2023, 1, 1), datetime(2023, 3, 1), datetime(2023, 5, 1)]
        data = []
        for d in dates:
            data.append(
                {"user_name": "U1", "accountId": "A1", "date": d, "patternId": "P1"}
            )
        df = pd.DataFrame(data)

        recurring_patterns = pd.DataFrame(
            {
                "patternId": ["P1"],
                "patternTypeBusiness": ["DateOfMonth"],
                "deviceId": ["D1"],
            }
        )

        flags = detect_skipped_payment(
            df,
            recurring_patterns,
            ["DateOfMonth"],
            45,
            min_transactions=3,
            bimonthly_gap_threshold=50,
        )
        self.assertTrue(flags.empty)


if __name__ == "__main__":
    unittest.main()
