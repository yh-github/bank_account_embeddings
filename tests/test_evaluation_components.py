import unittest
import numpy as np
from hierarchical.evaluation.evaluate import (
    calculate_confidence,
    compute_lift_curve,
    generate_granular_curve,
    extract_tensor_features,
)


class TestEvaluationComponents(unittest.TestCase):
    def test_calculate_confidence(self):
        # Hypergeometric test logic
        # k=10, N=100, M=20, n=20
        # If we drew 20 items, and 20 were positive (M), and we got 10 positives (k).
        # This should be high confidence?

        # Simple case: k=0, should be 0
        self.assertEqual(calculate_confidence(0, 100, 20, 20), 0.0)

        # Case: Random chance.
        # N=100, M=50 (50% positive). n=10. Expected k=5.
        # If k=5, p-value is roughly 0.5. Confidence ~50%.
        # If k=10 (all pos), p-value is very low. Confidence ~100%.
        conf = calculate_confidence(10, 100, 50, 10)
        self.assertGreater(conf, 99.0)

    def test_compute_lift_curve(self):
        # Perfect prediction
        y_true = np.array([1, 1, 0, 0])
        y_scores = np.array([0.9, 0.8, 0.2, 0.1])

        # At K=50% (Top 2). TP=2. Precision=1.0.
        # Base rate = 0.5. Lift = 1.0 / 0.5 = 2.0.

        lifts, confs, recalls, precisions, f1s, n_founds, depths = compute_lift_curve(
            y_true, y_scores, ks=[50]
        )

        self.assertAlmostEqual(lifts["lift_50"], 2.0)
        self.assertAlmostEqual(precisions["precision_50"], 1.0)
        self.assertAlmostEqual(recalls["recall_50"], 1.0)
        self.assertAlmostEqual(f1s["f1_50"], 1.0)  # 2*1*1 / 2 = 1.0

    def test_compute_lift_curve_random(self):
        # Worst prediction (reversed)
        y_true = np.array([1, 1, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])

        # At K=50% (Top 2). Top 2 are [0, 0]. TP=0.
        lifts, _, _, _, _, _, _ = compute_lift_curve(y_true, y_scores, ks=[50])
        self.assertEqual(lifts["lift_50"], 0.0)

    def test_generate_granular_curve(self):
        y_true = np.array([1, 0] * 50)  # 100 items, 50 pos
        y_scores = np.linspace(1, 0, 100)  # Perfect sort if matched
        # But y_true is alternating 1,0,1,0...
        # So top 50 scores match [1, 0, 1, 0...]
        # Precision at 50% (50 items) -> 25 positives.
        # Lift = (25/50) / (50/100) = 0.5 / 0.5 = 1.0 (Random)

        rows = generate_granular_curve(y_true, y_scores, "FLAG", 7, "XGB", True)

        # Check a row near K=50
        row_50 = next(r for r in rows if r["k_pct"] == 50.0)
        self.assertAlmostEqual(row_50["lift"], 1.0, delta=0.1)
        self.assertEqual(row_50["n_total"], 100)

    def test_extract_tensor_features(self):
        # Mock item structure
        item = {
            "n_days": 5,
            "days": [
                {"pos": {"amounts": [100.0, 50.0]}},  # Day 1
                {"neg": {"amounts": [-20.0], "balance": [[1000.0] * 7]}},  # Day 2
            ],
        }

        # extract_tensor_features returns [txn_count, n_days, total_debit, total_credit, avg_debit, n_large_debits, bal0...bal6]
        # Total 6 static + 7 bal = 13 features

        feats = extract_tensor_features(item)
        self.assertEqual(len(feats), 13)

        self.assertEqual(feats[0], 3)  # 3 txns
        self.assertEqual(feats[1], 5)  # n_days
        self.assertEqual(feats[2], -20.0)  # total_debit
        self.assertEqual(feats[3], 150.0)  # total_credit
        self.assertEqual(feats[4], -20.0)  # avg_debit (-20/1)
        self.assertEqual(feats[5], 0)  # n_large_debits
        self.assertEqual(feats[6], 1000.0)  # start_balance from last day neg stream


if __name__ == "__main__":
    unittest.main()
