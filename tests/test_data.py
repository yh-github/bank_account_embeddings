"""Unit tests for data loader."""

import unittest
import torch
from hierarchical.data.dataset import collate_hierarchical


class TestData(unittest.TestCase):
    def test_collate(self):
        # Mock item structure (existing test)
        pos_data = {
            "amounts": torch.tensor([10.0]),
            "n_txns": 1,
            "cat_group": torch.tensor([1]),
            "cat_sub": torch.tensor([1]),
            "cat_cp": torch.tensor([1]),
            "dates": torch.zeros(1, 4),
        }
        neg_data = {
            "amounts": torch.tensor([]),
            "n_txns": 0,
            "cat_group": torch.tensor([], dtype=torch.long),
            "cat_sub": torch.tensor([], dtype=torch.long),
            "cat_cp": torch.tensor([], dtype=torch.long),
            "dates": torch.zeros(0, 4),
        }

        item1 = {
            "days": [
                {
                    "pos": pos_data,
                    "neg": neg_data,
                    "n_txns": 1,
                    "meta": {"month": 1, "dow": 0, "dom": 1},
                }
            ],
            "day_dates": torch.tensor([100]),
            "n_days": 1,
            "account_id": "A1",
        }
        batch = [item1]
        out = collate_hierarchical(batch)
        self.assertIsInstance(out, dict)
        self.assertIn("pos", out)
        self.assertEqual(out["pos"]["cat_group"].shape[0], 1)


if __name__ == "__main__":
    unittest.main()
