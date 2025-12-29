"""Unit tests for models."""

import unittest

import torch

from hierarchical.models.transaction import TransactionEncoder
from hierarchical.models.day import DayEncoder
from hierarchical.models.account import AccountEncoder


class TestTransactionEncoder(unittest.TestCase):
    """Tests for TransactionEncoder."""

    def setUp(self):
        self.encoder = TransactionEncoder(
            num_categories_group=10,
            num_categories_sub=10,
            num_counter_parties=10,
            embedding_dim=32,
            use_counter_party=True,
            use_balance=False,
        )

    def test_forward_shape(self):
        """Forward should output [B, L, embedding_dim]."""
        B, L = 5, 10
        cat_group = torch.randint(0, 10, (B, L))
        cat_sub = torch.randint(0, 10, (B, L))
        counter_party = torch.randint(0, 10, (B, L))
        amount = torch.randn(B, L, 1)
        dates = torch.zeros(B, L, 4)

        out = self.encoder(
            category_group_ids=cat_group,
            category_sub_ids=cat_sub,
            counter_party_ids=counter_party,
            amounts=amount,
            dates=dates,
            balance_features=None,
        )
        self.assertEqual(out.shape, (B, L, 32))

    def test_forward_without_counter_party(self):
        """Encoder without counter_party should work."""
        encoder = TransactionEncoder(
            num_categories_group=10,
            num_categories_sub=10,
            num_counter_parties=10,
            embedding_dim=32,
            use_counter_party=False,
            use_balance=False,
        )
        B, L = 3, 5
        out = encoder(
            category_group_ids=torch.randint(0, 10, (B, L)),
            category_sub_ids=torch.randint(0, 10, (B, L)),
            counter_party_ids=None,
            amounts=torch.randn(B, L, 1),
            dates=torch.zeros(B, L, 4),
        )
        self.assertEqual(out.shape, (B, L, 32))

    def test_amount_2d_input(self):
        """Amounts can be [B, L] instead of [B, L, 1]."""
        B, L = 4, 8
        out = self.encoder(
            category_group_ids=torch.randint(0, 10, (B, L)),
            category_sub_ids=torch.randint(0, 10, (B, L)),
            counter_party_ids=torch.randint(0, 10, (B, L)),
            amounts=torch.randn(B, L),  # 2D input
            dates=torch.zeros(B, L, 4),
        )
        self.assertEqual(out.shape, (B, L, 32))


class TestDayEncoder(unittest.TestCase):
    """Tests for DayEncoder."""

    def setUp(self):
        txn_enc = TransactionEncoder(
            num_categories_group=10,
            num_categories_sub=10,
            num_counter_parties=10,
            embedding_dim=32,
            use_counter_party=True,
            use_balance=False,
        )
        self.day_enc = DayEncoder(
            txn_encoder=txn_enc, hidden_dim=32, num_layers=1, num_heads=2
        )

    def _make_stream_data(self, N: int, T: int) -> dict:
        """Helper to create stream data dict."""
        return {
            "cat_group": torch.randint(0, 10, (N, T)),
            "cat_sub": torch.randint(0, 10, (N, T)),
            "cat_cp": torch.randint(0, 10, (N, T)),
            "amounts": torch.randn(N, T),
            "dates": torch.zeros(N, T, 4),
            "balance": None,
            "mask": torch.ones(N, T, dtype=torch.bool),
            "has_data": torch.ones(N, dtype=torch.bool),
        }

    def test_forward_shape(self):
        """DayEncoder should output [N_Days, hidden_dim]."""
        N_Days, T = 6, 8
        pos_data = self._make_stream_data(N_Days, T)
        neg_data = self._make_stream_data(N_Days, T)

        out = self.day_enc(pos_data, neg_data)
        self.assertEqual(out.shape, (N_Days, 32))

    def test_empty_stream(self):
        """DayEncoder should handle empty streams gracefully."""
        N_Days, T = 4, 5
        pos_data = self._make_stream_data(N_Days, T)
        # Empty negative stream
        neg_data = self._make_stream_data(N_Days, T)
        neg_data["mask"] = torch.zeros(N_Days, T, dtype=torch.bool)
        neg_data["has_data"] = torch.zeros(N_Days, dtype=torch.bool)

        out = self.day_enc(pos_data, neg_data)
        self.assertEqual(out.shape, (N_Days, 32))
        self.assertFalse(torch.isnan(out).any())


class TestAccountEncoder(unittest.TestCase):
    """Tests for AccountEncoder."""

    def setUp(self):
        txn_enc = TransactionEncoder(
            num_categories_group=10,
            num_categories_sub=10,
            num_counter_parties=10,
            embedding_dim=32,
            use_counter_party=True,
            use_balance=False,
        )
        day_enc = DayEncoder(
            txn_encoder=txn_enc, hidden_dim=32, num_layers=1, num_heads=2
        )
        self.acc_enc = AccountEncoder(
            day_encoder=day_enc, hidden_dim=32, num_layers=1, num_heads=2
        )

    def _make_batch_data(self, B: int, D: int, T: int) -> dict:
        """Helper to create batch data dict matching collate output."""

        def make_stream(B: int, D: int, T: int) -> dict:
            return {
                "cat_group": torch.randint(0, 10, (B, D, T)),
                "cat_sub": torch.randint(0, 10, (B, D, T)),
                "cat_cp": torch.randint(0, 10, (B, D, T)),
                "amounts": torch.randn(B, D, T),
                "dates": torch.zeros(B, D, T, 4),
                "balance": None,
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

    def test_forward_shape(self):
        """AccountEncoder should output [B, hidden_dim]."""
        B, D, T = 4, 10, 5
        batch = self._make_batch_data(B, D, T)

        out = self.acc_enc(batch)
        self.assertEqual(out.shape, (B, 32))

    def test_single_account(self):
        """Should work with batch size of 1."""
        batch = self._make_batch_data(1, 7, 4)
        out = self.acc_enc(batch)
        self.assertEqual(out.shape, (1, 32))

    def test_variable_day_counts(self):
        """Should handle variable day counts via masking."""
        B, D, T = 3, 15, 6
        batch = self._make_batch_data(B, D, T)
        # Mask some days as padding
        batch["meta"]["day_mask"][0, 10:] = False
        batch["meta"]["day_mask"][1, 12:] = False

        out = self.acc_enc(batch)
        self.assertEqual(out.shape, (B, 32))
        self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
