"""Unit tests for balance calculator."""

import unittest
from datetime import date, datetime

import pandas as pd
import numpy as np

from hierarchical.data.balance import (
    calculate_balance_per_transaction,
    calculate_daily_balances,
)


class TestBalanceCalculator(unittest.TestCase):
    """Tests for balance calculation logic."""

    def setUp(self):
        # Sample Transactions
        self.txn_data = {
            'accountId': ['A1', 'A1', 'A1'],
            'id': ['T1', 'T2', 'T3'],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'amount': [-100.0, -50.0, 200.0],  # Net: +50
            'direction': ['D', 'D', 'C']
        }
        self.df_txn = pd.DataFrame(self.txn_data)

        # Sample Account
        # Final balance should be Initial + Net_Change
        # If Initial was 1000, final is 1050.
        self.acc_data = {
            'accountId': ['A1'],
            'availableBalance': [1050.0],
            'balanceDateTime': ['2023-01-10']
        }
        self.df_acc = pd.DataFrame(self.acc_data)

    def test_calculate_balance_per_transaction_basic(self):
        """Should correctly reconstruct running balances."""
        df_res = calculate_balance_per_transaction(
            self.df_txn, self.df_acc, include_pending=False
        )
        
        # Expected Logic:
        # Final = 1050
        # Total Txn = -100 -50 + 200 = +50
        # Initial = 1050 - 50 = 1000
        # T1 (Jan 1): 1000 + (-100) = 900
        # T2 (Jan 2): 900 + (-50) = 850
        # T3 (Jan 3): 850 + 200 = 1050
        
        self.assertEqual(len(df_res), 3)
        self.assertAlmostEqual(df_res.iloc[0]['balance_after'], 900.0)
        self.assertAlmostEqual(df_res.iloc[1]['balance_after'], 850.0)
        self.assertAlmostEqual(df_res.iloc[2]['balance_after'], 1050.0)

    def test_calculate_balance_sign_flip_detection(self):
        """Should detect incorrect signs and flip them."""
        # Case: Direction D (Debit) but amount is positive
        # Calculator should detect this heuristic and flip amounts to negative
        txn_data = {
            'accountId': ['A2', 'A2'],
            'id': ['TX1', 'TX2'],
            'date': ['2023-01-01', '2023-01-02'],
            'amount': [100.0, 200.0],  # Positive but Direction 'D'
            'direction': ['D', 'D']
        }
        df_txn = pd.DataFrame(txn_data)
        
        acc_data = {
            'accountId': ['A2'],
            'availableBalance': [700.0],
            'balanceDateTime': ['2023-01-10']
        }
        df_acc = pd.DataFrame(acc_data)
        
        df_res = calculate_balance_per_transaction(
            df_txn, df_acc, include_pending=False
        )
        
        # Logic: 
        # Total (flipped) = -100 - 200 = -300
        # Initial = 700 - (-300) = 1000
        # TX1: 1000 - 100 = 900
        # TX2: 900 - 200 = 700
        
        # Verify amounts were flipped in processing (output contains signed amounts)
        # Note: The function returns 'amount' column as processed.
        self.assertLess(df_res.iloc[0]['amount'], 0) # Should be negative
        self.assertAlmostEqual(df_res.iloc[0]['balance_after'], 900.0)
        self.assertAlmostEqual(df_res.iloc[1]['balance_after'], 700.0)

    def test_calculate_daily_balances(self):
        """Should aggregate to daily granularity."""
        # Multiple txns on same day
        txn_data = {
            'accountId': ['A3', 'A3'],
            'id': ['T_AM', 'T_PM'],
            'date': ['2023-02-01', '2023-02-01'], # Same day
            'amount': [-10.0, -20.0],
            'direction': ['D', 'D']
        }
        df_txn = pd.DataFrame(txn_data)
        
        acc_data = {
            'accountId': ['A3'],
            'availableBalance': [970.0],
            'balanceDateTime': ['2023-02-05']
        }
        df_acc = pd.DataFrame(acc_data)
        
        # Total = -30. Initial = 970 - (-30) = 1000
        # End of day 2023-02-01: 1000 - 10 - 20 = 970
        
        df_daily = calculate_daily_balances(df_txn, df_acc, include_pending=False)
        
        self.assertEqual(len(df_daily), 1)
        self.assertEqual(df_daily.iloc[0]['date'], date(2023, 2, 1))
        self.assertAlmostEqual(df_daily.iloc[0]['balance'], 970.0)

    def test_missing_account_balance(self):
        """Should handle missing account info gracefully (warn and output NaNs)."""
        # Txn exists but Account ID not in df_acc
        df_res = calculate_balance_per_transaction(
            self.df_txn, 
            pd.DataFrame(columns=['accountId', 'availableBalance', 'balanceDateTime'])
        )
        # Should run but produce NaNs for balance_after
        self.assertTrue(df_res['balance_after'].isna().all())

    def test_pending_transactions(self):
        """Should add synthetic transaction for pending amount."""
        # Pending amount = -50 (e.g. pending debit)
        # Balance is 1000. 
        # The logic adds a "ROLLBACK" transaction.
        # Logic in implementation: "synthetic rollback transactions".
        # Code: id='PENDING_ROLLBACK', date=balanceDateTime, amount=estimatedPendingAmount
        # direction based on sign.
        # This seems to treat 'estimatedPendingAmount' as something to be ADDED to stream?
        # Let's verify behavior.
        
        acc_data = {
            'accountId': ['A1'],
            'availableBalance': [1000.0],
            'balanceDateTime': ['2023-01-05'],
            'estimatedPendingAmount': [-50.0]
        }
        df_acc = pd.DataFrame(acc_data)
        # No historic txns for simplicity
        df_txn = pd.DataFrame(columns=['accountId', 'id', 'date', 'amount', 'direction'])
        
        # Note: validation requires columns
        df_txn = pd.DataFrame({
             'accountId': ['A1'], 'id': ['DUMMY'], 'date': ['2023-01-01'], 
             'amount': [0.0], 'direction': ['C']
        })
        
        # Function requires at least correct columns
        df_res = calculate_balance_per_transaction(
            df_txn, df_acc, include_pending=True
        )
        
        # Should have dummy + pending
        self.assertEqual(len(df_res), 2)
        # Pending should be last (date 2023-01-05 vs 2023-01-01)
        self.assertEqual(df_res.iloc[1]['id'], 'PENDING_ROLLBACK')
        self.assertEqual(df_res.iloc[1]['amount'], -50.0)
        self.assertAlmostEqual(df_res.iloc[0]['balance_after'], 1050.0) # Initial was 1050
        
        # Wait, math check:
        # Total amount in stream = -50.
        # Available Balance = 1000.
        # Initial = 1000 - (-50) = 1050.
        # Txn 1 (Pending): 1050 + (-50) = 1000.
        # So "Pending" is treated as having *already happened* in the calculation of "Available"?
        # Usually "Available" excludes pending, "Ledger" includes it (or vice versa depending on bank).
        # The code treats it as a transaction to be included in the history.


if __name__ == '__main__':
    unittest.main()
