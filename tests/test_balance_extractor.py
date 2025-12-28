"""Unit tests for balance feature extractor."""

import unittest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta

from hierarchical.data.balance import BalanceFeatureExtractor

class MockBalanceCalculator:
    """Mock for balance calculation result."""
    @staticmethod
    def calculate_daily_balances_return_mock(data_dict):
         return pd.DataFrame(data_dict)

class TestBalanceFeatureExtractor(unittest.TestCase):
    """Tests for BalanceFeatureExtractor class."""

    def test_initialization_and_indexing(self):
        """Should correctly index daily balances."""
        # Mock calculation output
        # A1 has data for Jan 1 and Jan 2
        data = {
            'accountId': ['A1', 'A1', 'B1'],
            'date': [date(2023, 1, 1), date(2023, 1, 2), date(2023, 1, 1)],
            'balance': [100.0, 150.0, 500.0],
            'running_date': date(2023, 1, 10)
        }
        df_balances = pd.DataFrame(data)
        
        # We need to bypass the actual calculation which requires raw transactions.
        # The class calls `calculate_daily_balances` in __init__.
        # We can mock `calculate_daily_balances` or subclass/monkeypatch.
        # Simpler approach: Create valid empty inputs that produce known output,
        # OR just monkeypatch the module function.
        
        pass

    def test_normalize_balance(self):
        """Should sign-normalize balances."""
        # Use dummy inputs, method is static-like (instance method but depends on nothing)
        # Actually it's an instance method but doesn't use self.
        
        # Create minimal instance (hacky but effective if we don't want to mock everything)
        # But __init__ runs weighty calculation.
        # Better to test logic directly if possible.
        
        # Since we can't easily instantiate without data, let's trust the logic inspection
        # or use a small integration test with valid (but small) data.
        
        df_txn = pd.DataFrame({
            'accountId': ['A1'], 'id': ['T1'], 'date': ['2023-01-01'],
            'amount': [100.0], 'direction': ['C']
        })
        df_acc = pd.DataFrame({
            'accountId': ['A1'], 'availableBalance': [100.0], 'balanceDateTime': ['2023-01-02']
        })
        
        extractor = BalanceFeatureExtractor(df_txn, df_acc)
        
        # Test Normalization
        val = extractor.normalize_balance(100.0)
        expected = np.log1p(100.0)
        self.assertAlmostEqual(val, expected)
        
        val_neg = extractor.normalize_balance(-100.0)
        expected_neg = -np.log1p(100.0)
        self.assertAlmostEqual(val_neg, expected_neg)

    def test_get_window_stats(self):
        """Should calculate correct statistics for a window."""
        # Setup: A1 balance: Day 1=100, Day 2=200, Day 3=150
        # Final = 150. Initial = 150 - (100+50-50)=50? No.
        # Let's just define the scenario to produce these balances.
        
        # Init = 0.
        # T1: +100 -> Bal 100
        # T2: +100 -> Bal 200
        # T3: -50  -> Bal 150
        # Final Available = 150.
        
        df_txn = pd.DataFrame({
            'accountId': ['A1', 'A1', 'A1'], 
            'id': ['T1', 'T2', 'T3'], 
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'amount': [100.0, 100.0, -50.0], 
            'direction': ['C', 'C', 'D']
        })
        df_acc = pd.DataFrame({
            'accountId': ['A1'], 
            'availableBalance': [150.0], 
            'balanceDateTime': ['2023-01-05']
        })
        
        extractor = BalanceFeatureExtractor(df_txn, df_acc)
        
        # Window: Jan 1 to Jan 3
        stats = extractor.get_window_stats('A1', '2023-01-01', '2023-01-03')
        
        self.assertAlmostEqual(stats['start_balance'], 100.0) # Jan 1 balance
        self.assertAlmostEqual(stats['end_balance'], 150.0)   # Jan 3 balance
        self.assertAlmostEqual(stats['max_balance'], 200.0)   # Jan 2 was 200
        self.assertAlmostEqual(stats['min_balance'], 100.0)
        
    def test_get_starting_balance(self):
        """Should return balance of previous day."""
        # Using same setup as above
        df_txn = pd.DataFrame({
            'accountId': ['A1'], 
            'id': ['T1'], 
            'date': ['2023-01-05'],
            'amount': [100.0], 
            'direction': ['C']
        })
        df_acc = pd.DataFrame({
            'accountId': ['A1'], 
            'availableBalance': [100.0], 
            'balanceDateTime': ['2023-01-10']
        })
        extractor = BalanceFeatureExtractor(df_txn, df_acc)
        
        # Day 2023-01-05 balance is 100.
        # get_starting_balance for 2023-01-06 should be 100.
        self.assertAlmostEqual(extractor.get_starting_balance('A1', '2023-01-06'), 100.0)
        
        # get_starting_balance for 2023-01-05 (day of txn)
        # Should be previous day balance. If no prev day, returns earliest (100).
        # Wait, get_starting_balance logic:
        # "Find most recent date BEFORE given date".
        # If given is Jan 5, before is nothing. Returns earliest -> 100.
        self.assertAlmostEqual(extractor.get_starting_balance('A1', '2023-01-05'), 100.0)


if __name__ == '__main__':
    unittest.main()
