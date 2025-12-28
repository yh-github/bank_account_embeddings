
import unittest
import pandas as pd
import numpy as np
import datetime
from hierarchical.data.balance import BalanceFeatureExtractor

class TestBalanceEdgeCases(unittest.TestCase):
    def setUp(self):
        # Basic Setup
        self.today = datetime.date.today()
        self.yesterday = self.today - datetime.timedelta(days=1)
        
    def test_missing_balance_header(self):
        """Test behavior when account info is missing availableBalance."""
        df_tx = pd.DataFrame([
            {'accountId': 'ACC_MISSING', 'id': 't1', 'date': self.yesterday, 'amount': -100, 'direction': 'D'},
        ])
        df_acc = pd.DataFrame([
            {'accountId': 'ACC_MISSING', 'availableBalance': np.nan, 'balanceDateTime': self.today}
        ])
        
        # Should not crash, but balance might be NaN or 0
        extractor = BalanceFeatureExtractor(df_tx, df_acc)
        bal = extractor.get_daily_balance('ACC_MISSING', self.yesterday)
        
        # Depending on implementation, might be NaN
        self.assertTrue(np.isnan(bal) or bal is None, f"Expected NaN or None, got {bal}")

    def test_transactions_after_header_date(self):
        """Test behavior when transactions occur AFTER the balance snapshot date."""
        # Snapshot was yesterday, but we have a txn today
        df_tx = pd.DataFrame([
            {'accountId': 'ACC_FUTURE', 'id': 't1', 'date': self.today, 'amount': -50, 'direction': 'D'},
        ])
        df_acc = pd.DataFrame([
            {'accountId': 'ACC_FUTURE', 'availableBalance': 1000.0, 'balanceDateTime': self.yesterday}
        ])
        
        extractor = BalanceFeatureExtractor(df_tx, df_acc)
        
        # Logic: 
        # Snapshot (Yesterday) = 1000. 
        # Txn (Today) = -50.
        # Initial Balance (Yesterday) = Snapshot - Sum(History<=Yesterday) 
        # History<=Yesterday is EMPTY. So Initial = 1000.
        # Balance After T1 (Today) = Initial + Amount = 1000 + (-50) = 950.
        
        bal_today = extractor.get_daily_balance('ACC_FUTURE', self.today)
        self.assertEqual(bal_today, 950.0)

    def test_out_of_order_transactions(self):
        """Test robustness against unsorted input."""
        df_tx = pd.DataFrame([
            {'accountId': 'ACC_ORDER', 'id': 't2', 'date': self.today, 'amount': -10, 'direction': 'D'},
            {'accountId': 'ACC_ORDER', 'id': 't1', 'date': self.yesterday, 'amount': -20, 'direction': 'D'},
        ])
        df_acc = pd.DataFrame([
            {'accountId': 'ACC_ORDER', 'availableBalance': 100.0, 'balanceDateTime': self.today}
        ])
        
        # Snapshot (Today) = 100.
        # History (<=Today) = t1(-20) + t2(-10) = -30.
        # Initial = 100 - (-30) = 130.
        # Balance(Yesterday) = 130 + (-20) = 110.
        # Balance(Today) = 110 + (-10) = 100.
        
        # Pass unsorted
        extractor = BalanceFeatureExtractor(df_tx, df_acc)
        
        bal_yesterday = extractor.get_daily_balance('ACC_ORDER', self.yesterday)
        bal_today = extractor.get_daily_balance('ACC_ORDER', self.today)
        
        self.assertEqual(bal_yesterday, 110.0)
        self.assertEqual(bal_today, 100.0)

if __name__ == '__main__':
    unittest.main()
