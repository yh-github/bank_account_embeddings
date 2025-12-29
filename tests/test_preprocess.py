import unittest
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from hierarchical.data.preprocess_tensors import (
    process_account_chunk,
    init_worker,
    patch_df
)

class MockVocab:
    def encode(self, x):
        return 1 # Returns index 1 for everything

class TestPreprocess(unittest.TestCase):
    def setUp(self):
        # Mock Vocabs
        self.vocabs = (MockVocab(), MockVocab(), MockVocab())
        
        # Mock Config
        self.config = {
            'max_days': 5,
            'max_txns': 3
        }
        
        # Mock Balance Cache
        # (account_id, date_obj) -> np.array
        # date_only in code seems to be datetime.date object from `df['date_only']`
        self.balance_cache = {}
        
        # Initialize Globals
        init_worker(self.vocabs, self.balance_cache, self.config)

    def test_patch_df(self):
        df = pd.DataFrame({'trId': [1], 'amount': [100]})
        patched = patch_df(df)
        self.assertIn('id', patched.columns)
        self.assertIn('direction', patched.columns)
        self.assertEqual(patched.iloc[0]['direction'], 'Credit')

    def test_process_account_chunk(self):
        # Data Setup
        data = [
            {'accountId': 'A1', 'date': '2023-01-01', 'amount': 100, 'personeticsCategoryGroupId': 'G1', 'personeticsSubCategoryId': 'S1', 'counterParty': 'C1'},
            {'accountId': 'A1', 'date': '2023-01-01', 'amount': -50, 'personeticsCategoryGroupId': 'G1', 'personeticsSubCategoryId': 'S1', 'counterParty': 'C1'},
            {'accountId': 'A1', 'date': '2023-01-02', 'amount': 20, 'personeticsCategoryGroupId': 'G1', 'personeticsSubCategoryId': 'S1', 'counterParty': 'C1'},
        ]
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df['date_only'] = df['date'].dt.date
        
        # Chunk format: list of (account_id, df)
        chunk = [('A1', df)]
        
        # Run
        results = process_account_chunk(chunk)
        
        # Verify
        self.assertEqual(len(results), 1)
        res = results[0]
        self.assertEqual(res['account_id'], 'A1')
        self.assertIn('days', res)
        days = res['days']
        self.assertEqual(len(days), 2) # 2 unique days
        
        # Check Day 1 (Jan 1)
        day1 = days[0]
        self.assertIn('pos', day1)
        self.assertIn('neg', day1)
        self.assertIsNotNone(day1['pos'])
        self.assertIsNotNone(day1['neg'])
        
        # Check tensor shapes match max_txns (which is 3) or actual count (1)?
        # Implementation of process_account_chunk usually returns tensors of length N (actual data) 
        # OR it packs them?
        # Let's see code... `process_account_chunk` calls `tensor(..., dtype=...)`.
        # It creates tensors from valid transactions.
        
        pos_tensors = day1['pos']
        self.assertEqual(pos_tensors['amounts'].shape[0], 1) # 1 pos txn
        # Logic centers the stream, so single item becomes 0
        self.assertTrue(np.isclose(pos_tensors['amounts'][0], 0.0, atol=1e-5))
        # Wait, preprocess_tensors lines 100+ usually implement extraction logic inline or call helper.
        # Assuming inline for now or robust enough to not crash.
        
    def test_process_account_chunk_empty_stream(self):
        # Day with only negative
        data = [
            {'accountId': 'A1', 'date': '2023-01-01', 'amount': -50, 'personeticsCategoryGroupId': 'G1', 'personeticsSubCategoryId': 'S1', 'counterParty': 'C1'},
        ]
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df['date_only'] = df['date'].dt.date
        
        chunk = [('A1', df)]
        results = process_account_chunk(chunk)
        
        day1 = results[0]['days'][0]
        self.assertIsNone(day1['pos']) # Should be None
        self.assertIsNotNone(day1['neg'])

if __name__ == '__main__':
    unittest.main()
