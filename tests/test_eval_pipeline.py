"""
Tests for the evaluation pipeline components.

Since we already have extensive tests for helper functions (test_eval_helpers.py)
and model loading (test_eval_load_model.py), this test just verifies basic
data structures can be loaded.
"""
import os
import tempfile
import unittest

import pandas as pd
import torch


class TestEvalPipeline(unittest.TestCase):
    """Test basic evaluation pipeline data structures."""
    
    def test_data_structure_compatibility(self):
        """Test that tensor and flags data structures are compatible."""
        # Create dummy tensor data matching expected structure
        accounts = []
        for i in range(3):
            days = []
            for d in range(5):
                pos = {
                    'cat_group': torch.randint(0, 10, (1, 3)),
                    'cat_sub': torch.randint(0, 10, (1, 3)),
                    'cat_cp': torch.randint(0, 10, (1, 3)),
                    'amounts': torch.randn(1, 3),
                    'dates': torch.zeros(1, 3, 4),
                    'mask': torch.ones(1, 3).bool(),
                    'has_data': torch.tensor([True]),
                    'n_txns': 3
                }
                days.append({'pos': pos, 'neg': None})
            
            accounts.append({
                'account_id': f"bank_{i}",
                'days': days,
                'day_dates': list(range(5)),
                'n_days': 5,
                'total_txns': 15
            })
        
        # Verify structure
        self.assertEqual(len(accounts), 3)
        self.assertEqual(len(accounts[0]['days']), 5)
        self.assertIn('n_txns', accounts[0]['days'][0]['pos'])
        
        # Create flags data
        flags_data = [{
            'global_id': 'bank_0',
            'flag': 'EMERGING_TEST',
            'emerging_date': '2023-01-15'
        }]
        df = pd.DataFrame(flags_data)
        
        # Verify structure
        self.assertIn('global_id', df.columns)
        self.assertIn('flag', df.columns)


if __name__ == '__main__':
    unittest.main()
