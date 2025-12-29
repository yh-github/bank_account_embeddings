import unittest
import os
import sys
import tempfile
import shutil
import torch
import pandas as pd
import numpy as np
from unittest.mock import patch

from hierarchical.models.transaction import TransactionEncoder
from hierarchical.models.day import DayEncoder
from hierarchical.models.account import AccountEncoder
import hierarchical.evaluation.evaluate as eval_script

class TestEvalPipeline(unittest.TestCase):
    def setUp(self):
        # Temp dir
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.test_dir = self.tmp_dir_obj.name
        
        # Paths
        self.ckpt_path = os.path.join(self.test_dir, 'model.pth')
        self.tensors_path = os.path.join(self.test_dir, 'tensors.pt')
        self.flags_path = os.path.join(self.test_dir, 'flags.csv')
        self.output_dir = os.path.join(self.test_dir, 'eval_out')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. Create Dummy Model
        self.hidden_dim = 32
        self.txn_dim = 32
        self.day_dim = 32
        self.acc_dim = 32
        
        txn_enc = TransactionEncoder(10, 10, 10, embedding_dim=self.txn_dim)
        day_enc = DayEncoder(txn_enc, hidden_dim=self.day_dim, num_layers=1)
        model = AccountEncoder(day_enc, hidden_dim=self.acc_dim, num_layers=1)
        
        # Save checkpont with config
        config = {
            'num_categories_group': 10,
            'num_categories_sub': 10,
            'num_counter_parties': 10,
            'hidden_dim': 32,
            'txn_dim': 32,
            'day_dim': 32,
            'account_dim': 32,
            'num_layers': 1,
            'day_num_layers': 1,
            'num_heads': 2, # Must divide 32
            'use_balance': True,
            'use_counter_party': True
        }
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config
        }, self.ckpt_path)
        
        # 2. Create Dummy Tensors
        # List of items
        # Need at least a few accounts
        self.accounts = []
        for i in range(5):
            acc_id = f"acc_{i}"
            # Create random day stream
            days = []
            for d in range(10): # 10 days
                 pos = {
                     'cat_group': torch.randint(0, 10, (1, 5)), # T=5
                     'cat_sub': torch.randint(0, 10, (1, 5)),
                     'cat_cp': torch.randint(0, 10, (1, 5)),
                     'amounts': torch.randn(1, 5),
                     'dates': torch.zeros(1, 5, 2), # simplified
                     'mask': torch.ones(1, 5).bool(),
                     'has_data': torch.tensor([True])
                 }
                 # Wrap in dict structure expected by collate (nested day dicts?)
                 # The preloaded data structure is:
                 # item['days'] = List[DayDict]
                 # DayDict = {'pos': ..., 'neg': ...}
                 # But tensors for PreloadedDataset usually store RAW lists or TENSORS?
                 # unified_eval_optimized calls embed_batch -> collate_hierarchical.
                 # collate expects list of Dicts with 'days'.
                 
                 # But wait, precomputed tensors might already be Tensors?
                 # The script says "Precomputed Tensors".
                 # Does it skip collate?
                 # load_data in unified_eval loads the .pt.
                 # Let's check unified_eval logic for data loading.
                 # If it accepts list of dicts, fine.
                 
                 day_item = {'pos': pos, 'neg': None}
                 days.append(day_item)
            
            self.accounts.append({
                'account_id': acc_id,
                'days': days,
                'day_dates': [f"2023-01-{d+1:02d}" for d in range(10)]
            })
            
        torch.save(self.accounts, self.tensors_path)
        
        # 3. Create Emerging Flags CSV
        # Needs accountId, flag_name, bank, emerging_date (required by evaluate.py)
        # acc_0, acc_1 -> Positive
        # acc_2, acc_3, acc_4 -> Negative (no flag)
        df = pd.DataFrame({
            'accountId': ['0', '1'],  # Account IDs without prefix
            'bank': ['testbank', 'testbank'],  # Added missing 'bank' column
            'flag_name': ['EMERGING_SAVINGS', 'EMERGING_SAVINGS'],
            'emerging_date': ['2023-01-08', '2023-01-09']  # Renamed from 'date' to 'emerging_date'
        })
        df.to_csv(self.flags_path, index=False)

    def tearDown(self):
        self.tmp_dir_obj.cleanup()

    def test_pipeline_execution(self):
        # Patch arguments
        test_args = [
            'evaluate.py',
            '--tensors_path', self.tensors_path,
            '--emerging_csv', self.flags_path,
            '--model_checkpoint', self.ckpt_path,
            '--output_dir', self.output_dir,
            '--hidden_dim', '32',
            '--batch_size', '4',
            '--T_values', '5', # Use T=5, our data has 10 days
            '--negatives_in_training', '100', # Dummy
            '--test_size', '0.5' # Ensure split works
        ]
        
        with patch.object(sys, 'argv', test_args):
            # Run
            # Warning: this might print a lot to stdout/stderr
            try:
                eval_script.main()
            except SystemExit as e:
                # argparse might exit 0 or 1
                if e.code != 0:
                    self.fail(f"Pipeline exited with code {e.code}")
            
        # Verify Outputs
        metrics_csv = os.path.join(self.output_dir, 'ensemble_summary_metrics.csv')
        curves_csv = os.path.join(self.output_dir, 'cumulative_lift_curves.csv')
        
        self.assertTrue(os.path.exists(metrics_csv), "Metrics CSV was not created")
        
        # Check content
        df_metrics = pd.read_csv(metrics_csv)
        self.assertGreater(len(df_metrics), 0, "Metrics CSV is empty")
        # Should have results for EMERGING_SAVINGS
        self.assertTrue('EMERGING_SAVINGS' in df_metrics['flag'].values)
        
        # We can't guarantee high lift on random data, but we can guarantee rows exist.
        
        # Check integrity of output
        # e.g. Lift should be numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(df_metrics['Lift@5%']))

if __name__ == '__main__':
    unittest.main()
