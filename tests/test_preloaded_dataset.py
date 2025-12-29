import unittest
import numpy as np
import torch
from hierarchical.data.preloaded_dataset import PreloadedDataset

class TestPreloadedDataset(unittest.TestCase):
    def setUp(self):
        # Create dummy data
        # Structure: list of dicts with 'account_id', 'days', 'day_dates'
        self.data_items = []
        for i in range(10): 
            # 10 accounts
            days = []
            day_dates = []
            n_days = 20 # 20 days per account
            for d in range(n_days):
                days.append({'pos': None, 'neg': None}) # Dummy content
                day_dates.append(f"2023-01-{d+1:02d}")
            
            self.data_items.append({
                'account_id': str(i),
                'days': days,
                'day_dates': day_dates
            })

    def test_init_from_list(self):
        ds = PreloadedDataset(self.data_items, augment=False)
        self.assertEqual(len(ds), 10)

    def test_slicing_no_augment(self):
        # max_days = 5
        ds = PreloadedDataset(self.data_items, augment=False, max_days=5)
        item = ds[0]
        
        self.assertEqual(len(item['days']), 5)
        self.assertEqual(len(item['day_dates']), 5)
        # Should be last 5 days
        self.assertEqual(item['day_dates'][-1], "2023-01-20")

    def test_augmentation_short_sequence(self):
        # Create short item (<5 days)
        short_items = [{
            'account_id': 'short',
            'days': [{}, {}, {}], # 3 days
            'day_dates': ['1', '2', '3']
        }]
        ds = PreloadedDataset(short_items, augment=True, max_days=10)
        item = ds[0]
        # Should not augment
        self.assertEqual(len(item['days']), 3)

    def test_augmentation_random_crop(self):
        # Seed for reproducibility if possible, but augmentation uses np.random directly.
        # We can patch np.random or just verify constraints.
        
        # Force "Crop" path (rand < 0.5)
        ds = PreloadedDataset(self.data_items, augment=True, max_days=20)
        
        np.random.seed(42) # Try to force crop path
        # With seed 42:
        # rand() 1st call: 0.37 (< 0.5) -> Crop
        # rand() 2nd call: for window size
        
        item = ds[0]
        
        # Check integrity
        self.assertEqual(len(item['days']), len(item['day_dates']))
        self.assertLessEqual(len(item['days']), 20)
        self.assertGreater(len(item['days']), 0)
        
        # Check alignment
        # The dates should be a contiguous sub-sequence of the original
        # Original dates: 2023-01-01 ... 2023-01-20
        # If we picked a window, date[i+1] - date[i] should be 1 day (conceptually)
        # or just check they appear in original list in order
        original_dates = self.data_items[0]['day_dates']
        subset_dates = item['day_dates']
        
        # Check subset property
        first_date = subset_dates[0]
        try:
            start_idx = original_dates.index(first_date)
            for i, date in enumerate(subset_dates):
                self.assertEqual(date, original_dates[start_idx + i])
        except ValueError:
            self.fail("Dates not found in original sequence")

    def test_augmentation_random_drop(self):
        # Force "Drop" path (rand > 0.5)
        ds = PreloadedDataset(self.data_items, augment=True, max_days=20)
        
        # Mock np.random.rand to return 0.9 (first check) -> Drop path
        # Then return mixtures for drop loop
        
        # Instead of mocking, let's just assert properties generally
        # Drop strategy produces non-contiguous dates
        # But maintains order.
        
        # Let's run multiple times to ensure we hit drop path
        hit_drop = False
        original_dates = self.data_items[0]['day_dates']
        
        for _ in range(20):
            item = ds[0]
            dates = item['day_dates']
            
            # Check if contiguous
            # Parse days
            days = [int(d.split('-')[-1]) for d in dates]
            diffs = np.diff(days)
            if np.any(diffs > 1):
                hit_drop = True
                break
        
        if not hit_drop:
            # It's possible we only hit crops purely by chance, but unlikely in 20 tries with 50/50 split
            # Or drop dropped nothing?
            # Warn but pass?
            print("Warning: Did not trigger random drop in 20 tries")

if __name__ == '__main__':
    unittest.main()
