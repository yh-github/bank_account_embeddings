"""Preloaded Dataset for efficiency."""

from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class PreloadedDataset(Dataset):
    """Dataset from pre-loaded data (list or .pt file)."""

    def __init__(
        self, 
        data_source: str | list, 
        augment: bool = False, 
        max_days: int = 360, 
        max_txns_per_day: int = 150
    ) -> None:
        """Initializes the PreloadedDataset.

        Args:
            data_source: Path to .pt file or list of data items.
            augment: Whether to apply augmentation.
            max_days: Maximum days to include.
            max_txns_per_day: Maximum transactions per day.
        """
        if isinstance(data_source, str):
            print(f"Loading preprocessed data from {data_source}...")
            self.data = torch.load(data_source, weights_only=False)
        else:
            self.data = data_source

        self.augment = augment
        self.max_days = max_days
        self.max_txns_per_day = max_txns_per_day

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        start_item = self.data[idx]

        # Create a fresh container for the return value
        item = {
            'account_id': start_item['account_id'],
            'days': start_item['days'],  # Reference for now
            'day_dates': start_item['day_dates']
        }

        # 2. Augment (or just slice)
        if self.augment:
            self.apply_augmentation(item)
        else:
            # Just Slice max_days if preproc was larger
            if len(item['days']) > self.max_days:
                item['days'] = item['days'][-self.max_days:]
                item['day_dates'] = item['day_dates'][-self.max_days:]

        return item

    def apply_augmentation(self, item: dict) -> None:
        """Applies random augmentation to the item."""
        days = item['days']  # Reference
        n_days = len(days)

        if n_days < 5:
            return  # No aug

        # Strategy: 50% Crop, 50% Random Drop
        if np.random.rand() < 0.5:
            # Temporal Crop
            window_size = max(5, int(n_days * np.random.uniform(0.6, 0.9)))
            window_size = min(window_size, self.max_days)  # Cap at Model Context

            start = np.random.randint(0, n_days - window_size + 1)

            # Create NEW list slice (safe)
            item['days'] = days[start:start + window_size]
            # Ensure day_dates is sliced similarly
            if isinstance(item['day_dates'], list):
                item['day_dates'] = item['day_dates'][start:start + window_size]
            else:
                 item['day_dates'] = item['day_dates'][start:start + window_size]


        else:
            # Random Drop
            indices_to_keep = []
            for i in range(n_days):
                if np.random.rand() > 0.2:
                    indices_to_keep.append(i)

            if not indices_to_keep:
                indices_to_keep = [0, 1]

            # Slice to max_days if still too big
            if len(indices_to_keep) > self.max_days:
                indices_to_keep = indices_to_keep[-self.max_days:]

            item['days'] = [days[i] for i in indices_to_keep]
            
            if isinstance(item['day_dates'], list):
                 item['day_dates'] = [item['day_dates'][i] for i in indices_to_keep]
            else:
                 item['day_dates'] = item['day_dates'][indices_to_keep]
