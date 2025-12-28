#!/usr/bin/env python3
"""Debug NaN in evaluation embeddings."""
import torch
import numpy as np
from pathlib import Path

# Load model and run inference on a few accounts
from hierarchical.models.transaction import TransactionEncoder
from hierarchical.models.day import DayEncoder
from hierarchical.models.account import AccountEncoder
from hierarchical.data.dataset import collate_hierarchical
from hierarchical.training.utils import recursive_to_device

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load checkpoint
ckpt_path = Path("results/exp_e2e_v1/model/model_best.pth")
print(f"Loading checkpoint: {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
state_dict = checkpoint if isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint else checkpoint.get('model_state_dict', checkpoint)

# Infer sizes
prefix = 'day_encoder.txn_encoder.'
cat_grp_size = state_dict.get(f'{prefix}cat_group_emb.weight', torch.zeros(100, 1)).shape[0]
cat_sub_size = state_dict.get(f'{prefix}cat_sub_emb.weight', torch.zeros(100, 1)).shape[0]
cat_cp_size = state_dict.get(f'{prefix}counter_party_emb.weight', torch.zeros(100, 1)).shape[0]
use_balance = f'{prefix}balance_proj.weight' in state_dict
use_cp = f'{prefix}counter_party_emb.weight' in state_dict

print(f"Vocab sizes: cat_grp={cat_grp_size}, cat_sub={cat_sub_size}, cat_cp={cat_cp_size}")
print(f"use_balance={use_balance}, use_cp={use_cp}")

# Build model
hidden_dim = 256
txn_encoder = TransactionEncoder(
    num_categories_group=cat_grp_size,
    num_categories_sub=cat_sub_size,
    num_counter_parties=cat_cp_size,
    embedding_dim=hidden_dim,
    use_balance=use_balance,
    use_counter_party=use_cp
)

day_encoder = DayEncoder(txn_encoder=txn_encoder, hidden_dim=hidden_dim, num_layers=2, num_heads=4)
model = AccountEncoder(day_encoder=day_encoder, hidden_dim=hidden_dim, num_layers=4, num_heads=4).to(device)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded successfully")

# Load a few tensors
print("\nLoading tensors...")
data = torch.load('results/exp_e2e_v1/data/pretrain_tensors.pt', weights_only=False)
print(f"Loaded {len(data)} accounts")

# Test with first 10 accounts
test_items = data[:10]

# Check for NaN in input data
print("\n=== Checking input data for NaN ===")
for item in test_items:
    acc_id = item['account_id']
    for i, day in enumerate(item['days']):
        for stream in ['pos', 'neg']:
            if day.get(stream):
                for key in ['amounts', 'balance']:
                    val = day[stream].get(key)
                    if val is not None:
                        arr = np.array(val) if not hasattr(val, 'cpu') else val.cpu().numpy()
                        if np.isnan(arr).any():
                            print(f"  NaN in {acc_id} day {i} {stream}.{key}")

# Collate and embed
print("\n=== Running inference ===")
batch = collate_hierarchical(test_items)
batch = recursive_to_device(batch, device)

with torch.no_grad():
    embeddings = model(batch)

emb_np = embeddings.cpu().numpy()
print(f"Embeddings shape: {emb_np.shape}")
print(f"Has NaN: {np.isnan(emb_np).any()}")
print(f"NaN count: {np.isnan(emb_np).sum()}")

if np.isnan(emb_np).any():
    print("\nAccounts with NaN embeddings:")
    for i, item in enumerate(test_items):
        if np.isnan(emb_np[i]).any():
            print(f"  {item['account_id']}: {np.isnan(emb_np[i]).sum()} NaNs")
else:
    print("\nNo NaN in first 10 accounts - testing more...")
    
    # Test with larger batch
    test_items = data[:100]
    batch = collate_hierarchical(test_items)
    batch = recursive_to_device(batch, device)
    with torch.no_grad():
        embeddings = model(batch)
    emb_np = embeddings.cpu().numpy()
    print(f"100 accounts - Has NaN: {np.isnan(emb_np).any()}")
