#!/usr/bin/env python3
"""
Optimized Evaluation Pipeline with Precomputed Tensors + Multi-T Analysis

Speed improvement: ~100 mins -> <5 mins
- Precompute tensors once for all test accounts
- Slice to different cutoffs for positives (no re-extraction)
- Batch GPU forward passes

Outputs:
- unified_comparison_scores.csv (per-account scores)
- ensemble_summary_metrics.csv (Lift@K + Confidence)
"""

import argparse
import logging
import os
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from scipy.stats import hypergeom
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression



# Cleanlab - confident learning for noisy labels
try:
    from cleanlab.classification import CleanLearning
    HAS_CLEANLAB = True
except ImportError:
    HAS_CLEANLAB = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Import model and data utilities
from hierarchical.models.transaction import TransactionEncoder
from hierarchical.models.day import DayEncoder
from hierarchical.models.account import AccountEncoder
# DayAutoEncoder removed for final release as it depends on external modules

from hierarchical.data.dataset import collate_hierarchical
from hierarchical.training.utils import recursive_to_device


def load_model(checkpoint_path, device, hidden_dim=128, model_type='hierarchical', args=None):
    """Load the trained model.
    
    Supports two checkpoint formats:
    1. New format: {'model_state_dict': ..., 'config': {...}}
    2. Legacy format: Just state_dict
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Detect checkpoint format
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        # New format with config
        config = checkpoint['config']
        state_dict = checkpoint['model_state_dict']
        logger.info(f"  Using checkpoint config: {config}")
        
        cat_grp_size = config['num_categories_group']
        cat_sub_size = config['num_categories_sub']
        cat_cp_size = config['num_counter_parties']
        use_balance = config.get('use_balance', True)
        use_cp = config.get('use_counter_party', True)
        d_txn = config.get('txn_dim', config.get('hidden_dim', hidden_dim))
        d_day = config.get('day_dim', config.get('hidden_dim', hidden_dim))
        d_acc = config.get('account_dim', config.get('hidden_dim', hidden_dim))
        num_layers = config.get('num_layers', 4)
        day_num_layers = config.get('day_num_layers', 2)
        num_heads = config.get('num_heads', 4)
        
        # Check for amount binning
        use_amount_binning = 'day_encoder.txn_encoder.amount_emb.weight' in state_dict
        num_amount_bins = 64
        if use_amount_binning:
            num_amount_bins = state_dict['day_encoder.txn_encoder.amount_emb.weight'].shape[0]
            logger.info(f"  Detected Amount Binning: {num_amount_bins} bins")
    else:
        # Legacy format - infer from state_dict
        state_dict = checkpoint if isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint else checkpoint.get('model_state_dict', checkpoint)
        
        prefix = 'day_encoder.txn_encoder.'
        cat_grp_size = state_dict.get(f'{prefix}cat_group_emb.weight', torch.zeros(100, 1)).shape[0]
        cat_sub_size = state_dict.get(f'{prefix}cat_sub_emb.weight', torch.zeros(100, 1)).shape[0]
        cat_cp_size = state_dict.get(f'{prefix}counter_party_emb.weight', torch.zeros(100, 1)).shape[0]
        
        use_balance = f'{prefix}balance_proj.weight' in state_dict
        use_cp = f'{prefix}counter_party_emb.weight' in state_dict
        
        use_amount_binning = f'{prefix}amount_emb.weight' in state_dict
        num_amount_bins = 64
        if use_amount_binning:
            num_amount_bins = state_dict[f'{prefix}amount_emb.weight'].shape[0]
            logger.info(f"  Detected Amount Binning: {num_amount_bins} bins")
        
        # Resolve dimensions from args or default
        d_txn = args.txn_dim if hasattr(args, 'txn_dim') and args.txn_dim else hidden_dim
        d_day = args.day_dim if hasattr(args, 'day_dim') and args.day_dim else hidden_dim
        d_acc = args.account_dim if hasattr(args, 'account_dim') and args.account_dim else hidden_dim
        
        # Detect number of layers from checkpoint keys
        max_layer_idx = 0
        for key in state_dict.keys():
            if 'transformer.layers.' in key:
                try:
                    parts = key.split('transformer.layers.')[1].split('.')
                    layer_idx = int(parts[0])
                    if layer_idx > max_layer_idx:
                        max_layer_idx = layer_idx
                except (ValueError, IndexError):
                    pass
        num_layers = max_layer_idx + 1
        day_num_layers = 2  # Default for DayEncoder
        num_heads = 4
        
        logger.info(f"  Inferred from state_dict: cat_grp={cat_grp_size}, cat_sub={cat_sub_size}, cat_cp={cat_cp_size}")
        logger.info(f"  Detected num_layers: {num_layers}")

    # 1. Transaction Encoder
    txn_encoder = TransactionEncoder(
        num_categories_group=cat_grp_size,
        num_categories_sub=cat_sub_size,
        num_counter_parties=cat_cp_size,
        embedding_dim=d_txn,
        use_balance=use_balance,
        use_counter_party=use_cp,
        use_amount_binning=use_amount_binning,
        num_amount_bins=num_amount_bins
    )
    
    # 2. Day Encoder
    day_encoder = DayEncoder(
        txn_encoder=txn_encoder,
        hidden_dim=d_day,
        num_layers=day_num_layers, 
        num_heads=num_heads
    )
    
    # 3. Account Encoder (Standard Hierarchical)
    model = AccountEncoder(
        day_encoder=day_encoder,
        hidden_dim=d_acc,
        num_layers=num_layers,
        num_heads=num_heads
    ).to(device)
    
    model.load_state_dict(state_dict)
        
    model.eval()
    return model


def slice_to_cutoff(item, cutoff_epoch_day):
    """Slice item's days to only include data before cutoff_epoch_day."""
    day_dates = item['day_dates']
    
    # Find the last day <= cutoff
    valid_indices = [i for i, d in enumerate(day_dates) if d <= cutoff_epoch_day]
    
    if not valid_indices:
        # No valid history - return minimal data
        valid_indices = [0]
    
    # Slice days
    sliced_item = {
        'account_id': item['account_id'],
        'days': [item['days'][i] for i in valid_indices],
        'day_dates': [day_dates[i] for i in valid_indices],
        'n_days': len(valid_indices)
    }
    return sliced_item

def extract_tensor_features(item):
    """
    Extract baseline features (amounts, counts) from tensor dict.
    Matching features from unified_temporal_comparison.py:
    - txn_count, n_days, total_debit, total_credit, avg_debit, n_large_debits
    """
    amounts = []
    
    # Check if item is valid
    if not item or 'days' not in item:
        return np.zeros(6, dtype=np.float32)
        
    for day in item['days']:
        # Pos (Credits)
        if 'pos' in day and day['pos'] is not None:
             amps = day['pos'].get('amounts')
             if amps is not None and len(amps) > 0:
                 if isinstance(amps, torch.Tensor):
                     amounts.append(amps.float().cpu().numpy())
                 else:
                     amounts.append(np.array(amps, dtype=np.float32))
        
        # Neg (Debits)
        if 'neg' in day and day['neg'] is not None:
             amps = day['neg'].get('amounts')
             if amps is not None and len(amps) > 0:
                 if isinstance(amps, torch.Tensor):
                     amounts.append(amps.float().cpu().numpy())
                 else:
                     amounts.append(np.array(amps, dtype=np.float32))
    
    if not amounts:
        return np.zeros(6, dtype=np.float32)
        
    all_amounts = np.concatenate(amounts)
    debits = all_amounts[all_amounts < 0]
    credits = all_amounts[all_amounts > 0]
    
    feat = np.zeros(6, dtype=np.float32)
    feat[0] = len(all_amounts) # txn_count
    feat[1] = item.get('n_days', 0) # n_days
    feat[2] = debits.sum() if len(debits) > 0 else 0 # total_debit
    feat[3] = credits.sum() if len(credits) > 0 else 0 # total_credit
    feat[4] = debits.mean() if len(debits) > 0 else 0 # avg_debit
    feat[5] = (debits < -500).sum() # n_large_debits
    
    # Extract Balance Features from the last day in the sequence
    balance_vec = np.zeros(7, dtype=np.float32)
    if 'days' in item and len(item['days']) > 0:
        # Find the last day that has balance info
        for day in reversed(item['days']):
            found = False
            for stream_key in ['pos', 'neg']:
                if stream_key in day and day[stream_key] is not None:
                    stream = day[stream_key]
                    if 'balance' in stream:
                        bal_data = stream['balance']
                        # Handle Tensor vs Numpy
                        if hasattr(bal_data, 'cpu'): bal_data = bal_data.cpu().numpy()
                        if len(bal_data) > 0:
                            balance_vec = bal_data[0] # Take first txn's balance row
                            found = True
                            break
            if found: break
            
    return np.concatenate([feat, balance_vec])

def calculate_confidence(k, N, M, n):
    """Hypergeometric confidence = 100 * (1 - p_value)."""
    if k == 0 or M == 0:
        return 0.0
    p_value = hypergeom.sf(k - 1, N, M, n)
    return 100 * (1 - p_value)



def compute_lift_curve(y_true, y_scores, ks=[1, 2, 3, 4, 5, 10, 20]):
    """Compute Lift, Recall, Precision, F1 at various K% values with confidence."""
    n = len(y_true)
    n_pos = y_true.sum()
    base_rate = n_pos / n if n > 0 else 0
    
    if n_pos == 0:
        return ({f'lift_{k}': 0.0 for k in ks}, 
                {f'conf_{k}': 0.0 for k in ks},
                {f'recall_{k}': 0.0 for k in ks},
                {f'precision_{k}': 0.0 for k in ks},
                {f'f1_{k}': 0.0 for k in ks},
                {f'n_found_{k}': 0 for k in ks},
                {f'depth_{k}': 0 for k in ks})
    
    # Sort by score descending
    sorted_idx = np.argsort(y_scores)[::-1]
    y_sorted = y_true[sorted_idx]
    
    lifts, confs, recalls, precisions, f1s, n_founds, depths = {}, {}, {}, {}, {}, {}, {}
    for k_pct in ks:
        depth = max(1, int(n * k_pct / 100))
        n_pos_at_k = y_sorted[:depth].sum()
        
        # Lift = (TP/K) / (P/N) = (TP * N) / (K * P)
        lift = (n_pos_at_k / depth) / base_rate if base_rate > 0 else 0
        conf = calculate_confidence(int(n_pos_at_k), n, int(n_pos), depth)
        
        # Recall@K = TP / P (what fraction of all positives did we capture?)
        recall = n_pos_at_k / n_pos if n_pos > 0 else 0
        
        # Precision@K = TP / K (what fraction of top K are positive?)
        precision = n_pos_at_k / depth
        
        # F1@K = harmonic mean
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        lifts[f'lift_{k_pct}'] = lift
        confs[f'conf_{k_pct}'] = conf
        recalls[f'recall_{k_pct}'] = recall
        precisions[f'precision_{k_pct}'] = precision
        f1s[f'f1_{k_pct}'] = f1
        n_founds[f'n_found_{k_pct}'] = int(n_pos_at_k)
        depths[f'depth_{k_pct}'] = depth
    
    return lifts, confs, recalls, precisions, f1s, n_founds, depths

def generate_granular_curve(y_true, y_scores, flag, T, method, is_hard):
    """Generate granular lift/recall/precision/F1 curve rows."""
    # Desired Ks: 0.1 to 3.0 (step 0.1), then 4.0 to 100.0 (step 1.0)
    ks_small = np.arange(0.1, 3.05, 0.1)
    ks_large = np.arange(4.0, 101.0, 1.0)
    ks = np.concatenate([ks_small, ks_large])
    
    n = len(y_true)
    n_pos = y_true.sum()
    base_rate = n_pos / n if n > 0 else 0
    
    if n_pos == 0: return []
    
    sorted_idx = np.argsort(y_scores)[::-1]
    y_sorted = y_true[sorted_idx]
    
    rows = []
    for k_pct in ks:
        depth = int(n * k_pct / 100)
        if depth == 0: depth = 1 # Safety
        
        n_pos_at_k = y_sorted[:depth].sum()
        lift = (n_pos_at_k / depth) / base_rate if base_rate > 0 else 0
        conf = calculate_confidence(int(n_pos_at_k), n, int(n_pos), depth)
        
        # New metrics
        recall = n_pos_at_k / n_pos if n_pos > 0 else 0
        precision = n_pos_at_k / depth
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        rows.append({
            'k_pct': k_pct,
            'lift': lift,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'confidence': conf,
            'depth': depth,
            'n_pos_at_k': n_pos_at_k,
            'n_total': n,
            'n_pos_total': int(n_pos),
            'prevalence': base_rate,
            'flag': flag,
            'T': T,
            'method': method,
            'hard_positive': is_hard
        })
    return rows
def embed_batch(model, batch_items, device):
    """Generate embeddings for a batch of items."""
    if not batch_items:
        return []
    
    # Collate into batch format
    batch = collate_hierarchical(batch_items)
    batch = recursive_to_device(batch, device)
    
    with torch.no_grad():
        embeddings = model(batch)
    
    return embeddings.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensors_path', required=True, help='Path to precomputed tensors .pt file')
    parser.add_argument('--emerging_csv', required=True, help='Path to emerging flags CSV')
    parser.add_argument('--model_checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', required=True, help='Output directory')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--txn_dim', type=int, help='Override transaction dimension')
    parser.add_argument('--day_dim', type=int, help='Override day dimension')
    parser.add_argument('--account_dim', type=int, help='Override account dimension')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test_size', type=float, default=0.3)
    parser.add_argument('--T_values', type=str, default='1,3,7', help='Comma-separated T values')
    parser.add_argument('--model_type', choices=['hierarchical', 'day_ae'], default='hierarchical', help='Model architecture type')
    parser.add_argument('--use_censored_positives', action='store_true', help='Use censored positives logic (TODO)')
    parser.add_argument('--csv_output_metrics', default='ensemble_summary_metrics.csv', help='Output CSV name for metrics')
    parser.add_argument('--csv_output_curves', default='cumulative_lift_curves.csv', help='Output CSV name for curves')
    parser.add_argument('--seeds', type=str, default='42', help='Comma-separated random seeds for train/test split (e.g. 42,123,456)')
    parser.add_argument('--negatives_in_training', type=str, default='1000,2000,5000,10000,-1', help='Comma-separated list of max train negatives values to loop over. -1 for no limit.')

    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    T_values = [int(t) for t in args.T_values.split(',')]
    logger.info(f"T values: {T_values}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load precomputed tensors
    logger.info(f"Loading precomputed tensors from {args.tensors_path}...")
    all_tensors = torch.load(args.tensors_path, weights_only=False)
    cache = {item['account_id']: item for item in all_tensors}
    logger.info(f"  Loaded {len(cache)} accounts")
    
    # 2. Load emerging flags
    logger.info(f"Loading emerging flags from {args.emerging_csv}...")
    emerging_df = pd.read_csv(args.emerging_csv)
    emerging_df['global_id'] = emerging_df['bank'] + '_' + emerging_df['accountId'].astype(str)
    emerging_df['emerging_date'] = pd.to_datetime(emerging_df['emerging_date'])
    
    # Create event lookup: account_id -> {flag: event_date}
    event_lookup = {}
    for _, row in emerging_df.iterrows():
        acc_id = row['global_id']
        if acc_id not in event_lookup:
            event_lookup[acc_id] = {}
        event_lookup[acc_id][row['flag_name']] = row['emerging_date']
    
    # Convert event dates to epoch days
    EPOCH = datetime(2020, 1, 1)
    for acc_id in event_lookup:
        for flag in event_lookup[acc_id]:
            dt = event_lookup[acc_id][flag]
            event_lookup[acc_id][flag] = (dt - EPOCH).days

    # 3. Load model
    logger.info(f"Loading model ({args.model_type}) from {args.model_checkpoint}...")
    model = load_model(args.model_checkpoint, device, args.hidden_dim, args.model_type, args)
    
    # 4. Get all flags
    all_flags = emerging_df['flag_name'].unique()
    logger.info(f"Evaluating {len(all_flags)} flags × {len(T_values)} T values")
    
    # 5. Build Global Feature Cache (Compute once for ALL accounts)
    logger.info("Building GLOBAL embedding and feature cache for ALL accounts...")
    
    all_account_ids = list(cache.keys())
    all_account_ids.sort()
    
    # Cache Negatives: (acc_id, T) -> (emb, feat) for "Hard/Random" offsets
    logger.info("  Processing potential negatives (ALL accounts with valid history)...")
    negative_cache_hard = {} 
    
    # Iterate T values
    # Filter criteria (matching primacy period from flag generation):
    MIN_CALENDAR_SPAN = 90   # Calendar days between first and last transaction
    MIN_ACTIVE_DAYS = 9      # Minimum days with at least 1 transaction
    
    for T in T_values:
        # Batch processing for efficiency
        batch_items, batch_ids = [], []
        skipped_span = 0
        skipped_active = 0
        
        # Process ALL accounts as potential negatives
        for acc_id in tqdm(all_account_ids, desc=f"Global Cache (T={T})"):
            item = cache[acc_id]
            day_dates = item['day_dates']  # List of epoch days (sorted)
            n_active_days = len(day_dates)
            
            if n_active_days == 0:
                skipped_active += 1
                continue
            
            # Calculate calendar span (last_date - first_date + 1)
            calendar_span = day_dates[-1] - day_dates[0] + 1
            
            # Filter 1: Calendar span must be >= 90 days
            if calendar_span < MIN_CALENDAR_SPAN:
                skipped_span += 1
                continue
            
            # Filter 2: Must have at least MIN_ACTIVE_DAYS with transactions
            if n_active_days < MIN_ACTIVE_DAYS:
                skipped_active += 1
                continue
            
            # Cutoff: Apply T days before end to match positive treatment
            # Positives use: event_day - T
            # Negatives use: last_day - T (simulating "predicting T days before now")
            cutoff_idx = n_active_days - T
            if cutoff_idx < 1:
                # Would have no history after cutoff
                skipped_active += 1
                continue
            
            cutoff_epoch_day = day_dates[cutoff_idx - 1]
            
            # Slice to cutoff
            item_sliced = slice_to_cutoff(cache[acc_id], cutoff_epoch_day)
            batch_items.append(item_sliced)
            batch_ids.append(acc_id)
            if acc_id not in negative_cache_hard: 
                negative_cache_hard[acc_id] = {}
            negative_cache_hard[acc_id][T] = {'feat': extract_tensor_features(item_sliced)}
        
        logger.info(f"  T={T}: Skipped {skipped_span} (span<{MIN_CALENDAR_SPAN}d), {skipped_active} (active<{MIN_ACTIVE_DAYS})")
        logger.info(f"  T={T}: Cached {len(batch_items)} accounts")

        # Embed
        for i in range(0, len(batch_items), args.batch_size):
            chunk = batch_items[i:i+args.batch_size]
            ids = batch_ids[i:i+args.batch_size]
            embs = embed_batch(model, chunk, device)
            for acc_id, emb in zip(ids, embs):
                negative_cache_hard[acc_id][T]['emb'] = emb

    # Cache Positives: Specific Cutoffs (Hard only - prediction task)
    logger.info("  Processing positives with specific cutoffs (Hard only)...")
    pos_emb_hard = {}  # (acc_id, flag, T) -> emb
    pos_feat_hard = {}
    

    
    for flag in tqdm(all_flags, desc="Flags"):
        flag_positives = emerging_df[emerging_df['flag_name'] == flag]
        
        for T in T_values:
            batch_h, ids_h = [], []
            
            for _, row in flag_positives.iterrows():
                acc_id = row['global_id']
                if acc_id not in cache: continue
                
                event_day = event_lookup.get(acc_id, {}).get(flag)
                if event_day is None: continue
                
                # Hard: Event - T (predict T days before emerging behavior)
                cut_h = event_day - T
                item_h = slice_to_cutoff(cache[acc_id], cut_h)
                batch_h.append(item_h)
                ids_h.append((acc_id, flag, T))
                pos_feat_hard[(acc_id, flag, T)] = extract_tensor_features(item_h)

            # Embed positives
            for i in range(0, len(batch_h), args.batch_size):
                embs = embed_batch(model, batch_h[i:i+args.batch_size], device)
                for k, e in zip(ids_h[i:i+args.batch_size], embs): 
                    pos_emb_hard[k] = e
    
    
    # 6. Loop over Seeds
    seeds = [int(s) for s in args.seeds.split(',')]
    logger.info(f"Running evaluation for seeds: {seeds}")
    
    all_results = []
    
    for seed in seeds:
        logger.info(f"\n[{seed}] Running seed {seed}...")
        
        # Train/Test Split
        np.random.seed(seed)
        train_accounts, test_accounts = train_test_split(all_account_ids, test_size=args.test_size, random_state=seed)
        
        # Parse negatives_in_training values
        neg_train_values = [int(x) for x in args.negatives_in_training.split(',')]
        
        import random
        
        train_ids_set = set(train_accounts) # Needed for filtering positives
        test_ids_set = set(test_accounts) # Use ALL test accounts
        
        # 7. Evaluate
        from sklearn.neural_network import MLPClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import make_pipeline
        
        # Loop over different negatives_in_training values
        for neg_train_val in neg_train_values:
            logger.info(f"  [Seed {seed}] negatives_in_training={neg_train_val}")
            
            # Subsample TRAIN negatives (only training needs subsampling)
            random.seed(seed)  # Reset seed for reproducibility per neg_train_val
            
            if neg_train_val > 0 and len(train_accounts) > neg_train_val:
                train_negative_ids = set(random.sample(train_accounts, neg_train_val))
            else:
                train_negative_ids = set(train_accounts)
            
            for flag in tqdm(all_flags, desc=f"Evaluating (Seed {seed}, negs={neg_train_val})"):
                flag_positive_ids = set(
                    row['global_id']
                    for _, row in emerging_df[emerging_df['flag_name'] == flag].iterrows()
                )
                
                for T in T_values:
                    # --- Build Training Set ---
                    X_train_emb, y_train = [], []
                    X_train_feat = []
                
                    # Train negatives are handled in the loop below (lines 620-629)

                    # Build from positives
                    pos_count = 0
                    for acc_id in flag_positive_ids:
                        if acc_id not in train_ids_set: continue # Must be in train split
                        
                        key = (acc_id, flag, T)
                        if key in pos_emb_hard:
                            X_train_emb.append(pos_emb_hard[key])
                            X_train_feat.append(pos_feat_hard[key])
                            y_train.append(1)
                            pos_count += 1
                    
                    # Build from negatives (using subsampled train list)
                    neg_count = 0
                    for acc_id in train_negative_ids:
                        # Skip if it is actually positive for this flag
                        if acc_id in flag_positive_ids: continue 
                        
                        if acc_id in negative_cache_hard and T in negative_cache_hard[acc_id]:
                            X_train_emb.append(negative_cache_hard[acc_id][T]['emb'])
                            X_train_feat.append(negative_cache_hard[acc_id][T]['feat'])
                            y_train.append(0)
                            neg_count += 1
                
                    if len(y_train) < 50 or pos_count < 10:
                        continue

                    X_train_emb = np.array(X_train_emb)
                    X_train_feat = np.array(X_train_feat)
                    y_train = np.array(y_train)
                    
                    # Train Models
                    n_pos = sum(y_train)
                    n_neg = len(y_train) - n_pos
                    scale = n_neg / n_pos if n_pos > 0 else 1
                    
                    # 1. XGB on Embeddings
                    clf_xgb_emb = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, scale_pos_weight=scale, random_state=seed, verbosity=0)
                    clf_xgb_emb.fit(X_train_emb, y_train)
                    
                    # 2. XGB on Features
                    clf_xgb_feat = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, scale_pos_weight=scale, random_state=seed, verbosity=0)
                    clf_xgb_feat.fit(X_train_feat, y_train)
                    
                    # 3. XGB on Balance
                    clf_xgb_bal = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, scale_pos_weight=scale, random_state=seed, verbosity=0)
                    has_bal = X_train_feat.shape[1] > 7
                    if has_bal:
                        clf_xgb_bal.fit(X_train_feat[:, -7:], y_train)
                    
                    # 4. XGB Hybrid (Emb + Feat)
                    clf_xgb_hybrid = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, scale_pos_weight=scale, random_state=seed, verbosity=0)
                    X_train_hybrid = np.concatenate([X_train_emb, X_train_feat], axis=1)
                    clf_xgb_hybrid.fit(X_train_hybrid, y_train)

                    # 5. Linear
                    clf_lin_emb = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed)
                    clf_lin_emb.fit(X_train_emb, y_train)
                    

                    # 7. CleanLearning (cleanlab) - confident learning for noisy labels
                    clf_pul_emb, clf_pul_feat = None, None
                    if HAS_CLEANLAB:
                        try:
                            base_lr_emb = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed, n_jobs=-1)
                            clf_pul_emb = CleanLearning(clf=base_lr_emb)
                            clf_pul_emb.fit(X_train_emb, y_train)
                            
                            base_lr_feat = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed, n_jobs=-1)
                            clf_pul_feat = CleanLearning(clf=base_lr_feat)
                            clf_pul_feat.fit(X_train_feat, y_train)
                        except Exception as e:
                            pass  # CleanLearning failed, continue without it
                    

                    
                    # --- Build Test Set ---
                    X_test_emb, X_test_feat, y_test = [], [], []
                
                    # Use ALL test accounts
                    for acc_id in test_accounts:
                        is_pos = acc_id in flag_positive_ids
                        if is_pos:
                            key = (acc_id, flag, T)
                            if key in pos_emb_hard:
                                X_test_emb.append(pos_emb_hard[key])
                                X_test_feat.append(pos_feat_hard[key])
                                y_test.append(1)
                        else:
                            # Negative
                            if acc_id in negative_cache_hard and T in negative_cache_hard[acc_id]:
                                X_test_emb.append(negative_cache_hard[acc_id][T]['emb'])
                                X_test_feat.append(negative_cache_hard[acc_id][T]['feat'])
                                y_test.append(0)

                    # Convert to numpy
                    X_test_emb = np.array(X_test_emb)
                    X_test_feat = np.array(X_test_feat)
                    y_test = np.array(y_test)

                    # Evaluate Helper
                    def evaluate_method(method_name, clf, X_emb, X_feat):
                        # Select correct features
                        if 'Feat' in method_name:
                            X = X_feat
                        elif 'Hybrid' in method_name:
                            X = np.concatenate([X_emb, X_feat], axis=1)
                        elif 'Bal' in method_name:
                            if not has_bal: return None
                            X = X_feat[:, -7:]
                        else:
                            X = X_emb
                        
                        if len(X) == 0: return None
                        probs = clf.predict_proba(X)[:, 1]
                        try: auc = roc_auc_score(y_test, probs)
                        except: auc = 0.5
                        lifts, confs, recalls, precisions, f1s, n_founds, depths = compute_lift_curve(y_test, probs)
                        
                        # Store seed info
                        n_total = len(y_test)
                        n_pos_test = int(sum(y_test))
                        prevalence = n_pos_test / n_total if n_total > 0 else 0
                        
                        metrics = {
                            'seed': seed,
                            'negatives_in_training': neg_train_val,
                            'flag': flag, 'T': T, 'method': method_name,
                            'auc': auc, 
                            'n_pos': n_pos_test,
                            'n_total': n_total,
                            'prevalence': prevalence
                        }
                        for k, v in lifts.items(): metrics[k] = v
                        for k, v in confs.items(): metrics[k] = v
                        for k, v in recalls.items(): metrics[k] = v
                        for k, v in precisions.items(): metrics[k] = v
                        for k, v in f1s.items(): metrics[k] = v
                        for k, v in n_founds.items(): metrics[k] = v
                        for k, v in depths.items(): metrics[k] = v
                        return metrics

                    # Run for all methods
                    methods = [
                        ('XGB_Emb', clf_xgb_emb),
                        ('XGB_Feat', clf_xgb_feat),
                        ('XGB_Hybrid', clf_xgb_hybrid),
                        ('XGB_Bal', clf_xgb_bal),
                        ('Linear_Emb', clf_lin_emb)
                    ]
                    
                    # Add PU Learning methods if available (CleanLearning)
                    if HAS_CLEANLAB and clf_pul_emb is not None:
                        methods.append(('PUL_Emb', clf_pul_emb))
                    if HAS_CLEANLAB and clf_pul_feat is not None:
                        methods.append(('PUL_Feat', clf_pul_feat))
                    
                    # Add ElkanotoPu if available
                    if HAS_PULEARN and clf_epu_emb is not None:
                        methods.append(('EPU_Emb', clf_epu_emb))
                    
                    for m_name, clf in methods:
                        if clf is None: continue
                        if m_name == 'XGB_Bal' and not has_bal: continue
                        res = evaluate_method(m_name, clf, X_test_emb, X_test_feat)
                        if res: all_results.append(res)

    # 8. Save results
    logger.info("Saving results...")
    if not all_results:
        logger.warning("No results generated! Check intersections of flags and tensors.")
        return

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / args.csv_output_metrics, index=False)
    
    logger.info("\n" + "="*80)
    logger.info("SUMMARY (Averaged across seeds)")
    logger.info("="*80)
    logger.info(f"{'Flag':<30} {'T':>2} {'Method':<10} {'AUC':>6} {'L@5':>7} {'L@10':>7}")
    logger.info("-"*80)
    
    # Aggregate for display
    summary_df = results_df.groupby(['flag', 'T', 'method'])[['auc', 'lift_5', 'lift_10']].mean().reset_index()
    
    for _, r in summary_df.sort_values(['flag', 'T', 'method']).iterrows():
        auc = r['auc']
        l5 = r['lift_5']
        l10 = r['lift_10']
        logger.info(f"{r['flag']:<30} {r['T']:>2} {r['method']:<10} {auc:>6.3f} {l5:>6.2f}x {l10:>6.2f}x")

if __name__ == '__main__':
    main()
