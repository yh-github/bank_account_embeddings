#!/usr/bin/env python3
"""
Offline Tensor Pre-processing

Converts raw transaction CSVs into pre-computed PyTorch tensors (.pt files).
This allows the GPU evaluation script to load data constantly without CPU bottlenecks.

Algorithm:
1. Load transactions
2. Pre-compute balance cache (shared memory)
3. Multiprocessing:
   - Group by account
   - Extract features (Categories, Amounts, Dates, Balance)
   - Create TensorDicts
4. Save to disk (e.g. 1 file per flag or sharded)
"""

import os
import argparse
import logging
import json
import pickle
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from datetime import datetime

# Reuse logic from fast eval
from hierarchical.data.vocab import load_vocabularies, CategoricalVocabulary, build_vocabularies
from hierarchical.data.loader import load_transactions, load_accounts
from hierarchical.data.balance import BalanceFeatureExtractor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for workers
_VOCABS = None
_BALANCE_CACHE = None
_CONFIG = None

def patch_df(df):
    """Ensure schema compatibility for BalanceCalculator and others."""
    cols = df.columns
    if 'id' not in cols and 'trId' in cols:
        df['id'] = df['trId']
    if 'direction' not in cols and 'amount' in cols:
        # Simple heuristic
        df['direction'] = df['amount'].apply(lambda x: 'Credit' if x > 0 else 'Debit')
    return df

def init_worker(vocabs, balance_cache, config):
    global _VOCABS, _BALANCE_CACHE, _CONFIG
    _VOCABS = vocabs
    _BALANCE_CACHE = balance_cache
    _CONFIG = config

def process_account_chunk(chunk_data):
    """
    Process a list of (account_id, transactions_df) tuples.
    Returns a list of tensor dicts.
    """
    cat_group_vocab, cat_sub_vocab, counter_party_vocab = _VOCABS
    max_days = _CONFIG['max_days']
    max_txns = _CONFIG['max_txns']
    
    results = []
    
    for account_id, df in chunk_data:
        # Group by day
        unique_days = sorted(df['date_only'].unique())
        days = unique_days[-max_days:]
        
        day_features = []
        for day in days:
            day_df = df[df['date_only'] == day]
            
            # Meta
            date_val = pd.to_datetime(day_df['date'].iloc[0])
            day_month = date_val.month
            day_weekend = 1 if date_val.dayofweek >= 5 else 0
            
            # Balance (from cache)
            cache_key = (account_id.upper(), day)
            base_bal = _BALANCE_CACHE.get(cache_key, np.zeros(7, dtype=np.float32))
            
            # Split Pos/Neg
            pos_df = day_df[day_df['amount'] > 0].head(max_txns)
            neg_df = day_df[day_df['amount'] <= 0].head(max_txns)
            
            streams = {}
            for name, sub_df in [('pos', pos_df), ('neg', neg_df)]:
                n = len(sub_df)
                if n == 0:
                    streams[name] = None
                    continue
                
                # Helpers
                def get_col(d, candidates, default='UNK'):
                    for c in candidates:
                        if c in d.columns: return d[c].tolist()
                    return [default] * len(d)
                
                # Encode
                c_grp = [cat_group_vocab.encode(x) for x in get_col(sub_df, ['personeticsCategoryGroupId', 'p_categoryGroupId'])]
                c_sub = [cat_sub_vocab.encode(x) for x in get_col(sub_df, ['personeticsSubCategoryId', 'p_subCategoryId'])]
                c_cp = [counter_party_vocab.encode(x) for x in get_col(sub_df, ['deviceId', 'counter_party'])]
                
                # Amounts
                amounts = sub_df['amount'].values.astype(np.float32)
                sign = np.sign(amounts)
                log_amounts = np.log1p(np.abs(amounts))
                norm_amounts = (log_amounts - log_amounts.mean()) / (log_amounts.std() + 1e-9)
                norm_amounts = (log_amounts - log_amounts.mean()) / (log_amounts.std() + 1e-9)
                amounts_feat = (sign * norm_amounts)
                
                # Store absolute magnitude for aux loss
                # Sum of absolute amounts (Total Volume)
                total_volume = np.sum(np.abs(amounts))
                log_total_volume = np.log1p(total_volume).astype(np.float32)
                
                # Dates
                sub_dates = pd.to_datetime(sub_df['date'])
                date_feats = np.stack([
                    sub_dates.dt.dayofweek.values,
                    sub_dates.dt.day.values - 1,
                    sub_dates.dt.month.values - 1,
                    (sub_dates - datetime(2020, 1, 1)).dt.days.values / 365.0
                ], axis=1).astype(np.float32)
                
                # Balance
                bal_expanded = np.tile(base_bal, (n, 1)).astype(np.float32)
                
                streams[name] = {
                    'cat_group': c_grp, # Keep as list for now
                    'cat_sub': c_sub,
                    'cat_cp': c_cp,
                    'amounts': amounts_feat,
                    'dates': date_feats, # numpy
                    'balance': bal_expanded, # numpy
                    'dates': date_feats, # numpy
                    'balance': bal_expanded, # numpy
                    'n_txns': n,
                    'log_total_volume': log_total_volume # Scalar target
                }
            
            day_features.append({
                'meta': {'month': day_month, 'weekend': day_weekend},
                'pos': streams.get('pos'),
                'neg': streams.get('neg')
            })
            
        from datetime import date as datelib
        epoch = datelib(2020, 1, 1)
        results.append({
            'account_id': account_id, 
            'days': day_features, 
            'n_days': len(day_features),
            'day_dates': [(d - epoch).days for d in days] # Store as int offsets
        })
        
    return results

def precompute_balance_cache(df: pd.DataFrame, balance_extractor) -> dict:
    # Same logic as before
    logger.info("Pre-computing balance cache...")
    df['date_only'] = pd.to_datetime(df['date']).dt.date
    unique_pairs = df[['accountId', 'date_only']].drop_duplicates()
    
    cache = {}
    forCtx = tqdm(unique_pairs.iterrows(), total=len(unique_pairs), desc="Balance Cache")
    for _, row in forCtx:
        acc_id = str(row['accountId']).upper()
        date_val = row['date_only']
        try:
            end_date = date_val - pd.Timedelta(days=1)
            start_date = end_date - pd.Timedelta(days=7)
            stats = balance_extractor.get_window_stats(acc_id, start_date, end_date)
            starting_bal = balance_extractor.get_starting_balance(acc_id, date_val) or 0.0
            
            feats = np.array([
                balance_extractor.normalize_balance(stats.get('start_balance')),
                balance_extractor.normalize_balance(stats.get('end_balance')),
                balance_extractor.normalize_balance(stats.get('min_balance')),
                balance_extractor.normalize_balance(stats.get('max_balance')),
                balance_extractor.normalize_balance(stats.get('avg_balance')),
                balance_extractor.normalize_balance(stats.get('balance_volatility')),
                balance_extractor.normalize_balance(starting_bal)
            ], dtype=np.float32)
            cache[(acc_id, date_val)] = feats
        except:
             cache[(acc_id, date_val)] = np.zeros(7, dtype=np.float32)
    return cache

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval'], default='eval')
    parser.add_argument('--data_file', help="For train mode: path to input CSV")
    parser.add_argument('--eval_data_dir', help="For eval mode: path to eval directory")
    parser.add_argument('--model_checkpoint', required=False, help="Path to existing model (to reuse vocab)")
    parser.add_argument('--vocab_dir', required=False, help="Directory to load/save vocabularies")
    parser.add_argument('--account_file', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--min_days', type=int, default=10)
    parser.add_argument('--censor_subcat', help="Optional: Censor (remove) specific subCategory ID from inputs")
    parser.add_argument('--max_days', type=int, default=180)
    parser.add_argument('--max_txns', type=int, default=20)
    args = parser.parse_args()
    
    # 1. Vocabs
    vocabs = None
    if args.model_checkpoint:
        vocab_dir = str(Path(args.model_checkpoint).parent / "vocabularies")
        logger.info(f"Loading vocabs from checkpoint dir: {vocab_dir}")
        vocabs = load_vocabularies(vocab_dir)
    elif args.vocab_dir:
        # Try load
        try:
            logger.info(f"Loading vocabs from {args.vocab_dir}...")
            vocabs = load_vocabularies(args.vocab_dir)
        except Exception:
            logger.info("Vocabs not found. Building...")
            # Need to load data to build vocab
            if not args.data_file: raise ValueError("--data_file required to build vocabs")
            df = load_transactions(args.data_file)
            df = patch_df(df)
            vocabs = build_vocabularies(df, args.vocab_dir)
    else:
        raise ValueError("Must provide --model_checkpoint or --vocab_dir")
    
    # 2. Balance Cache
    cache_source = args.data_file if args.mode == 'train' else str(Path(args.eval_data_dir).parent / "pretrain" / "pretrain_transactions.csv")
    
    if os.path.exists(cache_source):
        df_base = load_transactions(cache_source)
        df_base = patch_df(df_base)
        df_base['accountId'] = df_base['accountId'].astype(str)
        df_acc = load_accounts(args.account_file)
        extractor = BalanceFeatureExtractor(df_base, df_acc)
        balance_cache = precompute_balance_cache(df_base, extractor)
    else:
        logger.warning(f"Cache source {cache_source} not found. Balance features may be zero.")
        balance_cache = {}

    config = {'max_days': args.max_days, 'max_txns': args.max_txns}

    # Helper function to process a single DF
    def process_dataframe(df, output_path):
        # Apply Censorship
        if args.censor_subcat:
            init_len = len(df)
            
            # Normalize target: Strip 'C', 'c' or 'CG' prefix to match raw IDs (often ints)
            target = str(args.censor_subcat).upper().lstrip('C').lstrip('G')
            
            # Robust filtering for both schema versions
            col_name = None
            if 'personeticsSubCategoryId' in df.columns:
                 col_name = 'personeticsSubCategoryId'
            elif 'p_subCategoryId' in df.columns:
                 col_name = 'p_subCategoryId'
                 
            if col_name:
                 # Helper to normalize series
                 # Ensure we compare str vs str or handle mixed types
                 # Safe way: cast both to string and strip 'C'
                 series_str = df[col_name].astype(str).str.upper().str.lstrip('C').str.lstrip('G')
                 mask = series_str != target
                 df = df[mask]
            else:
                 logger.warning("Could not find subCategory column for censorship!")
                 
            drop_len = init_len - len(df)
            logger.info(f"  CENSORED {drop_len} transactions matching {args.censor_subcat} (Norm: {target})")

        logger.info(f"Processing {len(df)} transactions...")
        df['accountId'] = df['accountId'].astype(str)
        df['date'] = pd.to_datetime(df['date'])
        df['date_only'] = df['date'].dt.date
        df = df.sort_values(['accountId', 'date'])
        
        groups = []
        for acc_id, group in df.groupby('accountId'):
            if group['date_only'].nunique() >= args.min_days:
                groups.append((acc_id, group))
        
        if not groups:
            logger.warning("No accounts passed min_days filter!")
            return

        chunk_size = 100
        chunks = [groups[i:i + chunk_size] for i in range(0, len(groups), chunk_size)]
        
        logger.info(f"  {len(groups)} accounts, {len(chunks)} chunks. Using {cpu_count()} cores.")
        
        all_results = []
        with Pool(cpu_count(), initializer=init_worker, initargs=(vocabs, balance_cache, config)) as pool:
            for res in tqdm(pool.imap(process_account_chunk, chunks), total=len(chunks)):
                all_results.extend(res)
                
        # Save metadata
        metadata = vars(args)
        metadata_file = Path(args.output_dir) / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"  Saved metadata to {metadata_file}")

        logger.info(f"  Saving {len(all_results)} accounts to {output_path}")
        torch.save(all_results, output_path)

    # 3. Execution
    if args.mode == 'train':
        if not args.data_file: raise ValueError("--data_file required for train mode")
        output_file = Path(args.output_dir) / "pretrain_tensors.pt"
        df = load_transactions(args.data_file)
        df = patch_df(df)
        process_dataframe(df, output_file)
        
    else: # Eval
        if not args.eval_data_dir: raise ValueError("--eval_data_dir required for eval mode")
        eval_path = Path(args.eval_data_dir)
        flags = sorted([d for d in eval_path.iterdir() if d.is_dir()])
        os.makedirs(args.output_dir, exist_ok=True)
        
        for flag_dir in flags:
            flag_name = flag_dir.name
            txn_file = flag_dir / "transactions.csv"
            if not txn_file.exists(): continue
            
            output_file = Path(args.output_dir) / f"{flag_name}_tensors.pt"
            if output_file.exists():
                logger.info(f"Skipping {flag_name} (exists)")
                continue
                
            logger.info(f"Processing {flag_name}...")
            df = load_transactions(txn_file)
            df = patch_df(df)
            process_dataframe(df, output_file)

if __name__ == '__main__':
    main()
