#!/usr/bin/env python3
"""Refactored Evaluation Pipeline.

Modularized version of unified_eval_optimized.py.
"""

import argparse
import logging
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from xgboost import XGBClassifier

# Optional imports
try:
    from cleanlab.classification import CleanLearning
    HAS_CLEANLAB = True
except ImportError:
    HAS_CLEANLAB = False

try:
    from pulearn import BaggingPuClassifier
    HAS_PULEARN = True
except ImportError:
    HAS_PULEARN = False

# Import local modules
from hierarchical.evaluation.unified_eval_optimized import (
    compute_lift_curve,
    embed_batch as embed_batch_hierarchical,
    extract_tensor_features,
    slice_to_cutoff,
)
from hierarchical.models.account import AccountEncoder
from hierarchical.models.day import DayEncoder
from hierarchical.models.transaction import TransactionEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- 1. Model Loading ---
def load_model(
    checkpoint_path: str,
    device: str,
    hidden_dim: int = 128,
    num_heads: int = 4,
    num_layers: int = None,
    model_type: str = 'hierarchical'
) -> torch.nn.Module:
    """Load the trained model (Hierarchical)."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint if isinstance(checkpoint, dict) and 'model_state_dict' not in checkpoint else checkpoint.get('model_state_dict', checkpoint)
    
    # Infer prefix
    prefix = 'day_encoder.txn_encoder.'
    if model_type == 'day_ae':
        raise ValueError("DayAutoEncoder not supported in this minimal release.")

    # Infer sizes
    cat_grp_size = state_dict.get(f'{prefix}cat_group_emb.weight', torch.zeros(100, 1)).shape[0]
    cat_sub_size = state_dict.get(f'{prefix}cat_sub_emb.weight', torch.zeros(100, 1)).shape[0]
    cat_cp_size = state_dict.get(f'{prefix}counter_party_emb.weight', torch.zeros(100, 1)).shape[0]
    use_balance = f'{prefix}balance_proj.weight' in state_dict
    use_cp = f'{prefix}counter_party_emb.weight' in state_dict
    
    use_amount_binning = f'{prefix}amount_emb.weight' in state_dict
    num_amount_bins = 64
    if use_amount_binning:
        num_amount_bins = state_dict[f'{prefix}amount_emb.weight'].shape[0]

    # Detect Dimensions
    # Txn Dim (Input to DayEncoder)
    # The output of TransactionEncoder is determined by `norm.weight`
    txn_dim_weight = state_dict.get(f'{prefix}norm.weight')
    if txn_dim_weight is not None:
        txn_dim = txn_dim_weight.shape[0]
    else:
        txn_dim = hidden_dim # Fallback
        
    # Day Dim (Input to AccountEncoder)
    # Output of DayEncoder determined by `day_encoder.norm.weight`
    day_dim_weight = state_dict.get('day_encoder.norm.weight')
    if day_dim_weight is not None:
        day_dim = day_dim_weight.shape[0]
    else:
        day_dim = hidden_dim

    # Account Dim (Final Output)
    # Determined by `norm.weight`
    acc_dim_weight = state_dict.get('norm.weight')
    if acc_dim_weight is not None:
        acc_dim = acc_dim_weight.shape[0]
    else:
        acc_dim = hidden_dim
        
    logger.info(f"  Detected Dims: Txn={txn_dim} -> Day={day_dim} -> Account={acc_dim}")

    # Initialize Modules
    txn_encoder = TransactionEncoder(
        num_categories_group=cat_grp_size,
        num_categories_sub=cat_sub_size,
        num_counter_parties=cat_cp_size,
        embedding_dim=txn_dim,
        use_balance=use_balance,
        use_counter_party=use_cp,
        use_amount_binning=use_amount_binning,
        num_amount_bins=num_amount_bins
    )
    
    # Detect Layers Independently
    if num_layers is None:
        # 1. Day Encoder Layers
        max_day_layer = 0
        has_day_layers = False
        for key in state_dict.keys():
            if 'day_encoder.transformer.layers.' in key:
                has_day_layers = True
                try:
                    parts = key.split('day_encoder.transformer.layers.')[1].split('.')
                    layer_idx = int(parts[0])
                    if layer_idx > max_day_layer: max_day_layer = layer_idx
                except Exception: pass
        
        day_layers = max_day_layer + 1 if has_day_layers else 2 
        
        # 2. Account Encoder Layers (Root transformer)
        max_acc_layer = 0
        has_acc_layers = False
        for key in state_dict.keys():
            if key.startswith('transformer.layers.'):
                has_acc_layers = True
                try:
                    parts = key.split('transformer.layers.')[1].split('.')
                    layer_idx = int(parts[0])
                    if layer_idx > max_acc_layer: max_acc_layer = layer_idx
                except Exception: pass
        
        acc_layers = max_acc_layer + 1 if has_acc_layers else 2
        
        logger.info(f"Auto-detected layers -> Day: {day_layers}, Account: {acc_layers}")
    else:
        day_layers = num_layers
        acc_layers = num_layers
        logger.info(f"Using specified layers: {day_layers} (Global)")

    day_encoder = DayEncoder(
        txn_encoder=txn_encoder,
        hidden_dim=day_dim,
        num_layers=day_layers, 
        num_heads=num_heads
    )
    
    model = AccountEncoder(
        day_encoder=day_encoder,
        hidden_dim=acc_dim,
        num_layers=acc_layers,
        num_heads=num_heads
    ).to(device)
    
    # Load State Dict
    try:
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        logger.warning(f"Strict load failed: {e}. Retrying with strict=False")
        model.load_state_dict(state_dict, strict=False)
        
    model.eval()
    return model


# --- 2. Caching Logic ---
def cache_embeddings(
    model: torch.nn.Module,
    cache: dict,
    T_values: list[int],
    all_flags: np.ndarray,
    emerging_df: pd.DataFrame,
    batch_size: int,
    device: str
) -> tuple[dict, dict, dict]:
    """Computes and caches embeddings for Negatives and Positives."""
    # Create event lookup
    event_lookup: dict[str, dict[str, int]] = {}
    EPOCH = datetime(2020, 1, 1)

    for _, row in emerging_df.iterrows():
        acc_id = row['global_id'].split('_', 1)[1] if '_' in row['global_id'] else row['global_id']
        if acc_id not in event_lookup: event_lookup[acc_id] = {}
        dt = pd.to_datetime(row['emerging_date'])
        event_lookup[acc_id][row['flag_name']] = (dt - EPOCH).days

    # Cache Negatives
    negative_cache_hard: dict[str, dict[int, dict]] = {}
    all_account_ids = list(cache.keys())
    
    MIN_CALENDAR_SPAN = 90
    MIN_ACTIVE_DAYS = 9
    
    logger.info("Computing Global Negative Cache...")
    for T in T_values:
        batch_items, batch_ids = [], []
        
        for acc_id in tqdm(all_account_ids, desc=f"Negatives (T={T})"):
            item = cache[acc_id]
            day_dates = item.get('day_dates', [])
            if not isinstance(day_dates, (list, np.ndarray, torch.Tensor)) or len(day_dates) == 0:
                continue
            
            # Assuming day_dates sorted
            if isinstance(day_dates, torch.Tensor):
                first_date = int(day_dates[0])
                last_date = int(day_dates[-1])
            else:
                first_date = day_dates[0]
                last_date = day_dates[-1]

            span = last_date - first_date + 1
            if span < MIN_CALENDAR_SPAN or len(day_dates) < MIN_ACTIVE_DAYS:
                continue
            
            # Slice T days from end
            cutoff_idx = len(day_dates) - T
            if cutoff_idx <= 0: continue
            
            cutoff_epoch_day = day_dates[cutoff_idx - 1]
            item_sliced = slice_to_cutoff(item, int(cutoff_epoch_day))
            
            batch_items.append(item_sliced)
            batch_ids.append(acc_id)
            
            if acc_id not in negative_cache_hard: negative_cache_hard[acc_id] = {}
            negative_cache_hard[acc_id][T] = {'feat': extract_tensor_features(item_sliced)}
            
        # Embed Batch
        for i in range(0, len(batch_items), batch_size):
            chunk = batch_items[i:i+batch_size]
            ids = batch_ids[i:i+batch_size]
            embs = embed_batch_hierarchical(model, chunk, device)
            for acc_id, emb in zip(ids, embs):
                negative_cache_hard[acc_id][T]['emb'] = emb
                
    # Cache Positives
    pos_emb_hard = {}
    pos_feat_hard = {}
    
    logger.info("Computing Specific Positive Cache...")
    for flag in tqdm(all_flags, desc="Flags"):
        flag_positives = emerging_df[emerging_df['flag_name'] == flag]
        for T in T_values:
            batch_h, ids_h = [], []
            for _, row in flag_positives.iterrows():
                acc_id = row['global_id'].split('_', 1)[1] if '_' in row['global_id'] else row['global_id']
                if acc_id not in cache: continue
                
                event_day = event_lookup.get(acc_id, {}).get(flag)
                if event_day is None: continue
                
                # Cutoff: Event - T
                cut_h = event_day - T
                item_h = slice_to_cutoff(cache[acc_id], cut_h)
                batch_h.append(item_h)
                ids_h.append((acc_id, flag, T))
                pos_feat_hard[(acc_id, flag, T)] = extract_tensor_features(item_h)
            
            # Embed
            for i in range(0, len(batch_h), batch_size):
                chunk = batch_h[i:i+batch_size]
                ids_chunk = ids_h[i:i+batch_size]
                embs = embed_batch_hierarchical(model, chunk, device)
                for key, emb in zip(ids_chunk, embs):
                    pos_emb_hard[key] = emb
                    
    return negative_cache_hard, pos_emb_hard, pos_feat_hard


# --- 3. Data Preparation ---
def build_training_set(
    train_negative_ids: set,
    flag_positive_ids: set,
    pos_emb_hard: dict,
    pos_feat_hard: dict,
    negative_cache_hard: dict,
    T: int,
    train_ids_set: set,
    flag: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds the training set for a specific configuration."""
    X_train_emb, X_train_feat, y_train = [], [], []

    # 1. Positives
    for acc_id in flag_positive_ids:
        if acc_id not in train_ids_set: continue
        
        key = (acc_id, flag, T)
        if key in pos_emb_hard:
            X_train_emb.append(pos_emb_hard[key])
            X_train_feat.append(pos_feat_hard[key])
            y_train.append(1)

    # 2. Negatives
    for acc_id in train_negative_ids:
        if acc_id in flag_positive_ids: continue  # Skip if actually positive
        
        if acc_id in negative_cache_hard and T in negative_cache_hard[acc_id]:
            X_train_emb.append(negative_cache_hard[acc_id][T]['emb'])
            X_train_feat.append(negative_cache_hard[acc_id][T]['feat'])
            y_train.append(0)
            
    return np.array(X_train_emb), np.array(X_train_feat), np.array(y_train)


# --- 4. Training & Evaluation ---
def train_and_evaluate(
    X_train_emb: np.ndarray,
    X_train_feat: np.ndarray,
    y_train: np.ndarray,
    X_test_emb: np.ndarray,
    X_test_feat: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    neg_train_val: int,
    flag: str,
    T: int,
    has_cleanlab: bool,
    has_pulearn: bool
) -> list[dict]:
    """Trains models and evaluates them."""
    results = []
    
    if len(y_train) < 50 or sum(y_train) < 10: return results

    n_pos = sum(y_train)
    n_neg = len(y_train) - n_pos
    scale = n_neg / n_pos if n_pos > 0 else 1
    
    xgb_params = {
        'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1,
        'scale_pos_weight': scale, 'random_state': seed, 'verbosity': 0
    }
    
    has_bal = X_train_feat.shape[1] > 7

    classifiers = {}
    classifiers['XGB_Emb'] = XGBClassifier(**xgb_params).fit(X_train_emb, y_train)
    classifiers['XGB_Feat'] = XGBClassifier(**xgb_params).fit(X_train_feat, y_train)
    classifiers['XGB_Hybrid'] = XGBClassifier(**xgb_params).fit(np.concatenate([X_train_emb, X_train_feat], axis=1), y_train)
    
    if has_bal:
        classifiers['XGB_Bal'] = XGBClassifier(**xgb_params).fit(X_train_feat[:, -7:], y_train)
        
    classifiers['Linear_Emb'] = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed).fit(X_train_emb, y_train)

    if has_cleanlab:
        try:
            base_lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed, n_jobs=-1)
            classifiers['PUL_Emb'] = CleanLearning(clf=base_lr).fit(X_train_emb, y_train)
        except Exception: pass

    # Evaluate
    for method_name, clf in classifiers.items():
        if 'Feat' in method_name: X_test = X_test_feat
        elif 'Hybrid' in method_name: X_test = np.concatenate([X_test_emb, X_test_feat], axis=1)
        elif 'Bal' in method_name: X_test = X_test_feat[:, -7:]
        else: X_test = X_test_emb

        if len(X_test) == 0: continue
        
        try:
            probs = clf.predict_proba(X_test)[:, 1]
            try: auc = roc_auc_score(y_test, probs)
            except Exception: auc = 0.5
            lifts, _, _, _, _, _, _ = compute_lift_curve(y_test, probs)
            
            metrics = {
                'seed': seed,
                'negatives_in_training': neg_train_val,
                'flag': flag, 'T': T, 'method': method_name,
                'auc': auc, 'n_pos': int(sum(y_test)), 'n_total': len(y_test)
            }
            metrics.update(lifts)
            results.append(metrics)
            
            # Helper log for monitoring
            if method_name in ['XGB_Emb', 'XGB_Feat', 'PUL_Emb']:
                logger.info(f"{flag[:20]:<20} T={T} {method_name:<10} AUC={auc:.3f} L@10={lifts.get('lift_10',0):.2f}x (Negs={neg_train_val})")

        except Exception as e:
            logger.error(f"Error evaluating {method_name}: {e}")

    return results


# --- 5. Main Orchestrator ---
def main() -> None:
    """Main execution function."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensors_path', required=True)
    parser.add_argument('--emerging_csv', required=True)
    parser.add_argument('--model_checkpoint', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=None, help="Force number of layers")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--test_size', type=float, default=0.3)
    parser.add_argument('--T_values', type=str, default='1,3,7')
    parser.add_argument('--seeds', type=str, default='42')
    parser.add_argument('--negatives_in_training', type=str, default='1000,2000,5000,10000,-1')
    parser.add_argument('--csv_output_metrics', default='ensemble_summary_metrics.csv')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Data & Model
    model = load_model(args.model_checkpoint, device, args.hidden_dim, args.num_heads, args.num_layers, 'hierarchical')
    tensors = torch.load(args.tensors_path, weights_only=False)
    cache = {item['account_id']: item for item in tensors}  # Assuming list of dicts
    emerging_df = pd.read_csv(args.emerging_csv)
    emerging_df['global_id'] = emerging_df['bank'] + '_' + emerging_df['accountId'].astype(str)
    
    all_account_ids = list(cache.keys())
    all_flags = emerging_df['flag_name'].unique()
    T_values = [int(t) for t in args.T_values.split(',')]
    neg_train_configs = [int(x) for x in args.negatives_in_training.split(',')]
    seeds = [int(x) for x in args.seeds.split(',')]
    
    # Pre-compute Global Caches
    neg_cache_hard, pos_emb_hard, pos_feat_hard = cache_embeddings(
        model, cache, T_values, all_flags, emerging_df, args.batch_size, device
    )
    
    all_results = []
    
    # Loop
    for seed in seeds:
        logger.info(f"Running Seed {seed}")
        np.random.seed(seed)
        random.seed(seed)
        train_ids, test_ids = train_test_split(all_account_ids, test_size=args.test_size, random_state=seed)
        train_ids_set = set(train_ids)
        test_ids_set = set(test_ids)
        
        # Pre-sample negative sets for this seed
        neg_id_sets = {}
        for neg_val in neg_train_configs:
            if neg_val > 0 and len(train_ids) > neg_val:
                neg_id_sets[neg_val] = set(random.sample(train_ids, neg_val))
            else:
                neg_id_sets[neg_val] = set(train_ids)

        for flag in tqdm(all_flags, desc=f"Seed {seed} Flags"):
            flag_pos_ids = set(
                row['global_id'].split('_', 1)[1] if '_' in row['global_id'] else row['global_id']
                for _, row in emerging_df[emerging_df['flag_name'] == flag].iterrows()
            )
            
            for T in T_values:
                # 1. Build Test Data (Global Test Set) - Computed ONCE per Flag/T
                X_test_emb, X_test_feat, y_test = [], [], []
                for acc_id in test_ids:
                    is_pos = acc_id in flag_pos_ids
                    if is_pos:
                        key = (acc_id, flag, T)
                        if key in pos_emb_hard:
                            X_test_emb.append(pos_emb_hard[key])
                            X_test_feat.append(pos_feat_hard[key])
                            y_test.append(1)
                    else:
                         if acc_id in neg_cache_hard and T in neg_cache_hard[acc_id]:
                            X_test_emb.append(neg_cache_hard[acc_id][T]['emb'])
                            X_test_feat.append(neg_cache_hard[acc_id][T]['feat'])
                            y_test.append(0)
                
                X_test_emb = np.array(X_test_emb)
                X_test_feat = np.array(X_test_feat)
                y_test = np.array(y_test)
                
                # Check if valid test set
                if len(y_test) == 0 or sum(y_test) == 0: continue

                # 2. Iterate Negative Configs reusing Test Data
                for neg_val in neg_train_configs:
                    train_neg_ids = neg_id_sets[neg_val]
                    
                    # Build Training Data
                    X_train_emb, X_train_feat, y_train = build_training_set(
                        train_neg_ids, flag_pos_ids, pos_emb_hard, pos_feat_hard, neg_cache_hard, T, train_ids_set, flag
                    )
                    
                    # Eval
                    res = train_and_evaluate(
                        X_train_emb, X_train_feat, y_train, X_test_emb, X_test_feat, y_test, 
                        seed, neg_val, flag, T, HAS_CLEANLAB, HAS_PULEARN
                    )
                    all_results.extend(res)

    # Save
    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df.to_csv(output_dir / args.csv_output_metrics, index=False)
        logger.info(f"Saved results to {args.csv_output_metrics}")
    else:
        logger.warning("No results to save.")

    logger.info("Evaluation Complete")


if __name__ == '__main__':
    main()
