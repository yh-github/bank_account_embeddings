"""Main training script for the Hierarchical V6 Model."""

import argparse
import json
import logging
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


from hierarchical.data.balance import BalanceFeatureExtractor
from hierarchical.data.dataset import HierarchicalDataset, collate_hierarchical
from hierarchical.data.loader import (
    load_accounts,
    load_transactions,
    load_joint_bank_data
)
from hierarchical.data.preloaded_dataset import PreloadedDataset
from hierarchical.data.vocab import build_vocabularies, load_vocabularies
from hierarchical.models.account import AccountEncoder
from hierarchical.models.day import DayEncoder
from hierarchical.models.transaction import TransactionEncoder
from hierarchical.training.losses import contrastive_loss
from hierarchical.training.utils import recursive_to_device, setup_logger

logger = logging.getLogger(__name__)


class TwoViewDataset(Dataset):
    """Dataset wrapper that returns two views of the same data item."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple[dict, dict]:
        item = self.dataset[idx]
        return item, item


def collate_two_views_hierarchical(batch: list[tuple[dict, dict]]) -> tuple[dict, dict]:
    """Collates a batch of two views."""
    view1 = [item[0] for item in batch]
    view2 = [item[1] for item in batch]
    return collate_hierarchical(view1), collate_hierarchical(view2)


def train(args: argparse.Namespace) -> None:
    """Main training function for Hierarchical V6 Model.

    Args:
        args: Parsed command line arguments.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training Hierarchical V6 on {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    setup_logger(args.output_dir)

    # 1. Load Data
    balance_extractor = None
    train_ds_base: Dataset
    val_ds_base: Dataset
    c_grp: dict
    c_sub: dict
    c_cp: dict

    if args.preprocessed_file:
        logger.info(f"Loading preprocessed data from {args.preprocessed_file}...")
        all_data = torch.load(args.preprocessed_file, weights_only=False)

        if args.vocab_dir:
            vocab_dir = args.vocab_dir
        else:
            vocab_dir = os.path.join(args.output_dir, "vocabularies")
        logger.info(f"Loading vocabs from {vocab_dir}...")
        c_grp, c_sub, c_cp = load_vocabularies(vocab_dir)

        # Shuffle and Split
        random.shuffle(all_data)
        idx = int(len(all_data) * 0.9)
        train_data = all_data[:idx]
        val_data = all_data[idx:]

        logger.info(f"Train Accounts: {len(train_data)}, Val: {len(val_data)}")

        train_ds_base = PreloadedDataset(
            train_data,
            augment=args.augment,
            max_days=args.max_days,
            max_txns_per_day=args.max_txns_per_day
        )
        val_ds_base = PreloadedDataset(
            val_data,
            augment=False,
            max_days=args.max_days,
            max_txns_per_day=args.max_txns_per_day
        )

    elif args.bank_config:
        logger.info(f"Loading Joint Data from {args.bank_config}...")
        with open(args.bank_config, 'r') as f:
            bank_configs = json.load(f)
        df, df_acc = load_joint_bank_data(bank_configs, args.cutoff_date)

        if df_acc is not None:
            balance_extractor = BalanceFeatureExtractor(df, df_acc)
            logger.info("Joint Balance Feature Extractor initialized.")

        # 2. Build Vocabularies
        vocab_dir = os.path.join(args.output_dir, "vocabularies")
        c_grp, c_sub, c_cp = build_vocabularies(df, vocab_dir)

        # 3. Split with Seed and Save
        random_seed = 42
        np.random.seed(random_seed)

        accounts = df['accountId'].unique()
        np.random.shuffle(accounts)
        idx = int(len(accounts) * 0.9)
        train_ids = accounts[:idx]
        val_ids = accounts[idx:]

        logger.info(f"Train Accounts: {len(train_ids)}, Val: {len(val_ids)}")

        # Save Splits
        np.save(os.path.join(args.output_dir, "train_ids.npy"), train_ids)
        np.save(os.path.join(args.output_dir, "val_ids.npy"), val_ids)
        logger.info(f"Saved splits to {args.output_dir}")

        # 4. Datasets
        if args.augment:
             logger.warning("Augmentation requested but AugmentedDataset removed. Using standard Dataset.")

        train_ds_base = HierarchicalDataset(
            df, c_grp, c_sub, c_cp,
            balance_extractor=balance_extractor,
            account_ids=train_ids,
            max_days=args.max_days, min_days=5,
            max_txns_per_day=args.max_txns_per_day
        )

        val_ds_base = HierarchicalDataset(
            df, c_grp, c_sub, c_cp,
            balance_extractor=balance_extractor,
            account_ids=val_ids,
            max_days=args.max_days, min_days=5,
            max_txns_per_day=args.max_txns_per_day
        )

    else:
        logger.info(f"Loading transactions from {args.data_file}...")
        df = load_transactions(args.data_file)

        # LEAKAGE FIX: Temporal Cutoff
        if args.cutoff_date:
            cutoff_dt = pd.to_datetime(args.cutoff_date)
            initial_len = len(df)
            df = df[df['date'] < cutoff_dt]
            logger.info(f"Leakage Prevention: Filtered data < {args.cutoff_date}. Kept {len(df):,} ({len(df)/initial_len:.1%}) txns.")

        if args.account_file and os.path.exists(args.account_file):
            logger.info(f"Loading accounts from {args.account_file}...")
            df_acc = load_accounts(args.account_file)
            balance_extractor = BalanceFeatureExtractor(df, df_acc)
            logger.info("Balance Feature Extractor initialized.")

        # 2. Build Vocabularies
        vocab_dir = os.path.join(args.output_dir, "vocabularies")
        c_grp, c_sub, c_cp = build_vocabularies(df, vocab_dir)

        # 3. Split
        accounts = df['accountId'].unique()
        np.random.shuffle(accounts)
        idx = int(len(accounts) * 0.9)
        train_ids = accounts[:idx]
        val_ids = accounts[idx:]

        logger.info(f"Train Accounts: {len(train_ids)}, Val: {len(val_ids)}")

        # 4. Datasets
        if args.augment:
             logger.warning("Augmentation requested but AugmentedDataset removed. Using standard Dataset.")

        train_ds_base = HierarchicalDataset(
            df, c_grp, c_sub, c_cp,
            balance_extractor=balance_extractor,
            account_ids=train_ids,
            max_days=args.max_days, min_days=5,
            max_txns_per_day=args.max_txns_per_day
        )

        val_ds_base = HierarchicalDataset(
            df, c_grp, c_sub, c_cp,
            balance_extractor=balance_extractor,
            account_ids=val_ids,
            max_days=args.max_days, min_days=5,
            max_txns_per_day=args.max_txns_per_day
        )

    train_ds = TwoViewDataset(train_ds_base)
    val_ds = TwoViewDataset(val_ds_base)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_two_views_hierarchical, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_two_views_hierarchical, num_workers=args.num_workers
    )

    # 5. Model Initialization
    logger.info("Initializing Model...")
    
    # Resolve Dimensions
    txn_dim = args.txn_dim if args.txn_dim else args.hidden_dim
    day_dim = args.day_dim if args.day_dim else args.hidden_dim
    acc_dim = args.account_dim if args.account_dim else args.hidden_dim
    
    logger.info(f"Dimensions: Txn={txn_dim} -> Day={day_dim} -> Account={acc_dim}")

    txn_enc = TransactionEncoder(
        num_categories_group=len(c_grp),
        num_categories_sub=len(c_sub),
        num_counter_parties=len(c_cp),
        embedding_dim=txn_dim,
        use_balance=args.use_balance,
        use_counter_party=args.use_counter_party
    )

    day_enc = DayEncoder(
        txn_enc,
        hidden_dim=day_dim,
        num_heads=4,
        num_layers=2,
        dropout=args.dropout
    ).to(device)

    if args.gradient_checkpointing:
        logger.info("Enabling Gradient Checkpointing for Day Encoder")
        day_enc.use_checkpointing = True

    model = AccountEncoder(
        day_enc,
        hidden_dim=acc_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout
    ).to(device)



    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    logger.info(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 6. Training Loop
    best_val_loss = float('inf')
    early_stop_counter = 0
    scaler = torch.amp.GradScaler('cuda' if device == 'cuda' else 'cpu', enabled=args.use_amp)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for batch1, batch2 in pbar:
            optimizer.zero_grad()
            b1 = recursive_to_device(batch1, device)
            b2 = recursive_to_device(batch2, device)

            # Mixed Precision Forward
            with torch.amp.autocast('cuda' if device == 'cuda' else 'cpu', enabled=args.use_amp):
                emb1 = model(b1)
                emb2 = model(b2)
                loss = contrastive_loss(emb1, emb2)

            if torch.isnan(loss):
                logger.warning(f"NaN loss in epoch {epoch}, batch skipped.")
                continue

            # Backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} Train Loss: {avg_loss:.4f}")

        # Validation
        val_loss = validate(model, val_loader, device, use_amp=args.use_amp)
        logger.info(f"Epoch {epoch+1} Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save checkpoint with hyperparameters for proper model reconstruction
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'config': {
                    'num_categories_group': len(c_grp),
                    'num_categories_sub': len(c_sub),
                    'num_counter_parties': len(c_cp),
                    'txn_dim': txn_dim,
                    'day_dim': day_dim,
                    'account_dim': acc_dim,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'day_num_layers': 2,  # DayEncoder uses 2 layers
                    'num_heads': args.num_heads,
                    'use_balance': args.use_balance,
                    'use_counter_party': args.use_counter_party,
                    'dropout': args.dropout,
                }
            }
            torch.save(checkpoint, os.path.join(args.output_dir, "model_best.pth"))
            logger.info("  Saved Best Model (with config)")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= 5:
                logger.info("Early stopping triggered")
                break


def validate(model: nn.Module, val_loader: DataLoader, device: str, use_amp: bool = False) -> float:
    """Validation loop."""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch1, batch2 in val_loader:
            b1 = recursive_to_device(batch1, device)
            b2 = recursive_to_device(batch2, device)
            
            with torch.amp.autocast('cuda' if device == 'cuda' else 'cpu', enabled=use_amp):
                emb1 = model(b1)
                emb2 = model(b2)
                loss = contrastive_loss(emb1, emb2)
            
            val_loss += loss.item()
    return val_loss / len(val_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', required=False)
    parser.add_argument('--account_file', required=False)
    parser.add_argument('--bank_config', required=False, help="JSON file with list of bank configs")
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--vocab_dir', required=False, help="Directory containing vocab files")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_dim', type=int, default=128, help="Legacy: Default dim if others not specified")
    
    # Pyramid Architecture Args
    parser.add_argument('--txn_dim', type=int, default=None)
    parser.add_argument('--day_dim', type=int, default=None)
    parser.add_argument('--account_dim', type=int, default=None)
    
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--gradient_checkpointing', action='store_true', help="Enable gradient checkpointing (memory optimization)")

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--cutoff_date', type=str, default=None, help="YYYY-MM-DD to convert pre-training to strictly past data")

    # New High-Memory Args
    parser.add_argument('--augment', action='store_true', help="Enable augmentation (Multi-view/Masking)")
    parser.add_argument('--max_txns_per_day', type=int, default=30, help="Max txns per day (default 30)")
    parser.add_argument('--max_days', type=int, default=180, help="Max days context (default 180)")

    parser.add_argument('--preprocessed_file', type=str, default=None, help="Path to .pt file with preprocessed data")

    # Feature Flags
    parser.add_argument('--use_balance', action='store_true', help="Enable balance features")
    parser.add_argument('--no_counter_party', action='store_true', help="Disable counter_party features")
    parser.add_argument('--use_amp', action='store_true', help="Enable Automatic Mixed Precision")

    args = parser.parse_args()

    # Logic to resolve flags
    args.use_counter_party = not args.no_counter_party

    if not args.data_file and not args.bank_config and not args.preprocessed_file:
        parser.error("Either --data_file, --bank_config, or --preprocessed_file must be provided.")

    train(args)
