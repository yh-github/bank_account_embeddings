#!/usr/bin/env python3
"""Consolidate transaction and account data from multiple banks.

Usage:
    python -m scripts.data_prep.consolidate_data --config config/local.yaml

Or with environment variable:
    export EMBEDDER_CONFIG=config/local.yaml
    python -m scripts.data_prep.consolidate_data
"""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from hierarchical.config import load_config, get_path, get_banks


# Columns to keep for transactions (after processing)
TXN_KEEP_COLS = [
    "id",
    "accountId",
    "date",
    "amount",
    "description",
    "personeticsCategoryGroupId",
    "personeticsSubCategoryId",
    "deviceId",
    "direction",
    "status",
    "isForeignCurrency",
]

# Columns to keep for accounts
ACC_KEEP_COLS = [
    "id",
    "type",
    "status",
    "currency",
    "balance",
    "availableBalance",
    "balanceDateTime",
]


def consolidate_data(config_path: str | None = None) -> None:
    """Consolidate data from multiple banks into single files.

    Args:
        config_path: Optional path to config file.
    """
    # Load configuration
    config = load_config(config_path)

    data_dir = get_path("data_dir", required=True)
    output_dir = get_path("output_dir", required=False) or Path("./output")
    banks = get_banks()

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Banks to process: {banks}")

    # ----------------------------
    # Consolidate Transactions
    # ----------------------------
    print("\nConsolidating Transactions...")
    output_txn = output_dir / "transactions.csv"

    first_chunk = True
    total_txns = 0

    for bank in tqdm(banks, desc="Processing Banks (Txn)"):
        txn_file = data_dir / bank / "DTransaction_output.csv"
        if not txn_file.exists():
            print(f"  Skipping {bank}: transaction file not found")
            continue

        try:
            df = pd.read_csv(txn_file, low_memory=False)

            # Ensure required columns exist
            for col in TXN_KEEP_COLS:
                if col not in df.columns:
                    df[col] = None

            # Filter columns
            df = df[[c for c in TXN_KEEP_COLS if c in df.columns]].copy()

            # Apply filters
            # 1. Status = 'Cleared' (ignore Pending)
            if "status" in df.columns:
                df = df[df["status"] == "Cleared"]

            # 2. isForeignCurrency = False
            if "isForeignCurrency" in df.columns:
                df = df[df["isForeignCurrency"] != True]  # noqa: E712

            # 3. Non-zero amounts
            if "amount" in df.columns:
                df = df[df["amount"] != 0]

            # Add bank identifier and prefix account IDs
            df["bank"] = bank
            df["original_id"] = df["id"]
            df["id"] = bank + "_" + df["id"].astype(str)
            df["accountId"] = bank + "_" + df["accountId"].astype(str)

            # Write to file
            mode = "w" if first_chunk else "a"
            header = first_chunk
            df.to_csv(output_txn, mode=mode, header=header, index=False, quoting=1)

            total_txns += len(df)
            first_chunk = False

        except Exception as e:
            print(f"  Error processing transactions for {bank}: {e}")

    print(f"Saved {total_txns:,} transactions to {output_txn}")

    # ----------------------------
    # Consolidate Accounts
    # ----------------------------
    print("\nConsolidating Accounts...")
    output_acc = output_dir / "accounts.csv"

    first_chunk_acc = True
    total_accs = 0

    for bank in tqdm(banks, desc="Processing Banks (Acc)"):
        acc_file = data_dir / bank / "DAccount_output.csv"
        if not acc_file.exists():
            continue

        try:
            df = pd.read_csv(acc_file, low_memory=False)

            # Ensure columns exist
            for col in ACC_KEEP_COLS:
                if col not in df.columns:
                    df[col] = None

            # Filter columns
            df = df[ACC_KEEP_COLS].copy()

            # Add bank identifier and prefix IDs
            df["original_id"] = df["id"]
            df["bank"] = bank
            df["id"] = bank + "_" + df["id"].astype(str)

            # Write to file
            mode = "w" if first_chunk_acc else "a"
            header = first_chunk_acc
            df.to_csv(output_acc, mode=mode, header=header, index=False, quoting=1)

            total_accs += len(df)
            first_chunk_acc = False

        except Exception as e:
            print(f"  Error reading accounts for {bank}: {e}")

    print(f"Saved {total_accs:,} accounts to {output_acc}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Consolidate bank data")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (or set HIERARCHICAL_CONFIG env var)",
    )
    args = parser.parse_args()

    consolidate_data(args.config)


if __name__ == "__main__":
    main()
