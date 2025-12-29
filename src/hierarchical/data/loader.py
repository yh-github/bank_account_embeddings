"""Data loading utilities."""

import pandas as pd


def load_transactions(filepath: str) -> pd.DataFrame:
    """Loads transactions from a CSV or Parquet file.

    Standardizes column names to 'date' and 'amount'.

    Args:
        filepath: Path to the data file.

    Returns:
        DataFrame containing transactions.
    """
    if str(filepath).endswith(".parquet"):
        df = pd.read_parquet(filepath)
    else:
        # Robust loading matching consolidation writer (quoting=1)
        import csv

        # Use python engine for maximum robustness against quoting issues
        print("DEBUG: Using Python Engine for loading transactions...")
        df = pd.read_csv(filepath, quoting=csv.QUOTE_ALL, engine="python")

    # Ensure standard columns
    if "transactionDate" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"transactionDate": "date"})
    if "transactionAmount" in df.columns and "amount" not in df.columns:
        df = df.rename(columns={"transactionAmount": "amount"})

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    return df


def load_accounts(filepath: str) -> pd.DataFrame:
    """Loads account metadata from a CSV file.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame containing account info.
    """
    import csv

    return pd.read_csv(filepath, quoting=csv.QUOTE_ALL, engine="python")


def load_joint_bank_data(
    bank_configs: list[dict[str, str]], cutoff_date: str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Loads and merges data from multiple banks.

    Args:
        bank_configs: List of dicts containing configuration for each bank.
            Expected keys: 'name', 'txn_file', 'acc_file'.
            Optional keys: 'cutoff_date'.
        cutoff_date: Optional global cutoff date string 'YYYY-MM-DD'.
            Bank-specific cutoff takes precedence if present.

    Returns:
        A tuple of:
        - merged_txns: DataFrame containing all transactions.
        - merged_accounts: DataFrame containing all accounts, or None.
    """
    import logging
    import os

    logger = logging.getLogger(__name__)

    all_txns = []
    all_accounts = []

    logger.info(f"Loading data for {len(bank_configs)} banks...")

    for config in bank_configs:
        bank_name = config["name"]
        logger.info(f"  Loading {bank_name}...")

        # Load Txns
        df = load_transactions(config["txn_file"])

        # Apply Cutoff (Optimization: Filter before merge)
        # Priority: Config Cutoff > Global Cutoff (allows relative split)
        bank_cutoff = config.get("cutoff_date", cutoff_date)

        if bank_cutoff:
            logger.info(f"    Applying cutoff: {bank_cutoff}")
            df = df[pd.to_datetime(df["date"]) < pd.to_datetime(bank_cutoff)]

        # Prefix Account IDs
        df["accountId"] = f"{bank_name}_" + df["accountId"].astype(str)
        all_txns.append(df)

        # Load Accounts (if available)
        acc_file = config.get("acc_file")
        if acc_file and os.path.exists(acc_file):
            acc = load_accounts(acc_file)
            # Standardize column names
            name_map = {
                "account_id": "accountId",
                "accountID": "accountId",
                "Account ID": "accountId",
                "id": "accountId",
            }
            acc = acc.rename(columns=name_map)
            # Special case: 'id' maps to accountId only if accountId doesn't exist
            if "id" in acc.columns and "accountId" not in acc.columns:
                acc = acc.rename(columns={"id": "accountId"})

            if "accountId" in acc.columns:
                acc["accountId"] = f"{bank_name}_" + acc["accountId"].astype(str)
                all_accounts.append(acc)
            else:
                logger.warning(
                    f"  Warning: 'accountId' not found in {acc_file}. Columns: {acc.columns.tolist()}"
                )

    # Merge
    logger.info("  Merging datasets...")
    merged_txns = pd.concat(all_txns, ignore_index=True)

    merged_accounts = None
    if all_accounts:
        merged_accounts = pd.concat(all_accounts, ignore_index=True)

    logger.info(
        f"Joint Dataset: {len(merged_txns):,} transactions, "
        f"{len(merged_txns['accountId'].unique()):,} accounts."
    )

    return merged_txns, merged_accounts
