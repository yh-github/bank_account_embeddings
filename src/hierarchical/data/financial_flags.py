#!/usr/bin/env python3
"""
Financial Activity Flags Detection

Detects two types of financial events from transaction data:
1. Emerging - Category first appears after initial observation period (indicates behavioral change)
2. Skipped Payment - Gap >= threshold days between consecutive payments in a recurring pattern

Note: "Emerging" flags indicate a change from the user's observed baseline behavior,
not necessarily a "first-ever" event (data represents a limited 6-month snapshot).

Usage:
    python financial_flags.py
    python financial_flags.py --config path/to/config.yaml
"""

import pandas as pd
import yaml
import argparse
import os
import re
from dateutil.relativedelta import relativedelta


# ============================================
# CONFIGURATION
# ============================================


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    print(f"Loading configuration from {config_path}...")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_target_subcategories(config: dict) -> list:
    """Get list of subcategory IDs that belong to target category groups."""
    target_groups = config.get("target_category_groups", [])
    categories = config.get("categories", {})

    target_subcategories = [
        subcat_id
        for subcat_id, info in categories.items()
        if info.get("group") in target_groups
    ]
    return target_subcategories


def generate_flag_name(sub_category_name: str, prefix: str) -> str:
    """
    Convert subcategory name to flag name format.

    Example: 'Buy Now Pay Later' -> 'EMERGING_BUY_NOW_PAY_LATER'
    """
    # Remove special characters, replace spaces with underscores, uppercase
    normalized = re.sub(r"[^a-zA-Z0-9\s]", "", sub_category_name)
    normalized = re.sub(r"\s+", "_", normalized.strip())
    return f"{prefix}_{normalized.upper()}"


# ============================================
# DATA LOADING
# ============================================


def load_transactions(file_path: str) -> pd.DataFrame:
    """Load DTransaction.csv with date parsing and validation."""
    print(f"Loading transactions from {file_path}...")

    df = pd.read_csv(file_path, low_memory=False)

    # Drop duplicate transactions by id
    initial_count = len(df)
    df = df.drop_duplicates(subset=["id"], keep="first")
    if len(df) < initial_count:
        print(f"  Dropped {initial_count - len(df):,} duplicate transactions by id")

    # Parse date column
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Basic statistics
    print(f"  Loaded {len(df):,} transactions")
    print(
        f"  Date range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
    )
    print(f"  Users: {df['user_name'].nunique():,}")

    return df


def load_accounts(file_path: str) -> pd.DataFrame:
    """Load DAccount.csv with relevant columns for filtering."""
    print(f"Loading accounts from {file_path}...")

    # Only load columns we need
    df = pd.read_csv(file_path, usecols=["id", "type"], low_memory=False)

    print(f"  Loaded {len(df):,} accounts")
    print(f"  Account types: {df['type'].unique().tolist()}")

    return df


def load_recurring_patterns(file_path: str) -> pd.DataFrame:
    """Load DRecurringPattern.csv with relevant columns for pattern type filtering."""
    print(f"Loading recurring patterns from {file_path}...")

    # Only load columns we need
    df = pd.read_csv(
        file_path,
        usecols=["patternId", "patternTypeBusiness", "deviceId"],
        low_memory=False,
    )

    print(f"  Loaded {len(df):,} patterns")
    print(f"  Pattern types: {df['patternTypeBusiness'].value_counts().to_dict()}")

    return df


def filter_transactions(
    transactions_df: pd.DataFrame,
    accounts_df: pd.DataFrame,
    account_type: str,
    direction: str,
) -> pd.DataFrame:
    """
    Filter transactions by account type and direction.

    1. Left join transactions with accounts on accountId = id
    2. Filter by account type (e.g., 'Checking')
    3. Filter by direction (e.g., 'D' for debit/outgoing)
    """
    print("\nFiltering transactions...")
    print(f"  Account type: {account_type}")
    print(
        f"  Direction: {direction} ({'Debit/Outgoing' if direction == 'D' else 'Credit/Incoming'})"
    )

    initial_count = len(transactions_df)

    # Left join with accounts
    df = transactions_df.merge(
        accounts_df,
        left_on="accountId",
        right_on="id",
        how="left",
        suffixes=("", "_account"),
    )

    # Filter by account type
    if account_type:
        df = df[df["type"] == account_type]
        print(f"  After account type filter: {len(df):,} transactions")

    # Filter by direction
    if direction:
        df = df[df["direction"] == direction]
        print(f"  After direction filter: {len(df):,} transactions")

    print(
        f"  Filtered from {initial_count:,} to {len(df):,} transactions ({len(df) / initial_count * 100:.1f}%)"
    )

    return df


def enrich_with_categories(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """Add sub_category_name column from YAML config mapping."""
    print("Enriching with category names...")

    # Create mapping from subcategory ID to name
    name_map = {
        subcat_id: info.get("name", subcat_id) for subcat_id, info in categories.items()
    }

    # Add sub_category_name column
    df["sub_category_name"] = df["personeticsSubCategoryId"].map(name_map)

    return df


# ============================================
# FLAG DETECTION: EMERGING
# ============================================


def detect_emerging(
    df: pd.DataFrame, target_subcategories: list, categories: dict, primacy_months: int
) -> pd.DataFrame:
    """
    Detect emerging financial activity flags per account.

    Identifies categories that first appear after the initial observation period,
    indicating a change from the account's baseline behavior (not necessarily "first-ever").

    Logic:
    1. Calculate account_start_date per accountId: MIN(date) across all transactions
    2. For each target subcategory:
       a. Find emerging_date per accountId (first appearance of category)
       b. Calculate months_since_start
       c. Flag if months_since_start >= primacy_months

    Note: Same user with multiple accounts may have multiple emerging flags for the same category.

    Returns DataFrame with flagged records.
    """
    print(f"\nDetecting Emerging flags (primacy period: {primacy_months} months)...")

    results = []

    # Calculate account start date per accountId (earliest transaction date)
    account_start = df.groupby("accountId")["date"].min().reset_index()
    account_start.columns = ["accountId", "account_start_date"]

    # Filter to target subcategories
    target_df = df[df["personeticsSubCategoryId"].isin(target_subcategories)].copy()

    if target_df.empty:
        print("  No transactions found for target subcategories")
        return pd.DataFrame()

    # Process each subcategory
    for subcat_id in target_subcategories:
        subcat_info = categories.get(subcat_id, {})
        subcat_name = subcat_info.get("name", subcat_id)

        # Filter to this subcategory
        subcat_df = target_df[target_df["personeticsSubCategoryId"] == subcat_id]

        if subcat_df.empty:
            continue

        print(f"  Processing {subcat_id} ({subcat_name})...")

        # Find emerging date per account for this subcategory (first appearance)
        # Get the first transaction (by date) for each account
        first_txn_idx = subcat_df.groupby("accountId")["date"].idxmin()
        first_txns = subcat_df.loc[
            first_txn_idx, ["user_name", "accountId", "date"]
        ].copy()
        first_txns.columns = ["user_name", "accountId", "emerging_date"]

        # Merge with account start dates
        merged = first_txns.merge(account_start, on="accountId", how="left")

        # Calculate months since start
        merged["months_since_start"] = merged.apply(
            lambda row: relativedelta(
                row["emerging_date"], row["account_start_date"]
            ).months
            + relativedelta(row["emerging_date"], row["account_start_date"]).years * 12
            + relativedelta(row["emerging_date"], row["account_start_date"]).days
            / 30.44,
            axis=1,
        )

        # Flag if category emerged after primacy period
        flagged = merged[merged["months_since_start"] >= primacy_months].copy()

        if not flagged.empty:
            flagged["flag_name"] = generate_flag_name(subcat_name, "EMERGING")
            flagged["subcategoryId"] = subcat_id
            flagged["sub_category_name"] = subcat_name
            flagged["categoryGroupId"] = subcat_info.get("group", "")
            flagged["category_group_name"] = subcat_info.get("group_name", "")
            results.append(flagged)

    if results:
        result_df = pd.concat(results, ignore_index=True)
        print(f"  Found {len(result_df):,} Emerging flags")
        return result_df[
            [
                "user_name",
                "accountId",
                "flag_name",
                "categoryGroupId",
                "category_group_name",
                "subcategoryId",
                "sub_category_name",
                "emerging_date",
                "account_start_date",
                "months_since_start",
            ]
        ]
    else:
        print("  Found 0 Emerging flags")
        return pd.DataFrame()


# ============================================
# FLAG DETECTION: SKIPPED PAYMENT
# ============================================


def detect_skipped_payment(
    df: pd.DataFrame,
    recurring_patterns_df: pd.DataFrame,
    pattern_types: list,
    gap_threshold_days: int,
    min_transactions: int = 4,
    bimonthly_gap_threshold: int = 50,
    monthly_payments_config: dict = None,
) -> pd.DataFrame:
    """
    Detect skipped payment flags across ALL recurring patterns (not limited to specific categories).

    Logic:
    1. Filter: patternId is not null
    2. Join with DRecurringPattern to get patternTypeBusiness
    3. Filter by allowed pattern types (e.g., DateOfMonth, NoisyDateOfMonth)
    4. For each pattern:
       a. Group by (user_name, patternId)
       b. For each group with min_transactions+ transactions:
          - Sort by date, calculate gaps between consecutive payments
          - Flag if any gap >= gap_threshold_days (or pattern-specific threshold)
       c. Exclude bi-monthly patterns: patterns with exactly 3 transactions
          where ALL gaps > bimonthly_gap_threshold (likely ~60-day cycles)

    Note: Recurring patterns have minimum 3 transactions; we use 4 by default
    to ensure sufficient history for gap detection.

    Args:
        monthly_payments_config: Optional dict with 'pattern_type' and 'gap_threshold_days'
                                 for MonthlyPayments with different threshold

    Returns DataFrame with flagged records.
    """
    print(
        f"\nDetecting Skipped Payment flags (gap threshold: {gap_threshold_days} days)..."
    )
    print(f"  Pattern types filter: {pattern_types}")
    if monthly_payments_config:
        print(
            f"  MonthlyPayments filter: {monthly_payments_config['pattern_type']} with {monthly_payments_config['gap_threshold_days']} days threshold"
        )
    print(
        f"  Bi-monthly exclusion: patterns with 3 txns and all gaps > {bimonthly_gap_threshold} days"
    )

    # Check if patternId column exists
    if "patternId" not in df.columns:
        print(
            "  Warning: patternId column not found. Skipping skipped payment detection."
        )
        return pd.DataFrame()

    # Filter to transactions with patternId
    pattern_df = df[df["patternId"].notna()].copy()
    print(f"  Transactions with patternId: {len(pattern_df):,}")

    if pattern_df.empty:
        print("  No transactions with patternId found")
        return pd.DataFrame()

    # Join with recurring patterns to get patternTypeBusiness
    pattern_df = pattern_df.merge(
        recurring_patterns_df[["patternId", "patternTypeBusiness"]],
        on="patternId",
        how="left",
    )

    # Build combined pattern types filter (include MonthlyPayments if configured)
    all_pattern_types = list(pattern_types) if pattern_types else []
    monthly_payments_type = None
    monthly_payments_threshold = None
    if monthly_payments_config:
        monthly_payments_type = monthly_payments_config.get("pattern_type")
        monthly_payments_threshold = monthly_payments_config.get("gap_threshold_days")
        if monthly_payments_type and monthly_payments_type not in all_pattern_types:
            all_pattern_types.append(monthly_payments_type)

    # Filter by pattern types
    if all_pattern_types:
        pattern_df = pattern_df[
            pattern_df["patternTypeBusiness"].isin(all_pattern_types)
        ]
        print(f"  After pattern type filter: {len(pattern_df):,} transactions")

    if pattern_df.empty:
        print("  No transactions found for specified pattern types")
        return pd.DataFrame()

    results = []
    bimonthly_excluded = 0

    # Group by user and pattern (no category filter)
    for (user_name, pattern_id), group in pattern_df.groupby(
        ["user_name", "patternId"]
    ):
        # Skip groups with fewer than min_transactions
        if len(group) < min_transactions:
            continue

        # Sort by date and calculate gaps
        sorted_group = group.sort_values("date")
        dates = sorted_group["date"].values

        # Get pattern info from first transaction
        device_id = (
            sorted_group["deviceId"].iloc[0]
            if "deviceId" in sorted_group.columns
            else None
        )
        account_id = (
            sorted_group["accountId"].iloc[0]
            if "accountId" in sorted_group.columns
            else None
        )
        pattern_type = sorted_group["patternTypeBusiness"].iloc[0]

        # Determine threshold based on pattern type
        if pattern_type == monthly_payments_type and monthly_payments_threshold:
            threshold = monthly_payments_threshold
        else:
            threshold = gap_threshold_days

        # Calculate all gaps between consecutive payments
        gaps = []
        for i in range(1, len(dates)):
            gap_days = (pd.Timestamp(dates[i]) - pd.Timestamp(dates[i - 1])).days
            gaps.append((i, gap_days))

        # Exclude bi-monthly patterns: exactly 3 transactions with ALL gaps > threshold
        # These are likely regular bi-monthly payments (~60 day cycles), not skipped payments
        if len(dates) == 3 and all(g[1] > bimonthly_gap_threshold for g in gaps):
            bimonthly_excluded += 1
            continue

        # Flag gaps that exceed threshold (pattern-type specific)
        for i, gap_days in gaps:
            if gap_days >= threshold:
                results.append(
                    {
                        "user_name": user_name,
                        "accountId": account_id,
                        "flag_name": "SKIPPED_PAYMENT",
                        "patternId": pattern_id,
                        "patternTypeBusiness": pattern_type,
                        "deviceId": device_id,
                        "payment_before_gap": pd.Timestamp(dates[i - 1]).strftime(
                            "%Y-%m-%d"
                        ),
                        "payment_after_gap": pd.Timestamp(dates[i]).strftime(
                            "%Y-%m-%d"
                        ),
                        "gap_days": gap_days,
                    }
                )

    print(f"  Excluded {bimonthly_excluded:,} bi-monthly patterns")

    if results:
        result_df = pd.DataFrame(results)
        print(f"  Found {len(result_df):,} Skipped Payment flags")
        return result_df
    else:
        print("  Found 0 Skipped Payment flags")
        return pd.DataFrame()


# ============================================
# MAIN
# ============================================


def main(config_path: str = "config.yaml"):
    """
    Main entry point.

    1. Load config from YAML
    2. Load transaction, account, and recurring pattern data
    3. Filter by account type and direction
    4. Enrich with category names
    5. Run Emerging detection (for target categories)
    6. Run Skipped Payment detection (for all patterns of specified types)
    7. Save outputs to CSV
    8. Print summary statistics
    """
    print("=" * 60)
    print("Financial Activity Flags Detection")
    print("=" * 60)

    # Load configuration
    config = load_config(config_path)

    # Get paths from config
    transaction_file = config["paths"]["transaction_file"]
    account_file = config["paths"].get("account_file")
    recurring_pattern_file = config["paths"].get("recurring_pattern_file")
    output_dir = config["paths"]["output_dir"]

    # Get parameters
    params = config["parameters"]
    primacy_months = params.get("primacy_period_months", 4)
    gap_threshold_days = params.get("gap_threshold_days", 45)
    min_transactions = params.get("min_transactions", 4)
    bimonthly_gap_threshold = params.get("bimonthly_gap_threshold", 50)

    # Get filters
    filters = config.get("filters", {})
    account_type = filters.get("account_type")
    direction = filters.get("direction")

    # Get skipped payment filters
    skipped_filters = config.get("skipped_payment_filters", {})
    pattern_types = skipped_filters.get("pattern_types", [])
    monthly_payments_config = skipped_filters.get("monthly_payments")

    # Get categories
    categories = config.get("categories", {})

    # Get target subcategories (for Emerging detection only)
    target_subcategories = get_target_subcategories(config)
    print(f"  Target subcategories (Emerging): {len(target_subcategories)}")

    # Load transaction data
    df = load_transactions(transaction_file)

    # Load recurring patterns (for Skipped Payment detection)
    recurring_patterns_df = None
    if recurring_pattern_file and os.path.exists(recurring_pattern_file):
        recurring_patterns_df = load_recurring_patterns(recurring_pattern_file)
    elif recurring_pattern_file:
        print(f"  Recurring pattern file not found: {recurring_pattern_file}")

    # Load account data and filter (if account_file provided)
    if account_file:
        accounts_df = load_accounts(account_file)
        df = filter_transactions(df, accounts_df, account_type, direction)
    elif direction:
        # Filter by direction only if no account file
        print(f"\nFiltering by direction: {direction}")
        df = df[df["direction"] == direction]
        print(f"  Filtered to {len(df):,} transactions")

    # Enrich with category names
    df = enrich_with_categories(df, categories)

    # Detect Emerging flags (for target categories only)
    emerging_flags = detect_emerging(
        df, target_subcategories, categories, primacy_months
    )

    # Detect Skipped Payment flags (for all patterns of specified types)
    skipped_payment_flags = pd.DataFrame()
    if recurring_patterns_df is not None:
        skipped_payment_flags = detect_skipped_payment(
            df,
            recurring_patterns_df,
            pattern_types,
            gap_threshold_days,
            min_transactions,
            bimonthly_gap_threshold,
            monthly_payments_config,
        )
    else:
        print(
            "\nSkipping Skipped Payment detection (no recurring pattern file configured)"
        )

    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)

    # Save outputs
    print("\nSaving outputs...")

    if not emerging_flags.empty:
        emerging_path = os.path.join(output_dir, "emerging_flags.csv")
        emerging_flags.to_csv(emerging_path, index=False)
        print(f"  {emerging_path} ({len(emerging_flags)} rows)")
    else:
        print("  emerging_flags.csv (0 rows - not saved)")

    if not skipped_payment_flags.empty:
        skipped_payment_path = os.path.join(output_dir, "skipped_payment_flags.csv")
        skipped_payment_flags.to_csv(skipped_payment_path, index=False)
        print(f"  {skipped_payment_path} ({len(skipped_payment_flags)} rows)")
    else:
        print("  skipped_payment_flags.csv (0 rows - not saved)")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial Activity Flags Detection")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to configuration file"
    )
    args = parser.parse_args()

    main(args.config)
