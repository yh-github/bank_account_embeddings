"""Balance feature extraction and calculation."""

import datetime
import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_balance_per_transaction(
    df_transactions: pd.DataFrame,
    df_accounts: pd.DataFrame,
    include_pending: bool = True
) -> pd.DataFrame:
    """Calculates running balance after each transaction.

    Args:
        df_transactions: Transaction data with columns: accountId, id, date, amount, direction.
        df_accounts: Account data with columns: accountId, availableBalance, balanceDateTime.
            Optional: estimatedPendingAmount.
        include_pending: If True and estimatedPendingAmount exists, adds synthetic transactions
            to account for pending amounts.

    Returns:
        Original transactions DataFrame with 'balance_after' column added,
        sorted by (accountId, date, id).
    """
    # Step 1: Prepare and validate inputs
    df_txn, df_acc = _prepare_inputs(df_transactions.copy(), df_accounts.copy())

    # Step 2: Normalize amount signs (D=negative, C=positive)
    df_txn = _normalize_amounts(df_txn)

    # Step 3: Optionally add pending transactions
    if include_pending:
        df_txn = _add_pending_transactions(df_txn, df_acc)

    # Step 4: Calculate running balance per transaction
    df_result = _calculate_running_balance(df_txn, df_acc)

    return df_result


def _prepare_inputs(
    df_transactions: pd.DataFrame,
    df_accounts: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validates and prepares input DataFrames."""
    # Required columns validation
    txn_required = ['accountId', 'id', 'date', 'amount', 'direction']
    acc_required = ['accountId', 'availableBalance', 'balanceDateTime']

    missing_txn = [col for col in txn_required if col not in df_transactions.columns]
    missing_acc = [col for col in acc_required if col not in df_accounts.columns]

    if missing_txn:
        raise ValueError(f"DTransaction missing required columns: {missing_txn}")
    if missing_acc:
        raise ValueError(f"DAccount missing required columns: {missing_acc}")

    # Normalize accountId to uppercase string
    df_transactions['accountId'] = df_transactions['accountId'].astype(str).str.upper()
    df_accounts['accountId'] = df_accounts['accountId'].astype(str).str.upper()

    # Convert dates
    df_transactions['date'] = pd.to_datetime(df_transactions['date']).dt.date
    df_accounts['balanceDateTime'] = pd.to_datetime(df_accounts['balanceDateTime']).dt.date

    # Keep only latest balanceDateTime per account
    df_accounts = df_accounts.sort_values(
        ['accountId', 'balanceDateTime'],
        ascending=[True, False]
    ).drop_duplicates('accountId')

    # Check and warn about duplicates
    original_count = len(df_transactions)
    df_transactions = df_transactions.drop_duplicates(subset=['accountId', 'id'])
    duplicates_removed = original_count - len(df_transactions)
    if duplicates_removed > 0:
        logger.warning(f"Removed {duplicates_removed} duplicate transactions (by accountId, id)")

    # Check for transactions after balanceDateTime
    df_txn_with_balance_date = df_transactions.merge(
        df_accounts[['accountId', 'balanceDateTime']],
        on='accountId',
        how='left'
    )
    future_txns = df_txn_with_balance_date[
        df_txn_with_balance_date['date'] > df_txn_with_balance_date['balanceDateTime']
    ]
    if len(future_txns) > 0:
        affected_accounts = future_txns['accountId'].nunique()
        logger.warning(
            f"Found {len(future_txns)} transactions after balanceDateTime in {affected_accounts} accounts. "
            "This may cause incorrect balance calculation."
        )

    # Sort transactions
    df_transactions = df_transactions.sort_values(['accountId', 'date', 'id']).reset_index(drop=True)

    return df_transactions, df_accounts


def _normalize_amounts(df_transactions: pd.DataFrame) -> pd.DataFrame:
    """Ensures consistent sign convention for amounts (D=negative, C=positive)."""
    # Check if direction 'D' transactions are mostly positive
    if 'D' in df_transactions['direction'].values:
        d_mask = df_transactions['direction'] == 'D'
        d_vals = df_transactions.loc[d_mask, 'amount']
        if len(d_vals) > 0:
            d_positive_ratio = (d_vals > 0).mean()
            if d_positive_ratio > 0.99:
                df_transactions['amount'] = df_transactions['amount'] * -1
                logger.info("Flipped amount signs (D transactions were positive, now negative)")
                return df_transactions

    # Check if direction 'C' transactions are mostly negative
    if 'C' in df_transactions['direction'].values:
        c_mask = df_transactions['direction'] == 'C'
        c_vals = df_transactions.loc[c_mask, 'amount']
        if len(c_vals) > 0:
            c_negative_ratio = (c_vals < 0).mean()
            if c_negative_ratio > 0.99:
                df_transactions['amount'] = df_transactions['amount'] * -1
                logger.info("Flipped amount signs (C transactions were negative, now positive)")

    return df_transactions


def _add_pending_transactions(
    df_transactions: pd.DataFrame,
    df_accounts: pd.DataFrame
) -> pd.DataFrame:
    """Adds synthetic transactions for estimatedPendingAmount."""
    if 'estimatedPendingAmount' not in df_accounts.columns:
        return df_transactions

    # Get accounts that exist in transactions AND have non-zero pending amount
    accounts_in_txn = df_transactions['accountId'].unique()
    pending_mask = (
        (df_accounts['accountId'].isin(accounts_in_txn)) &
        (df_accounts['estimatedPendingAmount'] != 0) &
        (df_accounts['estimatedPendingAmount'].notna())
    )
    accounts_with_pending = df_accounts[pending_mask].copy()

    if len(accounts_with_pending) == 0:
        return df_transactions

    # Create synthetic rollback transactions
    rollback_txns = pd.DataFrame({
        'accountId': accounts_with_pending['accountId'],
        'id': 'PENDING_ROLLBACK',
        'date': accounts_with_pending['balanceDateTime'],
        'amount': accounts_with_pending['estimatedPendingAmount'],
        'direction': np.where(
            accounts_with_pending['estimatedPendingAmount'] > 0,
            'D',  # Positive pending = debit
            'C'   # Negative pending = credit
        )
    })

    # Add any columns from original transactions that are missing
    for col in df_transactions.columns:
        if col not in rollback_txns.columns:
            rollback_txns[col] = np.nan

    # Combine and re-sort
    df_combined = pd.concat([df_transactions, rollback_txns], ignore_index=True)
    df_combined = df_combined.sort_values(['accountId', 'date', 'id']).reset_index(drop=True)

    logger.info(f"Added {len(rollback_txns)} synthetic pending transactions")

    return df_combined


def _calculate_running_balance(
    df_transactions: pd.DataFrame,
    df_accounts: pd.DataFrame
) -> pd.DataFrame:
    """Core logic to calculate running balance."""
    # Merge availableBalance AND balanceDateTime to transactions
    df_result = df_transactions.merge(
        df_accounts[['accountId', 'availableBalance', 'balanceDateTime']],
        on='accountId',
        how='left'
    )

    # Check for accounts without balance info
    missing_balance = df_result['availableBalance'].isna().sum()
    if missing_balance > 0:
        affected = df_result[df_result['availableBalance'].isna()]['accountId'].nunique()
        logger.warning(f"{affected} accounts have no availableBalance in DAccount. Their balance_after will be NaN.")

    # Calculate sum of amounts per account, BUT ONLY for transactions <= balanceDateTime
    # This ensures we calculate the correct 'initial_balance' relative to the snapshot.
    
    # 1. Filter mask for historical transactions (contributing to the snapshot)
    # Handle potentially missing dates or NaT by treating them as 'past' or 'future' carefully.
    # Here assuming standard flow: if date <= balanceDateTime, it's history.
    is_historical = df_result['date'] <= df_result['balanceDateTime']
    
    # 2. Sum historical amounts
    historical_sums = df_result[is_historical].groupby('accountId')['amount'].sum().reset_index()
    historical_sums.columns = ['accountId', 'historical_total_amount']

    df_result = df_result.merge(historical_sums, on='accountId', how='left')
    
    # Fill NaN for accounts with no historical txns (initial balance = snapshot)
    df_result['historical_total_amount'] = df_result['historical_total_amount'].fillna(0.0)

    # 3. Calculate initial balance (balance before first transaction in the dataset)
    # Logic: Snapshot = Initial + Sum(History)
    # Initial = Snapshot - Sum(History)
    df_result['initial_balance'] = df_result['availableBalance'] - df_result['historical_total_amount']

    # 4. Calculate cumulative sum per account (in sorted order) across ALL transactions
    df_result['cumsum_amount'] = df_result.groupby('accountId')['amount'].cumsum()

    # 5. Final balance after each transaction
    # Balance[t] = Initial + CumSum[t]
    # This works for both historical AND future transactions correctly.
    df_result['balance_after'] = df_result['initial_balance'] + df_result['cumsum_amount']

    # Clean up helper columns
    cols_to_drop = ['availableBalance', 'balanceDateTime', 'historical_total_amount', 'initial_balance', 'cumsum_amount']
    df_result = df_result.drop(columns=cols_to_drop)

    return df_result


def calculate_daily_balances(
    df_transactions: pd.DataFrame,
    df_accounts: pd.DataFrame,
    include_pending: bool = True
) -> pd.DataFrame:
    """Calculates daily closing balance and running date for each account.

    Args:
        df_transactions: Transaction data.
        df_accounts: Account data.
        include_pending: Whether to include estimatedPendingAmount.

    Returns:
        DataFrame with daily balances, columns: ['accountId', 'date', 'balance', 'running_date'].
        Sorted by (accountId, date).
    """
    # 1. Calculate balance after every transaction
    df_detailed = calculate_balance_per_transaction(
        df_transactions,
        df_accounts,
        include_pending=include_pending
    )

    # 2. Extract strictly date part
    # Handle NaT or non-date inputs gracefully if possible, but expecting valid inputs
    try:
        if not df_detailed.empty and not isinstance(df_detailed['date'].iloc[0], (datetime.date, type(pd.NaT))):
             df_detailed['date'] = pd.to_datetime(df_detailed['date']).dt.date
    except Exception:
        pass # Fallback

    if df_detailed.empty:
        return pd.DataFrame(columns=['accountId', 'date', 'balance', 'running_date'])

    # 3. Find the global "running_date" (max date in the dataset)
    running_date = df_detailed['date'].max()

    # 4. Group by accountId and date, taking the last 'balance_after'
    df_daily = df_detailed.groupby(['accountId', 'date']).last().reset_index()

    # 5. Select and rename columns
    df_daily = df_daily[['accountId', 'date', 'balance_after']].rename(
        columns={'balance_after': 'balance'}
    )

    # 6. Add running_date column
    df_daily['running_date'] = running_date

    return df_daily


class BalanceFeatureExtractor:
    """Extracts balance-related features for account embeddings.

    Pre-computes daily balances from transactions and accounts data,
    then provides efficient lookups.
    """

    def __init__(
        self,
        df_transactions: pd.DataFrame,
        df_accounts: pd.DataFrame,
        include_pending: bool = True
    ) -> None:
        """Initializes the BalanceFeatureExtractor.

        Args:
            df_transactions: Transaction data.
            df_accounts: Account data.
            include_pending: Whether to include estimatedPendingAmount in balance calculation.
        """
        # Normalize account column name
        df_acc = df_accounts.copy()
        if 'id' in df_acc.columns and 'accountId' not in df_acc.columns:
            df_acc = df_acc.rename(columns={'id': 'accountId'})

        # Compute daily balances
        logger.info("BalanceFeatureExtractor: Computing daily balances...")
        self.daily_balances = calculate_daily_balances(
            df_transactions,
            df_acc,
            include_pending=include_pending
        )

        # Store running_date
        if not df_transactions.empty:
            if hasattr(df_transactions['date'], 'dt'):
                self.running_date = df_transactions['date'].max().date()
            else:
                self.running_date = pd.to_datetime(df_transactions['date']).max().date()
        else:
            self.running_date = datetime.date.today()

        # Build lookup index: (accountId, date) -> balance
        self._balance_index: dict[tuple[str, Any], float] = {}
        for _, row in self.daily_balances.iterrows():
            key = (str(row['accountId']).upper(), row['date'])
            self._balance_index[key] = row['balance']

        # Build per-account sorted date lists for efficient range queries
        self._account_data: dict[str, pd.DataFrame] = {}
        for acc_id, group in self.daily_balances.groupby('accountId'):
            self._account_data[str(acc_id).upper()] = group.sort_values('date').reset_index(drop=True)

        logger.info(
            f"BalanceFeatureExtractor: Indexed {len(self._balance_index)} daily balances "
            f"for {len(self._account_data)} accounts. Running date: {self.running_date}"
        )

    def get_daily_balance(
        self,
        account_id: str,
        date: Any
    ) -> float | None:
        """Get the closing balance for an account on a specific date.

        Args:
            account_id: The account identifier.
            date: The date to look up.

        Returns:
            The balance on that date, or None if no data.
        """
        date = self._normalize_date(date)
        key = (str(account_id).upper(), date)
        return self._balance_index.get(key, None)

    def get_starting_balance(
        self,
        account_id: str,
        date: Any
    ) -> float | None:
        """Get the starting balance for an account on a specific date.

        This is the closing balance from the previous day with transactions.

        Args:
            account_id: The account identifier.
            date: The date to look up.

        Returns:
            The starting balance, or None if no data available.
        """
        date = self._normalize_date(date)
        acc_id = str(account_id).upper()
        if acc_id not in self._account_data:
            return None

        df = self._account_data[acc_id]
        # Find the most recent date before the given date
        before_mask = df['date'] < date
        if before_mask.any():
            prev_row = df[before_mask].iloc[-1]
            return float(prev_row['balance'])
        else:
            # No previous date, return earliest balance
            if len(df) > 0:
                return float(df.iloc[0]['balance'])
            return None

    def get_window_stats(
        self,
        account_id: str,
        start_date: Any,
        end_date: Any
    ) -> dict[str, float | None]:
        """Get summary statistics for an account's balance within a time window.

        Args:
            account_id: The account identifier.
            start_date: Start of the window (inclusive).
            end_date: End of the window (inclusive).

        Returns:
            Dictionary with statistics:
            - start_balance, end_balance, balance_change
            - min_balance, max_balance, avg_balance, balance_volatility
        """
        start_date = self._normalize_date(start_date)
        end_date = self._normalize_date(end_date)
        
        acc_id = str(account_id).upper()
        empty_result: dict[str, float | None] = {
            'start_balance': None,
            'end_balance': None,
            'balance_change': None,
            'min_balance': None,
            'max_balance': None,
            'avg_balance': None,
            'balance_volatility': None
        }

        if acc_id not in self._account_data:
            return empty_result

        df = self._account_data[acc_id]

        # Filter to window
        mask = (df['date'] >= start_date) & (df['date'] <= end_date)
        window_df = df[mask]

        if len(window_df) == 0:
            return empty_result

        balances = window_df['balance'].values
        
        start_balance = float(window_df.iloc[0]['balance'])
        end_balance = float(window_df.iloc[-1]['balance'])

        return {
            'start_balance': start_balance,
            'end_balance': end_balance,
            'balance_change': end_balance - start_balance,
            'min_balance': float(np.min(balances)),
            'max_balance': float(np.max(balances)),
            'avg_balance': float(np.mean(balances)),
            'balance_volatility': float(np.std(balances)) if len(balances) > 1 else 0.0
        }

    @staticmethod
    def _normalize_date(date: Any) -> datetime.date:
        """Helper to normalize date input to datetime.date object."""
        if isinstance(date, str):
            pd_ts = pd.to_datetime(date)
            # handle NaT
            if pd.isna(pd_ts):
               return datetime.date.today() # fallback
            return pd_ts.date()
        elif isinstance(date, pd.Timestamp):
            return date.date()
        elif isinstance(date, datetime.datetime):
            return date.date()
        return date

    def normalize_balance(self, balance: float | None) -> float:
        """Normalize a balance value using sign-aware log transform.

        Formula: sign(x) * log(1 + |x|)
        """
        if balance is None:
            return 0.0
        sign = 1 if balance >= 0 else -1
        return float(sign * np.log1p(abs(balance)))
