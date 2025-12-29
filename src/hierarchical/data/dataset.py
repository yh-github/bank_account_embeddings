"""Hierarchical dataset and collation logic."""

import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .vocab import CategoricalVocabulary, V4FeatureExtractor

logger = logging.getLogger(__name__)


class HierarchicalDataset(Dataset):
    """Dataset for the Hierarchical Model.

    Processes transactions into daily sequences of positive/negative streams.
    """

    def __init__(
        self,
        transactions: pd.DataFrame,
        cat_group_vocab: CategoricalVocabulary,
        cat_sub_vocab: CategoricalVocabulary,
        counter_party_vocab: CategoricalVocabulary,
        balance_extractor: Any = None,
        max_days: int = 180,
        max_txns_per_day: int = 20,
        min_days: int = 5,
        account_ids: list[str] | None = None,
        epoch_date: datetime = datetime(2020, 1, 1),
    ) -> None:
        """Initializes the HierarchicalDataset.

        Args:
            transactions: DataFrame containing transaction data.
            cat_group_vocab: Vocabulary for category groups.
            cat_sub_vocab: Vocabulary for sub-categories.
            counter_party_vocab: Vocabulary for counter parties.
            balance_extractor: Optional balance feature extractor.
            max_days: Maximum number of days to look back.
            max_txns_per_day: Maximum transactions per stream per day.
            min_days: Minimum active days required for an account.
            account_ids: List of account IDs to include (for filtering).
            epoch_date: Reference date for temporal normalization.
        """
        self.max_days = max_days
        self.max_txns_per_day = max_txns_per_day
        self.min_days = min_days
        self.epoch_date = epoch_date

        self.feature_extractor = V4FeatureExtractor(
            cat_group_vocab,
            cat_sub_vocab,
            counter_party_vocab,
            balance_extractor,
            epoch_date,
        )

        # Filter by account IDs if provided
        if account_ids is not None:
            transactions = transactions[
                transactions["accountId"].isin(account_ids)
            ].copy()

        # Ensure date column exists and is datetime
        if "date" not in transactions.columns:
            if "transactionDate" in transactions.columns:
                transactions["date"] = transactions["transactionDate"]
        transactions["date"] = pd.to_datetime(transactions["date"])

        # Sort by Date within Account
        self.transactions = transactions.sort_values(["accountId", "date"])

        # Ensure accountId is string
        self.transactions["accountId"] = self.transactions["accountId"].astype(str)

        # Group by Account
        self.grouped = self.transactions.groupby("accountId")

        # Identify valid accounts
        self.accounts: list[str] = []
        for account_id, group in self.grouped:
            unique_days = group["date"].dt.date.nunique()
            if unique_days >= min_days:
                self.accounts.append(str(account_id))

        logger.info(
            f"HierarchicalDataset: {len(self.accounts)} accounts (min_days={min_days})"
        )

    def __len__(self) -> int:
        return len(self.accounts)

    def _pack_stream(
        self, df: pd.DataFrame, account_id: str
    ) -> dict[str, torch.Tensor | int | float] | None:
        """Packs a stream of transactions into tensors."""
        n_tx = len(df)
        if n_tx == 0:
            return None

        def get_col(
            df_: pd.DataFrame, candidates: list[str], default: Any = None
        ) -> Any:
            for c in candidates:
                if c in df_.columns:
                    return df_[c]
            return [default] * len(df_)

        # Encode categorical features
        c_grp = [
            self.feature_extractor.cat_group_vocab.encode(x)
            for x in get_col(
                df, ["personeticsCategoryGroupId", "p_categoryGroupId"], "UNK"
            )
        ]
        c_sub = [
            self.feature_extractor.cat_sub_vocab.encode(x)
            for x in get_col(df, ["personeticsSubCategoryId", "p_subCategoryId"], "UNK")
        ]
        c_cp = [
            self.feature_extractor.counter_party_vocab.encode(x)
            for x in get_col(df, ["deviceId", "counter_party"], "UNK")
        ]

        # Extract numerical features
        amounts = self.feature_extractor.normalize_amounts(df["amount"].values)
        d_feats = self.feature_extractor.extract_date_features(df["date"])

        bal_feats = None
        if self.feature_extractor.balance_extractor:
            feats = self.feature_extractor.extract_balance_features(
                account_id, df["date"]
            )
            if feats is not None:
                bal_feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

        total_vol = np.sum(np.abs(df["amount"].values))
        log_total_vol = float(np.log1p(total_vol))

        return {
            "cat_group": torch.tensor(c_grp, dtype=torch.long),
            "cat_sub": torch.tensor(c_sub, dtype=torch.long),
            "cat_cp": torch.tensor(c_cp, dtype=torch.long),
            "amounts": torch.tensor(amounts, dtype=torch.float32),
            "dates": torch.tensor(d_feats, dtype=torch.float32),
            "balance": torch.tensor(bal_feats, dtype=torch.float32)
            if bal_feats is not None
            else None,
            "n_txns": n_tx,
            "log_total_volume": torch.tensor(log_total_vol, dtype=torch.float32),
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        account_id = self.accounts[idx]
        group = self.grouped.get_group(account_id)

        # Split into days
        daily_groups_iter = group.groupby(group["date"].dt.date)
        daily_groups = [g for _, g in daily_groups_iter]

        # Truncate to max history
        if len(daily_groups) > self.max_days:
            daily_groups = daily_groups[-self.max_days :]

        day_items = []
        day_dates_list = []

        for daily_txns in daily_groups:
            # Split into POS (credits) and NEG (debits)
            pos_txns = daily_txns[daily_txns["amount"] > 0].copy()
            neg_txns = daily_txns[daily_txns["amount"] < 0].copy()

            # Sort by absolute amount (descending for POS, ascending (most negative) for NEG)
            # Actually, intention is "largest magnitude".
            # Exp 14 logic: pos desc, neg asc (meaning -100 before -5).
            pos_txns = pos_txns.sort_values("amount", ascending=False)
            neg_txns = neg_txns.sort_values("amount", ascending=True)

            # Truncate
            limit = self.max_txns_per_day
            if len(pos_txns) > limit:
                pos_txns = pos_txns.iloc[:limit]
            if len(neg_txns) > limit:
                neg_txns = neg_txns.iloc[:limit]

            day_pos = self._pack_stream(pos_txns, account_id)
            day_neg = self._pack_stream(neg_txns, account_id)

            # Calendar metadata
            d_obj = daily_txns["date"].iloc[0]
            is_weekend = 1 if d_obj.weekday() >= 5 else 0
            month = d_obj.month

            day_items.append(
                {
                    "pos": day_pos,
                    "neg": day_neg,
                    "meta": {"is_weekend": is_weekend, "month": month},
                    "day_offset": (d_obj - self.epoch_date).days,
                }
            )
            day_dates_list.append((d_obj - self.epoch_date).days)

        return {
            "account_id": account_id,
            "days": day_items,
            "day_dates": day_dates_list,  # Keep as list, collate converts
        }


def collate_hierarchical(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collates a batch of HierarchicalDataset items into padded tensors.

    Args:
        batch: List of items from HierarchicalDataset.__getitem__.

    Returns:
        A dictionary containing padded tensors for 'pos' and 'neg' streams,
        and metadata.
    """
    batch_size = len(batch)
    max_days = max(len(item["days"]) for item in batch)

    # Determine max sequence lengths per stream
    max_tx_pos = 1
    max_tx_neg = 1
    for item in batch:
        for day in item["days"]:
            if day["pos"]:
                max_tx_pos = max(max_tx_pos, day["pos"]["n_txns"])
            if day["neg"]:
                max_tx_neg = max(max_tx_neg, day["neg"]["n_txns"])

    # Check for optional features presence
    has_balance = False
    has_cp = False

    # Inspect the first valid day in the batch to detect optional features
    for item in batch:
        for day in item["days"]:
            for stream in [day["pos"], day["neg"]]:
                if stream:
                    if stream.get("balance") is not None:
                        has_balance = True
                    if stream.get("cat_cp") is not None:
                        has_cp = True
            if has_balance or has_cp:
                break  # Optimization: stop checking once found (assuming consistency)
        if has_balance or has_cp:
            break

    def init_stream_tensors(T_dim: int) -> dict[str, torch.Tensor | None]:
        tensors: dict[str, torch.Tensor | None] = {
            "cat_group": torch.zeros(batch_size, max_days, T_dim, dtype=torch.long),
            "cat_sub": torch.zeros(batch_size, max_days, T_dim, dtype=torch.long),
            "amounts": torch.zeros(batch_size, max_days, T_dim, dtype=torch.float32),
            "dates": torch.zeros(batch_size, max_days, T_dim, 4, dtype=torch.float32),
            "mask": torch.zeros(batch_size, max_days, T_dim, dtype=torch.bool),
            "has_data": torch.zeros(batch_size, max_days, dtype=torch.bool),
            "log_total_volume": torch.zeros(batch_size, max_days, dtype=torch.float32),
        }

        tensors["cat_cp"] = (
            torch.zeros(batch_size, max_days, T_dim, dtype=torch.long)
            if has_cp
            else None
        )
        tensors["balance"] = (
            torch.zeros(batch_size, max_days, T_dim, 7, dtype=torch.float32)
            if has_balance
            else None
        )

        return tensors

    pos_tensors = init_stream_tensors(max_tx_pos)
    neg_tensors = init_stream_tensors(max_tx_neg)

    day_mask = torch.zeros(batch_size, max_days, dtype=torch.bool)
    day_timesteps = torch.zeros(batch_size, max_days, dtype=torch.long)
    day_month = torch.zeros(batch_size, max_days, dtype=torch.long)
    day_weekend = torch.zeros(batch_size, max_days, dtype=torch.long)

    for i, item in enumerate(batch):
        days = item["days"]
        n_days = len(days)

        day_mask[i, :n_days] = True

        # Determine day_dates
        # item['day_dates'] can be a list of ints or a tensor
        day_dates_raw = item["day_dates"]
        if isinstance(day_dates_raw, torch.Tensor):
            day_dates_val = day_dates_raw.to(dtype=torch.long)
        else:
            day_dates_val = torch.tensor(day_dates_raw, dtype=torch.long)
        day_timesteps[i, :n_days] = day_dates_val

        for d, day in enumerate(days):
            day_month[i, d] = day["meta"]["month"]
            day_weekend[i, d] = day["meta"].get("is_weekend", 0)

            def fill_stream(
                target_dict: dict[str, torch.Tensor | None],
                source_data: dict[str, Any] | None,
            ) -> None:
                if source_data is None:
                    return

                n = source_data["n_txns"]
                target_dict["has_data"][i, d] = True
                target_dict["mask"][i, d, :n] = True

                # Helper to ensure tensor
                def as_tensor(val, dtype):
                    if isinstance(val, (list, np.ndarray)):
                        return torch.tensor(val, dtype=dtype)
                    return val

                # Fill tensors
                target_dict["cat_group"][i, d, :n] = as_tensor(
                    source_data["cat_group"], torch.long
                )
                target_dict["cat_sub"][i, d, :n] = as_tensor(
                    source_data["cat_sub"], torch.long
                )
                target_dict["amounts"][i, d, :n] = as_tensor(
                    source_data["amounts"], torch.float32
                )
                target_dict["dates"][i, d, :n] = as_tensor(
                    source_data["dates"], torch.float32
                )

                if (
                    target_dict["cat_cp"] is not None
                    and source_data.get("cat_cp") is not None
                ):
                    target_dict["cat_cp"][i, d, :n] = as_tensor(
                        source_data["cat_cp"], torch.long
                    )

                if (
                    target_dict["balance"] is not None
                    and source_data.get("balance") is not None
                ):
                    target_dict["balance"][i, d, :n] = as_tensor(
                        source_data["balance"], torch.float32
                    )

                if "log_total_volume" in source_data:
                    # Should be a scalar tensor or float. Force float for numpy scalars.
                    target_dict["log_total_volume"][i, d] = float(
                        source_data["log_total_volume"]
                    )

            fill_stream(pos_tensors, day["pos"])
            fill_stream(neg_tensors, day["neg"])

    return {
        "account_id": [item["account_id"] for item in batch],
        "pos": pos_tensors,
        "neg": neg_tensors,
        "meta": {
            "day_mask": day_mask,
            "day_dates": day_timesteps,
            "day_month": day_month,
            "day_weekend": day_weekend,
        },
    }
