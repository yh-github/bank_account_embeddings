"""Categorical vocabulary management and feature extraction."""

import logging
import os
import pickle
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CategoricalVocabulary:
    """Manages vocabularies for categorical features.

    Attributes:
        token2id: Mapping from token string to integer ID.
        id2token: Mapping from integer ID to token string.
        next_id: Next available ID for a new token.
    """

    def __init__(self) -> None:
        """Initializes a new CategoricalVocabulary."""
        self.token2id: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
        self.id2token: dict[int, str] = {0: "<PAD>", 1: "<UNK>"}
        self.next_id: int = 2

    def fit(self, tokens: list[Any]) -> None:
        """Builds the vocabulary from a list of tokens.

        Args:
            tokens: A list of tokens (strings, numbers, etc.) to include.
                None, empty strings, and NaNs are ignored.
        """
        unique_tokens = set(tokens) - {None, "", np.nan}
        # Sort for deterministic vocabulary generation
        for token in sorted(unique_tokens, key=str):
            token_str = str(token)
            if token_str not in self.token2id:
                self.token2id[token_str] = self.next_id
                self.id2token[self.next_id] = token_str
                self.next_id += 1

    def encode(self, token: Any) -> int:
        """Converts a token to its integer ID.

        Args:
            token: The token to encode.

        Returns:
            The integer ID of the token. Returns 0 (PAD) for empty/null tokens,
            and 1 (UNK) for tokens not in the vocabulary.
        """
        if (
            token is None
            or token == ""
            or (isinstance(token, float) and np.isnan(token))
        ):
            return 0  # PAD
        return self.token2id.get(str(token), 1)  # UNK if not found

    def decode(self, token_id: int) -> str:
        """Converts an integer ID back to its token string.

        Args:
            token_id: The ID to decode.

        Returns:
            The string representation of the token, or '<UNK>' if unknown.
        """
        return self.id2token.get(token_id, "<UNK>")

    def __len__(self) -> int:
        return len(self.token2id)

    def save(self, path: str) -> None:
        """Saves the vocabulary to a file using pickle.

        Args:
            path: The file path to save to.
        """
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "token2id": self.token2id,
                    "id2token": self.id2token,
                    "next_id": self.next_id,
                },
                f,
            )

    @classmethod
    def load(cls, path: str) -> "CategoricalVocabulary":
        """Loads a vocabulary from a pickle file.

        Args:
            path: The file path to load from.

        Returns:
            The loaded CategoricalVocabulary instance.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        vocab = cls()
        vocab.token2id = data["token2id"]
        vocab.id2token = data["id2token"]
        vocab.next_id = data["next_id"]
        return vocab


class V4FeatureExtractor:
    """Extracts and normalizes features for the V4 model.

    Handles date features, balance features, and amount normalization.
    """

    def __init__(
        self,
        cat_group_vocab: CategoricalVocabulary,
        cat_sub_vocab: CategoricalVocabulary,
        counter_party_vocab: CategoricalVocabulary,
        balance_extractor: Any = None,
        epoch_date: datetime = datetime(2020, 1, 1),
    ) -> None:
        """Initializes the feature extractor.

        Args:
            cat_group_vocab: Vocabulary for category groups.
            cat_sub_vocab: Vocabulary for sub-categories.
            counter_party_vocab: Vocabulary for counter parties.
            balance_extractor: Optional extractor for balance features.
            epoch_date: Reference date for normalizing temporal features.
        """
        self.cat_group_vocab = cat_group_vocab
        self.cat_sub_vocab = cat_sub_vocab
        self.counter_party_vocab = counter_party_vocab
        self.balance_extractor = balance_extractor
        self.epoch_date = epoch_date

    def extract_date_features(self, dates: pd.Series) -> np.ndarray:
        """Extracts date features from a series of dates.

        Features: [day_of_week, day_of_month, month, normalized_days]

        Args:
            dates: A pandas Series containing datetime objects.

        Returns:
            A [N, 4] numpy array of float32 features.
        """
        dates_dt = pd.to_datetime(dates)

        dow = dates_dt.dt.dayofweek.values  # 0-6
        dom = dates_dt.dt.day.values - 1  # 0-30
        month = dates_dt.dt.month.values - 1  # 0-11

        # Days since epoch, normalized to [0, 1] range (roughly per year)
        days_since_epoch = (dates_dt - self.epoch_date).dt.days.values
        norm_days = days_since_epoch / 365.0

        return np.stack([dow, dom, month, norm_days], axis=1).astype(np.float32)

    def extract_balance_features(
        self, account_id: str, dates: pd.Series
    ) -> np.ndarray | None:
        """Extracts balance features for a given account and dates.

        Features: [starting_bal, start, end, min, max, avg, std]

        Args:
            account_id: The account ID string.
            dates: A pandas Series of transaction dates.

        Returns:
            A [N, 7] numpy array of float32 features, or None if no balance
            extractor is available. Returns zeros on extraction failure.
        """
        if self.balance_extractor is None:
            return None

        try:
            # Get balance stats for this account's date range
            balance_stats = []
            for date in dates:
                # Calculate window dates (7-day lookback, ending YESTERDAY)
                # This prevents leakage of today's EOD balance into today's features
                current_tx_date = pd.to_datetime(date)
                end_date = current_tx_date - pd.Timedelta(days=1)
                start_date = end_date - pd.Timedelta(days=7)

                # get_window_stats should return a dict
                stats_dict = self.balance_extractor.get_window_stats(
                    account_id, start_date, end_date
                )

                start_bal_val = (
                    self.balance_extractor.get_starting_balance(account_id, date) or 0.0
                )

                stats = [
                    stats_dict.get("start_balance", 0.0) or 0.0,
                    stats_dict.get("end_balance", 0.0) or 0.0,
                    stats_dict.get("min_balance", 0.0) or 0.0,
                    stats_dict.get("max_balance", 0.0) or 0.0,
                    stats_dict.get("avg_balance", 0.0) or 0.0,
                    stats_dict.get("balance_volatility", 0.0) or 0.0,
                    start_bal_val,
                ]
                balance_stats.append(stats)

            return np.array(balance_stats, dtype=np.float32)
        except Exception as e:
            logger.error(f"Balance extraction failed for {account_id}: {e}")
            return np.zeros((len(dates), 7), dtype=np.float32)

    @staticmethod
    def normalize_amounts(amounts: np.ndarray) -> np.ndarray:
        """Normalizes transaction amounts using log transformation.

        Method: sign(x) * ((log(1 + |x|) - mean) / std)

        Args:
            amounts: A numpy array of transaction amounts.

        Returns:
            A standardized numpy array of float32 values.
        """
        # Preserve sign, apply log to absolute value
        sign = np.sign(amounts)
        abs_amounts = np.abs(amounts)
        log_amounts = np.log1p(abs_amounts)  # log(1 + x) to handle zeros

        # Standardize
        mean = log_amounts.mean()
        std = log_amounts.std() + 1e-9
        normalized = (log_amounts - mean) / std

        return (sign * normalized).astype(np.float32)


def build_vocabularies(
    df: pd.DataFrame, save_dir: str | None = None
) -> tuple[CategoricalVocabulary, CategoricalVocabulary, CategoricalVocabulary]:
    """Builds vocabularies from transaction data.

    Supports multiple column naming conventions (e.g. 'personeticsCategoryGroupId',
    'p_categoryGroupId', etc.).

    Args:
        df: DataFrame containing transaction columns.
        save_dir: Optional directory path to save the generated vocabularies.

    Returns:
        A tuple of (cat_group_vocab, cat_sub_vocab, counter_party_vocab).
    """
    logger.info("Building vocabularies...")

    cat_group_vocab = CategoricalVocabulary()
    cat_sub_vocab = CategoricalVocabulary()
    counter_party_vocab = CategoricalVocabulary()

    # Helper to find column
    def fit_vocab(vocab: CategoricalVocabulary, col_names: list[str]) -> None:
        for col in col_names:
            if col in df.columns:
                vocab.fit(df[col].dropna().unique().tolist())
                return

    fit_vocab(
        cat_group_vocab,
        ["personeticsCategoryGroupId", "p_categoryGroupId", "categoryGroupId"],
    )
    fit_vocab(
        cat_sub_vocab, ["personeticsSubCategoryId", "p_subCategoryId", "subCategoryId"]
    )
    fit_vocab(counter_party_vocab, ["deviceId", "counter_party", "counterParty"])

    logger.info(f"  Category groups: {len(cat_group_vocab)}")
    logger.info(f"  Sub-categories: {len(cat_sub_vocab)}")
    logger.info(f"  Counter parties: {len(counter_party_vocab)}")

    # Save if requested
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        cat_group_vocab.save(os.path.join(save_dir, "cat_group_vocab.pkl"))
        cat_sub_vocab.save(os.path.join(save_dir, "cat_sub_vocab.pkl"))
        counter_party_vocab.save(os.path.join(save_dir, "counter_party_vocab.pkl"))
        logger.info(f"Vocabularies saved to {save_dir}")

    return cat_group_vocab, cat_sub_vocab, counter_party_vocab


def load_vocabularies(
    save_dir: str,
) -> tuple[CategoricalVocabulary, CategoricalVocabulary, CategoricalVocabulary]:
    """Loads vocabularies from a directory.

    Args:
        save_dir: The directory containing the vocabulary files.

    Returns:
        A tuple of (cat_group_vocab, cat_sub_vocab, counter_party_vocab).
    """
    cat_group_vocab = CategoricalVocabulary.load(
        os.path.join(save_dir, "cat_group_vocab.pkl")
    )
    cat_sub_vocab = CategoricalVocabulary.load(
        os.path.join(save_dir, "cat_sub_vocab.pkl")
    )
    counter_party_vocab = CategoricalVocabulary.load(
        os.path.join(save_dir, "counter_party_vocab.pkl")
    )
    return cat_group_vocab, cat_sub_vocab, counter_party_vocab
