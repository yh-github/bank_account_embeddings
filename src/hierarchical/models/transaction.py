"""Transaction Encoder module."""

import torch
import torch.nn as nn


class TransactionEncoder(nn.Module):
    """Encodes individual transactions using entity embeddings and numerical features.

    Supports categorical features (Category Group, Sub Category, Counter Party)
    and numerical features (Amount, Date, Balance).
    """

    def __init__(
        self,
        num_categories_group: int,
        num_categories_sub: int,
        num_counter_parties: int,
        category_dim: int = 32,
        counter_party_dim: int = 64,
        embedding_dim: int = 128,
        use_balance: bool = False,
        use_counter_party: bool = True,
        use_amount_binning: bool = False,
        num_amount_bins: int = 64,
        amount_bin_min: float = 0.0,
        amount_bin_max: float = 15.0,  # Cover log1p($1M) ~ 13.8
        dropout: float = 0.1,
    ) -> None:
        """Initializes the TransactionEncoder.

        Args:
            num_categories_group: Size of category group vocabulary.
            num_categories_sub: Size of sub-category vocabulary.
            num_counter_parties: Size of counter party vocabulary.
            category_dim: Embedding dimension for categories.
            counter_party_dim: Embedding dimension for counter parties.
            embedding_dim: Output embedding dimension.
            use_balance: Whether to use balance features.
            use_counter_party: Whether to use counter party embeddings.
            use_amount_binning: Whether to use binning for amounts (vs linear).
            num_amount_bins: Number of bins for amount discretization.
            amount_bin_min: Minimum value for amount binning.
            amount_bin_max: Maximum value for amount binning.
            dropout: Dropout rate.
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.use_balance = use_balance
        self.use_counter_party = use_counter_party
        self.use_amount_binning = use_amount_binning

        self.num_amount_bins = num_amount_bins
        self.amount_bin_min = amount_bin_min
        self.amount_bin_max = amount_bin_max

        # Categorical embeddings
        self.cat_group_emb = nn.Embedding(
            num_categories_group, category_dim, padding_idx=0
        )
        self.cat_sub_emb = nn.Embedding(num_categories_sub, category_dim, padding_idx=0)

        if use_counter_party:
            self.counter_party_emb = nn.Embedding(
                num_counter_parties, counter_party_dim, padding_idx=0
            )

        # Numerical projections
        if use_amount_binning:
            # Embedding for discrete bins
            self.amount_emb = nn.Embedding(num_amount_bins, 16)
        else:
            # Linear projection for continuous
            self.amount_proj = nn.Linear(1, 16)

        # Date encoding: day of week (7) + day of month (31)
        self.date_day_of_week_emb = nn.Embedding(7, 8)
        self.date_day_of_month_emb = nn.Embedding(31, 8)

        # Balance features
        if use_balance:
            # starting_balance (1) + stats (6) = 7 features
            self.balance_proj = nn.Linear(7, 16)

        # Combine all features
        total_dim = category_dim * 2
        if use_counter_party:
            total_dim += counter_party_dim

        total_dim += 16  # Amount
        total_dim += 8 * 2  # Date (dow, dom)

        if use_balance:
            total_dim += 16

        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        category_group_ids: torch.Tensor,
        category_sub_ids: torch.Tensor,
        counter_party_ids: torch.Tensor | None = None,
        amounts: torch.Tensor | None = None,
        dates: torch.Tensor | None = None,
        balance_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            category_group_ids: [Batch, Seq]
            category_sub_ids: [Batch, Seq]
            counter_party_ids: [Batch, Seq] or None
            amounts: [Batch, Seq] or [Batch, Seq, 1] - log-normalized amounts
            dates: [Batch, Seq, 4] - [dow, dom, month, linear]
            balance_features: [Batch, Seq, 7] or None

        Returns:
            Encoded transactions: [Batch, Seq, embedding_dim]
        """
        # Categorical embeddings
        cat_group_emb = self.cat_group_emb(category_group_ids)  # [B, L, cat_dim]
        cat_sub_emb = self.cat_sub_emb(category_sub_ids)  # [B, L, cat_dim]

        features = [cat_group_emb, cat_sub_emb]

        if self.use_counter_party:
            if counter_party_ids is None:
                raise ValueError("Model expects counter_party_ids but None provided.")
            cp_emb = self.counter_party_emb(counter_party_ids)  # [B, L, cp_dim]
            features.append(cp_emb)

        # Amount Encoding
        if amounts is None:
            raise ValueError("Model expects amounts but None provided.")

        if self.use_amount_binning:
            # Discretize amounts into bins
            val = amounts
            if val.dim() == 3:
                val = val.squeeze(-1)

            # Clip to range
            val = val.clamp(min=self.amount_bin_min, max=self.amount_bin_max)
            # Normalize to 0-1
            val_norm = (val - self.amount_bin_min) / (
                self.amount_bin_max - self.amount_bin_min
            )
            # Scale to indices
            bin_indices = (val_norm * (self.num_amount_bins - 1)).long()

            amount_emb = self.amount_emb(bin_indices)
        else:
            # Linear Projection
            amount_input = amounts
            if amount_input.dim() == 2:
                amount_input = amount_input.unsqueeze(-1)
            amount_emb = self.amount_proj(amount_input)

        features.append(amount_emb)

        # Date encoding
        if dates is None:
            raise ValueError("Model expects dates but None provided.")

        dow_emb = self.date_day_of_week_emb(dates[..., 0].long())
        dom_emb = self.date_day_of_month_emb(dates[..., 1].long())

        features.extend([dow_emb, dom_emb])

        if self.use_balance:
            if balance_features is not None:
                balance_emb = self.balance_proj(balance_features)
                features.append(balance_emb)
            else:
                # Balance features enabled but missing -> PAD with zeros
                B, L = amounts.shape[:2]
                device = amounts.device
                features.append(torch.zeros(B, L, 16, device=device))

        # Concatenate and fuse
        combined = torch.cat(features, dim=-1)
        txn_embeddings = self.fusion(combined)
        txn_embeddings = self.norm(txn_embeddings)

        return txn_embeddings
