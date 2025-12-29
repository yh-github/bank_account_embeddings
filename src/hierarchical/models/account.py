"""Account Encoder module: Hierarchical aggregation."""

import torch
import torch.nn as nn

from .day import DayEncoder


class AccountEncoder(nn.Module):
    """Hierarchical Account Encoder.

    Architecture:
    1. Input: Sequence of Day Data (variable length).
    2. Shared DayEncoder transforms Daily Data -> Day Embeddings.
    3. Adds Day Positional Encodings + Calendar Embeddings.
    4. Sequence Transformer (or SetTransformer) aggregates Day Embeddings.
    5. Global Pooling (Mean).
    """

    def __init__(
        self,
        day_encoder: DayEncoder,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        max_days: int = 512,
        dropout: float = 0.1,
    ) -> None:
        """Initializes the AccountEncoder.

        Args:
            day_encoder: Pre-initialized DayEncoder instance.
            hidden_dim: Hidden dimension size.
            num_layers: Number of transformer layers for the account level.
            num_heads: Number of attention heads.
            max_days: Maximum number of days in the sequence (for positional encoding).
            dropout: Dropout rate.
        """
        super().__init__()

        self.day_encoder = day_encoder
        self.hidden_dim = hidden_dim

        # Input Projection (if Pyramid)
        start_dim = day_encoder.hidden_dim
        if start_dim != hidden_dim:
            self.input_proj = nn.Linear(start_dim, hidden_dim)
        else:
            self.input_proj = None

        # Positional Encoding for DAYS (The Sequence)
        self.pos_embedding = nn.Embedding(max_days, hidden_dim)

        # Calendar Embeddings
        # Month: 1-12 (0 padding) -> 13
        self.month_embedding = nn.Embedding(13, hidden_dim)
        # Weekend: 0 or 1 -> 2
        self.weekend_embedding = nn.Embedding(2, hidden_dim)

        # Standard Transformer for day sequence
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, batch_data: dict) -> torch.Tensor:
        """Forward pass.

        Args:
            batch_data: Dict containing 'pos', 'neg', 'meta' keys (from collate).
              meta: {'day_mask': [B, D], 'day_month': [B, D], 'day_weekend': [B, D]}

        Returns:
             Account Embeddings [B, hidden_dim]
        """
        # Unpack Meta
        meta = batch_data["meta"]
        day_mask = meta["day_mask"]  # [B, D]
        day_month = meta["day_month"]  # [B, D]
        day_weekend = meta["day_weekend"]  # [B, D]

        B, D = day_mask.shape

        # Helper to flatten dict of tensors
        def flatten_stream(stream_data: dict) -> dict:
            flat = {}
            for k, v in stream_data.items():
                if v is None:
                    flat[k] = None
                    continue
                # v shape: [B, D, ...] -> [B*D, ...]
                flat[k] = v.view(B * D, *v.shape[2:])
            return flat

        # 1. Flatten inputs for DayEncoder
        # We process all days of all accounts in parallel sharing weights
        flat_pos = flatten_stream(batch_data["pos"])
        flat_neg = flatten_stream(batch_data["neg"])

        # 2. Encode Days
        # Output: [B*D, DayDim]
        flat_day_embs = self.day_encoder(flat_pos, flat_neg)

        # 3. Reshape back to Sequence
        # [B, D, DayDim]
        day_embs = flat_day_embs.view(B, D, -1)

        # Project if needed
        if self.input_proj is not None:
            day_embs = self.input_proj(day_embs)

        # Now day_embs is [B, D, AccountDim]

        # 4. Add Embeddings
        # Positions
        positions = torch.arange(D, device=day_embs.device).unsqueeze(0).expand(B, D)
        pos_emb = self.pos_embedding(positions)

        # Calendar
        month_emb = self.month_embedding(day_month)
        weekend_emb = self.weekend_embedding(day_weekend)

        # Combine
        sequence_input = day_embs + pos_emb + month_emb + weekend_emb

        # 5. Transformer Sequence
        key_padding_mask = ~day_mask  # True for PADDED positions

        processed = self.transformer(
            sequence_input, src_key_padding_mask=key_padding_mask
        )

        # 6. Global Pooling
        mask_expanded = day_mask.unsqueeze(-1).float()
        sum_emb = (processed * mask_expanded).sum(dim=1)
        cnt = mask_expanded.sum(dim=1).clamp(min=1e-9)

        account_emb = sum_emb / cnt

        return self.norm(self.output_proj(account_emb))
