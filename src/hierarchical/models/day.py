"""Day Encoder module for aggregating daily transactions."""

import torch
import torch.nn as nn


from .transaction import TransactionEncoder


class DayEncoder(nn.Module):
    """Encodes a set of transactions (representing one day) into a single Day Vector.

    Architecture:
    1. Encode individual transactions (Category + Date + Amount).
    2. Zero Positional Encoding (Permutation Invariant).
    3. Transformer or SetTransformer Encoder.
    4. Pooling (Mean) masked by valid transactions.
    """

    def __init__(
        self,
        txn_encoder: TransactionEncoder,
        hidden_dim: int = 128,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1
    ) -> None:
        """Initializes the DayEncoder.

        Args:
            txn_encoder: Pre-initialized TransactionEncoder instance.
            hidden_dim: Hidden dimension size.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            dropout: Dropout rate.
        """
        super().__init__()

        self.txn_encoder = txn_encoder
        self.hidden_dim = hidden_dim

        # Input Projection (if Pyramid)
        start_dim = txn_encoder.embedding_dim
        if start_dim != hidden_dim:
            self.input_proj = nn.Linear(start_dim, hidden_dim)
        else:
            self.input_proj = None
            
        self.use_checkpointing = False # Default Off

        # Standard Transformer Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        # self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers) 
        # Replaced with ModuleList to support checkpointing
        import copy
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])

        # Projection to combine Inflow + Outflow
        # Concatenate: Dim * 2 -> Dim
        self.combine_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)

    def _encode_stream(self, stream_data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encodes a single stream (Pos or Neg).

        Args:
            stream_data: Dictionary containing stream tensors:
                - cat_group, cat_sub, amounts, dates, mask, has_data
                - (Optional) cat_cp, balance

        Returns:
            Pooled vector [N_Days, Dim]
        """
        # Unpack inputs
        cat_group = stream_data['cat_group']
        cat_sub = stream_data['cat_sub']
        cat_cp = stream_data.get('cat_cp')
        amounts = stream_data['amounts']
        dates = stream_data['dates']
        balance = stream_data.get('balance')
        mask = stream_data['mask']
        has_data = stream_data['has_data']

        # 1. Encode Transactions
        # [N, T, Dim]
        txn_emb = self.txn_encoder(
            cat_group, cat_sub, cat_cp, amounts, dates, balance
        )

        # Project if needed
        if self.input_proj is not None:
             txn_emb = self.input_proj(txn_emb)

        input_mask = mask.bool()
        if (~input_mask).all():
            # All padded - return zero
            return torch.zeros(mask.size(0), self.hidden_dim, device=txn_emb.device)

        # 2. Transformer
        # Transformer expects True for IGNORED positions
        key_padding_mask = ~input_mask

        # processed = self.transformer(txn_emb, src_key_padding_mask=key_padding_mask)
        
        # Manual Layer Loop with Optional Checkpointing
        x = txn_emb
        for layer in self.layers:
            if self.use_checkpointing and x.requires_grad:
                # checkpoint requires inputs to require_grad
                x = torch.utils.checkpoint.checkpoint(layer, x, src_key_padding_mask=key_padding_mask, use_reentrant=False)
            else:
                x = layer(x, src_key_padding_mask=key_padding_mask)
        
        processed = x

        # 3. Pooling (Mean)
        mask_expanded = input_mask.unsqueeze(-1).float()
        sum_emb = (processed * mask_expanded).sum(dim=1)
        cnt = mask_expanded.sum(dim=1).clamp(min=1e-9)

        pooled = sum_emb / cnt

        # If has_data is False (day level), ensure zero
        has_data_mask = has_data.unsqueeze(-1).float()
        return pooled * has_data_mask

    def forward(
        self,
        pos_data: dict[str, torch.Tensor],
        neg_data: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Forward pass.

        Args:
             pos_data: Dictionary of flattened tensors [B*D, T] for positive stream.
             neg_data: Dictionary of flattened tensors [B*D, T] for negative stream.

        Returns:
            Day embeddings [B*D, hidden_dim]
        """
        # Encode Positive Stream
        h_pos = self._encode_stream(pos_data)

        # Encode Negative Stream
        h_neg = self._encode_stream(neg_data)

        # Combine
        combined = torch.cat([h_pos, h_neg], dim=-1)
        day_emb = self.combine_proj(combined)

        return self.norm(day_emb)
