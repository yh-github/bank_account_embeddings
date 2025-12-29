"""Contrastive loss functions."""

import torch
import torch.nn.functional as F


def contrastive_loss(
    emb1: torch.Tensor, emb2: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    """Computes the SimCLR (InfoNCE) loss.

    Args:
        emb1: First view embeddings [Batch, Dim].
        emb2: Second view embeddings [Batch, Dim].
        temperature: Softmax temperature scaling.

    Returns:
        Scalar loss tensor.
    """
    emb1 = F.normalize(emb1, dim=1)
    emb2 = F.normalize(emb2, dim=1)

    # Cosine Similarity Matrix [Batch, Batch]
    logits = torch.matmul(emb1, emb2.T) / temperature
    labels = torch.arange(len(emb1), device=emb1.device)

    loss_1 = F.cross_entropy(logits, labels)
    loss_2 = F.cross_entropy(logits.T, labels)

    return (loss_1 + loss_2) / 2


def contrastive_loss_with_hard_negatives(
    emb_anchor: torch.Tensor,
    emb_positive: torch.Tensor,
    emb_hard_neg: torch.Tensor | None,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Computes SimCLR loss with explicit hard negatives.

    Args:
        emb_anchor: Anchor embeddings (B, D).
        emb_positive: Positive pair embeddings (B, D).
        emb_hard_neg: Hard negative embeddings (B, D), or None.
        temperature: Softmax temperature.

    Returns:
        Scalar loss tensor.
    """
    if emb_hard_neg is None:
        return contrastive_loss(emb_anchor, emb_positive, temperature)

    emb_anchor = F.normalize(emb_anchor, dim=1)
    emb_positive = F.normalize(emb_positive, dim=1)
    emb_hard_neg = F.normalize(emb_hard_neg, dim=1)

    # 1. Similarity block: Anchor vs Batch Positives (B, B)
    logits_batch = torch.matmul(emb_anchor, emb_positive.T) / temperature

    # 2. Similarity vector: Anchor vs its Hard Negative (B, 1)
    # Only anchor[i] vs hard_neg[i] matters
    logits_hard = (
        torch.einsum("nc,nc->n", [emb_anchor, emb_hard_neg]).unsqueeze(-1) / temperature
    )

    # 3. Concatenate: [Batch_Positives, Hard_Negative] -> (B, B+1)
    # The positive for anchor[i] is at index i in logits_batch.
    # So the target label is just 'i'.
    logits_full = torch.cat([logits_batch, logits_hard], dim=1)

    labels = torch.arange(len(emb_anchor), device=emb_anchor.device)

    return F.cross_entropy(logits_full, labels)
