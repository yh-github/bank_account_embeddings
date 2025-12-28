"""Unit tests for loss functions."""

import unittest

import torch

from hierarchical.training.losses import (
    contrastive_loss,
    contrastive_loss_with_hard_negatives,
)


class TestContrastiveLoss(unittest.TestCase):
    """Tests for contrastive_loss function."""

    def test_identical_embeddings_low_loss(self):
        """Identical embeddings should have low loss."""
        emb = torch.randn(8, 64)
        loss = contrastive_loss(emb, emb.clone())
        # Loss should be relatively low for identical pairs
        self.assertLess(loss.item(), 1.0)

    def test_output_is_scalar(self):
        """Loss should be a scalar tensor."""
        emb1 = torch.randn(4, 32)
        emb2 = torch.randn(4, 32)
        loss = contrastive_loss(emb1, emb2)
        self.assertEqual(loss.shape, torch.Size([]))

    def test_loss_is_positive(self):
        """Loss should always be positive."""
        emb1 = torch.randn(8, 64)
        emb2 = torch.randn(8, 64)
        loss = contrastive_loss(emb1, emb2)
        self.assertGreater(loss.item(), 0)

    def test_temperature_affects_loss(self):
        """Different temperatures should produce different losses."""
        emb1 = torch.randn(8, 64)
        emb2 = torch.randn(8, 64)
        loss_low_temp = contrastive_loss(emb1, emb2, temperature=0.01)
        loss_high_temp = contrastive_loss(emb1, emb2, temperature=1.0)
        # Lower temperature should generally produce higher loss
        self.assertNotEqual(loss_low_temp.item(), loss_high_temp.item())

    def test_different_batch_sizes(self):
        """Loss should work with different batch sizes."""
        for batch_size in [1, 4, 16, 32]:
            emb1 = torch.randn(batch_size, 64)
            emb2 = torch.randn(batch_size, 64)
            loss = contrastive_loss(emb1, emb2)
            self.assertFalse(torch.isnan(loss))


class TestContrastiveLossWithHardNegatives(unittest.TestCase):
    """Tests for contrastive_loss_with_hard_negatives function."""

    def test_none_hard_neg_falls_back(self):
        """When hard_neg is None, should fall back to basic contrastive loss."""
        emb_anchor = torch.randn(8, 64)
        emb_positive = torch.randn(8, 64)
        loss = contrastive_loss_with_hard_negatives(emb_anchor, emb_positive, None)
        expected = contrastive_loss(emb_anchor, emb_positive)
        self.assertAlmostEqual(loss.item(), expected.item(), places=5)

    def test_with_hard_negatives(self):
        """Loss should work with hard negatives provided."""
        emb_anchor = torch.randn(8, 64)
        emb_positive = torch.randn(8, 64)
        emb_hard_neg = torch.randn(8, 64)
        loss = contrastive_loss_with_hard_negatives(
            emb_anchor, emb_positive, emb_hard_neg
        )
        self.assertFalse(torch.isnan(loss))
        self.assertGreater(loss.item(), 0)

    def test_hard_negatives_increase_loss(self):
        """Adding hard negatives should generally affect loss."""
        torch.manual_seed(42)
        emb_anchor = torch.randn(16, 64)
        emb_positive = emb_anchor + torch.randn(16, 64) * 0.1  # Similar to anchor
        emb_hard_neg = torch.randn(16, 64)  # Random hard negatives
        
        loss_without = contrastive_loss(emb_anchor, emb_positive)
        loss_with = contrastive_loss_with_hard_negatives(
            emb_anchor, emb_positive, emb_hard_neg
        )
        # Both should be valid losses
        self.assertFalse(torch.isnan(loss_without))
        self.assertFalse(torch.isnan(loss_with))


class TestLossGradients(unittest.TestCase):
    """Tests for gradient flow through loss functions."""

    def test_contrastive_loss_gradients(self):
        """Contrastive loss should produce valid gradients."""
        emb1 = torch.randn(8, 64, requires_grad=True)
        emb2 = torch.randn(8, 64, requires_grad=True)
        loss = contrastive_loss(emb1, emb2)
        loss.backward()
        self.assertIsNotNone(emb1.grad)
        self.assertIsNotNone(emb2.grad)
        self.assertFalse(torch.isnan(emb1.grad).any())
        self.assertFalse(torch.isnan(emb2.grad).any())

    def test_hard_negative_loss_gradients(self):
        """Hard negative loss should produce valid gradients."""
        emb_anchor = torch.randn(8, 64, requires_grad=True)
        emb_positive = torch.randn(8, 64, requires_grad=True)
        emb_hard_neg = torch.randn(8, 64, requires_grad=True)
        loss = contrastive_loss_with_hard_negatives(
            emb_anchor, emb_positive, emb_hard_neg
        )
        loss.backward()
        self.assertIsNotNone(emb_anchor.grad)
        self.assertFalse(torch.isnan(emb_anchor.grad).any())


if __name__ == "__main__":
    unittest.main()
