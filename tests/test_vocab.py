"""Unit tests for vocabulary and feature extraction."""

import os
import tempfile
import unittest
from datetime import datetime

import numpy as np
import pandas as pd

from hierarchical.data.vocab import (
    CategoricalVocabulary,
    V4FeatureExtractor,
    build_vocabularies,
    load_vocabularies,
)


class TestCategoricalVocabulary(unittest.TestCase):
    """Tests for CategoricalVocabulary class."""

    def test_init_has_pad_and_unk(self):
        """Vocabulary should initialize with PAD (0) and UNK (1)."""
        vocab = CategoricalVocabulary()
        self.assertEqual(vocab.encode("<PAD>"), 0)
        self.assertEqual(vocab.encode("<UNK>"), 1)
        self.assertEqual(len(vocab), 2)

    def test_fit_adds_tokens(self):
        """Fit should add new tokens to vocabulary."""
        vocab = CategoricalVocabulary()
        vocab.fit(["apple", "banana", "cherry"])
        self.assertEqual(len(vocab), 5)  # PAD, UNK + 3 tokens
        # Tokens should be encoded sequentially starting from 2
        self.assertGreaterEqual(vocab.encode("apple"), 2)
        self.assertGreaterEqual(vocab.encode("banana"), 2)
        self.assertGreaterEqual(vocab.encode("cherry"), 2)

    def test_fit_ignores_nulls(self):
        """Fit should ignore None, empty strings, and NaN."""
        vocab = CategoricalVocabulary()
        vocab.fit(["valid", None, "", np.nan])
        self.assertEqual(len(vocab), 3)  # PAD, UNK + 1 valid token

    def test_encode_unknown_returns_unk(self):
        """Encoding unknown token should return UNK (1)."""
        vocab = CategoricalVocabulary()
        vocab.fit(["known"])
        self.assertEqual(vocab.encode("unknown"), 1)

    def test_encode_none_returns_pad(self):
        """Encoding None should return PAD (0)."""
        vocab = CategoricalVocabulary()
        self.assertEqual(vocab.encode(None), 0)
        self.assertEqual(vocab.encode(""), 0)
        self.assertEqual(vocab.encode(np.nan), 0)

    def test_decode_returns_token(self):
        """Decode should return original token string."""
        vocab = CategoricalVocabulary()
        vocab.fit(["test_token"])
        token_id = vocab.encode("test_token")
        self.assertEqual(vocab.decode(token_id), "test_token")

    def test_decode_unknown_id(self):
        """Decode of unknown ID should return UNK."""
        vocab = CategoricalVocabulary()
        self.assertEqual(vocab.decode(999), "<UNK>")

    def test_save_and_load(self):
        """Vocabulary should be savable and loadable."""
        vocab = CategoricalVocabulary()
        vocab.fit(["alpha", "beta", "gamma"])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "vocab.pkl")
            vocab.save(path)
            loaded = CategoricalVocabulary.load(path)

        self.assertEqual(len(loaded), len(vocab))
        self.assertEqual(loaded.encode("alpha"), vocab.encode("alpha"))
        self.assertEqual(loaded.encode("gamma"), vocab.encode("gamma"))

    def test_fit_is_deterministic(self):
        """Fit should produce deterministic vocabulary (sorted)."""
        vocab1 = CategoricalVocabulary()
        vocab2 = CategoricalVocabulary()
        tokens = ["z_last", "a_first", "m_middle"]
        vocab1.fit(tokens)
        vocab2.fit(tokens)
        for token in tokens:
            self.assertEqual(vocab1.encode(token), vocab2.encode(token))


class TestV4FeatureExtractor(unittest.TestCase):
    """Tests for V4FeatureExtractor class."""

    def setUp(self):
        self.cat_group_vocab = CategoricalVocabulary()
        self.cat_sub_vocab = CategoricalVocabulary()
        self.counter_party_vocab = CategoricalVocabulary()
        self.extractor = V4FeatureExtractor(
            self.cat_group_vocab,
            self.cat_sub_vocab,
            self.counter_party_vocab,
            epoch_date=datetime(2020, 1, 1),
        )

    def test_extract_date_features_shape(self):
        """Date features should have shape [N, 4]."""
        dates = pd.Series([datetime(2020, 1, 15), datetime(2020, 6, 20)])
        features = self.extractor.extract_date_features(dates)
        self.assertEqual(features.shape, (2, 4))
        self.assertEqual(features.dtype, np.float32)

    def test_extract_date_features_dow(self):
        """Day of week should be in range 0-6."""
        # Jan 6, 2020 is a Monday (dow=0)
        dates = pd.Series([datetime(2020, 1, 6)])
        features = self.extractor.extract_date_features(dates)
        self.assertEqual(features[0, 0], 0)  # Monday

    def test_extract_date_features_dom(self):
        """Day of month should be 0-indexed (0-30)."""
        dates = pd.Series([datetime(2020, 1, 15)])
        features = self.extractor.extract_date_features(dates)
        self.assertEqual(features[0, 1], 14)  # 15th day, 0-indexed

    def test_normalize_amounts_sign_preservation(self):
        """Amount normalization should preserve relative sign."""
        # Use asymmetric values to avoid zero after standardization
        amounts = np.array([1000.0, 100.0, -100.0, -1000.0])
        normalized = V4FeatureExtractor.normalize_amounts(amounts)
        # Larger positive should be > smaller positive
        self.assertGreater(normalized[0], normalized[1])
        # Larger negative (more negative) should be < smaller negative
        self.assertLess(normalized[3], normalized[2])

    def test_normalize_amounts_handles_zeros(self):
        """Amount normalization should handle zeros."""
        amounts = np.array([0.0, 100.0])
        normalized = V4FeatureExtractor.normalize_amounts(amounts)
        # Should not raise, zeros handled via log1p
        self.assertIsInstance(normalized, np.ndarray)
        self.assertEqual(normalized.dtype, np.float32)


class TestBuildVocabularies(unittest.TestCase):
    """Tests for vocabulary building functions."""

    def test_build_vocabularies_returns_three(self):
        """build_vocabularies should return 3 vocabularies."""
        df = pd.DataFrame(
            {
                "personeticsCategoryGroupId": ["CG1", "CG2", "CG1"],
                "personeticsSubCategoryId": ["SC1", "SC2", "SC3"],
                "deviceId": ["D1", "D2", "D1"],
            }
        )
        cat_grp, cat_sub, counter_party = build_vocabularies(df)
        self.assertIsInstance(cat_grp, CategoricalVocabulary)
        self.assertIsInstance(cat_sub, CategoricalVocabulary)
        self.assertIsInstance(counter_party, CategoricalVocabulary)

    def test_build_and_load_vocabularies(self):
        """Vocabularies should round-trip through save/load."""
        df = pd.DataFrame(
            {
                "personeticsCategoryGroupId": ["CG100", "CG200"],
                "personeticsSubCategoryId": ["SC100", "SC200"],
                "deviceId": ["DEV1", "DEV2"],
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cat_grp, cat_sub, counter_party = build_vocabularies(df, save_dir=tmpdir)
            loaded_grp, loaded_sub, loaded_cp = load_vocabularies(tmpdir)

        self.assertEqual(len(loaded_grp), len(cat_grp))
        self.assertEqual(loaded_grp.encode("CG100"), cat_grp.encode("CG100"))


if __name__ == "__main__":
    unittest.main()
