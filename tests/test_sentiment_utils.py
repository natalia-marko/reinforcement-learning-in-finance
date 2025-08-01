"""Minimal unit-tests for sentiment_utils

Run with `pytest -q` (after installing pytest).

These tests avoid external API calls and instead focus on pure-python
logic that could affect numeric correctness.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import sentiment_utils as su


def test_row_weighted_bias_free():
    """Ensure sentiment_weighted ignores zero-count sources (fix for bias)."""
    sample = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "month": ["2024-01", "2024-02"],
            "sentiment_mean": [0.2, 0.3],
            # google present only in first row
            "google_sentiment_mean": [0.1, 0.0],
            "google_news_count": [10, 0],
            # yahoo present only in second row
            "yahoo_sentiment_mean": [0.0, -0.2],
            "yahoo_news_count": [0, 5],
        }
    )

    out = su.aggregate_monthly_sentiment_enhanced(sample)
    # Row-wise expected values
    # row0 -> (0.1*10) / 10 = 0.1
    # row1 -> (-0.2*5) / 5 = -0.2
    assert np.isclose(out.loc[0, "sentiment_weighted"], 0.1)
    assert np.isclose(out.loc[1, "sentiment_weighted"], -0.2)


def test_get_cache_path(tmp_path):
    """Cache path should live under news_cache and end with expected pattern."""
    p = su.get_cache_path("yahoo", "TEST", 2024)
    assert p.name == "yahoo_TEST_2024.json"
    # path should reside in the global CACHE_DIR
    assert p.parent == su.CACHE_DIR


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-q"])