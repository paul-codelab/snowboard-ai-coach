import numpy as np

from snowcoach.smooth import clean, hampel, smooth


def test_hampel_removes_spike():
    x = np.full(21, 10.0) + np.random.default_rng(0).normal(0, 0.5, 21)
    x[10] = 60
    y = hampel(x)
    assert np.isnan(y[10])
    assert np.isfinite(y[:10]).all()


def test_nan_stays_nan_and_gaps_not_bridged():
    x = np.array([1, 2, np.nan, np.nan, np.nan, 5, 6], dtype=float)
    y = smooth(x, half=1)
    assert np.isnan(y[2:5]).all()
    assert np.isfinite(y[[0, 1, 5, 6]]).all()


def test_clean_keeps_trend():
    t = np.linspace(0, 2, 60)
    x = 20 * np.sin(np.pi * t)
    y = clean(x, fps=30)
    assert np.nanmax(np.abs(y - x)) < 2
