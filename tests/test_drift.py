import numpy as np
import pandas as pd
import pytest

from drift import drift_report, population_stability_index, summarize_shift

RNG = np.random.default_rng(7)


def test_identical_samples_have_zero_drift():
    ref = pd.Series(RNG.normal(0, 1, 2000))
    assert population_stability_index(ref, ref.copy()) == pytest.approx(0.0, abs=1e-9)


def test_shifted_mean_raises_psi():
    ref = pd.Series(RNG.normal(0, 1, 5000))
    same = pd.Series(RNG.normal(0, 1, 5000))
    shifted = pd.Series(RNG.normal(2.0, 1, 5000))
    assert population_stability_index(ref, shifted) > population_stability_index(ref, same)


def test_large_shift_is_flagged_large():
    ref = pd.Series(RNG.normal(0, 1, 5000))
    shifted = pd.Series(RNG.normal(3.0, 1, 5000))
    assert population_stability_index(ref, shifted) > 0.25


def test_summarize_shift_values():
    ref = pd.Series([1.0, 2.0, 3.0])
    cur = pd.Series([2.0, 3.0, 4.0])
    shift = summarize_shift(ref, cur)
    assert shift["mean_shift"] == pytest.approx(1.0)
    assert shift["std_ratio"] == pytest.approx(1.0)


def test_drift_report_only_shared_numeric_columns():
    ref = pd.DataFrame({"a": RNG.normal(size=500), "b": RNG.normal(size=500), "label": ["x"] * 500})
    cur = pd.DataFrame({"a": RNG.normal(1.5, 1, 500), "b": RNG.normal(size=500)})
    report = drift_report(ref, cur)
    assert list(report.index) == ["a", "b"]
    assert report.loc["a", "psi"] > report.loc["b", "psi"]


def test_empty_after_dropna_raises():
    with pytest.raises(ValueError):
        population_stability_index(pd.Series([np.nan]), pd.Series([1.0]))
