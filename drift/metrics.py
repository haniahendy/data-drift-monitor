"""Numeric drift metrics between a reference and a current sample.

PSI thresholds used here are the common industry rule of thumb:
< 0.1 no significant shift, 0.1-0.25 moderate, > 0.25 large.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _binned_shares(reference: pd.Series, current: pd.Series, bins: int):
    ref = reference.dropna().to_numpy(dtype=float)
    cur = current.dropna().to_numpy(dtype=float)
    if ref.size == 0 or cur.size == 0:
        raise ValueError("reference and current must both be non-empty after dropping NaN")
    edges = np.histogram_bin_edges(ref, bins=bins)
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    eps = 1e-6
    ref_share = np.maximum(ref_counts / ref_counts.sum(), eps)
    cur_share = np.maximum(cur_counts / cur_counts.sum(), eps)
    return ref_share, cur_share


def population_stability_index(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI between two numeric samples. Higher means more drift."""
    ref_share, cur_share = _binned_shares(reference, current, bins)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def summarize_shift(reference: pd.Series, current: pd.Series) -> dict:
    """Relative change in mean and standard deviation."""
    ref_mean, cur_mean = float(reference.mean()), float(current.mean())
    ref_std, cur_std = float(reference.std()), float(current.std())
    return {
        "reference_mean": ref_mean,
        "current_mean": cur_mean,
        "mean_shift": cur_mean - ref_mean,
        "reference_std": ref_std,
        "current_std": cur_std,
        "std_ratio": (cur_std / ref_std) if ref_std else float("inf"),
    }


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Per-column PSI and shift summary for shared numeric columns."""
    shared = [c for c in reference.columns if c in current.columns and pd.api.types.is_numeric_dtype(reference[c])]
    if not shared:
        raise ValueError("no shared numeric columns")
    rows = []
    for col in shared:
        psi = population_stability_index(reference[col], current[col], bins=bins)
        shift = summarize_shift(reference[col], current[col])
        rows.append({
            "column": col,
            "psi": round(psi, 4),
            "drift_level": "large" if psi > 0.25 else ("moderate" if psi > 0.1 else "none"),
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in shift.items()},
        })
    return pd.DataFrame(rows).set_index("column")
