"""Bootstrap and multiplicity helpers (numpy/pandas only)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_sharpe(
    daily_returns: pd.Series,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
    periods_per_year: float = 252.0,
) -> dict[str, float]:
    """Percentile CI for Sharpe via simple bootstrap of daily returns."""
    r = daily_returns.dropna().astype(float).values
    if len(r) < 10:
        return {"sharpe_mean": float("nan"), "sharpe_p5": float("nan"), "sharpe_p95": float("nan")}
    rng = np.random.default_rng(seed)
    sharpes = []
    for _ in range(n_bootstrap):
        sample = rng.choice(r, size=len(r), replace=True)
        mu = sample.mean()
        sd = sample.std()
        if sd > 0:
            sharpes.append((mu / sd) * np.sqrt(periods_per_year))
    if not sharpes:
        return {"sharpe_mean": float("nan"), "sharpe_p5": float("nan"), "sharpe_p95": float("nan")}
    a = np.array(sharpes)
    return {
        "sharpe_mean": float(a.mean()),
        "sharpe_p5": float(np.percentile(a, 5)),
        "sharpe_p95": float(np.percentile(a, 95)),
    }


def bonferroni_alpha(alpha: float, n_tests: int) -> float:
    """Adjusted per-test alpha if treating tests as independent (conservative)."""
    if n_tests <= 0:
        return alpha
    return alpha / n_tests


def permutation_sharpe_pvalue(
    daily_returns: pd.Series,
    *,
    observed_sharpe: float,
    n_perm: int = 1000,
    seed: int = 0,
    periods_per_year: float = 252.0,
) -> float:
    """Approximate one-sided p-value: share of permuted Sharpes >= observed (label shuffle)."""
    r = daily_returns.dropna().astype(float).values
    if len(r) < 10:
        return float("nan")
    rng = np.random.default_rng(seed)

    def sharpe(x: np.ndarray) -> float:
        mu = x.mean()
        sd = x.std()
        if sd <= 0:
            return float("nan")
        return (mu / sd) * np.sqrt(periods_per_year)

    cnt = 0
    for _ in range(n_perm):
        perm = rng.permutation(r)
        s = sharpe(perm)
        if not np.isnan(s) and s >= observed_sharpe:
            cnt += 1
    return (1 + cnt) / (1 + n_perm)
