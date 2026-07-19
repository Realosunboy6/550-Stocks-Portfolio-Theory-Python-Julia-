"""Mean-variance optimization: GMV, max Sharpe, targets, efficient frontier."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DEFAULT_RF
from .core import portfolio_stats, solve_weights


def gmv(mu: pd.Series, cov: pd.DataFrame, bounds=(0.0, 1.0), groups=None,
        gamma: float = 0.0) -> pd.Series:
    """Global minimum-variance portfolio. gamma adds L2 regularization
    (gamma * ||w||^2) to spread weights across more assets."""
    C = cov.values
    return solve_weights(lambda w: w @ C @ w + gamma * (w @ w), len(mu),
                         bounds, groups, index=mu.index)


def max_sharpe(mu: pd.Series, cov: pd.DataFrame, rf: float = DEFAULT_RF,
               bounds=(0.0, 1.0), groups=None, gamma: float = 0.0) -> pd.Series:
    m, C = mu.values, cov.values

    def neg_sharpe(w):
        vol = np.sqrt(w @ C @ w)
        base = -(w @ m - rf) / vol if vol > 0 else 0.0
        return base + gamma * (w @ w)

    return solve_weights(neg_sharpe, len(mu), bounds, groups, index=mu.index)


def min_vol_at_return(mu: pd.Series, cov: pd.DataFrame, target_return: float,
                      bounds=(0.0, 1.0), groups=None) -> pd.Series:
    C = cov.values
    extra = [{"type": "eq", "fun": lambda w: w @ mu.values - target_return}]
    return solve_weights(lambda w: w @ C @ w, len(mu), bounds, groups,
                         extra_constraints=extra, index=mu.index)


def max_return_at_vol(mu: pd.Series, cov: pd.DataFrame, target_vol: float,
                      bounds=(0.0, 1.0), groups=None) -> pd.Series:
    C = cov.values
    extra = [{"type": "ineq",
              "fun": lambda w: target_vol ** 2 - w @ C @ w}]
    return solve_weights(lambda w: -(w @ mu.values), len(mu), bounds, groups,
                         extra_constraints=extra, index=mu.index)


def frontier(mu: pd.Series, cov: pd.DataFrame, n_points: int = 30,
             rf: float = DEFAULT_RF, bounds=(0.0, 1.0), groups=None) -> pd.DataFrame:
    """Efficient frontier: one row per target return with vol/sharpe/weights."""
    w_gmv = gmv(mu, cov, bounds, groups)
    r_lo = float(w_gmv @ mu)
    r_hi = float(mu.max())
    rows = []
    for tr in np.linspace(r_lo, r_hi, n_points):
        try:
            w = min_vol_at_return(mu, cov, tr, bounds, groups)
        except Exception:
            continue
        stats = portfolio_stats(w.values, mu, cov, rf)
        rows.append({"target_return": tr, **stats,
                     **{f"w_{a}": w[a] for a in mu.index}})
    return pd.DataFrame(rows)
