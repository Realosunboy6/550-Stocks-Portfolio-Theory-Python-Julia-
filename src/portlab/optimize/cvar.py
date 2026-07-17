"""CVaR (expected shortfall) portfolio optimization via the
Rockafellar-Uryasev linear program — scipy.linprog, no cvxpy required."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from ..config import DEFAULT_ALPHA


def min_cvar(
    rets: pd.DataFrame,
    alpha: float = DEFAULT_ALPHA,
    min_return: float | None = None,
    bounds=(0.0, 1.0),
) -> pd.Series:
    """Minimize historical CVaR_alpha of per-period portfolio returns.

    min_return: optional lower bound on mean per-period return.
    Decision variables: [w (n), zeta (1), u (T)] minimizing
    zeta + 1/((1-alpha)T) * sum(u),  u_t >= -r_t·w - zeta,  u_t >= 0.
    """
    R = rets.values
    T, n = R.shape
    inv = 1.0 / ((1 - alpha) * T)

    c = np.concatenate([np.zeros(n), [1.0], np.full(T, inv)])

    # -R w - zeta - u <= 0  ->  rows: t
    rows, cols, vals = [], [], []
    for t in range(T):
        rows.extend([t] * n); cols.extend(range(n)); vals.extend(-R[t])
        rows.append(t); cols.append(n); vals.append(-1.0)
        rows.append(t); cols.append(n + 1 + t); vals.append(-1.0)
    A_ub = coo_matrix((vals, (rows, cols)), shape=(T, n + 1 + T)).tocsr()
    b_ub = np.zeros(T)

    if min_return is not None:
        mean_row = np.concatenate([-R.mean(axis=0), [0.0], np.zeros(T)])
        A_ub = coo_matrix(np.vstack([A_ub.toarray(), mean_row]))
        b_ub = np.append(b_ub, -min_return)

    A_eq = np.concatenate([np.ones(n), [0.0], np.zeros(T)]).reshape(1, -1)
    b_eq = [1.0]

    lo, hi = bounds
    var_bounds = [(lo, hi)] * n + [(None, None)] + [(0, None)] * T

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=var_bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"CVaR LP failed: {res.message}")
    w = res.x[:n]
    return pd.Series(w / w.sum(), index=rets.columns, name="weight")


def cvar_frontier(rets: pd.DataFrame, alphas=(0.90, 0.95, 0.975, 0.99),
                  bounds=(0.0, 1.0)) -> pd.DataFrame:
    """Min-CVaR portfolios across confidence levels (original Phase 6 sweep)."""
    from ..metrics import cvar_historical
    rows = []
    for a in alphas:
        w = min_cvar(rets, alpha=a, bounds=bounds)
        pr = rets @ w
        rows.append({"alpha": a, "mean_return": pr.mean(),
                     "cvar": cvar_historical(pr, a),
                     **{f"w_{c}": w[c] for c in rets.columns}})
    return pd.DataFrame(rows)
