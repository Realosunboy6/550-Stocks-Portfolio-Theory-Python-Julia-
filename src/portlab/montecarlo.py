"""Monte Carlo simulation of portfolio growth and retirement withdrawals.

Return models (matching Portfolio Visualizer's four, plus block bootstrap):
  - 'bootstrap'        IID resampling of historical portfolio returns
  - 'block_bootstrap'  circular block resampling (preserves autocorrelation)
  - 'normal'           parameterized normal from given mean/vol
  - 'student_t'        parameterized fat-tailed Student-t
  - 'statistical'      multivariate normal from asset mu/cov + weights

Withdrawal rules (PV's five):
  - none, fixed real amount, fixed percentage, smoothed percentage
    (rolling average balance), RMD-style 1/remaining-life-expectancy,
    custom per-period schedule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import MONTHS_PER_YEAR


@dataclass
class MonteCarloResult:
    balances: np.ndarray          # (n_sims, T+1) nominal balances
    withdrawals: np.ndarray       # (n_sims, T) per-period withdrawals taken
    freq: int = MONTHS_PER_YEAR

    @property
    def success_rate(self) -> float:
        """Fraction of paths that never hit zero."""
        return float((self.balances.min(axis=1) > 0).mean())

    def percentiles(self, qs=(10, 25, 50, 75, 90)) -> pd.DataFrame:
        years = np.arange(self.balances.shape[1]) / self.freq
        data = {f"p{q}": np.percentile(self.balances, q, axis=0) for q in qs}
        return pd.DataFrame(data, index=pd.Index(years, name="year"))

    def ending_stats(self) -> pd.Series:
        end = self.balances[:, -1]
        return pd.Series({
            "Success Rate": self.success_rate,
            "Median Ending Balance": float(np.median(end)),
            "Mean Ending Balance": float(end.mean()),
            "10th Percentile": float(np.percentile(end, 10)),
            "90th Percentile": float(np.percentile(end, 90)),
            "Prob. Ending < Initial": float((end < self.balances[0, 0]).mean()),
        })


def _simulate_returns(
    model: str, n_sims: int, T: int, rng: np.random.Generator,
    hist_returns: pd.Series | None = None,
    mean_annual: float = 0.07, vol_annual: float = 0.15, t_dof: float = 5.0,
    asset_mu: pd.Series | None = None, asset_cov: pd.DataFrame | None = None,
    weights: pd.Series | None = None,
    freq: int = MONTHS_PER_YEAR, block: int = 6,
) -> np.ndarray:
    """(n_sims, T) matrix of per-period simple portfolio returns."""
    if model in ("bootstrap", "block_bootstrap"):
        if hist_returns is None or len(hist_returns) < 12:
            raise ValueError("bootstrap models need a historical return series")
        h = hist_returns.values
        if model == "bootstrap":
            idx = rng.integers(0, len(h), size=(n_sims, T))
            return h[idx]
        n_blocks = int(np.ceil(T / block))
        starts = rng.integers(0, len(h), size=(n_sims, n_blocks))
        offsets = np.arange(block)
        idx = (starts[:, :, None] + offsets[None, None, :]) % len(h)
        return h[idx].reshape(n_sims, -1)[:, :T]

    mu_p = mean_annual / freq
    sd_p = vol_annual / np.sqrt(freq)
    if model == "normal":
        return rng.normal(mu_p, sd_p, size=(n_sims, T))
    if model == "student_t":
        raw = rng.standard_t(t_dof, size=(n_sims, T))
        scale = sd_p / np.sqrt(t_dof / (t_dof - 2))
        return mu_p + raw * scale
    if model == "statistical":
        if asset_mu is None or asset_cov is None or weights is None:
            raise ValueError("statistical model needs asset_mu, asset_cov, weights")
        w = weights.reindex(asset_mu.index).fillna(0.0).values
        w = w / w.sum()
        mu_port = float(w @ asset_mu.values) / freq
        var_port = float(w @ asset_cov.values @ w) / freq
        return rng.normal(mu_port, np.sqrt(var_port), size=(n_sims, T))
    raise ValueError(f"unknown model {model!r}")


def monte_carlo(
    initial: float = 1_000_000.0,
    years: int = 30,
    n_sims: int = 5_000,
    model: str = "bootstrap",
    hist_returns: pd.Series | None = None,
    mean_annual: float = 0.07,
    vol_annual: float = 0.15,
    t_dof: float = 5.0,
    asset_mu: pd.Series | None = None,
    asset_cov: pd.DataFrame | None = None,
    weights: pd.Series | None = None,
    freq: int = MONTHS_PER_YEAR,
    withdrawal: str = "none",
    withdrawal_amount: float = 0.0,       # annual, today's dollars (fixed rule)
    withdrawal_pct: float = 0.04,          # annual (percentage rules)
    smoothing_periods: int = 12,           # smoothed-percentage rule
    life_expectancy: float = 90.0,         # RMD rule: age-based horizon
    current_age: float = 60.0,
    custom_schedule: pd.Series | None = None,  # per-period amounts (+contribution/-withdrawal)
    contribution_annual: float = 0.0,      # annual contribution (accumulation phase)
    inflation_annual: float = 0.025,
    seed: int = 42,
) -> MonteCarloResult:
    """Simulate portfolio survival under a withdrawal/contribution policy."""
    rng = np.random.default_rng(seed)
    T = years * freq
    R = _simulate_returns(model, n_sims, T, rng, hist_returns, mean_annual,
                          vol_annual, t_dof, asset_mu, asset_cov, weights, freq)

    infl_p = (1 + inflation_annual) ** (1 / freq) - 1
    balances = np.empty((n_sims, T + 1))
    balances[:, 0] = initial
    withdrawals = np.zeros((n_sims, T))
    rolling = np.full((n_sims, max(smoothing_periods, 1)), initial)

    for t in range(T):
        bal = balances[:, t]
        cpi = (1 + infl_p) ** t

        if withdrawal == "none":
            wd = np.zeros(n_sims)
        elif withdrawal == "fixed":
            wd = np.full(n_sims, withdrawal_amount / freq * cpi)
        elif withdrawal == "fixed_pct":
            wd = bal * (withdrawal_pct / freq)
        elif withdrawal == "smoothed_pct":
            wd = rolling.mean(axis=1) * (withdrawal_pct / freq)
        elif withdrawal == "rmd":
            age = current_age + t / freq
            remaining = max(life_expectancy - age, 1.0)
            wd = bal / remaining / freq
        elif withdrawal == "custom":
            if custom_schedule is None:
                raise ValueError("withdrawal='custom' needs custom_schedule")
            amt = float(custom_schedule.iloc[t]) if t < len(custom_schedule) else 0.0
            wd = np.full(n_sims, max(-amt, 0.0) * cpi)
        else:
            raise ValueError(f"unknown withdrawal rule {withdrawal!r}")

        contrib = contribution_annual / freq * cpi
        if withdrawal == "custom" and custom_schedule is not None \
                and t < len(custom_schedule) and float(custom_schedule.iloc[t]) > 0:
            contrib += float(custom_schedule.iloc[t]) * cpi

        wd = np.minimum(wd, np.maximum(bal + contrib, 0.0))
        new_bal = (bal + contrib - wd) * (1 + R[:, t])
        new_bal = np.maximum(new_bal, 0.0)
        balances[:, t + 1] = new_bal
        withdrawals[:, t] = wd
        rolling = np.roll(rolling, 1, axis=1)
        rolling[:, 0] = new_bal

    return MonteCarloResult(balances=balances, withdrawals=withdrawals, freq=freq)
