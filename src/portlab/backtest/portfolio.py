"""Portfolio backtesting engine — the portlab equivalent of Portfolio
Visualizer's "Backtest Portfolio" tool.

Supports fixed target weights, periodic contributions/withdrawals (optionally
inflation-adjusted), periodic or tolerance-band rebalancing, benchmark
comparison, and CPI-deflated (real) results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import DEFAULT_RF, TRADING_DAYS
from .. import metrics as M

REBALANCE_MONTHS = {"never": None, "monthly": 1, "quarterly": 3,
                    "semiannually": 6, "annually": 12}


@dataclass
class BacktestResult:
    balance: pd.Series                 # end-of-period portfolio value
    returns: pd.Series                 # per-period time-weighted returns
    weights: pd.DataFrame              # start-of-period effective weights
    cashflows: pd.Series               # external flows (+contribution/-withdrawal)
    benchmark_returns: pd.Series | None = None
    real_balance: pd.Series | None = None
    periods: int = TRADING_DAYS
    rf: float = DEFAULT_RF
    name: str = "Portfolio"
    rebalance_dates: list = field(default_factory=list)

    def summary(self) -> pd.DataFrame:
        s = M.summary(self.returns, rf=self.rf, periods=self.periods,
                      bench=self.benchmark_returns, name=self.name)
        s["Ending Balance"] = float(self.balance.iloc[-1])
        if self.real_balance is not None:
            s["Ending Balance (real)"] = float(self.real_balance.iloc[-1])
        initial = self.balance.iloc[0] / (1 + self.returns.iloc[0])
        flows = pd.Series(
            [-initial, *(-self.cashflows.iloc[1:])],
            index=self.balance.index)
        flows.iloc[-1] += self.balance.iloc[-1]
        s["Money-Weighted Return (IRR)"] = M.money_weighted_return(flows)
        cols = [s.rename(self.name)]
        if self.benchmark_returns is not None:
            cols.append(M.summary(self.benchmark_returns, rf=self.rf,
                                  periods=self.periods, name="Benchmark"))
        return pd.concat(cols, axis=1)

    def drawdowns(self, top: int = 10) -> pd.DataFrame:
        return M.drawdown_table(self.returns, top)


def _is_period_boundary(prev: pd.Timestamp, cur: pd.Timestamp, months: int) -> bool:
    """True when `cur` starts a new k-month block (calendar-aligned)."""
    if prev.year != cur.year or prev.month != cur.month:
        return ((cur.year * 12 + cur.month) % months) == 0 or months == 1
    return False


def backtest_portfolio(
    returns: pd.DataFrame,
    weights: dict[str, float] | pd.Series,
    initial: float = 10_000.0,
    cashflow: float = 0.0,
    cashflow_freq: str = "monthly",
    cashflow_inflation_adjusted: bool = False,
    rebalance: str = "annually",
    rebalance_band: float | None = None,
    benchmark: pd.Series | None = None,
    cpi: pd.Series | None = None,
    rf: float = DEFAULT_RF,
    periods: int = TRADING_DAYS,
    name: str = "Portfolio",
) -> BacktestResult:
    """Simulate a target-weight portfolio through historical returns.

    returns: per-period simple returns per asset (daily or monthly).
    weights: target allocation, will be normalized to sum to 1.
    cashflow: amount added (+) or withdrawn (-) at each cashflow boundary.
    cashflow_freq: 'monthly' | 'quarterly' | 'annually'.
    cashflow_inflation_adjusted: grow the cashflow with CPI (requires cpi).
    rebalance: 'never' | 'monthly' | 'quarterly' | 'semiannually' | 'annually'
        | 'bands' (use rebalance_band as absolute weight deviation trigger).
    cpi: monthly CPI index series; enables real (inflation-adjusted) balances.
    """
    w_target = pd.Series(weights, dtype=float)
    w_target = w_target / w_target.sum()
    rets = returns[list(w_target.index)].dropna(how="any")
    if rets.empty:
        raise ValueError("no overlapping return history for the requested assets")

    cf_months = {"monthly": 1, "quarterly": 3, "annually": 12}[cashflow_freq]
    if rebalance != "bands":
        if rebalance not in REBALANCE_MONTHS:
            raise ValueError(f"rebalance must be one of {[*REBALANCE_MONTHS, 'bands']}")
        reb_months = REBALANCE_MONTHS[rebalance]
    else:
        reb_months = None
        if rebalance_band is None:
            raise ValueError("rebalance='bands' requires rebalance_band")

    # Inflation factor for growing cashflows.
    infl_factor = None
    if cashflow_inflation_adjusted and cpi is not None:
        cpi_m = cpi.reindex(rets.index, method="ffill")
        infl_factor = (cpi_m / cpi_m.iloc[0]).fillna(1.0)

    holdings = initial * w_target.values          # dollar value per asset
    bal, twr, wts, flows, reb_dates = [], [], [], [], []
    prev_date = rets.index[0]

    for i, (date, r) in enumerate(rets.iterrows()):
        # --- external cashflow at period start (skip the very first period)
        flow = 0.0
        if i > 0 and cashflow != 0.0 and _is_period_boundary(prev_date, date, cf_months):
            flow = cashflow * (float(infl_factor.loc[date]) if infl_factor is not None else 1.0)
            total = holdings.sum()
            if total + flow <= 0:      # ruined by withdrawal
                holdings = np.zeros_like(holdings)
            else:
                holdings = holdings * (1 + flow / total)

        # --- rebalance check at period start
        total = holdings.sum()
        if total > 0 and i > 0:
            cur_w = holdings / total
            do_reb = False
            if reb_months is not None and _is_period_boundary(prev_date, date, reb_months):
                do_reb = True
            elif rebalance == "bands" and np.abs(cur_w - w_target.values).max() > rebalance_band:
                do_reb = True
            if do_reb:
                holdings = total * w_target.values
                reb_dates.append(date)

        start_total = holdings.sum()
        wts.append(holdings / start_total if start_total > 0 else w_target.values)

        # --- apply market returns
        holdings = holdings * (1 + r.values)
        end_total = holdings.sum()
        twr.append(end_total / start_total - 1 if start_total > 0 else 0.0)
        bal.append(end_total)
        flows.append(flow)
        prev_date = date

    idx = rets.index
    balance = pd.Series(bal, index=idx, name=name)
    port_rets = pd.Series(twr, index=idx, name=name)
    weights_df = pd.DataFrame(wts, index=idx, columns=w_target.index)
    cashflows = pd.Series(flows, index=idx)

    bench = None
    if benchmark is not None:
        bench = benchmark.reindex(idx).dropna()

    real_balance = None
    if cpi is not None:
        cpi_m = cpi.reindex(idx, method="ffill")
        real_balance = balance * (cpi_m.iloc[0] / cpi_m)

    return BacktestResult(balance=balance, returns=port_rets, weights=weights_df,
                          cashflows=cashflows, benchmark_returns=bench,
                          real_balance=real_balance, periods=periods, rf=rf,
                          name=name, rebalance_dates=reb_dates)


def compare_portfolios(
    returns: pd.DataFrame,
    allocations: dict[str, dict[str, float]],
    benchmark: pd.Series | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, dict[str, BacktestResult]]:
    """Backtest several allocations (PV lets you compare 3 — we allow any number)."""
    results = {nm: backtest_portfolio(returns, w, benchmark=benchmark, name=nm, **kwargs)
               for nm, w in allocations.items()}
    table = pd.concat([r.summary().iloc[:, 0] for r in results.values()], axis=1)
    return table, results


def backtest_dynamic(
    returns: pd.DataFrame,
    schedule: dict,
    initial: float = 10_000.0,
    rf: float = DEFAULT_RF,
    periods: int = TRADING_DAYS,
    name: str = "Dynamic Portfolio",
) -> BacktestResult:
    """PV-style dynamic-allocation backtest: weights change on given dates.

    schedule: {date_like: {ticker: weight}} — on each date the portfolio is
    rebalanced to the new targets; between dates weights drift with returns.
    The union of all tickers across the schedule defines the asset set.
    """
    sched = {pd.Timestamp(d): pd.Series(w, dtype=float) / sum(w.values())
             for d, w in schedule.items()}
    dates = sorted(sched)
    assets = sorted({a for w in sched.values() for a in w.index})
    rets = returns[assets].dropna(how="any")
    rets = rets.loc[rets.index >= dates[0]]
    if rets.empty:
        raise ValueError("no return history after the first schedule date")

    holdings = None
    bal, twr, wts, reb_dates = [], [], [], []
    next_i = 0
    for date, r in rets.iterrows():
        while next_i < len(dates) and dates[next_i] <= date:
            target = sched[dates[next_i]].reindex(assets).fillna(0.0)
            total = holdings.sum() if holdings is not None else initial
            holdings = total * target.values
            reb_dates.append(date)
            next_i += 1
        start_total = holdings.sum()
        wts.append(holdings / start_total if start_total > 0 else np.zeros(len(assets)))
        holdings = holdings * (1 + r.values)
        end_total = holdings.sum()
        twr.append(end_total / start_total - 1 if start_total > 0 else 0.0)
        bal.append(end_total)

    idx = rets.index
    return BacktestResult(
        balance=pd.Series(bal, index=idx, name=name),
        returns=pd.Series(twr, index=idx, name=name),
        weights=pd.DataFrame(wts, index=idx, columns=assets),
        cashflows=pd.Series(0.0, index=idx), periods=periods, rf=rf,
        name=name, rebalance_dates=reb_dates)
