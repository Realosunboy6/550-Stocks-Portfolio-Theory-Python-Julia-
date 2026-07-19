# Glossary

Plain-English definitions of every term portlab uses.

- **Alpha** — return beyond what factor/benchmark exposure explains. Positive alpha after a
  proper factor regression is rare.
- **Beta** — sensitivity to the benchmark: beta 1.2 means ~1.2% move when the benchmark moves 1%.
- **CAGR** — compound annual growth rate; the smoothed yearly return that turns start value into
  end value.
- **CDaR** — conditional drawdown at risk; the average of the worst drawdowns.
- **Cornish-Fisher / modified VaR** — VaR adjusted for fat tails and skew instead of assuming a
  normal distribution.
- **CVaR / expected shortfall** — the average loss in the worst (1−α) of periods; deeper than VaR.
- **Drawdown** — percent decline from the highest point so far.
- **Efficient frontier** — the set of portfolios with the highest return for each risk level.
- **Factor regression** — explaining returns with systematic drivers (market, size, value,
  momentum, quality...).
- **Glide path** — gradually shifting allocation (usually stocks → bonds) as a goal approaches.
- **Guardrails (Guyton-Klinger)** — retirement spending rules that cut/raise withdrawals when
  the withdrawal rate drifts too far from the initial plan.
- **HRP** — hierarchical risk parity; clusters similar assets and splits risk across clusters,
  avoiding unstable matrix inversion.
- **Kelly fraction** — bet size that maximizes long-run compound growth (aggressive; many use half-Kelly).
- **Ledoit-Wolf shrinkage** — a statistically safer covariance estimate that pulls extreme
  sample values toward a structured target.
- **Leverage** — investing more than your equity by borrowing; amplifies both gains and losses.
- **Maintenance margin** — minimum equity/position ratio before a forced sell-down (margin call).
- **Max drawdown** — the worst peak-to-trough loss over the whole history.
- **Monte Carlo** — simulating thousands of possible futures to estimate outcome probabilities.
- **Rebalancing** — trading back to target weights; "bands" trigger only past a threshold.
- **Risk parity / risk contribution** — sizing positions so each contributes equally to risk.
- **Sharpe ratio** — excess return per unit of volatility; the standard risk-adjusted score.
- **Sortino ratio** — like Sharpe but only penalizes downside volatility.
- **Tracking error** — volatility of the return difference vs a benchmark.
- **VPW** — variable percentage withdrawal; annuity-style spending that adapts to the balance and
  never technically runs out.
- **Walk-forward backtest** — repeatedly train on the past, trade the next block; the honest way
  to evaluate a strategy.
