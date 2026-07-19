# API Overview

| Module | Key functions |
|---|---|
| `portlab.data` | `get_prices`, `get_returns`, `get_asset_class_returns` (1926+), `get_mirror_returns`, `LAZY_PORTFOLIOS`, `get_ff_factors`, `get_cpi`, `get_shiller_cape` |
| `portlab.metrics` | `summary`, `cagr`, `sharpe`, `sortino`, `omega`, `calmar`, `treynor`, `m_squared`, `var_historical/parametric`, `cvar_historical/parametric`, `cdar`, `drawdown_table`, `best_worst`, `rolling_sharpe/vol/beta`, trade stats |
| `portlab.covariance` | `get_cov(method="ledoit_wolf" | "sample" | "ewma" | "semi")`, `psd_fix` |
| `portlab.optimize` | `gmv`, `max_sharpe` (with `gamma` L2), `frontier`, `equal_risk_contribution`, `hrp`, `min_cvar`, `min_cdar`, `max_sortino`, `kelly`, `max_omega`, `min_max_drawdown`, `min_tracking_error`, `black_litterman`, `resampled_weights`, `geometric_frontier` |
| `portlab.backtest` | `backtest_portfolio` (cashflows, rebalancing, leverage, taxes), `backtest_dynamic`, `glide_path`, `RollingBacktest`, stress tools |
| `portlab.montecarlo` | `monte_carlo` (5 return models, 7 withdrawal rules, glide paths), `glide_weights` |
| `portlab.factor` | `factor_regression`, `rolling_exposures`, `attribution`, `match_exposure`, `compare_funds` |
| `portlab.analytics` | `performance_table`, `screener`, `pca`, `correlation_matrix`, `cointegration_test`, `income_history` |
| `portlab.tactical` | `ma_timing`, `dual_momentum`, `relative_strength`, `target_volatility`, `adaptive_allocation`, `seasonal`, `cape_valuation`, `evaluate` |
| `portlab.report` | `tear_sheet` (also `BacktestResult.tear_sheet`) |
| `portlab.rebalance` | `rebalance_trades` |
| `portlab.plots` | growth, drawdown, frontier (+transition), heatmaps, fan chart, risk budget, attribution, monthly heatmap |

Full signatures are in the docstrings — every function has one.
