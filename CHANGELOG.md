# Changelog

All notable changes to portlab. Format: [Keep a Changelog](https://keepachangelog.com); versioning: [SemVer](https://semver.org).

## [0.2.0] - 2026-07-19

### Added
- Parametric + Cornish-Fisher VaR, parametric CVaR, CDaR metric
- ~18 new metrics: Treynor, M², tail ratio, win rate/payoff/profit factor, gain-to-pain,
  Sterling/Burke/pain ratios, recovery factor, R², Kelly fraction, best/worst table, rolling Sharpe/vol/beta
- Hierarchical Risk Parity (`optimize.hrp`), semicovariance estimator, min-CDaR LP, L2 regularization (`gamma`)
- One-line HTML tear sheet (`portlab.report.tear_sheet`, `BacktestResult.tear_sheet`)
- Rebalancing trade calculator (`rebalance_trades`), dividend income history, FX conversion helper
- Frontier transition map, monthly-returns heatmap, rolling-beta chart
- Multi-period returns (`periods_lag`), per-period geometric mean, arithmetic wealth/drawdown variants
- Docs site (mkdocs-material) with formula reference, examples gallery, glossary; PyPI release workflow
- Streamlit web app (`app.py`)

### Earlier (0.1.0, unreleased on PyPI)
- Full tool coverage: backtesting (leverage, taxes, cashflows, band rebalancing),
  14 optimizers, Monte Carlo with 7 withdrawal rules and glide paths, factor analysis, tactical models,
  1926+ asset-class history, 17 lazy portfolios, free data layer with caching and GitHub-mirror fallbacks.
