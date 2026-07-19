# Worked Examples

Ten questions, each answerable in a few lines. Open the matching Colab notebook from the
[README](https://github.com/Realosunboy6/free-portfolio-visualizer) to run them interactively.

## 1. Did 60/40 survive the 1970s? *(notebook 08)*
```python
from portlab.data.asset_classes import get_asset_class_returns
from portlab import backtest_portfolio
rets = get_asset_class_returns(["US Stock Market", "10-Year Treasury"], start="1972-01-01")
backtest_portfolio(rets, {"US Stock Market": 60, "10-Year Treasury": 40}, periods=12).summary()
```

## 2. Is my fund's alpha just factor exposure? *(notebook 04)*
```python
from portlab.data.factors import get_ff_factors
from portlab.factor import factor_regression
factor_regression(monthly_fund_returns, get_ff_factors("ff5_mom"), model="ff5_mom")
```

## 3. Can I retire on $800k? *(notebook 03)*
```python
from portlab import monte_carlo
monte_carlo(initial=800_000, years=30, withdrawal="guardrails",
            withdrawal_pct=0.05, model="block_bootstrap",
            hist_returns=my_portfolio_monthly).success_rate
```

## 4. What trades rebalance my account today?
```python
from portlab import rebalance_trades
rebalance_trades({"VTI": 71_000, "BND": 22_000}, {"VTI": 0.6, "BND": 0.4}, cash_to_add=1_000)
```

## 5. What would 2x leverage have done? *(notebook 01)*
```python
backtest_portfolio(rets, {"VTI": 1.0}, leverage=2.0, debt_rate=0.05,
                   maintenance_margin=0.25).summary()
```

## 6. How different is my portfolio after taxes? *(notebook 01)*
```python
backtest_portfolio(rets, alloc, tax_dividend=0.15, tax_capgains=0.15,
                   div_yield={"VTI": 0.015, "BND": 0.03}).summary()
```

## 7. Which famous lazy portfolio wins? *(notebook 01)*
```python
from portlab.data import LAZY_PORTFOLIOS
from portlab.backtest import compare_portfolios
table, _ = compare_portfolios(rets, LAZY_PORTFOLIOS)
```

## 8. Is HRP more robust than max-Sharpe here? *(notebook 02)*
```python
from portlab.optimize import hrp, max_sharpe
hrp(rets), max_sharpe(mu, cov)
```

## 9. What drives my portfolio risk? *(notebook 02)*
```python
from portlab import plots
plots.risk_budget_chart(weights, cov)
```

## 10. Send my results to someone
```python
result.tear_sheet("report.html")     # one file, open anywhere
```
