# Formula Reference

Per-period simple returns \(r_t\), \(n\) observations, \(P\) periods per year
(252 daily, 12 monthly), risk-free rate \(r_f\) (annual).

## Returns

| Metric | Formula |
|---|---|
| Simple return | \(r_t = P_t/P_{t-1} - 1\) |
| Log return | \(\ln(P_t/P_{t-1})\) |
| CAGR | \(\left(\prod_t (1+r_t)\right)^{P/n} - 1\) |
| Geometric mean (per period) | \(\left(\prod_t (1+r_t)\right)^{1/n} - 1\) |
| Annualized volatility | \(\sigma\sqrt{P}\) |

## Risk-adjusted ratios

| Metric | Formula |
|---|---|
| Sharpe | \(\dfrac{\bar{r} - r_f/P}{\sigma}\sqrt{P}\) |
| Sortino | \(\dfrac{\bar{r} - r_f/P}{\sigma_{down}}\sqrt{P}\), \(\sigma_{down} = \sqrt{\tfrac{1}{n}\sum \min(r_t - r_f/P, 0)^2}\) |
| Calmar | CAGR / \(|\text{MaxDD}|\) |
| Omega(\(\theta\)) | \(\sum \max(r_t-\theta,0) \big/ \sum \max(\theta-r_t,0)\) |
| Treynor | (CAGR − \(r_f\)) / \(\beta\) |
| M² | \(r_f + \text{Sharpe} \times \sigma_{bench}\sqrt{P}\) |
| Information ratio | \(\bar{a}/\sigma_a \cdot \sqrt{P}\) with active return \(a_t = r_t - b_t\) |

## Tail risk

| Metric | Formula |
|---|---|
| Historical VaR\(_\alpha\) | \(-q_{1-\alpha}(r)\) |
| Parametric VaR | \(\sigma z_\alpha - \mu\) |
| Cornish-Fisher VaR | z adjusted: \(z + \tfrac{(z^2-1)S}{6} + \tfrac{(z^3-3z)K}{24} - \tfrac{(2z^3-5z)S^2}{36}\) |
| Historical CVaR | \(-\mathbb{E}[r \mid r \le q_{1-\alpha}]\) |
| Parametric CVaR | \(\sigma\,\phi(z_\alpha)/(1-\alpha) - \mu\) |
| CDaR\(_\alpha\) | mean of the worst \((1-\alpha)\) drawdowns |

## Drawdowns

Wealth \(W_t = \prod_{s\le t}(1+r_s)\); drawdown \(D_t = W_t/\max_{s\le t} W_s - 1\).
Max drawdown \(= \min_t D_t\); Ulcer index \(= \sqrt{\overline{D_t^2}}\);
pain index \(= \overline{|D_t|}\); recovery factor = total return / |MaxDD|.

## Portfolio math

Expected return \(\mu_p = w^\top\mu\); variance \(\sigma_p^2 = w^\top\Sigma w\);
risk contribution of asset i \(= w_i(\Sigma w)_i / \sigma_p^2\) (sums to 1);
Kelly fraction \(= \mu/\sigma^2\) per period.

## Withdrawal rules

- **VPW**: withdraw \(\text{PMT}(r, N, 1) = \dfrac{r}{(1-(1+r)^{-N})(1+r)}\) of the current
  balance with \(N\) = years to the depletion age and \(r\) the expected real return.
- **Guyton-Klinger**: inflation raise capped at 6%; skip the raise after a losing year when the
  current rate exceeds the initial; cut 10% when rate > 1.2× initial (if >15y remain);
  raise 10% when rate < 0.8× initial.
