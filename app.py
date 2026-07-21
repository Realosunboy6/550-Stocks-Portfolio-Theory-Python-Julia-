"""portlab web app — run locally with `streamlit run app.py`, or deploy free
on Streamlit Community Cloud (share.streamlit.io → this repo → app.py).

Every page is a thin form over the portlab package; no logic lives here.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from portlab import backtest_portfolio, metrics, monte_carlo, optimize as opt, plots
from portlab.covariance import corr_from_cov, get_cov
from portlab.data import LAZY_PORTFOLIOS, get_returns
from portlab.report import tear_sheet

st.set_page_config(page_title="portlab — portfolio analysis", page_icon="📈",
                   layout="wide")


@st.cache_data(ttl=3600, show_spinner="Downloading prices…")
def load_returns(tickers: tuple[str, ...], start: str) -> pd.DataFrame:
    return get_returns(list(tickers), start)


def parse_tickers(raw: str) -> list[str]:
    return [t.strip().upper() for t in raw.split(",") if t.strip()]


st.sidebar.title("📈 portlab")
st.sidebar.caption("Free, open-source portfolio analysis — "
                   "[GitHub](https://github.com/Realosunboy6/free-portfolio-visualizer)")
page = st.sidebar.radio("Tool", ["Backtest", "Optimize", "Retirement", "Analytics"])
start = st.sidebar.text_input("Start date", "2015-01-01")

if page == "Backtest":
    st.header("Backtest Portfolio")
    preset = st.selectbox("Preset", ["Custom"] + list(LAZY_PORTFOLIOS))
    if preset == "Custom":
        c1, c2 = st.columns(2)
        raw_t = c1.text_input("Tickers", "VTI, TLT, GLD")
        raw_w = c2.text_input("Weights", "60, 30, 10")
        ticks = parse_tickers(raw_t)
        alloc = dict(zip(ticks, [float(x) for x in raw_w.split(",")]))
    else:
        alloc = LAZY_PORTFOLIOS[preset]
        ticks = list(alloc)
        st.write(alloc)
    c1, c2, c3, c4 = st.columns(4)
    initial = c1.number_input("Initial $", 1000, 100_000_000, 10_000)
    cashflow = c2.number_input("Monthly cashflow ($, − = withdraw)", value=0)
    rebalance = c3.selectbox("Rebalancing", ["annually", "quarterly", "monthly", "never"])
    bench_t = c4.text_input("Benchmark", "SPY").strip().upper()

    if st.button("Run backtest", type="primary"):
        rets = load_returns(tuple(ticks + [bench_t]), start)
        res = backtest_portfolio(rets, alloc, initial=initial, cashflow=cashflow,
                                 rebalance=rebalance, benchmark=rets[bench_t])
        both = pd.concat([res.returns.rename("Portfolio"),
                          rets[bench_t].rename(bench_t)], axis=1)
        a, b, c, d = st.columns(4)
        a.metric("CAGR", f"{metrics.cagr(res.returns):.2%}")
        b.metric("Sharpe", f"{metrics.sharpe(res.returns):.2f}")
        c.metric("Max Drawdown", f"{metrics.max_drawdown(res.returns):.1%}")
        d.metric("Ending Balance", f"${res.balance.iloc[-1]:,.0f}")
        st.plotly_chart(plots.growth_chart(both, initial=initial), use_container_width=True)
        st.plotly_chart(plots.drawdown_chart(both), use_container_width=True)
        st.plotly_chart(plots.monthly_heatmap(res.returns), use_container_width=True)
        st.dataframe(res.summary())
        st.download_button("⬇️ Download tear sheet (HTML)",
                           tear_sheet(res.returns, rets[bench_t], output=None),
                           file_name="tear_sheet.html", mime="text/html")

elif page == "Optimize":
    st.header("Optimization & Efficient Frontier")
    ticks = parse_tickers(st.text_input("Tickers", "VTI, VEA, VWO, TLT, IEF, GLD, VNQ"))
    rf = st.slider("Risk-free rate", 0.0, 0.08, 0.03, 0.005)
    if st.button("Optimize", type="primary"):
        rets = load_returns(tuple(ticks), start)
        mu, cov = rets.mean() * 252, get_cov(rets)
        ports = {"Equal Weight": pd.Series(1 / len(mu), index=mu.index),
                 "GMV": opt.gmv(mu, cov),
                 "Max Sharpe": opt.max_sharpe(mu, cov, rf=rf),
                 "Risk Parity": opt.equal_risk_contribution(cov),
                 "HRP": opt.hrp(rets),
                 "Min CVaR": opt.min_cvar(rets)}
        fr = opt.frontier(mu, cov, n_points=30, rf=rf)
        stats = {n: opt.portfolio_stats(w.values, mu, cov, rf) for n, w in ports.items()}
        st.plotly_chart(plots.frontier_chart(
            fr, mu, cov, highlight={n: (s["volatility"], s["return"])
                                    for n, s in stats.items()}), use_container_width=True)
        st.plotly_chart(plots.frontier_transition_chart(fr), use_container_width=True)
        st.dataframe(pd.DataFrame(ports).style.format("{:.1%}"))
        st.plotly_chart(plots.risk_budget_chart(ports["Max Sharpe"], cov),
                        use_container_width=True)
        st.plotly_chart(plots.corr_heatmap(corr_from_cov(cov)), use_container_width=True)

elif page == "Retirement":
    st.header("Monte Carlo Retirement Planner")
    c1, c2, c3 = st.columns(3)
    initial = c1.number_input("Starting balance $", 10_000, 100_000_000, 1_000_000)
    years = c2.slider("Years", 5, 60, 30)
    rule = c3.selectbox("Withdrawal rule", ["fixed", "fixed_pct", "smoothed_pct",
                                            "rmd", "vpw", "guardrails", "none"])
    c1, c2, c3 = st.columns(3)
    amount = c1.number_input("Annual withdrawal $ (fixed)", value=40_000)
    pct = c2.slider("Withdrawal % (percentage rules)", 0.01, 0.10, 0.04, 0.005)
    ticks = parse_tickers(c3.text_input("Portfolio", "VTI, BND"))
    if st.button("Simulate", type="primary"):
        rets = load_returns(tuple(ticks), start)
        port = (1 + rets.mean(axis=1)).resample("ME").prod() - 1
        res = monte_carlo(initial=initial, years=years, n_sims=3000,
                          model="block_bootstrap", hist_returns=port,
                          withdrawal=rule, withdrawal_amount=amount,
                          withdrawal_pct=pct)
        st.metric("Success rate", f"{res.success_rate:.1%}")
        st.plotly_chart(plots.fan_chart(res.percentiles((5, 25, 50, 75, 95))),
                        use_container_width=True)
        st.dataframe(res.ending_stats().to_frame("value"))

else:
    st.header("Asset Analytics")
    ticks = parse_tickers(st.text_input("Tickers", "SPY, QQQ, TLT, GLD, BTC-USD"))
    if st.button("Analyze", type="primary"):
        from portlab import analytics
        rets = load_returns(tuple(ticks), start)
        st.plotly_chart(plots.corr_heatmap(analytics.correlation_matrix(rets)),
                        use_container_width=True)
        st.dataframe(analytics.performance_table(rets).style.format("{:.3f}"))
