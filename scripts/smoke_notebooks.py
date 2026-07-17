#!/usr/bin/env python3
"""Headless smoke test for the Colab notebooks.

Executes every code cell of each notebook in-process with SMOKE=True and the
network data layer replaced by deterministic synthetic data, so it runs in CI
or sandboxes with no market-data access. Real end-to-end behavior is exercised
in Colab; this catches API breakage between portlab and the notebooks.

Usage: python scripts/smoke_notebooks.py [notebook.ipynb ...]
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("PORTLAB_CACHE", tempfile.mkdtemp(prefix="portlab_smoke_"))

import numpy as np
import pandas as pd
import plotly.graph_objects as go

go.Figure.show = lambda self, *a, **k: None   # headless: never render

import portlab.data.factors as _factors
import portlab.data.macro as _macro
import portlab.data.prices as _prices


def _synthetic_prices(tickers, start, end=None, interval="1d", **kw):
    idx = pd.bdate_range(start, end or pd.Timestamp.today().normalize())
    out = {}
    for t in tickers:
        rng = np.random.default_rng(abs(hash(t)) % (2**32))
        rets = rng.normal(0.0004, 0.012, len(idx))
        out[t] = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(out, index=idx)


def _synthetic_ff(model="ff3", freq="monthly", start="1990-01-01", **kw):
    idx = pd.date_range(start, periods=360, freq="ME")
    rng = np.random.default_rng(0)
    cols = {"Mkt-RF": (0.006, 0.045), "SMB": (0.001, 0.025), "HML": (0.001, 0.028),
            "RMW": (0.002, 0.02), "CMA": (0.002, 0.018), "Mom": (0.004, 0.04)}
    df = pd.DataFrame({c: rng.normal(m, s, len(idx)) for c, (m, s) in cols.items()},
                      index=idx)
    df["RF"] = 0.002
    return df


def _synthetic_fred(series, start="1950-01-01", **kw):
    idx = pd.date_range(start, pd.Timestamp.today(), freq="MS")
    if series.startswith("CPI"):
        return pd.Series(100 * 1.0025 ** np.arange(len(idx)), index=idx, name=series)
    return pd.Series(3.0, index=idx, name=series)


_prices.download_prices = _synthetic_prices
_factors.get_ff_factors = _synthetic_ff
_macro.get_fred = _synthetic_fred


def run_notebook(path: Path) -> None:
    nb = json.loads(path.read_text())
    ns: dict = {"display": lambda *a, **k: None, "get_ipython": lambda: None}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if "%pip" in src or "pip install" in src:
            continue
        src = src.replace("SMOKE = False", "SMOKE = True")
        # strip Colab magics/shell lines if any slipped in
        src = "\n".join(l for l in src.splitlines()
                        if not l.lstrip().startswith(("!", "%")))
        try:
            exec(compile(src, f"{path.name}:cell{i}", "exec"), ns)
        except Exception:
            print(f"FAILED {path.name} cell {i}:\n{src[:400]}", file=sys.stderr)
            raise


def main() -> int:
    root = Path(__file__).resolve().parent.parent / "notebooks"
    targets = [Path(a) for a in sys.argv[1:]] or sorted(root.glob("*.ipynb"))
    for p in targets:
        print(f"-- {p.name}")
        run_notebook(p)
        print(f"   OK")
    print(f"{len(targets)} notebooks passed smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
