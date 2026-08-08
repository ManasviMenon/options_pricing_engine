# Options Pricing Engine

A numerically stable Black-Scholes options pricing engine in Python. Prices calls/puts, computes Greeks, solves implied volatility from market prices, and prices full option chains under 5 ms. Validated against live Apple options data with a Power BI dashboard.

## Features

- **Pricing** — vectorized Black-Scholes for European calls and puts, with explicit handling for deep ITM/OTM, near-expiry, and zero-vol regimes to avoid catastrophic cancellation.
- **Greeks** — Delta, Gamma, Vega, Theta, Rho, computed analytically and by finite differences (cross-validated to 6+ decimals, with optimal step sizing).
- **Implied volatility** — Newton-Raphson with a Brent's-method fallback and arbitrage-bound checks.
- **Batch pricing** — full chains (200 contracts) priced in under 5 ms via NumPy broadcasting.
- **Market data** — fetches live option chains (yfinance), prices every contract, and benchmarks the solver's IV against the source (~0.87 correlation on liquid contracts after correcting for dividends).

## Install

```bash
uv pip install -e .
```

## Usage

```python
from options_engine import price_call, greeks_analytic, implied_volatility_one

price_call(S=100, K=100, T=1.0, r=0.05, sigma=0.25)
greeks_analytic(100, 100, 1.0, 0.05, 0.25, "call")
implied_volatility_one(market_price=10.45, S=100, K=100, T=1.0, r=0.05)
```

Fetch and price live data:

```bash
python src/options_engine/data.py
```

## Tests

```bash
pytest
```

~63 tests: Hull reference prices, put-call parity to sub-nanocent precision (1,000 cases), IV round-trip, analytic vs. numerical Greek agreement, no-NaN robustness, and a 5 ms chain benchmark.

## Structure

```
src/options_engine/
    pricing.py       Black-Scholes prices + stability regimes
    greeks.py        analytic + finite-difference Greeks
    implied_vol.py   Newton-Raphson + Brent IV solver
    batch.py         chain pricing + benchmarks
    data.py          live market data pipeline
tests/               test suite
notebooks/           numerical stability studies
powerbi/             dashboard
```

## Stack

Python, NumPy, SciPy, pandas, Matplotlib, yfinance, pytest, Power BI.