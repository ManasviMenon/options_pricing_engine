from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

import numpy as np
import pandas as pd

from options_engine.implied_vol import implied_volatility_one
from options_engine.greeks import greeks_analytic


# Default risk-free rate (approximate; could be pulled from a T-bill feed)
_DEFAULT_R = 0.05


def _years_to_expiry(expiry_str: str, now: Optional[datetime] = None) -> float:
    """Convert a 'YYYY-MM-DD' expiry string to years-from-now."""
    if now is None:
        now = datetime.now()
    expiry = datetime.strptime(expiry_str, "%Y-%m-%d")
    delta_days = (expiry - now).days
    return max(delta_days / 365.0, 1e-6)  # floor to avoid T=0


def fetch_option_chain(
    ticker_symbol: str = "AAPL",
    max_expiries: int = 10,
    option_type: Literal["call", "put"] = "call",
    r: float = _DEFAULT_R,
) -> pd.DataFrame:

    import yfinance as yf

    ticker = yf.Ticker(ticker_symbol)

    # Spot price = most recent close
    hist = ticker.history(period="1d")
    if hist.empty:
        raise RuntimeError(f"Could not fetch spot price for {ticker_symbol}")
    spot = float(hist["Close"].iloc[-1])
    dividend_yield = 0.005

    # Available expiries
    all_expiries = ticker.options
    if not all_expiries:
        raise RuntimeError(f"No option expiries available for {ticker_symbol}")
    expiries = all_expiries[:max_expiries]

    rows = []
    now = datetime.now()

    for expiry_str in expiries:
        T = _years_to_expiry(expiry_str, now)

        chain = ticker.option_chain(expiry_str)
        contracts = chain.calls if option_type == "call" else chain.puts

        for _, contract in contracts.iterrows():
            strike = float(contract["strike"])
            bid = float(contract.get("bid", np.nan))
            ask = float(contract.get("ask", np.nan))
            last = float(contract.get("lastPrice", np.nan))
            yahoo_iv = float(contract.get("impliedVolatility", np.nan))

            # Use mid price if bid/ask available, else last price
            # Quality filters: only price liquid, tradeable contracts
            # 1. Need a real two-sided market (both bid and ask positive)
            if not (np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0):
                continue

            # 2. Reject absurdly wide spreads (illiquid contract)
            spread = ask - bid
            mid = (bid + ask) / 2.0
            if spread / mid > 0.5:  # spread more than 50% of mid = junk
                continue

            # 3. Reject near-worthless contracts (price under 10 cents)
            if mid < 0.10:
                continue

            market_price = mid
            # 4. Only keep contracts near the money (0.85 < S/K < 1.15)
            # Deep ITM/OTM options are illiquid and their IVs are unreliable
            moneyness = spot / strike
            if moneyness < 0.85 or moneyness > 1.15:
                continue

            # 5. Skip near-expiry contracts (< 7 days) - IV is unreliable there
            if T < 7 / 365.0:
                continue

            # Run our IV solver
            our_iv = implied_volatility_one(
                market_price, spot, strike, T, r - dividend_yield, option_type
            )

            # Skip if our solver couldn't recover a sensible IV
            if np.isnan(our_iv):
                continue

            # Compute Greeks at our recovered IV
            g = greeks_analytic(spot, strike, T, r- dividend_yield, our_iv, option_type)

            rows.append({
                "expiry": expiry_str,
                "T": T,
                "strike": strike,
                "spot": spot,
                "market_price": market_price,
                "bid": bid,
                "ask": ask,
                "our_iv": our_iv,
                "yahoo_iv": yahoo_iv,
                "delta": float(g.delta[0]),
                "gamma": float(g.gamma[0]),
                "vega": float(g.vega[0]),
                "theta": float(g.theta[0]),
                "rho": float(g.rho[0]),
                "moneyness": spot / strike,
            })

    df = pd.DataFrame(rows)
    return df


def iv_comparison_stats(df: pd.DataFrame) -> dict:
    """
    Compare our implied vol against Yahoo's on the same contracts.
    Returns summary statistics of the difference.
    """
    valid = df.dropna(subset=["our_iv", "yahoo_iv"])
    valid = valid[valid["yahoo_iv"] > 0]  # Yahoo sometimes reports 0

    if len(valid) == 0:
        return {"n": 0}

    diff = valid["our_iv"] - valid["yahoo_iv"]
    return {
        "n": int(len(valid)),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "median_abs_diff": float(np.median(np.abs(diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "correlation": float(np.corrcoef(valid["our_iv"], valid["yahoo_iv"])[0, 1]),
    }


def save_chain_to_csv(
    df: pd.DataFrame,
    path: str = "data/option_chainv2.csv",
) -> None:
    """Save the enriched chain to CSV for Power BI or further analysis."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Saved {len(df)} contracts to {path}")


if __name__ == "__main__":
    # Quick demo when run directly
    print("Fetching AAPL option chain...")
    df = fetch_option_chain("AAPL", max_expiries=10, option_type="call")
    print(f"\nFetched and priced {len(df)} contracts.")
    print("\nFirst few rows:")
    print(df[["expiry", "strike", "market_price", "our_iv", "yahoo_iv",
              "delta", "gamma"]].head(10).to_string())

    print("\n--- IV Solver vs Yahoo comparison ---")
    stats = iv_comparison_stats(df)
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")

    save_chain_to_csv(df)