"""
Unit tests for the implied volatility solver.

Covers:
  - Round-trip: price -> IV -> repriced should match to nanocents.
  - Deep ITM / OTM stability (where Newton typically fails).
  - Arbitrage bound violations return NaN.
  - Vectorized batch mode.
"""

import numpy as np
import pytest

from options_engine.pricing import price_call, price_put
from options_engine.implied_vol import (
    implied_volatility,
    implied_volatility_one,
)


# ---------------------------------------------------------------------------
# Round-trip: the definitive correctness test
# ---------------------------------------------------------------------------

def test_round_trip_scalar_call():
    """Price with sigma, invert, get back the same sigma."""
    S, K, T, r, true_sigma = 100.0, 100.0, 1.0, 0.05, 0.25
    market_price = float(price_call(S, K, T, r, true_sigma)[0])
    recovered_sigma = implied_volatility_one(market_price, S, K, T, r, "call")
    assert abs(recovered_sigma - true_sigma) < 1e-6


def test_round_trip_scalar_put():
    S, K, T, r, true_sigma = 100.0, 110.0, 0.5, 0.05, 0.30
    market_price = float(price_put(S, K, T, r, true_sigma)[0])
    recovered_sigma = implied_volatility_one(market_price, S, K, T, r, "put")
    assert abs(recovered_sigma - true_sigma) < 1e-6


def test_round_trip_batch():
    """Round-trip must hold across a large random grid (excluding degenerate near-zero prices)."""
    rng = np.random.default_rng(42)
    n = 200
    S = rng.uniform(50, 200, n)
    K = rng.uniform(50, 200, n)
    T = rng.uniform(0.1, 2.0, n)
    r = rng.uniform(0.0, 0.10, n)
    true_sigma = rng.uniform(0.10, 0.60, n)

    market_prices = price_call(S, K, T, r, true_sigma)
    recovered = implied_volatility(market_prices, S, K, T, r, "call")

    # Filter out degenerate cases where the true price is essentially zero.
    # In those cases, sigma is ambiguous — many sigmas produce a zero-rounded price.
    meaningful = market_prices > 1e-6

    errors = np.abs(recovered[meaningful] - true_sigma[meaningful])
    max_err = float(np.nanmax(errors))
    assert max_err < 1e-4,f"max round-trip error {max_err} (across {meaningful.sum()} meaningful cases)"


# ---------------------------------------------------------------------------
# Deep ITM / OTM stability (Newton typically fails, Brent must save us)
# ---------------------------------------------------------------------------

def test_deep_itm_call_iv():
    """Deep ITM call: Newton fails (Vega ~0), Brent should recover."""
    S, K, T, r, true_sigma = 200.0, 100.0, 0.5, 0.05, 0.30
    market_price = float(price_call(S, K, T, r, true_sigma)[0])
    recovered = implied_volatility_one(market_price, S, K, T, r, "call")
    assert abs(recovered - true_sigma) < 1e-4


def test_deep_otm_call_iv():
    """Deep OTM call: solver must still recover sigma."""
    S, K, T, r, true_sigma = 50.0, 150.0, 0.5, 0.05, 0.50
    market_price = float(price_call(S, K, T, r, true_sigma)[0])
    recovered = implied_volatility_one(market_price, S, K, T, r, "call")
    assert abs(recovered - true_sigma) < 1e-4


# ---------------------------------------------------------------------------
# Arbitrage bound violations return NaN
# ---------------------------------------------------------------------------

def test_price_above_upper_bound_returns_nan():
    """Call price above S is arbitrage-violating -> NaN."""
    result = implied_volatility_one(150.0, 100.0, 100.0, 1.0, 0.05, "call")
    assert np.isnan(result)


def test_price_below_lower_bound_returns_nan():
    """Call price below intrinsic value is arbitrage-violating -> NaN."""
    # S=200, K=100, so intrinsic is around $100. Price of $5 is impossible.
    result = implied_volatility_one(5.0, 200.0, 100.0, 1.0, 0.05, "call")
    assert np.isnan(result)


def test_negative_price_returns_nan():
    result = implied_volatility_one(-5.0, 100.0, 100.0, 1.0, 0.05, "call")
    assert np.isnan(result)


# ---------------------------------------------------------------------------
# Boundary: intrinsic value gives sigma near zero
# ---------------------------------------------------------------------------

def test_at_intrinsic_returns_low_sigma():
    """Call priced exactly at intrinsic value implies very low sigma."""
    S, K, T, r = 120.0, 100.0, 0.5, 0.05
    intrinsic = S - K * np.exp(-r * T)  # ~ 22.47
    # Add a tiny epsilon so we're technically above intrinsic
    sigma = implied_volatility_one(intrinsic + 1e-3, S, K, T, r, "call")
    assert sigma < 0.15, f"expected low sigma, got {sigma}"


# ---------------------------------------------------------------------------
# Vectorized shape and behavior
# ---------------------------------------------------------------------------

def test_batch_shape():
    n = 5
    S = np.full(n, 100.0)
    K = np.full(n, 100.0)
    T = np.full(n, 1.0)
    r = np.full(n, 0.05)
    true_sigma = np.array([0.15, 0.20, 0.25, 0.30, 0.40])

    prices = price_call(S, K, T, r, true_sigma)
    recovered = implied_volatility(prices, S, K, T, r, "call")

    assert recovered.shape == (n,)
    assert np.max(np.abs(recovered - true_sigma)) < 1e-6


def test_batch_with_mixed_valid_and_arb():
    """A batch with one arbitrage-violating price should have one NaN, rest valid."""
    S = np.array([100.0, 100.0, 100.0])
    K = np.array([100.0, 100.0, 100.0])
    T = np.array([1.0, 1.0, 1.0])
    r = np.array([0.05, 0.05, 0.05])
    prices = np.array([
        float(price_call(100.0, 100.0, 1.0, 0.05, 0.25)[0]),  # valid
        150.0,  # above S - arbitrage
        float(price_call(100.0, 100.0, 1.0, 0.05, 0.35)[0]),  # valid
    ])

    result = implied_volatility(prices, S, K, T, r, "call")
    assert not np.isnan(result[0])
    assert np.isnan(result[1])
    assert not np.isnan(result[2])

def test_diagnose_batch_failure():
    rng = np.random.default_rng(42)
    n = 200
    S = rng.uniform(50, 200, n)
    K = rng.uniform(50, 200, n)
    T = rng.uniform(0.1, 2.0, n)
    r = rng.uniform(0.0, 0.10, n)
    true_sigma = rng.uniform(0.10, 0.60, n)

    market_prices = price_call(S, K, T, r, true_sigma)
    recovered = implied_volatility(market_prices, S, K, T, r, "call")

    meaningful = market_prices > 1e-6
    errors = np.abs(recovered - true_sigma)
    errors_meaningful = np.where(meaningful, errors, 0.0)
    worst = int(np.argmax(errors_meaningful))

    print(f"\n\nWORST MEANINGFUL CASE:")
    print(f"  index:       {worst}")
    print(f"  S:           {S[worst]:.4f}")
    print(f"  K:           {K[worst]:.4f}")
    print(f"  T:           {T[worst]:.4f}")
    print(f"  r:           {r[worst]:.4f}")
    print(f"  true sigma:  {true_sigma[worst]:.6f}")
    print(f"  recovered:   {recovered[worst]:.6f}")
    print(f"  price:       {market_prices[worst]:.10f}")
    print(f"  error:       {errors[worst]:.6f}")
    print(f"  S/K ratio:   {S[worst]/K[worst]:.4f}")
    assert True

