import numpy as np
import pytest

from options_engine.pricing import price_call, price_put
from options_engine.batch import (
    ChainResult,
    price_chain,
    benchmark_chain,
)

def test_chain_matches_individual_pricer():
    """Every price in the chain must match a direct pricer call."""
    spot = 100.0
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    expiries = np.array([0.25, 0.5, 1.0, 2.0])
    r = 0.05
    sigma = 0.25

    result = price_chain(spot, strikes, expiries, r, sigma, "call")

    for i, K in enumerate(strikes):
        for j, T in enumerate(expiries):
            expected = float(price_call(spot, K, T, r, sigma)[0])
            assert abs(float(result.prices[i, j]) - expected) < 1e-12


def test_chain_put_matches_individual_pricer():
    spot = 100.0
    strikes = np.array([80.0, 100.0, 120.0])
    expiries = np.array([0.5, 1.0])
    r = 0.05
    sigma = 0.30

    result = price_chain(spot, strikes, expiries, r, sigma, "put")

    for i, K in enumerate(strikes):
        for j, T in enumerate(expiries):
            expected = float(price_put(spot, K, T, r, sigma)[0])
            assert abs(float(result.prices[i, j]) - expected) < 1e-12


def test_chain_with_vol_surface():
    """sigma can be a full grid, not just a scalar."""
    spot = 100.0
    strikes = np.array([90.0, 100.0, 110.0])
    expiries = np.array([0.25, 1.0])
    r = 0.05
    # Volatility smile: higher vol for OTM strikes
    sigma_surface = np.array([
        [0.30, 0.28],
        [0.22, 0.20],
        [0.28, 0.26],
    ])

    result = price_chain(spot, strikes, expiries, r, sigma_surface, "call")

    for i, K in enumerate(strikes):
        for j, T in enumerate(expiries):
            expected = float(price_call(spot, K, T, r, sigma_surface[i, j])[0])
            assert abs(float(result.prices[i, j]) - expected) < 1e-12


def test_chain_shape():
    """Output shape matches (n_strikes, n_expiries)."""
    spot = 100.0
    strikes = np.linspace(80, 120, 10)
    expiries = np.linspace(0.1, 2.0, 5)
    result = price_chain(spot, strikes, expiries, 0.05, 0.25, "call")

    assert result.prices.shape == (10, 5)
    assert result.n_contracts == 50


def test_chain_bad_sigma_shape_raises():
    """Mismatched sigma grid shape should raise ValueError."""
    spot = 100.0
    strikes = np.array([90.0, 100.0, 110.0])
    expiries = np.array([0.5, 1.0])
    wrong_sigma = np.ones((2, 2))  # should be (3, 2)

    with pytest.raises(ValueError):
        price_chain(spot, strikes, expiries, 0.05, wrong_sigma, "call")

def test_chain_without_greeks_is_none():
    result = price_chain(100.0, [100.0], [1.0], 0.05, 0.25, "call")
    assert result.greeks is None


def test_chain_with_greeks_populated():
    strikes = np.array([90.0, 100.0, 110.0])
    expiries = np.array([0.5, 1.0])
    result = price_chain(100.0, strikes, expiries, 0.05, 0.25, "call",
                         include_greeks=True)

    assert result.greeks is not None
    assert result.greeks.delta.shape == (3, 2)
    assert result.greeks.gamma.shape == (3, 2)
    assert result.greeks.vega.shape == (3, 2)
    assert result.greeks.theta.shape == (3, 2)
    assert result.greeks.rho.shape == (3, 2)


def test_chain_greeks_deltas_positive_for_call():
    """Call deltas should all be in (0, 1)."""
    strikes = np.array([80.0, 100.0, 120.0])
    expiries = np.array([0.5, 1.0])
    result = price_chain(100.0, strikes, expiries, 0.05, 0.25, "call",
                         include_greeks=True)
    assert np.all(result.greeks.delta > 0)
    assert np.all(result.greeks.delta < 1)

def test_benchmark_200_contract_chain_under_5ms():
    """
    THE HEADLINE BENCHMARK.
    Price a 200-contract chain (20 strikes x 10 expiries) in under 5ms
    on average across many runs.
    """
    stats = benchmark_chain(n_strikes=20, n_expiries=10, n_runs=200,
                            include_greeks=False)

    assert stats["n_contracts"] == 200
    assert stats["mean_ms"] < 5.0, (
        f"200-contract chain took {stats['mean_ms']:.3f}ms on average "
        f"(target: <5ms). Full stats: {stats}"
    )


def test_benchmark_500_contract_chain_reasonable():
    """500-contract chain should still be quick (under 15ms)."""
    stats = benchmark_chain(n_strikes=25, n_expiries=20, n_runs=100,
                            include_greeks=False)
    assert stats["mean_ms"] < 15.0, f"500-contract chain: {stats}"


def test_benchmark_with_greeks_under_15ms():
    """200-contract chain WITH Greeks should complete under 15ms."""
    stats = benchmark_chain(n_strikes=20, n_expiries=10, n_runs=100,
                            include_greeks=True)
    assert stats["mean_ms"] < 15.0, f"200-contract chain + greeks: {stats}"