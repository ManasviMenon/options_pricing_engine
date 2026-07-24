import numpy as np
import pytest

from options_engine.pricing import d1_d2, price_call, price_put


# ---------------------------------------------------------------------------
# Hull textbook reference (correctness anchor)
# ---------------------------------------------------------------------------

def test_hull_reference_call():
    """S=42, K=40, T=0.5, r=0.10, sigma=0.20 -> Call ~= 4.7594."""
    price = float(price_call(42.0, 40.0, 0.5, 0.10, 0.20)[0])
    assert abs(price - 4.7594) < 1e-3


def test_hull_reference_put():
    """S=42, K=40, T=0.5, r=0.10, sigma=0.20 -> Put ~= 0.8086."""
    price = float(price_put(42.0, 40.0, 0.5, 0.10, 0.20)[0])
    assert abs(price - 0.8086) < 1e-3


# ---------------------------------------------------------------------------
# Put-call parity: C - P = S - K * exp(-r*T)
# ---------------------------------------------------------------------------

def test_put_call_parity_scalar():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
    C = float(price_call(S, K, T, r, sigma)[0])
    P = float(price_put(S, K, T, r, sigma)[0])
    lhs = C - P
    rhs = S - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-12


def test_put_call_parity_batch():
    """Parity must hold across a large random grid to machine precision."""
    rng = np.random.default_rng(42)
    n = 1000
    S = rng.uniform(50, 200, n)
    K = rng.uniform(50, 200, n)
    T = rng.uniform(0.01, 2.0, n)
    r = rng.uniform(0.0, 0.10, n)
    sigma = rng.uniform(0.05, 0.80, n)

    C = price_call(S, K, T, r, sigma)
    P = price_put(S, K, T, r, sigma)
    lhs = C - P
    rhs = S - K * np.exp(-r * T)
    max_err = float(np.max(np.abs(lhs - rhs)))
    assert max_err < 1e-9, f"max parity error {max_err}"


def test_put_call_parity_deep_itm():
    """Parity must hold even through the deep-ITM branch."""
    S, K, T, r, sigma = 500.0, 100.0, 0.5, 0.05, 0.30
    C = float(price_call(S, K, T, r, sigma)[0])
    P = float(price_put(S, K, T, r, sigma)[0])
    lhs = C - P
    rhs = S - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-10


def test_put_call_parity_deep_otm():
    """Parity must hold in deep-OTM regime too."""
    S, K, T, r, sigma = 50.0, 300.0, 0.5, 0.05, 0.30
    C = float(price_call(S, K, T, r, sigma)[0])
    P = float(price_put(S, K, T, r, sigma)[0])
    lhs = C - P
    rhs = S - K * np.exp(-r * T)
    assert abs(lhs - rhs) < 1e-10


# ---------------------------------------------------------------------------
# Boundary limits (regime handling)
# ---------------------------------------------------------------------------

def test_zero_time_call_itm():
    """At T=0, ITM call = intrinsic S - K."""
    assert float(price_call(120.0, 100.0, 0.0, 0.05, 0.30)[0]) == pytest.approx(20.0, abs=1e-14)


def test_zero_time_call_otm():
    """At T=0, OTM call = 0."""
    assert float(price_call(80.0, 100.0, 0.0, 0.05, 0.30)[0]) == pytest.approx(0.0, abs=1e-14)


def test_zero_time_put_itm():
    assert float(price_put(80.0, 100.0, 0.0, 0.05, 0.30)[0]) == pytest.approx(20.0, abs=1e-14)


def test_zero_time_put_otm():
    assert float(price_put(120.0, 100.0, 0.0, 0.05, 0.30)[0]) == pytest.approx(0.0, abs=1e-14)


def test_zero_vol_call_deterministic():
    """At sigma=0, call = max(F - K, 0) * exp(-rT), F = S*exp(rT)."""
    S, K, T, r = 100.0, 100.0, 1.0, 0.05
    F = S * np.exp(r * T)
    expected = max(F - K, 0.0) * np.exp(-r * T)
    assert float(price_call(S, K, T, r, 0.0)[0]) == pytest.approx(expected, abs=1e-12)


def test_zero_vol_put_deterministic():
    S, K, T, r = 100.0, 120.0, 1.0, 0.05
    F = S * np.exp(r * T)
    expected = max(K - F, 0.0) * np.exp(-r * T)
    assert float(price_put(S, K, T, r, 0.0)[0]) == pytest.approx(expected, abs=1e-12)


# ---------------------------------------------------------------------------
# Deep-tail stability
# ---------------------------------------------------------------------------

def test_deep_itm_call_matches_intrinsic_plus_carry():
    """Deep ITM call -> S - K*exp(-rT), the intrinsic + carry limit."""
    S, K, T, r, sigma = 1000.0, 50.0, 0.25, 0.05, 0.20
    price = float(price_call(S, K, T, r, sigma)[0])
    expected = S - K * np.exp(-r * T)
    assert price == pytest.approx(expected, rel=1e-8)


def test_deep_otm_call_near_zero():
    """Deep OTM call -> 0."""
    price = float(price_call(10.0, 1000.0, 0.25, 0.05, 0.20)[0])
    assert 0.0 <= price < 1e-6


# ---------------------------------------------------------------------------
# Monotonicity (economic sanity)
# ---------------------------------------------------------------------------

def test_call_monotonic_in_S():
    Ss = np.linspace(50, 200, 50)
    prices = price_call(Ss, 100.0, 0.5, 0.05, 0.30)
    assert np.all(np.diff(prices) > 0), "call must strictly increase in S"


def test_call_monotonic_decreasing_in_K():
    Ks = np.linspace(50, 200, 50)
    prices = price_call(100.0, Ks, 0.5, 0.05, 0.30)
    assert np.all(np.diff(prices) < 0), "call must strictly decrease in K"


def test_call_monotonic_in_sigma():
    sigmas = np.linspace(0.05, 1.0, 50)
    prices = price_call(100.0, 100.0, 0.5, 0.05, sigmas)
    assert np.all(np.diff(prices) > 0), "call must strictly increase in sigma"


def test_put_monotonic_in_sigma():
    sigmas = np.linspace(0.05, 1.0, 50)
    prices = price_put(100.0, 100.0, 0.5, 0.05, sigmas)
    assert np.all(np.diff(prices) > 0), "put must strictly increase in sigma"


def test_call_monotonic_in_T():
    Ts = np.linspace(0.01, 2.0, 50)
    prices = price_call(100.0, 100.0, Ts, 0.05, 0.30)
    assert np.all(np.diff(prices) > 0), "ATM call must strictly increase in T"


# ---------------------------------------------------------------------------
# Vectorization consistency
# ---------------------------------------------------------------------------

def test_scalar_matches_array():
    scalar = float(price_call(100.0, 100.0, 1.0, 0.05, 0.20)[0])
    arr = float(
        price_call(
            np.array([100.0]), np.array([100.0]),
            np.array([1.0]), np.array([0.05]), np.array([0.20]),
        )[0]
    )
    assert scalar == pytest.approx(arr, abs=1e-14)


def test_broadcasting_shape():
    S = np.array([90.0, 100.0, 110.0])
    prices = price_call(S, 100.0, 1.0, 0.05, 0.20)
    assert prices.shape == (3,)
    assert prices[2] > prices[1] > prices[0]


def test_d1_minus_d2_equals_sigma_sqrt_T():
    """Mathematical identity that must hold to machine precision."""
    S, K, T, r, sigma = 100.0, 110.0, 0.5, 0.05, 0.25
    d1, d2 = d1_d2(S, K, T, r, sigma)
    assert float(d1 - d2) == pytest.approx(sigma * np.sqrt(T), abs=1e-14)


# ---------------------------------------------------------------------------
# Robustness: no NaN or inf under extreme inputs
# ---------------------------------------------------------------------------

def test_no_nan_extreme_grid():
    """Every combination of extreme inputs must produce a finite, non-negative price."""
    S_vals = [1e-6, 1e6]
    K_vals = [1e-6, 1e6]
    T_vals = [1e-12, 100.0]
    r_vals = [0.0, 0.20]
    sigma_vals = [1e-12, 5.0]

    for S in S_vals:
        for K in K_vals:
            for T in T_vals:
                for r in r_vals:
                    for sigma in sigma_vals:
                        c = float(price_call(S, K, T, r, sigma)[0])
                        p = float(price_put(S, K, T, r, sigma)[0])
                        assert np.isfinite(c), f"call NaN at {S=} {K=} {T=} {r=} {sigma=}"
                        assert np.isfinite(p), f"put NaN at {S=} {K=} {T=} {r=} {sigma=}"
                        assert c >= -1e-9
                        assert p >= -1e-9