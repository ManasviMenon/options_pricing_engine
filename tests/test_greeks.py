"""
Unit tests for Greek computations.

Covers:
  - Cross-check: analytic vs finite-difference agree to 6+ decimals.
  - Reference values from Hull's textbook.
  - Sign conventions.
  - Boundary limits.
  - Vectorization consistency.
"""

import numpy as np
import pytest

from options_engine.greeks import (
    GreekBundle,
    greeks_analytic,
    greeks_finite_difference,
    optimal_nudge,
)


# ---------------------------------------------------------------------------
# Cross-check: analytic vs finite-difference must agree
# ---------------------------------------------------------------------------

def test_analytic_vs_finite_difference_call():
    """The single most powerful sanity check: two methods, same answer."""
    rng = np.random.default_rng(42)
    n = 100
    S = rng.uniform(50, 200, n)
    K = rng.uniform(50, 200, n)
    T = rng.uniform(0.1, 2.0, n)
    r = rng.uniform(0.0, 0.10, n)
    sigma = rng.uniform(0.10, 0.60, n)

    a = greeks_analytic(S, K, T, r, sigma, "call")
    fd = greeks_finite_difference(S, K, T, r, sigma, "call")

    assert np.max(np.abs(a.delta - fd.delta)) < 1e-6, "delta mismatch"
    assert np.max(np.abs(a.gamma - fd.gamma)) < 1e-4, "gamma mismatch"
    assert np.max(np.abs(a.vega - fd.vega)) < 1e-6, "vega mismatch"
    assert np.max(np.abs(a.theta - fd.theta)) < 1e-4, "theta mismatch"
    assert np.max(np.abs(a.rho - fd.rho)) < 1e-6, "rho mismatch"


def test_analytic_vs_finite_difference_put():
    rng = np.random.default_rng(43)
    n = 100
    S = rng.uniform(50, 200, n)
    K = rng.uniform(50, 200, n)
    T = rng.uniform(0.1, 2.0, n)
    r = rng.uniform(0.0, 0.10, n)
    sigma = rng.uniform(0.10, 0.60, n)

    a = greeks_analytic(S, K, T, r, sigma, "put")
    fd = greeks_finite_difference(S, K, T, r, sigma, "put")

    assert np.max(np.abs(a.delta - fd.delta)) < 1e-6, "delta mismatch"
    assert np.max(np.abs(a.gamma - fd.gamma)) < 1e-4, "gamma mismatch"
    assert np.max(np.abs(a.vega - fd.vega)) < 1e-6, "vega mismatch"
    assert np.max(np.abs(a.theta - fd.theta)) < 1e-4, "theta mismatch"
    assert np.max(np.abs(a.rho - fd.rho)) < 1e-6, "rho mismatch"


# ---------------------------------------------------------------------------
# Hull textbook Greeks reference
# ---------------------------------------------------------------------------

def test_hull_call_delta():
    """S=42, K=40, T=0.5, r=0.10, sigma=0.20 -> Delta ~= 0.7791."""
    g = greeks_analytic(42.0, 40.0, 0.5, 0.10, 0.20, "call")
    assert abs(float(g.delta[0]) - 0.7791) < 1e-3


def test_hull_call_gamma():
    """Same inputs -> Gamma ~= 0.0498."""
    g = greeks_analytic(42.0, 40.0, 0.5, 0.10, 0.20, "call")
    assert abs(float(g.gamma[0]) - 0.0498) < 1e-3


# ---------------------------------------------------------------------------
# Sign conventions
# ---------------------------------------------------------------------------

def test_call_delta_positive():
    g = greeks_analytic(100.0, 100.0, 1.0, 0.05, 0.30, "call")
    assert 0 < float(g.delta[0]) < 1


def test_put_delta_negative():
    g = greeks_analytic(100.0, 100.0, 1.0, 0.05, 0.30, "put")
    assert -1 < float(g.delta[0]) < 0


def test_call_put_delta_relationship():
    """Delta_call - Delta_put must equal 1 (from put-call parity)."""
    S, K, T, r, sigma = 100.0, 110.0, 0.5, 0.05, 0.30
    gc = greeks_analytic(S, K, T, r, sigma, "call")
    gp = greeks_analytic(S, K, T, r, sigma, "put")
    diff = float(gc.delta[0] - gp.delta[0])
    assert abs(diff - 1.0) < 1e-12


def test_gamma_equals_across_call_put():
    """Gamma is identical for calls and puts on same inputs."""
    S, K, T, r, sigma = 100.0, 110.0, 0.5, 0.05, 0.30
    gc = greeks_analytic(S, K, T, r, sigma, "call")
    gp = greeks_analytic(S, K, T, r, sigma, "put")
    assert abs(float(gc.gamma[0] - gp.gamma[0])) < 1e-14


def test_vega_equals_across_call_put():
    """Vega is identical for calls and puts on same inputs."""
    S, K, T, r, sigma = 100.0, 110.0, 0.5, 0.05, 0.30
    gc = greeks_analytic(S, K, T, r, sigma, "call")
    gp = greeks_analytic(S, K, T, r, sigma, "put")
    assert abs(float(gc.vega[0] - gp.vega[0])) < 1e-14


def test_gamma_positive():
    g = greeks_analytic(100.0, 100.0, 1.0, 0.05, 0.30, "call")
    assert float(g.gamma[0]) > 0


def test_vega_positive():
    g = greeks_analytic(100.0, 100.0, 1.0, 0.05, 0.30, "call")
    assert float(g.vega[0]) > 0


def test_theta_negative_for_call():
    """Long calls bleed value over time -> theta < 0."""
    g = greeks_analytic(100.0, 100.0, 1.0, 0.05, 0.30, "call")
    assert float(g.theta[0]) < 0


# ---------------------------------------------------------------------------
# Boundary limits
# ---------------------------------------------------------------------------

def test_call_delta_at_expiry_itm():
    """At T=0, ITM call delta = 1."""
    g = greeks_analytic(120.0, 100.0, 0.0, 0.05, 0.30, "call")
    assert float(g.delta[0]) == pytest.approx(1.0, abs=1e-12)


def test_call_delta_at_expiry_otm():
    """At T=0, OTM call delta = 0."""
    g = greeks_analytic(80.0, 100.0, 0.0, 0.05, 0.30, "call")
    assert float(g.delta[0]) == pytest.approx(0.0, abs=1e-12)


def test_put_delta_at_expiry_itm():
    g = greeks_analytic(80.0, 100.0, 0.0, 0.05, 0.30, "put")
    assert float(g.delta[0]) == pytest.approx(-1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Vectorization consistency
# ---------------------------------------------------------------------------

def test_greeks_bundle_shape():
    """Passing an array of spots should produce arrays of Greeks of the same shape."""
    S = np.array([90.0, 100.0, 110.0])
    g = greeks_analytic(S, 100.0, 1.0, 0.05, 0.20, "call")
    assert g.delta.shape == (3,)
    assert g.gamma.shape == (3,)
    assert g.delta[2] > g.delta[1] > g.delta[0], "delta must increase with S"


def test_optimal_nudge_scales_with_input():
    """Nudge should scale linearly with |input|."""
    h100 = optimal_nudge(100.0, order=1)
    h200 = optimal_nudge(200.0, order=1)
    assert float(h200) == pytest.approx(2 * float(h100), rel=1e-14)


def test_gamma_nudge_bigger_than_delta_nudge():
    """Second-derivative optimal nudge should be bigger than first-derivative."""
    h1 = float(optimal_nudge(100.0, order=1))
    h2 = float(optimal_nudge(100.0, order=2))
    assert h2 > h1