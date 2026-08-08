from __future__ import annotations

from typing import Literal, Union

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

from options_engine.pricing import d1_d2, price_call, price_put

ArrayLike = Union[float, np.ndarray]

# Solver configuration
_SIGMA_LOW = 1e-4        # smallest sigma we search over (0.01%)
_SIGMA_HIGH = 5.0        # largest sigma we search over (500%)
_NEWTON_TOL = 1e-8       # convergence tolerance for Newton-Raphson
_NEWTON_MAX_ITER = 50    # max Newton iterations before falling back
_VEGA_MIN = 1e-10        # Vega threshold below which Newton fails
_BRENT_TOL = 1e-10       # convergence tolerance for Brent

# Arbitrage-bound tolerance (slack for floating-point noise)
_ARB_TOL = 1e-10


def _arbitrage_bounds(S, K, T, r, option_type):
    """Return (lower_bound, upper_bound) on option price."""
    disc_K = K * np.exp(-r * T)
    if option_type == "call":
        lower = np.maximum(S - disc_K, 0.0)
        upper = S
    else:
        lower = np.maximum(disc_K - S, 0.0)
        upper = disc_K
    return lower, upper


def _vega_at(S, K, T, r, sigma):
    """Compute raw Vega (not per-pct) at a specific sigma. Used by Newton."""
    d1, _ = d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


def _newton_raphson_one(
    market_price, S, K, T, r, option_type, initial_guess
):
    """
    Newton-Raphson for a single option. Returns (sigma, converged_bool).

    Iterates: sigma_new = sigma - (BSM(sigma) - market_price) / Vega(sigma)
    Bails out if:
      - Vega too small (unresponsive to sigma changes)
      - sigma wanders outside [_SIGMA_LOW, _SIGMA_HIGH]
      - hits _NEWTON_MAX_ITER without converging
    """
    sigma = initial_guess
    pricer = price_call if option_type == "call" else price_put

    for _ in range(_NEWTON_MAX_ITER):
        model_price = float(pricer(S, K, T, r, sigma)[0])
        residual = model_price - market_price

        if abs(residual) < _NEWTON_TOL:
            return sigma, True

        vega = float(_vega_at(S, K, T, r, sigma))
        if abs(vega) < _VEGA_MIN:
            return sigma, False  # unresponsive - can't proceed

        sigma_next = sigma - residual / vega

        if sigma_next < _SIGMA_LOW or sigma_next > _SIGMA_HIGH:
            return sigma, False  # wandered out of bounds

        sigma = sigma_next

    return sigma, False  # max iter without converging


def _brent_one(market_price, S, K, T, r, option_type):
    """
    Brent's method for a single option. Bracketed root-finding on the
    residual function BSM(sigma) - market_price. Always converges if
    the residual has opposite signs at the two endpoints.
    """
    pricer = price_call if option_type == "call" else price_put

    def residual(sigma):
        return float(pricer(S, K, T, r, sigma)[0]) - market_price

    try:
        sigma = brentq(
            residual,
            _SIGMA_LOW,
            _SIGMA_HIGH,
            xtol=_BRENT_TOL,
            maxiter=100,
        )
        return sigma
    except (ValueError, RuntimeError):
        return np.nan


def _initial_guess(market_price, S, K, T, r, option_type):
    """
    Manaster-Koehler initial guess for Newton-Raphson.
    Good enough that Newton usually converges in 3-5 iterations.
    """
    return np.sqrt(2 * abs(np.log(S / K) + r * T) / T)


def implied_volatility_one(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: Literal["call", "put"] = "call",
) -> float:
    """
    Solve for implied volatility given a single market option price.

    Returns np.nan if:
      - Price is outside the no-arbitrage bounds.
      - Neither Newton nor Brent converges.
    """
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)
    market_price = float(market_price)

    # Arbitrage pre-check
    lower, upper = _arbitrage_bounds(S, K, T, r, option_type)
    lower = float(lower)
    upper = float(upper)
    if market_price < lower - _ARB_TOL or market_price > upper + _ARB_TOL:
        return np.nan

    # Try Newton-Raphson first
# Try Newton-Raphson first
    guess = _initial_guess(market_price, S, K, T, r, option_type)
    guess = max(_SIGMA_LOW, min(_SIGMA_HIGH, guess))  # clamp to bracket
    sigma, converged = _newton_raphson_one(
        market_price, S, K, T, r, option_type, guess
    )

    # Verify Newton's answer by re-checking the residual with a tighter tolerance
    if converged:
        pricer = price_call if option_type == "call" else price_put
        model_price = float(pricer(S, K, T, r, sigma)[0])
        if abs(model_price - market_price) < 1e-10:
            return sigma
        # Otherwise Newton lied — fall through to Brent

    # Fall back to Brent
    return _brent_one(market_price, S, K, T, r, option_type)

def implied_volatility(
    market_price: ArrayLike,
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    option_type: Literal["call", "put"] = "call",
) -> np.ndarray:
    """
    Vectorized implied volatility solver. Returns an array of sigmas.
    Any position that couldn't be solved returns np.nan.
    """
    market_price = np.atleast_1d(np.asarray(market_price, dtype=float))
    S = np.atleast_1d(np.asarray(S, dtype=float))
    K = np.atleast_1d(np.asarray(K, dtype=float))
    T = np.atleast_1d(np.asarray(T, dtype=float))
    r = np.atleast_1d(np.asarray(r, dtype=float))

    market_price, S, K, T, r = np.broadcast_arrays(market_price, S, K, T, r)

    out = np.empty_like(market_price)
    for i in range(market_price.size):
        out.flat[i] = implied_volatility_one(
            float(market_price.flat[i]),
            float(S.flat[i]),
            float(K.flat[i]),
            float(T.flat[i]),
            float(r.flat[i]),
            option_type,
        )
    return out