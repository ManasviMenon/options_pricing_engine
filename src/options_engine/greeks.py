"""
Greek computations for European options.

Two parallel implementations of every Greek:
    (1) Analytic - closed-form derivatives of BSM. Fast, exact.
    (2) Finite-difference - central differences with optimal nudge size.
        Model-agnostic sanity check.

If the two implementations disagree, one has a bug. This is the single
best correctness test in numerical finance code.

Optimal nudge sizes:
    First derivatives:  h ~= eps^(1/3) * |x|   (~6e-6 * |x|)
    Second derivatives: h ~= eps^(1/4) * |x|   (~1e-4 * |x|)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
from scipy.stats import norm

from options_engine.pricing import d1_d2, price_call, price_put

ArrayLike = Union[float, np.ndarray]

# Machine epsilon for float64 (~2.22e-16)
_EPS = np.finfo(float).eps

# Optimal nudge multipliers derived from balancing truncation and roundoff
_H_FIRST = _EPS ** (1 / 3)   # for first derivatives (~6e-6)
_H_SECOND = _EPS ** (1 / 4)  # for second derivatives (~1e-4)

# Regime thresholds (kept in sync with pricing.py)
_T_EPS = 1e-10
_SIGMA_EPS = 1e-10


@dataclass
class GreekBundle:
    """
    Container for the five standard Greeks plus the price.

    Sign conventions:
        delta: [+] for calls, [-] for puts
        gamma: [+] for both (long options are long convexity)
        vega:  scaled per +1% change in sigma (market convention)
        theta: scaled per -1 calendar day (value lost per day owned)
        rho:   scaled per +1% change in r (market convention)
    """
    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    theta: np.ndarray
    rho: np.ndarray

    def as_dict(self) -> dict:
        return {
            "price": self.price,
            "delta": self.delta,
            "gamma": self.gamma,
            "vega": self.vega,
            "theta": self.theta,
            "rho": self.rho,
        }


def optimal_nudge(x: ArrayLike, order: int = 1) -> np.ndarray:
    """
    Compute the optimal finite-difference nudge for input x.

    order=1 for first derivatives (delta, vega, theta, rho).
    order=2 for second derivatives (gamma).
    """
    x_abs = np.abs(np.asarray(x, dtype=float))
    x_abs = np.maximum(x_abs, 1.0)
    if order == 1:
        return _H_FIRST * x_abs
    elif order == 2:
        return _H_SECOND * x_abs
    else:
        raise ValueError(f"order must be 1 or 2, got {order}")


def _broadcast(S, K, T, r, sigma):
    """Broadcast the five inputs to matching-shape float arrays."""
    arrs = [
        np.atleast_1d(np.asarray(x, dtype=float))
        for x in (S, K, T, r, sigma)
    ]
    return np.broadcast_arrays(*arrs)


# ============================================================================
# ANALYTIC GREEKS - closed-form derivatives of BSM
# ============================================================================

def _analytic_call_greeks(S, K, T, r, sigma):
    """Raw closed-form Greeks for a European call. Assumes safe inputs."""
    d1, d2 = d1_d2(S, K, T, r, sigma)
    sqrt_T = np.sqrt(T)
    phi_d1 = norm.pdf(d1)
    disc = np.exp(-r * T)

    delta = norm.cdf(d1)
    gamma = phi_d1 / (S * sigma * sqrt_T)
    vega = S * phi_d1 * sqrt_T
    theta = -(S * phi_d1 * sigma) / (2 * sqrt_T) - r * K * disc * norm.cdf(d2)
    rho = K * T * disc * norm.cdf(d2)
    return delta, gamma, vega, theta, rho


def _analytic_put_greeks(S, K, T, r, sigma):
    """Raw closed-form Greeks for a European put. Assumes safe inputs."""
    d1, d2 = d1_d2(S, K, T, r, sigma)
    sqrt_T = np.sqrt(T)
    phi_d1 = norm.pdf(d1)
    disc = np.exp(-r * T)

    delta = norm.cdf(d1) - 1.0
    gamma = phi_d1 / (S * sigma * sqrt_T)
    vega = S * phi_d1 * sqrt_T
    theta = -(S * phi_d1 * sigma) / (2 * sqrt_T) + r * K * disc * norm.cdf(-d2)
    rho = -K * T * disc * norm.cdf(-d2)
    return delta, gamma, vega, theta, rho


def greeks_analytic(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    option_type: Literal["call", "put"] = "call",
    *,
    vega_per_pct: bool = True,
    rho_per_pct: bool = True,
    theta_per_day: bool = True,
) -> GreekBundle:
    """Vectorized analytic Greeks with degeneracy handling and market conventions."""
    S, K, T, r, sigma = _broadcast(S, K, T, r, sigma)
    price = price_call(S, K, T, r, sigma) if option_type == "call" else price_put(S, K, T, r, sigma)

    delta = np.zeros_like(S)
    gamma = np.zeros_like(S)
    vega = np.zeros_like(S)
    theta = np.zeros_like(S)
    rho = np.zeros_like(S)

    # Regime 1: T -> 0
    near_expiry = T < _T_EPS
    if np.any(near_expiry):
        if option_type == "call":
            itm = S[near_expiry] > K[near_expiry]
            delta[near_expiry] = itm.astype(float)
        else:
            itm = S[near_expiry] < K[near_expiry]
            delta[near_expiry] = -itm.astype(float)

    # Regime 2: sigma -> 0 (deterministic stock)
    zero_vol = (sigma < _SIGMA_EPS) & ~near_expiry
    if np.any(zero_vol):
        F = S[zero_vol] * np.exp(r[zero_vol] * T[zero_vol])
        disc = np.exp(-r[zero_vol] * T[zero_vol])
        if option_type == "call":
            itm = F > K[zero_vol]
            delta[zero_vol] = itm.astype(float)
            rho[zero_vol] = np.where(itm, K[zero_vol] * T[zero_vol] * disc, 0.0)
            theta[zero_vol] = np.where(itm, -r[zero_vol] * K[zero_vol] * disc, 0.0)
        else:
            itm = F < K[zero_vol]
            delta[zero_vol] = np.where(itm, -1.0, 0.0)
            rho[zero_vol] = np.where(itm, -K[zero_vol] * T[zero_vol] * disc, 0.0)
            theta[zero_vol] = np.where(itm, r[zero_vol] * K[zero_vol] * disc, 0.0)

    # Regime 3: normal
    normal = ~(near_expiry | zero_vol)
    if np.any(normal):
        if option_type == "call":
            d, g, v, th, rh = _analytic_call_greeks(
                S[normal], K[normal], T[normal], r[normal], sigma[normal]
            )
        else:
            d, g, v, th, rh = _analytic_put_greeks(
                S[normal], K[normal], T[normal], r[normal], sigma[normal]
            )
        delta[normal] = d
        gamma[normal] = g
        vega[normal] = v
        theta[normal] = th
        rho[normal] = rh

    # Market conventions
    if vega_per_pct:
        vega = vega * 0.01
    if rho_per_pct:
        rho = rho * 0.01
    if theta_per_day:
        theta = theta / 365.0

    return GreekBundle(price=price, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


# ============================================================================
# FINITE-DIFFERENCE GREEKS - central differences with optimal nudge
# ============================================================================

def greeks_finite_difference(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    r: ArrayLike,
    sigma: ArrayLike,
    option_type: Literal["call", "put"] = "call",
    *,
    h_S: Union[float, None] = None,
    h_sigma: Union[float, None] = None,
    h_T: Union[float, None] = None,
    h_r: Union[float, None] = None,
    vega_per_pct: bool = True,
    rho_per_pct: bool = True,
    theta_per_day: bool = True,
) -> GreekBundle:
    """Vectorized finite-difference Greeks. Nudges default to optimal sizes."""
    S, K, T, r, sigma = _broadcast(S, K, T, r, sigma)
    pricer = price_call if option_type == "call" else price_put

    # Nudge sizes - scale by max |input| in the batch
    hS = float(_H_FIRST * max(float(np.max(np.abs(S))), 1.0)) if h_S is None else h_S
    hSg = float(_H_SECOND * max(float(np.max(np.abs(S))), 1.0)) if h_S is None else h_S
    hSig = float(_H_FIRST * max(float(np.max(np.abs(sigma))), 1.0)) if h_sigma is None else h_sigma
    hT = float(_H_FIRST * max(float(np.max(np.abs(T))), 1.0)) if h_T is None else h_T
    hR = float(_H_FIRST * max(float(np.max(np.abs(r))), 1.0)) if h_r is None else h_r

    p0 = pricer(S, K, T, r, sigma)

    # Delta = dPrice/dS
    delta = (pricer(S + hS, K, T, r, sigma) - pricer(S - hS, K, T, r, sigma)) / (2 * hS)

    # Gamma = d^2 Price / dS^2 (bigger nudge because second derivative)
    gamma = (
        pricer(S + hSg, K, T, r, sigma)
        - 2 * pricer(S, K, T, r, sigma)
        + pricer(S - hSg, K, T, r, sigma)
    ) / (hSg * hSg)

    # Vega = dPrice/dsigma
    vega = (pricer(S, K, T, r, sigma + hSig) - pricer(S, K, T, r, sigma - hSig)) / (2 * hSig)

    # Theta = -dPrice/dT (value lost per unit time)
    T_up = T + hT
    T_dn = np.maximum(T - hT, _T_EPS)
    theta = -(pricer(S, K, T_up, r, sigma) - pricer(S, K, T_dn, r, sigma)) / (T_up - T_dn)

    # Rho = dPrice/dr
    rho = (pricer(S, K, T, r + hR, sigma) - pricer(S, K, T, r - hR, sigma)) / (2 * hR)

    if vega_per_pct:
        vega = vega * 0.01
    if rho_per_pct:
        rho = rho * 0.01
    if theta_per_day:
        theta = theta / 365.0

    return GreekBundle(price=p0, delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)