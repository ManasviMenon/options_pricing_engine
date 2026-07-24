"""
Black-Scholes-Merton pricer for European options.

"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

# Thresholds below which we route to the closed-form limit instead of the
# BSM formula. Chosen small enough to be indistinguishable from zero in
# any realistic use, large enough to safely avoid float underflow.
_T_EPS = 1e-10          # ~3 milliseconds of a year
_SIGMA_EPS = 1e-10      # essentially zero volatility
_DEEP_ITM_THRESHOLD = 8.0  # |d1| > 8 -> N(d1) is 1.0 to machine precision


def d1_d2(S, K, T, r, sigma):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.asarray(T, dtype=float)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    sigma_sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    return d1, d2


def _bsm_call_raw(S, K, T, r, sigma):
    """Bare Black-Scholes call formula. Assumes safe inputs (T>0, sigma>0)."""
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _bsm_put_raw(S, K, T, r, sigma):
    """Bare Black-Scholes put formula. Assumes safe inputs (T>0, sigma>0)."""
    d1, d2 = d1_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _broadcast_inputs(S, K, T, r, sigma):
    """Broadcast the five inputs to a common shape as float arrays."""
    arrs = [
        np.atleast_1d(np.asarray(x, dtype=float))
        for x in (S, K, T, r, sigma)
    ]
    return np.broadcast_arrays(*arrs)


def price_call(S, K, T, r, sigma):
    
    S, K, T, r, sigma = _broadcast_inputs(S, K, T, r, sigma)
    out = np.empty_like(S)

    # Regime 1: near expiry -> intrinsic value
    near_expiry = T < _T_EPS
    if np.any(near_expiry):
        out[near_expiry] = np.maximum(S[near_expiry] - K[near_expiry], 0.0)

    # Regime 2: zero volatility -> deterministic discounted payoff
    zero_vol = (sigma < _SIGMA_EPS) & ~near_expiry
    if np.any(zero_vol):
        F = S[zero_vol] * np.exp(r[zero_vol] * T[zero_vol])
        out[zero_vol] = np.maximum(F - K[zero_vol], 0.0) * np.exp(-r[zero_vol] * T[zero_vol])

    # Regime 3: normal case (including deep-tail branch for stability)
    normal = ~(near_expiry | zero_vol)
    if np.any(normal):
        d1n, _ = d1_d2(S[normal], K[normal], T[normal], r[normal], sigma[normal])
        deep_itm = d1n > _DEEP_ITM_THRESHOLD

        normal_prices = np.empty(int(normal.sum()), dtype=float)

        if np.any(deep_itm):
            # Route through the stable OTM put + parity
            put_stable = _bsm_put_raw(
                S[normal][deep_itm],
                K[normal][deep_itm],
                T[normal][deep_itm],
                r[normal][deep_itm],
                sigma[normal][deep_itm],
            )
            forward_minus_disc_strike = S[normal][deep_itm] - K[normal][deep_itm] * np.exp(
                -r[normal][deep_itm] * T[normal][deep_itm]
            )
            normal_prices[deep_itm] = put_stable + forward_minus_disc_strike

        stable = ~deep_itm
        if np.any(stable):
            normal_prices[stable] = _bsm_call_raw(
                S[normal][stable],
                K[normal][stable],
                T[normal][stable],
                r[normal][stable],
                sigma[normal][stable],
            )

        out[normal] = normal_prices

    return out


def price_put(S, K, T, r, sigma):
    S, K, T, r, sigma = _broadcast_inputs(S, K, T, r, sigma)
    out = np.empty_like(S)

    near_expiry = T < _T_EPS
    if np.any(near_expiry):
        out[near_expiry] = np.maximum(K[near_expiry] - S[near_expiry], 0.0)

    zero_vol = (sigma < _SIGMA_EPS) & ~near_expiry
    if np.any(zero_vol):
        F = S[zero_vol] * np.exp(r[zero_vol] * T[zero_vol])
        out[zero_vol] = np.maximum(K[zero_vol] - F, 0.0) * np.exp(-r[zero_vol] * T[zero_vol])

    normal = ~(near_expiry | zero_vol)
    if np.any(normal):
        d1n, _ = d1_d2(S[normal], K[normal], T[normal], r[normal], sigma[normal])
        deep_itm_put = d1n < -_DEEP_ITM_THRESHOLD

        normal_prices = np.empty(int(normal.sum()), dtype=float)

        if np.any(deep_itm_put):
            call_stable = _bsm_call_raw(
                S[normal][deep_itm_put],
                K[normal][deep_itm_put],
                T[normal][deep_itm_put],
                r[normal][deep_itm_put],
                sigma[normal][deep_itm_put],
            )
            forward_minus_disc_strike = S[normal][deep_itm_put] - K[normal][deep_itm_put] * np.exp(
                -r[normal][deep_itm_put] * T[normal][deep_itm_put]
            )
            normal_prices[deep_itm_put] = call_stable - forward_minus_disc_strike

        stable = ~deep_itm_put
        if np.any(stable):
            normal_prices[stable] = _bsm_put_raw(
                S[normal][stable],
                K[normal][stable],
                T[normal][stable],
                r[normal][stable],
                sigma[normal][stable],
            )

        out[normal] = normal_prices

    return out