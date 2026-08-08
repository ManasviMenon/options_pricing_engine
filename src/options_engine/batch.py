from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Optional, Union

import numpy as np

from options_engine.pricing import price_call, price_put
from options_engine.greeks import (
    GreekBundle,
    greeks_analytic,
)

ArrayLike = Union[float, np.ndarray]


@dataclass
class ChainResult:
    strikes: np.ndarray
    expiries: np.ndarray
    prices: np.ndarray
    greeks: Optional[GreekBundle] = None

    @property
    def n_contracts(self) -> int:
        return int(self.prices.size)


def price_chain(
    spot: float,
    strikes: ArrayLike,
    expiries: ArrayLike,
    r: float,
    sigma: Union[float, np.ndarray],
    option_type: Literal["call", "put"] = "call",
    *,
    include_greeks: bool = False,
) -> ChainResult:
   
    strikes = np.atleast_1d(np.asarray(strikes, dtype=float))
    expiries = np.atleast_1d(np.asarray(expiries, dtype=float))
    n_k = strikes.size
    n_t = expiries.size

    # Build the (n_strikes, n_expiries) grid via broadcasting
    K_grid = strikes[:, None]     # column vector
    T_grid = expiries[None, :]    # row vector
    S_grid = np.full((n_k, n_t), spot, dtype=float)
    r_grid = np.full((n_k, n_t), r, dtype=float)

    if np.isscalar(sigma):
        sigma_grid = np.full((n_k, n_t), float(sigma), dtype=float)
    else:
        sigma_grid = np.asarray(sigma, dtype=float)
        if sigma_grid.shape != (n_k, n_t):
            raise ValueError(
                f"sigma array shape {sigma_grid.shape} does not match "
                f"chain shape ({n_k}, {n_t})"
            )

    # Vectorized pricing across the entire grid at once
    if option_type == "call":
        prices = price_call(S_grid, K_grid, T_grid, r_grid, sigma_grid)
    else:
        prices = price_put(S_grid, K_grid, T_grid, r_grid, sigma_grid)

    prices = prices.reshape(n_k, n_t)

    greeks = None
    if include_greeks:
        greeks = greeks_analytic(
            S_grid, K_grid, T_grid, r_grid, sigma_grid, option_type
        )
        # Reshape each field to (n_k, n_t) - greeks_analytic returns flat arrays
        greeks = GreekBundle(
            price=greeks.price.reshape(n_k, n_t),
            delta=greeks.delta.reshape(n_k, n_t),
            gamma=greeks.gamma.reshape(n_k, n_t),
            vega=greeks.vega.reshape(n_k, n_t),
            theta=greeks.theta.reshape(n_k, n_t),
            rho=greeks.rho.reshape(n_k, n_t),
        )

    return ChainResult(
        strikes=strikes,
        expiries=expiries,
        prices=prices,
        greeks=greeks,
    )


def benchmark_chain(
    n_strikes: int = 20,
    n_expiries: int = 10,
    n_runs: int = 100,
    include_greeks: bool = False,
    option_type: Literal["call", "put"] = "call",
) -> dict:

    # Realistic input parameters
    spot = 100.0
    strikes = np.linspace(80, 120, n_strikes)
    expiries = np.linspace(0.05, 2.0, n_expiries)
    r = 0.05
    sigma = 0.25

    # Warm up (JIT-like caching in numpy/scipy)
    for _ in range(3):
        price_chain(spot, strikes, expiries, r, sigma, option_type,
                    include_greeks=include_greeks)

    # Timed runs
    times_ms = np.empty(n_runs, dtype=float)
    for i in range(n_runs):
        t0 = time.perf_counter()
        price_chain(spot, strikes, expiries, r, sigma, option_type,
                    include_greeks=include_greeks)
        times_ms[i] = (time.perf_counter() - t0) * 1000.0

    return {
        "n_contracts": n_strikes * n_expiries,
        "n_runs": n_runs,
        "mean_ms": float(np.mean(times_ms)),
        "median_ms": float(np.median(times_ms)),
        "min_ms": float(np.min(times_ms)),
        "max_ms": float(np.max(times_ms)),
        "std_ms": float(np.std(times_ms)),
    }