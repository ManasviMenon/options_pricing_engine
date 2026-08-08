"""
options_engine - real-time Black-Scholes options pricing engine.

Public API:
    price_call, price_put               - vectorized BSM prices
    d1_d2                                - the two Z-scores
    greeks_analytic                      - closed-form Greeks
    greeks_finite_difference             - nudge-and-measure Greeks
    optimal_nudge                        - theoretically-optimal step size
    GreekBundle                          - container for price + all Greeks
    implied_volatility                   - vectorized IV solver
    implied_volatility_one                - scalar IV solver
    price_chain                          - batch pricer for full chains
    benchmark_chain                       - performance harness
    ChainResult                          - container for chain output
"""

from options_engine.pricing import (
    price_call,
    price_put,
    d1_d2,
)
from options_engine.greeks import (
    greeks_analytic,
    greeks_finite_difference,
    optimal_nudge,
    GreekBundle,
)
from options_engine.implied_vol import (
    implied_volatility,
    implied_volatility_one,
)
from options_engine.batch import (
    price_chain,
    benchmark_chain,
    ChainResult,
)

__all__ = [
    "price_call",
    "price_put",
    "d1_d2",
    "greeks_analytic",
    "greeks_finite_difference",
    "optimal_nudge",
    "GreekBundle",
    "implied_volatility",
    "implied_volatility_one",
    "price_chain",
    "benchmark_chain",
    "ChainResult",
]

__version__ = "0.4.0"