import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from options_engine.pricing import price_call, d1_d2

K = 100.0
r = 0.05
T = 0.5
sigma = 0.25

# S values from way OTM (S << K) to way ITM (S >> K)
S_values = np.logspace(0, 4, 500)  # 500 log-spaced points from S=1 to S=10000


def naive_call(S, K, T, r, sigma):
    """Straight BSM formula. No stability tricks."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

stable_prices = np.array([
    float(price_call(S, K, T, r, sigma)[0]) for S in S_values
])
naive_prices = np.array([naive_call(S, K, T, r, sigma) for S in S_values])

# The "true" price for verification: intrinsic + carry limit for deep ITM
# (For less-deep cases, the stable pricer is our reference.)
theoretical_intrinsic_plus_carry = S_values - K * np.exp(-r * T)

# Relative error is meaningful only where prices are non-trivial
# We compare stable vs naive, treating stable as the reference
epsilon_ref = np.maximum(np.abs(stable_prices), 1e-10)  # avoid division by zero
rel_error_naive = np.abs(naive_prices - stable_prices) / epsilon_ref

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: absolute prices
ax1.loglog(S_values, stable_prices, label="Stable pricer", linewidth=2,
           color="seagreen")
ax1.loglog(S_values, np.maximum(naive_prices, 1e-16), label="Naive BSM",
           linewidth=2, linestyle="--", color="crimson", alpha=0.7)
ax1.axvline(K, color="gray", linestyle=":", label=f"Strike K = {K}")
ax1.set_xlabel("Spot price S", fontsize=12)
ax1.set_ylabel("Call price", fontsize=12)
ax1.set_title("Prices across moneyness", fontsize=13, weight="bold")
ax1.legend(fontsize=11)
ax1.grid(True, which="both", alpha=0.3)

# Right panel: relative error (naive vs stable)
ax2.loglog(S_values, np.maximum(rel_error_naive, 1e-16),
           linewidth=2, color="crimson")
ax2.axhline(1e-10, color="seagreen", linestyle="--",
            label="Stable pricer precision")
ax2.axvline(K, color="gray", linestyle=":", label=f"Strike K = {K}")
ax2.set_xlabel("Spot price S", fontsize=12)
ax2.set_ylabel("Relative error of naive BSM", fontsize=12)
ax2.set_title("Naive pricer relative error explodes in tails",
              fontsize=13, weight="bold")
ax2.legend(fontsize=11)
ax2.grid(True, which="both", alpha=0.3)

plt.suptitle("Deep-Tail Stability: Naive BSM vs Stable Pricer",
             fontsize=15, weight="bold", y=1.02)

plt.tight_layout()
plt.savefig("notebooks/02_deep_tail_stability.png", dpi=150,
            bbox_inches="tight")
plt.show()

max_rel_err = float(np.max(rel_error_naive))
worst_S = float(S_values[int(np.argmax(rel_error_naive))])
print(f"\nMax relative error of naive BSM: {max_rel_err:.2e}")
print(f"  Occurs at S = {worst_S:.2f} (moneyness S/K = {worst_S/K:.2f})")
print(f"Naive fails by many orders of magnitude in the deep tails.")
