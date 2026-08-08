import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from options_engine.pricing import price_call
from options_engine.implied_vol import implied_volatility_one

K = 100.0
r = 0.05
true_sigma = 0.25

moneyness = np.linspace(0.5, 2.0, 40)     # S/K from 0.5 (deep OTM) to 2.0 (deep ITM)
expiries = np.linspace(0.05, 2.0, 40)     # T from 2 weeks to 2 years

n_m = len(moneyness)
n_t = len(expiries)

errors = np.full((n_m, n_t), np.nan)
prices = np.full((n_m, n_t), np.nan)

for i, m in enumerate(moneyness):
    S = m * K
    for j, T in enumerate(expiries):
        # Compute market price at the true sigma
        price = float(price_call(S, K, T, r, true_sigma)[0])
        prices[i, j] = price
        # Skip degenerate near-zero prices where sigma is ambiguous
        if price < 1e-4:
            continue
        recovered = implied_volatility_one(price, S, K, T, r, "call")
        if not np.isnan(recovered):
            errors[i, j] = abs(recovered - true_sigma)


fig, ax = plt.subplots(figsize=(10, 8))

# Mask NaN (degenerate) regions so they show as blank
masked = np.ma.masked_invalid(errors)

# Clip to plotting range
min_err = 1e-14
max_err = 1e-2
plot_data = np.clip(masked, min_err, max_err)

im = ax.pcolormesh(
    expiries, moneyness, plot_data,
    norm=LogNorm(vmin=min_err, vmax=max_err),
    cmap="viridis_r", shading="auto"
)

ax.set_xlabel("Time to expiry (years)", fontsize=12)
ax.set_ylabel("Moneyness (S/K)", fontsize=12)
ax.set_title(
    "IV Solver Recovery Error Across Parameter Space\n"
    "(σ_true = 0.25, grid of 1600 test points)",
    fontsize=13, weight="bold"
)

# Reference lines
ax.axhline(1.0, color="white", linestyle="--", linewidth=1, alpha=0.7)
ax.text(0.1, 1.02, "ATM (S=K)", color="white", fontsize=10, weight="bold")

cbar = fig.colorbar(im, ax=ax, label="|σ_recovered − σ_true|")
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
plt.savefig("notebooks/03_iv_solver_reliability.png", dpi=150,
            bbox_inches="tight")
plt.show()

valid = ~np.isnan(errors)
if np.any(valid):
    print(f"\nValid recovery cases:  {int(valid.sum())} / {n_m * n_t}")
    print(f"Median error:          {float(np.nanmedian(errors)):.2e}")
    print(f"Max error:             {float(np.nanmax(errors)):.2e}")
    print(f"Cases below 1e-6:      {int(np.sum(errors < 1e-6))} "
          f"({100 * np.sum(errors < 1e-6) / valid.sum():.1f}%)")