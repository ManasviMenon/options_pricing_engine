import numpy as np
import matplotlib.pyplot as plt

from options_engine.pricing import price_call
from options_engine.greeks import greeks_analytic


S = 100.0
K = 100.0
T = 1.0
r = 0.05
sigma = 0.25

# Machine epsilon and the theoretical optimal nudge for first derivatives
EPS = np.finfo(float).eps
H_OPTIMAL = EPS ** (1 / 3) * S   # sweet spot for delta


true_delta = float(
    greeks_analytic(S, K, T, r, sigma, "call").delta[0]
)
print(f"Analytic delta (truth): {true_delta:.10f}")

nudge_sizes = np.logspace(-14, 0, 200)  # 200 log-spaced points

fd_deltas = np.empty_like(nudge_sizes)
for i, h in enumerate(nudge_sizes):
    up = float(price_call(S + h, K, T, r, sigma)[0])
    dn = float(price_call(S - h, K, T, r, sigma)[0])
    fd_deltas[i] = (up - dn) / (2 * h)

errors = np.abs(fd_deltas - true_delta)

fig, ax = plt.subplots(figsize=(10, 6))

ax.loglog(nudge_sizes, errors, linewidth=2, color="steelblue",
          label="Finite-difference error")

# Mark the theoretical sweet spot
ax.axvline(H_OPTIMAL, color="crimson", linestyle="--", linewidth=1.5,
           label=f"Theoretical optimum ≈ {H_OPTIMAL:.2e}")

ax.set_xlabel("Nudge size h", fontsize=12)
ax.set_ylabel("|Δ_fd − Δ_analytic|", fontsize=12)
ax.set_title("Finite-Difference Delta: The Nudge-Size Sweet Spot",
             fontsize=14, weight="bold")

# Annotate the two failure regions
ax.text(1e-13, 1e-2, "Roundoff error\ndominates",
        ha="left", va="center", fontsize=11, color="dimgray",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow"))
ax.text(1e-1, 1e-2, "Truncation error\ndominates",
        ha="right", va="center", fontsize=11, color="dimgray",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow"))

ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper center", fontsize=11)

plt.tight_layout()

# Save it to the notebooks folder
plt.savefig("notebooks/01_delta_ucurve.png", dpi=150, bbox_inches="tight")
plt.show()

best_idx = int(np.argmin(errors))
print(f"\nEmpirical best nudge:    {nudge_sizes[best_idx]:.2e}")
print(f"Theoretical optimum:     {H_OPTIMAL:.2e}")
print(f"Error at best nudge:     {errors[best_idx]:.2e}")
print(f"Error at h=1e-14:        {errors[0]:.2e}")
print(f"Error at h=1e0:          {errors[-1]:.2e}")