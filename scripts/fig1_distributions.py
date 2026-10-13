"""
Figure 1: Distributions of A1 and A2 (histograms).
Shows the full dedup sample with median and mean lines.
"""
import numpy as np
import matplotlib.pyplot as plt
from _common import arrays, FIGS

arr = arrays()
A1 = arr["A1"]
A2 = arr["A2"]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# ---- Panel (a): A1 distribution ----
ax = axes[0]
xmax = 0.9
bins = np.arange(0, xmax + 0.001, 0.04)
ax.hist(A1[A1 <= xmax], bins=bins, color="steelblue", edgecolor="black",
        linewidth=0.6, alpha=0.85)
ax.axvline(np.median(A1), color="crimson", linestyle="--", linewidth=1.6,
           label=f"median = {np.median(A1):.3f}")
ax.axvline(A1.mean(), color="darkorange", linestyle=":", linewidth=1.6,
           label=f"mean = {A1.mean():.3f}")
ax.set_xlabel(r"$A_1$")
ax.set_ylabel("Number of galaxies")
ax.set_title(fr"(a) $A_1$ distribution, $N = {len(A1)}$")
ax.legend(loc="upper right", frameon=False)
ax.set_xlim(0, xmax)

# ---- Panel (b): A2 distribution ----
ax = axes[1]
bins = np.arange(0, xmax + 0.001, 0.04)
ax.hist(A2[A2 <= xmax], bins=bins, color="darkorange", edgecolor="black",
        linewidth=0.6, alpha=0.85)
ax.axvline(np.median(A2), color="crimson", linestyle="--", linewidth=1.6,
           label=f"median = {np.median(A2):.3f}")
ax.axvline(A2.mean(), color="steelblue", linestyle=":", linewidth=1.6,
           label=f"mean = {A2.mean():.3f}")
ax.set_xlabel(r"$A_2$")
ax.set_ylabel("Number of galaxies")
ax.set_title(fr"(b) $A_2$ distribution, $N = {len(A2)}$")
ax.legend(loc="upper right", frameon=False)
ax.set_xlim(0, xmax)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(FIGS / f"fig1.{ext}")
plt.close()

print(f"Figure 1 saved -> {FIGS/'fig1.pdf'}")
print(f"  A1: N={len(A1)}, median={np.median(A1):.3f}, mean={A1.mean():.3f}")
print(f"  A2: N={len(A2)}, median={np.median(A2):.3f}, mean={A2.mean():.3f}")
