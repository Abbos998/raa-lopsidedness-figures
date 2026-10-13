"""
Figure 3: A1-A2 coupling (scatter + binned trend).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr, spearmanr
from _common import arrays, FIGS

arr = arrays()
A1 = arr["A1_j"]
A2 = arr["A2_j"]

print(f"Joint sample: N = {len(A1)}")
r_p, p_p = pearsonr(A1, A2)
r_s, p_s = spearmanr(A1, A2)
slope, intercept, _, _, se = stats.linregress(A1, A2)
print(f"  Pearson r = {r_p:+.3f}")
print(f"  Spearman rho = {r_s:+.3f}")
print(f"  Linear fit: A2 = {slope:.3f} A1 + {intercept:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

# ---- Panel (a): scatter + linear fit ----
ax = axes[0]
ax.scatter(A1, A2, s=18, color="teal", alpha=0.6, edgecolor="none")
xfit = np.linspace(0, A1.max()*1.05, 100)
ax.plot(xfit, slope*xfit + intercept, "r-", linewidth=2,
        label=fr"$A_2 = {slope:.2f}\,A_1 + {intercept:.2f}$")
ax.plot(xfit, xfit, "k--", linewidth=1, alpha=0.5, label=r"$y = x$")
ax.set_xlabel(r"$A_1$ ($m=1$ mode)")
ax.set_ylabel(r"$A_2$ ($m=2$ mode)")
ax.set_xlim(0, max(A1.max(), A2.max())*1.05)
ax.set_ylim(0, max(A1.max(), A2.max())*1.05)
ax.text(0.03, 0.97,
        fr"Pearson $r = {r_p:+.3f}$""\n"
        fr"Spearman $\rho = {r_s:+.3f}$""\n"
        fr"$p < 10^{{{int(np.log10(p_s))}}}$",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white",
                  alpha=0.9, edgecolor="gray"))
ax.legend(loc="lower right", frameon=False)

# ---- Panel (b): binned trend, colour by A1 ----
ax = axes[1]
sc = ax.scatter(A1, A2, c=A1, s=18, alpha=0.65,
                cmap="viridis", edgecolor="none", label="individual")
cb = plt.colorbar(sc, ax=ax, pad=0.02)
cb.set_label(r"$A_1$", fontsize=10)

# Binned medians
bins = np.array([0, 0.1, 0.2, 0.3, 0.5, 1.0])
xc, yc, ylo, yhi, Ns = [], [], [], [], []
for i in range(len(bins)-1):
    sel = (A1 >= bins[i]) & (A1 < bins[i+1])
    if sel.sum() >= 5:
        xc.append(0.5*(bins[i]+bins[i+1]))
        yc.append(np.median(A2[sel]))
        ylo.append(np.percentile(A2[sel], 16))
        yhi.append(np.percentile(A2[sel], 84))
        Ns.append(sel.sum())
xc = np.array(xc); yc = np.array(yc)
ylo = np.array(ylo); yhi = np.array(yhi)
ax.errorbar(xc, yc, yerr=[yc-ylo, yhi-yc],
            fmt="o-", color="darkred", markersize=10,
            capsize=4, linewidth=2, label=r"median $\pm 1\sigma$")
ax.set_xlabel(r"$A_1$")
ax.set_ylabel(r"$A_2$")
ax.set_xlim(0, max(A1.max(), A2.max())*1.05)
ax.set_ylim(0, max(A1.max(), A2.max())*1.05)
ax.legend(loc="upper right", frameon=False)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(FIGS / f"fig3.{ext}")
plt.close()
print(f"Figure 3 saved -> {FIGS/'fig3.pdf'}")
