"""
Figure 5: A1 vs physical parameters (M_B, V_rot, sigma_c, z) — 2x2 grid.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, kendalltau
from _common import arrays, FIGS

arr = arrays()

panels = [
    ("A1_B", "BMAG",   r"$M_B$ (absolute $B$ magnitude)",            True),
    ("A1_V", "Vrot",   r"$V_{\rm rot}$ (km s$^{-1}$)",               False),
    ("A1_s", "Sigmac", r"$\sigma_c$ (km s$^{-1}$)",                  False),
    ("A1_z", "zred",   r"redshift $z$",                              False),
]
titles = [
    "(a) $A_1$ vs luminosity",
    "(b) $A_1$ vs rotation velocity",
    "(c) $A_1$ vs central velocity dispersion",
    "(d) $A_1$ vs redshift (selection check)",
]

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

for ax, (ak, pk, xlab, invert), title in zip(axes.flat, panels, titles):
    y = arr[ak]
    x = arr[pk]
    N = len(x)
    rP, pP = pearsonr(x, y)
    tau, ptau = kendalltau(x, y)

    print(f"{title}: N={N}, Pearson r={rP:+.3f} (p={pP:.1e}), tau={tau:+.3f} (p={ptau:.1e})")

    # Scatter coloured by A1
    sc = ax.scatter(x, y, c=y, s=18, alpha=0.65, cmap="viridis",
                    edgecolor="none")
    plt.colorbar(sc, ax=ax, pad=0.02, label=r"$A_1$")

    # Binned median
    xmin, xmax = x.min(), x.max()
    nbins = 8 if pk != "zred" else 6
    edges = np.linspace(xmin, xmax, nbins + 1)
    xc, ym, yl, yh = [], [], [], []
    for i in range(nbins):
        sel = (x >= edges[i]) & (x < edges[i+1])
        n_in_bin = sel.sum()
        if n_in_bin >= 5:
            xc.append(0.5*(edges[i]+edges[i+1]))
            ym.append(np.median(y[sel]))
            if n_in_bin >= 10:
                # Dense bin: 16-84 percentile (= +/- 1 sigma for a Normal)
                yl.append(np.percentile(y[sel], 16))
                yh.append(np.percentile(y[sel], 84))
            else:
                # Sparse bin: 25-75 percentile (IQR, more robust for small N)
                yl.append(np.percentile(y[sel], 25))
                yh.append(np.percentile(y[sel], 75))
    if xc:
        xc = np.array(xc); ym = np.array(ym)
        yl = np.array(yl); yh = np.array(yh)
        ax.errorbar(xc, ym, yerr=[ym-yl, yh-ym],
                    fmt="o-", color="red", markersize=7, capsize=3,
                    linewidth=1.5, label=r"median $\pm 1\sigma$")

    ax.set_xlabel(xlab)
    ax.set_ylabel(r"$A_1$")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 1.0)
    if invert:
        ax.invert_xaxis()

    # Stat box
    p_str_P = f"{pP:.1e}"
    p_str_K = f"{ptau:.1e}"
    ax.text(0.03, 0.97,
            fr"$N = {N}$""\n"
            fr"$r_P = {rP:+.3f}$ ($p = {p_str_P}$)""\n"
            fr"$\tau = {tau:+.3f}$ ($p = {p_str_K}$)",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white",
                      alpha=0.9, edgecolor="gray"))
    ax.legend(loc="upper right", frameon=False, fontsize=9)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(FIGS / f"fig5.{ext}")
plt.close()
print(f"\nFigure 5 saved -> {FIGS/'fig5.pdf'}")
