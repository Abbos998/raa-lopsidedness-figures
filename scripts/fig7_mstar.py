"""
Figure 7: Stellar-mass subsample — A1 vs stellar mass M* (from Table 1 cross-match
with the deduplicated Table 2).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
from _common import load_table1_mstar, load_table2_dedup, FIGS

# Cross-match deduplicated Table 2 with Table 1 (which has logMstar)
t1 = load_table1_mstar()
t2_rows = load_table2_dedup()

mstar_sample = []
for r in t2_rows:
    if r["A1"] is None or r["A1"] > 1.0:
        continue
    if r["norm"] in t1:
        lm = t1[r["norm"]]["logMstar"]
        if lm is not None:
            mstar_sample.append({
                "norm": r["norm"],
                "A1": r["A1"],
                "A2": r["A2"],
                "logMstar": lm,
                "TType": r["TType"],
            })

print(f"Stellar-mass subsample: N = {len(mstar_sample)}")

A1 = np.array([g["A1"] for g in mstar_sample])
lMs = np.array([g["logMstar"] for g in mstar_sample])

print(f"  logM* range : [{lMs.min():.2f}, {lMs.max():.2f}]")
print(f"  logM* median: {np.median(lMs):.2f}")
print(f"  A1 median:    {np.median(A1):.3f}")

rP, pP = pearsonr(lMs, A1)
rS, pS = spearmanr(lMs, A1)
tau, ptau = kendalltau(lMs, A1)
print(f"  Pearson r = {rP:+.3f} (p = {pP:.2e})")
print(f"  Spearman rho = {rS:+.3f} (p = {pS:.2e})")
print(f"  Kendall tau = {tau:+.3f} (p = {ptau:.2e})")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ---- Panel (a): scatter + binned trend ----
ax = axes[0]
sc = ax.scatter(lMs, A1, c=A1, s=22, alpha=0.65, cmap="viridis",
                edgecolor="none")
plt.colorbar(sc, ax=ax, pad=0.02, label=r"$A_1$")

# Binned medians
mbins = np.arange(8.0, 11.6, 0.4)
xc, ym, yl, yh, Ns = [], [], [], [], []
for i in range(len(mbins)-1):
    sel = (lMs >= mbins[i]) & (lMs < mbins[i+1])
    if sel.sum() >= 5:
        xc.append(np.median(lMs[sel]))
        ym.append(np.median(A1[sel]))
        yl.append(np.percentile(A1[sel], 16))
        yh.append(np.percentile(A1[sel], 84))
        Ns.append(sel.sum())
xc = np.array(xc); ym = np.array(ym)
yl = np.array(yl); yh = np.array(yh)
ax.errorbar(xc, ym, yerr=[ym-yl, yh-ym],
            fmt="o-", color="crimson", markersize=10, capsize=4,
            linewidth=2, label=r"median $\pm 1\sigma$ ($N \geq 5$)")
ax.set_xlabel(r"$\log(M_\star / M_\odot)$")
ax.set_ylabel(r"$A_1$")
ax.set_title(fr"(a) $A_1$ vs total stellar mass ($N={len(mstar_sample)}$)")
ax.set_ylim(0, 0.7)
ax.text(0.03, 0.97,
        fr"$N = {len(mstar_sample)}$""\n"
        fr"$r_P = {rP:+.3f}$ ($p = {pP:.1e}$)""\n"
        fr"$\rho_S = {rS:+.3f}$",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white",
                  alpha=0.9, edgecolor="gray"))
ax.legend(loc="lower left", frameon=False)

# ---- Panel (b): stellar mass distribution ----
ax = axes[1]
ax.hist(lMs, bins=np.arange(8.0, 11.6, 0.2),
        color="steelblue", edgecolor="black", linewidth=0.5, alpha=0.85)
ax.axvline(np.median(lMs), color="crimson", linestyle="--",
           linewidth=1.6, label=f"median = {np.median(lMs):.2f}")
ax.set_xlabel(r"$\log(M_\star / M_\odot)$")
ax.set_ylabel("Number of galaxies")
ax.set_title(fr"(b) Stellar mass distribution in stellar-mass subsample")
ax.legend(loc="upper left", frameon=False)
ax.text(0.03, 0.72,
        f"Range: [{lMs.min():.2f}, {lMs.max():.2f}]\n"
        f"Mean = {lMs.mean():.2f}\n"
        f"Std  = {lMs.std():.2f}",
        transform=ax.transAxes, fontsize=10, va="top",
        bbox=dict(boxstyle="round", facecolor="white",
                  alpha=0.9, edgecolor="gray"))

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(FIGS / f"fig7.{ext}")
plt.close()
print(f"Figure 7 saved -> {FIGS/'fig7.pdf'}")

# Save CSV for TNG comparison
with open(FIGS.parent / "data" / "stellar_mass_subsample.csv", "w") as f:
    f.write("norm_name,logMstar,TType,A1,A2\n")
    for g in sorted(mstar_sample, key=lambda x: x["logMstar"]):
        a2 = f"{g['A2']:.4f}" if g["A2"] is not None else ""
        tt = f"{g['TType']:.1f}" if g["TType"] is not None else ""
        f.write(f"{g['norm']},{g['logMstar']:.3f},{tt},{g['A1']:.4f},{a2}\n")
print(f"Stellar-mass subsample CSV saved -> {FIGS.parent/'data'/'stellar_mass_subsample.csv'}")
