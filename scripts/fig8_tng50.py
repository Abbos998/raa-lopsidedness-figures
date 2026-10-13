"""
Figure 8: Observations vs TNG50-1 comparison.
Three panels: (a) A1 distributions, (b) median A1 vs M*, (c) cumulative.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, mannwhitneyu, pearsonr, spearmanr
from _common import load_table1_mstar, load_table2_dedup, load_tng50, FIGS

# ---- Load observational stellar-mass subsample (deduplicated) ----
t1 = load_table1_mstar()
t2 = load_table2_dedup()

obs_rows = []
for r in t2:
    if r["A1"] is None or r["A1"] > 1.0:
        continue
    if r["norm"] in t1:
        lm = t1[r["norm"]]["logMstar"]
        if lm is not None:
            obs_rows.append({"logMstar": lm, "A1": r["A1"]})
obs = pd.DataFrame(obs_rows)
print(f"Obs stellar-mass subsample: N = {len(obs)}")
print(f"  logM* range: [{obs['logMstar'].min():.2f}, {obs['logMstar'].max():.2f}]")

# ---- Load TNG50 ----
tng_full = load_tng50()
print(f"TNG50 full: N = {len(tng_full)}")

# Mass-range restricted TNG50 (same M* range as obs)
m_lo = obs["logMstar"].min()
m_hi = obs["logMstar"].max()
tng_matched = tng_full[(tng_full["logMstar"] >= m_lo) & (tng_full["logMstar"] <= m_hi)].reset_index(drop=True)
print(f"TNG50 mass-range restricted: N = {len(tng_matched)}")

# ---- Statistical tests ----
ks = ks_2samp(obs["A1"], tng_matched["A1"])
mw = mannwhitneyu(obs["A1"], tng_matched["A1"], alternative="two-sided")
print(f"\nKS: D = {ks.statistic:.3f}, p = {ks.pvalue:.2e}")
print(f"MW: p = {mw.pvalue:.2e}")

ratio = obs["A1"].median() / tng_matched["A1"].median()
print(f"Median ratio obs/TNG50 = {ratio:.2f}x")

for name, df in [("Obs", obs), ("TNG50 matched", tng_matched)]:
    rp, pp = pearsonr(df["logMstar"], df["A1"])
    rs, ps = spearmanr(df["logMstar"], df["A1"])
    print(f"  {name:15s}: Pearson r={rp:+.3f}, Spearman rho={rs:+.3f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

# ---- Panel (a): A1 distributions ----
ax = axes[0]
bins = np.linspace(0, 0.6, 25)
ax.hist(obs["A1"], bins=bins, density=True, alpha=0.55,
        color="steelblue", edgecolor="black", linewidth=0.4,
        label="Observations")
ax.hist(tng_matched["A1"], bins=bins, density=True, alpha=0.55,
        color="crimson", edgecolor="black", linewidth=0.4,
        label="TNG50-1 mass-range restricted")
ax.axvline(obs["A1"].median(), color="steelblue", linestyle="--", linewidth=1.4)
ax.axvline(tng_matched["A1"].median(), color="crimson", linestyle="--", linewidth=1.4)
ax.set_xlabel(r"$A_1$")
ax.set_ylabel("Probability density")
ax.legend(loc="upper right", frameon=False, fontsize=9)
ax.set_xlim(0, 0.6)
ax.text(0.60, 0.60,
        fr"KS $D = {ks.statistic:.3f}$""\n"
        fr"$p = {ks.pvalue:.2e}$""\n"
        fr"MW $p = {mw.pvalue:.2e}$",
        transform=ax.transAxes, va="center", ha="left", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white",
                  alpha=0.9, edgecolor="gray"))

# ---- Panel (b): median A1 vs M* ----
ax = axes[1]
mbins = np.arange(9.4, 11.2, 0.3)
xc = 0.5 * (mbins[:-1] + mbins[1:])

def binned_median(df, mbins):
    med, lo, hi, N = [], [], [], []
    for i in range(len(mbins)-1):
        sel = (df["logMstar"] >= mbins[i]) & (df["logMstar"] < mbins[i+1])
        if sel.sum() >= 3:
            med.append(df["A1"][sel].median())
            lo.append(np.percentile(df["A1"][sel], 16))
            hi.append(np.percentile(df["A1"][sel], 84))
            N.append(sel.sum())
        else:
            med.append(np.nan); lo.append(np.nan)
            hi.append(np.nan); N.append(0)
    return np.array(med), np.array(lo), np.array(hi), N

obs_med, obs_lo, obs_hi, obs_N = binned_median(obs, mbins)
tng_med, tng_lo, tng_hi, tng_N = binned_median(tng_matched, mbins)

ax.fill_between(xc, obs_lo, obs_hi, alpha=0.20, color="steelblue")
ax.plot(xc, obs_med, "o-", color="steelblue", markersize=9,
        linewidth=2, label="Observations (VII/296)")
ax.fill_between(xc, tng_lo, tng_hi, alpha=0.20, color="crimson")
ax.plot(xc, tng_med, "s-", color="crimson", markersize=9,
        linewidth=2, label="TNG50-1 (TNGGalaxyLab)")
ax.set_xlabel(r"$\log(M_\star / M_\odot)$")
ax.set_ylabel(r"median $A_1$")
ax.set_xlim(9.4, 11.2)
ax.set_ylim(0, 0.36)
ax.legend(loc="upper right", frameon=False)
ax.grid(True, alpha=0.3, linestyle=":")

# ---- Panel (c): cumulative distributions ----
ax = axes[2]
for arr, col, lbl in [
    (obs["A1"], "steelblue", "Observations"),
    (tng_matched["A1"], "crimson", "TNG50 mass-range restricted"),
    (tng_full["A1"], "purple", "TNG50 full"),
]:
    xs = np.sort(arr.values)
    cdf = np.arange(1, len(arr)+1) / len(arr)
    ax.plot(xs, cdf, color=col, linewidth=1.8, label=lbl)
ax.set_xlabel(r"$A_1$")
ax.set_ylabel("Cumulative fraction")
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.set_xlim(0, 0.6)
ax.set_ylim(0, 1.02)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(FIGS / f"fig8.{ext}")
plt.close()
print(f"\nFigure 8 saved -> {FIGS/'fig8.pdf'}")
