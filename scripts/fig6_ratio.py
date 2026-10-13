"""
Figure 6: A2/A1 mode ratio distribution + variation with T-type.
"""
import numpy as np
import matplotlib.pyplot as plt
from _common import arrays, FIGS, load_table2_dedup

arr = arrays()
A1 = arr["A1_j"]
A2 = arr["A2_j"]

# Threshold to keep ratio numerically stable
sel = A1 > 0.02
ratio = A2[sel] / A1[sel]
A1_sel = A1[sel]

print(f"A2/A1 ratio: N = {len(ratio)}")
print(f"  median = {np.median(ratio):.3f}")
print(f"  mean   = {ratio.mean():.3f}")
print(f"  Fraction A2>A1: {(ratio>1).sum()/len(ratio)*100:.1f}%")

# Get T-type for joint sample
rows = load_table2_dedup()
joint_T = [(r["A1"], r["A2"], r["TType"]) for r in rows
           if r["A1"] is not None and r["A2"] is not None and r["TType"] is not None]
jt = np.array(joint_T)
mask = (jt[:,0] <= 1.0) & (jt[:,1] <= 1.0) & (jt[:,0] > 0.02)
A1_T = jt[mask, 0]
A2_T = jt[mask, 1]
TT_T = jt[mask, 2]
ratio_T = A2_T / A1_T

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

# ---- Panel (a): overall ratio distribution ----
ax = axes[0]
log_ratio = np.log10(ratio)
bins = np.linspace(-1.5, 1.5, 40)
ax.hist(log_ratio, bins=bins, color="purple", edgecolor="black",
        linewidth=0.6, alpha=0.75)
ax.axvline(0, color="black", linestyle="-", linewidth=1, alpha=0.6,
           label=r"$A_2 = A_1$")
ax.axvline(np.log10(np.median(ratio)), color="crimson", linestyle="--",
           linewidth=1.6, label=fr"median = {np.median(ratio):.2f}")
ax.set_xlabel(r"$\log_{10}(A_2 / A_1)$")
ax.set_ylabel("Number of galaxies")
ax.set_title(fr"(a) Distribution of $A_2 / A_1$ ratio ($N={len(ratio)}$)")
ax.legend(loc="upper right", frameon=False)
ax.set_xlim(-1.5, 1.5)

# ---- Panel (b): ratio vs T-type ----
ax = axes[1]
bin_edges = [(-3, 0, "E/S0"), (1, 2, "S0/a-Sa"), (3, 4, "Sb-Sbc"),
             (5, 6, "Sc-Scd"), (7, 8, "Sd-Sdm"), (9, 10, "Sm/Im")]
positions = []
data_list = []
labels = []
Ns = []
for i, (lo, hi, lbl) in enumerate(bin_edges):
    sel_T = (TT_T >= lo) & (TT_T <= hi)
    if sel_T.sum() >= 3:
        positions.append(i)
        data_list.append(np.log10(ratio_T[sel_T]))
        labels.append(lbl)
        Ns.append(sel_T.sum())

bp = ax.boxplot(data_list, positions=positions, widths=0.6,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="black", linewidth=1.5),
                whis=1.5)
colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(data_list)))
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.75)
for p, n in zip(positions, Ns):
    ax.text(p, 0.95, f"$N={n}$", ha="center", fontsize=9,
            transform=ax.get_xaxis_transform())
ax.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.5)
ax.set_xticks(positions)
ax.set_xticklabels(labels, rotation=0, fontsize=10)
ax.set_xlabel("Morphological $T$-type")
ax.set_ylabel(r"$\log_{10}(A_2 / A_1)$")
ax.set_title(fr"(b) Mode ratio by morphology")
ax.set_ylim(-1.5, 1.5)

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(FIGS / f"fig6.{ext}")
plt.close()
print(f"Figure 6 saved -> {FIGS/'fig6.pdf'}")
