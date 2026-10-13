"""
Figure 4: A1 and A2 vs morphological T-type (boxplots).
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kendalltau
from _common import arrays, FIGS

arr = arrays()
A1 = arr["A1_T"]
T1 = arr["TType_1"]
A2 = arr["A2_T"]
T2 = arr["TType_2"]

print(f"A1 vs T-type: N = {len(A1)}")
tau1, p1 = kendalltau(T1, A1)
print(f"  Kendall tau = {tau1:+.3f} (p = {p1:.2e})")
print(f"A2 vs T-type: N = {len(A2)}")
tau2, p2 = kendalltau(T2, A2)
print(f"  Kendall tau = {tau2:+.3f} (p = {p2:.2e})")

# T-type bin edges (de Vaucouleurs)
bin_edges = [(-3, 0, "E/S0"), (1, 2, "S0/a-Sa"), (3, 4, "Sb-Sbc"),
             (5, 6, "Sc-Scd"), (7, 8, "Sd-Sdm"), (9, 10, "Sm/Im")]

def bin_data(T, A, edges):
    out = []
    labels = []
    Ns = []
    for lo, hi, lbl in edges:
        sel = (T >= lo) & (T <= hi)
        if sel.sum() > 0:
            out.append(A[sel])
            labels.append(lbl)
            Ns.append(sel.sum())
    return out, labels, Ns


fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

# ---- Panel (a): A1 vs T-type ----
ax = axes[0]
data1, lbls1, Ns1 = bin_data(T1, A1, bin_edges)
positions = np.arange(len(data1))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(data1)))
bp = ax.boxplot(data1, positions=positions, widths=0.6,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="black", linewidth=1.5),
                whis=1.5)
for patch, c in zip(bp['boxes'], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.75)
# Annotate N
for p, n in zip(positions, Ns1):
    ax.text(p, 0.95, f"$N={n}$", ha="center", fontsize=9,
            transform=ax.get_xaxis_transform())
ax.set_xticks(positions)
ax.set_xticklabels(lbls1, rotation=0, fontsize=10)
ax.set_xlabel("Morphological $T$-type")
ax.set_ylabel(r"$A_1$")
ax.set_ylim(0, 1.0)
ax.set_title(fr"(a) $A_1$ vs $T$-type")
ax.text(0.03, 0.94,
        fr"$\tau = {tau1:+.3f}$""\n"fr"$p = 10^{{{int(np.log10(p1))}}}$",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white",
                  alpha=0.9, edgecolor="gray"))

# ---- Panel (b): A2 vs T-type ----
ax = axes[1]
data2, lbls2, Ns2 = bin_data(T2, A2, bin_edges)
positions = np.arange(len(data2))
colors2 = plt.cm.plasma(np.linspace(0.2, 0.9, len(data2)))
bp2 = ax.boxplot(data2, positions=positions, widths=0.6,
                 patch_artist=True, showfliers=False,
                 medianprops=dict(color="black", linewidth=1.5),
                 whis=1.5)
for patch, c in zip(bp2['boxes'], colors2):
    patch.set_facecolor(c)
    patch.set_alpha(0.75)
for p, n in zip(positions, Ns2):
    ax.text(p, 0.95, f"$N={n}$", ha="center", fontsize=9,
            transform=ax.get_xaxis_transform())
ax.set_xticks(positions)
ax.set_xticklabels(lbls2, rotation=0, fontsize=10)
ax.set_xlabel("Morphological $T$-type")
ax.set_ylabel(r"$A_2$")
ax.set_ylim(0, 1.0)
ax.set_title(fr"(b) $A_2$ vs $T$-type")
p2_str = f"{p2:.2f}" if p2 >= 0.001 else fr"10^{{{int(np.log10(p2))}}}"
ax.text(0.03, 0.94,
        fr"$\tau = {tau2:+.3f}$""\n"fr"$p = {p2_str}$",
        transform=ax.transAxes, va="top", fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white",
                  alpha=0.9, edgecolor="gray"))

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(FIGS / f"fig4.{ext}")
plt.close()
print(f"Figure 4 saved -> {FIGS/'fig4.pdf'}")
