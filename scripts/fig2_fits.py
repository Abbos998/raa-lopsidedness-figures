"""
Figure 2: Analytic distribution fits (log-normal, Rayleigh, exponential)
superposed on observed A1 and A2 histograms.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm, rayleigh, expon, kstest
from _common import arrays, FIGS


def fit_and_plot(ax, data, label, color_data):
    bins = np.arange(0, 1.0, 0.04)
    ax.hist(data, bins=bins, density=True, color=color_data,
            edgecolor="black", linewidth=0.6, alpha=0.65,
            label=fr"Observed ($N={len(data)}$)")

    xplot = np.linspace(0.005, 1.0, 500)

    # Log-normal (MLE)
    lnA = np.log(data)
    mu, sigma = lnA.mean(), lnA.std()
    pdf_ln = lognorm.pdf(xplot, sigma, 0, np.exp(mu))
    ks_ln = kstest(data, 'lognorm', args=(sigma, 0, np.exp(mu))).statistic
    ax.plot(xplot, pdf_ln, "g--", linewidth=2,
            label=fr"Log-normal (KS $D={ks_ln:.3f}$)")

    # Rayleigh
    sigma_R = np.sqrt(0.5 * (data**2).mean())
    pdf_R = rayleigh.pdf(xplot, 0, sigma_R)
    ks_R = kstest(data, 'rayleigh', args=(0, sigma_R)).statistic
    ax.plot(xplot, pdf_R, "r-", linewidth=2,
            label=fr"Rayleigh (KS $D={ks_R:.3f}$)")

    # Exponential
    lam = 1.0 / data.mean()
    pdf_E = expon.pdf(xplot, 0, 1/lam)
    ks_E = kstest(data, 'expon', args=(0, 1/lam)).statistic
    ax.plot(xplot, pdf_E, "m:", linewidth=2,
            label=fr"Exponential (KS $D={ks_E:.3f}$)")

    ax.set_xlabel(label)
    ax.set_ylabel("Probability density")
    ax.set_xlim(0, 1.0)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    return mu, sigma, sigma_R, lam


arr = arrays()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

print("=== A1 fits ===")
mu1, s1, sR1, l1 = fit_and_plot(axes[0], arr["A1"], r"$A_1$", "steelblue")
axes[0].set_title(fr"(a) $A_1$ distribution with fits")
print(f"  log-normal: mu={mu1:.3f}, sigma={s1:.3f}")

print("\n=== A2 fits ===")
mu2, s2, sR2, l2 = fit_and_plot(axes[1], arr["A2"], r"$A_2$", "darkorange")
axes[1].set_title(fr"(b) $A_2$ distribution with fits")
print(f"  log-normal: mu={mu2:.3f}, sigma={s2:.3f}")

plt.tight_layout()
for ext in ("pdf", "png"):
    plt.savefig(FIGS / f"fig2.{ext}")
plt.close()
print(f"\nFigure 2 saved -> {FIGS/'fig2.pdf'}")
