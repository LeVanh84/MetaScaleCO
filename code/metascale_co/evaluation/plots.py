"""Publication figures (matplotlib -> vector PDF + PNG preview).

Colour rules: one fixed colour per method across every figure (validated
categorical palette, adjacent-pair CVD dE >= 8), solid hairline grid, markers as
secondary encoding, legends on every multi-series figure.

Usage:  uv run python -m metascale_co.evaluation.plots
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..config import FIG, RES, read_json

# ------------------------------------------------------------------ style
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

COLOR = {"MS-Myopic": SLOT[0], "MS-UCB-epi": SLOT[1], "Fixed-tuned": SLOT[2],
         "Fixed-light": SLOT[3], "UCB1-AOS": SLOT[4], "Stagnation": SLOT[5],
         "MS-Myopic+STOP": SLOT[6], "MS-Batch (pooled)": SLOT[7],
         "Oracle-1step": INK}
MARKER = {"MS-Myopic": "o", "MS-UCB-epi": "D", "Fixed-tuned": "s", "Fixed-light": "v",
          "UCB1-AOS": "^", "Stagnation": "P", "MS-Myopic+STOP": "X", "Oracle-1step": "",
          "MS-Batch (pooled)": "o"}
LABEL = {"MS-Myopic": "MetaScale-Myopic", "MS-UCB-epi": "MetaScale-UCB-epi (epistemic σ)",
         "MS-UCB": "MetaScale-UCB (aleatoric σ, Eq. 27)", "MS-DiffOnly": "MetaScale, no history",
         "MS-2op": "MetaScale, {light, strong} only", "Fixed-tuned": "Fixed op., tuned per budget",
         "Fixed-light": "Fixed light op. (GenSCO-like)", "UCB1-AOS": "UCB1 operator selection",
         "Stagnation": "Stagnation-triggered kick", "Random": "Random operator",
         "Oracle-1step": "One-step oracle (upper bound)", "MS-Myopic+STOP": "MetaScale + learned STOP"}
MAIN = ["Fixed-light", "Stagnation", "UCB1-AOS", "Fixed-tuned", "MS-UCB-epi", "MS-Myopic",
        "Oracle-1step"]
DS_LABEL = {"tsp100-uniform": "TSP-100 uniform (in-distribution)",
            "tsp200-uniform": "TSP-200 uniform (size shift)",
            "tsp100-clustered": "TSP-100 clustered (geometry shift)"}
SHORT = {"tsp100-uniform": "ID", "tsp200-uniform": "size", "tsp100-clustered": "geom."}

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
    "axes.titlecolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "axes.facecolor": SURFACE, "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "lines.linewidth": 1.4,
})


def nice_log(ax, axis="y"):
    """Log axis with plain-number ticks at 1-2-5 steps."""
    loc = matplotlib.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0))
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(loc); a.set_major_formatter(fmt)
    a.set_minor_locator(matplotlib.ticker.NullLocator())


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"  {FIG.name}/{name}.pdf (+ .png)")


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(means, [2.5, 97.5])


def _log2_budget_axis(ax, budgets):
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.minorticks_off()


def _line(ax, x, y, name, lo=None, hi=None, **kw):
    c = COLOR.get(name, INK)
    if name == "Oracle-1step":
        ax.plot(x, y, color=INK, lw=1.0, ls=(0, (1.5, 1.5)), label=LABEL[name], zorder=2)
        return
    ax.plot(x, y, color=c, marker=MARKER.get(name, "o"), ms=3.6, mec=SURFACE, mew=0.6,
            label=LABEL.get(name, name), zorder=3, **kw)
    if lo is not None:
        ax.fill_between(x, lo, hi, color=c, alpha=0.13, lw=0, zorder=1)


# ------------------------------------------------------------------ figures
def fig_scaling(df, datasets, budgets):
    pi = df.groupby(["dataset", "budget", "algorithm", "instance"]).gap.mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45), sharey=False)
    for ax, ds in zip(axes, datasets):
        for name in MAIN:
            m, lo, hi = [], [], []
            for B in budgets:
                v = pi[(pi.dataset == ds) & (pi.budget == B) & (pi.algorithm == name)].gap.values
                m.append(v.mean()); a, b = boot_ci(v); lo.append(a); hi.append(b)
            _line(ax, budgets, m, name, lo, hi)
        _log2_budget_axis(ax, budgets)
        ax.set_yscale("log"); nice_log(ax)
        ax.set_title(DS_LABEL[ds], loc="left")
        ax.set_xlabel("Compute budget $B$ (units of $C_0$)")
    axes[0].set_ylabel("Optimality gap to LKH (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.13), handlelength=2.2,
               columnspacing=1.4)
    fig.tight_layout()
    save(fig, "fig1_quality_compute_scaling")


def fig_compute_and_wallclock(df, datasets, budgets, ref):
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.4))
    main = df[~df.algorithm.str.endswith("+STOP")]
    st = df[df.algorithm.str.endswith("+STOP")]
    for j, ds in enumerate(datasets):
        ax = axes[0, j]
        for name in ["Fixed-tuned", ref]:
            g = main[(main.dataset == ds) & (main.algorithm == name)].groupby("budget")
            _line(ax, g.used_units.mean().values, g.gap.mean().values, name)
        s = st[st.dataset == ds].groupby("tau").agg(gap=("gap", "mean"), used=("used_units", "mean"))
        s = s.sort_values("used")
        ax.plot(s.used, s.gap, color=COLOR["MS-Myopic+STOP"], marker="X", ms=4, mec=SURFACE,
                mew=0.6, lw=1.0, label=LABEL["MS-Myopic+STOP"] + r" ($\tau$ sweep)", zorder=4)
        ax.set_xscale("log", base=2); ax.set_yscale("log"); nice_log(ax)
        ax.set_xticks(budgets); ax.set_xticklabels([str(b) for b in budgets]); ax.minorticks_off()
        ax.set_title(DS_LABEL[ds], loc="left")
        ax.set_xlabel("Compute actually used ($C_0$ units)")
        ax = axes[1, j]
        for name in ["Fixed-tuned", "UCB1-AOS", ref]:
            g = main[(main.dataset == ds) & (main.algorithm == name)].groupby("budget")
            _line(ax, g.wall_ms.mean().values, g.gap.mean().values, name)
        ax.set_xscale("log"); ax.set_yscale("log"); nice_log(ax); nice_log(ax, "x")
        ax.set_xlabel("Wall-clock per run (ms)")
    axes[0, 0].set_ylabel("Gap (%)"); axes[1, 0].set_ylabel("Gap (%)")
    h0, l0 = axes[0, 0].get_legend_handles_labels()
    h1, l1 = axes[1, 0].get_legend_handles_labels()
    seen, H, L = set(), [], []
    for h, l in zip(h0 + h1, l0 + l1):
        if l not in seen:
            seen.add(l); H.append(h); L.append(l)
    fig.legend(H, L, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06), columnspacing=1.4)
    axes[0, 0].text(-0.28, 1.08, "a", transform=axes[0, 0].transAxes, fontsize=10, weight="bold")
    axes[1, 0].text(-0.28, 1.08, "b", transform=axes[1, 0].transAxes, fontsize=10, weight="bold")
    fig.tight_layout()
    save(fig, "fig3_compute_frontier_and_wallclock")


def fig_calibration(cal, datasets, ref):
    c = cal[cal.algorithm == ref]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.4))
    lim = 0
    for ax, ds in zip(axes, datasets):
        d = c[c.dataset == ds]
        q = pd.qcut(d.mu.rank(method="first"), 15, labels=False)
        g = d.groupby(q).agg(mu=("mu", "mean"), gain=("gain", "mean"), sd=("gain", "std"),
                             n=("gain", "size"))
        se = 1.96 * g.sd / np.sqrt(g.n)
        ax.errorbar(g.mu, g.gain, yerr=se, fmt="o", ms=3.6, color=SLOT[0], mec=SURFACE, mew=0.6,
                    elinewidth=0.8, capsize=0)
        hi = max(g.mu.max(), g.gain.max()) * 1.1
        lim = max(lim, hi)
        rho = stats.spearmanr(d.mu, d.gain).statistic
        ece = np.average(np.abs(g.mu - g.gain), weights=g.n)
        ax.text(0.97, 0.04, f"Spearman ρ = {rho:.2f}\nmean abs. calib. error = {ece:.3f} pp",
                transform=ax.transAxes, va="bottom", ha="right", color=INK2, fontsize=6.8)
        ax.set_title(DS_LABEL[ds], loc="left")
        ax.set_xlabel("Predicted gain μ (%), 15 quantile bins")
    for ax in axes:
        ax.plot([0, lim], [0, lim], color=MUTED, lw=0.8, zorder=0)
        ax.set_xscale("symlog", linthresh=0.05); ax.set_yscale("symlog", linthresh=0.05)
        ax.set_xlim(-0.01, lim); ax.set_ylim(-0.01, lim)
    axes[0].set_ylabel("Realised one-step gain G (%)")
    fig.tight_layout()
    save(fig, "fig5_voc_calibration")


def fig_batch(datasets, ref):
    b = pd.read_csv(RES / "batch_results.csv")
    Bb = int(b.budget.max())
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    names = ["Fixed-tuned (uniform split)", f"{ref} (uniform split)", "MS-Batch (pooled)"]
    cols = [COLOR["Fixed-tuned"], COLOR[ref], COLOR["MS-Batch (pooled)"]]
    mks = ["s", "o", "o"]
    cells = [(ds, B) for ds in datasets for B in sorted(b.budget.unique())]
    for k, (n, c, m) in enumerate(zip(names, cols, mks)):
        for i, (ds, B) in enumerate(cells):
            v = b[(b.dataset == ds) & (b.budget == B) & (b.method == n)].groupby("instance").gap.mean()
            lo, hi = boot_ci(v.values)
            x = i + (k - 1) * 0.22
            ax.plot([x, x], [lo, hi], color=c, lw=1.0)
            ax.plot(x, v.mean(), marker=m, color=c, ms=4, mec=SURFACE, mew=0.6, ls="",
                    label=n.replace(ref, SHORTNAME[ref]) if i == 0 else None)
    ax.set_xticks(range(len(cells)))
    ax.set_xticklabels([f"{SHORT[d]}\nB={B}" for d, B in cells])
    ax.set_ylabel("Mean gap over the batch (%)")
    ax.set_title("(a) pooled vs uniform budget, batches of 10", loc="left")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, columnspacing=1.0)
    ax = axes[1]
    d = b[(b.method == "MS-Batch (pooled)") & (b.dataset == datasets[0]) & (b.budget == Bb)]
    ax.scatter(d.gap0, d.spent, s=9, color=COLOR["MS-Batch (pooled)"], edgecolor=SURFACE,
               linewidth=0.4, zorder=3)
    ax.axhline(Bb, color=MUTED, lw=0.8, zorder=1, label=f"uniform share ({Bb} units)")
    ax.legend(loc="upper left")
    ax.set_xlabel("Initial gap of the instance (%)")
    ax.set_ylabel("Compute received ($C_0$ units)")
    ax.set_title(f"(b) who gets the pooled budget (ID, B={Bb})", loc="left")
    fig.tight_layout()
    save(fig, "fig7_batch_allocation")


SHORTNAME = {"MS-Myopic": "MetaScale-Myopic", "MS-UCB-epi": "MetaScale-UCB-epi",
             "MS-UCB": "MetaScale-UCB"}


def main():
    tuned = read_json("tuned.json")
    ref = tuned["reference"]
    LABEL[ref] = LABEL[ref] + " (ref.)"
    df = pd.read_csv(RES / "results.csv")
    cal = pd.read_csv(RES / "calibration.csv")
    datasets = [d for d in ["tsp100-uniform", "tsp200-uniform", "tsp100-clustered"]
                if d in set(df.dataset)]
    budgets = sorted(df.budget.unique())
    main_df = df[~df.algorithm.str.endswith("+STOP")]
    fig_scaling(main_df, datasets, budgets)
    fig_compute_and_wallclock(df, datasets, budgets, ref)
    fig_calibration(cal, datasets, ref)
    fig_batch(datasets, ref)


if __name__ == "__main__":
    main()
