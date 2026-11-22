"""Statistics (MSO conventions):
  * unit of analysis = test instance (gap averaged over its 3 seeds), n = 100 paired
    observations per (dataset, budget) cell;
  * Wilcoxon signed-rank (paired, two-sided) of every method vs the reference,
    Holm correction across opponents within a cell;
  * Vargha-Delaney A12 (probability that the reference has a lower gap);
  * Friedman test over all 21 cells + average ranks + Holm post-hoc vs reference
    (Demsar 2006);
  * anytime AUC (Eq. 44) at B = 128.

Usage:  uv run python -m metascale_co.evaluation.stats
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ..config import RES, read_json
from .. import methods as M

ALPHA = 0.05


def a12(x, y):
    """P(X < Y) + 0.5 P(X = Y)  (X = reference; lower gap is better)."""
    x, y = np.asarray(x), np.asarray(y)
    lt = (x[:, None] < y[None, :]).mean()
    eq = (x[:, None] == y[None, :]).mean()
    return lt + 0.5 * eq


def a12_label(a):
    d = abs(a - 0.5)
    return "negligible" if d < 0.06 else "small" if d < 0.14 else "medium" if d < 0.21 else "large"


def holm(pvals):
    p = np.asarray(pvals)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def per_instance(df):
    return df.groupby(["dataset", "budget", "algorithm", "instance"]).gap.mean().reset_index()


def cell_tests(pi, ref, algos):
    rows = []
    for (ds, B), g in pi.groupby(["dataset", "budget"]):
        w = g.pivot(index="instance", columns="algorithm", values="gap")
        opp = [a for a in algos if a != ref]
        ps, recs = [], []
        for a in opp:
            diff = w[a] - w[ref]
            p = stats.wilcoxon(w[ref], w[a]).pvalue if np.any(diff != 0) else 1.0
            ps.append(p)
            recs.append(dict(dataset=ds, budget=B, algorithm=a, mean=w[a].mean(), std=w[a].std(),
                             median=w[a].median(), p_raw=p, A12=a12(w[ref], w[a])))
        for r, padj in zip(recs, holm(ps)):
            r["p_holm"] = padj
            better = r["mean"] < w[ref].mean()
            r["sign"] = ("+" if better else "-") if padj < ALPHA else "≈"   # opponent vs ref
            r["A12_label"] = a12_label(r["A12"])
            rows.append(r)
        rows.append(dict(dataset=ds, budget=B, algorithm=ref, mean=w[ref].mean(), std=w[ref].std(),
                         median=w[ref].median(), p_raw=np.nan, p_holm=np.nan, A12=np.nan,
                         sign="ref", A12_label=""))
    return pd.DataFrame(rows)


def friedman(pi, ref, algos):
    cell_means = pi.groupby(["dataset", "budget", "algorithm"]).gap.mean().unstack()[algos]
    ranks = cell_means.rank(axis=1)
    stat, p = stats.friedmanchisquare(*[cell_means[a] for a in algos])
    k, n = len(algos), len(cell_means)
    se = np.sqrt(k * (k + 1) / (6.0 * n))
    avg = ranks.mean()
    z = {a: (avg[a] - avg[ref]) / se for a in algos if a != ref}
    pz = {a: 2 * stats.norm.sf(abs(v)) for a, v in z.items()}
    padj = dict(zip(pz, holm(list(pz.values()))))
    out = pd.DataFrame({"avg_rank": avg})
    out["p_holm_vs_ref"] = pd.Series(padj)
    return out.sort_values("avg_rank"), stat, p, n


def latex_table(ct, ref, algos, budgets, datasets, path):
    cols = [(d, b) for d in datasets for b in budgets]
    lines = [r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{{len(budgets)}}}{{c}}{{{d}}}" for d in datasets)
             + r" \\",
             "Method & " + " & ".join(f"$B{{=}}{b}$" for _, b in cols) + r" \\", r"\midrule"]
    best = {c: ct[(ct.dataset == c[0]) & (ct.budget == c[1]) &
                  ct.algorithm.isin([a for a in algos if a != "Oracle-1step"])]["mean"].min()
            for c in cols}
    for a in algos:
        cells = []
        for c in cols:
            r = ct[(ct.dataset == c[0]) & (ct.budget == c[1]) & (ct.algorithm == a)].iloc[0]
            v = f"{r['mean']:.2f}"
            if np.isclose(r["mean"], best[c]):
                v = rf"\textbf{{{v}}}"
            s = "" if r["sign"] == "ref" else {"+": "$^{+}$", "-": "$^{-}$", "≈": r"$^{\approx}$"}[r["sign"]]
            cells.append(v + s)
        name = a + (" (ref)" if a == ref else "") + (r" $\dagger$" if a == "Oracle-1step" else "")
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(lines))


def main():
    tuned = read_json("tuned.json")
    ref = tuned["reference"]
    df = pd.read_csv(RES / "results.csv")
    main_df = df[~df.algorithm.str.endswith("+STOP")]
    pi = per_instance(main_df)
    algos = M.ALL
    ct = cell_tests(pi, ref, algos)
    ct.to_csv(RES / "stats_cells.csv", index=False)

    deploy = [a for a in algos if a != "Oracle-1step"]
    fr, stat, p, n = friedman(pi, ref, deploy)
    fr.to_csv(RES / "friedman_ranks.csv")
    datasets = list(dict.fromkeys(df.dataset))
    bud = sorted(df.budget.unique())
    sel = sorted({int(bud[len(bud) // 3]), int(bud[(2 * len(bud)) // 3]), int(bud[-1])})
    latex_table(ct, ref, algos, sel, datasets, RES / "table_main.tex")

    print(f"Reference: {ref}.  Friedman over {n} cells x {len(deploy)} methods: "
          f"chi2={stat:.2f}, p={p:.2e}")
    print(fr.round(4).to_string())
    print("\nWin/tie/loss of each opponent vs reference over 21 cells "
          "(+ = opponent significantly better, - = significantly worse):")
    wtl = ct[ct.sign != "ref"].groupby("algorithm").sign.value_counts().unstack(fill_value=0)
    print(wtl.reindex(columns=["+", "≈", "-"], fill_value=0).to_string())

    # anytime AUC at B=128 (Eq. 44): mean of the gap curve over the compute grid
    z = np.load(RES / "anytime.npz")
    auc = [dict(dataset=k.split("|")[0], algorithm=k.split("|")[1], auc_mean=v.mean(1).mean(),
                auc_std=v.mean(1).std()) for k, v in z.items()]
    auc = pd.DataFrame(auc)
    auc.to_csv(RES / "anytime_auc.csv", index=False)
    print("\nAnytime AUC (mean gap over 0..128 C0; lower = better):")
    print(auc.pivot(index="algorithm", columns="dataset", values="auc_mean").round(3).to_string())

    # learned stopping summary
    st = df[df.algorithm.str.endswith("+STOP")]
    s = st.groupby(["dataset", "tau"]).agg(gap=("gap", "mean"), used=("used_units", "mean")).round(3)
    s.to_csv(RES / "stopping.csv")
    print(f"\nLearned stopping (B={int(df.budget.max())}):")
    print(s.to_string())

    # controller overhead
    ov = main_df[main_df.budget == main_df.budget.max()].groupby(["dataset", "algorithm"]).wall_ms.mean().unstack(0)
    ov.to_csv(RES / "wallclock_Bmax.csv")


if __name__ == "__main__":
    main()
