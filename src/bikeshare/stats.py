"""Helpers for the hypothesis-testing notebook: effect sizes, bootstrap, post-hoc and permutation tests."""
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def hedges_g(a, b) -> float:
    """Standardised mean difference (a - b) with the small-sample correction."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    pooled_sd = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / pooled_sd * (1 - 3 / (4 * (na + nb) - 9)))


def cluster_bootstrap_diff(values, in_group, clusters, n_boot=5000, seed=0):
    """Difference in means (group minus the rest) with a 95% CI from resampling whole clusters.

    Hours on the same day move together, so resampling days instead of hours keeps
    that dependence and gives an honest interval.
    """
    df = pd.DataFrame({"v": np.asarray(values, dtype=float), "g": np.asarray(in_group, dtype=bool),
                       "c": np.asarray(clusters)})
    agg = df.groupby(["c", "g"])["v"].agg(["sum", "count"]).unstack(fill_value=0)
    sum_in, n_in = agg[("sum", True)].to_numpy(), agg[("count", True)].to_numpy()
    sum_out, n_out = agg[("sum", False)].to_numpy(), agg[("count", False)].to_numpy()
    picks = np.random.default_rng(seed).integers(0, len(agg), size=(n_boot, len(agg)))
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = sum_in[picks].sum(1) / n_in[picks].sum(1) - sum_out[picks].sum(1) / n_out[picks].sum(1)
    observed = sum_in.sum() / n_in.sum() - sum_out.sum() / n_out.sum()
    return float(observed), np.nanpercentile(boot, [2.5, 97.5])


def cluster_bootstrap_mean(values, clusters, n_boot=5000, seed=0):
    """Mean with a 95% CI from resampling whole clusters (days)."""
    df = pd.DataFrame({"v": np.asarray(values, dtype=float), "c": np.asarray(clusters)})
    agg = df.groupby("c")["v"].agg(["sum", "count"])
    picks = np.random.default_rng(seed).integers(0, len(agg), size=(n_boot, len(agg)))
    boot = agg["sum"].to_numpy()[picks].sum(1) / agg["count"].to_numpy()[picks].sum(1)
    return float(df["v"].mean()), np.percentile(boot, [2.5, 97.5])


def eta_squared(groups) -> float:
    """Share of the variance in the outcome explained by group membership."""
    values = np.concatenate([np.asarray(g, dtype=float) for g in groups])
    grand = values.mean()
    between = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    return float(between / ((values - grand) ** 2).sum())


def games_howell(groups: dict) -> pd.DataFrame:
    """Pairwise mean comparisons that do not assume equal variances (post-hoc for Welch's ANOVA)."""
    k = len(groups)
    rows = []
    for (name_a, a), (name_b, b) in combinations(groups.items(), 2):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
        t = (a.mean() - b.mean()) / np.sqrt(va + vb)
        df = (va + vb) ** 2 / (va**2 / (len(a) - 1) + vb**2 / (len(b) - 1))
        p_adj = stats.studentized_range.sf(abs(t) * np.sqrt(2), k, df)
        rows.append({"group_a": name_a, "group_b": name_b, "mean_diff": a.mean() - b.mean(),
                     "t": t, "df": df, "p_adj": p_adj})
    return pd.DataFrame(rows)


def total_variation_distance(p, q) -> float:
    """Share of rides that would have to move to another hour to turn profile p into profile q."""
    return float(0.5 * np.abs(np.asarray(p, dtype=float) - np.asarray(q, dtype=float)).sum())


def permutation_test_profiles(profiles, labels, n_perm=10_000, seed=0):
    """Test whether two groups of days have different hourly profiles by shuffling labels across days.

    `profiles` has one row per day (hourly shares that sum to 1) and `labels` holds two group names.
    Returns the observed distance, the p-value and the permutation distribution.
    """
    X = np.asarray(profiles, dtype=float)
    labels = np.asarray(labels)
    in_a = labels == np.unique(labels)[0]
    observed = total_variation_distance(X[in_a].mean(0), X[~in_a].mean(0))
    rng = np.random.default_rng(seed)
    n_a = int(in_a.sum())
    null = np.empty(n_perm)
    for i in range(n_perm):
        order = rng.permutation(len(X))
        null[i] = total_variation_distance(X[order[:n_a]].mean(0), X[order[n_a:]].mean(0))
    p_value = (np.sum(null >= observed) + 1) / (n_perm + 1)
    return observed, float(p_value), null
