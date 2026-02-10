from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Optional plotting deps (only needed if plot=True)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # keep importable even in non-plot environments
    plt = None
    sns = None


def _format_p(p: float) -> str:
    """Pretty p-value formatting consistent across functions."""
    if p is None or np.isnan(p):
        return "nan"
    return f"{p:.3f}" if float(p) >= 0.0001 else "< 0.0001"


def correlate(
    df: pd.DataFrame,
    x: str,
    y: str,
    how: str = "pearson",
    plot: bool = True,
    **kwargs: Any,
) -> Tuple[float, float, int]:
    """
    Perform bivariate correlation between x and y using Pearson or Spearman.
    Drops missing values listwise.

    Parameters
    ----------
    df : pd.DataFrame
    x, y : str
        Column names
    how : {"pearson", "spearman"}
    plot : bool
        If True, make a regression plot (requires seaborn + matplotlib).
    **kwargs :
        Passed to seaborn.regplot.

    Returns
    -------
    r : float
    p : float
    n : int
    """
    corr_df = df[[x, y]].dropna()
    n = len(corr_df)

    if n < 3:
        raise ValueError(f"Not enough complete cases to correlate: n={n} (need at least 3).")

    # scipy returns nan (and emits warnings) when one input is constant.
    if corr_df[x].nunique(dropna=True) < 2 or corr_df[y].nunique(dropna=True) < 2:
        raise ValueError("Correlation is undefined when one variable is constant after dropping NA.")

    how = how.lower().strip()
    if how == "pearson":
        r, p = stats.pearsonr(corr_df[x], corr_df[y])

        if plot:
            if sns is None or plt is None:
                raise ImportError("Plotting requires seaborn and matplotlib.")
            sns.regplot(data=corr_df, x=x, y=y, **kwargs)
            plt.show()

    elif how == "spearman":
        r, p = stats.spearmanr(corr_df[x], corr_df[y])

        if plot:
            if sns is None or plt is None:
                raise ImportError("Plotting requires seaborn and matplotlib.")
            ranked = corr_df.rank()
            sns.regplot(data=ranked, x=x, y=y, **kwargs)
            plt.xlabel(f"{x} (rank)")
            plt.ylabel(f"{y} (rank)")

            plt.show()
    else:
        raise ValueError('"how" must be either "pearson" or "spearman".')

    print(f"r = {r:.3f}\np = {_format_p(p)}\nn = {n}")
    return float(r), float(p), int(n)


def compare_groups(
    data: pd.DataFrame,
    grouping_var: str,
    dependent_var: str,
    group_order: Optional[Tuple[Any, Any]] = None,
    plot: bool = True,
    equal_var_alpha: float = 0.05,
    parametric: bool = True,
) -> Dict[str, Any]:
    """
    Compare two independent groups.

    If parametric=True, runs Levene's test + independent samples t-test,
    reports CI (if available), and Cohen's d.

    If parametric=False, runs Mann–Whitney U (a.k.a. Wilcoxon rank-sum for
    independent samples) and returns t-test-specific outputs as nan.

    Parameters
    ----------
    data : pd.DataFrame
    grouping_var : str
        Grouping column with exactly two groups (after dropping NA).
    dependent_var : str
        Outcome column.
    group_order : tuple, optional
        Explicit order like (control_value, treatment_value).
        If None, inferred from sorted unique values.
    plot : bool
        If True, show a boxplot (requires seaborn + matplotlib).
    equal_var_alpha : float
        Threshold for Levene test; p > alpha -> assume equal variances.
    parametric : bool
        If True, use independent samples t-test. If False, use Mann–Whitney U.

    Returns
    -------
    results : dict
    """
    df = data[[grouping_var, dependent_var]].dropna()

    unique_groups = list(df[grouping_var].unique())
    if len(unique_groups) != 2:
        raise ValueError(
            f"{grouping_var} must have exactly 2 groups after dropping NA. Found: {unique_groups}"
        )

    if group_order is None:
        # stable ordering
        try:
            ordered_groups = list(np.sort(np.asarray(unique_groups)))
        except Exception:
            ordered_groups = unique_groups
        g1, g2 = ordered_groups[0], ordered_groups[1]
    else:
        g1, g2 = group_order
        if g1 not in unique_groups or g2 not in unique_groups:
            raise ValueError(
                f"group_order must contain the two observed group values {unique_groups}. Got: {group_order}"
            )

    group1 = df.loc[df[grouping_var] == g1, dependent_var].dropna()
    group2 = df.loc[df[grouping_var] == g2, dependent_var].dropna()

    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        raise ValueError(f"Need at least 2 observations per group. Got n1={n1}, n2={n2}")

    # Descriptives (always computed)
    mean_diff = float(group1.mean() - group2.mean())
    var1 = float(group1.var(ddof=1))
    var2 = float(group2.var(ddof=1))

    # Default outputs
    levene_stat, levene_p = np.nan, np.nan
    equal_var: Optional[bool] = None
    t_stat, p_value = np.nan, np.nan
    lower_bound, higher_bound = np.nan, np.nan
    cohen_d = np.nan
    u_stat = np.nan
    test_name = "ttest_ind" if parametric else "mannwhitneyu"

    if parametric:
        # Levene's test for equal variances
        levene_stat, levene_p = stats.levene(group1, group2, center="median")
        equal_var = bool(levene_p > equal_var_alpha)

        # Independent samples t-test
        t_result = stats.ttest_ind(group1, group2, equal_var=equal_var)
        t_stat, p_value = float(t_result.statistic), float(t_result.pvalue)

        # 95% CI of mean difference (if scipy provides it)
        if hasattr(t_result, "confidence_interval"):
            ci = t_result.confidence_interval()
            lower_bound, higher_bound = float(ci.low), float(ci.high)

        # Cohen's d (pooled SD)
        pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        if pooled_sd != 0 and not np.isnan(pooled_sd):
            cohen_d = float((group1.mean() - group2.mean()) / pooled_sd)
    else:
        # Mann–Whitney U (independent samples). Keep two-sided by default.
        try:
            u_res = stats.mannwhitneyu(group1, group2, alternative="two-sided")
        except TypeError:
            # Older SciPy without alternative= parameter
            u_res = stats.mannwhitneyu(group1, group2)
        u_stat, p_value = float(u_res.statistic), float(u_res.pvalue)

    results = {
        "group1_value": g1,
        "group2_value": g2,
        "n1": n1,
        "n2": n2,
        "levene_stat": float(levene_stat),
        "levene_p": float(levene_p),
        "equal_var": equal_var,
        "group1_variance": var1,
        "group2_variance": var2,
        "mean_difference": mean_diff,
        "mean_difference_95CI": (lower_bound, higher_bound),
        "t_stat": t_stat,
        "u_stat": u_stat,
        "p_value": p_value,
        "cohen_d": cohen_d,
        "test": test_name,
    }

    if parametric:
        print(f"Levene's test: statistic = {levene_stat:.3f}, p-value = {_format_p(float(levene_p))}")
        print("Equal variances assumed." if equal_var else "Equal variances NOT assumed.")
    print(f"\nGroup 1 ({g1}) N = {n1}, mean = {group1.mean():.3f}")
    print(f"Group 2 ({g2}) N = {n2}, mean = {group2.mean():.3f}")
    print(f"\nMean difference (g1-g2) = {mean_diff:.3f}")
    if parametric:
        print(f"Mean difference 95% CI = [{lower_bound:.3f} - {higher_bound:.3f}]")
        print(f"t = {t_stat:.3f}, p = {_format_p(p_value)}")
        print(f"Cohen's d = {cohen_d:.3f}")
    else:
        print(f"U = {u_stat:.3f}, p = {_format_p(p_value)}")

    if plot:
        if sns is None or plt is None:
            raise ImportError("Plotting requires seaborn and matplotlib.")
        plt.figure(figsize=(4, 6))
        ax = sns.boxplot(
            data=df,
            x=grouping_var,
            y=dependent_var,
            hue=grouping_var,
            dodge=False,
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        
        plt.xlabel(grouping_var)
        plt.ylabel(dependent_var)
        plt.show()

    return results


def paired_ttest(
    df: pd.DataFrame,
    x: str,
    y: str,
    plot: bool = True,
    parametric: bool = True,
    **kwargs: Any,
) -> Tuple[float, float, int, float]:
    """
    Paired-samples comparison between x and y.
    Drops missing values listwise.

    If parametric=True, runs paired t-test (scipy.stats.ttest_rel).
    If parametric=False, runs paired Wilcoxon signed-rank test (scipy.stats.wilcoxon).

    Returns
    -------
    stat : float
    p : float
    n : int
    cohens_dz : float
        Cohen's dz = mean(diff) / sd(diff)
    """
    test_df = df[[x, y]].dropna()
    n = len(test_df)
    if n < 2:
        raise ValueError(f"Not enough pairs: n={n} (need at least 2).")

    diffs = test_df[x] - test_df[y]

    if parametric:
        stat, p = stats.ttest_rel(test_df[x], test_df[y])
        stat, p = float(stat), float(p)
    else:
        if np.allclose(diffs.to_numpy(), 0):
            raise ValueError("Wilcoxon test is undefined when all paired differences are 0.")
        w_res = stats.wilcoxon(test_df[x], test_df[y])
        stat, p = float(w_res.statistic), float(w_res.pvalue)

    mean_diff = float(diffs.mean())
    if parametric:
        # 95% CI of mean difference (manual)
        se = float(stats.sem(diffs))
        t_crit = float(stats.t.ppf(0.975, df=n - 1))
        lower_bound = mean_diff - t_crit * se
        higher_bound = mean_diff + t_crit * se
    else:
        lower_bound, higher_bound = np.nan, np.nan

    sd_diff = float(diffs.std(ddof=1))
    cohens_dz = np.nan if (not parametric or sd_diff == 0) else float(mean_diff / sd_diff)

    if plot:
        if sns is None or plt is None:
            raise ImportError("Plotting requires seaborn and matplotlib.")
        plot_df = test_df.melt(var_name="Condition", value_name="Value")
        plt.figure(figsize=(4, 6))
        ax = sns.boxplot(
            data=plot_df,
            x="Condition",
            y="Value",
            hue="Condition",
            **kwargs,
        )
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        plt.show()

    stat_label = "t" if parametric else "W"
    msg = (
        f"{stat_label} = {stat:.3f}\n"
        f"p = {_format_p(p)}\n"
        f"n = {n}\n"
        f"Mean difference = {mean_diff:.3f}"
    )
    if parametric:
        msg += (
            f"\nMean difference 95% CI = [{lower_bound:.3f} - {higher_bound:.3f}]\n"
            f"Cohen's dz = {cohens_dz:.3f}"
        )
    print(msg)

    return stat, p, int(n), cohens_dz
