"""
Statistics used across the v2 analyses, as thin wrappers over scipy and statsmodels
(October 2026). Percentages in, percentages out where noted; every resampling routine
takes a seed so that results are reproducible.
"""
import numpy as np
from scipy import stats as sps
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

SEED = 0


def wilson(k, n):
    """Wilson 95% interval for k/n, in percent."""
    if n == 0:
        return (float("nan"), float("nan"))
    lo, hi = proportion_confint(k, n, alpha=0.05, method="wilson")
    return 100 * lo, 100 * hi


def mcnemar_exact(b, c):
    """Exact two-sided McNemar p from the discordant counts b and c."""
    if b + c == 0:
        return 1.0
    return float(mcnemar([[0, b], [c, 0]], exact=True).pvalue)


def holm(ps):
    """Holm-adjusted p-values, in the input order."""
    return list(multipletests(ps, method="holm")[1]) if ps else []


def signflip_p(d, n_resamples=20000, seed=SEED):
    """Two-sided paired permutation (sign-flip) p for mean(d) != 0."""
    d = np.asarray(d, float)
    if not len(d) or not d.any():
        return 1.0
    res = sps.permutation_test((d,), np.mean, permutation_type="samples", n_resamples=n_resamples,
                               alternative="two-sided", rng=np.random.default_rng(seed))
    return float(res.pvalue)


def spearman(a, b, n_resamples=20000, seed=SEED):
    """Spearman's rho with a two-sided permutation p-value."""
    rho = sps.spearmanr(a, b).statistic
    res = sps.permutation_test((np.asarray(a, float), np.asarray(b, float)),
                               lambda x, y: sps.spearmanr(x, y).statistic, permutation_type="pairings",
                               n_resamples=n_resamples, alternative="two-sided", rng=np.random.default_rng(seed))
    return float(rho), float(res.pvalue)


def paired_mean_ci(a, b, n_resamples=2000, seed=SEED):
    """Mean of a - b (paired) and its 95% percentile-bootstrap interval."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a):
        return float("nan"), [float("nan"), float("nan")]
    if np.all(a == b):  # no discordant pairs: the bootstrap is degenerate
        return 0.0, [0.0, 0.0]
    res = sps.bootstrap((a, b), lambda x, y, axis=-1: np.mean(x - y, axis=axis), paired=True,
                        n_resamples=n_resamples, method="percentile", rng=np.random.default_rng(seed))
    return float(np.mean(a - b)), [float(res.confidence_interval.low), float(res.confidence_interval.high)]


def mean_ci(d, n_resamples=20000, seed=SEED):
    """Mean of d and its 95% percentile-bootstrap interval."""
    d = np.asarray(d, float)
    res = sps.bootstrap((d,), np.mean, n_resamples=n_resamples, method="percentile", rng=np.random.default_rng(seed))
    return float(d.mean()), [float(res.confidence_interval.low), float(res.confidence_interval.high)]


def cluster_bootstrap(n_clusters, statistic, n_resamples=2000, seed=SEED):
    """Percentile-bootstrap distribution of statistic(cluster_indices) over resampled clusters.

    statistic takes an integer array of cluster indices (with repeats) and returns a scalar or a
    1-D array. Returns the (n_resamples, ...) array of replicates and the 2.5/97.5 percentiles."""
    res = sps.bootstrap((np.arange(n_clusters),), statistic, vectorized=False, n_resamples=n_resamples,
                        method="percentile", rng=np.random.default_rng(seed))
    return np.asarray(res.bootstrap_distribution).T, np.asarray(res.confidence_interval.low), np.asarray(res.confidence_interval.high)


def stratified_bootstrap(strata, statistic, n_resamples=2000, seed=SEED):
    """Like cluster_bootstrap, but resamples indices within each stratum (scipy's bootstrap draws each
    sample independently), so every draw keeps the strata sizes. statistic takes the concatenated index
    array. Returns (replicates, lo, hi)."""
    strata = np.asarray(strata)
    groups = tuple(np.flatnonzero(strata == s) for s in sorted(set(strata.tolist())))
    res = sps.bootstrap(groups, lambda *g: statistic(np.concatenate(g)), vectorized=False, paired=False,
                        n_resamples=n_resamples, method="percentile", rng=np.random.default_rng(seed))
    return np.asarray(res.bootstrap_distribution).T, np.asarray(res.confidence_interval.low), np.asarray(res.confidence_interval.high)
