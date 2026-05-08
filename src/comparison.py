from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .schemas import SpecParamResult


@dataclass
class ComparisonResult:
    exponent_diff: float
    offset_diff: float
    peak_count_match: bool
    peak_center_diffs: list[float]
    peak_power_diffs: list[float]
    peak_bw_diffs: list[float]
    spectral_r_squared: float | None
    td_r_squared: float | None


def compare_results(
    spectral: SpecParamResult,
    time_domain: SpecParamResult,
) -> ComparisonResult:
    """Compute parameter-wise differences between spectral and time-domain results."""

    exp_diff = spectral.aperiodic.exponent - time_domain.aperiodic.exponent
    off_diff = spectral.aperiodic.offset - time_domain.aperiodic.offset

    # Match peaks by center frequency (greedy nearest-neighbor)
    spec_peaks = sorted(spectral.peaks, key=lambda p: p.center_frequency)
    td_peaks = sorted(time_domain.peaks, key=lambda p: p.center_frequency)

    center_diffs: list[float] = []
    power_diffs: list[float] = []
    bw_diffs: list[float] = []

    td_used: set[int] = set()
    for sp in spec_peaks:
        best_j = -1
        best_dist = float("inf")
        for j, tp in enumerate(td_peaks):
            if j in td_used:
                continue
            dist = abs(sp.center_frequency - tp.center_frequency)
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j >= 0 and best_dist < 5.0:
            td_used.add(best_j)
            tp = td_peaks[best_j]
            center_diffs.append(sp.center_frequency - tp.center_frequency)
            power_diffs.append(sp.power - tp.power)
            bw_diffs.append(sp.bandwidth - tp.bandwidth)

    return ComparisonResult(
        exponent_diff=exp_diff,
        offset_diff=off_diff,
        peak_count_match=len(spectral.peaks) == len(time_domain.peaks),
        peak_center_diffs=center_diffs,
        peak_power_diffs=power_diffs,
        peak_bw_diffs=bw_diffs,
        spectral_r_squared=spectral.r_squared,
        td_r_squared=time_domain.r_squared,
    )


def compute_agreement_metrics(comparisons: list[ComparisonResult]) -> dict:
    """Compute RMSE, correlation, and Bland-Altman stats across a sweep."""

    if not comparisons:
        return {}

    exp_diffs = np.array([c.exponent_diff for c in comparisons])
    off_diffs = np.array([c.offset_diff for c in comparisons])

    all_center = []
    for c in comparisons:
        all_center.extend(c.peak_center_diffs)
    center_diffs = np.array(all_center) if all_center else np.array([])

    return {
        "exponent_rmse": float(np.sqrt(np.mean(exp_diffs**2))),
        "exponent_bias": float(np.mean(exp_diffs)),
        "offset_rmse": float(np.sqrt(np.mean(off_diffs**2))),
        "offset_bias": float(np.mean(off_diffs)),
        "peak_center_rmse": float(np.sqrt(np.mean(center_diffs**2))) if len(center_diffs) > 0 else None,
        "peak_center_bias": float(np.mean(center_diffs)) if len(center_diffs) > 0 else None,
        "n_comparisons": len(comparisons),
    }


def tost_equivalence(
    diffs: np.ndarray,
    bound: float,
    alpha: float = 0.05,
) -> dict:
    """Two One-Sided Tests for equivalence within [-bound, +bound].

    Returns dict with p-values and whether equivalence is established.
    """
    from scipy.stats import t as t_dist

    n = len(diffs)
    if n < 2:
        return {"equivalent": False, "p_upper": 1.0, "p_lower": 1.0, "n": n}

    mean_d = float(np.mean(diffs))
    se = float(np.std(diffs, ddof=1) / np.sqrt(n))

    if se < 1e-12:
        p_upper = 0.0 if mean_d < bound else 1.0
        p_lower = 0.0 if mean_d > -bound else 1.0
    else:
        t_upper = (mean_d - bound) / se
        t_lower = (mean_d + bound) / se
        p_upper = float(t_dist.cdf(t_upper, df=n - 1))
        p_lower = float(1.0 - t_dist.cdf(t_lower, df=n - 1))

    return {
        "equivalent": max(p_upper, p_lower) < alpha,
        "p_upper": p_upper,
        "p_lower": p_lower,
        "mean_diff": mean_d,
        "bound": bound,
        "n": n,
    }
