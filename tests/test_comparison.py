import numpy as np
import pytest

from src.comparison import (
    compare_results,
    compute_agreement_metrics,
    tost_equivalence,
)
from src.eeg_generator import generate_eeg_signal
from src.schemas import AperiodicParams, PeriodicPeak
from src.spectral_specparam import fit_spectral_specparam
from src.time_domain_wrapper import fit_time_domain


def _make_signal(aperiodic, peaks=None, duration=30, seed=42):
    return generate_eeg_signal(
        aperiodic, peaks, sfreq=256, duration=duration, random_seed=seed
    )


class TestCompareResults:
    def test_basic_comparison(self):
        peak = PeriodicPeak(center_frequency=10.0, power=0.8, bandwidth=2.0)
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=1.5), [peak])
        spec = fit_spectral_specparam(sig)
        td = fit_time_domain(sig)
        comp = compare_results(spec, td)
        assert abs(comp.exponent_diff) < 0.5
        assert isinstance(comp.peak_count_match, bool)

    def test_peak_matching(self):
        peaks = [
            PeriodicPeak(center_frequency=10.0, power=1.0, bandwidth=2.0),
            PeriodicPeak(center_frequency=25.0, power=0.8, bandwidth=1.5),
        ]
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=1.5), peaks, seed=33)
        spec = fit_spectral_specparam(sig)
        td = fit_time_domain(sig)
        comp = compare_results(spec, td)
        assert len(comp.peak_center_diffs) > 0


class TestAgreementMetrics:
    def test_sweep_metrics(self):
        comparisons = []
        for exp in [1.0, 1.5, 2.0]:
            peak = PeriodicPeak(center_frequency=10.0, power=0.8, bandwidth=2.0)
            sig = _make_signal(
                AperiodicParams(offset=1.5, exponent=exp), [peak],
                seed=int(exp * 100),
            )
            spec = fit_spectral_specparam(sig)
            td = fit_time_domain(sig)
            comparisons.append(compare_results(spec, td))

        metrics = compute_agreement_metrics(comparisons)
        assert metrics["n_comparisons"] == 3
        assert metrics["exponent_rmse"] < 0.5
        assert "offset_rmse" in metrics


class TestTOST:
    def test_equivalent_data(self):
        diffs = np.array([0.01, -0.02, 0.03, 0.0, -0.01, 0.02, -0.03, 0.01])
        result = tost_equivalence(diffs, bound=0.1)
        assert result["equivalent"] is True

    def test_non_equivalent_data(self):
        diffs = np.array([0.5, 0.6, 0.4, 0.55, 0.45])
        result = tost_equivalence(diffs, bound=0.1)
        assert result["equivalent"] is False

    def test_exponent_equivalence_sweep(self):
        """Run TOST on exponent diffs across the parameter sweep."""
        diffs = []
        for exp in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
            peak = PeriodicPeak(center_frequency=10.0, power=0.8, bandwidth=2.0)
            sig = _make_signal(
                AperiodicParams(offset=1.5, exponent=exp), [peak],
                seed=int(exp * 100),
            )
            spec = fit_spectral_specparam(sig)
            td = fit_time_domain(sig)
            diffs.append(spec.aperiodic.exponent - td.aperiodic.exponent)

        result = tost_equivalence(np.array(diffs), bound=0.2)
        print(f"\nTOST exponent: mean={result['mean_diff']:.4f} "
              f"p_upper={result['p_upper']:.4f} p_lower={result['p_lower']:.4f} "
              f"equiv={result['equivalent']}")


class TestRSquared:
    def test_td_r_squared_present(self):
        peak = PeriodicPeak(center_frequency=10.0, power=0.8, bandwidth=2.0)
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=1.5), [peak])
        td = fit_time_domain(sig)
        assert td.r_squared is not None
        assert td.r_squared > 0.5

    def test_r_squared_high_for_clean_signal(self):
        peak = PeriodicPeak(center_frequency=10.0, power=1.0, bandwidth=2.0)
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=2.0), [peak])
        td = fit_time_domain(sig)
        assert td.r_squared is not None
        assert td.r_squared > 0.9
