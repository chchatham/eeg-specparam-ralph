"""Final regression test suite for spectral - time-domain equivalence.

Runs both pipelines across a grid of synthetic signals with known ground truth
and asserts agreement within defined tolerances.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.schemas import AperiodicParams, PeriodicPeak
from src.eeg_generator import generate_eeg_signal
from src.spectral_specparam import fit_spectral_specparam
from src.time_domain_wrapper import fit_time_domain
from src.comparison import compare_results, tost_equivalence


EXPONENTS = [1.0, 1.25, 1.5, 1.75, 2.0]
PEAK_CONFIGS = [
    [],
    [PeriodicPeak(center_frequency=10.0, power=0.8, bandwidth=2.0)],
    [
        PeriodicPeak(center_frequency=10.0, power=0.8, bandwidth=2.0),
        PeriodicPeak(center_frequency=25.0, power=0.6, bandwidth=1.5),
    ],
]
SFREQ = 256.0
DURATION = 30.0
OFFSET = 1.5
BASE_SEED = 100

EXP_TOLERANCE = 0.35
OFFSET_TOLERANCE = 0.5
PEAK_CENTER_TOLERANCE = 4.0


def _generate_case(exponent, peaks, seed):
    aperiodic = AperiodicParams(offset=OFFSET, exponent=exponent)
    signal = generate_eeg_signal(
        aperiodic, peaks, sfreq=SFREQ, duration=DURATION, random_seed=seed,
    )
    spectral = fit_spectral_specparam(signal)
    td = fit_time_domain(signal, method="psd")
    comp = compare_results(spectral, td)
    return spectral, td, comp


@pytest.fixture(scope="module")
def all_cases():
    cache = {}
    for exp in EXPONENTS:
        for pi, peaks in enumerate(PEAK_CONFIGS):
            seed = BASE_SEED + int(exp * 100) + pi
            cache[(exp, pi)] = _generate_case(exp, peaks, seed)
    return cache


class TestRegressionEquivalence:
    @pytest.mark.parametrize("exponent", EXPONENTS)
    @pytest.mark.parametrize("peak_idx", range(len(PEAK_CONFIGS)),
                             ids=["0peaks", "1peak", "2peaks"])
    def test_convergence(self, exponent, peak_idx, all_cases):
        _, td, _ = all_cases[(exponent, peak_idx)]
        assert td.converged, f"TD failed to converge for exp={exponent}, peaks={peak_idx}"

    @pytest.mark.parametrize("exponent", EXPONENTS)
    @pytest.mark.parametrize("peak_idx", range(len(PEAK_CONFIGS)),
                             ids=["0peaks", "1peak", "2peaks"])
    def test_exponent_agreement(self, exponent, peak_idx, all_cases):
        spectral, td, comp = all_cases[(exponent, peak_idx)]
        assert abs(comp.exponent_diff) < EXP_TOLERANCE, (
            f"Exponent diff {comp.exponent_diff:.4f} exceeds {EXP_TOLERANCE} "
            f"(spectral={spectral.aperiodic.exponent:.4f}, td={td.aperiodic.exponent:.4f})"
        )

    @pytest.mark.parametrize("exponent", EXPONENTS)
    @pytest.mark.parametrize("peak_idx", range(len(PEAK_CONFIGS)),
                             ids=["0peaks", "1peak", "2peaks"])
    def test_offset_agreement(self, exponent, peak_idx, all_cases):
        _, _, comp = all_cases[(exponent, peak_idx)]
        assert abs(comp.offset_diff) < OFFSET_TOLERANCE, (
            f"Offset diff {comp.offset_diff:.4f} exceeds {OFFSET_TOLERANCE}"
        )

    @pytest.mark.parametrize("exponent", EXPONENTS)
    @pytest.mark.parametrize("peak_idx", [1, 2], ids=["1peak", "2peaks"])
    def test_peak_center_agreement(self, exponent, peak_idx, all_cases):
        _, _, comp = all_cases[(exponent, peak_idx)]
        for i, cd in enumerate(comp.peak_center_diffs):
            assert abs(cd) < PEAK_CENTER_TOLERANCE, (
                f"Peak {i} center diff {cd:.3f} Hz exceeds {PEAK_CENTER_TOLERANCE} Hz"
            )

    @pytest.mark.parametrize("exponent", EXPONENTS)
    def test_r_squared_acceptable(self, exponent, all_cases):
        _, td, _ = all_cases[(exponent, 1)]
        assert td.r_squared is not None and td.r_squared > 0.85, (
            f"TD r_squared {td.r_squared} below 0.85 for exp={exponent}"
        )


class TestBatchEquivalence:
    def test_convergence_rate(self, all_cases):
        converged = sum(1 for (_, td, _) in all_cases.values() if td.converged)
        rate = converged / len(all_cases)
        assert rate >= 0.95, f"Convergence rate {rate:.1%} below 95%"

    def test_tost_exponent(self, all_cases):
        diffs = np.array([comp.exponent_diff for _, _, comp in all_cases.values()])
        result = tost_equivalence(diffs, bound=0.3)
        assert result["equivalent"], (
            f"TOST exponent failed: mean_diff={result['mean_diff']:.4f}, "
            f"p={max(result['p_upper'], result['p_lower']):.4f}"
        )

    def test_tost_offset(self, all_cases):
        diffs = np.array([comp.offset_diff for _, _, comp in all_cases.values()])
        result = tost_equivalence(diffs, bound=0.4)
        assert result["equivalent"], (
            f"TOST offset failed: mean_diff={result['mean_diff']:.4f}, "
            f"p={max(result['p_upper'], result['p_lower']):.4f}"
        )

    def test_exponent_rmse(self, all_cases):
        diffs = np.array([comp.exponent_diff for _, _, comp in all_cases.values()])
        rmse = float(np.sqrt(np.mean(diffs**2)))
        assert rmse < 0.2, f"Exponent RMSE {rmse:.4f} exceeds 0.2"

    def test_offset_rmse(self, all_cases):
        diffs = np.array([comp.offset_diff for _, _, comp in all_cases.values()])
        rmse = float(np.sqrt(np.mean(diffs**2)))
        assert rmse < 0.35, f"Offset RMSE {rmse:.4f} exceeds 0.35"


# ---------------------------------------------------------------------------
# Phase 8E: ACF equivalence sweep
# ---------------------------------------------------------------------------

ACF_EXPONENTS = [1.0, 1.25, 1.5, 1.75, 2.0]
ACF_PEAK_CONFIGS = [
    [],
    [PeriodicPeak(center_frequency=10.0, power=0.5, bandwidth=2.0)],
    [
        PeriodicPeak(center_frequency=10.0, power=0.5, bandwidth=2.0),
        PeriodicPeak(center_frequency=20.0, power=0.3, bandwidth=1.5),
    ],
]


@pytest.fixture(scope="module")
def acf_cases():
    cache = {}
    for exp in ACF_EXPONENTS:
        for pi, peaks in enumerate(ACF_PEAK_CONFIGS):
            seed = 200 + int(exp * 100) + pi
            aperiodic = AperiodicParams(offset=0.0, exponent=exp)
            signal = generate_eeg_signal(
                aperiodic, peaks, sfreq=SFREQ, duration=DURATION, random_seed=seed,
            )
            td_acf = fit_time_domain(signal, method="acf")
            spectral = fit_spectral_specparam(signal)
            cache[(exp, pi)] = (spectral, td_acf, exp)
    return cache


class TestACFEquivalence:
    @pytest.mark.parametrize("exponent", ACF_EXPONENTS)
    @pytest.mark.parametrize("peak_idx", range(len(ACF_PEAK_CONFIGS)),
                             ids=["0peaks", "1peak", "2peaks"])
    def test_acf_convergence(self, acf_cases, exponent, peak_idx):
        _, td, _ = acf_cases[(exponent, peak_idx)]
        assert td.converged

    @pytest.mark.parametrize("exponent", ACF_EXPONENTS)
    @pytest.mark.parametrize("peak_idx", range(len(ACF_PEAK_CONFIGS)),
                             ids=["0peaks", "1peak", "2peaks"])
    def test_acf_exponent_vs_ground_truth(self, acf_cases, exponent, peak_idx):
        _, td, true_exp = acf_cases[(exponent, peak_idx)]
        assert td.converged
        assert abs(td.aperiodic.exponent - true_exp) < 0.15, (
            f"ACF exp={td.aperiodic.exponent:.3f} vs truth={true_exp}"
        )

    @pytest.mark.parametrize("exponent", ACF_EXPONENTS)
    @pytest.mark.parametrize("peak_idx", range(len(ACF_PEAK_CONFIGS)),
                             ids=["0peaks", "1peak", "2peaks"])
    def test_acf_vs_spectral(self, acf_cases, exponent, peak_idx):
        spec, td, _ = acf_cases[(exponent, peak_idx)]
        if td.converged and spec.converged:
            assert abs(td.aperiodic.exponent - spec.aperiodic.exponent) < 0.15

    def test_acf_exponent_rmse(self, acf_cases):
        errors = []
        for (exp, _), (_, td, true_exp) in acf_cases.items():
            if td.converged:
                errors.append(td.aperiodic.exponent - true_exp)
        rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))
        assert rmse < 0.10, f"ACF exponent RMSE {rmse:.4f} exceeds 0.10"

    def test_acf_convergence_rate(self, acf_cases):
        n = len(acf_cases)
        converged = sum(1 for _, td, _ in acf_cases.values() if td.converged)
        assert converged / n >= 0.95

    def test_acf_diagnostics_present(self, acf_cases):
        for _, td, _ in acf_cases.values():
            if td.converged:
                assert td.diagnostics is not None
                assert td.diagnostics.fit_domain == "acf"
