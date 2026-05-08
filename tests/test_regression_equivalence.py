"""Final regression test suite for spectral ↔ time-domain equivalence.

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
    td = fit_time_domain(signal)
    comp = compare_results(spectral, td)
    return spectral, td, comp


class TestRegressionEquivalence:
    @pytest.mark.parametrize("exponent", EXPONENTS)
    @pytest.mark.parametrize("peak_idx", range(len(PEAK_CONFIGS)),
                             ids=["0peaks", "1peak", "2peaks"])
    def test_convergence(self, exponent, peak_idx):
        _, td, _ = _generate_case(exponent, PEAK_CONFIGS[peak_idx],
                                  seed=BASE_SEED + int(exponent * 100) + peak_idx)
        assert td.converged, f"TD failed to converge for exp={exponent}, peaks={peak_idx}"

    @pytest.mark.parametrize("exponent", EXPONENTS)
    @pytest.mark.parametrize("peak_idx", range(len(PEAK_CONFIGS)),
                             ids=["0peaks", "1peak", "2peaks"])
    def test_exponent_agreement(self, exponent, peak_idx):
        spectral, td, comp = _generate_case(
            exponent, PEAK_CONFIGS[peak_idx],
            seed=BASE_SEED + int(exponent * 100) + peak_idx,
        )
        assert abs(comp.exponent_diff) < EXP_TOLERANCE, (
            f"Exponent diff {comp.exponent_diff:.4f} exceeds {EXP_TOLERANCE} "
            f"(spectral={spectral.aperiodic.exponent:.4f}, td={td.aperiodic.exponent:.4f})"
        )

    @pytest.mark.parametrize("exponent", EXPONENTS)
    @pytest.mark.parametrize("peak_idx", range(len(PEAK_CONFIGS)),
                             ids=["0peaks", "1peak", "2peaks"])
    def test_offset_agreement(self, exponent, peak_idx):
        _, _, comp = _generate_case(
            exponent, PEAK_CONFIGS[peak_idx],
            seed=BASE_SEED + int(exponent * 100) + peak_idx,
        )
        assert abs(comp.offset_diff) < OFFSET_TOLERANCE, (
            f"Offset diff {comp.offset_diff:.4f} exceeds {OFFSET_TOLERANCE}"
        )

    @pytest.mark.parametrize("exponent", EXPONENTS)
    def test_peak_center_agreement(self, exponent):
        peaks = PEAK_CONFIGS[1]
        _, _, comp = _generate_case(
            exponent, peaks,
            seed=BASE_SEED + int(exponent * 100) + 1,
        )
        for i, cd in enumerate(comp.peak_center_diffs):
            assert abs(cd) < PEAK_CENTER_TOLERANCE, (
                f"Peak {i} center diff {cd:.3f} Hz exceeds {PEAK_CENTER_TOLERANCE} Hz"
            )

    @pytest.mark.parametrize("exponent", EXPONENTS)
    def test_r_squared_acceptable(self, exponent):
        peaks = PEAK_CONFIGS[1]
        _, td, _ = _generate_case(
            exponent, peaks,
            seed=BASE_SEED + int(exponent * 100) + 1,
        )
        assert td.r_squared is not None and td.r_squared > 0.85, (
            f"TD r_squared {td.r_squared} below 0.85 for exp={exponent}"
        )


class TestBatchEquivalence:
    @pytest.fixture(scope="class")
    def sweep_results(self):
        results = []
        for exp in EXPONENTS:
            for pi, peaks in enumerate(PEAK_CONFIGS):
                seed = BASE_SEED + int(exp * 100) + pi
                spectral, td, comp = _generate_case(exp, peaks, seed)
                results.append({
                    "exponent": exp, "peak_idx": pi,
                    "spectral": spectral, "td": td, "comp": comp,
                })
        return results

    def test_convergence_rate(self, sweep_results):
        converged = sum(1 for r in sweep_results if r["td"].converged)
        rate = converged / len(sweep_results)
        assert rate >= 0.95, f"Convergence rate {rate:.1%} below 95%"

    def test_tost_exponent(self, sweep_results):
        diffs = np.array([r["comp"].exponent_diff for r in sweep_results])
        result = tost_equivalence(diffs, bound=0.3)
        assert result["equivalent"], (
            f"TOST exponent failed: mean_diff={result['mean_diff']:.4f}, "
            f"p={max(result['p_upper'], result['p_lower']):.4f}"
        )

    def test_tost_offset(self, sweep_results):
        diffs = np.array([r["comp"].offset_diff for r in sweep_results])
        result = tost_equivalence(diffs, bound=0.4)
        assert result["equivalent"], (
            f"TOST offset failed: mean_diff={result['mean_diff']:.4f}, "
            f"p={max(result['p_upper'], result['p_lower']):.4f}"
        )

    def test_exponent_rmse(self, sweep_results):
        diffs = np.array([r["comp"].exponent_diff for r in sweep_results])
        rmse = float(np.sqrt(np.mean(diffs**2)))
        assert rmse < 0.2, f"Exponent RMSE {rmse:.4f} exceeds 0.2"

    def test_offset_rmse(self, sweep_results):
        diffs = np.array([r["comp"].offset_diff for r in sweep_results])
        rmse = float(np.sqrt(np.mean(diffs**2)))
        assert rmse < 0.35, f"Offset RMSE {rmse:.4f} exceeds 0.35"
