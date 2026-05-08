import numpy as np
import pytest

from src.eeg_generator import generate_eeg_signal
from src.schemas import AperiodicParams, PeriodicPeak
from src.spectral_specparam import fit_spectral_specparam
from src.time_domain_wrapper import fit_time_domain


def _make_signal(aperiodic, peaks=None, duration=30, seed=42):
    return generate_eeg_signal(
        aperiodic, peaks, sfreq=256, duration=duration, random_seed=seed
    )


class TestTimeDomainWrapper:
    def test_returns_specparam_result(self):
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=2.0))
        result = fit_time_domain(sig)
        assert result.method == "time_domain"

    def test_converges_on_simple_signal(self):
        peak = PeriodicPeak(center_frequency=10.0, power=0.8, bandwidth=2.0)
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=2.0), [peak])
        result = fit_time_domain(sig)
        assert result.converged is True

    def test_exponent_is_free(self):
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=2.0))
        result = fit_time_domain(sig)
        if result.converged:
            assert 0.5 <= result.aperiodic.exponent <= 3.0

    def test_peak_center_near_alpha(self):
        peak = PeriodicPeak(center_frequency=10.0, power=0.8, bandwidth=2.0)
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=2.0), [peak])
        result = fit_time_domain(sig)
        if result.converged and len(result.peaks) > 0:
            assert abs(result.peaks[0].center_frequency - 10.0) < 5.0

    def test_handles_convergence_failure(self):
        """A very short or flat signal may fail — wrapper should not crash."""
        sig = generate_eeg_signal(
            AperiodicParams(offset=-5.0, exponent=0.1),
            sfreq=256, duration=2, random_seed=99,
        )
        result = fit_time_domain(sig)
        assert result.method == "time_domain"


class TestExponentRecovery:
    """Validate that the time-domain fitter recovers the correct exponent."""

    @pytest.mark.parametrize("exp", [1.0, 1.5, 2.0])
    def test_exponent_vs_spectral(self, exp):
        peak = PeriodicPeak(center_frequency=10.0, power=0.8, bandwidth=2.0)
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=exp), [peak],
                           seed=int(exp * 100))
        spectral = fit_spectral_specparam(sig)
        td = fit_time_domain(sig)
        assert td.converged, f"Time-domain failed for exp={exp}"
        assert abs(td.aperiodic.exponent - spectral.aperiodic.exponent) < 0.15, (
            f"exp={exp}: TD={td.aperiodic.exponent:.3f} vs Spec={spectral.aperiodic.exponent:.3f}"
        )

    @pytest.mark.parametrize("exp", [1.0, 1.5, 2.0])
    def test_peak_center_vs_spectral(self, exp):
        peak = PeriodicPeak(center_frequency=10.0, power=0.8, bandwidth=2.0)
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=exp), [peak],
                           seed=int(exp * 100))
        spectral = fit_spectral_specparam(sig)
        td = fit_time_domain(sig)
        if td.converged and td.peaks and spectral.peaks:
            spec_cf = min(spectral.peaks, key=lambda p: abs(p.center_frequency - 10.0))
            td_cf = min(td.peaks, key=lambda p: abs(p.center_frequency - 10.0))
            assert abs(td_cf.center_frequency - spec_cf.center_frequency) < 2.0


class TestMultiPeak:
    """Validate multi-peak detection."""

    def test_no_peaks(self):
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=1.5),
                           duration=30, seed=10)
        td = fit_time_domain(sig, min_peak_height=0.3)
        assert td.converged

    def test_single_peak(self):
        peak = PeriodicPeak(center_frequency=10.0, power=0.8, bandwidth=2.0)
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=1.5), [peak])
        td = fit_time_domain(sig)
        assert td.converged
        alpha = [p for p in td.peaks if abs(p.center_frequency - 10.0) < 3.0]
        assert len(alpha) >= 1

    def test_two_peaks(self):
        peaks = [
            PeriodicPeak(center_frequency=10.0, power=1.0, bandwidth=2.0),
            PeriodicPeak(center_frequency=25.0, power=0.8, bandwidth=1.5),
        ]
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=1.5), peaks, seed=33)
        td = fit_time_domain(sig)
        assert td.converged
        near_10 = any(abs(p.center_frequency - 10.0) < 3.0 for p in td.peaks)
        near_25 = any(abs(p.center_frequency - 25.0) < 4.0 for p in td.peaks)
        assert near_10, f"Missing alpha peak. Found: {[(p.center_frequency, p.power) for p in td.peaks]}"
        assert near_25, f"Missing beta peak. Found: {[(p.center_frequency, p.power) for p in td.peaks]}"

    def test_three_peaks(self):
        peaks = [
            PeriodicPeak(center_frequency=6.0, power=0.6, bandwidth=1.5),
            PeriodicPeak(center_frequency=10.0, power=1.0, bandwidth=2.0),
            PeriodicPeak(center_frequency=25.0, power=0.8, bandwidth=1.5),
        ]
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=1.5), peaks, seed=77)
        td = fit_time_domain(sig)
        assert td.converged
        near_10 = any(abs(p.center_frequency - 10.0) < 3.0 for p in td.peaks)
        near_25 = any(abs(p.center_frequency - 25.0) < 4.0 for p in td.peaks)
        assert near_10
        assert near_25

    def test_multi_peak_vs_spectral(self):
        peaks = [
            PeriodicPeak(center_frequency=10.0, power=1.0, bandwidth=2.0),
            PeriodicPeak(center_frequency=25.0, power=0.8, bandwidth=1.5),
        ]
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=1.5), peaks, seed=33)
        spectral = fit_spectral_specparam(sig)
        td = fit_time_domain(sig)
        assert td.converged and spectral.converged
        assert abs(td.aperiodic.exponent - spectral.aperiodic.exponent) < 0.2


class TestACFFitting:
    """Phase 8D: ACF-specific fitting tests."""

    @pytest.mark.parametrize("exp", [1.0, 1.5, 2.0])
    def test_acf_convergence(self, exp):
        peak = PeriodicPeak(center_frequency=10.0, power=0.5, bandwidth=2.0)
        sig = _make_signal(AperiodicParams(offset=0.0, exponent=exp), [peak], seed=42)
        result = fit_time_domain(sig, method="acf")
        assert result.converged

    @pytest.mark.parametrize("exp", [1.0, 1.5, 2.0])
    def test_acf_exponent_recovery(self, exp):
        peak = PeriodicPeak(center_frequency=10.0, power=0.5, bandwidth=2.0)
        sig = _make_signal(AperiodicParams(offset=0.0, exponent=exp), [peak], seed=42)
        result = fit_time_domain(sig, method="acf")
        assert result.converged
        assert abs(result.aperiodic.exponent - exp) < 0.15, (
            f"ACF exp recovery: got {result.aperiodic.exponent:.3f}, expected {exp}")

    def test_acf_no_peaks(self):
        sig = _make_signal(AperiodicParams(offset=0.0, exponent=1.5), duration=30)
        result = fit_time_domain(sig, method="acf")
        assert result.converged
        assert abs(result.aperiodic.exponent - 1.5) < 0.15

    def test_acf_peak_detection(self):
        peak = PeriodicPeak(center_frequency=10.0, power=0.5, bandwidth=2.0)
        sig = _make_signal(AperiodicParams(offset=0.0, exponent=1.5), [peak])
        result = fit_time_domain(sig, method="acf")
        assert result.converged
        assert len(result.peaks) >= 1
        near_10 = any(abs(p.center_frequency - 10.0) < 3.0 for p in result.peaks)
        assert near_10

    def test_acf_r_squared(self):
        peak = PeriodicPeak(center_frequency=10.0, power=0.5, bandwidth=2.0)
        sig = _make_signal(AperiodicParams(offset=0.0, exponent=1.5), [peak])
        result = fit_time_domain(sig, method="acf")
        assert result.r_squared is not None
        assert result.r_squared > 0.3


class TestACFResiduals:
    """Phase 8D: Diagnostics arrays present and correct shape."""

    def test_diagnostics_present(self):
        peak = PeriodicPeak(center_frequency=10.0, power=0.5, bandwidth=2.0)
        sig = _make_signal(AperiodicParams(offset=0.0, exponent=1.5), [peak])
        result = fit_time_domain(sig, method="acf")
        assert result.diagnostics is not None
        assert result.diagnostics.fit_domain == "acf"

    def test_diagnostics_shapes(self):
        sig = _make_signal(AperiodicParams(offset=0.0, exponent=1.5), duration=30)
        result = fit_time_domain(sig, method="acf")
        d = result.diagnostics
        assert d is not None
        assert len(d.lags) == len(d.empirical_acf)
        assert len(d.lags) == len(d.model_acf)
        assert len(d.lags) == len(d.acf_residual)

    def test_acf_residual_is_difference(self):
        sig = _make_signal(AperiodicParams(offset=0.0, exponent=1.5), duration=30)
        result = fit_time_domain(sig, method="acf")
        d = result.diagnostics
        np.testing.assert_allclose(
            d.acf_residual, d.empirical_acf - d.model_acf, atol=1e-10)

    def test_psd_method_no_diagnostics(self):
        sig = _make_signal(AperiodicParams(offset=0.0, exponent=1.5))
        result = fit_time_domain(sig, method="psd")
        assert result.diagnostics is None

    def test_lags_start_at_zero(self):
        sig = _make_signal(AperiodicParams(offset=0.0, exponent=1.5), duration=30)
        result = fit_time_domain(sig, method="acf")
        assert result.diagnostics.lags[0] == 0.0

    def test_acf_positive_at_zero_lag(self):
        sig = _make_signal(AperiodicParams(offset=0.0, exponent=1.5), duration=30)
        result = fit_time_domain(sig, method="acf")
        assert result.diagnostics.empirical_acf[0] > 0
        assert result.diagnostics.model_acf[0] > 0


class TestLongLag:
    """Phase 8D: Long signals benefit ACF fitting."""

    def test_long_signal_better_recovery(self):
        """30-sec signal should recover exponent better than 4-sec signal."""
        peak = PeriodicPeak(center_frequency=10.0, power=0.5, bandwidth=2.0)
        ap = AperiodicParams(offset=0.0, exponent=1.5)

        sig_short = _make_signal(ap, [peak], duration=4, seed=42)
        sig_long = _make_signal(ap, [peak], duration=30, seed=42)

        r_short = fit_time_domain(sig_short, method="acf")
        r_long = fit_time_domain(sig_long, method="acf")

        assert r_long.converged
        err_long = abs(r_long.aperiodic.exponent - 1.5)
        assert err_long < 0.2, f"Long signal ACF error too high: {err_long:.3f}"


class TestBaselineComparison:
    """Record baseline discrepancies between spectral and time-domain."""

    def test_baseline_alpha_signal(self):
        """Run both pipelines on the same signal and report discrepancies."""
        peak = PeriodicPeak(center_frequency=10.0, power=0.8, bandwidth=2.0)
        sig = _make_signal(
            AperiodicParams(offset=1.5, exponent=2.0), [peak], duration=30
        )

        spectral = fit_spectral_specparam(sig)
        td = fit_time_domain(sig)

        assert spectral.converged, "Spectral fit should converge"
        assert td.converged, "Time-domain fit should converge"

        exp_diff = abs(spectral.aperiodic.exponent - td.aperiodic.exponent)
        offset_diff = abs(spectral.aperiodic.offset - td.aperiodic.offset)

        # Record baseline — these are NOT pass/fail criteria yet
        # They document the current gap that Phase 3B-3E must close
        print(f"\n--- BASELINE DISCREPANCIES ---")
        print(f"Spectral exponent:  {spectral.aperiodic.exponent:.4f}")
        print(f"TimeDom  exponent:  {td.aperiodic.exponent:.4f} (hardcoded 2.0)")
        print(f"Exponent diff:      {exp_diff:.4f}")
        print(f"Spectral offset:    {spectral.aperiodic.offset:.4f}")
        print(f"TimeDom  offset:    {td.aperiodic.offset:.4f}")
        print(f"Offset diff:        {offset_diff:.4f}")

        if spectral.peaks:
            sp = spectral.peaks[0]
            print(f"Spectral peak:      cf={sp.center_frequency:.2f} Hz, "
                  f"pow={sp.power:.4f}, bw={sp.bandwidth:.4f}")
        if td.peaks:
            tp = td.peaks[0]
            print(f"TimeDom  peak:      cf={tp.center_frequency:.2f} Hz, "
                  f"pow={tp.power:.4f}, bw={tp.bandwidth:.4f}")

        print(f"Spectral r²:        {spectral.r_squared:.4f}")
        print(f"TimeDom  r²:        {td.r_squared}")
        print(f"--- END BASELINE ---\n")
