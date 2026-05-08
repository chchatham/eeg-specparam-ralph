import numpy as np
import pytest

from src.eeg_generator import generate_eeg_signal
from src.schemas import AperiodicParams, PeriodicPeak
from src.spectral_specparam import fit_spectral_specparam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal(aperiodic, peaks=None, duration=30, seed=42):
    return generate_eeg_signal(
        aperiodic, peaks, sfreq=256, duration=duration, random_seed=seed
    )


# ---------------------------------------------------------------------------
# Basic integration
# ---------------------------------------------------------------------------

class TestFitSpectralSpecparam:
    def test_returns_specparam_result(self):
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=1.5))
        result = fit_spectral_specparam(sig)
        assert result.method == "spectral"
        assert result.converged is True
        assert result.frequency_range == (1.0, 40.0)

    def test_r_squared_present(self):
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=1.5))
        result = fit_spectral_specparam(sig)
        assert result.r_squared is not None
        assert result.r_squared > 0.5

    def test_aperiodic_exponent_recovery(self):
        """Spectral SpecParam should recover the exponent within ~0.3."""
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=1.5))
        result = fit_spectral_specparam(sig)
        assert abs(result.aperiodic.exponent - 1.5) < 0.3

    @pytest.mark.parametrize("exp", [1.0, 1.5, 2.0])
    def test_exponent_sweep(self, exp):
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=exp), seed=int(exp * 100))
        result = fit_spectral_specparam(sig)
        assert abs(result.aperiodic.exponent - exp) < 0.4

    def test_alpha_peak_detected(self):
        """A strong alpha peak should appear in the results near 10 Hz."""
        peak = PeriodicPeak(center_frequency=10.0, power=1.0, bandwidth=2.0)
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=1.5), [peak])
        result = fit_spectral_specparam(sig)
        alpha_peaks = [
            p for p in result.peaks if abs(p.center_frequency - 10.0) < 3.0
        ]
        assert len(alpha_peaks) >= 1

    def test_two_peaks_detected(self):
        peaks = [
            PeriodicPeak(center_frequency=10.0, power=1.0, bandwidth=2.0),
            PeriodicPeak(center_frequency=25.0, power=0.8, bandwidth=1.5),
        ]
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=1.5), peaks)
        result = fit_spectral_specparam(sig)
        near_10 = any(abs(p.center_frequency - 10.0) < 3.0 for p in result.peaks)
        near_25 = any(abs(p.center_frequency - 25.0) < 3.0 for p in result.peaks)
        assert near_10
        assert near_25


# ---------------------------------------------------------------------------
# Knee mode
# ---------------------------------------------------------------------------

class TestKneeMode:
    def test_knee_mode_converges(self):
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=2.0, knee=5.0))
        result = fit_spectral_specparam(sig, aperiodic_mode="knee")
        assert result.converged is True
        assert result.aperiodic.knee is not None

    def test_fixed_mode_no_knee(self):
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=1.5))
        result = fit_spectral_specparam(sig, aperiodic_mode="fixed")
        assert result.aperiodic.knee is None


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_short_signal(self):
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=1.5), duration=2)
        result = fit_spectral_specparam(sig)
        assert result.converged is True

    def test_custom_freq_range(self):
        sig = _make_signal(AperiodicParams(offset=1.5, exponent=1.5))
        result = fit_spectral_specparam(sig, freq_range=(2.0, 30.0))
        assert result.frequency_range == (2.0, 30.0)
        assert result.converged is True

    def test_channel_selection(self):
        sig = generate_eeg_signal(
            AperiodicParams(offset=1.5, exponent=1.5),
            sfreq=256, duration=10, n_channels=3, random_seed=0,
        )
        r0 = fit_spectral_specparam(sig, channel=0)
        r2 = fit_spectral_specparam(sig, channel=2)
        assert r0.converged and r2.converged
        assert r0.aperiodic.exponent != r2.aperiodic.exponent  # different noise draws
