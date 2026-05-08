import numpy as np
import pytest

from src.eeg_generator import compute_target_psd, generate_eeg_signal, validate_signal_psd
from src.schemas import AperiodicParams, PeriodicPeak


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_aperiodic():
    return AperiodicParams(offset=1.0, exponent=1.5)


@pytest.fixture
def alpha_peak():
    return PeriodicPeak(center_frequency=10.0, power=0.8, bandwidth=2.0)


@pytest.fixture
def multi_peaks():
    return [
        PeriodicPeak(center_frequency=10.0, power=0.8, bandwidth=2.0),
        PeriodicPeak(center_frequency=22.0, power=0.4, bandwidth=1.5),
    ]


# ---------------------------------------------------------------------------
# compute_target_psd
# ---------------------------------------------------------------------------

class TestComputeTargetPsd:
    def test_shape_matches_freqs(self, simple_aperiodic):
        freqs = np.arange(0, 129)
        psd = compute_target_psd(freqs, simple_aperiodic, [])
        assert psd.shape == freqs.shape

    def test_dc_is_zero(self, simple_aperiodic):
        freqs = np.fft.rfftfreq(512, d=1.0 / 256.0)
        psd = compute_target_psd(freqs, simple_aperiodic, [])
        assert psd[0] == 0.0

    def test_decreases_with_frequency(self, simple_aperiodic):
        freqs = np.arange(1, 50, dtype=float)
        psd = compute_target_psd(freqs, simple_aperiodic, [])
        assert np.all(np.diff(psd) < 0)

    def test_steeper_exponent_decays_faster(self):
        freqs = np.arange(1, 50, dtype=float)
        psd_low = compute_target_psd(freqs, AperiodicParams(1.0, 1.0), [])
        psd_high = compute_target_psd(freqs, AperiodicParams(1.0, 2.0), [])
        ratio = psd_low[-1] / psd_low[0]
        ratio_steep = psd_high[-1] / psd_high[0]
        assert ratio_steep < ratio

    def test_peak_adds_power(self, simple_aperiodic, alpha_peak):
        freqs = np.arange(1, 50, dtype=float)
        psd_no_peak = compute_target_psd(freqs, simple_aperiodic, [])
        psd_with_peak = compute_target_psd(freqs, simple_aperiodic, [alpha_peak])
        idx_10 = np.argmin(np.abs(freqs - 10.0))
        assert psd_with_peak[idx_10] > psd_no_peak[idx_10]

    def test_knee_model(self):
        freqs = np.arange(1, 50, dtype=float)
        ap = AperiodicParams(offset=1.0, exponent=2.0, knee=5.0)
        psd = compute_target_psd(freqs, ap, [])
        assert psd[0] > 0
        assert psd[-1] < psd[0]


# ---------------------------------------------------------------------------
# generate_eeg_signal
# ---------------------------------------------------------------------------

class TestGenerateEegSignal:
    def test_output_shape(self, simple_aperiodic):
        sig = generate_eeg_signal(simple_aperiodic, sfreq=256, duration=5, n_channels=3)
        assert sig.data.shape == (3, 256 * 5)

    def test_metadata(self, simple_aperiodic, alpha_peak):
        sig = generate_eeg_signal(
            simple_aperiodic, [alpha_peak], sfreq=512, duration=8
        )
        assert sig.sfreq == 512.0
        assert sig.duration == 8.0
        assert sig.ground_truth_aperiodic == simple_aperiodic
        assert sig.ground_truth_peaks == [alpha_peak]

    def test_no_peaks_default(self, simple_aperiodic):
        sig = generate_eeg_signal(simple_aperiodic)
        assert sig.ground_truth_peaks == []

    def test_reproducibility(self, simple_aperiodic):
        s1 = generate_eeg_signal(simple_aperiodic, random_seed=42)
        s2 = generate_eeg_signal(simple_aperiodic, random_seed=42)
        np.testing.assert_array_equal(s1.data, s2.data)

    def test_different_seeds_differ(self, simple_aperiodic):
        s1 = generate_eeg_signal(simple_aperiodic, random_seed=1)
        s2 = generate_eeg_signal(simple_aperiodic, random_seed=2)
        assert not np.allclose(s1.data, s2.data)

    def test_channels_independent(self, simple_aperiodic):
        sig = generate_eeg_signal(simple_aperiodic, n_channels=2, random_seed=0)
        corr = np.corrcoef(sig.data[0], sig.data[1])[0, 1]
        assert abs(corr) < 0.15

    def test_signal_finite(self, simple_aperiodic, alpha_peak):
        sig = generate_eeg_signal(simple_aperiodic, [alpha_peak], random_seed=7)
        assert np.all(np.isfinite(sig.data))


# ---------------------------------------------------------------------------
# validate_signal_psd — the Phase 1 sanity check
# ---------------------------------------------------------------------------

class TestValidateSignalPsd:
    def test_exponent_recovery(self):
        """PSD slope should recover the aperiodic exponent within 0.3."""
        ap = AperiodicParams(offset=1.5, exponent=1.5)
        sig = generate_eeg_signal(ap, sfreq=256, duration=30, random_seed=99)
        result = validate_signal_psd(sig)
        assert result["fitted_exponent"] is not None
        assert abs(result["fitted_exponent"] - 1.5) < 0.3

    @pytest.mark.parametrize("exp", [1.0, 1.5, 2.0])
    def test_exponent_sweep(self, exp):
        """Recover exponent across the plausible EEG range."""
        ap = AperiodicParams(offset=1.5, exponent=exp)
        sig = generate_eeg_signal(ap, sfreq=256, duration=30, random_seed=int(exp * 100))
        result = validate_signal_psd(sig)
        assert result["fitted_exponent"] is not None
        assert abs(result["fitted_exponent"] - exp) < 0.35

    def test_peak_detected(self, alpha_peak):
        """A strong alpha peak should be detected near 10 Hz."""
        ap = AperiodicParams(offset=1.5, exponent=1.5)
        sig = generate_eeg_signal(ap, [alpha_peak], sfreq=256, duration=30, random_seed=42)
        result = validate_signal_psd(sig)
        assert len(result["detected_peak_freqs"]) == 1
        detected = result["detected_peak_freqs"][0]
        assert detected is not None
        assert abs(detected - 10.0) < 2.0

    def test_multiple_peaks_detected(self, multi_peaks):
        ap = AperiodicParams(offset=1.5, exponent=1.5)
        sig = generate_eeg_signal(ap, multi_peaks, sfreq=256, duration=30, random_seed=55)
        result = validate_signal_psd(sig)
        assert len(result["detected_peak_freqs"]) == 2
        for det, exp_f in zip(result["detected_peak_freqs"], [10.0, 22.0]):
            assert det is not None
            assert abs(det - exp_f) < 3.0

    def test_no_peaks_no_detections(self):
        ap = AperiodicParams(offset=1.5, exponent=1.5)
        sig = generate_eeg_signal(ap, sfreq=256, duration=10, random_seed=0)
        result = validate_signal_psd(sig)
        assert result["detected_peak_freqs"] == []
        assert result["expected_peak_freqs"] == []


# ---------------------------------------------------------------------------
# EEG band coverage
# ---------------------------------------------------------------------------

class TestEegBands:
    """Verify we can generate peaks in each standard EEG band."""

    @pytest.mark.parametrize(
        "band,center",
        [
            ("delta", 2.5),
            ("theta", 6.0),
            ("alpha", 10.0),
            ("beta", 20.0),
            ("gamma", 40.0),
        ],
    )
    def test_band_peak(self, band, center):
        ap = AperiodicParams(offset=1.5, exponent=1.5)
        peak = PeriodicPeak(center_frequency=center, power=1.0, bandwidth=1.5)
        sig = generate_eeg_signal(ap, [peak], sfreq=256, duration=20, random_seed=77)
        result = validate_signal_psd(sig, freq_range=(0.5, 50.0))
        det = result["detected_peak_freqs"][0]
        assert det is not None
        assert abs(det - center) < 3.0
