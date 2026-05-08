"""Tests for ACF math infrastructure (Phase 8A).

Validates: IRFFT scaling (R(0) == total power), AR(1) known ACF,
chi=2 Lorentzian ACF shape, empirical ACF properties, and
model/empirical consistency.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.timedomain_specparam import (
    _aperiodic_model,
    _compute_empirical_acf,
    _compute_model_psd_linear,
    _gaussian_peak,
    _model_acf_via_irfft,
)


# ---------------------------------------------------------------------------
# _compute_model_psd_linear
# ---------------------------------------------------------------------------

class TestComputeModelPsdLinear:
    def test_aperiodic_only(self):
        f = np.linspace(0, 128, 500)
        b, k, chi = 2.0, 1.0, 1.5
        psd = _compute_model_psd_linear(f, b, k, chi, [])
        expected = 10.0 ** _aperiodic_model(f, b, k, chi)
        expected[0] = 0.0
        np.testing.assert_allclose(psd, expected)

    def test_with_peaks(self):
        f = np.linspace(0, 128, 500)
        b, k, chi = 2.0, 1.0, 1.5
        peaks = [(0.5, 10.0, 2.0)]
        psd = _compute_model_psd_linear(f, b, k, chi, peaks)
        log_total = _aperiodic_model(f, b, k, chi) + _gaussian_peak(f, 0.5, 10.0, 2.0)
        expected = 10.0 ** log_total
        expected[0] = 0.0
        np.testing.assert_allclose(psd, expected)

    def test_dc_zeroed(self):
        f = np.array([0.0, 1.0, 2.0])
        psd = _compute_model_psd_linear(f, 2.0, 1.0, 1.5, [])
        assert psd[0] == 0.0

    def test_no_dc_in_grid(self):
        f = np.array([1.0, 2.0, 3.0])
        psd = _compute_model_psd_linear(f, 2.0, 1.0, 1.5, [])
        assert psd[0] > 0.0

    def test_non_negative(self):
        f = np.linspace(0, 128, 1000)
        psd = _compute_model_psd_linear(f, 2.0, 0.5, 2.0, [(0.3, 10, 2)])
        assert np.all(psd >= 0)

    def test_multi_peak(self):
        f = np.linspace(0, 128, 500)
        b, k, chi = 1.0, 0.5, 1.5
        peaks = [(0.4, 10.0, 2.0), (0.2, 20.0, 1.5)]
        psd = _compute_model_psd_linear(f, b, k, chi, peaks)
        log_total = _aperiodic_model(f, b, k, chi)
        for a, c, w in peaks:
            log_total = log_total + _gaussian_peak(f, a, c, w)
        expected = 10.0 ** log_total
        expected[0] = 0.0
        np.testing.assert_allclose(psd, expected)


# ---------------------------------------------------------------------------
# _model_acf_via_irfft
# ---------------------------------------------------------------------------

class TestModelAcfViaIrfft:
    def test_r0_equals_total_power(self):
        """R(0) must equal the one-sided integral of the model PSD."""
        sfreq = 256.0
        n_fft = 8192
        b, k, chi = 2.0, 1.0, 1.5
        peaks = [(0.5, 10.0, 2.0)]

        f_grid = np.arange(n_fft // 2 + 1) * (sfreq / n_fft)
        model_psd = _compute_model_psd_linear(f_grid, b, k, chi, peaks)

        df = sfreq / n_fft
        total_power = df * (model_psd[0] + np.sum(model_psd[1:-1]) + model_psd[-1])

        acf = _model_acf_via_irfft(b, k, chi, peaks, sfreq, n_fft, max_lag_samples=1)
        np.testing.assert_allclose(acf[0], total_power, rtol=1e-10)

    def test_r0_aperiodic_only(self):
        """R(0) check with no peaks."""
        sfreq = 256.0
        n_fft = 8192
        b, k, chi = 1.5, 0.5, 2.0

        f_grid = np.arange(n_fft // 2 + 1) * (sfreq / n_fft)
        model_psd = _compute_model_psd_linear(f_grid, b, k, chi, [])
        df = sfreq / n_fft
        total_power = df * (model_psd[0] + np.sum(model_psd[1:-1]) + model_psd[-1])

        acf = _model_acf_via_irfft(b, k, chi, [], sfreq, n_fft, max_lag_samples=1)
        np.testing.assert_allclose(acf[0], total_power, rtol=1e-10)

    def test_acf_decays_aperiodic(self):
        """Aperiodic-only ACF should decay monotonically."""
        acf = _model_acf_via_irfft(2.0, 1.0, 1.5, [], 256.0, 8192, 200)
        assert acf[0] > acf[10] > acf[50] > acf[100]

    def test_acf_positive_at_zero(self):
        acf = _model_acf_via_irfft(1.0, 0.5, 1.5, [], 256.0, 4096, 10)
        assert acf[0] > 0

    def test_chi2_exponential_decay(self):
        """For chi=2 (Lorentzian PSD), ACF should resemble exponential decay.

        Continuous Lorentzian S(f) = A/(k + f^2) has ACF ~ exp(-2*pi*sqrt(k)*|tau|).
        Discrete version with finite bandwidth deviates, but correlation should be high.
        """
        sfreq = 512.0
        n_fft = 32768
        b, k = 0.0, 1.0
        acf = _model_acf_via_irfft(b, k, 2.0, [], sfreq, n_fft, max_lag_samples=400)

        lags_sec = np.arange(400) / sfreq
        expected = np.exp(-2 * np.pi * np.sqrt(k) * lags_sec)
        expected *= acf[0]

        corr = np.corrcoef(acf[:200], expected[:200])[0, 1]
        assert corr > 0.95

    def test_peak_creates_oscillation_in_acf(self):
        """A periodic peak should create oscillatory structure in the ACF."""
        sfreq = 256.0
        n_fft = 16384
        peaks = [(1.0, 10.0, 1.0)]
        acf = _model_acf_via_irfft(1.0, 0.5, 1.5, peaks, sfreq, n_fft, 256)

        # ACF of a narrow peak at 10 Hz should oscillate at ~10 Hz
        # Check that there's a local max near lag = 1/10 sec = 0.1 sec = 25.6 samples
        target_lag = int(round(sfreq / 10.0))
        window = acf[target_lag - 3 : target_lag + 4]
        assert np.max(window) > acf[target_lag // 2]

    def test_different_n_fft_consistent(self):
        """Doubling n_fft should produce similar ACF (better freq resolution)."""
        params = (1.5, 0.5, 1.5, [(0.3, 10.0, 2.0)], 256.0)
        acf_small = _model_acf_via_irfft(*params, n_fft=4096, max_lag_samples=100)
        acf_large = _model_acf_via_irfft(*params, n_fft=16384, max_lag_samples=100)
        corr = np.corrcoef(acf_small, acf_large)[0, 1]
        assert corr > 0.99


# ---------------------------------------------------------------------------
# _compute_empirical_acf
# ---------------------------------------------------------------------------

class TestComputeEmpiricalAcf:
    def test_r0_is_variance(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(2000)
        acf = _compute_empirical_acf(x, max_lag_samples=10)
        np.testing.assert_allclose(acf[0], np.var(x), rtol=0.01)

    def test_white_noise_decorrelated(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(10000)
        acf = _compute_empirical_acf(x, max_lag_samples=100)
        assert np.all(np.abs(acf[1:]) < 0.1 * acf[0])

    def test_sinusoidal_acf(self):
        """ACF of sin(2*pi*f*t) should be cosine at the same frequency."""
        sfreq = 256.0
        t = np.arange(10000) / sfreq
        x = np.sin(2 * np.pi * 10.0 * t)
        acf = _compute_empirical_acf(x, max_lag_samples=64)

        lags_sec = np.arange(64) / sfreq
        expected = 0.5 * np.cos(2 * np.pi * 10.0 * lags_sec)
        np.testing.assert_allclose(acf / acf[0], expected / expected[0], atol=0.007)

    def test_ar1_exponential_decay(self):
        """ACF of AR(1) with coefficient phi should decay as phi^|m|."""
        rng = np.random.default_rng(42)
        phi = 0.8
        n = 200000
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = phi * x[i - 1] + rng.standard_normal()

        max_lag = 12
        acf = _compute_empirical_acf(x, max_lag_samples=max_lag)
        expected = acf[0] * phi ** np.arange(max_lag)
        np.testing.assert_allclose(acf, expected, rtol=0.1)

    def test_output_length(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(500)
        for ml in [10, 50, 200]:
            acf = _compute_empirical_acf(x, max_lag_samples=ml)
            assert len(acf) == ml

    def test_mean_centering(self):
        """Adding a constant offset should not change the ACF."""
        rng = np.random.default_rng(42)
        x = rng.standard_normal(1000)
        acf1 = _compute_empirical_acf(x, max_lag_samples=50)
        acf2 = _compute_empirical_acf(x + 100.0, max_lag_samples=50)
        np.testing.assert_allclose(acf1, acf2, atol=1e-10)


# ---------------------------------------------------------------------------
# Model/empirical consistency
# ---------------------------------------------------------------------------

class TestAcfConsistency:
    def test_model_matches_empirical_for_generated_signal(self):
        """Empirical ACF of a signal drawn from the model PSD should
        correlate highly with the model ACF."""
        rng = np.random.default_rng(42)
        sfreq = 256.0
        duration = 30.0
        n = int(sfreq * duration)
        n_fft = 2 * n

        b, k, chi = 1.0, 0.5, 1.5
        peaks = [(0.3, 10.0, 2.0)]

        f_grid = np.arange(n_fft // 2 + 1) * (sfreq / n_fft)
        model_psd = _compute_model_psd_linear(f_grid, b, k, chi, peaks)

        psd_two = model_psd.copy()
        psd_two[1:-1] /= 2.0
        amplitudes = np.sqrt(np.maximum(psd_two * n_fft * sfreq, 0.0))
        phases = rng.uniform(0, 2 * np.pi, len(amplitudes))
        spectrum = amplitudes * np.exp(1j * phases)
        spectrum[0] = 0.0
        if n_fft % 2 == 0:
            spectrum[-1] = np.abs(spectrum[-1])
        signal = np.fft.irfft(spectrum, n=n_fft)[:n]

        max_lag = int(sfreq * 2)
        acf_emp = _compute_empirical_acf(signal, max_lag)
        acf_model = _model_acf_via_irfft(b, k, chi, peaks, sfreq, n_fft, max_lag)

        corr = np.corrcoef(acf_emp, acf_model)[0, 1]
        assert corr > 0.9

    def test_r0_magnitudes_comparable(self):
        """R(0) of empirical and model should be in the same ballpark."""
        rng = np.random.default_rng(123)
        sfreq = 256.0
        n = int(sfreq * 30)
        n_fft = 2 * n

        b, k, chi = 1.5, 1.0, 1.5

        f_grid = np.arange(n_fft // 2 + 1) * (sfreq / n_fft)
        model_psd = _compute_model_psd_linear(f_grid, b, k, chi, [])
        psd_two = model_psd.copy()
        psd_two[1:-1] /= 2.0
        amplitudes = np.sqrt(np.maximum(psd_two * n_fft * sfreq, 0.0))
        phases = rng.uniform(0, 2 * np.pi, len(amplitudes))
        spectrum = amplitudes * np.exp(1j * phases)
        spectrum[0] = 0.0
        if n_fft % 2 == 0:
            spectrum[-1] = np.abs(spectrum[-1])
        signal = np.fft.irfft(spectrum, n=n_fft)[:n]

        acf_emp = _compute_empirical_acf(signal, 1)
        acf_model = _model_acf_via_irfft(b, k, chi, [], sfreq, n_fft, 1)

        ratio = acf_emp[0] / acf_model[0]
        assert 0.5 < ratio < 2.0
