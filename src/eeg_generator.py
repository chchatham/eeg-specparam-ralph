from __future__ import annotations

import numpy as np
from scipy.signal import welch

from .schemas import AperiodicParams, EEGSignal, PeriodicPeak


def compute_target_psd(
    freqs: np.ndarray,
    aperiodic: AperiodicParams,
    peaks: list[PeriodicPeak],
) -> np.ndarray:
    """Compute target one-sided PSD (linear power) from specparam-style parameters.

    Aperiodic model (log10 space):
      fixed:  log10(PSD) = offset - exponent * log10(f)
      knee:   log10(PSD) = offset - log10(knee + f^exponent)

    Periodic peaks are additive Gaussians in log10 space on top of the aperiodic.
    """
    freqs_safe = np.array(freqs, dtype=float)
    has_dc = freqs_safe[0] == 0.0
    if has_dc:
        freqs_safe[0] = freqs_safe[1] if len(freqs_safe) > 1 else 1.0

    if aperiodic.knee is not None and aperiodic.knee > 0:
        log_psd_ap = aperiodic.offset - np.log10(
            aperiodic.knee + freqs_safe ** aperiodic.exponent
        )
    else:
        log_psd_ap = aperiodic.offset - aperiodic.exponent * np.log10(freqs_safe)

    log_psd_peaks = np.zeros_like(freqs, dtype=float)
    for peak in peaks:
        log_psd_peaks += peak.power * np.exp(
            -((freqs - peak.center_frequency) ** 2) / (2 * peak.bandwidth ** 2)
        )

    psd = 10.0 ** (log_psd_ap + log_psd_peaks)
    if has_dc:
        psd[0] = 0.0
    return psd


def generate_eeg_signal(
    aperiodic: AperiodicParams,
    peaks: list[PeriodicPeak] | None = None,
    sfreq: float = 256.0,
    duration: float = 10.0,
    n_channels: int = 1,
    random_seed: int | None = None,
) -> EEGSignal:
    """Generate synthetic artifact-free EEG via spectral synthesis (Timmer & Koenig 1995).

    Constructs a target PSD from specparam-style parameters, then generates
    time-domain signals whose periodogram expectation matches that PSD.
    """
    if peaks is None:
        peaks = []

    rng = np.random.default_rng(random_seed)
    n_samples = int(sfreq * duration)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sfreq)
    n_freqs = len(freqs)

    target_psd = compute_target_psd(freqs, aperiodic, peaks)

    # Timmer & Koenig spectral synthesis:
    # For one-sided PSD S(f), the rfft coefficient X_k should satisfy
    # E[2|X_k|^2 / (N*fs)] = S(f_k)  →  E[|X_k|^2] = S(f_k)*N*fs/2
    # With X_k = sigma*(Z_r + j*Z_i), E[|X_k|^2] = 2*sigma^2
    # → sigma = sqrt(S(f_k)*N*fs/4)
    scale = np.sqrt(target_psd * n_samples * sfreq / 4.0)
    scale[0] = 0.0

    data = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        zr = rng.normal(size=n_freqs)
        zi = rng.normal(size=n_freqs)
        coeffs = scale * (zr + 1j * zi)

        coeffs[0] = 0.0
        if n_samples % 2 == 0:
            coeffs[-1] = scale[-1] * np.sqrt(2) * zr[-1]

        data[ch] = np.fft.irfft(coeffs, n=n_samples)

    return EEGSignal(
        data=data,
        sfreq=sfreq,
        duration=duration,
        ground_truth_aperiodic=aperiodic,
        ground_truth_peaks=list(peaks),
    )


def validate_signal_psd(
    signal: EEGSignal,
    freq_range: tuple[float, float] = (1.0, 40.0),
    channel: int = 0,
) -> dict:
    """Compute PSD and check it against the signal's ground-truth parameters.

    Returns a dict with fitted exponent/offset and detected peak frequencies.
    """
    data = signal.data[channel]
    nperseg = min(len(data), int(signal.sfreq * 2))
    freqs, psd = welch(data, fs=signal.sfreq, nperseg=nperseg)

    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_fit = freqs[mask]
    psd_fit = psd[mask]

    peak_centers = [p.center_frequency for p in signal.ground_truth_peaks]
    ap_mask = np.ones(len(freqs_fit), dtype=bool)
    for cf in peak_centers:
        ap_mask &= np.abs(freqs_fit - cf) > 3.0

    fitted_exponent = None
    fitted_offset = None
    if ap_mask.sum() > 2:
        log_f = np.log10(freqs_fit[ap_mask])
        log_psd = np.log10(psd_fit[ap_mask])
        coeffs = np.polyfit(log_f, log_psd, 1)
        fitted_exponent = -coeffs[0]
        fitted_offset = coeffs[1]

    detected_peak_freqs: list[float | None] = []
    for peak in signal.ground_truth_peaks:
        window = (freqs >= peak.center_frequency - peak.bandwidth * 3) & (
            freqs <= peak.center_frequency + peak.bandwidth * 3
        )
        if window.any():
            detected_peak_freqs.append(float(freqs[window][np.argmax(psd[window])]))
        else:
            detected_peak_freqs.append(None)

    return {
        "freqs": freqs,
        "psd": psd,
        "fitted_exponent": fitted_exponent,
        "fitted_offset": fitted_offset,
        "expected_exponent": signal.ground_truth_aperiodic.exponent,
        "expected_offset": signal.ground_truth_aperiodic.offset,
        "detected_peak_freqs": detected_peak_freqs,
        "expected_peak_freqs": peak_centers,
    }
