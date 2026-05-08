from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import welch
import warnings


def _aperiodic_model(f, b, k, chi):
    """Log10 of the aperiodic PSD: b - log10(k + f^chi)."""
    return b - np.log10(np.maximum(k, 1e-12) + f**chi)


def _gaussian_peak(f, a, c, w):
    """Single Gaussian peak in log10 PSD space."""
    return a * np.exp(-((f - c) ** 2) / (2.0 * w**2))


def _detect_peaks(freqs, residual, min_peak_height=0.1, min_peak_width=0.5,
                  max_peak_width=12.0, peak_threshold=2.0, max_n_peaks=6):
    """Iteratively detect Gaussian peaks from the flattened log PSD residual.

    Returns list of (amplitude, center_freq, bandwidth) tuples.
    """
    peaks = []
    residual = residual.copy()
    edge_margin = 1.5  # Hz — ignore peaks within this of freq range edges

    for _ in range(max_n_peaks):
        # Mask out edge regions for peak detection
        interior = (freqs >= freqs[0] + edge_margin) & (freqs <= freqs[-1] - edge_margin)
        if not interior.any():
            break

        interior_residual = np.where(interior, residual, -np.inf)
        max_idx = np.argmax(interior_residual)
        max_val = residual[max_idx]

        if max_val < min_peak_height:
            break

        below = residual[interior & (residual < max_val * 0.5)]
        noise_std = np.std(below) if len(below) > 2 else 0.1
        if noise_std > 0 and max_val < peak_threshold * noise_std:
            break

        center = freqs[max_idx]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    _gaussian_peak, freqs, np.maximum(residual, 0),
                    p0=[max_val, center, 2.0],
                    bounds=([0.0, freqs[0] + edge_margin, min_peak_width],
                            [max_val * 2, freqs[-1] - edge_margin, max_peak_width]),
                    maxfev=2000,
                )
            a_fit, c_fit, w_fit = popt
            if a_fit >= min_peak_height:
                peaks.append((float(a_fit), float(c_fit), float(w_fit)))
                residual -= _gaussian_peak(freqs, *popt)
        except RuntimeError:
            break

    return peaks


def _r_squared(observed: np.ndarray, predicted: np.ndarray) -> float | None:
    ss_res = np.sum((observed - predicted) ** 2)
    ss_tot = np.sum((observed - np.mean(observed)) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None


def fit_time_domain_specparam(time_series, sfreq, max_lag_sec=1.0,
                              freq_range=(1.0, 40.0), max_n_peaks=6,
                              min_peak_height=0.1,
                              peak_width_limits=(0.5, 12.0),
                              peak_threshold=2.0):
    """
    Fit the time-domain specparam model to a time series.

    Three-stage approach:
    1. Fit the aperiodic component on the Welch PSD (log10 space)
    2. Detect peaks iteratively from the flattened residual
    3. Refit the full model with all detected peaks

    The model PSD is:
        log10(S(f)) = b - log10(k + f^chi) + sum_i a_i * Gauss(f; c_i, w_i)

    Returns params_dict with keys 'aperiodic' (b, k, chi), 'peaks'
    (list of (a, c, w) tuples), and 'r_squared', or None on failure.
    """
    n_samples = len(time_series)
    nperseg = min(n_samples, int(sfreq * 2))
    freqs, psd = welch(time_series, fs=sfreq, nperseg=nperseg)

    mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    freqs_fit = freqs[mask]
    psd_fit = psd[mask]
    log_psd = np.log10(np.maximum(psd_fit, 1e-30))

    # Stage 1: Fit aperiodic
    ap_lower = [-10.0, 0.0, 0.5]
    ap_upper = [10.0, 50.0, 3.0]
    ap_p0 = [np.mean(log_psd), 0.1, 1.5]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ap_popt, _ = curve_fit(
                _aperiodic_model, freqs_fit, log_psd,
                p0=ap_p0, bounds=(ap_lower, ap_upper), maxfev=5000,
            )
    except RuntimeError:
        return None

    b_init, k_init, chi_init = ap_popt

    # Stage 2: Detect peaks from residual
    log_ap = _aperiodic_model(freqs_fit, *ap_popt)
    residual = log_psd - log_ap
    detected_peaks = _detect_peaks(
        freqs_fit, residual,
        min_peak_height=min_peak_height,
        min_peak_width=peak_width_limits[0],
        max_peak_width=peak_width_limits[1],
        peak_threshold=peak_threshold,
        max_n_peaks=max_n_peaks,
    )

    if not detected_peaks:
        model_log = _aperiodic_model(freqs_fit, b_init, k_init, chi_init)
        return {"aperiodic": (b_init, k_init, chi_init), "peaks": [],
                "r_squared": _r_squared(log_psd, model_log)}

    # Stage 3: Refit full model with aperiodic + all peaks jointly
    n_peaks = len(detected_peaks)

    def full_model(f, *params):
        b, k, chi = params[0], params[1], params[2]
        result = _aperiodic_model(f, b, k, chi)
        for i in range(n_peaks):
            a_i = params[3 + i * 3]
            c_i = params[4 + i * 3]
            w_i = params[5 + i * 3]
            result = result + _gaussian_peak(f, a_i, c_i, w_i)
        return result

    p0 = [b_init, k_init, chi_init]
    lower = [-10.0, 0.0, 0.5]
    upper = [10.0, 50.0, 3.0]
    for a_pk, c_pk, w_pk in detected_peaks:
        p0.extend([a_pk, c_pk, w_pk])
        lower.extend([0.0, freq_range[0], peak_width_limits[0]])
        upper.extend([3.0, freq_range[1], peak_width_limits[1]])

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                full_model, freqs_fit, log_psd,
                p0=p0, bounds=(lower, upper), maxfev=15000,
            )
    except RuntimeError:
        return {"aperiodic": (b_init, k_init, chi_init), "peaks": detected_peaks}

    b_f, k_f, chi_f = popt[0], popt[1], popt[2]
    final_peaks = []
    for i in range(n_peaks):
        a_i = float(popt[3 + i * 3])
        c_i = float(popt[4 + i * 3])
        w_i = float(popt[5 + i * 3])
        if a_i >= min_peak_height:
            final_peaks.append((a_i, c_i, w_i))

    model_log_psd = full_model(freqs_fit, *popt)

    return {
        "aperiodic": (float(b_f), float(k_f), float(chi_f)),
        "peaks": final_peaks,
        "r_squared": _r_squared(log_psd, model_log_psd),
    }
