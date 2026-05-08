from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit, least_squares
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


def _unpack_peaks(params, offset, n_peaks):
    """Extract list of (a, c, w) tuples from a flat parameter vector."""
    return [(float(params[offset + i * 3]),
             float(params[offset + i * 3 + 1]),
             float(params[offset + i * 3 + 2])) for i in range(n_peaks)]


def _build_peak_bounds(detected_peaks, freq_range, peak_width_limits):
    """Build p0/lower/upper lists for peak parameters."""
    p0, lower, upper = [], [], []
    for a_pk, c_pk, w_pk in detected_peaks:
        p0.extend([a_pk, c_pk, w_pk])
        lower.extend([0.0, freq_range[0], peak_width_limits[0]])
        upper.extend([3.0, freq_range[1], peak_width_limits[1]])
    return p0, lower, upper


def _full_psd_model(f, n_peaks, *params):
    """Aperiodic + N Gaussian peaks model in log10-PSD space."""
    result = _aperiodic_model(f, params[0], params[1], params[2])
    for a, c, w in _unpack_peaks(params, 3, n_peaks):
        result = result + _gaussian_peak(f, a, c, w)
    return result


def _compute_model_psd_linear(f_grid, b, k, chi, peaks):
    """Build linear-scale PSD from model params on an rfft frequency grid.

    Returns S(f) = 10^(aperiodic + peaks) in V²/Hz.
    DC bin (f=0) is zeroed for consistency with mean-centered empirical ACF.
    """
    log10_psd = _aperiodic_model(f_grid, b, k, chi)
    for a, c, w in peaks:
        log10_psd = log10_psd + _gaussian_peak(f_grid, a, c, w)
    psd_linear = 10.0 ** log10_psd
    if len(f_grid) > 0 and f_grid[0] == 0.0:
        psd_linear[0] = 0.0
    return psd_linear


def _model_acf_via_irfft(b, k, chi, peaks, sfreq, n_fft, max_lag_samples):
    """Compute model autocovariance via IRFFT of the model PSD.

    Converts one-sided PSD to two-sided before irfft, then scales by sfreq
    so that R(0) equals the total variance (one-sided PSD integral).
    """
    f_grid = np.arange(n_fft // 2 + 1) * (sfreq / n_fft)
    model_psd = _compute_model_psd_linear(f_grid, b, k, chi, peaks)
    psd_two = model_psd.copy()
    psd_two[1:-1] /= 2.0
    acf_full = np.fft.irfft(psd_two, n=n_fft) * sfreq
    return acf_full[:max_lag_samples]


def _compute_empirical_acf(x, max_lag_samples):
    """Compute empirical autocovariance via FFT (biased estimator).

    R[m] = (1/N) * sum_{t=0}^{N-1-m} x_c[t] * x_c[t+m]
    where x_c = x - mean(x).  Uses n_fft = 2*N for linear correlation.
    """
    x_c = x - np.mean(x)
    n = len(x_c)
    n_fft = 2 * n
    X = np.fft.rfft(x_c, n=n_fft)
    S = np.abs(X) ** 2
    acf_full = np.fft.irfft(S, n=n_fft) / n
    return acf_full[:max_lag_samples]


def _fit_acf_stages(time_series, sfreq, b_init, k_init, chi_init, detected_peaks,
                    freq_range, peak_width_limits, min_peak_height):
    """ACF-domain fitting with chi anchored to PSD-refined estimate.

    Both empirical and model ACFs are band-limited to freq_range before
    comparison, so the fit only sees frequencies where the parametric model
    is valid (avoids sub-Hz power mismatch).  Fits the normalized ACF shape
    (R/R(0)) to decouple b, then recovers b analytically from R(0).

    Chi is strongly regularized toward the PSD-refined init because the
    SpecParam model's aperiodic-periodic separation is not preserved in
    the ACF domain (additive in log-PSD becomes multiplicative in linear PSD).
    """
    n_samples = len(time_series)
    duration = n_samples / sfreq
    acf_max_lag_sec = min(duration / 2, 15.0)
    max_lag_samples = max(int(acf_max_lag_sec * sfreq), 10)
    n_fft = 2 * n_samples

    f_grid = np.arange(n_fft // 2 + 1) * (sfreq / n_fft)
    band_mask = (f_grid >= freq_range[0]) & (f_grid <= freq_range[1])

    x_c = time_series - np.mean(time_series)
    X = np.fft.rfft(x_c, n=n_fft)
    S_emp = np.abs(X) ** 2
    S_emp_band = np.where(band_mask, S_emp, 0.0)
    empirical_acf = np.fft.irfft(S_emp_band, n=n_fft)[:max_lag_samples] / n_samples

    lags = np.arange(max_lag_samples) / sfreq
    rho_emp = empirical_acf / max(abs(float(empirical_acf[0])), 1e-12)
    lag_weights = np.exp(-np.arange(max_lag_samples) / (2.0 * sfreq))

    def _band_model_acf(b, k, chi, peaks_list):
        model_psd = _compute_model_psd_linear(f_grid, b, k, chi, peaks_list)
        model_psd_band = np.where(band_mask, model_psd, 0.0)
        psd_two = model_psd_band.copy()
        psd_two[1:-1] /= 2.0
        return np.fft.irfft(psd_two, n=n_fft)[:max_lag_samples] * sfreq

    def _recover_b(k, chi, peaks_list):
        acf0 = _band_model_acf(0.0, k, chi, peaks_list)[0]
        return float(np.log10(max(empirical_acf[0], 1e-30))
                     - np.log10(max(abs(acf0), 1e-30)))

    n_peaks = len(detected_peaks)
    chi_reg_weight = 30.0

    def acf_residual(params):
        k, chi = params[0], params[1]
        peaks = _unpack_peaks(params, 2, n_peaks)
        model = _band_model_acf(0.0, k, chi, peaks)
        r0 = max(abs(model[0]), 1e-12)
        shape_resid = lag_weights * (rho_emp - model / r0)
        reg = np.array([chi_reg_weight * (chi - chi_init)])
        return np.concatenate([shape_resid, reg])

    pk_p0, pk_lo, pk_hi = _build_peak_bounds(detected_peaks, freq_range, peak_width_limits)
    p0 = [k_init, chi_init] + pk_p0
    lower = [0.0, 0.5] + pk_lo
    upper = [50.0, 3.0] + pk_hi

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = least_squares(
            acf_residual, x0=p0, bounds=(lower, upper),
            method='trf', max_nfev=15000,
        )

    popt = result.x
    k_f, chi_f = float(popt[0]), float(popt[1])
    all_peaks = _unpack_peaks(popt, 2, n_peaks)

    b_f = _recover_b(k_f, chi_f, all_peaks)
    model_acf = _band_model_acf(b_f, k_f, chi_f, all_peaks)
    final_peaks = [p for p in all_peaks if p[0] >= min_peak_height]

    return {
        "aperiodic": (b_f, k_f, chi_f),
        "peaks": final_peaks,
        "r_squared": _r_squared(empirical_acf, model_acf),
        "lags": lags,
        "empirical_acf": empirical_acf,
        "model_acf": model_acf,
        "acf_residual": empirical_acf - model_acf,
    }


def fit_time_domain_specparam(time_series, sfreq, max_lag_sec=1.0,
                              freq_range=(1.0, 40.0), max_n_peaks=6,
                              min_peak_height=0.1,
                              peak_width_limits=(0.5, 12.0),
                              peak_threshold=2.0,
                              method="psd"):
    """
    Fit the time-domain specparam model to a time series.

    method="psd": fit in log10-PSD domain via curve_fit (legacy).
    method="acf": fit in autocovariance domain via least_squares (IRFFT model ACF).
    Both use PSD initialization for initial parameter guesses.

    Returns params_dict with keys 'aperiodic' (b, k, chi), 'peaks'
    (list of (a, c, w) tuples), and 'r_squared', or None on failure.
    ACF method also returns 'lags', 'empirical_acf', 'model_acf', 'acf_residual'.
    """
    if method not in ("psd", "acf"):
        raise ValueError(f"method must be 'psd' or 'acf', got '{method}'")
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

    if method == "acf":
        # Refine aperiodic via joint PSD fit — aperiodic-only fit overestimates chi when peaks present.
        b_ref, k_ref, chi_ref = float(b_init), float(k_init), float(chi_init)
        if detected_peaks:
            n_pk = len(detected_peaks)
            pk_p0, pk_lo, pk_hi = _build_peak_bounds(detected_peaks, freq_range, peak_width_limits)
            jp0 = [b_init, k_init, chi_init] + pk_p0
            jlo = [-10.0, 0.0, 0.5] + pk_lo
            jhi = [10.0, 50.0, 3.0] + pk_hi
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    jpopt, _ = curve_fit(
                        lambda f, *p: _full_psd_model(f, n_pk, *p),
                        freqs_fit, log_psd,
                        p0=jp0, bounds=(jlo, jhi), maxfev=10000,
                    )
                b_ref, k_ref, chi_ref = float(jpopt[0]), float(jpopt[1]), float(jpopt[2])
                detected_peaks = [p for p in _unpack_peaks(jpopt, 3, n_pk)
                                  if p[0] >= min_peak_height]
            except RuntimeError:
                pass
        return _fit_acf_stages(
            time_series, sfreq,
            b_ref, k_ref, chi_ref,
            detected_peaks, freq_range, peak_width_limits, min_peak_height,
        )

    # --- PSD method (existing behavior) ---
    if not detected_peaks:
        model_log = _aperiodic_model(freqs_fit, b_init, k_init, chi_init)
        return {"aperiodic": (b_init, k_init, chi_init), "peaks": [],
                "r_squared": _r_squared(log_psd, model_log)}

    # Stage 3: Refit full model with aperiodic + all peaks jointly
    n_peaks = len(detected_peaks)
    pk_p0, pk_lo, pk_hi = _build_peak_bounds(detected_peaks, freq_range, peak_width_limits)
    p0 = [b_init, k_init, chi_init] + pk_p0
    lower = [-10.0, 0.0, 0.5] + pk_lo
    upper = [10.0, 50.0, 3.0] + pk_hi

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(
                lambda f, *p: _full_psd_model(f, n_peaks, *p),
                freqs_fit, log_psd,
                p0=p0, bounds=(lower, upper), maxfev=15000,
            )
    except RuntimeError:
        return {"aperiodic": (b_init, k_init, chi_init), "peaks": detected_peaks}

    b_f, k_f, chi_f = float(popt[0]), float(popt[1]), float(popt[2])
    final_peaks = [p for p in _unpack_peaks(popt, 3, n_peaks) if p[0] >= min_peak_height]
    model_log_psd = _full_psd_model(freqs_fit, n_peaks, *popt)

    return {
        "aperiodic": (float(b_f), float(k_f), float(chi_f)),
        "peaks": final_peaks,
        "r_squared": _r_squared(log_psd, model_log_psd),
    }
