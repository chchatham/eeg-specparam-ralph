from __future__ import annotations

import numpy as np
from scipy.signal import welch
from specparam import SpectralModel

from .schemas import AperiodicParams, EEGSignal, PeriodicPeak, SpecParamResult


def fit_spectral_specparam(
    signal: EEGSignal,
    channel: int = 0,
    freq_range: tuple[float, float] = (1.0, 40.0),
    aperiodic_mode: str = "fixed",
    peak_width_limits: tuple[float, float] = (0.5, 12.0),
    max_n_peaks: int = 6,
    min_peak_height: float = 0.1,
    peak_threshold: float = 2.0,
) -> SpecParamResult:
    """Run spectral SpecParam (FOOOF) on one channel and return a SpecParamResult."""

    data = signal.data[channel]
    nperseg = min(len(data), int(signal.sfreq * 2))
    freqs, psd = welch(data, fs=signal.sfreq, nperseg=nperseg)

    sm = SpectralModel(
        aperiodic_mode=aperiodic_mode,
        verbose=False,
        metrics=["gof_rsquared"],
        algorithm_settings={
            "peak_width_limits": list(peak_width_limits),
            "max_n_peaks": max_n_peaks,
            "min_peak_height": min_peak_height,
            "peak_threshold": peak_threshold,
        },
    )
    sm.add_data(freqs, psd, freq_range=list(freq_range))
    sm.fit()

    converged = bool(sm.results.has_model)
    if not converged:
        return SpecParamResult(
            aperiodic=AperiodicParams(offset=0.0, exponent=0.0),
            peaks=[],
            r_squared=None,
            frequency_range=freq_range,
            method="spectral",
            converged=False,
        )

    ap = sm.get_params("aperiodic")
    if aperiodic_mode == "knee":
        aperiodic = AperiodicParams(
            offset=float(ap[0]),
            exponent=float(ap[2]),
            knee=float(ap[1]),
        )
    else:
        aperiodic = AperiodicParams(
            offset=float(ap[0]),
            exponent=float(ap[1]),
        )

    pk = sm.get_params("peak")
    peaks: list[PeriodicPeak] = []
    if pk.ndim == 2 and pk.shape[0] > 0:
        for row in pk:
            peaks.append(
                PeriodicPeak(
                    center_frequency=float(row[0]),
                    power=float(row[1]),
                    bandwidth=float(row[2]) / 2.0,
                )
            )

    r_squared = sm.get_metrics("gof_rsquared")

    return SpecParamResult(
        aperiodic=aperiodic,
        peaks=peaks,
        r_squared=float(r_squared) if r_squared is not None else None,
        frequency_range=freq_range,
        method="spectral",
        converged=True,
    )
