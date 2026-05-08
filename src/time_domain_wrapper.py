from __future__ import annotations

from .schemas import AperiodicParams, EEGSignal, PeriodicPeak, SpecParamResult
from .timedomain_specparam import fit_time_domain_specparam


def fit_time_domain(
    signal: EEGSignal,
    channel: int = 0,
    freq_range: tuple[float, float] = (1.0, 40.0),
    max_lag_sec: float = 1.0,
    max_n_peaks: int = 6,
    min_peak_height: float = 0.1,
    peak_width_limits: tuple[float, float] = (0.5, 12.0),
    peak_threshold: float = 2.0,
) -> SpecParamResult:
    """Run the time-domain SpecParam and return a SpecParamResult."""

    data = signal.data[channel]
    result = fit_time_domain_specparam(
        data, signal.sfreq,
        max_lag_sec=max_lag_sec,
        freq_range=freq_range,
        max_n_peaks=max_n_peaks,
        min_peak_height=min_peak_height,
        peak_width_limits=peak_width_limits,
        peak_threshold=peak_threshold,
    )

    if result is None:
        return SpecParamResult(
            aperiodic=AperiodicParams(offset=0.0, exponent=0.0),
            peaks=[],
            r_squared=None,
            frequency_range=freq_range,
            method="time_domain",
            converged=False,
        )

    b, k, chi = result["aperiodic"]

    aperiodic = AperiodicParams(
        offset=float(b),
        exponent=float(chi),
        knee=float(k),
    )

    peaks: list[PeriodicPeak] = []
    for a, c, w in result["peaks"]:
        peaks.append(
            PeriodicPeak(
                center_frequency=float(c),
                power=float(a),
                bandwidth=float(w),
            )
        )

    return SpecParamResult(
        aperiodic=aperiodic,
        peaks=peaks,
        r_squared=result.get("r_squared"),
        frequency_range=freq_range,
        method="time_domain",
        converged=True,
    )
