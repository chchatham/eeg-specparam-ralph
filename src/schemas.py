from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass
class AperiodicParams:
    offset: float
    exponent: float
    knee: float | None = None


@dataclass
class PeriodicPeak:
    center_frequency: float
    power: float
    bandwidth: float


@dataclass
class SpecParamResult:
    aperiodic: AperiodicParams
    peaks: list[PeriodicPeak]
    r_squared: float | None
    frequency_range: tuple[float, float]
    method: str
    converged: bool


@dataclass
class EEGSignal:
    data: np.ndarray
    sfreq: float
    duration: float
    ground_truth_aperiodic: AperiodicParams
    ground_truth_peaks: list[PeriodicPeak] = field(default_factory=list)
