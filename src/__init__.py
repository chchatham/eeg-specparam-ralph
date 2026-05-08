from .schemas import AperiodicParams, EEGSignal, FitDiagnostics, PeriodicPeak, SpecParamResult
from .eeg_generator import generate_eeg_signal, compute_target_psd, validate_signal_psd
from .spectral_specparam import fit_spectral_specparam
from .time_domain_wrapper import fit_time_domain
from .comparison import ComparisonResult, compare_results, compute_agreement_metrics, tost_equivalence

__all__ = [
    "AperiodicParams",
    "EEGSignal",
    "FitDiagnostics",
    "PeriodicPeak",
    "SpecParamResult",
    "generate_eeg_signal",
    "compute_target_psd",
    "validate_signal_psd",
    "fit_spectral_specparam",
    "fit_time_domain",
    "ComparisonResult",
    "compare_results",
    "compute_agreement_metrics",
    "tost_equivalence",
]
