# Progress

## Last Updated
Iteration 12 — Phase 8 complete (all sub-phases 8A–8G). ACF fitting pipeline built, validated, deployed.

## What Exists
- `pyproject.toml`, `src/__init__.py`, `tests/__init__.py`
- `src/schemas.py` — AperiodicParams, PeriodicPeak, FitDiagnostics, SpecParamResult, EEGSignal
- `src/eeg_generator.py` — Timmer & Koenig spectral synthesis + PSD validation
- `src/spectral_specparam.py` — wrapper around specparam v2.0 (SpectralModel API)
- `src/timedomain_specparam.py` — PSD-based + ACF-based fitting (method="psd"|"acf")
- `src/time_domain_wrapper.py` — maps dict output to SpecParamResult, threads method param (default="acf"), populates FitDiagnostics
- `src/timedomain_specparam_original.py` — unmodified reference (do not edit)
- `src/comparison.py` — ComparisonResult, agreement metrics, TOST equivalence
- `src/app.py` — FastAPI web UI with PSD plot + ACF plot + parameter comparison table + sweep pages
- `Dockerfile` — Python 3.12-slim, pip install, expose 8000
- `railway.toml` — DOCKERFILE builder, healthcheck at /health, PORT env var
- Tests: 213 total, all passing
  - test_acf_math.py (21) — IRFFT scaling, AR(1), chi=2 shape, empirical ACF, model/empirical consistency
  - test_eeg_generator.py (25), test_spectral_specparam.py (13)
  - test_time_domain_wrapper.py (33) — PSD wrapper (17) + ACF fitting/residuals/long-lag (16)
  - test_comparison.py (8)
  - test_regression_equivalence.py (113) — PSD sweep (65) + ACF equivalence sweep (48)

## Current Agreement

### PSD-based TD fitter
| Metric         | Value   |
|----------------|---------|
| Exponent RMSE  | < 0.15  |
| Offset RMSE    | < 0.1   |
| Peak center RMSE | < 2 Hz |
| PSD r_squared  | > 0.9   |

### ACF-based TD fitter (default)
| Metric            | Value   |
|-------------------|---------|
| Exponent RMSE     | 0.032   |
| Max exponent error| 0.052   |
| Convergence rate  | 100%    |
| ACF r_squared     | ~0.55-0.65 |

## What's Broken
Nothing. All 213 tests pass. Default method switched to "acf".

## Decisions Made (Do Not Revisit)
1–31: (see prior entries — still valid)
32. **PSD joint refit before ACF stage** — aperiodic-only PSD fit overestimates chi when peaks are present. Joint PSD refit gives chi_init with error < 0.03. ACF stage regularizes toward it (weight=30).
33. **ACF domain cannot independently determine chi** — SpecParam's log-additive separability doesn't transfer to ACF domain. Chi must be anchored by PSD analysis.
34. **Default method switched to "acf"** — wrapper `fit_time_domain` defaults to method="acf". Core function `fit_time_domain_specparam` still defaults to "psd" for backward compatibility. PSD regression tests use explicit `method="psd"`.
35. **ACF r_squared is structurally lower (~0.6)** than PSD r_squared (~0.99) because the ACF has more structure (oscillations, long-lag decay) that the model doesn't capture perfectly. This is expected and not a quality issue — the parameter recovery is excellent.

## Known Issues
- Spectral SpecParam detects more spurious noise peaks than TD (TD is closer to ground truth peak count)
- ACF r_squared (~0.6) is lower than PSD r_squared (~0.99) — structural, not a bug

## Current Focus
Phase 8 complete. Only remaining unchecked item: Phase 6 — "Deployed and accessible on Railway."

## What was done in Iteration 12
- **Fixed `_model_acf_via_irfft` 2x scaling bug**: convert one-sided PSD to two-sided before irfft. Fixed both `_model_acf_via_irfft` and `_band_model_acf`.
- **Fixed Phase 8A tests**: updated total power formulas and signal generation scaling.
- **Discovered and resolved ACF chi bias**: tried long-lag approach (underestimated), then added PSD joint refit before ACF stage with strong regularization. Chi error < 0.06 across all conditions.
- **Phase 8C**: Added FitDiagnostics dataclass, diagnostics field on SpecParamResult, method parameter on wrapper, FitDiagnostics export.
- **Phase 8D**: Added 16 ACF-specific tests (convergence, exponent recovery, diagnostics, long-lag).
- **Phase 8E**: Added 48 ACF equivalence tests. Full sweep: 100% convergence, RMSE=0.032, max error=0.052.
- **Phase 8F**: Added ACF plot panel to app.py with empirical/model ACF and residual subplot.
- **Phase 8G**: Switched wrapper default to method="acf". All 213 tests pass.

## Next Steps
1. Deploy to Railway (Phase 6 — only unchecked checkbox remaining).
