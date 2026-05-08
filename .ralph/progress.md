# Progress

## Last Updated
Iteration 8 — Phase 7 complete: API parity & export.

## What Exists
- `pyproject.toml`, `src/__init__.py`, `tests/__init__.py`
- `src/schemas.py` — shared dataclasses (AperiodicParams, PeriodicPeak, SpecParamResult, EEGSignal)
- `src/eeg_generator.py` — Timmer & Koenig spectral synthesis + PSD validation
- `src/spectral_specparam.py` — wrapper around specparam v2.0 (SpectralModel API)
- `src/timedomain_specparam.py` — PSD-based fitting with free chi, multi-peak, r_squared
- `src/time_domain_wrapper.py` — maps dict output to SpecParamResult
- `src/timedomain_specparam_original.py` — unmodified reference (do not edit)
- `src/comparison.py` — ComparisonResult, agreement metrics, TOST equivalence
- `src/app.py` — FastAPI web UI with Plotly PSD plot + parameter comparison table
- `Dockerfile` — Python 3.12-slim, pip install, expose 8000
- `railway.toml` — DOCKERFILE builder, healthcheck at /health, PORT env var
- Tests: 128 total, all passing
  - test_eeg_generator.py (25), test_spectral_specparam.py (13)
  - test_time_domain_wrapper.py (17), test_comparison.py (8)
  - test_regression_equivalence.py (65) — full parameter sweep regression suite (module-scoped fixture cache)

## Current Agreement (across exponent sweep)
| Metric         | Value   |
|----------------|---------|
| Exponent RMSE  | < 0.15  |
| Offset RMSE    | < 0.1   |
| Peak center RMSE | < 2 Hz |
| TD r_squared   | > 0.9   |

## What's Broken
Nothing — all tests pass. App runs locally on port 8000/8001.

## Decisions Made (Do Not Revisit)
1–15: (see prior entries)
16. Multi-peak: iterative detection from flattened residual, edge guards at ±1.5 Hz.
17. R-squared computed on log10 PSD residuals (SS_res / SS_tot).
18. TOST equivalence test uses scipy t-distribution with user-specified bounds.
19. Web framework: FastAPI + Plotly (server-side HTML rendering via `plotly.io.to_html`).
20. Switched from ACF-based to PSD-based fitting for time-domain — ACF approach fails for chi >= 1.5.
21. specparam v2.0 API: `SpectralModel`, metrics via `metrics=["gof_rsquared"]`, peak settings via `algorithm_settings` dict.

## Decisions Made (continued)
22. Spectral BW = 2*sigma; our PeriodicPeak.bandwidth stores sigma. Divide by 2 when extracting from specparam.
23. Shared CSS via `_BASE_CSS` constant in app.py; page-specific rules appended per template.
24. PSD plot reconstruction uses `compute_target_psd()` instead of manual formula — single source of truth.
25. `peak_width_limits` and `peak_threshold` exposed as kwargs with defaults matching prior hardcoded values (0.5–12.0 Hz, 2.0×noise). Stage 3 refit bounds also use these limits.

## Known Issues
- Spectral SpecParam detects more spurious noise peaks than TD (TD is closer to ground truth peak count)

## Current Focus
**Deploy to Railway** — the only remaining unchecked item.

## What was done in Iteration 8
- Added `peak_width_limits` and `peak_threshold` params to `fit_time_domain_specparam()`, threaded to `_detect_peaks` and Stage 3 refit bounds
- Added same params to `fit_time_domain()` wrapper, threaded through
- Populated `src/__init__.py` with all public API exports (14 symbols)
- All 128 tests pass, smoke test with custom peak params confirmed working

## Next Steps
1. Deploy updated code to Railway (only unchecked checkbox remaining)
