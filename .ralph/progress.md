# Progress

## Last Updated
Iteration 6 — Bland-Altman plot, sweep heatmap, and regression test suite added.

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
- Tests: 123 total, all passing
  - test_eeg_generator.py (25), test_spectral_specparam.py (13)
  - test_time_domain_wrapper.py (17), test_comparison.py (8)
  - test_regression_equivalence.py (60) — full parameter sweep regression suite

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

## Known Issues
- Peak bandwidth ~2x different from spectral SpecParam (different fitting methodology)
- Spectral SpecParam detects more spurious noise peaks than TD (TD is closer to ground truth peak count)
- Not yet deployed to Railway (config is ready, needs `railway up`)
- No README yet

## Current Focus
**Remaining work to complete all phases:**
1. Deploy to Railway (Phase 6) — needs `railway up` with auth
2. Write README with deployment instructions (Phase 6)

## Next Steps
1. Deploy to Railway with `railway up`
2. Write README with deployment instructions
