# Progress

## Last Updated
Iteration 17 — Phase 10I-10J: CSS architecture consolidation and verification.

## What Exists
- `pyproject.toml`, `src/__init__.py`, `tests/__init__.py`
- `src/schemas.py` — AperiodicParams, PeriodicPeak, FitDiagnostics, SpecParamResult, EEGSignal
- `src/eeg_generator.py` — Timmer & Koenig spectral synthesis + PSD validation
- `src/spectral_specparam.py` — wrapper around specparam v2.0 (SpectralModel API)
- `src/timedomain_specparam.py` — PSD-based + ACF-based fitting (method="psd"|"acf")
- `src/time_domain_wrapper.py` — maps dict output to SpecParamResult, threads method param (default="acf"), populates FitDiagnostics
- `src/timedomain_specparam_original.py` — unmodified reference (do not edit)
- `src/comparison.py` — ComparisonResult, agreement metrics, TOST equivalence
- `src/app.py` — FastAPI web UI: overview landing page, AJAX simulator, sweep pages, ACF/PSD plots
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
36. **Cache deterministic sweeps** — overview mini-sweep and parameter sweep results are cached (module-level singleton and `@lru_cache`). No reason to recompute on every request.

## Known Issues
- Spectral SpecParam detects more spurious noise peaks than TD (TD is closer to ground truth peak count)
- ACF r_squared (~0.6) is lower than PSD r_squared (~0.99) — structural, not a bug

## Current Focus
Phase 10J — Deploy to Railway. All Phase 10 UI redesign items complete except Docker build + deploy.

## What was done in Iteration 17
- **Phase 10I complete:** CSS architecture consolidation
  - Created `_SIMULATOR_CSS` module-level constant with all simulator-specific CSS
  - Consolidated `_SHINY_CSS = _BASE_CSS + _OVERVIEW_CSS + _SIMULATOR_CSS` — single constant used by all pages
  - Added print stylesheet (hide sidebar, full-width main) and responsive breakpoints to `_SIMULATOR_CSS`
  - Updated overview, simulate, and sweep page templates to use `{_SHINY_CSS}` instead of inline CSS
- **Phase 10J verification (partial):**
  - All 213 tests pass
  - All 5 routes return 200
  - Visual checks: cards, sidebar, navbar, hero, metrics, AJAX/AbortController all present
  - Docker not available locally — will build on Railway

## What was done in Iteration 16
- **Phase 10A-10H complete:** Full R Shiny-style UI redesign
  - Sidebar + main panel layout for `/simulate` page (300px fixed sidebar, collapsible sections)
  - Card/panel system wrapping plots, tables, and metrics
  - Dark navbar with hamburger toggle for mobile
  - Shiny-style input controls (sliders, wells, help text, smooth peak animation)
  - Overview hero section, theorem boxes, CSS flow diagram, example cards, metrics row
  - CSS custom properties for theming, typography hierarchy
  - Skeleton loading animations, expand-to-fullscreen plots
  - Sweep page with metrics row, styled controls, card layout

## What was done in Iteration 15
- **Simplify:** Three-agent code review found and fixed 7 issues:
  1. Extracted `_assemble_peaks()` — deduped peak param assembly across `simulate_page` and `simulate_compute`
  2. Extracted `_fit_signal()` — deduped generate→fit→compare core used in 3 places
  3. Extracted `_peak_fields_html()` — deduped 39 lines of repeated HTML for peak field groups
  4. Cached `_run_mini_sweep()` with module-level singleton — eliminates 5-8s recomputation on every `GET /`
  5. Added `@lru_cache(maxsize=8)` on `_run_sweep()` — eliminates repeated sweep computation
  6. Consolidated `.equiv-badge`/`.badge` CSS into single `.badge` in `_BASE_CSS`
  7. Replaced throwaway `len([1.0,...])` with literal `5`
- **Deploy:** Committed, pushed to origin/main, deployed to Railway via `railway up --detach`
- All 213 tests pass. All 5 routes returning 200 in production.

## Routes
- `GET /` — overview landing page (cached 15-run mini-sweep)
- `GET /simulate` — interactive simulator with AJAX auto-compute
- `GET /simulate/compute` — JSON API for AJAX updates
- `GET /sweep` — parameter space sweep (quick/full modes, cached per seed)
- `GET /health` — health check

## Next Steps
Phase 10J remaining: commit all Phase 10 changes, push to origin/main, deploy to Railway.
All changes are HTML/CSS only — no logic changes, all 213 tests pass.
