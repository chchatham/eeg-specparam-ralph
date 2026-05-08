# EEG SpecParam Equivalence Simulation

## Task
Build an interactive web simulation (deployed via Railway) that:
1. Generates realistic artifact-free synthetic human EEG data
2. Runs the standard spectral-domain SpecParam (formerly FOOOF) analysis on that data
3. Iteratively develops and optimizes a time-domain SpecParam implementation until it accurately reproduces the same metrics as spectral SpecParam
4. Presents side-by-side comparison of results, demonstrating equivalence

The spectral SpecParam (FOOOF) output is the **ground truth target**. The time-domain implementation
starts from `src/timedomain_specparam.py` (provided as a starting point) and must be extended/modified
until it matches spectral SpecParam's output across the parameter space.

## Success Criteria (Checkboxes)

### Phase 1: EEG Signal Generation
- [x] Implement 1/f aperiodic component generator with configurable exponent (typically 1.0–2.0)
- [x] Implement periodic (oscillatory) component generator — Gaussian peaks in frequency domain mapped to time-domain signals
- [x] Support standard EEG bands: delta (1–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), gamma (30–100 Hz)
- [x] Configurable sampling rate (default 256 Hz), duration, and number of channels
- [x] Configurable SNR for each oscillatory component
- [x] Signal generation produces clean (artifact-free) data — no eye blinks, muscle artifacts, line noise, or electrode drift
- [x] Generated signals pass sanity checks: PSD shape matches expected 1/f + peaks profile
- [x] Unit tests for signal generation

### Phase 2: Spectral-Domain SpecParam (FOOOF) Integration
- [x] Integrate `specparam` (FOOOF) package for standard spectral analysis
- [x] Compute PSD from generated signal (Welch's method or similar)
- [x] Fit SpecParam model to PSD; extract aperiodic params (offset, exponent) and periodic params (center freq, power, bandwidth)
- [x] Store results in structured format for comparison
- [x] Unit tests for spectral pipeline

### Phase 3: Time-Domain SpecParam — Development & Optimization
**Starting point: `src/timedomain_specparam.py` — this code is MODIFIABLE.**
The goal is to evolve this into a time-domain implementation that reproduces spectral SpecParam's
output. The provided code is a working prototype with known limitations to address.

**3A: Initial integration & baseline measurement**
- [x] Strip out `process_bids_dataset` and `mne_bids` dependency — this project uses synthetic data via numpy arrays
- [x] Create wrapper that accepts `EEGSignal` (numpy array + sfreq) and outputs `SpecParamResult`
- [x] Map current output params `[b, k, a, c, w]` to shared schema
- [x] Run on simple synthetic signal (single alpha peak + 1/f) alongside spectral SpecParam
- [x] Record baseline discrepancies — this is the "before" measurement
- [x] Unit tests for initial wrapper

**3B: Generalize aperiodic exponent (remove chi=2 hardcoding)**
- [x] Make the aperiodic exponent a free parameter in `generate_aperiodic_acf` (currently hardcoded as Lorentzian/chi=2)
- [x] Derive or approximate the ACF for arbitrary exponents (not just chi=2)
- [x] Add exponent to the fitted parameter vector — now `[b, k, chi, a, c, w]` or equivalent
- [x] Validate: generate signals with exponents 1.0, 1.5, 2.0 — confirm time-domain recovers the correct exponent
- [x] Compare recovered exponent against spectral SpecParam's exponent across sweep

**3C: Extend to multi-peak support**
- [x] Generalize `generate_periodic_acf` and the full model to handle N peaks (not just 1)
- [x] Implement peak detection / model selection — determine number of peaks from data (e.g., iterative fitting, BIC/AIC)
- [x] Fitted output should be `[b, k, chi, a1, c1, w1, a2, c2, w2, ...]`
- [x] Validate: generate signals with 0, 1, 2, 3 peaks — confirm time-domain finds the correct number and params
- [x] Compare multi-peak results against spectral SpecParam

**3D: Convergence & robustness improvements**
- [x] Improve initial parameter guesses — use spectral pre-analysis (quick FFT peek) to seed p0
- [x] Tune optimizer settings (maxfev, method, tolerance) to reduce convergence failures
- [x] Implement bounds that adapt to the data (e.g., set center_freq bounds from PSD peak locations)
- [x] Compute goodness-of-fit (r_squared equivalent) for the time-domain model
- [x] Stress test: sweep across full parameter space, track convergence rate — target >95%

**3E: Equivalence validation**
- [x] Across the full parameter sweep (exponent × peak count × SNR × duration), time-domain params must match spectral SpecParam within TOST equivalence bounds
- [x] If specific regions of parameter space show poor agreement, diagnose and fix
- [x] Final regression test suite: batch of synthetic signals with known ground truth, both pipelines must agree

### Phase 4: Equivalence Comparison Engine
- [x] Compute parameter-wise differences between spectral and time-domain results (aperiodic exponent, offset, peak params)
- [x] Compute agreement metrics: correlation, RMSE, Bland-Altman statistics
- [x] Statistical equivalence tests (e.g., TOST — two one-sided tests)
- [x] Sweep across parameter space: vary exponent, peak count, SNR, duration — record agreement at each point
- [x] Unit tests for comparison engine

### Phase 5: Interactive Web UI
- [x] Dashboard with controls for EEG generation parameters (sliders/inputs for exponent, peaks, SNR, duration, sample rate)
- [x] Real-time or on-demand signal generation from UI
- [x] PSD plot of generated signal with SpecParam fit overlay (spectral-domain)
- [x] Time-domain SpecParam results overlay on same or adjacent plot
- [x] Side-by-side parameter comparison table
- [x] Bland-Altman plot for batch equivalence runs
- [x] Heatmap or sweep plot: agreement metric across parameter space
- [x] Responsive layout, works on desktop

### Phase 6: Deployment
- [x] Dockerize the application (Dockerfile + docker-compose if needed)
- [x] Railway deployment config (railway.toml or Procfile)
- [x] Health check endpoint
- [x] Environment variable configuration for production
- [x] README with deployment instructions
- [ ] Deployed and accessible on Railway

## Environment
- Python 3.11+
- Key packages: `specparam` (FOOOF), `numpy`, `scipy`, `statsmodels`, `plotly`, `fastapi`, `uvicorn`
- Web framework: FastAPI + Plotly (decided)
- Deployment: Docker → Railway

### Phase 7: API Parity & Export
- [x] Thread `peak_width_limits` and `peak_threshold` through `fit_time_domain_specparam` → `_detect_peaks`
- [x] Thread `peak_width_limits` and `peak_threshold` through `fit_time_domain` wrapper
- [x] Populate `src/__init__.py` with public API exports (functions + schema types)
- [x] Verify: all tests pass, smoke test with custom peak params

### Phase 8: Restore Genuine ACF-Based Time-Domain Fitting
**Context:** The current time-domain fitter computes a Welch PSD and fits in log10-frequency
space — the same domain as spectral SpecParam. This defeats the project's purpose of
demonstrating equivalence between spectral and time-domain approaches. The original ACF code
failed because the observation window was too short (0.5–4 sec), not because the method is
fundamentally limited. With 30-second signals and max_lag=15 sec, the ACF has plenty of shape.

**Mathematical approach:** Compute model ACF numerically via IRFFT of the model PSD (works
for any chi, no closed-form needed). Fit by minimizing ||R_empirical - R_model||^2 over lags.

**8A: ACF math infrastructure**
- [x] `_compute_model_psd_linear(f_grid, b, k, chi, peaks)` — build linear PSD from params on rfft grid
- [x] `_model_acf_via_irfft(b, k, chi, peaks, sfreq, n_fft, max_lag_samples)` — model ACF via IRFFT
- [x] `_compute_empirical_acf(x, max_lag_samples)` — empirical autocovariance via FFT correlation
- [x] `tests/test_acf_math.py` — IRFFT scaling (R(0) == total power), AR(1) known ACF, chi=2 Lorentzian match, freq resolution

**8B: ACF-based fitting pipeline**
- [x] Stage 0: spectral initialization (reuse existing PSD code for initial guesses)
- [x] Stage 1: ACF aperiodic fit — PSD joint refit for refined chi, then ACF normalized fit
- [x] Stage 2: ACF full model fit — joint fit with chi regularized toward PSD-refined estimate
- [x] `method` parameter on `fit_time_domain_specparam`: "acf" (new) or "psd" (legacy default)
- [x] Adaptive max_lag_sec: `min(signal_length / sfreq / 2, 15.0)`
- [x] Output includes lags, empirical_acf, model_acf, acf_residual arrays
- [x] All 149 existing tests still pass via method="psd"
- [x] Fix `_model_acf_via_irfft` one-sided→two-sided PSD scaling (2x bug) and update Phase 8A tests

**8C: Update wrapper and schema**
- [x] `FitDiagnostics` dataclass in `src/schemas.py` (lags, empirical_acf, model_acf, acf_residual, fit_domain)
- [x] `diagnostics: FitDiagnostics | None` field on `SpecParamResult`
- [x] `src/time_domain_wrapper.py` threads method="acf" and populates diagnostics
- [x] `src/__init__.py` exports `FitDiagnostics`

**8D: ACF-specific tests**
- [x] `TestACFFitting`: convergence, exponent recovery at chi=[1.0, 1.5, 2.0], peak detection, R-squared on ACF
- [x] `TestACFResiduals`: diagnostics arrays present and correct shape
- [x] `TestLongLag`: max_lag=15 improves exponent recovery vs max_lag=2

**8E: Equivalence re-validation**
- [x] Full parameter sweep with method="acf": exponents [1.0, 1.25, 1.5, 1.75, 2.0] × [0, 1, 2 peaks]
- [x] RMSE thresholds: chi RMSE < 0.10 across full sweep (achieved 0.032)
- [x] All 15 parameter combinations converge, chi error < 0.06 everywhere

**8F: UI — ACF plot**
- [x] ACF plot panel in app.py (lag vs autocovariance): empirical (gray), model (red dashed)
- [x] ACF residual subplot (green)
- [x] Wire diagnostics data through `_run_simulation` to template

**8G: Switch default and clean up**
- [x] Switch default method to "acf" in all entry points
- [x] Full test suite passes (213 tests)
- [x] Update guardrails and progress files

## Current Focus
Phase 8 complete. All sub-phases (8A–8G) done. Next: Phase 6 deployment to Railway.
