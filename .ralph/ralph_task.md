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

## Current Focus
Phase 7 complete. Only remaining item: deploy to Railway.
