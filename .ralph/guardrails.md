# Guardrails

Append-only. If something broke or wasted a loop, add a sign here.

## Domain Constraints (EEG / Signal Processing)

### 🚧 SIGN: Artifact-free means artifact-free by construction
Do NOT simulate artifacts (blinks, EMG, line noise) and then try to clean them.
Generate only: 1/f aperiodic background + band-limited oscillatory components + white noise floor.
The point of the simulation is controlled ground truth — artifacts defeat the purpose.

### 🚧 SIGN: Use physically plausible parameter ranges
- Aperiodic exponent: 1.0 to 2.0 (not 0.5, not 3.0)
- Alpha peak: 8–13 Hz center, bandwidth 1–3 Hz
- Sampling rate: 256 Hz or 512 Hz (not 1000+ unless explicitly requested)
- Signal duration: 2–60 seconds typical
- Amplitude units: µV (microvolts), typical scalp EEG range ~10–100 µV
If in doubt, check MNE or SpecParam docs for standard simulation parameters.

### 🚧 SIGN: SpecParam (FOOOF) expects power spectral density, not raw FFT
Always use Welch's method (or multitaper) to compute PSD before passing to SpecParam.
Frequency range for fitting: typically 1–40 Hz or 1–50 Hz. Do not fit 0 Hz (DC) or above Nyquist/2.

### 🚧 SIGN: The `specparam` package was renamed from `fooof`
Import as `from specparam import SpectralModel` (not `from fooof import FOOOF`).
pip install: `pip install specparam`

### 🚧 SIGN: Time-domain code is the STARTING POINT, not a black box
`src/timedomain_specparam.py` is the initial implementation to iterate on. It SHOULD be modified
to achieve parity with spectral SpecParam. However:
- Keep `src/timedomain_specparam_original.py` as an unmodified copy for reference/diffing
- Every modification must be validated against spectral SpecParam output (run comparison after each change)
- Log what was changed, why, and whether it improved agreement in `.ralph/activity.log`
- If a change worsens agreement, revert it and add a guardrail explaining why it failed

### 🚧 SIGN: Known limitations to address in the time-domain code
These are NOT constraints to work around — they are the development targets:
1. **Aperiodic exponent hardcoded at chi=2 (Lorentzian)** → ✅ DONE — free parameter
2. **Single periodic peak only** → ✅ DONE — N peaks with iterative detection
3. **No goodness-of-fit metric** → ✅ DONE — r_squared on log10 PSD residuals
4. **Fixed initial guesses (alpha-centric)** → ✅ DONE — spectral pre-analysis seeds p0
5. **BIDS I/O bundled in** → ✅ DONE — stripped, accepts numpy arrays directly

### 🚧 SIGN: Strip `process_bids_dataset` and `mne_bids` dependency early
The BIDS loader in the time-domain code is unnecessary for this project.
Remove it in Phase 3A to avoid carrying a heavy unused dependency.
The entry point is `fit_time_domain_specparam(time_series, sfreq)` with numpy arrays.

### 🚧 SIGN: `fit_time_domain_specparam` returns None on failure — handle it
When `curve_fit` fails to converge, the function returns `(None, tau, empirical_acf)`.
The wrapper MUST check for None and return a structured error/empty result, not crash.
Convergence failures should be logged and counted in sweep analyses.

### 🚧 SIGN: Time-domain ACF fitting needs unnormalized autocovariance
The code computes `acf(...) * np.var(time_series)` to get autocovariance, not normalized ACF.
If you preprocess the signal (e.g., z-score), you'll change the variance and break the fit.
Pass the signal as-is (in µV) to `fit_time_domain_specparam`.

### 🚧 SIGN: Default bounds in time-domain code may need adjustment for synthetic signals
Lower/upper bounds are: b∈[-5,5], k∈[0,50], a∈[0,3], c∈[1,40], w∈[0.5,5].
Initial guess centers on alpha (c=10 Hz). If generating signals with peaks outside these
ranges, the fit will clip or fail. The wrapper should allow overriding bounds/p0 if needed,
but the defaults are fine for typical alpha-band testing.

### 🚧 SIGN: Spectral SpecParam output is GROUND TRUTH for comparison
When developing the time-domain implementation, the target is: "does it produce the same
numbers as spectral SpecParam on the same signal?" The spectral version (official `specparam`
package) is the reference. If the two disagree, the time-domain code needs to change.

### 🚧 SIGN: Measure before and after every modification to time-domain code
Before changing `timedomain_specparam.py`:
1. Run both pipelines on a standard test signal set
2. Record agreement metrics (RMSE, correlation per parameter)
3. Make the change
4. Re-run and compare — did agreement improve?
5. If worse → revert, add guardrail about why. If better → commit, log in activity.log.
Never make multiple changes between measurements.

### 🚧 SIGN: Preserve mathematical rigor when generalizing the ACF model
The existing ACF derivations (aperiodic: Lorentzian, periodic: Taylor expansion of
exponentiated Gaussian) are analytically grounded. When generalizing:
- For arbitrary aperiodic exponents, derive or cite the correct ACF form — do not just
  "make it a parameter" in the Lorentzian formula without checking the math.
- For multi-peak, the periodic ACF is additive (sum of independent oscillatory ACFs)
  before convolution with aperiodic — verify this is implemented correctly.
- If an analytical form doesn't exist for a generalization, document the approximation
  used and its known limits.

### 🚧 SIGN: ACF fitting requires long observation windows for steep spectral slopes
**Discovered in Phase 3B, revised in Phase 8.** The original ACF approach failed for chi >= 1.5
because the observation window was too short (0.5–4 sec), not because the method is fundamentally
limited. With 30-second signals and max_lag=15 sec, even slowly-decaying ACFs (chi=2.0) show
their full shape. The original failures were:
1. Numerical cosine transform with fixed freq grid → scale mismatch
2. irfft with signal's freq grid → optimizer finds wrong local minimum at short lags
3. Normalized ACF fitting with max_lag=0.5–4 sec → insufficient shape for chi >= 1.5
**Key insight:** The issue was observation-window limited, not method-limited.
**Rule:** Use adaptive max_lag = min(signal_length / sfreq / 2, 15.0). For signals shorter than
~8 sec, ACF fitting for steep exponents may still be unreliable — fall back to PSD method.

### 🚧 SIGN: ACF of log-additive PSD model requires IRFFT, not convolution
**Discovered in Phase 8.** The original `timedomain_specparam_original.py` computed
`np.convolve(aperiodic_acf, periodic_acf)`. This is mathematically wrong for the SpecParam model.
The model is additive in log10-PSD: `log10(S) = aperiodic + peaks`, so in linear PSD:
`S = 10^(aperiodic + peaks)` — the peaks are **multiplicative**, not additive.
Convolution in ACF domain corresponds to multiplication in PSD domain (not exponentiation).
**Correct approach:** Build the full linear model PSD, then IRFFT to get the model ACF.
This is exact (no approximation) and works for any chi and any number of peaks.

### 🚧 SIGN: Use numerical IRFFT for model ACF, not closed-form
The original code used a closed-form Lorentzian ACF (only valid for chi=2). For arbitrary chi,
no simple closed-form exists. The numerical approach — IRFFT of the model PSD on a dense rfft
grid — is exact and general. **Must convert one-sided PSD → two-sided before irfft** (see sign
above). Verify scaling: `R_model(0)` must equal `integral_0^{f_nyq} S_one(f) df` = `df * (S[0]/2 + sum(S[1:-1]) + S[-1]/2)` using trapezoidal rule, OR equivalently the variance of a signal drawn from this PSD.

### 🚧 SIGN: Signal generation from model PSD requires correct amplitude scaling
**Discovered in Phase 8A.** When synthesizing a signal with a prescribed PSD S(f) via IRFFT
of random-phase spectrum, the rfft coefficient magnitudes must be:
`|X[k]| = sqrt(n_fft * sfreq * S[k])` where S is in V²/Hz.
Using `sqrt(S * sfreq / n_fft)` (off by factor n_fft) produces a signal with vanishingly
small variance. This matters for ACF consistency tests and future signal generation utilities.

### 🚧 SIGN: One-sided PSD must be converted to two-sided before np.fft.irfft
**Discovered in Phase 8B.** `np.fft.irfft(a, n)` treats its input as DFT coefficients for
the non-negative frequencies, and internally doubles non-DC/Nyquist bins during reconstruction
(because `X[k] = conj(X[n-k])` for real signals). If you pass a **one-sided PSD** (which is
already 2× the two-sided PSD for f > 0), the result is doubled.
**Symptom:** `R_model(0) ≈ 2 × Var(signal)` when it should equal `Var(signal)`.
**Fix:** Convert one-sided to two-sided before irfft: `S_two = S_one.copy(); S_two[1:-1] /= 2`.
Verified: flat S_one=1 gives irfft*sfreq = 256 (wrong 2×), S_two gives 128 (correct = f_nyq).
**Impact:** `_model_acf_via_irfft` and `_band_model_acf` in `_fit_acf_stages` both need this fix.
Phase 8A tests `test_r0_equals_total_power` used the same wrong formula, so they pass despite the bug.

### 🚧 SIGN: Band-limit both ACFs to freq_range to avoid sub-Hz mismatch
**Discovered in Phase 8B.** The signal generator uses a no-knee model (`b - chi*log10(f)`,
which diverges as f→0), but the fitter uses a knee model (`b - log10(k + f^chi)`, which
levels off). In the PSD domain this doesn't matter (fit range = [1, 40] Hz), but the
full-band ACF integrates over all frequencies including sub-Hz. Result: empirical ACF
decays ~2× slower than model ACF with correct parameters (rho_emp(0.5s) = 0.75 vs
rho_model(0.5s) = 0.49), pushing chi to upper bound.
**Fix:** Zero out frequencies outside `freq_range` in both `|X|^2` (empirical) and
`S_model` (model) before IRFFT. This makes both ACFs see the same frequency band.

### 🚧 SIGN: ACF chi estimation requires PSD-refined initialization
**Discovered in Phase 8B.** The SpecParam model is log-additive in PSD: `log10(S) = aperiodic + peaks`.
This makes aperiodic and periodic components cleanly separable in log-PSD domain. In the ACF domain,
the model becomes multiplicative: `S = 10^(aperiodic) * 10^(peaks)`. This breaks separability — the
ACF residual landscape has its minimum at a different chi than the true value when peaks are present.
**Symptom:** Joint ACF fit overestimates chi by 0.2–0.8 when peaks are present.
**Fix:** Run a joint PSD fit (aperiodic + peaks) before ACF fitting to get a refined chi_init.
Use strong regularization (weight=30) toward this PSD-refined chi in the ACF stage.
**Corollary:** Fitting only long lags (where peaks have decayed) doesn't work because
band-limited ACFs have insufficient information at long lags.

### 🚧 SIGN: Edge guards needed in peak detection
**Discovered in Phase 3C.** Iterative peak detection from the flattened residual can find
spurious peaks at the edges of the frequency range (e.g., 1.04 Hz, 39.9 Hz), which distort
the aperiodic fit. Solution: `edge_margin = 1.5` Hz — reject candidate peaks within 1.5 Hz
of the freq_range boundaries. This fixed exponent recovery for exp=1.0 (was 1.22, now ~1.0).

### 🚧 SIGN: specparam v2.0 API details
- Metric names: `'gof_rsquared'`, `'error_mae'` (not `'r_squared'`)
- Peak settings passed via `algorithm_settings` dict, not as direct kwargs
- `sm.add_data(freqs, psd, freq_range=list(freq_range))` then `sm.fit()`
- `sm.get_params('aperiodic')` → fixed mode: `[offset, exponent]`, knee mode: `[offset, knee, exponent]`
- `sm.get_params('peak')` → shape `(n_peaks, 3)`: `[center_freq, power, bandwidth]`

### 🚧 SIGN: specparam reports bandwidth as 2*sigma, not sigma
`sm.get_params('peak')` returns `[CF, PW, BW]` where **BW = 2 × standard deviation**.
Our `PeriodicPeak.bandwidth` schema stores sigma (std dev). So when extracting from
specparam, always divide by 2: `bandwidth = float(row[2]) / 2.0`.
The time-domain fitter uses sigma natively — no conversion needed there.
Forgetting this makes the spectral PSD overlay ~2x too wide and breaks bandwidth comparisons.

### 🚧 SIGN: DC component zeroing must be conditional
In `compute_target_psd()`, only zero out `psd[0]` when `freqs[0] == 0.0`.
Unconditionally zeroing `psd[0]` corrupts the lowest non-DC frequency bin if the
frequency array doesn't start at 0. Check `has_dc = freqs_safe[0] == 0.0` first.

### 🚧 SIGN: f-string formatting with conditionals
`f"{value:.4f if condition else 'N/A'}"` causes ValueError. Pre-compute the formatted
string: `formatted = f"{value:.4f}" if condition else "N/A"`, then use `{formatted}` in
the f-string.

## Technical Constraints

### 🚧 SIGN: Railway free tier has memory limits
Keep batch parameter sweeps bounded. Default sweep should complete in <60s and use <512MB.
Offer "quick" vs "full" sweep options in the UI.

### 🚧 SIGN: Plotly is heavy — use server-side rendering or lazy loading
If using Plotly for interactive plots, don't render 10 plots simultaneously on page load.
Generate on demand or use tabs/accordion.

### 🚧 SIGN: Pin dependency versions in requirements.txt
`specparam`, `numpy`, `scipy` version interactions matter. Pin them.

## Phase 9 Constraints (Overview Page & Simulator)

### 🚧 SIGN: MathJax LaTeX strings must NOT be in f-strings
LaTeX uses `{}` which conflicts with Python f-string interpolation. Define all math
content as module-level string constants (regular strings, not f-strings), then interpolate
them into the HTML f-string template: `{MATH_APERIODIC}`. Never write LaTeX directly
inside an f-string — it will either fail or silently eat braces.

### 🚧 SIGN: AJAX compute endpoint must mirror simulate_page params exactly
`GET /simulate/compute` must accept the same `Query()` parameters with the same defaults
as `simulate_page()`. If they diverge, the JS fetch will pass params the endpoint doesn't
expect, or vice versa. Extract param definitions to a shared pattern if possible.

### 🚧 SIGN: Range slider and number input must not both have `name` attribute
If both a `<input type="range">` and `<input type="number">` for the same parameter have
the same `name`, the form will submit duplicate query params. Only one input should carry
the `name` attribute; the other syncs via JS.

### 🚧 SIGN: Overview mini-sweep must be bounded
The live mini-sweep on the overview page runs on every page load. Keep it to ≤15 runs
(5 exponents × 3 peak configs) with duration=10s signals. Must complete in <15 seconds
and use <256MB. If it grows, add a loading indicator.

### 🚧 SIGN: Plotly.react requires div ID, not innerHTML replacement
For AJAX plot updates, use `Plotly.react(divId, data, layout)` — do not replace innerHTML
of the plot container. innerHTML replacement leaks Plotly's internal state and causes
memory issues on repeated updates. The div must have a stable ID.

### 🚧 SIGN: AbortController prevents stale AJAX responses
When the user changes a parameter while a previous fetch is in-flight, abort the old
request before starting the new one. Without this, a slow old request can overwrite
the results of a newer fast request, showing stale data.

### 🚧 SIGN: Range slider uses `data-sync` attribute, not `name`
Range sliders use `data-sync="param_name"` to link to the corresponding `<input type="number" name="param_name">`.
Only the number input carries the `name` attribute. JS syncs them via `input` events on both elements.
If you add a new slider-number pair, follow this pattern — giving both a `name` produces duplicate query params.

### 🚧 SIGN: Initial Plotly render must use Plotly.react with JSON, not to_html
When AJAX updates use `Plotly.react(divId, data, layout)`, the initial server-side render
must also use `Plotly.react` with JSON data (via `plotly.io.to_json`), not `plotly.io.to_html`.
Mixing approaches creates two Plotly instances in the same container — the first from `to_html`'s
inline script, the second from `Plotly.react`. Use `to_json` for initial data, render it via
`Plotly.react` in a `<script>` block, and reuse the same div ID for AJAX updates.

## Process Constraints

### 🚧 SIGN: Run tests after every change
`pytest tests/ -v` — if it fails, fix before moving on. Do not accumulate breakage.

### 🚧 SIGN: Do not skip PSD validation in Phase 1
The entire project depends on the synthetic signal being correct. If the PSD of the generated signal doesn't show clear 1/f + peaks, everything downstream is meaningless. Write the sanity check BEFORE moving to Phase 2.

### 🚧 SIGN: Keep spectral and time-domain result schemas identical
Define one dataclass/schema for SpecParam results. Both pipelines must output to it.
If the schemas diverge, the comparison engine breaks silently.
