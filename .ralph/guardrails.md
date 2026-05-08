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

### 🚧 SIGN: ACF-based fitting is fundamentally limited for steep spectral slopes
**Discovered in Phase 3B.** The original plan was to fit the time-domain model via ACF matching.
This works for chi=1.0 but FAILS for chi >= 1.5 because the normalized ACF barely decays
within the observation window (0.53 at 4 seconds for chi=2.0), giving insufficient shape info.
Multiple ACF approaches were tried and abandoned:
1. Numerical cosine transform with fixed freq grid → scale mismatch
2. irfft with signal's freq grid → optimizer finds wrong local minimum
3. Normalized ACF fitting → works for chi=1.0, fails for chi >= 1.5
**Solution:** Switched to PSD-based fitting in log10 space. This matches spectral SpecParam
within 0.015 for exponent across the full sweep. Do NOT revisit ACF-based approaches.

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

## Process Constraints

### 🚧 SIGN: Run tests after every change
`pytest tests/ -v` — if it fails, fix before moving on. Do not accumulate breakage.

### 🚧 SIGN: Do not skip PSD validation in Phase 1
The entire project depends on the synthetic signal being correct. If the PSD of the generated signal doesn't show clear 1/f + peaks, everything downstream is meaningless. Write the sanity check BEFORE moving to Phase 2.

### 🚧 SIGN: Keep spectral and time-domain result schemas identical
Define one dataclass/schema for SpecParam results. Both pipelines must output to it.
If the schemas diverge, the comparison engine breaks silently.
