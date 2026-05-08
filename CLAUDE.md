# EEG SpecParam Equivalence Simulator

## Session Recovery (READ THIS FIRST)
If you're starting a new session, recovering from compaction, or running in a Ralph loop:
1. Read `.ralph/ralph_task.md` — the anchor. Has all checkboxes. Defines "done."
2. Read `.ralph/progress.md` — what exists, what's broken, what to do next.
3. Read `.ralph/guardrails.md` — learned constraints. Follow every sign.
4. Do NOT re-read the full codebase unless progress.md says something is broken. Trust the files.
5. Pick up the "Current Focus" from progress.md and work on it.
6. Before exiting or if context feels heavy: update progress.md with what you did and what's next.

## Compaction Instructions
When compacting this conversation, preserve:
- Current task and its completion state
- Any new guardrails discovered this session
- Any new known issues
- The exact next step to take
Do NOT preserve: file contents already read, full API/command outputs, failed approaches
(log failures to .ralph/errors.log instead).

## Project Purpose
Interactive web simulation that generates realistic artifact-free synthetic EEG data,
runs both spectral-domain SpecParam (FOOOF) and a novel time-domain SpecParam on it,
and presents side-by-side equivalence comparison. Deployed via Railway.

## Architecture
- `src/eeg_generator.py` — synthetic EEG signal generation (1/f + oscillatory components)
- `src/spectral_specparam.py` — wrapper around official `specparam` package (PSD → fit)
- `src/timedomain_specparam.py` — time-domain ACF model and fitter (**actively developed — this is the code being optimized**)
- `src/timedomain_specparam_original.py` — unmodified copy of the initial implementation (for reference/diffing)
- `src/time_domain_wrapper.py` — project wrapper that calls timedomain_specparam and maps output to shared schema
- `src/comparison.py` — equivalence metrics: correlation, RMSE, Bland-Altman, TOST
- `src/schemas.py` — shared dataclasses for SpecParam results (both pipelines output to this)
- `src/app.py` — web application (FastAPI/Dash TBD)
- `tests/` — pytest suite
- `Dockerfile` + `railway.toml` — deployment config

### Time-Domain Code (`src/timedomain_specparam.py`) — Starting State
The initial implementation has these functions:
- `generate_aperiodic_acf(tau, b, k)` — ACF for aperiodic component (chi=2 Lorentzian ONLY — to be generalized)
- `generate_periodic_acf(tau, a, c, w, M=5)` — ACF for single periodic peak via Taylor expansion (to be extended to N peaks)
- `time_domain_specparam_model(tau, b, k, a, c, w)` — full model: convolve aperiodic × periodic ACF
- `fit_time_domain_specparam(time_series, sfreq, max_lag_sec)` — compute empirical ACF, fit model via curve_fit
- `process_bids_dataset(...)` — BIDS I/O (TO BE REMOVED — not needed)

Current fitted params: `[b, k, a, c, w]` — offset, knee, peak_amp, center_freq, bandwidth
Target fitted params: `[b, k, chi, a1, c1, w1, ..., aN, cN, wN]` — with free exponent and N peaks

## Key Schemas / Interfaces

### SpecParamResult (shared output schema — both pipelines must produce this)
```python
@dataclass
class AperiodicParams:
    offset: float        # y-intercept of 1/f fit (log-log space)
    exponent: float      # slope of 1/f fit. Both pipelines must output this as a free parameter.
    knee: float | None   # knee frequency. Spectral: optional. Time-domain: param `k`.

@dataclass
class PeriodicPeak:
    center_frequency: float   # Hz
    power: float              # amplitude above aperiodic
    bandwidth: float          # Hz (std dev of Gaussian)

@dataclass
class SpecParamResult:
    aperiodic: AperiodicParams
    peaks: list[PeriodicPeak]   # Both pipelines must support 0+ peaks
    r_squared: float | None     # goodness of fit
    frequency_range: tuple[float, float]  # Hz range used for fitting
    method: str                 # "spectral" or "time_domain"
    converged: bool             # True if fit succeeded
```

### Parameter Mapping (spectral ↔ time-domain) — CURRENT vs TARGET
```
Spectral SpecParam          Time-Domain (CURRENT)        Time-Domain (TARGET)
─────────────────           ─────────────────────        ────────────────────
offset                      b (offset)                   b (offset) — direct
exponent (free, ~1-2)       FIXED at 2.0                 chi (free) — TO IMPLEMENT
knee (optional)             k (knee freq)                k (knee freq)
N peaks (0+)                1 peak only                  N peaks — TO IMPLEMENT
  peak center_freq          c                            c1, c2, ... cN
  peak power                a                            a1, a2, ... aN
  peak bandwidth            w                            w1, w2, ... wN
r_squared                   not computed                 TO IMPLEMENT
```

### EEGSignal (generator output)
```python
@dataclass
class EEGSignal:
    data: np.ndarray           # shape (n_channels, n_samples), units µV
    sfreq: float               # sampling rate Hz
    duration: float            # seconds
    ground_truth_aperiodic: AperiodicParams
    ground_truth_peaks: list[PeriodicPeak]
```

## Environment
- Python 3.11+
- Key deps: `specparam`, `numpy`, `scipy`, `plotly` (or `dash`), `fastapi`, `uvicorn`
- Optional: `mne` for PSD utilities
- Test: `pytest tests/ -v`
- Docker: `docker build -t eeg-sim . && docker run -p 8000:8000 eeg-sim`

## Design Principles
1. Artifact-free by construction — never generate-then-clean
2. Official `specparam` package for spectral-domain — no reimplementation (this is ground truth)
3. Time-domain code is actively developed — modify it to achieve parity with spectral SpecParam
4. Every modification to time-domain code must be measured: run comparison before and after
5. Single shared schema for both pipeline outputs — divergence = bug
6. Ground truth params travel with the signal — always available for validation
7. All comparison must include formal equivalence tests, not just correlation
8. Keep `timedomain_specparam_original.py` as unmodified reference — never edit the original copy
