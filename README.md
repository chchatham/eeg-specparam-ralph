# EEG SpecParam Equivalence Simulator

Interactive web simulation that generates synthetic artifact-free EEG data, runs both spectral-domain SpecParam (FOOOF) and a time-domain SpecParam implementation, and presents side-by-side equivalence comparison.

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run the app
python -m src.app
# Open http://localhost:8000
```

## Features

- **Single simulation** (`/`) — configure EEG parameters (exponent, peaks, duration, sample rate), generate a synthetic signal, and compare spectral vs. time-domain SpecParam fits with an interactive PSD plot and parameter comparison table.
- **Batch sweep** (`/sweep`) — run a grid of simulations across exponent x peak power space. Includes Bland-Altman plots, parameter heatmaps, RMSE metrics, and TOST equivalence tests. Quick (20 runs) and full (36 runs) modes available.

## Architecture

| File | Purpose |
|------|---------|
| `src/eeg_generator.py` | Synthetic EEG via Timmer & Koenig spectral synthesis |
| `src/spectral_specparam.py` | Wrapper around official `specparam` package (ground truth) |
| `src/timedomain_specparam.py` | Time-domain PSD-based fitting with free exponent, multi-peak |
| `src/time_domain_wrapper.py` | Maps time-domain output to shared `SpecParamResult` schema |
| `src/comparison.py` | Agreement metrics, RMSE, Bland-Altman stats, TOST equivalence |
| `src/schemas.py` | Shared dataclasses (`SpecParamResult`, `EEGSignal`, etc.) |
| `src/app.py` | FastAPI web application with Plotly visualizations |

## Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

123 tests covering signal generation, both fitting pipelines, the comparison engine, and a 60-test regression suite that validates equivalence across the full parameter space.

## Docker

```bash
docker build -t eeg-sim .
docker run -p 8000:8000 eeg-sim
```

## Deploy to Railway

```bash
# Install Railway CLI: https://docs.railway.app/guides/cli
railway login
railway up
```

The `railway.toml` configures the Dockerfile builder, health check at `/health`, and auto-restart on failure. The app listens on the `PORT` environment variable (default 8000).

## Dependencies

- Python 3.11+
- numpy, scipy, specparam (v2.0+), statsmodels, plotly, fastapi, uvicorn
