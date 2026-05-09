from __future__ import annotations

from functools import lru_cache

import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from scipy.signal import welch

from .schemas import AperiodicParams, PeriodicPeak
from .eeg_generator import compute_target_psd, generate_eeg_signal
from .spectral_specparam import fit_spectral_specparam
from .time_domain_wrapper import fit_time_domain
from .comparison import compare_results, compute_agreement_metrics, tost_equivalence

app = FastAPI(title="EEG SpecParam Equivalence Simulator")

# MathJax strings — module-level constants to avoid f-string {} conflicts (guardrail)
_MATH_APERIODIC = r"$$\log_{10}(\text{PSD}(f)) = b - \log_{10}(k + f^\chi)$$"
_MATH_PERIODIC = r"$$G(f) = a \cdot \exp\!\left(-\frac{(f - c)^2}{2w^2}\right)$$"
_MATH_FULL = r"$$S(f) = 10^{\,L(f)\;+\;\sum_{i=1}^{N} G_i(f)}$$"
_MATH_ACF = r"$$R[m] = \text{IFFT}\!\big(S_{\text{two-sided}}\big) \cdot f_s$$"
_MATH_FITTING = (
    r"$$\hat{\theta} = \arg\min_\theta "
    r"\sum_{m=0}^{M}\!\big(R_{\text{emp}}[m] - R_{\text{model}}[m;\,\theta]\big)^2$$"
)

_BASE_CSS = """\
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 20px; background: #f8f9fa; }}
h1 {{ color: #2c3e50; margin-bottom: 5px; }}
.subtitle {{ color: #7f8c8d; margin-bottom: 20px; }}
.plot {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
         margin-bottom: 20px; }}
table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px;
         overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 20px; }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }}
th {{ background: #2c3e50; color: white; }}
nav {{ margin-bottom: 20px; }}
nav a {{ margin-right: 16px; padding: 8px 16px; background: #3498db; color: white;
         text-decoration: none; border-radius: 4px; }}
nav a.active {{ background: #2c3e50; }}
nav a:hover {{ background: #2980b9; }}
.badge {{ display: inline-block; padding: 2px 10px; border-radius: 4px; color: white;
          font-weight: bold; font-size: 0.85em; }}"""


def _nav_html(active: str) -> str:
    items = [
        ("Overview", "/"),
        ("Simulator", "/simulate"),
        ("Quick Sweep", "/sweep?mode=quick"),
        ("Full Sweep", "/sweep?mode=full"),
    ]
    links = []
    for label, href in items:
        cls = ' class="active"' if label == active else ""
        links.append(f'<a href="{href}"{cls}>{label}</a>')
    return f'<nav>{"".join(links)}</nav>'


def _assemble_peaks(n_peaks, peak1_center, peak1_power, peak1_bw,
                    peak2_center, peak2_power, peak2_bw,
                    peak3_center, peak3_power, peak3_bw):
    centers = [peak1_center, peak2_center, peak3_center][:n_peaks]
    powers = [peak1_power, peak2_power, peak3_power][:n_peaks]
    bws = [peak1_bw, peak2_bw, peak3_bw][:n_peaks]
    return centers, powers, bws


def _peak_fields_html(n: int, center: float, power: float, bw: float, n_peaks: int) -> str:
    vis = "" if n_peaks >= n else "display:none"
    p = f"peak{n}"
    return (
        f'<div id="{p}-fields" style="{vis}">\n'
        f'<h4 style="margin: 12px 0 6px;">Peak {n}</h4>\n'
        f'<div class="grid">\n'
        f'  <div class="field"><label>Center freq (Hz)</label>\n'
        f'    <div class="range-row"><input type="range" min="1" max="50" step="0.5" value="{center}" data-sync="{p}_center">\n'
        f'    <input type="number" name="{p}_center" value="{center}" step="0.5" min="1" max="50"></div></div>\n'
        f'  <div class="field"><label>Power (log10)</label>\n'
        f'    <div class="range-row"><input type="range" min="0" max="3" step="0.1" value="{power}" data-sync="{p}_power">\n'
        f'    <input type="number" name="{p}_power" value="{power}" step="0.1" min="0" max="3"></div></div>\n'
        f'  <div class="field"><label>Bandwidth (Hz)</label>\n'
        f'    <div class="range-row"><input type="range" min="0.5" max="12" step="0.5" value="{bw}" data-sync="{p}_bw">\n'
        f'    <input type="number" name="{p}_bw" value="{bw}" step="0.5" min="0.5" max="12"></div></div>\n'
        f'</div>\n'
        f'</div>'
    )


def _fit_signal(signal):
    spectral = fit_spectral_specparam(signal)
    td = fit_time_domain(signal)
    comp = compare_results(spectral, td)
    return spectral, td, comp


@app.get("/health")
def health():
    return {"status": "ok"}


def _run_simulation(
    exponent: float,
    offset: float,
    peak_centers: list[float],
    peak_powers: list[float],
    peak_bws: list[float],
    sfreq: float,
    duration: float,
    seed: int,
):
    aperiodic = AperiodicParams(offset=offset, exponent=exponent)
    peaks = [
        PeriodicPeak(center_frequency=c, power=p, bandwidth=b)
        for c, p, b in zip(peak_centers, peak_powers, peak_bws)
    ]

    signal = generate_eeg_signal(
        aperiodic, peaks, sfreq=sfreq, duration=duration, random_seed=seed,
    )
    spectral, td, comp = _fit_signal(signal)
    freqs_psd, psd_vals = welch(signal.data[0], fs=sfreq, nperseg=min(len(signal.data[0]), int(sfreq * 2)))

    return spectral, td, comp, freqs_psd, psd_vals


def _build_psd_plot(freqs, psd, spectral, td):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=freqs, y=psd, mode="lines", name="Empirical PSD",
        line=dict(color="gray", width=1),
    ))

    mask = (freqs >= 1) & (freqs <= 40)
    f_fit = freqs[mask]

    spec_psd = compute_target_psd(f_fit, spectral.aperiodic, spectral.peaks)
    fig.add_trace(go.Scatter(
        x=f_fit, y=spec_psd, mode="lines", name="Spectral SpecParam",
        line=dict(color="blue", width=2),
    ))

    if td.converged:
        td_psd = compute_target_psd(f_fit, td.aperiodic, td.peaks)
        fig.add_trace(go.Scatter(
            x=f_fit, y=td_psd, mode="lines", name="Time-Domain SpecParam",
            line=dict(color="red", width=2, dash="dash"),
        ))

    fig.update_xaxes(type="log", title_text="Frequency (Hz)", range=[0, np.log10(50)])
    fig.update_yaxes(type="log", title_text="Power (µV²/Hz)")
    fig.update_layout(
        height=400, margin=dict(l=60, r=20, t=30, b=50),
        legend=dict(x=0.65, y=0.95),
    )
    return fig


def _build_acf_plot(td_acf):
    if not td_acf.converged or td_acf.diagnostics is None:
        return None

    d = td_acf.diagnostics
    max_plot_lag = min(len(d.lags), int(2.0 * (1.0 / d.lags[1])) if d.lags[1] > 0 else 512)
    lags = d.lags[:max_plot_lag]
    emp = d.empirical_acf[:max_plot_lag]
    model = d.model_acf[:max_plot_lag]
    resid = d.acf_residual[:max_plot_lag]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                        vertical_spacing=0.08)

    fig.add_trace(go.Scatter(
        x=lags, y=emp, mode="lines", name="Empirical ACF",
        line=dict(color="gray", width=1),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=lags, y=model, mode="lines", name="Model ACF",
        line=dict(color="red", width=2, dash="dash"),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=lags, y=resid, mode="lines", name="Residual",
        line=dict(color="#27ae60", width=1),
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

    fig.update_xaxes(title_text="Lag (sec)", row=2, col=1)
    fig.update_yaxes(title_text="Autocovariance", row=1, col=1)
    fig.update_yaxes(title_text="Residual", row=2, col=1)
    fig.update_layout(
        height=450, margin=dict(l=60, r=20, t=30, b=50),
        legend=dict(x=0.65, y=0.95),
    )
    return fig


def _build_comparison_table(spectral, td, comp):
    rows = []
    rows.append(f"<tr><td>Exponent</td><td>{spectral.aperiodic.exponent:.4f}</td>"
                f"<td>{td.aperiodic.exponent:.4f}</td><td>{comp.exponent_diff:.4f}</td></tr>")
    rows.append(f"<tr><td>Offset</td><td>{spectral.aperiodic.offset:.4f}</td>"
                f"<td>{td.aperiodic.offset:.4f}</td><td>{comp.offset_diff:.4f}</td></tr>")
    td_r2 = f"{td.r_squared:.4f}" if td.r_squared is not None else "N/A"
    sp_r2 = f"{spectral.r_squared:.4f}" if spectral.r_squared is not None else "N/A"
    rows.append(f"<tr><td>R²</td><td>{sp_r2}</td><td>{td_r2}</td><td></td></tr>")
    rows.append(f"<tr><td># Peaks</td><td>{len(spectral.peaks)}</td>"
                f"<td>{len(td.peaks)}</td><td>{'Match' if comp.peak_count_match else 'Differ'}</td></tr>")

    for i, (cd, pd, bd) in enumerate(zip(comp.peak_center_diffs, comp.peak_power_diffs, comp.peak_bw_diffs)):
        rows.append(f"<tr><td>Peak {i+1} center diff</td><td colspan='2'></td><td>{cd:.3f} Hz</td></tr>")
        rows.append(f"<tr><td>Peak {i+1} power diff</td><td colspan='2'></td><td>{pd:.4f}</td></tr>")

    return "\n".join(rows)


_OVERVIEW_CSS = """\
.section {{ background: white; padding: 24px 28px; border-radius: 8px;
           box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 24px; }}
.section h2 {{ color: #2c3e50; margin-top: 0; border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; }}
.math-block {{ background: #fdfdfe; border-left: 3px solid #3498db; padding: 12px 16px;
              margin: 12px 0; border-radius: 0 4px 4px 0; overflow-x: auto; }}
.pipeline {{ background: #f1f3f5; padding: 16px; border-radius: 6px; font-family: monospace;
             font-size: 0.9em; line-height: 1.8; white-space: pre; overflow-x: auto; }}
.example-cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
.card {{ background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 16px; }}
.card h4 {{ margin: 0 0 8px; color: #2c3e50; }}
.card p {{ font-size: 0.9em; color: #555; margin: 0 0 12px; }}
.card a {{ display: inline-block; padding: 6px 14px; background: #3498db; color: white;
           text-decoration: none; border-radius: 4px; font-size: 0.85em; }}
.card a:hover {{ background: #2980b9; }}
.metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }}
.metric-box {{ background: #f8f9fa; border-radius: 6px; padding: 12px; text-align: center; }}
.metric-box .value {{ font-size: 1.6em; font-weight: bold; color: #2c3e50; }}
.metric-box .label {{ font-size: 0.85em; color: #7f8c8d; }}"""


_mini_sweep_cache: list[dict] | None = None


def _run_mini_sweep():
    global _mini_sweep_cache
    if _mini_sweep_cache is not None:
        return _mini_sweep_cache

    exponents = [1.0, 1.25, 1.5, 1.75, 2.0]
    peak_configs = [
        [],
        [PeriodicPeak(center_frequency=10.0, power=0.8, bandwidth=2.0)],
        [PeriodicPeak(center_frequency=10.0, power=0.8, bandwidth=2.0),
         PeriodicPeak(center_frequency=25.0, power=0.6, bandwidth=1.5)],
    ]
    results = []
    for i, exp in enumerate(exponents):
        for j, peaks in enumerate(peak_configs):
            aperiodic = AperiodicParams(offset=1.5, exponent=exp)
            signal = generate_eeg_signal(
                aperiodic, peaks, sfreq=256.0, duration=10.0,
                random_seed=100 + i * 10 + j,
            )
            spectral, td, comp = _fit_signal(signal)
            results.append({"exponent": exp, "n_peaks": len(peaks),
                            "spectral": spectral, "td": td, "comp": comp})
    _mini_sweep_cache = results
    return results


@app.get("/", response_class=HTMLResponse)
def overview_page():
    sweep = _run_mini_sweep()
    comparisons = [r["comp"] for r in sweep]
    metrics = compute_agreement_metrics(comparisons)
    n_runs = len(sweep)
    converged = sum(1 for r in sweep if r["td"].converged)

    exp_diffs = np.array([c.exponent_diff for c in comparisons])
    tost_exp = tost_equivalence(exp_diffs, bound=0.2)
    tost_label = "EQUIVALENT" if tost_exp["equivalent"] else "NOT EQUIVALENT"
    tost_color = "#27ae60" if tost_exp["equivalent"] else "#e74c3c"
    tost_p = max(tost_exp["p_upper"], tost_exp["p_lower"])

    exp_rmse = f"{metrics.get('exponent_rmse', 0):.4f}"
    off_rmse = f"{metrics.get('offset_rmse', 0):.4f}"

    ba_fig = _build_bland_altman(sweep)
    ba_json = plotly.io.to_json(ba_fig)

    return f"""<!DOCTYPE html>
<html>
<head>
<title>EEG SpecParam Equivalence Simulator</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js" async></script>
<style>
{_BASE_CSS}
{_OVERVIEW_CSS}
</style>
</head>
<body>
<h1>EEG SpecParam Equivalence Simulator</h1>
<p class="subtitle">Demonstrating equivalence between spectral and time-domain SpecParam on synthetic EEG</p>
{_nav_html("Overview")}

<div class="section">
<h2>Introduction</h2>
<p>SpecParam (formerly FOOOF) is the standard algorithm for parameterizing neural power spectra
into aperiodic (1/f-like) and periodic (oscillatory) components. It operates in the frequency
domain by fitting a mixture model to the power spectral density (PSD).</p>
<p>This project develops and validates a <strong>time-domain</strong> equivalent that fits the
same parametric model to the autocorrelation function (ACF) instead of the PSD. Because the ACF
and PSD form a Fourier transform pair, they carry identical information &mdash; but the time-domain
approach opens the door to applications where spectral estimation is impractical (short epochs,
non-stationary data, real-time BCI).</p>
<p>The simulator below generates artifact-free synthetic EEG, runs both pipelines on it, and
presents side-by-side equivalence results.</p>
</div>

<div class="section">
<h2>Mathematical Background</h2>
<p><strong>Aperiodic component</strong> &mdash; modeled as a Lorentzian in log-log space with
offset <em>b</em>, knee <em>k</em>, and spectral exponent &chi;:</p>
<div class="math-block">{_MATH_APERIODIC}</div>

<p><strong>Periodic peaks</strong> &mdash; each oscillatory component is a Gaussian in
log-power space with amplitude <em>a</em>, center frequency <em>c</em>, and bandwidth <em>w</em>:</p>
<div class="math-block">{_MATH_PERIODIC}</div>

<p><strong>Full model</strong> &mdash; aperiodic and periodic are additive in log-PSD space,
so the linear PSD is their exponentiated sum:</p>
<div class="math-block">{_MATH_FULL}</div>

<p><strong>ACF via inverse FFT</strong> &mdash; the model autocovariance is computed numerically
by taking the inverse FFT of the two-sided model PSD:</p>
<div class="math-block">{_MATH_ACF}</div>

<p><strong>Time-domain fitting</strong> &mdash; the ACF-based fitter minimizes the sum of squared
residuals between empirical and model ACFs over lags 0 to <em>M</em>:</p>
<div class="math-block">{_MATH_FITTING}</div>

<h3>Three-Stage Fitting Pipeline</h3>
<ol>
<li><strong>Stage 0: Spectral initialization</strong> &mdash; quick PSD fit to seed initial parameter guesses</li>
<li><strong>Stage 1: PSD joint refit</strong> &mdash; refine &chi; estimate with peaks, then ACF normalized fit</li>
<li><strong>Stage 2: ACF full model</strong> &mdash; joint fit of all parameters with &chi; regularized toward PSD-refined estimate</li>
</ol>
</div>

<div class="section">
<h2>Architecture</h2>
<div class="pipeline">EEGSignal (numpy)
    &darr;
+---+-------------------+---+
|   Spectral Pipeline   |   Time-Domain Pipeline   |
|   Welch PSD &rarr; specparam  |   ACF via FFT &rarr; IRFFT fit   |
|   &darr;                  |   &darr;                      |
|   SpecParamResult     |   SpecParamResult         |
+---+-------------------+---+
    &darr;                      &darr;
    +--- ComparisonResult ---+
         (RMSE, TOST, Bland-Altman)</div>
<p style="margin-top:12px;"><strong>Key modules:</strong></p>
<table>
<tr><th>Module</th><th>Purpose</th></tr>
<tr><td><code>eeg_generator.py</code></td><td>Artifact-free synthetic EEG (1/f + Gaussian peaks)</td></tr>
<tr><td><code>spectral_specparam.py</code></td><td>Wrapper around official <code>specparam</code> package</td></tr>
<tr><td><code>timedomain_specparam.py</code></td><td>ACF-based and PSD-based time-domain fitters</td></tr>
<tr><td><code>time_domain_wrapper.py</code></td><td>Maps raw output to shared <code>SpecParamResult</code> schema</td></tr>
<tr><td><code>comparison.py</code></td><td>Agreement metrics, TOST equivalence tests</td></tr>
<tr><td><code>schemas.py</code></td><td>Shared dataclasses (<code>SpecParamResult</code>, <code>EEGSignal</code>)</td></tr>
</table>
</div>

<div class="section">
<h2>Live Equivalence Summary</h2>
<p>Mini-sweep: {n_runs} runs (5 exponents &times; 3 peak configs: 0, 1, 2 peaks).
Each signal is 10 sec at 256 Hz.</p>
<div class="metrics-grid">
  <div class="metric-box"><div class="value">{converged}/{n_runs}</div><div class="label">Converged</div></div>
  <div class="metric-box"><div class="value">{exp_rmse}</div><div class="label">Exponent RMSE</div></div>
  <div class="metric-box"><div class="value">{off_rmse}</div><div class="label">Offset RMSE</div></div>
  <div class="metric-box"><div class="value"><span class="badge" style="background:{tost_color}">{tost_label}</span></div>
    <div class="label">TOST (bound=0.2, p={tost_p:.4f})</div></div>
</div>
<div class="plot" style="margin-top:16px;">
<h3>Bland-Altman: Spectral vs Time-Domain</h3>
<div id="ba-plot"></div>
</div>
</div>

<div class="section">
<h2>Example Configurations</h2>
<p>Click any card to open the simulator with that configuration pre-loaded.</p>
<div class="example-cards">
  <div class="card">
    <h4>Clean 1/f (No Peaks)</h4>
    <p>Pure aperiodic signal with exponent 1.5. Tests baseline aperiodic recovery.</p>
    <a href="/simulate?exponent=1.5&offset=1.5&n_peaks=0&duration=30&sfreq=256&seed=42">Try in Simulator &rarr;</a>
  </div>
  <div class="card">
    <h4>Strong Alpha Rhythm</h4>
    <p>Prominent 10 Hz alpha peak over a 1/f background. The most common EEG signature.</p>
    <a href="/simulate?exponent=1.5&offset=1.5&n_peaks=1&peak1_center=10&peak1_power=1.2&peak1_bw=2.0&duration=30&sfreq=256&seed=42">Try in Simulator &rarr;</a>
  </div>
  <div class="card">
    <h4>Alpha + Beta</h4>
    <p>Two-peak model: alpha (10 Hz) and beta (25 Hz). Tests multi-peak detection.</p>
    <a href="/simulate?exponent=1.5&offset=1.5&n_peaks=2&peak1_center=10&peak1_power=0.8&peak1_bw=2.0&peak2_center=25&peak2_power=0.6&peak2_bw=1.5&duration=30&sfreq=256&seed=42">Try in Simulator &rarr;</a>
  </div>
  <div class="card">
    <h4>Steep Spectral Slope</h4>
    <p>Exponent 2.5 with alpha peak. Tests recovery at the upper edge of the exponent range.</p>
    <a href="/simulate?exponent=2.5&offset=1.5&n_peaks=1&peak1_center=10&peak1_power=0.8&peak1_bw=2.0&duration=30&sfreq=256&seed=42">Try in Simulator &rarr;</a>
  </div>
  <div class="card">
    <h4>Short Clinical Recording</h4>
    <p>Only 5 seconds of data. Tests robustness with limited observation windows.</p>
    <a href="/simulate?exponent=1.5&offset=1.5&n_peaks=1&peak1_center=10&peak1_power=0.8&peak1_bw=2.0&duration=5&sfreq=256&seed=42">Try in Simulator &rarr;</a>
  </div>
</div>
</div>

<div class="section">
<h2>Using the Simulator</h2>
<h3>Parameter Guide</h3>
<table>
<tr><th>Parameter</th><th>Range</th><th>Description</th></tr>
<tr><td>Exponent (&chi;)</td><td>0.5 &ndash; 3.0</td><td>Slope of the 1/f background. Typical EEG: 1.0&ndash;2.0.</td></tr>
<tr><td>Offset (b)</td><td>-3.0 &ndash; 5.0</td><td>Y-intercept of the aperiodic fit in log-log space.</td></tr>
<tr><td>Peak center</td><td>1 &ndash; 50 Hz</td><td>Center frequency of each oscillatory peak.</td></tr>
<tr><td>Peak power</td><td>0 &ndash; 3</td><td>Amplitude of the peak above the aperiodic floor (log10 units).</td></tr>
<tr><td>Peak bandwidth</td><td>0.5 &ndash; 12 Hz</td><td>Width of the Gaussian peak (standard deviation in Hz).</td></tr>
<tr><td>Duration</td><td>2 &ndash; 120 sec</td><td>Signal length. Longer = better ACF recovery for steep exponents.</td></tr>
<tr><td>Sample rate</td><td>128 &ndash; 1024 Hz</td><td>Sampling frequency. 256 Hz is standard for scalp EEG.</td></tr>
</table>
<h3>Interpreting the Plots</h3>
<ul>
<li><strong>PSD plot:</strong> Gray = empirical Welch PSD. Blue = spectral SpecParam fit. Red dashed = time-domain fit.
Good agreement means red and blue traces overlap.</li>
<li><strong>ACF plot:</strong> Gray = empirical autocovariance. Red dashed = model ACF from time-domain fit.
The residual (green) should be small and unstructured.</li>
</ul>
<p>For a full parameter-space analysis, try the <a href="/sweep?mode=quick">Quick Sweep</a>
or <a href="/sweep?mode=full">Full Sweep</a>.</p>
</div>

<script>
var baData = {ba_json};
Plotly.react("ba-plot", baData.data, baData.layout);
</script>
</body>
</html>"""


@app.get("/simulate", response_class=HTMLResponse)
def simulate_page(
    exponent: float = Query(1.5, ge=0.5, le=3.0),
    offset: float = Query(1.5, ge=-3.0, le=5.0),
    peak1_center: float = Query(10.0, ge=1, le=50),
    peak1_power: float = Query(0.8, ge=0, le=3),
    peak1_bw: float = Query(2.0, ge=0.5, le=12),
    n_peaks: int = Query(1, ge=0, le=3),
    peak2_center: float = Query(25.0, ge=1, le=50),
    peak2_power: float = Query(0.6, ge=0, le=3),
    peak2_bw: float = Query(1.5, ge=0.5, le=12),
    peak3_center: float = Query(40.0, ge=1, le=50),
    peak3_power: float = Query(0.4, ge=0, le=3),
    peak3_bw: float = Query(2.0, ge=0.5, le=12),
    sfreq: float = Query(256, ge=128, le=1024),
    duration: float = Query(30, ge=2, le=120),
    seed: int = Query(42, ge=0, le=99999),
):
    peak_centers, peak_powers, peak_bws = _assemble_peaks(
        n_peaks, peak1_center, peak1_power, peak1_bw,
        peak2_center, peak2_power, peak2_bw,
        peak3_center, peak3_power, peak3_bw,
    )

    spectral, td, comp, freqs, psd = _run_simulation(
        exponent, offset, peak_centers, peak_powers, peak_bws,
        sfreq, duration, seed,
    )

    psd_fig = _build_psd_plot(freqs, psd, spectral, td)
    acf_fig = _build_acf_plot(td)

    table_html = _build_comparison_table(spectral, td, comp)

    psd_json = plotly.io.to_json(psd_fig)
    acf_json = plotly.io.to_json(acf_fig) if acf_fig else "null"

    return f"""<!DOCTYPE html>
<html>
<head>
<title>EEG SpecParam Equivalence Simulator</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
{_BASE_CSS}
form {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 20px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }}
.field {{ display: flex; flex-direction: column; }}
.field label {{ font-size: 0.85em; color: #555; margin-bottom: 3px; }}
.field input[type="number"] {{ padding: 6px 8px; border: 1px solid #ddd; border-radius: 4px; width: 80px; }}
.field input[type="range"] {{ width: 100%; margin-top: 4px; }}
.range-row {{ display: flex; align-items: center; gap: 8px; }}
button {{ background: #3498db; color: white; border: none; padding: 10px 24px;
          border-radius: 4px; cursor: pointer; font-size: 1em; margin-top: 12px; }}
button:hover {{ background: #2980b9; }}
.plot-wrapper {{ position: relative; }}
.loading-overlay {{ position: absolute; inset: 0; background: rgba(255,255,255,0.7);
                    display: none; align-items: center; justify-content: center;
                    z-index: 10; border-radius: 8px; }}
.loading-overlay.active {{ display: flex; }}
.spinner {{ width: 36px; height: 36px; border: 4px solid #ddd; border-top-color: #3498db;
            border-radius: 50%; animation: spin 0.8s linear infinite; }}
@keyframes spin {{ to {{ transform: rotate(360deg); }} }}
</style>
</head>
<body>
<h1>EEG SpecParam Equivalence Simulator</h1>
<p class="subtitle">Compare spectral and time-domain SpecParam on synthetic EEG</p>

{_nav_html("Simulator")}

<form id="sim-form" method="get" action="/simulate">
<div class="grid">
  <div class="field"><label>Exponent (1/f slope)</label>
    <div class="range-row"><input type="range" min="0.5" max="3.0" step="0.1" value="{exponent}" data-sync="exponent">
    <input type="number" name="exponent" value="{exponent}" step="0.1" min="0.5" max="3.0"></div></div>
  <div class="field"><label>Offset (log power)</label>
    <div class="range-row"><input type="range" min="-3" max="5" step="0.5" value="{offset}" data-sync="offset">
    <input type="number" name="offset" value="{offset}" step="0.5" min="-3" max="5"></div></div>
  <div class="field"><label>Number of peaks</label>
    <div class="range-row"><input type="range" min="0" max="3" step="1" value="{n_peaks}" data-sync="n_peaks">
    <input type="number" name="n_peaks" value="{n_peaks}" min="0" max="3"></div></div>
  <div class="field"><label>Duration (sec)</label>
    <div class="range-row"><input type="range" min="2" max="120" step="1" value="{duration}" data-sync="duration">
    <input type="number" name="duration" value="{duration}" step="1" min="2" max="120"></div></div>
  <div class="field"><label>Sample rate (Hz)</label>
    <div class="range-row"><input type="range" min="128" max="1024" step="64" value="{sfreq}" data-sync="sfreq">
    <input type="number" name="sfreq" value="{sfreq}" step="64" min="128" max="1024"></div></div>
  <div class="field"><label>Seed</label>
    <input type="number" name="seed" value="{seed}" min="0" max="99999"></div>
</div>
{_peak_fields_html(1, peak1_center, peak1_power, peak1_bw, n_peaks)}
{_peak_fields_html(2, peak2_center, peak2_power, peak2_bw, n_peaks)}
{_peak_fields_html(3, peak3_center, peak3_power, peak3_bw, n_peaks)}
<button type="submit">Generate &amp; Compare</button>
</form>

<div class="plot plot-wrapper">
<div class="loading-overlay" id="loading"><div class="spinner"></div></div>
<h3>Power Spectral Density</h3>
<div id="psd-plot"></div>
</div>

<div class="plot plot-wrapper" id="acf-container" style="{"" if acf_fig else "display:none"}">
<h3>Autocovariance (ACF Fit)</h3>
<div id="acf-plot"></div>
</div>

<h3>Parameter Comparison</h3>
<table>
<tr><th>Parameter</th><th>Spectral</th><th>Time-Domain</th><th>Difference</th></tr>
<tbody id="comparison-table">{table_html}</tbody>
</table>

<script>
(function() {{
  var psdData = {psd_json};
  Plotly.react("psd-plot", psdData.data, psdData.layout);

  var acfData = {acf_json};
  if (acfData) Plotly.react("acf-plot", acfData.data, acfData.layout);

  var form = document.getElementById("sim-form");
  var loading = document.getElementById("loading");
  var controller = null;
  var timer = null;

  function showPeakFields() {{
    var n = parseInt(form.querySelector("[name=n_peaks]").value) || 0;
    document.getElementById("peak1-fields").style.display = n >= 1 ? "" : "none";
    document.getElementById("peak2-fields").style.display = n >= 2 ? "" : "none";
    document.getElementById("peak3-fields").style.display = n >= 3 ? "" : "none";
  }}

  form.querySelectorAll("input[type=range]").forEach(function(slider) {{
    var name = slider.getAttribute("data-sync");
    var num = form.querySelector("input[name=" + name + "]");
    slider.addEventListener("input", function() {{ num.value = slider.value; }});
    num.addEventListener("input", function() {{ slider.value = num.value; }});
  }});

  function getParams() {{
    var params = new URLSearchParams();
    form.querySelectorAll("input[name]").forEach(function(el) {{
      params.set(el.name, el.value);
    }});
    return params;
  }}

  function fetchResults() {{
    if (controller) controller.abort();
    controller = new AbortController();
    loading.classList.add("active");

    var params = getParams();
    history.replaceState(null, "", "/simulate?" + params.toString());

    fetch("/simulate/compute?" + params.toString(), {{ signal: controller.signal }})
      .then(function(r) {{ return r.json(); }})
      .then(function(data) {{
        loading.classList.remove("active");
        var psd = JSON.parse(data.psd_plot);
        Plotly.react("psd-plot", psd.data, psd.layout);

        var acfEl = document.getElementById("acf-container");
        if (data.acf_plot) {{
          var acf = JSON.parse(data.acf_plot);
          Plotly.react("acf-plot", acf.data, acf.layout);
          acfEl.style.display = "";
        }} else {{
          acfEl.style.display = "none";
        }}

        document.getElementById("comparison-table").innerHTML = data.table_html;
      }})
      .catch(function(e) {{
        if (e.name !== "AbortError") {{
          loading.classList.remove("active");
          console.error(e);
        }}
      }});
  }}

  form.addEventListener("input", function() {{
    showPeakFields();
    clearTimeout(timer);
    timer = setTimeout(fetchResults, 500);
  }});

  form.addEventListener("submit", function(e) {{
    e.preventDefault();
    clearTimeout(timer);
    fetchResults();
  }});
}})();
</script>
</body>
</html>"""


@app.get("/simulate/compute")
def simulate_compute(
    exponent: float = Query(1.5, ge=0.5, le=3.0),
    offset: float = Query(1.5, ge=-3.0, le=5.0),
    peak1_center: float = Query(10.0, ge=1, le=50),
    peak1_power: float = Query(0.8, ge=0, le=3),
    peak1_bw: float = Query(2.0, ge=0.5, le=12),
    n_peaks: int = Query(1, ge=0, le=3),
    peak2_center: float = Query(25.0, ge=1, le=50),
    peak2_power: float = Query(0.6, ge=0, le=3),
    peak2_bw: float = Query(1.5, ge=0.5, le=12),
    peak3_center: float = Query(40.0, ge=1, le=50),
    peak3_power: float = Query(0.4, ge=0, le=3),
    peak3_bw: float = Query(2.0, ge=0.5, le=12),
    sfreq: float = Query(256, ge=128, le=1024),
    duration: float = Query(30, ge=2, le=120),
    seed: int = Query(42, ge=0, le=99999),
):
    peak_centers, peak_powers, peak_bws = _assemble_peaks(
        n_peaks, peak1_center, peak1_power, peak1_bw,
        peak2_center, peak2_power, peak2_bw,
        peak3_center, peak3_power, peak3_bw,
    )

    spectral, td, comp, freqs, psd = _run_simulation(
        exponent, offset, peak_centers, peak_powers, peak_bws,
        sfreq, duration, seed,
    )

    psd_fig = _build_psd_plot(freqs, psd, spectral, td)
    acf_fig = _build_acf_plot(td)
    table_html = _build_comparison_table(spectral, td, comp)

    return JSONResponse({
        "psd_plot": plotly.io.to_json(psd_fig),
        "acf_plot": plotly.io.to_json(acf_fig) if acf_fig else None,
        "table_html": table_html,
        "converged": td.converged,
    })


@lru_cache(maxsize=8)
def _run_sweep(exponents: tuple[float, ...], peak_powers: tuple[float, ...], base_seed: int = 42):
    results = []
    for i, exp in enumerate(exponents):
        for j, pp in enumerate(peak_powers):
            aperiodic = AperiodicParams(offset=1.5, exponent=exp)
            peaks = []
            if pp > 0:
                peaks = [PeriodicPeak(center_frequency=10.0, power=pp, bandwidth=2.0)]

            signal = generate_eeg_signal(
                aperiodic, peaks, sfreq=256.0, duration=10.0,
                random_seed=base_seed + i * 100 + j,
            )
            spectral, td, comp = _fit_signal(signal)
            results.append({
                "exponent": exp,
                "peak_power": pp,
                "spectral": spectral,
                "td": td,
                "comp": comp,
            })
    return results


def _build_bland_altman(results: list[dict]):
    spec_exps = np.array([r["spectral"].aperiodic.exponent for r in results])
    td_exps = np.array([r["td"].aperiodic.exponent for r in results])
    spec_offs = np.array([r["spectral"].aperiodic.offset for r in results])
    td_offs = np.array([r["td"].aperiodic.offset for r in results])

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Exponent", "Offset"))

    for col, (spec_vals, td_vals, name) in enumerate([
        (spec_exps, td_exps, "Exponent"),
        (spec_offs, td_offs, "Offset"),
    ], start=1):
        means = (spec_vals + td_vals) / 2
        diffs = spec_vals - td_vals
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        upper_loa = mean_diff + 1.96 * std_diff
        lower_loa = mean_diff - 1.96 * std_diff

        colors = [r["exponent"] for r in results]

        fig.add_trace(go.Scatter(
            x=means, y=diffs, mode="markers",
            marker=dict(color=colors, colorscale="Viridis", size=8,
                        colorbar=dict(title="Exp") if col == 2 else dict(title="")),
            name=name,
            showlegend=False,
        ), row=1, col=col)

        x_range = [float(np.min(means)) - 0.1, float(np.max(means)) + 0.1]
        fig.add_trace(go.Scatter(
            x=x_range, y=[mean_diff, mean_diff], mode="lines",
            line=dict(color="red", dash="solid"), name="Mean diff",
            showlegend=(col == 1),
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=x_range, y=[upper_loa, upper_loa], mode="lines",
            line=dict(color="gray", dash="dash"), name="+1.96 SD",
            showlegend=(col == 1),
        ), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=x_range, y=[lower_loa, lower_loa], mode="lines",
            line=dict(color="gray", dash="dash"), name="-1.96 SD",
            showlegend=(col == 1),
        ), row=1, col=col)

        fig.update_xaxes(title_text=f"Mean {name}", row=1, col=col)
        fig.update_yaxes(title_text=f"Difference (Spec - TD)", row=1, col=col)

    fig.update_layout(height=400, margin=dict(l=60, r=20, t=40, b=50))
    return fig


def _build_sweep_heatmap(results: list[dict]):
    exponents = sorted(set(r["exponent"] for r in results))
    peak_powers = sorted(set(r["peak_power"] for r in results))

    exp_diff_grid = np.full((len(peak_powers), len(exponents)), np.nan)
    off_diff_grid = np.full((len(peak_powers), len(exponents)), np.nan)

    pp_idx = {p: i for i, p in enumerate(peak_powers)}
    exp_idx = {e: j for j, e in enumerate(exponents)}
    for r in results:
        i, j = pp_idx[r["peak_power"]], exp_idx[r["exponent"]]
        exp_diff_grid[i, j] = abs(r["comp"].exponent_diff)
        off_diff_grid[i, j] = abs(r["comp"].offset_diff)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Exponent |Diff|", "Offset |Diff|"),
        horizontal_spacing=0.15,
    )

    pp_labels = [f"{p:.1f}" for p in peak_powers]
    exp_labels = [f"{e:.2f}" for e in exponents]

    fig.add_trace(go.Heatmap(
        z=exp_diff_grid, x=exp_labels, y=pp_labels,
        colorscale="YlOrRd", zmin=0, zmax=0.3,
        colorbar=dict(title="|Diff|", x=0.42, len=0.9),
    ), row=1, col=1)

    fig.add_trace(go.Heatmap(
        z=off_diff_grid, x=exp_labels, y=pp_labels,
        colorscale="YlOrRd", zmin=0, zmax=0.5,
        colorbar=dict(title="|Diff|", x=1.0, len=0.9),
    ), row=1, col=2)

    fig.update_xaxes(title_text="Aperiodic Exponent", row=1, col=1)
    fig.update_xaxes(title_text="Aperiodic Exponent", row=1, col=2)
    fig.update_yaxes(title_text="Peak Power", row=1, col=1)
    fig.update_yaxes(title_text="Peak Power", row=1, col=2)
    fig.update_layout(height=400, margin=dict(l=60, r=20, t=40, b=50))
    return fig


@app.get("/sweep", response_class=HTMLResponse)
def sweep_page(
    mode: str = Query("quick", pattern="^(quick|full)$"),
    seed: int = Query(42, ge=0, le=9999),
):
    if mode == "full":
        exponents = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0)
        peak_powers = (0.0, 0.3, 0.6, 0.9, 1.2, 1.5)
    else:
        exponents = (1.0, 1.25, 1.5, 1.75, 2.0)
        peak_powers = (0.0, 0.4, 0.8, 1.2)

    results = _run_sweep(exponents, peak_powers, base_seed=seed)
    comparisons = [r["comp"] for r in results]
    metrics = compute_agreement_metrics(comparisons)

    exp_diffs = np.array([c.exponent_diff for c in comparisons])
    off_diffs = np.array([c.offset_diff for c in comparisons])
    tost_exp = tost_equivalence(exp_diffs, bound=0.2)
    tost_off = tost_equivalence(off_diffs, bound=0.3)

    ba_fig = _build_bland_altman(results)
    ba_html = plotly.io.to_html(ba_fig, full_html=False, include_plotlyjs="cdn")

    hm_fig = _build_sweep_heatmap(results)
    hm_html = plotly.io.to_html(hm_fig, full_html=False, include_plotlyjs=False)

    n_runs = len(results)
    converged = sum(1 for r in results if r["td"].converged)

    exp_rmse = f"{metrics.get('exponent_rmse', 0):.4f}"
    off_rmse = f"{metrics.get('offset_rmse', 0):.4f}"
    exp_bias = f"{metrics.get('exponent_bias', 0):.4f}"
    off_bias = f"{metrics.get('offset_bias', 0):.4f}"
    pk_rmse = f"{metrics.get('peak_center_rmse', 0):.3f}" if metrics.get('peak_center_rmse') is not None else "N/A"

    tost_exp_str = "EQUIVALENT" if tost_exp["equivalent"] else "NOT equivalent"
    tost_off_str = "EQUIVALENT" if tost_off["equivalent"] else "NOT equivalent"
    tost_exp_color = "#27ae60" if tost_exp["equivalent"] else "#e74c3c"
    tost_off_color = "#27ae60" if tost_off["equivalent"] else "#e74c3c"

    return f"""<!DOCTYPE html>
<html>
<head>
<title>EEG SpecParam Equivalence Simulator</title>
<style>
{_BASE_CSS}
a {{ color: #3498db; }}
</style>
</head>
<body>
<h1>EEG SpecParam Equivalence Simulator</h1>
<p class="subtitle">Bland-Altman analysis and parameter sweep across exponent x peak power</p>

{_nav_html("Quick Sweep" if mode == "quick" else "Full Sweep")}

<h3>Summary</h3>
<table>
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Sweep grid</td><td>{len(exponents)} exponents x {len(peak_powers)} peak powers = {n_runs} runs</td></tr>
<tr><td>Convergence</td><td>{converged}/{n_runs} ({100*converged/n_runs:.0f}%)</td></tr>
<tr><td>Exponent RMSE</td><td>{exp_rmse}</td></tr>
<tr><td>Exponent bias</td><td>{exp_bias}</td></tr>
<tr><td>Offset RMSE</td><td>{off_rmse}</td></tr>
<tr><td>Offset bias</td><td>{off_bias}</td></tr>
<tr><td>Peak center RMSE</td><td>{pk_rmse}</td></tr>
<tr><td>TOST exponent (bound=0.2)</td>
    <td><span class="badge" style="background:{tost_exp_color}">{tost_exp_str}</span>
    p = {max(tost_exp['p_upper'], tost_exp['p_lower']):.4f}</td></tr>
<tr><td>TOST offset (bound=0.3)</td>
    <td><span class="badge" style="background:{tost_off_color}">{tost_off_str}</span>
    p = {max(tost_off['p_upper'], tost_off['p_lower']):.4f}</td></tr>
</table>

<div class="plot">
<h3>Bland-Altman Plots</h3>
{ba_html}
</div>

<div class="plot">
<h3>Parameter Space Heatmap</h3>
{hm_html}
</div>

</body>
</html>"""


def main():
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
