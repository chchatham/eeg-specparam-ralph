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
:root {
  --primary: #2c3e50;
  --primary-light: #34495e;
  --accent: #3498db;
  --accent-hover: #2980b9;
  --success: #27ae60;
  --success-hover: #219a52;
  --warning: #f39c12;
  --danger: #e74c3c;
  --bg: #f4f6f9;
  --card-bg: #ffffff;
  --text: #343a40;
  --text-muted: #6c757d;
  --text-light: #95a5a6;
  --border: #dee2e6;
  --border-light: #ecf0f1;
  --well-bg: #f8f9fa;
  --shadow: 0 1px 3px rgba(0,0,0,0.1);
}
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 20px; background: var(--bg);
       color: var(--text); font-size: 0.9rem; line-height: 1.5; }
h1 { color: var(--primary); margin-bottom: 5px; font-size: 1.75rem; }
h2 { font-size: 1.35rem; color: var(--primary); }
h3 { font-size: 1.1rem; color: var(--primary); }
.subtitle { color: var(--text-muted); margin-bottom: 20px; }
code { font-family: 'SF Mono', 'Fira Code', Consolas, monospace; font-size: 0.88em; }
.plot { background: var(--card-bg); padding: 15px; border-radius: 8px; box-shadow: var(--shadow);
         margin-bottom: 20px; }
table { width: 100%; border-collapse: collapse; background: var(--card-bg); border-radius: 8px;
         overflow: hidden; box-shadow: var(--shadow); margin-bottom: 20px; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--border-light); }
th { background: var(--primary); color: white; }
.navbar { background: var(--primary); border-radius: 6px; margin-bottom: 20px; overflow: hidden; }
.nav-links { display: flex; }
.nav-toggle { display: none; background: none; border: none; color: rgba(255,255,255,0.8);
               padding: 10px 16px; cursor: pointer; font-size: 0.9em; width: 100%; text-align: left; }
.nav-toggle:hover { color: white; }
.navbar a { padding: 10px 18px; color: rgba(255,255,255,0.7); text-decoration: none;
             font-size: 0.9em; font-weight: 500; border-bottom: 3px solid transparent;
             transition: color 0.2s, border-color 0.2s; white-space: nowrap; }
.navbar a:hover { color: white; background: rgba(255,255,255,0.05); }
.navbar a.active { color: white; border-bottom-color: var(--accent); }
@media (max-width: 600px) {
  .nav-toggle { display: block; }
  .nav-links { display: none; flex-direction: column; }
  .navbar.expanded .nav-links { display: flex; }
  .navbar a { border-bottom: none; border-left: 3px solid transparent; }
  .navbar a.active { border-left-color: #3498db; }
}
.badge { display: inline-block; padding: 2px 10px; border-radius: 4px; color: white;
          font-weight: bold; font-size: 0.85em; }
.card { background: var(--card-bg); border-radius: 8px; box-shadow: var(--shadow);
         margin-bottom: 20px; overflow: hidden; }
.card-header { display: flex; justify-content: space-between; align-items: center;
                padding: 10px 16px; font-weight: 600; font-size: 0.95em;
                border-bottom: 1px solid var(--border-light); color: var(--primary); background: var(--well-bg); }
.card-body { padding: 16px; }"""


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
    return (
        '<nav class="navbar">'
        '<button class="nav-toggle" onclick="this.parentElement.classList.toggle(\'expanded\')" '
        'type="button">&#9776; Menu</button>'
        f'<div class="nav-links">{"".join(links)}</div>'
        '</nav>'
    )


def _assemble_peaks(n_peaks, peak1_center, peak1_power, peak1_bw,
                    peak2_center, peak2_power, peak2_bw,
                    peak3_center, peak3_power, peak3_bw):
    centers = [peak1_center, peak2_center, peak3_center][:n_peaks]
    powers = [peak1_power, peak2_power, peak3_power][:n_peaks]
    bws = [peak1_bw, peak2_bw, peak3_bw][:n_peaks]
    return centers, powers, bws


def _peak_fields_html(n: int, center: float, power: float, bw: float, n_peaks: int) -> str:
    cls = "peak-group visible" if n_peaks >= n else "peak-group"
    p = f"peak{n}"
    return (
        f'<div id="{p}-fields" class="{cls}">\n'
        f'<h4 style="margin: 12px 0 6px; font-size: 0.88em; color: #34495e;">Peak {n}</h4>\n'
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
.hero { background: linear-gradient(135deg, #2c3e50 0%, #34495e 50%, #2c3e50 100%);
         color: white; padding: 48px 32px; border-radius: 8px; text-align: center;
         margin-bottom: 24px; }
.hero h1 { color: white; font-size: 2rem; margin-bottom: 12px; border: none; }
.hero p { font-size: 1.1em; opacity: 0.85; max-width: 700px; margin: 0 auto 20px; line-height: 1.5; }
.cta-btn { display: inline-block; padding: 12px 28px; background: #27ae60; color: white;
            border-radius: 6px; text-decoration: none; font-weight: 600; font-size: 1.05em;
            transition: background 0.2s; }
.cta-btn:hover { background: #219a52; }
.theorem-box { background: #fafbfc; border: 1px solid #e0e4e8; border-radius: 8px;
                padding: 14px 20px; margin: 12px 0; text-align: center; }
.theorem-box .th-label { font-size: 0.75em; font-weight: 600; color: #95a5a6;
                          text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
.flow-diagram { text-align: center; padding: 16px 0; }
.flow-node { display: inline-block; padding: 10px 20px; border-radius: 6px;
              font-weight: 600; font-size: 0.9em; }
.flow-input { background: #2c3e50; color: white; }
.flow-spectral { background: #3498db; color: white; }
.flow-timedomain { background: #e74c3c; color: white; }
.flow-result { background: #ecf0f1; color: #2c3e50; border: 1px solid #bdc3c7; }
.flow-comparison { background: #27ae60; color: white; }
.flow-arrow { font-size: 1.4em; color: #95a5a6; margin: 6px 0; }
.flow-split { display: flex; justify-content: center; gap: 32px; margin: 8px 0; }
.flow-branch { display: flex; flex-direction: column; align-items: center; }
.flow-detail { font-size: 0.82em; color: #7f8c8d; margin: 4px 0; }
.example-cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }
.example-card { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 16px;
                 transition: transform 0.2s, box-shadow 0.2s; }
.example-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
.example-card h4 { margin: 0 0 6px; color: #2c3e50; }
.example-card .params { font-size: 0.78em; color: #7f8c8d; font-family: monospace; margin-bottom: 8px; }
.example-card p { font-size: 0.9em; color: #555; margin: 0 0 12px; }
.example-card a { display: inline-block; padding: 6px 14px; background: #3498db; color: white;
                   text-decoration: none; border-radius: 4px; font-size: 0.85em; transition: background 0.2s; }
.example-card a:hover { background: #2980b9; }
.metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
.metric-box { background: #f8f9fa; border-radius: 6px; padding: 16px; text-align: center;
               border: 1px solid #ecf0f1; }
.metric-box .value { font-size: 1.8em; font-weight: bold; color: #2c3e50; }
.metric-box .label { font-size: 0.82em; color: #7f8c8d; margin-top: 4px; }"""

_SIMULATOR_CSS = """\
body.sim-page { max-width: none; margin: 0; padding: 0; background: var(--bg); }
.sidebar { width: 300px; position: fixed; top: 0; bottom: 0; left: 0; overflow-y: auto;
            background: var(--card-bg); border-right: 1px solid var(--border); padding: 0;
            box-shadow: 2px 0 8px rgba(0,0,0,0.06); z-index: 100;
            transition: transform 0.3s ease; }
.sidebar-header { display: flex; justify-content: space-between; align-items: center;
                   padding: 14px 16px; border-bottom: 1px solid var(--border-light);
                   background: var(--primary); color: white; }
.sidebar-header h2 { margin: 0; font-size: 1.05rem; color: white; font-weight: 600; }
.sidebar-close { display: none; background: none; border: none; font-size: 1.4rem;
                  cursor: pointer; color: rgba(255,255,255,0.7); line-height: 1; padding: 0; }
.sidebar-close:hover { color: white; }
.sidebar-body { padding: 12px 16px 80px; }
.main-panel { margin-left: 300px; padding: 20px 28px; min-height: 100vh; }
.top-bar { display: flex; align-items: center; gap: 12px; margin-bottom: 4px; }
.sidebar-toggle { display: none; background: var(--primary); color: white; border: none;
                   font-size: 1.1rem; padding: 8px 11px; border-radius: 4px; cursor: pointer; }
.sidebar-toggle:hover { background: var(--primary-light); }
details { margin-bottom: 4px; border-bottom: 1px solid #f0f0f0; }
details[open] { margin-bottom: 8px; }
details summary { font-weight: 600; color: var(--primary); cursor: pointer; padding: 8px 0;
                   font-size: 0.9em; list-style: none; }
details summary::-webkit-details-marker { display: none; }
details summary::before { content: "\25B8  "; color: var(--text-light); font-size: 0.85em; }
details[open] > summary::before { content: "\25BE  "; }
details summary:hover { color: var(--accent); }
.well { background: var(--well-bg); border: 1px solid var(--border-light); border-radius: 6px;
         padding: 12px; margin: 6px 0 10px; }
.grid { display: flex; flex-direction: column; gap: 8px; }
.field { display: flex; flex-direction: column; margin-bottom: 10px; }
.field label { font-size: 0.82em; color: #555; margin-bottom: 3px; font-weight: 500; }
.field input[type="number"] { padding: 5px 7px; border: 1px solid var(--border); border-radius: 4px;
                               width: 72px; font-size: 0.9em; }
.field input[type="range"] { width: 100%; margin: 4px 0 0; accent-color: var(--accent); }
.range-row { display: flex; align-items: center; gap: 8px; }
.run-btn { width: 100%; background: var(--success); color: white; border: none; padding: 12px;
            border-radius: 6px; cursor: pointer; font-size: 1em; font-weight: 600;
            margin-top: 12px; }
.run-btn:hover { background: var(--success-hover); }
input[type="range"] { -webkit-appearance: none; appearance: none;
                       height: 6px; background: #e0e4e8; border-radius: 3px; outline: none; }
input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; width: 18px; height: 18px;
    background: var(--accent); border-radius: 50%; cursor: pointer; border: 2px solid white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2); margin-top: -6px; }
input[type="range"]::-moz-range-track { height: 6px; background: #e0e4e8;
                                         border-radius: 3px; border: none; }
input[type="range"]::-moz-range-thumb { width: 14px; height: 14px; background: var(--accent);
    border-radius: 50%; cursor: pointer; border: 2px solid white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2); }
.help-text { font-size: 0.73em; color: var(--text-light); margin-top: 2px; line-height: 1.3; }
.styled-select { width: 100%; padding: 6px 8px; border: 1px solid var(--border); border-radius: 4px;
                  font-size: 0.9em; background: white; color: var(--primary); cursor: pointer; }
.styled-select:focus { border-color: var(--accent); outline: none; box-shadow: 0 0 0 2px rgba(52,152,219,0.15); }
.peak-group { overflow: hidden; max-height: 0; opacity: 0;
               transition: max-height 0.3s ease, opacity 0.2s ease; }
.peak-group.visible { max-height: 300px; opacity: 1; }
.plot-wrapper { position: relative; }
.loading-overlay { position: absolute; inset: 0; background: rgba(255,255,255,0.85);
                    display: none; align-items: center; justify-content: center;
                    z-index: 10; border-radius: 8px; padding: 20px; }
.loading-overlay.active { display: flex; }
.skeleton-bar { width: 90%; height: 250px; border-radius: 6px;
                 background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
                 background-size: 200% 100%; animation: shimmer 1.5s infinite; }
@keyframes shimmer { 0% { background-position: 200% 0; } 100% { background-position: -200% 0; } }
.expand-btn { background: none; border: none; color: var(--text-muted); cursor: pointer;
               font-size: 1.1em; padding: 2px 6px; border-radius: 4px; line-height: 1; }
.expand-btn:hover { background: rgba(0,0,0,0.06); color: var(--primary); }
.card.fullscreen { position: fixed; inset: 0; z-index: 1000; border-radius: 0;
                    margin: 0; overflow-y: auto; }
.sweep-controls { display: flex; gap: 12px; align-items: center; margin-bottom: 16px; }
.sweep-controls label { font-size: 0.85em; color: var(--text-muted); font-weight: 500; }
.sweep-controls select { padding: 6px 10px; border: 1px solid var(--border); border-radius: 4px;
                          font-size: 0.9em; background: white; cursor: pointer; }
@media (max-width: 768px) {
  .sidebar { transform: translateX(-100%); }
  .sidebar.open { transform: translateX(0); }
  .sidebar-close { display: block; }
  .sidebar-toggle { display: block; }
  .main-panel { margin-left: 0; }
}
@media print {
  .sidebar, .navbar, .nav-toggle, .sidebar-toggle, .run-btn { display: none !important; }
  .main-panel { margin-left: 0; }
  body { max-width: none; }
}"""

_SHINY_CSS = _BASE_CSS + "\n" + _OVERVIEW_CSS + "\n" + _SIMULATOR_CSS


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
{_SHINY_CSS}
</style>
</head>
<body>
<div class="hero">
<h1>EEG SpecParam Equivalence Simulator</h1>
<p>Demonstrating that spectral-domain and time-domain SpecParam produce equivalent
results on synthetic EEG data</p>
<a href="/simulate" class="cta-btn">Launch Simulator</a>
</div>
{_nav_html("Overview")}

<div class="card">
<div class="card-header">Introduction</div>
<div class="card-body">
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
</div>

<div class="card">
<div class="card-header">Mathematical Background</div>
<div class="card-body">
<p><strong>Aperiodic component</strong> &mdash; modeled as a Lorentzian in log-log space with
offset <em>b</em>, knee <em>k</em>, and spectral exponent &chi;:</p>
<div class="theorem-box"><div class="th-label">Aperiodic Model</div>{_MATH_APERIODIC}</div>

<p><strong>Periodic peaks</strong> &mdash; each oscillatory component is a Gaussian in
log-power space with amplitude <em>a</em>, center frequency <em>c</em>, and bandwidth <em>w</em>:</p>
<div class="theorem-box"><div class="th-label">Periodic Component</div>{_MATH_PERIODIC}</div>

<p><strong>Full model</strong> &mdash; aperiodic and periodic are additive in log-PSD space,
so the linear PSD is their exponentiated sum:</p>
<div class="theorem-box"><div class="th-label">Full PSD Model</div>{_MATH_FULL}</div>

<p><strong>ACF via inverse FFT</strong> &mdash; the model autocovariance is computed numerically
by taking the inverse FFT of the two-sided model PSD:</p>
<div class="theorem-box"><div class="th-label">Model ACF</div>{_MATH_ACF}</div>

<p><strong>Time-domain fitting</strong> &mdash; the ACF-based fitter minimizes the sum of squared
residuals between empirical and model ACFs over lags 0 to <em>M</em>:</p>
<div class="theorem-box"><div class="th-label">Objective Function</div>{_MATH_FITTING}</div>

<h3>Three-Stage Fitting Pipeline</h3>
<ol>
<li><strong>Stage 0: Spectral initialization</strong> &mdash; quick PSD fit to seed initial parameter guesses</li>
<li><strong>Stage 1: PSD joint refit</strong> &mdash; refine &chi; estimate with peaks, then ACF normalized fit</li>
<li><strong>Stage 2: ACF full model</strong> &mdash; joint fit of all parameters with &chi; regularized toward PSD-refined estimate</li>
</ol>
</div>
</div>

<div class="card">
<div class="card-header">Architecture</div>
<div class="card-body">
<div class="flow-diagram">
  <div class="flow-node flow-input">EEGSignal (numpy)</div>
  <div class="flow-arrow">&darr;</div>
  <div class="flow-split">
    <div class="flow-branch">
      <div class="flow-node flow-spectral">Spectral Pipeline</div>
      <div class="flow-detail">Welch PSD &rarr; specparam</div>
      <div class="flow-arrow">&darr;</div>
      <div class="flow-node flow-result">SpecParamResult</div>
    </div>
    <div class="flow-branch">
      <div class="flow-node flow-timedomain">Time-Domain Pipeline</div>
      <div class="flow-detail">ACF via FFT &rarr; IRFFT fit</div>
      <div class="flow-arrow">&darr;</div>
      <div class="flow-node flow-result">SpecParamResult</div>
    </div>
  </div>
  <div class="flow-arrow">&darr;</div>
  <div class="flow-node flow-comparison">ComparisonResult (RMSE, TOST, Bland-Altman)</div>
</div>
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
</div>

<div class="card">
<div class="card-header">Live Equivalence Summary</div>
<div class="card-body">
<p style="color:#7f8c8d; margin-top:0;">Mini-sweep: {n_runs} runs (5 exponents &times; 3 peak configs).
Each signal is 10 sec at 256 Hz.</p>
<div class="metrics-grid">
  <div class="metric-box"><div class="value">{converged}/{n_runs}</div><div class="label">Converged</div></div>
  <div class="metric-box"><div class="value">{exp_rmse}</div><div class="label">Exponent RMSE</div></div>
  <div class="metric-box"><div class="value">{off_rmse}</div><div class="label">Offset RMSE</div></div>
  <div class="metric-box"><div class="value"><span class="badge" style="background:{tost_color}">{tost_label}</span></div>
    <div class="label">TOST (bound=0.2, p={tost_p:.4f})</div></div>
</div>
<div style="margin-top:16px;">
<h4 style="color:#2c3e50; margin-bottom:8px;">Bland-Altman: Spectral vs Time-Domain</h4>
<div id="ba-plot"></div>
</div>
</div>
</div>

<div class="card">
<div class="card-header">Example Configurations</div>
<div class="card-body">
<p style="color:#7f8c8d; margin-top:0;">Click any card to open the simulator with that configuration pre-loaded.</p>
<div class="example-cards">
  <div class="example-card">
    <h4>Clean 1/f (No Peaks)</h4>
    <div class="params">&chi;=1.5 &middot; 0 peaks &middot; 30s</div>
    <p>Pure aperiodic signal. Tests baseline aperiodic recovery.</p>
    <a href="/simulate?exponent=1.5&offset=1.5&n_peaks=0&duration=30&sfreq=256&seed=42">Try in Simulator &rarr;</a>
  </div>
  <div class="example-card">
    <h4>Strong Alpha Rhythm</h4>
    <div class="params">&chi;=1.5 &middot; 10 Hz (1.2) &middot; 30s</div>
    <p>Prominent 10 Hz alpha peak. The most common EEG signature.</p>
    <a href="/simulate?exponent=1.5&offset=1.5&n_peaks=1&peak1_center=10&peak1_power=1.2&peak1_bw=2.0&duration=30&sfreq=256&seed=42">Try in Simulator &rarr;</a>
  </div>
  <div class="example-card">
    <h4>Alpha + Beta</h4>
    <div class="params">&chi;=1.5 &middot; 10 Hz + 25 Hz &middot; 30s</div>
    <p>Two-peak model. Tests multi-peak detection.</p>
    <a href="/simulate?exponent=1.5&offset=1.5&n_peaks=2&peak1_center=10&peak1_power=0.8&peak1_bw=2.0&peak2_center=25&peak2_power=0.6&peak2_bw=1.5&duration=30&sfreq=256&seed=42">Try in Simulator &rarr;</a>
  </div>
  <div class="example-card">
    <h4>Steep Spectral Slope</h4>
    <div class="params">&chi;=2.5 &middot; 10 Hz (0.8) &middot; 30s</div>
    <p>Tests recovery at the upper edge of the exponent range.</p>
    <a href="/simulate?exponent=2.5&offset=1.5&n_peaks=1&peak1_center=10&peak1_power=0.8&peak1_bw=2.0&duration=30&sfreq=256&seed=42">Try in Simulator &rarr;</a>
  </div>
  <div class="example-card">
    <h4>Short Clinical Recording</h4>
    <div class="params">&chi;=1.5 &middot; 10 Hz (0.8) &middot; 5s</div>
    <p>Only 5 seconds of data. Tests robustness with limited windows.</p>
    <a href="/simulate?exponent=1.5&offset=1.5&n_peaks=1&peak1_center=10&peak1_power=0.8&peak1_bw=2.0&duration=5&sfreq=256&seed=42">Try in Simulator &rarr;</a>
  </div>
</div>
</div>
</div>

<div class="card">
<div class="card-header">Using the Simulator</div>
<div class="card-body">
<h4 style="margin-top:0; color:#2c3e50;">Parameter Guide</h4>
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
{_SHINY_CSS}
</style>
</head>
<body class="sim-page">

<div class="sidebar" id="sidebar">
  <div class="sidebar-header">
    <h2>Parameters</h2>
    <button class="sidebar-close" id="sidebar-close" type="button">&times;</button>
  </div>
  <div class="sidebar-body">
    <form id="sim-form" method="get" action="/simulate">
      <details open>
        <summary>Aperiodic Parameters</summary>
        <div class="well">
          <div class="field"><label>Exponent (&chi;)</label>
            <div class="range-row"><input type="range" min="0.5" max="3.0" step="0.1" value="{exponent}" data-sync="exponent">
            <input type="number" name="exponent" value="{exponent}" step="0.1" min="0.5" max="3.0"></div>
            <span class="help-text">1/f slope of the power spectrum. Typical EEG: 1.0&ndash;2.0</span></div>
          <div class="field"><label>Offset (b)</label>
            <div class="range-row"><input type="range" min="-3" max="5" step="0.5" value="{offset}" data-sync="offset">
            <input type="number" name="offset" value="{offset}" step="0.5" min="-3" max="5"></div>
            <span class="help-text">Y-intercept of the aperiodic fit in log-log space</span></div>
        </div>
      </details>
      <details open>
        <summary>Periodic Peaks</summary>
        <div class="well">
          <div class="field"><label>Number of peaks</label>
            <select name="n_peaks" class="styled-select">
              <option value="0" {"selected" if n_peaks == 0 else ""}>0 &mdash; aperiodic only</option>
              <option value="1" {"selected" if n_peaks == 1 else ""}>1 peak</option>
              <option value="2" {"selected" if n_peaks == 2 else ""}>2 peaks</option>
              <option value="3" {"selected" if n_peaks == 3 else ""}>3 peaks</option>
            </select></div>
          {_peak_fields_html(1, peak1_center, peak1_power, peak1_bw, n_peaks)}
          {_peak_fields_html(2, peak2_center, peak2_power, peak2_bw, n_peaks)}
          {_peak_fields_html(3, peak3_center, peak3_power, peak3_bw, n_peaks)}
        </div>
      </details>
      <details open>
        <summary>Signal Settings</summary>
        <div class="well">
          <div class="field"><label>Duration (sec)</label>
            <div class="range-row"><input type="range" min="2" max="120" step="1" value="{duration}" data-sync="duration">
            <input type="number" name="duration" value="{duration}" step="1" min="2" max="120"></div>
            <span class="help-text">Longer signals improve ACF recovery for steep exponents</span></div>
          <div class="field"><label>Sample rate (Hz)</label>
            <div class="range-row"><input type="range" min="128" max="1024" step="64" value="{sfreq}" data-sync="sfreq">
            <input type="number" name="sfreq" value="{sfreq}" step="64" min="128" max="1024"></div>
            <span class="help-text">256 Hz is standard for scalp EEG</span></div>
          <div class="field"><label>Seed</label>
            <input type="number" name="seed" value="{seed}" min="0" max="99999">
            <span class="help-text">Random seed for reproducibility</span></div>
        </div>
      </details>
      <button type="submit" class="run-btn">Run Simulation</button>
    </form>
  </div>
</div>

<div class="main-panel" id="main-panel">
  <div class="top-bar">
    <button class="sidebar-toggle" id="sidebar-toggle" type="button">&#9776;</button>
    <div>
      <h1 style="margin-bottom:2px;">EEG SpecParam Simulator</h1>
      <p class="subtitle" style="margin-bottom:8px;">Compare spectral and time-domain SpecParam on synthetic EEG</p>
    </div>
  </div>
  {_nav_html("Simulator")}

  <div class="card plot-wrapper">
    <div class="card-header"><span>Power Spectral Density</span><button class="expand-btn" type="button">&#x26F6;</button></div>
    <div class="card-body">
      <div class="loading-overlay" id="loading"><div class="skeleton-bar"></div></div>
      <div id="psd-plot"></div>
    </div>
  </div>

  <div class="card plot-wrapper" id="acf-container" style="{"" if acf_fig else "display:none"}">
    <div class="card-header"><span>Autocovariance (ACF Fit)</span><button class="expand-btn" type="button">&#x26F6;</button></div>
    <div class="card-body">
      <div id="acf-plot"></div>
    </div>
  </div>

  <div class="card">
    <div class="card-header"><span>Parameter Comparison</span></div>
    <div class="card-body" style="padding:0;">
      <table>
        <tr><th>Parameter</th><th>Spectral</th><th>Time-Domain</th><th>Difference</th></tr>
        <tbody id="comparison-table">{table_html}</tbody>
      </table>
    </div>
  </div>
</div>

<script>
(function() {{
  var psdData = {psd_json};
  Plotly.react("psd-plot", psdData.data, psdData.layout);

  var acfData = {acf_json};
  if (acfData) Plotly.react("acf-plot", acfData.data, acfData.layout);

  document.querySelectorAll(".expand-btn").forEach(function(btn) {{
    btn.addEventListener("click", function() {{
      var card = btn.closest(".card");
      card.classList.toggle("fullscreen");
      btn.innerHTML = card.classList.contains("fullscreen") ? "&#x2715;" : "&#x26F6;";
      setTimeout(function() {{
        var plotDiv = card.querySelector(".js-plotly-plot");
        if (plotDiv) Plotly.Plots.resize(plotDiv);
      }}, 100);
    }});
  }});

  var sidebar = document.getElementById("sidebar");
  document.getElementById("sidebar-toggle").addEventListener("click", function() {{
    sidebar.classList.toggle("open");
  }});
  document.getElementById("sidebar-close").addEventListener("click", function() {{
    sidebar.classList.remove("open");
  }});

  var form = document.getElementById("sim-form");
  var loading = document.getElementById("loading");
  var controller = null;
  var timer = null;

  function showPeakFields() {{
    var n = parseInt(form.querySelector("[name=n_peaks]").value) || 0;
    [1, 2, 3].forEach(function(i) {{
      var el = document.getElementById("peak" + i + "-fields");
      if (n >= i) {{ el.classList.add("visible"); }}
      else {{ el.classList.remove("visible"); }}
    }});
  }}

  form.querySelectorAll("input[type=range]").forEach(function(slider) {{
    var name = slider.getAttribute("data-sync");
    var num = form.querySelector("input[name=" + name + "]");
    slider.addEventListener("input", function() {{ num.value = slider.value; }});
    num.addEventListener("input", function() {{ slider.value = num.value; }});
  }});

  function getParams() {{
    var params = new URLSearchParams();
    form.querySelectorAll("input[name], select[name]").forEach(function(el) {{
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

  function onFormChange() {{
    showPeakFields();
    clearTimeout(timer);
    timer = setTimeout(fetchResults, 500);
  }}
  form.addEventListener("input", onFormChange);
  form.addEventListener("change", onFormChange);

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
{_SHINY_CSS}
a {{ color: var(--accent); }}
</style>
</head>
<body>
<h1>Parameter Space Sweep</h1>
<p class="subtitle">Bland-Altman analysis and equivalence metrics across exponent &times; peak power</p>

{_nav_html("Quick Sweep" if mode == "quick" else "Full Sweep")}

<div class="sweep-controls">
  <label>Mode:</label>
  <select onchange="window.location='/sweep?mode='+this.value+'&seed={seed}'">
    <option value="quick" {"selected" if mode == "quick" else ""}>Quick (20 runs)</option>
    <option value="full" {"selected" if mode == "full" else ""}>Full (36 runs)</option>
  </select>
  <label>Seed:</label>
  <input type="number" value="{seed}" min="0" max="9999" style="width:70px;padding:6px 8px;border:1px solid var(--border);border-radius:4px;"
         onchange="window.location='/sweep?mode={mode}&seed='+this.value">
</div>

<div class="card">
<div class="card-header">Summary</div>
<div class="card-body">
<p style="color:var(--text-muted); margin-top:0; font-size:0.9em;">
{len(exponents)} exponents &times; {len(peak_powers)} peak powers = {n_runs} runs</p>
<div class="metrics-grid">
  <div class="metric-box"><div class="value">{converged}/{n_runs}</div><div class="label">Converged</div></div>
  <div class="metric-box"><div class="value">{exp_rmse}</div><div class="label">Exponent RMSE</div></div>
  <div class="metric-box"><div class="value">{off_rmse}</div><div class="label">Offset RMSE</div></div>
  <div class="metric-box"><div class="value">{pk_rmse}</div><div class="label">Peak Center RMSE</div></div>
  <div class="metric-box"><div class="value"><span class="badge" style="background:{tost_exp_color}">{tost_exp_str}</span></div>
    <div class="label">TOST Exponent (p={max(tost_exp['p_upper'], tost_exp['p_lower']):.4f})</div></div>
  <div class="metric-box"><div class="value"><span class="badge" style="background:{tost_off_color}">{tost_off_str}</span></div>
    <div class="label">TOST Offset (p={max(tost_off['p_upper'], tost_off['p_lower']):.4f})</div></div>
</div>
</div>
</div>

<div class="card">
<div class="card-header">Bland-Altman Plots</div>
<div class="card-body">
{ba_html}
</div>
</div>

<div class="card">
<div class="card-header">Parameter Space Heatmap</div>
<div class="card-body">
{hm_html}
</div>
</div>

</body>
</html>"""


def main():
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
