from __future__ import annotations

import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from scipy.signal import welch

from .schemas import AperiodicParams, PeriodicPeak
from .eeg_generator import compute_target_psd, generate_eeg_signal
from .spectral_specparam import fit_spectral_specparam
from .time_domain_wrapper import fit_time_domain
from .comparison import compare_results, compute_agreement_metrics, tost_equivalence

app = FastAPI(title="EEG SpecParam Equivalence Simulator")

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
.controls {{ margin-bottom: 20px; }}
.controls a {{ margin-right: 16px; padding: 8px 16px; background: #3498db; color: white;
               text-decoration: none; border-radius: 4px; }}
.controls a.active {{ background: #2c3e50; }}
.controls a:hover {{ background: #2980b9; }}"""


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
    spectral = fit_spectral_specparam(signal)
    td = fit_time_domain(signal)
    comp = compare_results(spectral, td)

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


@app.get("/", response_class=HTMLResponse)
def dashboard(
    exponent: float = Query(1.5, ge=0.5, le=3.0),
    offset: float = Query(1.5, ge=-3.0, le=5.0),
    peak1_center: float = Query(10.0, ge=0, le=50),
    peak1_power: float = Query(0.8, ge=0, le=3),
    peak1_bw: float = Query(2.0, ge=0.1, le=10),
    n_peaks: int = Query(1, ge=0, le=3),
    peak2_center: float = Query(25.0, ge=0, le=50),
    peak2_power: float = Query(0.6, ge=0, le=3),
    peak2_bw: float = Query(1.5, ge=0.1, le=10),
    sfreq: float = Query(256, ge=64, le=512),
    duration: float = Query(30, ge=2, le=60),
    seed: int = Query(42, ge=0, le=9999),
):
    peak_centers = [peak1_center][:n_peaks]
    peak_powers = [peak1_power][:n_peaks]
    peak_bws = [peak1_bw][:n_peaks]
    if n_peaks >= 2:
        peak_centers.append(peak2_center)
        peak_powers.append(peak2_power)
        peak_bws.append(peak2_bw)

    spectral, td, comp, freqs, psd = _run_simulation(
        exponent, offset, peak_centers, peak_powers, peak_bws,
        sfreq, duration, seed,
    )

    psd_fig = _build_psd_plot(freqs, psd, spectral, td)
    psd_html = plotly.io.to_html(psd_fig, full_html=False, include_plotlyjs="cdn")

    acf_fig = _build_acf_plot(td)
    acf_html = plotly.io.to_html(acf_fig, full_html=False, include_plotlyjs=False) if acf_fig else ""

    table_html = _build_comparison_table(spectral, td, comp)

    return f"""<!DOCTYPE html>
<html>
<head>
<title>EEG SpecParam Equivalence Simulator</title>
<style>
{_BASE_CSS}
form {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 20px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }}
.field {{ display: flex; flex-direction: column; }}
.field label {{ font-size: 0.85em; color: #555; margin-bottom: 3px; }}
.field input {{ padding: 6px 8px; border: 1px solid #ddd; border-radius: 4px; }}
button {{ background: #3498db; color: white; border: none; padding: 10px 24px;
          border-radius: 4px; cursor: pointer; font-size: 1em; margin-top: 12px; }}
button:hover {{ background: #2980b9; }}
</style>
</head>
<body>
<h1>EEG SpecParam Equivalence Simulator</h1>
<p class="subtitle">Compare spectral and time-domain SpecParam on synthetic EEG</p>

<div class="controls">
  <a href="/" class="active">Single Simulation</a>
  <a href="/sweep?mode=quick">Quick Sweep</a>
  <a href="/sweep?mode=full">Full Sweep</a>
</div>

<form method="get">
<div class="grid">
  <div class="field"><label>Exponent (1/f slope)</label>
    <input type="number" name="exponent" value="{exponent}" step="0.1" min="0.5" max="3.0"></div>
  <div class="field"><label>Offset (log power)</label>
    <input type="number" name="offset" value="{offset}" step="0.5" min="-3" max="5"></div>
  <div class="field"><label>Number of peaks</label>
    <input type="number" name="n_peaks" value="{n_peaks}" min="0" max="3"></div>
  <div class="field"><label>Duration (sec)</label>
    <input type="number" name="duration" value="{duration}" step="5" min="2" max="60"></div>
  <div class="field"><label>Sample rate (Hz)</label>
    <input type="number" name="sfreq" value="{sfreq}" step="64" min="64" max="512"></div>
  <div class="field"><label>Seed</label>
    <input type="number" name="seed" value="{seed}" min="0" max="9999"></div>
</div>
{f'''<h4 style="margin: 12px 0 6px;">Peak 1</h4>
<div class="grid">
  <div class="field"><label>Center freq (Hz)</label>
    <input type="number" name="peak1_center" value="{peak1_center}" step="1" min="1" max="50"></div>
  <div class="field"><label>Power (log10)</label>
    <input type="number" name="peak1_power" value="{peak1_power}" step="0.1" min="0" max="3"></div>
  <div class="field"><label>Bandwidth (Hz)</label>
    <input type="number" name="peak1_bw" value="{peak1_bw}" step="0.5" min="0.1" max="10"></div>
</div>''' if n_peaks >= 1 else ''}
{f'''<h4 style="margin: 12px 0 6px;">Peak 2</h4>
<div class="grid">
  <div class="field"><label>Center freq (Hz)</label>
    <input type="number" name="peak2_center" value="{peak2_center}" step="1" min="1" max="50"></div>
  <div class="field"><label>Power (log10)</label>
    <input type="number" name="peak2_power" value="{peak2_power}" step="0.1" min="0" max="3"></div>
  <div class="field"><label>Bandwidth (Hz)</label>
    <input type="number" name="peak2_bw" value="{peak2_bw}" step="0.5" min="0.1" max="10"></div>
</div>''' if n_peaks >= 2 else ''}
<button type="submit">Generate &amp; Compare</button>
</form>

<div class="plot">
<h3>Power Spectral Density</h3>
{psd_html}
</div>

{f'<div class="plot"><h3>Autocovariance (ACF Fit)</h3>{acf_html}</div>' if acf_html else ''}

<h3>Parameter Comparison</h3>
<table>
<tr><th>Parameter</th><th>Spectral</th><th>Time-Domain</th><th>Difference</th></tr>
{table_html}
</table>

</body>
</html>"""


def _run_sweep(exponents: list[float], peak_powers: list[float], base_seed: int = 42):
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
            spectral = fit_spectral_specparam(signal)
            td = fit_time_domain(signal)
            comp = compare_results(spectral, td)
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
        exponents = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
        peak_powers = [0.0, 0.3, 0.6, 0.9, 1.2, 1.5]
    else:
        exponents = [1.0, 1.25, 1.5, 1.75, 2.0]
        peak_powers = [0.0, 0.4, 0.8, 1.2]

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
.badge {{ display: inline-block; padding: 2px 10px; border-radius: 4px; color: white; font-weight: bold; }}
a {{ color: #3498db; }}
</style>
</head>
<body>
<h1>EEG SpecParam Equivalence Simulator</h1>
<p class="subtitle">Bland-Altman analysis and parameter sweep across exponent x peak power</p>

<div class="controls">
  <a href="/">Single Simulation</a>
  <a href="/sweep?mode=quick&seed={seed}" {"class='active'" if mode == "quick" else ""}>Quick Sweep</a>
  <a href="/sweep?mode=full&seed={seed}" {"class='active'" if mode == "full" else ""}>Full Sweep</a>
</div>

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
