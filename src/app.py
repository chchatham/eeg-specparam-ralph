from __future__ import annotations

import numpy as np
import plotly
import plotly.graph_objects as go
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from scipy.signal import welch

from .schemas import AperiodicParams, PeriodicPeak
from .eeg_generator import generate_eeg_signal
from .spectral_specparam import fit_spectral_specparam
from .time_domain_wrapper import fit_time_domain
from .comparison import compare_results

app = FastAPI(title="EEG SpecParam Equivalence Simulator")


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

    # Spectral model curve
    sp = spectral.aperiodic
    log_spec = sp.offset - sp.exponent * np.log10(np.maximum(f_fit, 0.01))
    for pk in spectral.peaks:
        log_spec = log_spec + pk.power * np.exp(-((f_fit - pk.center_frequency)**2) / (2 * pk.bandwidth**2))
    fig.add_trace(go.Scatter(
        x=f_fit, y=10**log_spec, mode="lines", name="Spectral SpecParam",
        line=dict(color="blue", width=2),
    ))

    # Time-domain model curve
    if td.converged:
        tp = td.aperiodic
        k = tp.knee if tp.knee is not None else 0
        log_td = tp.offset - np.log10(np.maximum(k, 1e-12) + f_fit**tp.exponent)
        for pk in td.peaks:
            log_td = log_td + pk.power * np.exp(-((f_fit - pk.center_frequency)**2) / (2 * pk.bandwidth**2))
        fig.add_trace(go.Scatter(
            x=f_fit, y=10**log_td, mode="lines", name="Time-Domain SpecParam",
            line=dict(color="red", width=2, dash="dash"),
        ))

    fig.update_xaxes(type="log", title_text="Frequency (Hz)", range=[0, np.log10(50)])
    fig.update_yaxes(type="log", title_text="Power (µV²/Hz)")
    fig.update_layout(
        height=400, margin=dict(l=60, r=20, t=30, b=50),
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

    table_html = _build_comparison_table(spectral, td, comp)

    return f"""<!DOCTYPE html>
<html>
<head>
<title>EEG SpecParam Equivalence Simulator</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 1100px; margin: 0 auto; padding: 20px; background: #f8f9fa; }}
h1 {{ color: #2c3e50; margin-bottom: 5px; }}
.subtitle {{ color: #7f8c8d; margin-bottom: 20px; }}
form {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 20px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }}
.field {{ display: flex; flex-direction: column; }}
.field label {{ font-size: 0.85em; color: #555; margin-bottom: 3px; }}
.field input {{ padding: 6px 8px; border: 1px solid #ddd; border-radius: 4px; }}
button {{ background: #3498db; color: white; border: none; padding: 10px 24px;
          border-radius: 4px; cursor: pointer; font-size: 1em; margin-top: 12px; }}
button:hover {{ background: #2980b9; }}
.plot {{ background: white; padding: 15px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
         margin-bottom: 20px; }}
table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px;
         overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
th, td {{ padding: 8px 12px; text-align: left; border-bottom: 1px solid #eee; }}
th {{ background: #2c3e50; color: white; }}
</style>
</head>
<body>
<h1>EEG SpecParam Equivalence Simulator</h1>
<p class="subtitle">Compare spectral and time-domain SpecParam on synthetic EEG</p>

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

<h3>Parameter Comparison</h3>
<table>
<tr><th>Parameter</th><th>Spectral</th><th>Time-Domain</th><th>Difference</th></tr>
{table_html}
</table>

</body>
</html>"""


def main():
    import uvicorn
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
