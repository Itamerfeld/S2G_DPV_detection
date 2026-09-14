########################################################
# SIMULATION 2: fixed Q, vary SNR
########################################################
# Monte-Carlo Pd / Pfa / ROC for one S2G detector while SNR changes.
# STFT, signal model, and figure style match example.ipynb.
#
# Why these analysis settings (same as the notebook):
#   - overlap = 0 so adjacent frames share no samples
#   - nperseg = 8192 at 128 kHz so 640 Hz is not an integer number of
#     cycles per hop (that lock made wrapped-phase S2G look perfect)
#   - phase_mode = 'difference': wrapped frame-to-frame increment, no unwrap
#   - SNR is referenced to one detector bin via analysis_bandwidth()
#   - the tonal wanders and is AM'd; a CW tone is degenerate for S2G
#
"""Fixed-Q ROC vs SNR."""

import os
import pickle
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import utils as ut

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
show_plots = os.environ.get("S2G_HEADLESS") != "1"

########################################################
# PARAMETERS
########################################################
run_simulation = False
# One-shot feature-vs-frequency figure (cheap; does not use the pickle).
plot_feature_grid = True

# --- signal ---
fs = 128000
f0 = 640
duration = 60
snr_values = [0, -6, -12, -18]
num_iterations = 100
rng_seed = 0
# Must exceed one STFT bin (~15.6 Hz) so Hann sidelobes are not scored as FA.
tolerance = 25

# --- STFT (aligned with example.ipynb) ---
nperseg = 8192
nfft = None
overlap = 0.0
window = "hanning"
dc = 150
crop_freq = 8000

# --- S2G ---
mode = "wasserstein"
phase_mode = "difference"
wasserstein_circular = False
quantization_level = 5 if "laplacian" in mode else 3

signal_bw = ut.analysis_bandwidth(fs, nperseg, window)

# Tonal realism, calibrated on the dpv*_1m recordings (2–5 Hz wander, 0.4–0.9 AM).
freq_wander_hz = 2.0
am_depth = 0.5
blade_rate_hz = 40.0
blade_depth = 0.4

num_thresholds = 20
threshold_range = (2, 6)
thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)

MODE_TAG = f"{mode}_circ" if wasserstein_circular else mode
RESULT_NAME = f"simulation_2_set_Q_({quantization_level})_change_SNR_{MODE_TAG}"


def _s2g_detector():
    return ut.S2GDetector(
        fs=fs, nperseg=nperseg, overlap=overlap, nfft=nfft, window=window,
        dc=dc, crop_freq=crop_freq, quantization_levels=quantization_level,
        mode=mode, phase_mode=phase_mode, wasserstein_circular=wasserstein_circular,
    )


def _notebook_axes(fig):
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)", zeroline=False)
    fig.update_layout(
        font=dict(size=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(font=dict(size=13), bgcolor="rgba(255,255,255,0.85)"),
        margin=dict(t=50, b=55, l=70, r=30),
    )
    return fig


########################################################
# Monte-Carlo
########################################################
if run_simulation:
    print(
        f"{datetime.now():%H:%M:%S}  sim 2: Q={quantization_level}, mode={MODE_TAG}, "
        f"{num_iterations} iter × {len(snr_values)} SNR × {len(thresholds)} thr"
    )
    pd_rates = np.zeros((len(snr_values), len(thresholds)))
    fa_rates = np.zeros((len(snr_values), len(thresholds)))

    raw_signal = ut.simulate_raw_signal(
        f0, fs, duration,
        freq_wander_hz=freq_wander_hz, am_depth=am_depth,
        blade_rate_hz=blade_rate_hz, blade_depth=blade_depth,
        seed=rng_seed,
    )
    detector = _s2g_detector()

    for si, snr in enumerate(snr_values):
        pd = np.zeros((num_iterations, len(thresholds)))
        fa = np.zeros((num_iterations, len(thresholds)))
        for i in range(num_iterations):
            rx = ut.add_noise_to_signal(raw_signal, snr_db=snr, fs=fs, signal_bw=signal_bw, noise_type="white")
            feature, F = detector.get_feature_vector(rx)
            fix, f_mask = ut.evaluation_bins(F, f0, tolerance)
            for ti, threshold in enumerate(thresholds):
                pd[i, ti] = 1 if feature[fix] > threshold else 0
                fa[i, ti] = 1 if np.any(feature[f_mask] > threshold) else 0
        pd_rates[si, :] = np.mean(pd, axis=0)
        fa_rates[si, :] = np.mean(fa, axis=0)
        print(f"{datetime.now():%H:%M:%S}  finished SNR={snr} dB")

    with open(os.path.join(RESULTS_DIR, f"{RESULT_NAME}.pkl"), "wb") as f:
        pickle.dump({"pd_rates": pd_rates, "fa_rates": fa_rates}, f)
    print(f"saved {RESULT_NAME}.pkl")
else:
    with open(os.path.join(RESULTS_DIR, f"{RESULT_NAME}.pkl"), "rb") as f:
        results = pickle.load(f)
    pd_rates = results["pd_rates"]
    fa_rates = results["fa_rates"]

########################################################
# Pd / Pfa / ROC
########################################################
snr_colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
snr_linestyles = ["solid", "dash", "dot", "dashdot"]
snr_markers = ["circle", "square", "diamond", "triangle-up"]

fig_pd, fig_fa, fig_roc = go.Figure(), go.Figure(), go.Figure()
for si, snr in enumerate(snr_values):
    line = dict(color=snr_colors[si], width=2.4, dash=snr_linestyles[si])
    marker = dict(symbol=snr_markers[si], size=8, line=dict(width=1))
    name = f"SNR = {snr} dB"
    fig_pd.add_trace(go.Scatter(x=thresholds, y=pd_rates[si], mode="lines+markers", name=name, line=line, marker=marker))
    fig_fa.add_trace(go.Scatter(x=thresholds, y=fa_rates[si], mode="lines+markers", name=name, line=line, marker=marker))
    fig_roc.add_trace(go.Scatter(x=fa_rates[si], y=pd_rates[si], mode="lines+markers", name=name, line=line, marker=marker))

plot_base = os.path.join(PLOTS_DIR, RESULT_NAME)
for fig, xlabel, ylabel, suffix in (
    (fig_pd, "Threshold", "Detection rate", "pd"),
    (fig_fa, "Threshold", "False-alarm rate", "fa"),
    (fig_roc, "False-alarm rate", "Detection rate", "roc"),
):
    fig.update_xaxes(title_text=xlabel, title_font=dict(size=16))
    fig.update_yaxes(title_text=ylabel, title_font=dict(size=16))
    fig.update_layout(height=520, width=560, title=dict(
        text=f"Sim 2 · Q = {quantization_level} · {MODE_TAG} · phase {phase_mode}",
        font=dict(size=14), x=0.01, xanchor="left",
    ))
    _notebook_axes(fig)
    fig.write_image(f"{plot_base}_{suffix}.pdf")
    if show_plots:
        fig.show(renderer="browser")

########################################################
# Feature vs frequency across SNR (same layout as example.ipynb)
########################################################
if plot_feature_grid:
    raw_signal = ut.simulate_raw_signal(
        f0, fs, duration,
        freq_wander_hz=freq_wander_hz, am_depth=am_depth,
        blade_rate_hz=blade_rate_hz, blade_depth=blade_depth,
        seed=rng_seed,
    )
    detector = _s2g_detector()
    s2g_color = "#2ca02c" if mode == "wasserstein" else "#9467bd"

    fig = make_subplots(
        rows=1, cols=len(snr_values),
        shared_yaxes=True, horizontal_spacing=0.03,
        column_titles=[f"SNR = {snr} dB" for snr in snr_values],
    )
    for i, snr in enumerate(snr_values, start=1):
        rx = ut.add_noise_to_signal(raw_signal, snr_db=snr, fs=fs, signal_bw=signal_bw, noise_type="white")
        K, F = detector.get_feature_vector(rx)
        fig.add_trace(
            go.Scatter(x=F, y=K, mode="lines", showlegend=False, line=dict(color=s2g_color, width=1.4)),
            row=1, col=i,
        )
        fig.add_vline(x=f0, line_dash="dot", line_color="crimson", line_width=1, row=1, col=i)
    fig.update_layout(
        height=360, width=1400,
        title=dict(
            text=f"Sim 2 · {MODE_TAG} (Q = {quantization_level}) score vs frequency",
            font=dict(size=14), x=0.01, xanchor="left",
        ),
        font=dict(size=12), plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(t=55, b=55, l=60, r=20),
    )
    fig.update_xaxes(title_text="Frequency (Hz)", showgrid=True, gridcolor="rgba(0,0,0,0.12)", zeroline=False)
    fig.update_yaxes(title_text="Score", col=1, showgrid=True, gridcolor="rgba(0,0,0,0.12)", zeroline=False)
    fig.update_annotations(font_size=13)
    fig.write_image(f"{plot_base}_features.pdf")
    if show_plots:
        fig.show(renderer="browser")
    print("wrote feature-vs-frequency grid")
