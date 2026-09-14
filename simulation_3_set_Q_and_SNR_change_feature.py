########################################################
# SIMULATION 3: fixed SNR and Q, compare detectors
########################################################
# Monte-Carlo Pd / Pfa / ROC across the same detector set as example.ipynb,
# plus a feature-vs-frequency grid vs SNR (notebook cell "Run on simulation").
#
# Why these analysis settings (same as the notebook):
#   - overlap = 0 so adjacent frames share no samples
#   - nperseg = 8192 at 128 kHz so 640 Hz is not an integer number of
#     cycles per hop (that lock made wrapped-phase S2G look perfect)
#   - phase_mode = 'difference': wrapped frame-to-frame increment, no unwrap
#   - entropy is Alexandri local spectral entropy (ClassicDetector mode='entropy')
#   - cosine is uncentered cosine of consecutive linear-power frames
#   - DEMON is scored at the propeller-rate line (blade_rate_hz), not at f0
#   - SNR is referenced to one detector bin via analysis_bandwidth()
#
"""Compare classical and S2G features at one SNR."""

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

########################################################
# PARAMETERS
########################################################
run_simulation = False
show_plots = os.environ.get("S2G_HEADLESS") != "1"
save_plots = True
# One-shot feature-vs-frequency figure (cheap; does not use the pickle).
plot_feature_grid = True

# --- signal ---
fs = 128000
f0 = 640
duration = 60
snr = -18
# SNR columns for the notebook-style feature grid (independent of `snr` above).
snr_grid = [4, 2, 0, -2, -4, -6, -8, -10]
num_iterations = 100
rng_seed = 0
# Must exceed one STFT bin (~15.6 Hz) so Hann sidelobes are not scored as FA.
tolerance = 25

# --- STFT (aligned with example.ipynb) ---
nperseg = 8192
nfft = None
overlap = 0.0
window = "hanning"
# Flow-noise cut for STFT features. DEMON uses a lower cut so the blade-rate
# line (~40 Hz) stays in-band; dc=150 would delete it.
dc = 150
demon_dc = 20
crop_freq = 8000
norm_size = 5

# --- S2G ---
phase_mode = "difference"
laplacian_quantization_levels = 5
wasserstein_quantization_levels = 3

signal_bw = ut.analysis_bandwidth(fs, nperseg, window)

# Tonal realism, calibrated on the dpv*_1m recordings (2–5 Hz wander, 0.4–0.9 AM).
freq_wander_hz = 2.0
am_depth = 0.5
# Propeller-rate AM: score DEMON at this frequency, not at f0.
blade_rate_hz = 40.0
blade_depth = 0.4

num_thresholds = 20
threshold_range = (2, 6)
thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)

RESULT_NAME = f"simulation_3_SNR_{snr}dB"

########################################################
# Detectors (same set / construction as example.ipynb)
########################################################
_stft = dict(fs=fs, nperseg=nperseg, overlap=overlap, nfft=nfft, window=window, crop_freq=crop_freq)

detectors = {
    "welch": ut.ClassicDetector(**_stft, dc=dc, norm_size=norm_size, mode="welch"),
    "demon": ut.ClassicDetector(**_stft, dc=demon_dc, norm_size=norm_size, mode="demon"),
    "correlation": ut.ClassicDetector(**_stft, dc=dc, norm_size=norm_size, mode="correlation"),
    "entropy": ut.ClassicDetector(**_stft, dc=dc, norm_size=norm_size, mode="entropy"),
    "cosine": ut.ClassicDetector(**_stft, dc=dc, norm_size=norm_size, mode="cosine"),
    "laplacian": ut.S2GDetector(
        **_stft, dc=dc, mode="laplacian",
        quantization_levels=laplacian_quantization_levels, phase_mode=phase_mode,
    ),
    "wasserstein": ut.S2GDetector(
        **_stft, dc=dc, mode="wasserstein",
        quantization_levels=wasserstein_quantization_levels, phase_mode=phase_mode,
    ),
    "wasserstein_circular": ut.S2GDetector(
        **_stft, dc=dc, mode="wasserstein",
        quantization_levels=wasserstein_quantization_levels, phase_mode=phase_mode,
        wasserstein_circular=True,
    ),
}

# Welch is shown on the feature grid (notebook "Spectrum") but is not a
# competitive detector at these SNRs; omit it from the ROC pickle by default.
ROC_DETECTORS = [k for k in detectors if k != "welch"]

DISPLAY_NAME = {
    "welch": "Spectrum",
    "demon": "DEMON",
    "correlation": "Correlation",
    "entropy": "Entropy",
    "cosine": "Cosine",
    "laplacian": "S2G Laplacian",
    "wasserstein": "S2G Wasserstein",
    "wasserstein_circular": "S2G Wasserstein circular",
}

CLASSICAL = {"welch", "demon", "correlation", "entropy", "cosine"}
classical_color = "#1f77b4"
laplacian_color = "#9467bd"
wasserstein_color = "#2ca02c"


def _eval_freq(name):
    """DEMON is a modulation-spectrum feature: score it at the blade rate."""
    return blade_rate_hz if name == "demon" else f0


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
        f"{datetime.now():%Y-%m-%d %H:%M:%S}: sim 3 SNR={snr} dB, "
        f"{num_iterations} iter, detectors={ROC_DETECTORS}"
    )
    raw_signal = ut.simulate_raw_signal(
        f0, fs, duration,
        freq_wander_hz=freq_wander_hz, am_depth=am_depth,
        blade_rate_hz=blade_rate_hz, blade_depth=blade_depth,
        seed=rng_seed,
    )
    all_results = {}
    for name in ROC_DETECTORS:
        detector = detectors[name]
        detected = np.zeros((num_iterations, len(thresholds)))
        false_alarms = np.zeros((num_iterations, len(thresholds)))
        for i in range(num_iterations):
            rx = ut.add_noise_to_signal(raw_signal, snr_db=snr, fs=fs, signal_bw=signal_bw, noise_type="white")
            feature, F = detector.get_feature_vector(rx)
            fix, f_mask = ut.evaluation_bins(F, _eval_freq(name), tolerance)
            for ti, threshold in enumerate(thresholds):
                detected[i, ti] = 1 if feature[fix] > threshold else 0
                false_alarms[i, ti] = 1 if np.any(feature[f_mask] > threshold) else 0
        all_results[name] = {
            "pd_rates": np.mean(detected, axis=0),
            "fa_rates": np.mean(false_alarms, axis=0),
        }
        print(f"{datetime.now():%Y-%m-%d %H:%M:%S}: finished {name}")

    with open(os.path.join(RESULTS_DIR, f"{RESULT_NAME}.pkl"), "wb") as f:
        pickle.dump(all_results, f)
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}: saved {RESULT_NAME}.pkl")
else:
    with open(os.path.join(RESULTS_DIR, f"{RESULT_NAME}.pkl"), "rb") as f:
        all_results = pickle.load(f)

########################################################
# Pd / Pfa / ROC
########################################################
detector_colors = {
    "demon": "#1f77b4",
    "correlation": "#ff7f0e",
    "entropy": "#8c564b",
    "cosine": "#17becf",
    "laplacian": "#9467bd",
    "wasserstein": "#2ca02c",
    "wasserstein_circular": "#006400",
    "welch": "#7f7f7f",
}
detector_linestyles = {
    "demon": "solid",
    "correlation": "dash",
    "entropy": "dot",
    "cosine": "dashdot",
    "laplacian": "solid",
    "wasserstein": "dash",
    "wasserstein_circular": "longdash",
    "welch": "longdashdot",
}
detector_markers = {
    "demon": "circle",
    "correlation": "square",
    "entropy": "diamond",
    "cosine": "triangle-up",
    "laplacian": "x",
    "wasserstein": "cross",
    "wasserstein_circular": "triangle-down",
    "welch": "hexagon",
}

fig_pd, fig_fa, fig_roc = go.Figure(), go.Figure(), go.Figure()
for name, rates in all_results.items():
    pd_rates = rates["pd_rates"]
    fa_rates = rates["fa_rates"]
    line = dict(
        color=detector_colors.get(name, "#333333"),
        width=2.4,
        dash=detector_linestyles.get(name, "solid"),
    )
    marker = dict(symbol=detector_markers.get(name, "circle"), size=8, line=dict(width=1))
    label = DISPLAY_NAME.get(name, name)
    fig_pd.add_trace(go.Scatter(x=thresholds, y=pd_rates, mode="lines+markers", name=label, line=line, marker=marker))
    fig_fa.add_trace(go.Scatter(x=thresholds, y=fa_rates, mode="lines+markers", name=label, line=line, marker=marker))
    fig_roc.add_trace(go.Scatter(x=fa_rates, y=pd_rates, mode="lines+markers", name=label, line=line, marker=marker))

plot_base = os.path.join(PLOTS_DIR, RESULT_NAME)
for fig, xlabel, ylabel, suffix in (
    (fig_pd, "Threshold", "Detection rate", "pd"),
    (fig_fa, "Threshold", "False-alarm rate", "fa"),
    (fig_roc, "False-alarm rate", "Detection rate", "roc"),
):
    fig.update_xaxes(title_text=xlabel, title_font=dict(size=16))
    fig.update_yaxes(title_text=ylabel, title_font=dict(size=16))
    fig.update_layout(height=520, width=560, title=dict(
        text=f"Sim 3 · SNR = {snr} dB · phase {phase_mode}",
        font=dict(size=14), x=0.01, xanchor="left",
    ))
    _notebook_axes(fig)
    if save_plots:
        fig.write_image(f"{plot_base}_{suffix}.pdf")
    if show_plots:
        fig.show(renderer="browser")

########################################################
# Feature vs frequency across SNR (example.ipynb simulation cell)
########################################################
if plot_feature_grid:
    raw_signal = ut.simulate_raw_signal(
        f0, fs, duration,
        freq_wander_hz=freq_wander_hz, am_depth=am_depth,
        blade_rate_hz=blade_rate_hz, blade_depth=blade_depth,
        seed=rng_seed,
    )
    row_order = [
        "welch", "demon", "correlation", "entropy", "cosine",
        "laplacian", "wasserstein", "wasserstein_circular",
    ]
    row_titles = [DISPLAY_NAME[k] for k in row_order]
    fig = make_subplots(
        rows=len(row_order),
        cols=len(snr_grid),
        shared_xaxes=True,
        shared_yaxes=False,
        vertical_spacing=0.04,
        horizontal_spacing=0.02,
        column_titles=[f"SNR = {s} dB" for s in snr_grid],
        row_titles=row_titles,
    )
    for col, snr_i in enumerate(snr_grid, start=1):
        rx = ut.add_noise_to_signal(raw_signal, snr_db=snr_i, fs=fs, signal_bw=signal_bw, noise_type="white")
        for row, name in enumerate(row_order, start=1):
            scores, F = detectors[name].get_feature_vector(rx)
            color = (
                laplacian_color if name == "laplacian"
                else wasserstein_color if "wasserstein" in name
                else classical_color
            )
            fig.add_trace(
                go.Scatter(x=F, y=scores, mode="lines", showlegend=False, line=dict(color=color, width=1.4)),
                row=row, col=col,
            )
            marker_hz = _eval_freq(name)
            fig.add_vline(x=marker_hz, line_dash="dot", line_color="crimson", line_width=1, row=row, col=col)

    fig.update_traces(selector=dict(type="scatter"), cliponaxis=False)
    fig.update_layout(
        height=880, width=1400, font=dict(size=12),
        margin=dict(t=55, b=55, l=90, r=110),
        plot_bgcolor="white", paper_bgcolor="white",
        title=dict(
            text="Sim 3 · detector features vs frequency  (crimson: f0, or blade rate on DEMON)",
            font=dict(size=14), x=0.01, xanchor="left",
        ),
    )
    fig.update_annotations(font_size=13)
    for ann in fig.layout.annotations:
        if ann.text in row_titles or (ann.text or "").startswith("SNR"):
            ann.text = f"<b>{ann.text}</b>"
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)", zeroline=False, autorange=True)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)", zeroline=False, autorange=True, automargin=True)
    fig.update_xaxes(title_text="Frequency (Hz)", row=len(row_order))
    fig.update_yaxes(showticklabels=True, col=1)
    for col in range(2, len(snr_grid) + 1):
        fig.update_yaxes(showticklabels=False, col=col)
    if save_plots:
        fig.write_image(os.path.join(PLOTS_DIR, f"{RESULT_NAME}_features.pdf"))
    if show_plots:
        fig.show(renderer="browser")
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S}: wrote feature-vs-frequency grid")
