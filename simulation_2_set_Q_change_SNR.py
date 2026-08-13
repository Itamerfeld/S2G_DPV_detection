########################################################
# SIMULATION 2: SET Q AND CHANGE SNR
########################################################

# imports
import os
import pickle

import numpy as np
import plotly.graph_objects as go

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
fs = 128000  # sampling frequency (must match simulation_3 / nb_99)
f0 = 640  # signal frequency
duration = 60  # seconds
snr_values = [0, -6, -12, -18]
# snr = -6
num_iterations = 100
tolerance = 5  # Hz

nperseg = 8000
nfft = None # 2*nperseg
noverlap = 0.25
window = 'hanning'
dc = 20
crop_freq = 2000
norm_size = 9
mode = "wasserstein"
# quantization_bins = [2, 3, 5, 10, 20]
quantization_level = 5 if 'laplacian' in mode else 3

num_thresholds = 20
threshold_range = (2, 6)  # match simulation_3 threshold grid
thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)

########################################################
# Simulation Run
########################################################
if run_simulation:
    print(f"starting simulation for Q={quantization_level} levels and {len(snr_values)} SNR values, with {num_iterations} iterations per threshold")

    # initialize empty result arrays
    pd_rates = np.zeros((len(snr_values), len(thresholds)))
    fa_rates = np.zeros((len(snr_values), len(thresholds)))

    # generate raw signal
    raw_signal = ut.simulate_raw_signal(f0, fs, duration)
    detector = ut.S2GDetector(fs=fs, nperseg=nperseg, overlap=noverlap, nfft=nfft, window=window, dc=dc, crop_freq=crop_freq, quantization_levels=quantization_level, mode=mode)

    for si, snr in enumerate(snr_values):
        pd = np.zeros((num_iterations, len(thresholds)))
        fa = np.zeros((num_iterations, len(thresholds)))
        for i in range(num_iterations):
            rx = ut.add_noise_to_signal(raw_signal, snr_db=snr, fs=fs, signal_bw=fs/nperseg, noise_type='white')
            feature, F = detector.get_feature_vector(rx)

            fix = np.where(F >= f0)[0][0]
            f_mask = np.ones(len(F), dtype=bool)
            f_mask[fix] = False

            for ti, threshold in enumerate(thresholds):
                pd[i, ti] = 1 if feature[fix] > threshold else 0
                fa[i, ti] = 1 if np.any(feature[f_mask] > threshold) else 0

        pd_rates[si, :] = np.mean(pd, axis=0)
        fa_rates[si, :] = np.mean(fa, axis=0)

        print(f"completed simulation for SNR={snr}dB, threshold={threshold}, with {num_iterations} iterations")

    print(f"completed simulation for all parameters with mode={mode}")

########################################################
# Save Results
########################################################

if run_simulation:
    results = {"pd_rates": pd_rates, "fa_rates": fa_rates}
    with open(os.path.join(RESULTS_DIR, f"simulation_2_set_Q_({quantization_level})_change_SNR_{mode}.pkl"), "wb") as f:
        pickle.dump(results, f)

    print(f"results saved to simulation_2_set_Q_({quantization_level})_change_SNR_{mode}.pkl")

########################################################
# Plot Results
########################################################

if not run_simulation:
    with open(os.path.join(RESULTS_DIR, f"simulation_2_set_Q_({quantization_level})_change_SNR_{mode}.pkl"), "rb") as f:
        results = pickle.load(f)
    pd_rates = results["pd_rates"]
    fa_rates = results["fa_rates"]

q_colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple']
q_linestyles = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']
q_markers = ['circle', 'square', 'diamond', 'triangle-up', 'x', 'cross']
line_width = 3

fig_pd = go.Figure()
fig_fa = go.Figure()
fig_roc = go.Figure()

for si, snr in enumerate(snr_values):  # loop over SNR values
    line_style = dict(color=q_colors[si], width=line_width, dash=q_linestyles[si])
    marker_style = dict(symbol=q_markers[si], size=8, line=dict(width=1))
    fig_pd.add_trace(go.Scatter(x=thresholds, y=pd_rates[si, :], mode="lines+markers", name=f"SNR={snr}dB", line=line_style, marker=marker_style))
    fig_fa.add_trace(go.Scatter(x=thresholds, y=fa_rates[si, :], mode="lines+markers", name=f"SNR={snr}dB", line=line_style, marker=marker_style))
    fig_roc.add_trace(go.Scatter(x=fa_rates[si, :], y=pd_rates[si, :], mode="lines+markers", name=f"SNR={snr}dB", line=line_style, marker=marker_style))

plot_base = os.path.join(PLOTS_DIR, f"simulation_2_set_Q_({quantization_level})_change_SNR_{mode}")
layout_kwargs = dict(height=500, width=500, title_text="", legend=dict(font=dict(size=16)))
axis_kwargs = dict(title_font=dict(size=20))

fig_pd.update_xaxes(title_text="Threshold", **axis_kwargs)
fig_pd.update_yaxes(title_text="Detection Rate", **axis_kwargs)
fig_pd.update_layout(**layout_kwargs)
fig_pd.write_image(f"{plot_base}_pd.pdf")

fig_fa.update_xaxes(title_text="Threshold", **axis_kwargs)
fig_fa.update_yaxes(title_text="False Alarm Rate", **axis_kwargs)
fig_fa.update_layout(**layout_kwargs)
fig_fa.write_image(f"{plot_base}_fa.pdf")

fig_roc.update_xaxes(title_text="False Alarm Rate", **axis_kwargs)
fig_roc.update_yaxes(title_text="Detection Rate", **axis_kwargs)
fig_roc.update_layout(**layout_kwargs)
fig_roc.write_image(f"{plot_base}_roc.pdf")

if show_plots:
    fig_pd.show(renderer="browser")
    fig_fa.show(renderer="browser")
    fig_roc.show(renderer="browser")