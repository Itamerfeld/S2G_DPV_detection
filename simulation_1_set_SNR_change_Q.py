########################################################
# SIMULATION 1: SET SNR AND CHANGE Q
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
fs = 8000  # sampling frequency
f0 = 640  # signal frequency
duration = 60  # seconds
# snr_values = [-2, -4, -6, -8]
snr = -10
num_iterations = 100
tolerance = 5  # Hz

nperseg = 1000
nfft = None # 2*nperseg
noverlap = 0.25
window = 'hanning'
dc = 20
crop_freq = 8000
norm_size = 9
mode = "wasserstein"
quantization_bins = [3, 5, 10, 20, 100]

num_thresholds = 20
threshold_range = (2, 6)
thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)

########################################################
# Simulation Run
########################################################
# if run_simulation:
#     print(f"starting simulation for SNR={snr}dB and {mode} mode, with {num_iterations} iterations per threshold")

#     # initialize empty result arrays
#     all_pd_rates = np.zeros((len(quantization_bins), len(thresholds)))
#     all_fa_rates = np.zeros((len(quantization_bins), len(thresholds)))

#     # generate raw signal
#     raw_signal = ut.simulate_raw_signal(f0, fs, duration)
#     rx = ut.add_noise_to_signal(raw_signal, snr_db=snr, fs=fs, signal_bw=fs/nperseg, noise_type='white')

#     for qi, q in enumerate(quantization_bins):
#         detector = ut.S2GDetector(fs=fs, nperseg=nperseg, overlap=noverlap, nfft=nfft, window=window, dc=dc, crop_freq=crop_freq, quantization_levels=q, mode=mode)

#         for ti, threshold in enumerate(thresholds):

#             pd = np.zeros(num_iterations)
#             fa = np.zeros(num_iterations)
#             for i in range(num_iterations):
#                 rx = ut.add_noise_to_signal(raw_signal, snr_db=snr, fs=fs, signal_bw=fs/nperseg, noise_type='white')
#                 is_detected, detections, (K, F) = detector.detect(rx, threshold)

#                 if is_detected:
#                     pd[i] = 1 if np.any((F[detections] - f0) < tolerance) else 0
#                     fa[i] = 1 if np.any((F[detections] - f0) > tolerance) else 0
                    
#                 all_pd_rates[qi, ti] = np.mean(pd)
#                 all_fa_rates[qi, ti] = np.mean(fa)

#         print(f"{time.time()}: completed simulation for Q={q}, threshold={threshold}, with {num_iterations} iterations")

if run_simulation:
    print(f"starting simulation for SNR={snr}dB and {mode} mode, with {num_iterations} iterations per threshold")

    # initialize empty result arrays
    pd_rates = np.zeros((len(quantization_bins), len(thresholds)))
    fa_rates = np.zeros((len(quantization_bins), len(thresholds)))

    # generate raw signal
    raw_signal = ut.simulate_raw_signal(f0, fs, duration)

    # loop Q
    for qi, q in enumerate(quantization_bins):
        detector = ut.S2GDetector(fs=fs, nperseg=nperseg, overlap=noverlap, nfft=nfft, window=window, dc=dc, crop_freq=crop_freq, quantization_levels=q, mode=mode)
        pd = np.zeros((num_iterations, len(thresholds)))
        fa = np.zeros((num_iterations, len(thresholds)))

        # loop iterations
        for i in range(num_iterations):
            rx = ut.add_noise_to_signal(raw_signal, snr_db=snr, fs=fs, signal_bw=fs/nperseg, noise_type='white')
            feature, F = detector.get_feature_vector(rx)

            fix = np.where(F >= f0)[0][0]
            f_mask = np.ones(len(F), dtype=bool)
            f_mask[fix] = False

            for ti, threshold in enumerate(thresholds):
                pd[i, ti] = 1 if feature[fix] > threshold else 0
                fa[i,ti] = 1 if np.any(feature[f_mask] > threshold) else 0

        pd_rates[qi, :] = np.mean(pd, axis=0)
        fa_rates[qi, :] = np.mean(fa, axis=0)

    print(f"completed type 1simulation for all parameters with mode={mode}")

########################################################
# Save Results
########################################################

if run_simulation:
    results = {"pd_rates": pd_rates, "fa_rates": fa_rates}
    with open(os.path.join(RESULTS_DIR, f"simulation_1_set_SNR_({snr}dB)_change_Q_{mode}.pkl"), "wb") as f:
        pickle.dump(results, f)

    print(f"results saved to simulation_1_set_SNR_({snr}dB)_change_Q_{mode}.pkl")

########################################################
# Plot Results
########################################################

if not run_simulation:
    with open(os.path.join(RESULTS_DIR, f"simulation_1_set_SNR_({snr}dB)_change_Q_{mode}.pkl"), "rb") as f:
        results = pickle.load(f)
    pd_rates = results["pd_rates"]
    fa_rates = results["fa_rates"]

q_colors = ['red', 'orange', 'yellow', 'green', 'blue']
q_linestyles = ['solid', 'dash', 'dot', 'dashdot', 'longdash']
q_markers = ['circle', 'square', 'diamond', 'triangle-up', 'x']
line_width = 3

fig_pd = go.Figure()
fig_fa = go.Figure()
fig_roc = go.Figure()

for qi, q in enumerate(quantization_bins):  # loop over quantization bins
    line_style = dict(color=q_colors[qi], width=line_width, dash=q_linestyles[qi])
    marker_style = dict(symbol=q_markers[qi], size=8, line=dict(width=1))
    fig_pd.add_trace(go.Scatter(x=thresholds, y=pd_rates[qi, :], mode="lines+markers", name=f"Q={q}", line=line_style, marker=marker_style))
    fig_fa.add_trace(go.Scatter(x=thresholds, y=fa_rates[qi, :], mode="lines+markers", name=f"Q={q}", line=line_style, marker=marker_style))
    fig_roc.add_trace(go.Scatter(x=fa_rates[qi, :], y=pd_rates[qi, :], mode="lines+markers", name=f"Q={q}", line=line_style, marker=marker_style))

plot_base = os.path.join(PLOTS_DIR, f"simulation_1_set_SNR_({snr}dB)_change_Q_{mode}")
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