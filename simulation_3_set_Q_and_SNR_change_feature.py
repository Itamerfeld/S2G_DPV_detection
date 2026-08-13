########################################################
# SIMULATION 1: SET SNR AND CHANGE Q
########################################################

# imports
import os
from datetime import datetime
import pickle

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

fs = 128000  # sampling frequency
f0 = 640  # signal frequency
duration = 60  # seconds
snr = -18
num_iterations = 100
tolerance = 5  # Hz

nperseg = 8000
nfft = None # 2*nperseg
noverlap = 0.25
window = 'hanning'
dc = 20
crop_freq = 2000
norm_size = 9
num_thresholds = 20
threshold_range = (2, 6)
thresholds = np.linspace(threshold_range[0], threshold_range[1], num_thresholds)

########################################################
# Initialize Detectors
########################################################

detectors = {
    'laplacian': ut.S2GDetector(fs=fs, nperseg=nperseg, overlap=noverlap, nfft=nfft, window=window, dc=dc, crop_freq=crop_freq, quantization_levels=5, mode="laplacian"),
    'wasserstein': ut.S2GDetector(fs=fs, nperseg=nperseg, overlap=noverlap, nfft=nfft, window=window, dc=dc, crop_freq=crop_freq, quantization_levels=3, mode="wasserstein"),
    'correlation': ut.ClassicDetector(fs=fs, nperseg=nperseg, overlap=noverlap, nfft=nfft, window=window, dc=dc, crop_freq=crop_freq, norm_size=norm_size, mode="correlation"),
    'entropy': ut.ClassicDetector(fs=fs, nperseg=nperseg, overlap=noverlap, nfft=nfft, window=window, dc=dc, crop_freq=crop_freq, norm_size=norm_size, mode="entropy"),
    'demon': ut.ClassicDetector(fs=fs, nperseg=nperseg, overlap=noverlap, nfft=nfft, window=window, dc=dc, crop_freq=crop_freq, norm_size=norm_size, mode="demon")
    # 'welch': ut.ClassicDetector(fs=fs, nperseg=nperseg, overlap=noverlap, nfft=nfft, window=window, dc=dc, crop_freq=crop_freq, norm_size=norm_size, mode="welch")
}

########################################################
# Simulation Run
########################################################

if run_simulation:
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: starting simulation type 3 for SNR={snr}dB, with {num_iterations} iterations per threshold")

    raw_signal = ut.simulate_raw_signal(f0, fs, duration)

    all_results = {}

    for detector_name, detector in detectors.items():
        detected_all_iter = np.zeros((num_iterations, len(thresholds)))
        fa_all_iter = np.zeros((num_iterations, len(thresholds)))

        for i in range(num_iterations):
            rx = ut.add_noise_to_signal(raw_signal, snr_db=snr, fs=fs, signal_bw=fs/nperseg, noise_type='white')
            feature, F = detector.get_feature_vector(rx)
            fix = np.where(F >= f0)[0][0]
            f_mask = np.ones(len(F), dtype=bool)
            f_mask[fix] = False

            for ti, threshold in enumerate(thresholds):
                detected_all_iter[i, ti] = 1 if feature[fix] > threshold else 0
                fa_all_iter[i,ti] = 1 if np.any(feature[f_mask] > threshold) else 0

        all_results[detector_name] = {"pd_rates": np.mean(detected_all_iter, axis=0), "fa_rates": np.mean(fa_all_iter, axis=0)}
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: completed simulation for {detector_name} with SNR={snr}dB")

########################################################
# Save Results
########################################################

if run_simulation:
    with open(os.path.join(RESULTS_DIR, f"simulation_3_SNR_{snr}dB.pkl"), "wb") as f:
        pickle.dump(all_results, f)

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: results saved to simulation_3_SNR_{snr}dB.pkl")

########################################################
# Plot Results
########################################################

if not run_simulation:
    with open(os.path.join(RESULTS_DIR, f"simulation_3_SNR_{snr}dB.pkl"), "rb") as f:
        all_results = pickle.load(f)

detector_colors = {
    'laplacian': 'red',
    'wasserstein': 'orange',
    'correlation': 'green',
    'entropy': 'blue',
    'demon': 'purple',
    'welch': 'brown',
}
detector_linestyles = {
    'laplacian': 'solid',
    'wasserstein': 'dash',
    'correlation': 'dot',
    'entropy': 'dashdot',
    'demon': 'longdash',
    'welch': 'longdashdot',
}
detector_markers = {
    'laplacian': 'circle',
    'wasserstein': 'square',
    'correlation': 'diamond',
    'entropy': 'triangle-up',
    'demon': 'x',
    'welch': 'cross',
}
line_width = 3

fig = make_subplots(rows=1, cols=1, shared_xaxes=False)
for detector_name, detector_results in all_results.items():

    pd_rates = detector_results["pd_rates"]
    fa_rates = detector_results["fa_rates"]
    line_style = dict(color=detector_colors[detector_name], width=line_width, dash=detector_linestyles[detector_name])
    marker_style = dict(symbol=detector_markers[detector_name], size=8, line=dict(width=1))

    # plot ROC curve
    # fig.add_trace(go.Scatter(x=thresholds, y=pd_rates, mode="lines+markers", name=f"{detector_name}", showlegend=False, line=line_style, marker=marker_style), row=1, col=1)
    # fig.add_trace(go.Scatter(x=thresholds, y=fa_rates, mode="lines+markers", name=f"{detector_name}", showlegend=False, line=line_style, marker=marker_style), row=1, col=2)
    fig.add_trace(go.Scatter(x=fa_rates, y=pd_rates, mode="lines+markers", name=f"{detector_name}", line=line_style, marker=marker_style), row=1, col=1)

# fig.update_xaxes(title_text="Threshold", tickfont=dict(size=20), title_font=dict(size=20), row=1, col=1)
# fig.update_yaxes(title_text="False Alarm Probability", tickfont=dict(size=20), title_font=dict(size=20), row=1, col=1)
# fig.update_yaxes(title_text="Detection Probability", tickfont=dict(size=20), title_font=dict(size=20), row=1, col=2)
fig.update_xaxes(title_text="False Alarm Rate", tickfont=dict(size=16), title_font=dict(size=16), row=1, col=1)
fig.update_yaxes(title_text="Detection Rate", tickfont=dict(size=16), title_font=dict(size=16), row=1, col=1)
fig.update_layout(height=500, width=500)

if show_plots:
    fig.show()

if save_plots:
    fig.write_image(os.path.join(PLOTS_DIR, f"simulation_3_SNR_{snr}dB.pdf"))
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: plot saved to results/plots/simulation_3_SNR_{snr}dB.pdf")
