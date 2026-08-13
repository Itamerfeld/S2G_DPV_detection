########################################################
# DPV ANALYSIS - SCOOTER EXPERIMENT 600m
########################################################

# imports

import os
import gpxpy
import librosa
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import utils as ut

ROOT = os.path.dirname(os.path.abspath(__file__))


########################################################
# PARAMETERS
########################################################

# output
output_folder = os.path.join(ROOT, 'results', 'plots')
os.makedirs(output_folder, exist_ok=True)
to_show = os.environ.get("S2G_HEADLESS") != "1"
to_save = True
plot_title = 'DPV experiment 600m'
output_file_name = 'dpv_600m_detection_scores_plot'

# SpectrogramParameters
nperseg = 8000
overlap = 0.5
window = 'hanning'
dc = 20
crop_freq = 8000
norm_size = 9

# S2G Parameters
laplacian_quantization_levels = 5
wasserstein_quantization_levels = 3

# F0 Parameters
f = 633  # Hz

########################################################
# LOAD DATA
########################################################

# DPV data
src_folder = os.path.join(ROOT, 'data', 'croatia', '2407_1_600m')
file_list = os.listdir(src_folder)
file_list = [f for f in file_list if f.endswith('.wav')]
file_list.sort()

data = []
for file in file_list[6:-15]:
    _data, fs = librosa.load(os.path.join(src_folder, file))
    data.append(_data)
data = np.concatenate(data)
print("completed loading data")


# GPX
gpx_file = os.path.join(src_folder, '600m_route.gpx')
gpx_data = gpxpy.parse(open(gpx_file))

# get the first track
track = gpx_data.tracks[0]

# get the first segment
segment = track.segments[0]

rx_location = gpxpy.gpx.GPXWaypoint(44.330008, 14.697141)

distances = []
for point in segment.points:
    distance = rx_location.distance_2d(point)
    distances.append(distance)

distances = np.array(distances)
t_distances = np.linspace(0, len(data)/fs, len(distances))
print("completed loading GPS data")


# BG Noise WAVs
noise_src_folder = os.path.join(ROOT, 'data', 'croatia', '2507_1_1000m')
file_list = os.listdir(noise_src_folder)
file_list = [f for f in file_list if f.endswith('.wav')]
file_list.sort()

bg_data = []
for file in file_list[:5]:
    _data, fs = librosa.load(os.path.join(noise_src_folder, file))
    bg_data.append(_data)
bg_data = np.concatenate(bg_data)
print(f"bg_data final shape: {bg_data.shape} of duration: {len(bg_data)/fs} seconds")

# Add BG Noise to data
noise_factor = 10
bg_in_shape = np.concatenate([bg_data, bg_data, bg_data, bg_data, bg_data, bg_data, bg_data, bg_data])
noisy_data = data + noise_factor * bg_in_shape[:len(data)]
print("completed adding noise to data")


########################################################
# FEATURE EXTRACTION - no noise
########################################################

welch_detector = ut.WelchDetector(fs, nperseg=nperseg, overlap=overlap, window=window, dc=dc, crop_freq=crop_freq, norm_size=norm_size)
talmon_detector = ut.TalmonDetector(fs, nperseg=nperseg, overlap=overlap, window=window, dc=dc, crop_freq=crop_freq, norm_size=norm_size)
s2g_laplacian_detector = ut.S2GDetector(fs, nperseg=nperseg, overlap=overlap, window=window, dc=dc, crop_freq=crop_freq, mode='laplacian', quantization_levels=laplacian_quantization_levels)
s2g_wasserstein_detector = ut.S2GDetector(fs, nperseg=nperseg, overlap=overlap, window=window, dc=dc, crop_freq=crop_freq, mode='wasserstein', quantization_levels=wasserstein_quantization_levels)


F, T, Sxx, phasogram = ut.calc_spectrogram(data, fs=fs, nperseg=nperseg, percent_overlap=overlap, window=window, remove_dc=dc, crop_freq=crop_freq)

demon_scores = []
correlation_scores = []
entropy_scores = []
laplacian_s2g_scores = []
wasserstein_s2g_scores = []

step_size = 10 * fs
for si in np.arange(start=0, stop=(len(data)-60*fs), step=step_size):
    _data = data[si:si+60*fs]
    _time = (si + len(_data) / 2) // fs  # time corresponding to the center of the window

    # demon
    demon, _, _ = ut.demon_spectrum_narrowband(_data, fs, f, 3, remove_dc=dc, crop_freq=crop_freq, agg='max')
    demon_scores.append(demon)

    # correlation
    TD = talmon_detector.get_score_1fb(_data, f)
    correlation_scores.append(TD)

    # entropy
    entropy, _, _ = ut.spectral_entropy_narrowband(_data, fs, nperseg, overlap, window, target_freq=f, bandwidth=3, remove_dc=dc, crop_freq=crop_freq, agg='max')
    entropy_scores.append(entropy)

    # s2g score - laplacian
    K = s2g_laplacian_detector.get_score_1fb(_data, f)
    laplacian_s2g_scores.append(K)

    # s2g score - wasserstein
    K = s2g_wasserstein_detector.get_score_1fb(_data, f)
    wasserstein_s2g_scores.append(K)

detectors_taxis_clean = np.linspace(T[0], T[-1], len(laplacian_s2g_scores))

print("completed all feature extraction - no noise")

########################################################
# FEATURE EXTRACTION - signal with noise
########################################################

nF, nT, nSxx, nphasogram = ut.calc_spectrogram(noisy_data, fs=fs, nperseg=nperseg, percent_overlap=overlap, window=window, remove_dc=dc, crop_freq=crop_freq)

demon_scores_n = []
correlation_scores_n = []
entropy_scores_n = []
laplacian_s2g_scores_n = []
wasserstein_s2g_scores_n = []

step_size = 10 * fs
for si in np.arange(start=0, stop=(len(data)-60*fs), step=step_size):
    _data = noisy_data[si:si+60*fs]
    _time = (si + len(_data) / 2) // fs  # time corresponding to the center of the window

    # demon
    demon, _, _ = ut.demon_spectrum_narrowband(_data, fs, f, 3, remove_dc=dc, crop_freq=crop_freq, agg='max')
    demon_scores_n.append(demon)

    # correlation
    TD = talmon_detector.get_score_1fb(_data, f)
    correlation_scores_n.append(TD)


    # entropy
    entropy, _, _ = ut.spectral_entropy_narrowband(_data, fs, nperseg, overlap, window, target_freq=f, bandwidth=3, remove_dc=dc, crop_freq=crop_freq, agg='max')
    entropy_scores_n.append(entropy)

    # s2g score - laplacian
    K = s2g_laplacian_detector.get_score_1fb(_data, f)
    laplacian_s2g_scores_n.append(K)

    # s2g score - wasserstein
    K = s2g_wasserstein_detector.get_score_1fb(_data, f)
    wasserstein_s2g_scores_n.append(K)

detectors_taxis_noisy = np.linspace(nT[0], nT[-1], len(laplacian_s2g_scores_n))

print("completed all feature extraction - with noise")

########################################################
# FEATURE EXTRACTION - only BG noise
########################################################

F_bg, T_bg, Sxx_bg, phasogram_bg = ut.calc_spectrogram(bg_data, fs=fs, nperseg=nperseg, percent_overlap=overlap, window=window, remove_dc=dc, crop_freq=crop_freq)

demon_scores_bg = []
correlation_scores_bg = []
entropy_scores_bg = []
laplacian_s2g_scores_bg = []
wasserstein_s2g_scores_bg = []

step_size = 10 * fs
bg_stop = max(0, len(bg_data) - 60 * fs)
for si in np.arange(start=0, stop=bg_stop, step=step_size):
    _data = bg_data[si:si+60*fs]
    _time = (si + len(_data) / 2) // fs  # time corresponding to the center of the window

    # demon
    demon, _, _ = ut.demon_spectrum_narrowband(_data, fs, f, 3, remove_dc=dc, crop_freq=crop_freq, agg='max')
    demon_scores_bg.append(demon)

    # correlation
    TD = talmon_detector.get_score_1fb(_data, f)
    correlation_scores_bg.append(TD)

    # entropy
    entropy, _, _ = ut.spectral_entropy_narrowband(_data, fs, nperseg, overlap, window, target_freq=f, bandwidth=3, remove_dc=dc, crop_freq=crop_freq, agg='max')
    entropy_scores_bg.append(entropy)

    # s2g score - laplacian
    K = s2g_laplacian_detector.get_score_1fb(_data, f)
    laplacian_s2g_scores_bg.append(K)

    # s2g score - wasserstein
    K = s2g_wasserstein_detector.get_score_1fb(_data, f)
    wasserstein_s2g_scores_bg.append(K)

detectors_taxis_bg = np.linspace(nT[0], nT[-1], len(laplacian_s2g_scores_bg))

print("completed all feature extraction - only BG noise")


########################################################
# PLOT RESULTS
########################################################


# plot original results without noise

fig = make_subplots(rows=7,
                    cols=1, 
                    row_heights=[0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 
                    vertical_spacing=0.01, 
                    shared_xaxes=True,
                    row_titles=["Spectrogram", "Distance", "Demon", "Correlation", "Entropy", "S2G - Laplacian", "S2G - Wasserstein"])

# no noise
_ds = max(1, Sxx.shape[1] // 1000)
fig.add_trace(go.Heatmap(z=Sxx[:, ::_ds], x=T[::_ds], y=F, colorscale='viridis', showlegend=False, showscale=False), row=1, col=1)
fig.add_trace(go.Scatter(x=t_distances, y=distances, name='Distance', marker_color='black', showlegend=False), row=2, col=1)
t_turn = np.where(T >= 1100)[0][0]
ix_turn = int((t_turn / len(T)) * len(detectors_taxis_clean))
print(f"t_turn: {t_turn}, ix_turn: {ix_turn}")


fig.add_trace(go.Scatter(x=detectors_taxis_clean, y=demon_scores, name=f'Demon f={f} Hz', marker_color='blue', showlegend=False), row=3, col=1)
# dix = np.where(detectors_taxis_clean >= 804)[0][0]
# fig.add_vline(x=detectors_taxis_clean[dix], line_dash="dash", line_color="red", line_width=2, row=3, col=1)
th = np.max(demon_scores[ix_turn:])
fig.add_hline(y=th, line_dash="dash", line_color="red", line_width=2, row=3, col=1)

fig.add_trace(go.Scatter(x=detectors_taxis_clean, y=correlation_scores, name=f'Correlation f={f} Hz', marker_color='blue', showlegend=False), row=4, col=1)
# dix = np.where(detectors_taxis_clean >= 814)[0][0]
# fig.add_vline(x=detectors_taxis_clean[dix], line_dash="dash", line_color="red", line_width=2, row=4, col=1)
th = np.max(correlation_scores[ix_turn:])
fig.add_hline(y=th, line_dash="dash", line_color="red", line_width=2, row=4, col=1)

fig.add_trace(go.Scatter(x=detectors_taxis_clean, y=entropy_scores, name=f'Entropy f={f} Hz', marker_color='blue', showlegend=False), row=5, col=1)
# dix = np.where(detectors_taxis_clean >= 886)[0][0]
# fig.add_vline(x=detectors_taxis_clean[dix], line_dash="dash", line_color="red", line_width=2, row=5, col=1)
th = np.max(entropy_scores[ix_turn:])
fig.add_hline(y=th, line_dash="dash", line_color="red", line_width=2, row=5, col=1)

fig.add_trace(go.Scatter(x=detectors_taxis_clean, y=laplacian_s2g_scores, name=f'Score f={f} Hz (Laplacian)', marker_color='green', showlegend=False), row=6, col=1)
# dix = np.where(detectors_taxis_clean >= 886)[0][0]
# fig.add_vline(x=detectors_taxis_clean[dix], line_dash="dash", line_color="red", line_width=2, row=6, col=1)
th = np.max(laplacian_s2g_scores[ix_turn:])
fig.add_hline(y=th, line_dash="dash", line_color="red", line_width=2, row=6, col=1)

fig.add_trace(go.Scatter(x=detectors_taxis_clean, y=wasserstein_s2g_scores, name=f'Score f={f} Hz (Wasserstein)', marker_color='green', showlegend=False), row=7, col=1)
# dix = np.where(detectors_taxis_clean >= 494)[0][0] 
# fig.add_vline(x=detectors_taxis_clean[dix], line_dash="dash", line_color="red", line_width=2, row=7, col=1)
th = np.max(wasserstein_s2g_scores[ix_turn:])
fig.add_hline(y=th, line_dash="dash", line_color="red", line_width=2, row=7, col=1)

# update axes
fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
fig.update_yaxes(title_text="Dist. (m)", row=2, col=1)
fig.update_yaxes(title_text="DEMON", row=3, col=1)
fig.update_yaxes(title_text="Correlation", row=4, col=1)
fig.update_yaxes(title_text="Entropy", row=5, col=1)
fig.update_yaxes(title_text="S2G - Laplacian", row=6, col=1)
fig.update_yaxes(title_text="S2G - Wasserstein", row=7, col=1)
fig.update_xaxes(title_text="Time (s)", row=7, col=1)
fig.update_layout(height=800, width=1400)
fig.update_annotations(font_size=12)

if to_save:
    fig.write_image(os.path.join(output_folder, f"{output_file_name}_clean.pdf"))
if to_show:
    fig.show(renderer="browser")
print("completed plotting original results without noise")


# plot results with noise
fig = make_subplots(rows=7,
                    cols=1, 
                    row_heights=[0.4, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], 
                    vertical_spacing=0.01, 
                    shared_xaxes=True,
                    row_titles=["Spectrogram", "Distance", "Demon", "Correlation", "Entropy", "S2G - Laplacian", "S2G - Wasserstein"])

fig.add_trace(go.Heatmap(z=nSxx[:, ::_ds], x=nT[::_ds], y=nF, colorscale='viridis', showlegend=False, showscale=False), row=1, col=1)
fig.add_trace(go.Scatter(x=t_distances, y=distances, name='Distance', marker_color='black', showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(x=detectors_taxis_noisy, y=demon_scores_n, name=f'Demon f={f} Hz', marker_color='blue', showlegend=False), row=3, col=1)
# dix = np.where(detectors_taxis_noisy >= 804)[0][0]
# fig.add_vline(x=detectors_taxis_noisy[dix], line_dash="dash", line_color="red", line_width=2, row=3, col=1)
th = np.max(demon_scores_n[ix_turn:])
fig.add_hline(y=th, line_dash="dash", line_color="red", line_width=2, row=3, col=1)

fig.add_trace(go.Scatter(x=detectors_taxis_noisy, y=correlation_scores_n, name=f'Correlation f={f} Hz', marker_color='blue', showlegend=False), row=4, col=1)
# dix = np.where(detectors_taxis_noisy >= 814)[0][0]
# fig.add_vline(x=detectors_taxis_noisy[dix], line_dash="dash", line_color="red", line_width=2, row=4, col=1)
th = np.max(correlation_scores_n[ix_turn:])
fig.add_hline(y=th, line_dash="dash", line_color="red", line_width=2, row=4, col=1)

fig.add_trace(go.Scatter(x=detectors_taxis_noisy, y=entropy_scores_n, name=f'Entropy f={f} Hz', marker_color='blue', showlegend=False), row=5, col=1)
# dix = np.where(detectors_taxis_noisy >= 886)[0][0]
# fig.add_vline(x=detectors_taxis_noisy[dix], line_dash="dash", line_color="red", line_width=2, row=5, col=1)
th = np.max(entropy_scores_n[ix_turn:])
fig.add_hline(y=th, line_dash="dash", line_color="red", line_width=2, row=5, col=1)

fig.add_trace(go.Scatter(x=detectors_taxis_noisy, y=laplacian_s2g_scores_n, name=f'Score f={f} Hz (Laplacian)', marker_color='green', showlegend=False), row=6, col=1)
# dix = np.where(detectors_taxis_noisy >= 886)[0][0]
# fig.add_vline(x=detectors_taxis_noisy[dix], line_dash="dash", line_color="red", line_width=2, row=6, col=1)
th = np.max(laplacian_s2g_scores_n[ix_turn:])
fig.add_hline(y=th, line_dash="dash", line_color="red", line_width=2, row=6, col=1)

fig.add_trace(go.Scatter(x=detectors_taxis_noisy, y=wasserstein_s2g_scores_n, name=f'Score f={f} Hz (Wasserstein)', marker_color='green', showlegend=False), row=7, col=1)
# dix = np.where(detectors_taxis_noisy >= 494)[0][0]
# fig.add_vline(x=detectors_taxis_noisy[dix], line_dash="dash", line_color="red", line_width=2, row=7, col=1)
th = np.max(wasserstein_s2g_scores_n[ix_turn:])
fig.add_hline(y=th, line_dash="dash", line_color="red", line_width=2, row=7, col=1)

# update axes
fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
fig.update_yaxes(title_text="Dist. (m)", row=2, col=1)
fig.update_yaxes(title_text="DEMON", row=3, col=1)
fig.update_yaxes(title_text="Correlation", row=4, col=1)
fig.update_yaxes(title_text="Entropy", row=5, col=1)
fig.update_yaxes(title_text="S2G - Laplacian", row=6, col=1)
fig.update_yaxes(title_text="S2G - Wasserstein", row=7, col=1)
fig.update_xaxes(title_text="Time (s)", row=7, col=1)


fig.update_layout(height=800, width=1400)
fig.update_annotations(font_size=12)

if to_save:
    fig.write_image(os.path.join(output_folder, f"{output_file_name}_noisy.pdf"))
if to_show:
    fig.show(renderer="browser")
print("completed plotting results with noise")


# plot results with BG noise only
fig = make_subplots(rows=6,
                    cols=1, 
                    row_heights=[0.4, 0.12, 0.12, 0.12, 0.12, 0.12], 
                    vertical_spacing=0.01, 
                    shared_xaxes=True,
                    row_titles=["Spectrogram", "Demon", "Correlation", "Entropy", "S2G - Laplacian", "S2G - Wasserstein"])

fig.add_trace(go.Heatmap(z=Sxx_bg[:, ::_ds], x=T_bg[::_ds], y=F_bg, colorscale='viridis', showlegend=False, showscale=False), row=1, col=1)

fig.add_trace(go.Scatter(x=detectors_taxis_bg, y=demon_scores_bg, name=f'Demon f={f} Hz', marker_color='blue', showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(x=detectors_taxis_bg, y=correlation_scores_bg, name=f'Correlation f={f} Hz', marker_color='blue', showlegend=False), row=3, col=1)

fig.add_trace(go.Scatter(x=detectors_taxis_bg, y=entropy_scores_bg, name=f'Entropy f={f} Hz', marker_color='blue', showlegend=False), row=4, col=1)

fig.add_trace(go.Scatter(x=detectors_taxis_bg, y=laplacian_s2g_scores_bg, name=f'Score f={f} Hz (Laplacian)', marker_color='green', showlegend=False), row=5, col=1)

fig.add_trace(go.Scatter(x=detectors_taxis_bg, y=wasserstein_s2g_scores_bg, name=f'Score f={f} Hz (Wasserstein)', marker_color='green', showlegend=False), row=6, col=1)

# update axes
fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
fig.update_yaxes(title_text="DEMON", row=2, col=1)
fig.update_yaxes(title_text="Correlation", row=3, col=1)
fig.update_yaxes(title_text="Entropy", row=4, col=1)
fig.update_yaxes(title_text="S2G - Laplacian", row=5, col=1)
fig.update_yaxes(title_text="S2G - Wasserstein", row=6, col=1)
fig.update_xaxes(title_text="Time (s)", row=6, col=1)


fig.update_layout(height=800, width=1400)
fig.update_annotations(font_size=12)

if to_save:
    fig.write_image(os.path.join(output_folder, f"{output_file_name}_bg.pdf"))
if to_show:
    fig.show(renderer="browser")
print("completed plotting results with BG noise only")

print("DONE")
