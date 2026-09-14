########################################################
# DPV sea trial — 600 m Croatia run
########################################################
# Same analysis as example.ipynb cells "complete experiment data" and
# "added recorded background noise": inbound / outbound scores in 60 s
# hops, GPS range aligned to the acoustic turn, plus a background-only
# control from the 1000 m recordings.
#
# Why these analysis settings (same as the notebook):
#   - overlap = 0 so adjacent STFT frames share no samples
#   - nperseg = 8192 at 128 kHz: ~15.6 Hz bins, and the 632 Hz inbound
#     tone is not an integer number of cycles per hop
#   - phase_mode = 'difference': wrapped frame-to-frame increment, no unwrap
#   - entropy is Alexandri local spectral entropy
#   - cosine is uncentered cosine of consecutive linear-power frames
#   - correlation is not scored through the removed TalmonDetector
#
"""Croatia 600 m sea-trial scores, aligned with example.ipynb."""

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
output_folder = os.path.join(ROOT, "results", "plots")
os.makedirs(output_folder, exist_ok=True)
to_show = os.environ.get("S2G_HEADLESS") != "1"
to_save = True
output_file_name = "dpv_600m_detection_scores_plot"

# Recording / STFT (aligned with example.ipynb)
fs = 128000
nperseg = 8192
nfft = None
overlap = 0.0
window = "hanning"
dc = 150
crop_freq = 8000
norm_size = 5

# S2G
phase_mode = "difference"
laplacian_quantization_levels = 5
wasserstein_quantization_levels = 3

# Inbound (approach) and outbound (recede) tonals on this run.
f_in = 632
f_out = 859

# Sliding 60 s analysis windows, 10 s hop (same as the notebook).
window_s = 60
step_s = 10
noise_factor = 10
# Acoustic turn / CPA marker on the spectrogram time axis.
t_turn_s = 1100

SCORE_ORDER = [
    ("demon", "DEMON"),
    ("correlation", "Correlation"),
    ("entropy", "Entropy"),
    ("cosine", "Cosine"),
    ("laplacian", "S2G Laplacian"),
    ("wasserstein", "S2G Wasserstein"),
    ("wasserstein_circular", "S2G Wasserstein circular"),
]
N_SCORES = len(SCORE_ORDER)

inbound_color = "#1f77b4"
outbound_color = "#ff7f0e"
distance_color = "#222222"
turn_color = "#7f8c8d"
cpa_color = "#d35400"

########################################################
# Detectors
########################################################
_stft = dict(
    fs=fs, nperseg=nperseg, overlap=overlap, nfft=nfft,
    window=window, dc=dc, crop_freq=crop_freq,
)

s2g_laplacian = ut.S2GDetector(
    **_stft, mode="laplacian",
    quantization_levels=laplacian_quantization_levels, phase_mode=phase_mode,
)
s2g_wasserstein = ut.S2GDetector(
    **_stft, mode="wasserstein",
    quantization_levels=wasserstein_quantization_levels, phase_mode=phase_mode,
)
s2g_wasserstein_circular = ut.S2GDetector(
    **_stft, mode="wasserstein",
    quantization_levels=wasserstein_quantization_levels, phase_mode=phase_mode,
    wasserstein_circular=True,
)
classic = ut.ClassicDetector(**_stft, norm_size=norm_size)

# name -> callable(x) -> (scores, F)
EXTRACTORS = {
    "demon": lambda x: classic.get_feature_vector(x, mode="demon"),
    "correlation": lambda x: classic.get_feature_vector(x, mode="correlation"),
    "entropy": lambda x: classic.get_feature_vector(x, mode="entropy"),
    "cosine": lambda x: classic.get_feature_vector(x, mode="cosine"),
    "laplacian": s2g_laplacian.get_feature_vector,
    "wasserstein": s2g_wasserstein.get_feature_vector,
    "wasserstein_circular": s2g_wasserstein_circular.get_feature_vector,
}


def _score_at(scores, F, f_hz):
    """Value of `scores` at the bin nearest f_hz."""
    return float(scores[ut.nearest_frequency_index(F, f_hz)])


def extract_tonal_scores(x):
    """
    Slide a window_s block every step_s and record each detector at f_in / f_out.

    Returns
    -------
    scores_in, scores_out : dict[str, np.ndarray]
    n_windows : int
    """
    step = step_s * fs
    win = window_s * fs
    scores_in = {name: [] for name, _ in SCORE_ORDER}
    scores_out = {name: [] for name, _ in SCORE_ORDER}
    stop = max(0, len(x) - win)
    for si in np.arange(0, stop, step):
        block = x[si:si + win]
        for name, _ in SCORE_ORDER:
            vec, F = EXTRACTORS[name](block)
            scores_in[name].append(_score_at(vec, F, f_in))
            scores_out[name].append(_score_at(vec, F, f_out))
    scores_in = {k: np.asarray(v, dtype=float) for k, v in scores_in.items()}
    scores_out = {k: np.asarray(v, dtype=float) for k, v in scores_out.items()}
    n_windows = len(next(iter(scores_in.values())))
    return scores_in, scores_out, n_windows


def _panel_label(fig, text, row, extra=""):
    fig.add_annotation(
        text=f"<b>{text}</b>{extra}",
        xref="x domain", yref="y domain",
        x=0.006, y=0.94, xanchor="left", yanchor="top",
        showarrow=False, bgcolor="rgba(255,255,255,0.82)", borderpad=3,
        font=dict(size=12, color="#1a1a1a"),
        row=row, col=1,
    )


def plot_sea_trial(F, T, Sxx, distances, t_distances, taxis, scores_in, scores_out, title, outfile):
    """Notebook-style stack: spectrogram, range, then one panel per detector."""
    t_turn_idx = int(np.where(T >= t_turn_s)[0][0])
    t_turn = float(T[t_turn_idx])
    ix_turn = int((t_turn_idx / len(T)) * len(taxis))
    cpa_idx = int(np.argmin(distances))
    d_cpa = float(distances[cpa_idx])
    # GPS and recorder clocks differ; shift range so the GPS CPA sits on the acoustic turn.
    gps_clock_offset = t_turn - float(t_distances[cpa_idx])
    t_distances_aligned = t_distances + gps_clock_offset
    t_cpa = t_turn
    print(f"{outfile}: t_turn={t_turn:.1f}s, CPA={d_cpa:.1f}m, GPS clock offset={gps_clock_offset:+.1f}s")
    dist_at_detectors = np.interp(taxis, t_distances_aligned, distances)

    n_rows = 2 + N_SCORES
    fig = make_subplots(
        rows=n_rows, cols=1,
        row_heights=[0.26, 0.08] + [0.0943] * N_SCORES,
        vertical_spacing=0.014,
        shared_xaxes=True,
    )

    _ds = max(1, Sxx.shape[1] // 1200)
    zmin, zmax = np.nanpercentile(Sxx, [5, 99])
    T_ds = T[::_ds]
    Sxx_ds = Sxx[:, ::_ds]
    dist_at_spec = np.interp(T_ds, t_distances_aligned, distances)
    fig.add_trace(
        go.Heatmap(
            z=Sxx_ds, x=T_ds, y=F,
            customdata=np.broadcast_to(dist_at_spec, Sxx_ds.shape),
            colorscale="Viridis", zmin=zmin, zmax=zmax,
            showscale=False,
            hovertemplate="t = %{x:.0f} s<br>f = %{y:.1f} Hz<br>range = %{customdata:.0f} m<extra>Spectrogram</extra>",
        ),
        row=1, col=1,
    )
    fig.add_hline(y=f_in, line_dash="dot", line_color="white", line_width=1.3, row=1, col=1)
    fig.add_hline(y=f_out, line_dash="dot", line_color="#ffcc80", line_width=1.3, row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=t_distances_aligned, y=distances, mode="lines",
            line=dict(color=distance_color, width=1.8),
            name="Range", showlegend=False,
            hovertemplate="t = %{x:.0f} s<br>range = %{y:.0f} m<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[t_cpa], y=[d_cpa], mode="markers",
            marker=dict(color=cpa_color, size=9, symbol="diamond"),
            name=f"CPA ({d_cpa:.0f} m)",
            hovertemplate=f"CPA: {d_cpa:.0f} m at t = {t_cpa:.0f} s<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=inbound_color, width=1.8), name=f"Inbound ({f_in} Hz)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=outbound_color, width=1.8), name=f"Outbound ({f_out} Hz)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=turn_color, width=1.5, dash="dot"), name="Turn / CPA"), row=2, col=1)

    _panel_label(fig, "Spectrogram", 1, f"   <span style='font-weight:400'>{f_in} / {f_out} Hz</span>")
    _panel_label(fig, "Range to hydrophone", 2, f"   <span style='font-weight:400'>CPA = {d_cpa:.0f} m</span>")

    for i, (name, label) in enumerate(SCORE_ORDER):
        row = 3 + i
        y_in = scores_in[name]
        y_out = scores_out[name]
        th_in = float(np.max(y_in[ix_turn:])) if ix_turn < len(y_in) else float(np.max(y_in))
        th_out = float(np.max(y_out[:ix_turn])) if ix_turn > 0 else float(np.max(y_out))
        fig.add_trace(
            go.Scatter(
                x=taxis, y=y_in, mode="lines",
                line=dict(color=inbound_color, width=1.7),
                name=f"{label} inbound", showlegend=False,
                customdata=dist_at_detectors,
                hovertemplate=(
                    "t = %{x:.0f} s<br>range = %{customdata:.0f} m<br>"
                    "inbound score = %{y:.3g}<extra>" + label + "</extra>"
                ),
            ),
            row=row, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=taxis, y=y_out, mode="lines",
                line=dict(color=outbound_color, width=1.7),
                name=f"{label} outbound", showlegend=False,
                customdata=dist_at_detectors,
                hovertemplate=(
                    "t = %{x:.0f} s<br>range = %{customdata:.0f} m<br>"
                    "outbound score = %{y:.3g}<extra>" + label + "</extra>"
                ),
            ),
            row=row, col=1,
        )
        fig.add_hline(y=th_in, line_dash="dash", line_color=inbound_color, line_width=1.2, row=row, col=1)
        fig.add_hline(y=th_out, line_dash="dash", line_color=outbound_color, line_width=1.2, row=row, col=1)
        _panel_label(
            fig, label, row,
            f"   <span style='font-weight:400;color:{inbound_color}'>in = {th_in:.3g}</span>"
            f"   <span style='font-weight:400;color:{outbound_color}'>out = {th_out:.3g}</span>",
        )

    for row in range(1, n_rows + 1):
        fig.add_vline(x=t_turn, line_dash="dot", line_color=turn_color, line_width=1, row=row, col=1)

    fig.update_traces(selector=dict(type="scatter"), cliponaxis=False)
    fig.update_layout(
        height=1540, width=1500, font=dict(size=12),
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=16)),
        margin=dict(t=80, b=55, l=70, r=30),
        plot_bgcolor="white", paper_bgcolor="white", hovermode="x",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.012, x=1, xanchor="right",
            bgcolor="rgba(255,255,255,0.85)", font=dict(size=12),
        ),
    )
    fig.update_xaxes(
        range=[float(T[0]), float(T[-1])],
        showgrid=True, gridcolor="rgba(0,0,0,0.10)", zeroline=False,
        showticklabels=False, automargin=True,
    )
    fig.update_xaxes(showticklabels=True, title_text="Time (s)", row=n_rows, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.10)", zeroline=False, automargin=True, nticks=4)
    fig.update_yaxes(
        title_text="Frequency (Hz)", range=[float(F.min()), float(F.max())],
        showgrid=False, nticks=6, row=1, col=1,
    )
    fig.update_yaxes(title_text="Range (m)", row=2, col=1)
    fig.add_annotation(
        text=f"{f_in} Hz", x=0.995, y=f_in, xref="x domain", yref="y",
        xanchor="right", yanchor="bottom", showarrow=False,
        font=dict(size=11, color="white"), row=1, col=1,
    )
    fig.add_annotation(
        text=f"{f_out} Hz", x=0.995, y=f_out, xref="x domain", yref="y",
        xanchor="right", yanchor="top", showarrow=False,
        font=dict(size=11, color="#ffcc80"), row=1, col=1,
    )
    if to_save:
        fig.write_image(os.path.join(output_folder, outfile))
    if to_show:
        fig.show(renderer="browser")


def plot_background_only(F, T, Sxx, taxis, scores_in, scores_out, outfile):
    """Background-only control: no GPS range row."""
    n_rows = 1 + N_SCORES
    fig = make_subplots(
        rows=n_rows, cols=1,
        row_heights=[0.28] + [0.72 / N_SCORES] * N_SCORES,
        vertical_spacing=0.016,
        shared_xaxes=True,
    )
    _ds = max(1, Sxx.shape[1] // 1200)
    zmin, zmax = np.nanpercentile(Sxx, [5, 99])
    fig.add_trace(
        go.Heatmap(
            z=Sxx[:, ::_ds], x=T[::_ds], y=F,
            colorscale="Viridis", zmin=zmin, zmax=zmax,
            showscale=False, hoverinfo="skip",
        ),
        row=1, col=1,
    )
    fig.add_hline(y=f_in, line_dash="dot", line_color="white", line_width=1.3, row=1, col=1)
    fig.add_hline(y=f_out, line_dash="dot", line_color="#ffcc80", line_width=1.3, row=1, col=1)
    _panel_label(fig, "Spectrogram (recorded background)", 1, f"   <span style='font-weight:400'>{f_in} / {f_out} Hz</span>")

    for i, (name, label) in enumerate(SCORE_ORDER):
        row = 2 + i
        fig.add_trace(
            go.Scatter(x=taxis, y=scores_in[name], mode="lines",
                       line=dict(color=inbound_color, width=1.7),
                       name=f"{label} @ {f_in} Hz", showlegend=False),
            row=row, col=1,
        )
        fig.add_trace(
            go.Scatter(x=taxis, y=scores_out[name], mode="lines",
                       line=dict(color=outbound_color, width=1.7),
                       name=f"{label} @ {f_out} Hz", showlegend=False),
            row=row, col=1,
        )
        _panel_label(fig, label, row)

    fig.update_layout(
        height=1400, width=1500, font=dict(size=12),
        title=dict(
            text=f"Recorded background only · bins <b>{f_in} Hz</b> / <b>{f_out} Hz</b> · 60 s windows",
            x=0.01, xanchor="left", font=dict(size=16),
        ),
        margin=dict(t=80, b=55, l=70, r=30),
        plot_bgcolor="white", paper_bgcolor="white", hovermode="x",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.012, x=1, xanchor="right",
            bgcolor="rgba(255,255,255,0.85)", font=dict(size=12),
        ),
    )
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=inbound_color, width=1.8), name=f"{f_in} Hz"), row=2, col=1)
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=outbound_color, width=1.8), name=f"{f_out} Hz"), row=2, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.10)", zeroline=False, showticklabels=False, automargin=True)
    fig.update_xaxes(showticklabels=True, title_text="Time (s)", row=n_rows, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.10)", zeroline=False, automargin=True, nticks=4)
    fig.update_yaxes(title_text="Frequency (Hz)", range=[float(F.min()), float(F.max())], showgrid=False, nticks=6, row=1, col=1)
    if to_save:
        fig.write_image(os.path.join(output_folder, outfile))
    if to_show:
        fig.show(renderer="browser")


########################################################
# LOAD DATA
########################################################
src_folder = os.path.join(ROOT, "data", "croatia", "2407_1_600m")
file_list = sorted(f for f in os.listdir(src_folder) if f.endswith(".wav"))
# Same file slice as example.ipynb (trim dock / dead time at the ends).
data = []
for file in file_list[6:-15]:
    block, _ = librosa.load(os.path.join(src_folder, file), sr=fs)
    data.append(block)
data = np.concatenate(data)
print(f"loaded DPV audio: {len(data) / fs:.1f} s at {fs} Hz")

gpx_file = os.path.join(src_folder, "600m_route.gpx")
gpx_data = gpxpy.parse(open(gpx_file))
segment = gpx_data.tracks[0].segments[0]
rx_location = gpxpy.gpx.GPXWaypoint(44.330008, 14.697141)
distances = np.array([rx_location.distance_2d(p) for p in segment.points])
t_distances = np.linspace(0, len(data) / fs, len(distances))
print(f"loaded GPS: {len(distances)} points, CPA = {distances.min():.1f} m")

noise_src_folder = os.path.join(ROOT, "data", "croatia", "2507_1_1000m")
noise_files = sorted(f for f in os.listdir(noise_src_folder) if f.endswith(".wav"))
bg_data = []
for file in noise_files[:5]:
    block, _ = librosa.load(os.path.join(noise_src_folder, file), sr=fs)
    bg_data.append(block)
bg_data = np.concatenate(bg_data)
print(f"loaded background: {len(bg_data) / fs:.1f} s")

# Tile the shorter background onto the DPV recording (same as the notebook).
n_reps = int(np.ceil(len(data) / len(bg_data)))
bg_tiled = np.tile(bg_data, n_reps)[:len(data)]
noisy_data = data + noise_factor * bg_tiled
print(f"added tiled background ×{noise_factor}")

########################################################
# FEATURE EXTRACTION
########################################################
F, T, Sxx, _ = ut.calc_spectrogram(
    data, fs=fs, nperseg=nperseg, percent_overlap=overlap, nfft=nfft,
    window=window, remove_dc=dc, crop_freq=crop_freq,
)
print("extracting clean-run scores …")
scores_in, scores_out, n_win = extract_tonal_scores(data)
taxis_clean = np.linspace(T[0], T[-1], n_win)
print(f"completed clean-run extraction ({n_win} windows)")

nF, nT, nSxx, _ = ut.calc_spectrogram(
    noisy_data, fs=fs, nperseg=nperseg, percent_overlap=overlap, nfft=nfft,
    window=window, remove_dc=dc, crop_freq=crop_freq,
)
print("extracting noisy-run scores …")
scores_in_n, scores_out_n, n_win_n = extract_tonal_scores(noisy_data)
taxis_noisy = np.linspace(nT[0], nT[-1], n_win_n)
print(f"completed noisy-run extraction ({n_win_n} windows)")

F_bg, T_bg, Sxx_bg, _ = ut.calc_spectrogram(
    bg_data, fs=fs, nperseg=nperseg, percent_overlap=overlap, nfft=nfft,
    window=window, remove_dc=dc, crop_freq=crop_freq,
)
print("extracting background-only scores …")
scores_in_bg, scores_out_bg, n_win_bg = extract_tonal_scores(bg_data)
taxis_bg = np.linspace(T_bg[0], T_bg[-1], max(n_win_bg, 1))
print(f"completed background-only extraction ({n_win_bg} windows)")

########################################################
# PLOTS
########################################################
plot_sea_trial(
    F, T, Sxx, distances, t_distances, taxis_clean, scores_in, scores_out,
    title=f"Croatia sea trial · inbound <b>{f_in} Hz</b> · outbound <b>{f_out} Hz</b> · 60 s windows",
    outfile=f"{output_file_name}_clean.pdf",
)
print("wrote clean-run figure")

plot_sea_trial(
    nF, nT, nSxx, distances, t_distances, taxis_noisy, scores_in_n, scores_out_n,
    title=(
        f"Croatia sea trial + recorded BG ×{noise_factor} · "
        f"inbound <b>{f_in} Hz</b> · outbound <b>{f_out} Hz</b> · 60 s windows"
    ),
    outfile=f"{output_file_name}_noisy.pdf",
)
print("wrote noisy-run figure")

if n_win_bg > 0:
    plot_background_only(
        F_bg, T_bg, Sxx_bg, taxis_bg, scores_in_bg, scores_out_bg,
        outfile=f"{output_file_name}_bg.pdf",
    )
    print("wrote background-only figure")
else:
    print("skipped background-only figure (recording shorter than one 60 s window)")

print("DONE")
