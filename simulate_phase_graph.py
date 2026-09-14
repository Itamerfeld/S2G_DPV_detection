# imports and settings
import os
from copy import deepcopy
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import utils as ut

ROOT = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(ROOT, "results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
show_plots = os.environ.get("S2G_HEADLESS") != "1"

print("Imports complete.")

f0 = 636
fs = 128000
duration = 60
snr = 0
Q = 10
percent_overlap = 0.5
fft_nperseg = 8192
window = 'hann'
remove_dc = 20

# SNR is referenced to one detector bin, so snr_db is the SNR the detector actually sees
signal_bw = ut.analysis_bandwidth(fs, fft_nperseg, window)

sim_signal = ut.simulate_raw_signal(f0=f0, fs=fs, duration=duration)
noisy_sim_signal = ut.add_noise_to_signal(signal=sim_signal, snr_db=snr, fs=fs, signal_bw=signal_bw)
wn = np.random.randn(len(sim_signal))

# feature scores:
laplacian_scores = []
wasserstein_scores = []

### Plotting
signals = [sim_signal, noisy_sim_signal, wn]
column_titles = ["Pure Tonal", "Noisy (SNR=0 dB)", "White Noise"]
row_titles = ["S2G Graph", "Trans. Dist."]  # ["Phase Dist.", "S2G Graph", "Trans. Dist."]

specs = [
    # [{"type": "polar"} for _ in signals],   # Row 1: Barpolar (phase distribution)
    [{"type": "xy"}    for _ in signals],   # Row 2: graph (S2G)
    [{"type": "xy"}    for _ in signals],   # Row 3: Heatmap (transition matrix)
]

fig = make_subplots(
    rows=2, cols=len(signals),
    column_titles=column_titles,
    row_titles=row_titles,
    specs=specs,
    vertical_spacing=0.1
)

for ix, sig in enumerate(signals, start=1):
    F, T, Sxx, phase = ut.calc_spectrogram(sig, fs, nperseg=8192, percent_overlap=percent_overlap, window=window, remove_dc=remove_dc)

    # phase distribution (polar) - pick the bin closest to f0
    f_ix = np.argmin(np.abs(F - f0))
    f_phase = phase[f_ix, :]
    f_phase_wrapped = ut.wrap_phase(f_phase)
    hist_phase, bin_edges_phase = np.histogram(f_phase_wrapped, bins=Q, range=(0, 2 * np.pi), density=True)
    bin_centers = 0.5 * (bin_edges_phase[:-1] + bin_edges_phase[1:])
    bin_width = bin_edges_phase[1] - bin_edges_phase[0]

    # fig.add_trace(go.Barpolar(
    #     r=hist_phase,
    #     theta=np.degrees(bin_centers),   # Plotly's polar theta is in degrees
    #     width=np.degrees(bin_width),
    #     marker_color=hist_phase,
    #     marker_colorscale='Viridis',
    #     showlegend=False,
    # ), row=1, col=ix)

    # graph from S2G
    G = ut.get_s2g_graph(f_phase, n_levels=Q)
    fig_g = ut.draw_graph(G)
    for tr in fig_g.data:
        fig.add_trace(deepcopy(tr), row=1, col=ix)

    # transition matrix
    x_transitions = ut.get_s2g(f_phase, n_levels=Q)
    laplacian = ut.get_K(x_transitions, mode='laplacian')
    wasserstein = ut.get_K(x_transitions, mode='wasserstein')
    laplacian_scores.append(laplacian)
    wasserstein_scores.append(wasserstein)

    fig.add_trace(go.Heatmap(z=x_transitions, colorscale='Viridis', showscale=False), row=2, col=ix)

# fig.update_polars(
#     angularaxis=dict(
#         direction='counterclockwise',
#         rotation=0,
#         tickmode='array',
#         tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
#         ticktext=['0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '3π/2', '7π/4'],
#         tickfont=dict(size=20),
#     ),
#     radialaxis=dict(showticklabels=False),
# )
fig.update_layout(
    height=800, width=1200,
    font=dict(size=20),
    title_font=dict(size=20),
    margin=dict(t=140, r=140, b=100),   # leave room for titles + feature subtitles below row 2
)
fig.update_xaxes(tickfont=dict(size=20), title_font=dict(size=20))
fig.update_yaxes(tickfont=dict(size=20), title_font=dict(size=20))

# Add subtitle with features for each column (below 2nd row)
for ix, (lap, wass) in enumerate(zip(laplacian_scores, wasserstein_scores), start=1):
    fig.add_annotation(
        text=f"Laplacian: {lap:.2f}<br>Wasserstein: {wass:.2f}",
        x=0.5, y=-0.18,
        xref="x domain", yref="y domain",
        showarrow=False,
        yanchor="top",
        row=2, col=ix,
    )

# Match column/row title offsets and bold them.
title_offset = 0.06
column_title_texts = set(column_titles)
row_title_texts = set(row_titles)
for ann in fig.layout.annotations:
    ann.font = dict(size=20)
    if ann.text in column_title_texts:
        ann.text = f"<b>{ann.text}</b>"
        ann.y = 1 + title_offset
        ann.yanchor = 'bottom'
    elif ann.text in row_title_texts:
        ann.text = f"<b>{ann.text}</b>"
        ann.x = 1 + title_offset
        ann.xanchor = 'left'
# fig.write_image("../results/graph_examples_script.pdf")
fig.write_image(os.path.join(PLOTS_DIR, "simulate_phase_graph.pdf"))
if show_plots:
    fig.show()
