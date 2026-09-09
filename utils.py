# S2G utilities used by the evaluation scripts in this repository.

import networkx as nx

import numpy as np
from numpy import linalg as LA

from scipy import signal
from scipy.signal import find_peaks, hilbert
from scipy.stats import wasserstein_distance_nd

import plotly.graph_objects as go


## ====================
## DETECTORS
## ====================


class S2GDetector:
    def __init__(self, fs, nperseg, overlap=0., nfft=None, window='hanning', dc=20, crop_freq=None, norm_size=9, quantization_levels=10, mode="wasserstein", default_distance=2):
        self.fs = fs
        self.nperseg = nperseg
        self.overlap = overlap
        self.nfft = nfft
        self.window = window
        self.dc = dc
        self.crop_freq = crop_freq
        self.norm_size = norm_size
        self.quantization_levels = quantization_levels
        self.mode = mode
        self.default_distance = default_distance

    def detect(self, rx, threshold):
        K, F = self.get_feature_vector(rx)
        detections = find_peaks(K, height=threshold, distance=self.default_distance)[0]
        is_detected = len(detections) > 0
        return is_detected, detections, (K, F)

    def get_feature_vector(self, rx):
        F, T, Sxx, phasogram = calc_spectrogram(rx, self.fs, nperseg=self.nperseg, percent_overlap=self.overlap, nfft=self.nfft, window=self.window, remove_dc=self.dc, crop_freq=self.crop_freq)
        K = get_all_Ks(phasogram, F, n_levels=self.quantization_levels, mode=self.mode)
        return K, F

    # def get_score_1fb(self, rx, f0):
    #     F, T, Sxx, phasogram = calc_spectrogram(rx, self.fs, nperseg=self.nperseg, percent_overlap=self.overlap, nfft=self.nfft, window=self.window, remove_dc=self.dc, crop_freq=self.crop_freq)
    #     fix = np.where(F >= f0)[0][0]
    #     phase = phasogram[fix, :]
    #     transitions = get_s2g(phase, self.quantization_levels)
    #     score = get_K(transitions, mode=self.mode)
    #     return score


class ClassicDetector:
    def __init__(self, fs, nperseg, overlap=0., nfft=None, window='hanning', dc=20, crop_freq=None, norm_size=5, default_distance=3, default_mode='welch'):
        self.fs = fs
        self.nperseg = nperseg
        self.overlap = overlap
        self.nfft = nfft
        self.window = window
        self.dc = dc
        self.crop_freq = crop_freq
        self.norm_size = norm_size
        self.default_distance = default_distance
        self.default_mode = default_mode

    def detect(self, rx, threshold, mode=None):
        if mode is None:
            mode = self.default_mode
        scores, F = self.get_feature_vector(rx, mode=mode)
        detections = find_peaks(scores, height=threshold, distance=self.default_distance)[0]
        is_detected = len(detections) > 0
        return is_detected, detections, (scores, F)

    def get_feature_vector(self, rx, mode=None):
        if mode == 'welch':
            F, T, Sxx, phasogram = calc_spectrogram(rx, self.fs, nperseg=self.nperseg, percent_overlap=self.overlap, nfft=self.nfft, window=self.window, remove_dc=self.dc, crop_freq=self.crop_freq)
            scores = calc_welch_from_spectrogram(Sxx, normalization_window_size=self.norm_size)
        elif mode == 'correlation':
            scores, F = correlation_score(rx, self.fs, self.nperseg, self.overlap, self.window, self.dc, self.crop_freq)
        elif mode in ('demon_spectrum', 'demon'):
            scores, F = demon_spectrum(rx, self.fs, self.dc, self.crop_freq)
        elif mode in ('spectral_entropy', 'entropy'):
            scores, F = spectral_entropy(rx, self.fs, self.nperseg, self.overlap, self.window, self.dc, self.crop_freq)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        scores = (scores - np.min(scores)) / np.std(scores) if np.std(scores) > 0 else scores

        return scores, F


## ====================
## FEATURE EXTRACTORS
## ====================

def correlation_score(signal, fs, nperseg, percent_overlap, window, remove_dc=None, crop_freq=None):
    # Use linear power: Pearson on (nearly constant) CW tone power demeans the tone away.
    # Odd/even cross-energy keeps the DC power and peaks at stable tones.
    F, T, Sxx, _ = calc_spectrogram(signal, fs, nperseg, percent_overlap, window=window,
                                    remove_dc=remove_dc, crop_freq=crop_freq, logscale=False)
    Sxx_odd = Sxx[:, 1::2]
    Sxx_even = Sxx[:, ::2]
    n = min(Sxx_odd.shape[1], Sxx_even.shape[1])
    corr_scores = np.mean(Sxx_odd[:, :n] * Sxx_even[:, :n], axis=1)
    return corr_scores, F


def demon_spectrum(signal, fs, remove_dc=None, crop_freq=None):
    """
    DEMON: Detection of Envelope Modulation on Noise
    Returns envelope spectrum
    """
    analytic = hilbert(signal)
    envelope = np.abs(analytic)

    spectrum = np.fft.rfft(envelope)
    freqs = np.fft.rfftfreq(len(envelope), d=1/fs)

    if remove_dc is not None:
        dc_ix = np.where(freqs > remove_dc)[0][0]
        spectrum = spectrum[dc_ix:]
        freqs = freqs[dc_ix:]

    if crop_freq is not None:
        crop_ix = np.where(freqs > crop_freq)[0][0]
        spectrum = spectrum[:crop_ix]
        freqs = freqs[:crop_ix]

    return np.abs(spectrum), freqs


def demon_spectrum_narrowband(signal, fs, target_freq, bandwidth,
                              remove_dc=None, crop_freq=None, agg='max'):
    values, freqs = demon_spectrum(signal, fs, remove_dc=remove_dc, crop_freq=crop_freq)

    half_bw = bandwidth / 2
    band_mask = (freqs >= target_freq - half_bw) & (freqs <= target_freq + half_bw)
    if not np.any(band_mask):
        nearest_ix = np.argmin(np.abs(freqs - target_freq))
        band_mask[nearest_ix] = True
    band_values = values[band_mask]
    band_freqs = freqs[band_mask]

    if agg == 'mean':
        result = np.mean(band_values)
    elif agg == 'max':
        result = np.max(band_values)
    elif agg == 'sum':
        result = np.sum(band_values)
    else:
        raise ValueError(f"Unknown aggregation mode: {agg}")

    return result, band_values, band_freqs


def spectral_entropy(signal, fs, nperseg, percent_overlap, window, remove_dc=None, crop_freq=None):
    # Must use linear power (default spectrogram is log-scaled). A stable tone has a
    # more uniform temporal power distribution than bursty noise bins, so Shannon
    # entropy over time peaks at the tone — do not negate.
    _F, _T, _Sxx, _phasogram = calc_spectrogram(
        signal, fs=fs, nperseg=nperseg, percent_overlap=percent_overlap, window=window,
        remove_dc=remove_dc, crop_freq=crop_freq, logscale=False,
    )

    psd = np.maximum(_Sxx, 0.0)
    psd_sum = np.sum(psd, axis=1, keepdims=True) + 1e-12
    p = psd / psd_sum

    entropy = -(p * np.log(p + 1e-12)).sum(axis=1)

    return entropy, _F


def spectral_entropy_narrowband(signal, fs, nperseg, percent_overlap, window,
                                target_freq, bandwidth,
                                remove_dc=None, crop_freq=None, agg='max'):
    values, freqs = spectral_entropy(signal, fs, nperseg, percent_overlap, window,
                                     remove_dc=remove_dc, crop_freq=crop_freq)

    half_bw = bandwidth / 2
    band_mask = (freqs >= target_freq - half_bw) & (freqs <= target_freq + half_bw)
    if not np.any(band_mask):
        nearest_ix = np.argmin(np.abs(freqs - target_freq))
        band_mask[nearest_ix] = True
    band_values = values[band_mask]
    band_freqs = freqs[band_mask]

    if agg == 'mean':
        result = np.mean(band_values)
    elif agg == 'max':
        result = np.max(band_values)
    elif agg == 'sum':
        result = np.sum(band_values)
    else:
        raise ValueError(f"Unknown aggregation mode: {agg}")

    return result, band_values, band_freqs


## ====================
## S2G UTILITIES
## ====================

def normalize_data(x):
    x = x / 2  # this is necessary to avoid overflow in some cases due to super large values (probably a bug somewhere else)
    xmin = np.min(x)
    xmax = np.max(x)
    if xmax == xmin:
        return np.zeros_like(x, dtype=float)
    return (x - xmin) / (xmax - xmin)


def quantize_data(x, n_levels):
    x_quantized = np.floor(x * n_levels).astype(int)
    x_quantized[x_quantized == n_levels] = n_levels - 1  # Handle edge case
    return x_quantized


def get_s2g_transition_matrix(x_quantized, n_levels):
    transitions = np.zeros((n_levels, n_levels), dtype=int)
    for i in range(len(x_quantized)-1):
        transitions[x_quantized[i], x_quantized[i+1]] += 1
    return transitions


def get_s2g(x, n_levels):
    x = normalize_data(x)
    x = quantize_data(x, n_levels)
    transitions = get_s2g_transition_matrix(x, n_levels)
    return transitions


def get_s2g_edges(x_quantized):
    edges = []
    for i in range(1, len(x_quantized)):
        if x_quantized[i] != x_quantized[i-1]:
            edges.append((x_quantized[i-1], x_quantized[i]))
    return edges


def get_s2g_graph(x, n_levels):
    x = normalize_data(x)
    x = quantize_data(x, n_levels)
    nodes = list(range(n_levels))
    edges = get_s2g_edges(x)
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def get_K(transition_matrix, mode='wasserstein', uniform_M=None):
    if mode == 'wasserstein':
        if uniform_M is not None:
            uniform_dist = uniform_M
        else:
            uniform_dist = np.ones(transition_matrix.shape) / transition_matrix.size
        tm = np.asarray(transition_matrix, dtype=float)
        tm = tm / (tm.sum() + 1e-12)
        K = wasserstein_distance_nd(tm, uniform_dist)
    elif mode == 'edge_count':
        edge_count = np.count_nonzero(transition_matrix)
        K = edge_count / transition_matrix.size
    elif mode == 'laplacian':
        D = np.diag(np.sum(transition_matrix, axis=1))
        L = D - transition_matrix
        eigenvalues = LA.eigvals(L)
        eigenvalues = np.real(eigenvalues)
        eigenvalues = np.sort(eigenvalues)
        eigenvalues = eigenvalues[eigenvalues > 1e-6]
        K = eigenvalues[0] if eigenvalues.size > 0 else 0
    elif mode == 'all':
        K1 = get_K(transition_matrix, mode='wasserstein')
        K2 = get_K(transition_matrix, mode='edge_count')
        K3 = get_K(transition_matrix, mode='laplacian')
        K = (K1, K2, K3)
    else:
        raise ValueError(f"Unknown K calculation mode: {mode}")
    return K


def get_all_Ks(phase_matrix, frequencies, n_levels, mode='wasserstein'):
    Ks = []
    if mode == 'wasserstein':
        uniform_M = np.ones((n_levels, n_levels)) / (n_levels**2)
    else:
        uniform_M = None
    for f_idx, f in enumerate(frequencies):
        x = phase_matrix[f_idx, :]
        transition_matrix = get_s2g(x, n_levels=n_levels)
        K = get_K(transition_matrix, mode=mode, uniform_M=uniform_M)
        Ks.append(K)
    Ks = np.asarray(Ks, dtype=float)

    if mode == "edge_count":
        Ks = 1.0 - Ks
    elif mode == "laplacian":
        std = np.std(Ks)
        Ks = (np.max(Ks) - Ks) / std if std > 0 else Ks
    elif mode == "wasserstein":
        std = np.std(Ks)
        Ks = (Ks - np.min(Ks)) / std if std > 0 else Ks
    return Ks


## ====================
## SIMULATION UTILITIES
## ====================

def pink_noise(N):
    n = int(np.ceil(np.log2(N)))
    array = np.random.randn(n, N)
    array = np.cumsum(array, axis=0)
    weights = 1 / (2 ** np.arange(n))
    pink = np.dot(weights, array)
    return pink[:N]


def white_noise(N):
    return np.random.randn(N)


def simulate_raw_signal(f0, fs, duration):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * f0 * t)
    return sig


def add_noise_to_signal(signal, snr_db, fs, signal_bw, noise_type='white'):
    """
    Adds broad-band noise to a narrow-band signal based on an in-band SNR.
    """
    if noise_type == 'white':
        noise = white_noise(len(signal))
    elif noise_type == 'pink':
        noise = pink_noise(len(signal))
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    signal_power = np.mean(signal ** 2)
    desired_noise_power_in_band = signal_power / (10 ** (snr_db / 10))
    bandwidth_ratio = (fs / 2) / signal_bw
    total_desired_noise_power = desired_noise_power_in_band * bandwidth_ratio
    current_noise_power = np.mean(noise ** 2)
    scaling_factor = np.sqrt(total_desired_noise_power / current_noise_power)

    scaled_noise = noise * scaling_factor
    noisy_signal = signal + scaled_noise

    return noisy_signal


## ====================
## SIGNAL PROCESSING UTILITIES
## ====================

def rw_normalization(x, window_size=17):
    if window_size % 2 == 0:
        window_size += 1  # make it odd
    normalization_kernel = np.ones((window_size,)) / (window_size-1)
    normalization_kernel[window_size // 2] = 0
    smooth_x = signal.convolve(x, normalization_kernel, mode='same')
    ret = x / (smooth_x + 1e-10)

    ret[:window_size//2] = ret[window_size//2]
    ret[-window_size//2:] = ret[-window_size//2-1]
    return ret


def calc_spectrogram(x, fs, nperseg, percent_overlap, nfft=None, window='hamming', remove_dc=None, crop_freq=None, logscale=True):
    noverlap = int(nperseg * percent_overlap)
    step = nperseg - noverlap

    if window == 'flattop':
        window_vals = signal.windows.flattop(nperseg)
    elif hasattr(np, window):
        window_vals = getattr(np, window)(nperseg)
    elif hasattr(signal.windows, window):
        window_vals = signal.get_window(window, nperseg)
    else:
        raise ValueError(f"Window '{window}' not found in numpy or scipy.signal.windows")

    window_power = np.sum(window_vals**2)

    if nfft is None:
        nfft = nperseg

    n_segments = (len(x) - noverlap) // step
    f = np.fft.rfftfreq(nfft, 1/fs)
    t = np.arange(n_segments) * step / fs
    Sxx = np.zeros((n_segments, len(f)))
    Phase = np.zeros((n_segments, len(f)))

    for i in range(n_segments):
        start = i * step
        segment = x[start:start+nperseg]
        if len(segment) < nperseg:
            break
        segment = segment * window_vals
        spectrum = np.fft.rfft(segment, n=nfft)
        Sxx[i, :] = (np.abs(spectrum)**2) / (fs * window_power)
        Phase[i, :] = np.angle(spectrum)

    if remove_dc is not None:
        dc_band = f <= remove_dc
        f = f[~dc_band]
        Sxx = Sxx[:, ~dc_band]
        Phase = Phase[:, ~dc_band]

    if crop_freq is not None:
        if crop_freq > fs / 2:
            crop_freq = fs // 2 - 1  # Nyquist limit
        crop_band = f <= crop_freq
        f = f[crop_band]
        Sxx = Sxx[:, crop_band]
        Phase = Phase[:, crop_band]

    if logscale:
        Sxx = 10 * np.log10(Sxx + 1e-100)

    return f, t, Sxx.T, Phase.T


def calc_welch_from_spectrogram(Sxx, normalization_window_size=None):
    Pxx = np.mean(Sxx, axis=1)
    if normalization_window_size is not None:
        Pxx = rw_normalization(Pxx, window_size=normalization_window_size)
    Pxx = np.abs(Pxx - 1)
    return Pxx


## ====================
## GRAPH UTILITIES
## ====================

def draw_graph(G):
    pos = nx.spring_layout(G)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), mode='lines', showlegend=False)

    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        showlegend=False,
        marker=dict(showscale=False, colorscale='YlGnBu', reversescale=True, color=[], size=10, colorbar=dict(thickness=15, title=dict(text='Node Connections', side='right'), xanchor='left'), line_width=2))

    node_adjacencies = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))

    node_trace.marker.color = node_adjacencies
    node_trace.text = list(G.nodes())

    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, hovermode='closest', height=600, width=600,
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    return fig
