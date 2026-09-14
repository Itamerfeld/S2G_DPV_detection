# S2G utilities used by the evaluation scripts in this repository.

import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import networkx as nx

import numpy as np
from numpy import linalg as LA

from scipy import signal
from scipy.signal import find_peaks, hilbert

import ot

import plotly.graph_objects as go


## ====================
## DETECTORS
## ====================


class S2GDetector:
    def __init__(self, fs, nperseg, overlap=0., nfft=None, window='hanning', dc=20, crop_freq=None, norm_size=9, quantization_levels=10, mode="wasserstein", default_distance=2, phase_mode='wrapped', wasserstein_circular=False):
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
        self.phase_mode = phase_mode
        self.wasserstein_circular = wasserstein_circular

    def detect(self, rx, threshold, mode=None, phase_mode=None, quantization_levels=None, wasserstein_circular=None):
        K, F = self.get_feature_vector(rx, mode=mode, phase_mode=phase_mode, quantization_levels=quantization_levels, wasserstein_circular=wasserstein_circular)
        detections = find_peaks(K, height=threshold, distance=self.default_distance)[0]
        is_detected = len(detections) > 0
        return is_detected, detections, (K, F)

    def get_feature_vector(self, rx, mode=None, phase_mode=None, quantization_levels=None, wasserstein_circular=None):
        if mode is None:
            mode = self.mode
        if phase_mode is None:
            phase_mode = self.phase_mode
        if quantization_levels is None:
            quantization_levels = self.quantization_levels
        if wasserstein_circular is None:
            wasserstein_circular = self.wasserstein_circular
        F, T, Sxx, phasogram = calc_spectrogram(rx, self.fs, nperseg=self.nperseg, percent_overlap=self.overlap, nfft=self.nfft, window=self.window, remove_dc=self.dc, crop_freq=self.crop_freq)
        K = get_all_Ks(phasogram, F, n_levels=quantization_levels, mode=mode, phase_mode=phase_mode, wasserstein_circular=wasserstein_circular)
        return K, F


class ClassicDetector:
    def __init__(self, fs, nperseg, overlap=0., nfft=None, window='hanning', dc=20, crop_freq=None, norm_size=5, default_distance=3, default_mode='welch', correlation_whiten=True, mode=None):
        self.fs = fs
        self.nperseg = nperseg
        self.overlap = overlap
        self.nfft = nfft
        self.window = window
        self.dc = dc
        self.crop_freq = crop_freq
        self.norm_size = norm_size
        self.default_distance = default_distance
        # `mode` is accepted as an alias for default_mode (used by the simulation scripts).
        self.default_mode = default_mode if mode is None else mode
        # rw whitening of the correlation score helps on static recordings but hurts when
        # the source is moving (the tonal is modulated, so the local background tracks it).
        self.correlation_whiten = correlation_whiten

    def detect(self, rx, threshold, mode=None):
        if mode is None:
            mode = self.default_mode
        scores, F = self.get_feature_vector(rx, mode=mode)
        detections = find_peaks(scores, height=threshold, distance=self.default_distance)[0]
        is_detected = len(detections) > 0
        return is_detected, detections, (scores, F)

    def get_feature_vector(self, rx, mode=None):
        if mode is None:
            mode = self.default_mode
        if mode == 'welch':
            F, T, Sxx, phasogram = calc_spectrogram(rx, self.fs, nperseg=self.nperseg, percent_overlap=self.overlap, nfft=self.nfft, window=self.window, remove_dc=self.dc, crop_freq=self.crop_freq)
            scores = calc_welch_from_spectrogram(Sxx, normalization_window_size=self.norm_size)
        elif mode == 'correlation':
            scores, F = correlation_score(rx, self.fs, self.nperseg, self.overlap, self.window, self.dc, self.crop_freq,
                                          normalization_window_size=self.norm_size if self.correlation_whiten else None)
        elif mode in ('uncentered_cosine', 'cosine'):
            scores, F = uncentered_cosine_score(rx, self.fs, self.nperseg, self.overlap, self.window, self.dc, self.crop_freq)
        elif mode in ('demon_spectrum', 'demon'):
            scores, F = demon_spectrum(rx, self.fs, self.dc, self.crop_freq)
        elif mode in ('spectral_entropy', 'entropy', 'local_entropy'):
            scores, F = spectral_entropy(rx, self.fs, self.nperseg, self.overlap, self.window, self.dc, self.crop_freq,
                                         neighborhood=self.norm_size)
        elif mode == 'temporal_entropy':
            scores, F = temporal_entropy(rx, self.fs, self.nperseg, self.overlap, self.window, self.dc, self.crop_freq)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        scores = (scores - np.min(scores)) / np.std(scores) if np.std(scores) > 0 else scores

        return scores, F


## ====================
## FEATURE EXTRACTORS
## ====================

def correlation_score(signal, fs, nperseg, percent_overlap, window, remove_dc=None, crop_freq=None,
                      normalization_window_size=None):
    """
    Frame-to-frame cross-power of each bin, with the absolute level divided out.

    The raw cross-power decomposes as E[S_odd * S_even] = mu**2 + Cov(S_odd, S_even). The
    mu**2 level term spans ~54 dB across the band while the informative covariance term
    spans ~16 dB, so without dividing by mu**2 the feature is effectively a squared
    spectrum: it peaks on low-frequency flow noise instead of the tonal. Dividing it out
    leaves 1.0 for a stationary bin and >1 for one whose power is temporally correlated.
    """
    # Linear power: Pearson on (nearly constant) CW tone power demeans the tone away.
    F, T, Sxx, _ = calc_spectrogram(signal, fs, nperseg, percent_overlap, window=window,
                                    remove_dc=remove_dc, crop_freq=crop_freq, logscale=False)
    Sxx_odd = Sxx[:, 1::2]
    Sxx_even = Sxx[:, ::2]
    n = min(Sxx_odd.shape[1], Sxx_even.shape[1])
    cross_power = np.mean(Sxx_odd[:, :n] * Sxx_even[:, :n], axis=1)

    mean_power = np.mean(Sxx, axis=1)
    corr_scores = cross_power / (mean_power ** 2 + 1e-30)

    if normalization_window_size is not None:
        corr_scores = rw_normalization(corr_scores, window_size=normalization_window_size)

    return corr_scores, F


def uncentered_cosine_score(signal, fs, nperseg, percent_overlap, window, remove_dc=None, crop_freq=None):
    """
    Uncentered cosine similarity of consecutive linear-power frames per bin.

    mean(S_t S_{t+1}) / sqrt(mean(S_t^2) mean(S_{t+1}^2)) is ~0.5 for exponential
    spectrogram noise and → 1 for a bin whose power is stable. Unlike correlation_score,
    a constant-power tone is not cancelled.
    """
    F, T, Sxx, _ = calc_spectrogram(signal, fs, nperseg, percent_overlap, window=window,
                                    remove_dc=remove_dc, crop_freq=crop_freq, logscale=False)
    if Sxx.shape[1] < 2:
        return np.zeros(Sxx.shape[0]), F
    a, b = Sxx[:, :-1], Sxx[:, 1:]
    num = np.mean(a * b, axis=1)
    den = np.sqrt(np.mean(a ** 2, axis=1) * np.mean(b ** 2, axis=1) + 1e-30)
    return num / den, F


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


def temporal_entropy(signal, fs, nperseg, percent_overlap, window, remove_dc=None, crop_freq=None):
    """Shannon entropy of each bin's linear power across time (stability, not [3])."""
    _F, _T, _Sxx, _phasogram = calc_spectrogram(
        signal, fs=fs, nperseg=nperseg, percent_overlap=percent_overlap, window=window,
        remove_dc=remove_dc, crop_freq=crop_freq, logscale=False,
    )
    psd = np.maximum(_Sxx, 0.0)
    psd_sum = np.sum(psd, axis=1, keepdims=True) + 1e-12
    p = psd / psd_sum
    entropy = -(p * np.log(p + 1e-12)).sum(axis=1)
    return entropy, _F


def spectral_entropy(signal, fs, nperseg, percent_overlap, window, remove_dc=None, crop_freq=None,
                     neighborhood=5):
    """
    Local spectral entropy (Alexandri & Diamant 2024): Shannon entropy of the
    time-averaged linear spectrum in a sliding frequency neighborhood.

    A tonal concentrates mass in one bin of the window, so raw entropy is low.
    The returned score is H_max - H so find_peaks / thresholding treat a tone as a peak.
    """
    _F, _T, _Sxx, _phasogram = calc_spectrogram(
        signal, fs=fs, nperseg=nperseg, percent_overlap=percent_overlap, window=window,
        remove_dc=remove_dc, crop_freq=crop_freq, logscale=False,
    )
    Pxx = np.maximum(np.mean(_Sxx, axis=1), 0.0)
    window_size = int(neighborhood)
    if window_size < 1:
        window_size = 1
    if window_size % 2 == 0:
        window_size += 1
    half = window_size // 2
    padded = np.pad(Pxx, half, mode='edge')
    H = np.empty_like(Pxx, dtype=float)
    log_n = np.log(window_size)
    for k in range(len(Pxx)):
        w = padded[k:k + window_size]
        s = w.sum()
        if s <= 0:
            H[k] = log_n
            continue
        p = w / s
        H[k] = -(p * np.log(p + 1e-12)).sum()
    return log_n - H, _F


def spectral_entropy_narrowband(signal, fs, nperseg, percent_overlap, window,
                                target_freq, bandwidth,
                                remove_dc=None, crop_freq=None, agg='max', neighborhood=5):
    values, freqs = spectral_entropy(signal, fs, nperseg, percent_overlap, window,
                                     remove_dc=remove_dc, crop_freq=crop_freq,
                                     neighborhood=neighborhood)

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
    x_quantized = np.asarray(x_quantized)
    transitions = np.zeros((n_levels, n_levels), dtype=int)
    if x_quantized.size < 2:
        return transitions
    np.add.at(transitions, (x_quantized[:-1], x_quantized[1:]), 1)
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


def _row_normalize_transitions(transition_matrix):
    """Row-stochastic W as in Eq. (4): W_ij = count(i→j) / count(i). Empty rows stay 0."""
    W = np.asarray(transition_matrix, dtype=float)
    row_sum = W.sum(axis=1, keepdims=True)
    return np.divide(W, row_sum, out=np.zeros_like(W), where=row_sum > 0)


def _circular_distance(a, b, q):
    d = np.abs(a - b)
    return np.minimum(d, q - d)


@lru_cache(maxsize=32)
def _euclidean_edge_cost(q):
    """Ground cost ||(i,j)−(k,l)||_2 on the directed-edge grid."""
    ii, jj = np.divmod(np.arange(q * q), q)
    di = ii[:, None].astype(float) - ii[None, :]
    dj = jj[:, None].astype(float) - jj[None, :]
    return np.ascontiguousarray(np.hypot(di, dj))


@lru_cache(maxsize=32)
def _circular_edge_cost(q):
    """Ground cost hypot(d_Q(i,k), d_Q(j,l)) on the directed-edge grid."""
    ii, jj = np.divmod(np.arange(q * q), q)
    di = _circular_distance(ii[:, None], ii[None, :], q).astype(float)
    dj = _circular_distance(jj[:, None], jj[None, :], q).astype(float)
    return np.ascontiguousarray(np.hypot(di, dj))


def _w1_network_simplex(mu, nu, cost):
    """Exact discrete W1 via POT's network simplex (ot.emd2)."""
    mu = np.asarray(mu, dtype=np.float64).ravel()
    nu = np.asarray(nu, dtype=np.float64).ravel()
    cost = np.asarray(cost, dtype=np.float64)
    # POT requires the two masses to match to machine precision.
    mu = mu / mu.sum()
    nu = nu / nu.sum()
    return float(ot.emd2(mu, nu, cost))


def matrix_wasserstein(mu, nu, circular=False):
    """
    W1 between two Q×Q matrices treated as distributions on the directed-edge grid.

    Both matrices are flattened and renormalized to sum to 1 (joint transition mass).
    The transport problem is solved with POT's network simplex for every Q.

    circular=False (default): Euclidean ground metric ||(i,j)−(k,l)||_2.
    circular=True: d((i,j),(k,l)) = hypot(d_Q(i,k), d_Q(j,l)),
    d_Q(a,b) = min(|a-b|, Q-|a-b|). Exact circular cost at every Q (no embedding).
    """
    mu = np.asarray(mu, dtype=float).ravel()
    nu = np.asarray(nu, dtype=float).ravel()
    n = mu.size
    q = int(np.sqrt(n))
    if q * q != n:
        raise ValueError("Wasserstein support must be a square Q×Q matrix")
    if mu.sum() <= 0 or nu.sum() <= 0:
        return 0.0
    cost = _circular_edge_cost(q) if circular else _euclidean_edge_cost(q)
    return _w1_network_simplex(mu, nu, cost)


def get_K(transition_matrix, mode='wasserstein', uniform_M=None, wasserstein_circular=False):
    if mode == 'wasserstein':
        if uniform_M is not None:
            uniform_dist = uniform_M
        else:
            uniform_dist = np.ones(transition_matrix.shape) / transition_matrix.size
        tm = np.asarray(transition_matrix, dtype=float)
        tm = tm / (tm.sum() + 1e-12)
        K = matrix_wasserstein(tm, uniform_dist, circular=wasserstein_circular)
    elif mode == 'edge_count':
        edge_count = np.count_nonzero(transition_matrix)
        K = edge_count / transition_matrix.size
    elif mode == 'laplacian':
        W = _row_normalize_transitions(transition_matrix)
        D = np.diag(W.sum(axis=1))
        L = D - W
        eigenvalues = LA.eigvals(L)
        eigenvalues = np.real(eigenvalues)
        eigenvalues = np.sort(eigenvalues)
        eigenvalues = eigenvalues[eigenvalues > 1e-6]
        K = eigenvalues[0] if eigenvalues.size > 0 else 0
    elif mode == 'all':
        K1 = get_K(transition_matrix, mode='wasserstein', wasserstein_circular=wasserstein_circular)
        K2 = get_K(transition_matrix, mode='edge_count')
        K3 = get_K(transition_matrix, mode='laplacian')
        K = (K1, K2, K3)
    else:
        raise ValueError(f"Unknown K calculation mode: {mode}")
    return K


def _emd_workers(n_tasks):
    n_cpu = os.cpu_count() or 1
    return max(1, min(n_tasks, n_cpu))


def get_all_Ks(phase_matrix, frequencies, n_levels, mode='wasserstein', phase_mode='wrapped',
               wasserstein_circular=False):
    """
    Builds one S2G per frequency from wrapped phase in [0, 2π).

    phase_mode='wrapped' (also accepted as 'raw') uses the wrapped STFT phase.
    phase_mode='difference' uses the wrapped frame-to-frame increment, with no unwrap.
    wasserstein_circular selects the circular edge metric in matrix_wasserstein
    (exact network-simplex W1 at every Q).
    """
    if phase_mode in ('raw', 'wrapped'):
        phase_mode = 'wrapped'
    elif phase_mode != 'difference':
        raise ValueError(f"Unknown phase mode: {phase_mode}")

    if mode == 'wasserstein':
        uniform_M = np.ones((n_levels, n_levels)) / (n_levels**2)
        # Build the Q²×Q² cost once before workers share it (ot.emd2 releases the GIL).
        if wasserstein_circular:
            _circular_edge_cost(int(n_levels))
        else:
            _euclidean_edge_cost(int(n_levels))
    else:
        uniform_M = None

    def _score_bin(f_idx):
        x = wrap_phase(phase_matrix[f_idx, :])
        if phase_mode == 'difference':
            x = wrap_phase(np.diff(x))
        transition_matrix = get_s2g(x, n_levels=n_levels)
        return get_K(transition_matrix, mode=mode, uniform_M=uniform_M,
                     wasserstein_circular=wasserstein_circular)

    n_freq = len(frequencies)
    workers = _emd_workers(n_freq) if mode == 'wasserstein' else 1
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            Ks = list(ex.map(_score_bin, range(n_freq)))
    else:
        Ks = [_score_bin(f_idx) for f_idx in range(n_freq)]
    Ks = np.asarray(Ks, dtype=float)

    if mode == "edge_count":
        Ks = 1.0 - Ks
    elif mode == "laplacian":
        # After row-normalization, a tonal (cycle or self-loop) has a smaller Fiedler-like
        # value than a fully connected noise graph, so invert so that find_peaks sees a peak.
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
    """Unit-variance 1/f noise, generated by spectral shaping of white noise."""
    spectrum = np.fft.rfft(np.random.randn(N))
    freqs = np.fft.rfftfreq(N)
    scale = np.zeros_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    pink = np.fft.irfft(spectrum * scale, n=N)
    return pink / np.std(pink)


def white_noise(N):
    return np.random.randn(N)


def _slow_random_process(n, fs, tau, rng, rate=100.0):
    """
    Zero-mean unit-variance random process with correlation time tau seconds.
    Built at a low rate and interpolated up, since tau >> 1/fs.
    """
    m = max(4, int(np.ceil(n / fs * rate)) + 2)
    a = np.exp(-1.0 / (tau * rate))
    w = rng.standard_normal(m)
    x = np.empty(m)
    x[0] = w[0]
    for i in range(1, m):
        x[i] = a * x[i - 1] + np.sqrt(1.0 - a * a) * w[i]
    x -= x.mean()
    if x.std() > 0:
        x /= x.std()
    return np.interp(np.arange(n) / fs, np.arange(m) / rate, x)


def simulate_raw_signal(f0, fs, duration, freq_wander_hz=0.0, am_depth=0.0,
                        wander_tau=1.0, am_tau=0.3, seed=None,
                        blade_rate_hz=0.0, blade_depth=0.0):
    """
    Narrow-band tonal. The defaults give the original constant-frequency,
    constant-amplitude tone.

    A perfectly monochromatic tone is a degenerate case for phase-based detectors: its
    STFT phase advances by exactly (f0 * hop / fs mod 1) turns every frame, so the S2G
    walk is deterministic. Real machinery tonals wander and fade instead. Measured on the
    dpv*_1m recordings, the tonal has a frequency wander of 2-5 Hz std and an envelope
    std/mean of 0.4-0.9; pass freq_wander_hz and am_depth to reproduce that.

    blade_rate_hz / blade_depth add a sinusoidal propeller-rate AM so DEMON has a
    modulation line. Score DEMON at blade_rate_hz, not at f0. Both AM sources can
    be used together. Mean power is held at 0.125 so snr_db keeps its meaning.
    """
    n = int(fs * duration)
    t = np.arange(n) / fs
    rng = np.random.default_rng(seed)

    if freq_wander_hz > 0:
        f_inst = f0 + freq_wander_hz * _slow_random_process(n, fs, wander_tau, rng)
        phase = 2 * np.pi * np.cumsum(f_inst) / fs
    else:
        phase = 2 * np.pi * f0 * t

    envelope = np.ones(n, dtype=float)
    if am_depth > 0:
        envelope *= np.clip(1.0 + am_depth * _slow_random_process(n, fs, am_tau, rng), 0.0, None)
    if blade_rate_hz > 0 and blade_depth > 0:
        envelope *= np.clip(1.0 + blade_depth * np.cos(2.0 * np.pi * blade_rate_hz * t), 0.0, None)

    sig = 0.5 * envelope * np.sin(phase)

    power = np.mean(sig ** 2)
    if power > 0:
        sig *= np.sqrt(0.125 / power)
    return sig


def add_noise_to_signal(signal, snr_db, fs, signal_bw, noise_type='white', f0=None):
    """
    Adds broad-band noise to a narrow-band signal based on an in-band SNR.

    signal_bw is the bandwidth the SNR is referenced to; use analysis_bandwidth()
    so that snr_db is the SNR seen by a single detector bin. Coloured noise also
    needs f0, because its power is not spread evenly over [0, fs/2].
    """
    if noise_type == 'white':
        noise = white_noise(len(signal))
    elif noise_type == 'pink':
        noise = pink_noise(len(signal))
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")

    noise = noise - np.mean(noise)

    if noise_type == 'white':
        in_band_fraction = signal_bw / (fs / 2)
    else:
        if f0 is None:
            raise ValueError(f"f0 is required to set an in-band SNR for '{noise_type}' noise")
        freqs = np.fft.rfftfreq(len(noise), 1 / fs)
        psd = np.abs(np.fft.rfft(noise)) ** 2
        in_band = np.abs(freqs - f0) <= signal_bw / 2
        in_band_fraction = psd[in_band].sum() / psd.sum()

    signal_power = np.mean(signal ** 2)
    desired_noise_power_in_band = signal_power / (10 ** (snr_db / 10))
    total_desired_noise_power = desired_noise_power_in_band / in_band_fraction
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


def get_window_values(window, nperseg):
    if window == 'flattop':
        return signal.windows.flattop(nperseg)
    if hasattr(np, window):
        return getattr(np, window)(nperseg)
    if hasattr(signal.windows, window):
        return signal.get_window(window, nperseg)
    raise ValueError(f"Window '{window}' not found in numpy or scipy.signal.windows")


def analysis_bandwidth(fs, nperseg, window='hamming'):
    """
    Equivalent noise bandwidth (Hz) of a single bin of the spectrogram produced by
    calc_spectrogram. This is the bandwidth an in-band SNR should be referenced to,
    since it is the band over which each detector bin integrates noise.
    """
    window_vals = get_window_values(window, nperseg)
    return fs * np.sum(window_vals ** 2) / np.sum(window_vals) ** 2


def wrap_phase(phi):
    """Wrap phase to [0, 2π)."""
    return np.mod(np.asarray(phi, dtype=float), 2.0 * np.pi)


def nearest_frequency_index(F, f0):
    return int(np.argmin(np.abs(np.asarray(F) - f0)))


def evaluation_bins(F, f0, tolerance):
    """
    Nearest bin to f0, and a mask of bins outside ±tolerance Hz (leakage guard for FA).
    """
    F = np.asarray(F)
    fix = nearest_frequency_index(F, f0)
    fa_mask = np.abs(F - f0) > tolerance
    return fix, fa_mask


def calc_spectrogram(x, fs, nperseg, percent_overlap, nfft=None, window='hamming', remove_dc=None, crop_freq=None, logscale=True, return_complex=False):
    noverlap = int(nperseg * percent_overlap)
    step = nperseg - noverlap

    window_vals = get_window_values(window, nperseg)

    window_power = np.sum(window_vals**2)

    if nfft is None:
        nfft = nperseg

    n_segments = (len(x) - noverlap) // step
    f = np.fft.rfftfreq(nfft, 1/fs)
    t = np.arange(n_segments) * step / fs
    Sxx = np.zeros((n_segments, len(f)))
    Phase = np.zeros((n_segments, len(f)))
    X = np.zeros((n_segments, len(f)), dtype=np.complex128) if return_complex else None

    for i in range(n_segments):
        start = i * step
        segment = x[start:start+nperseg]
        if len(segment) < nperseg:
            break
        segment = segment * window_vals
        spectrum = np.fft.rfft(segment, n=nfft)
        Sxx[i, :] = (np.abs(spectrum)**2) / (fs * window_power)
        Phase[i, :] = wrap_phase(np.angle(spectrum))
        if return_complex:
            X[i, :] = spectrum

    if remove_dc is not None:
        dc_band = f <= remove_dc
        f = f[~dc_band]
        Sxx = Sxx[:, ~dc_band]
        Phase = Phase[:, ~dc_band]
        if return_complex:
            X = X[:, ~dc_band]

    if crop_freq is not None:
        if crop_freq > fs / 2:
            crop_freq = fs // 2 - 1  # Nyquist limit
        crop_band = f <= crop_freq
        f = f[crop_band]
        Sxx = Sxx[:, crop_band]
        Phase = Phase[:, crop_band]
        if return_complex:
            X = X[:, crop_band]

    if logscale:
        Sxx = 10 * np.log10(Sxx + 1e-100)

    if return_complex:
        return f, t, Sxx.T, Phase.T, X.T
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
