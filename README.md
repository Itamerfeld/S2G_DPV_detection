# S2G: Spectrogram-to-Graph Tonal Detection

Code and evaluation for **S2G**, a detector that turns the STFT phase of each frequency bin into a quantized transition graph and scores how unlike a random (uniform) graph that structure is. Stable tones produce concentrated phase-transition graphs; noise produces more uniform ones.

This repository contains the S2G implementation, synthetic ROC-style simulations, a phase-graph illustration, and a real DPV (diver propulsion vehicle) recording evaluation.

## Repository layout

```
utils.py                                  # S2G detector and supporting DSP
simulate_phase_graph.py                   # example graphs: tone / noisy tone / white noise
simulation_1_set_SNR_change_Q.py          # Pd / Pfa / ROC vs quantization level Q
simulation_2_set_Q_change_SNR.py          # Pd / Pfa / ROC vs SNR
simulation_3_set_Q_and_SNR_change_feature.py  # S2G vs correlation / entropy / DEMON
dpv_detection_for_paper.py                # field recording: scores vs range
data/croatia/                             # hydrophone WAVs + GPS track
results/                                  # precomputed simulation curves
```

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run scripts from any directory; paths are resolved relative to each file.

To skip opening a browser window (still writes PDF plots):

```bash
export S2G_HEADLESS=1
```

## S2G in brief

For a frequency bin, S2G:

1. Computes an STFT and takes the unwrapped-in-bin phase sequence.
2. Normalizes and quantizes phase into `Q` levels.
3. Builds a directed transition-count graph / matrix between levels.
4. Scores the matrix with either:
   - **Wasserstein** distance to a uniform transition distribution (higher → more tonal)
   - **Laplacian** algebraic connectivity (lower raw Fiedler-like value → more concentrated; the detector inverts and standardizes this for peak picking)

`S2GDetector` in `utils.py` exposes `get_feature_vector` (score vs frequency) and `get_score_1fb` (score at one frequency).

## Synthetic simulations

Default mode (`run_simulation = False`) **loads saved curves** from `results/` and writes PDFs under `results/plots/`. Set `run_simulation = True` at the top of a script to recompute (100 Monte Carlo trials; simulations 2 and 3 use 128 kHz / 60 s and can take hours).

```bash
export S2G_HEADLESS=1
python simulation_1_set_SNR_change_Q.py
python simulation_2_set_Q_change_SNR.py
python simulation_3_set_Q_and_SNR_change_feature.py
```

| Script | What it varies | Default setting |
| --- | --- | --- |
| `simulation_1_set_SNR_change_Q.py` | quantization levels `Q` | SNR = −10 dB, Wasserstein |
| `simulation_2_set_Q_change_SNR.py` | SNR | `Q = 3`, Wasserstein |
| `simulation_3_set_Q_and_SNR_change_feature.py` | detector / feature | SNR = −18 dB |

Phase-graph illustration (pure tone, SNR = 0 dB, white noise):

```bash
export S2G_HEADLESS=1
python simulate_phase_graph.py
```

## DPV field evaluation

`dpv_detection_for_paper.py` loads the Croatia 600 m DPV pass, overlays GPS range, and compares DEMON, correlation, spectral entropy, and S2G (Laplacian and Wasserstein) on:

- the original recording
- the recording plus added background noise from a 1000 m run
- background noise only

```bash
export S2G_HEADLESS=1
python dpv_detection_for_paper.py
```

This script is the slow one: it loads ~40 minutes of audio and scores 60 s windows every 10 s. Expect on the order of tens of minutes, depending on the machine.

PDFs are written to `results/plots/`.

## Data

Hydrophone WAV files are 16-bit, ~22 MB each (under GitHub’s 100 MB file limit), so they are stored in git without Git LFS.

- `data/croatia/2407_1_600m/` — full 600 m experiment (60 one-minute WAVs + GPS). The script skips the first 6 and last 15 files, matching the original analysis.
- `data/croatia/2507_1_1000m/` — the **five** background-noise WAVs the script actually reads (`file_list[:5]`). The unused remainder of that folder is about 1.8 GB and is omitted.

Total data in this repo is about **1.4 GB**.

## License

Copyright (c) 2026 Itamar Merfeld

This project is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).

You may use, modify, and share the code and data for **noncommercial** purposes (personal study, academic research, education, and use by noncommercial organizations). **For-profit / commercial use is not allowed.**

Required Notice: Copyright 2026 Itamar Merfeld

Please cite the associated paper if you use this work.
