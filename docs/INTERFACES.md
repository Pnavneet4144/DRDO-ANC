# Shared Interfaces (v0 — lead may revise; change only via PR)

Everyone builds against these so no one depends on an undocumented assumption.

## Audio
- Sample rate **16 kHz**, mono, float32 in [-1, 1]; `.wav` on disk
- Training clips: 4 s segments
- Real-time hop: 20 ms (320 samples) — P4 may tune after latency tests

## STFT (P1 owns)
- n_fft = 512, hop = 256, window = 512 (Hann)
- STFT lives inside the model wrapper; others pass waveforms only

## Enhancer signature (Python)
```python
def enhance(noisy: np.ndarray, reference: np.ndarray | None, sr: int = 16000) -> np.ndarray: ...
```
`reference` = reference-mic signal for LMS/NLMS (None for single-mic methods). Output length == input length.
Every method (spectral, Wiener, LMS, NLMS, AI, hybrid) uses this.

## Dataset manifest (P3 owns)
CSV columns: `id, clean_path, noise_path, noise_type, snr_db, split, mixed_path`
Splits: `train / val / test / test_unseen_noise`. After tag `dataset-v1` the split is frozen.

## Experiment config + results
Config YAML in `experiments/configs/`: `name, method, seed, dataset_version, params`.
Result JSON in `experiments/results/<name>.json`: `{name, method, snr, si_snr, stoi, pesq, latency_ms, commit, date}`

## Backend (P5 owns)
`POST /api/runs` accepts the result JSON; `GET /api/runs` lists/sorts them.
