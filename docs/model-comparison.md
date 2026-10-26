
# AI Model Comparison

Preliminary comparison of deep learning baseline architectures for active noise cancellation and speech enhancement.

## Architecture Comparison

| Model | Parameters | Measured CPU Latency | Input | Output | Causal |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **CRN** | 1,820,369 (~1.82M) | 1.45 ms / spectral frame `[1, 257, 1]`; 2.27 ms / 256-sample waveform frame; 10.3 ms / 1-second clip | Complex spectrum `[B, 257, F]` (magnitude used) | Complex spectrum `[B, 257, F]` (magnitude mask applied) | Yes, at inference (`eval()` mode) |
| **DCCRN** | 1,719,714 (~1.72M) | 3.76 ms / spectral frame `[1, 257, 1]`; 5.70 ms / 256-sample waveform frame; 41.9 ms / 1-second clip (63 frames) | Complex spectrum `[B, 257, F]` (real & imag used) | Complex spectrum `[B, 257, F]` (polar complex ratio mask applied) | Yes, at inference (`eval()` mode) |

### Notes

- **Measurement environment:** CPU measurements from development host (Linux x86_64, PyTorch CPU).
- **Benchmark protocol:** Model set to `eval()` mode; wrapped in `torch.inference_mode()`. Timing captures `time.perf_counter()` immediately before and after each individual model invocation, recording elapsed time per iteration across 100 runs (50 runs for 1-second clip) following initial warmup iterations.
- **Reproducible benchmark commands:**
  - *Single spectral frame `[1, 257, 1]` (median of 100 iterations after 20 warmup iterations):*
    ```bash
    python -c "import time, statistics, torch; from ai_engine.models.dccrn import DCCRN; m = DCCRN().eval(); x = torch.randn(1, 257, 1, dtype=torch.complex64); \
with torch.inference_mode(): \
    for _ in range(20): m(x) \
    l = [] \
    for _ in range(100): \
        t0 = time.perf_counter(); m(x); l.append((time.perf_counter() - t0) * 1000) \
    print(f'Median ms: {statistics.median(l):.3f}')"
    ```
  - *Isolated 256-sample waveform frame (`STFT → DCCRN → iSTFT`, median of 100 iterations after 20 warmup iterations):*
    ```bash
    python -c "import time, statistics, torch; from ai_engine.models.dccrn import DCCRN; from ai_engine.models.stft import STFT; m = DCCRN().eval(); stft = STFT(); x = torch.randn(1, 256); \
with torch.inference_mode(): \
    for _ in range(20): m.enhance_waveform(x, stft) \
    l = [] \
    for _ in range(100): \
        t0 = time.perf_counter(); m.enhance_waveform(x, stft); l.append((time.perf_counter() - t0) * 1000) \
    print(f'Median ms: {statistics.median(l):.3f}')"
    ```
  - *1-second waveform clip (16,000 samples / 63 frames, median of 50 iterations after 10 warmup iterations):*
    ```bash
    python -c "import time, statistics, torch; from ai_engine.models.dccrn import DCCRN; from ai_engine.models.stft import STFT; m = DCCRN().eval(); stft = STFT(); x = torch.randn(1, 16000); \
with torch.inference_mode(): \
    for _ in range(10): m.enhance_waveform(x, stft) \
    l = [] \
    for _ in range(50): \
        t0 = time.perf_counter(); m.enhance_waveform(x, stft); l.append((time.perf_counter() - t0) * 1000) \
    print(f'Median ms: {statistics.median(l):.3f}')"
    ```
- **Stateless benchmark & streaming caveats:**
  - *Recurrent state:* In these benchmark calls, recurrent (LSTM) hidden states `(h, c)` are re-initialized to zero on each call and are NOT retained across frames.
  - *Frame processing:* The 256-sample waveform frame benchmark tests the offline `enhance_waveform` pipeline on an isolated 256-sample tensor (which undergoes reflection padding in `STFT` yielding 2 frames and boundary taper in `iSTFT`). This is NOT equivalent to real-time streaming overlap-add.
  - *Real-time compliance:* While 5.70 ms per 256-sample frame on a desktop CPU is within the 16.0 ms audio frame duration (processing ratio ~0.36), **real-time streaming compliance cannot be claimed from a stateless benchmark**. True streaming latency with carried recurrent state and overlap-add ring buffers must be measured in Phase 4 Task B / streaming harness on the target embedded hardware.
- **CRN:** Convolutional Recurrent Network with a causal 2D convolutional encoder, 2-layer unidirectional LSTM bottleneck, and causal 2D transposed-convolutional decoder with skip connections. Employs a real-valued sigmoid magnitude mask applied to the noisy complex spectrum while preserving the noisy phase.
- **CRN causality:** Causal at inference (eval mode); BatchNorm statistics are pooled across time during training.
- **DCCRN:** Deep Complex Convolution Recurrent Network with causal complex 2D convolutional encoder, 2-layer unidirectional ComplexLSTM bottleneck, and causal complex 2D transposed-convolutional decoder with skip connections. Employs a polar complex ratio mask with bounded magnitude and phase shift applied to the noisy complex spectrum.
- **DCCRN causality:** Causal at inference (eval mode); ComplexBatchNorm statistics are pooled across time during training.

## How to Decide

Final model selection between CRN and DCCRN must not be based on architectural intuition or personal preference. It must be determined empirically under strictly standardized conditions:

1. **Frozen Dataset Evaluation:** Both models must be trained and evaluated on the exact same frozen dataset (`dataset-v1`), using identical speaker splits and noise conditions to ensure leakage-safe, fair comparisons.
2. **Objective Audio Metrics:** Model performance must be evaluated quantitatively using standard speech enhancement and active noise cancellation metrics:
   - **SI-SNR** (Scale-Invariant Signal-to-Noise Ratio improvement in dB)
   - **SNR** (Standard Signal-to-Noise Ratio improvement in dB)
   - **PESQ** (Perceptual Evaluation of Speech Quality, ITU-T P.862)
   - **STOI** (Short-Time Objective Intelligibility)
3. **Hardware & Latency Constraints:** Latency must be measured on the target deployment hardware (e.g., embedded / real-time streaming harness) with streaming state management enabled.
4. **Trade-off Analysis:** A model with marginal PESQ/STOI gains must be evaluated against its latency and compute footprint within the real-time frame budget (e.g., DCCRN complex convolution overhead vs. CRN real-valued processing). The final candidate must satisfy both audio enhancement targets and real-time execution bounds.

## AIEnhancer Integration Status

Both baseline architectures (CRN and DCCRN) are unified under the shared `AIEnhancer` wrapper (`ai_engine/models/enhancer.py`):

- **Protocol Conformance:** Implements both offline `Enhancer` and real-time `StreamingEnhancer` protocols defined in `common/interface.py`.
- **Offline Waveform Enhancement:** `enhance(noisy, reference=None, sr=16000)` enforces 16 kHz sample rate, validates 1-D float32 input in `[-1, 1]`, processes the waveform via `STFT -> Model -> iSTFT`, and returns a float32 NumPy array of identical shape with finiteness checks.
- **Spectral Enhancement:** `enhance_spec(noisy)` extracts complex spectrum `[257, F]` before inverse STFT.
- **Stateful Streaming:** `process_frame(frame, ref_frame=None)` accepts 256 samples (16.0 ms frame), maintains layer-wise causal convolution buffers and LSTM hidden/cell states across frames, and reconstructs time-domain audio with 1-frame algorithmic delay.
- **State Isolation:** `reset()` fully flushes analysis context, synthesis buffers, and internal recurrent/conv states for deterministic, leak-free streaming across calls.
- **Shared Transform Contract:** All STFT and iSTFT transformations strictly delegate to the shared `STFT` class (`ai_engine/models/stft.py`), eliminating redundant direct PyTorch transform calls from the enhancer module.

## Training Pipeline Workflow

The Phase 4 training pipeline is fully configurable via YAML and invoked via CLI:

```bash
python -m ai_engine.training.train --config experiments/configs/<config_name>.yaml
```

### Configured Baselines

- `experiments/configs/crn_baseline.yaml`: Causal magnitude mask CRN baseline (Adam optimizer, lr=0.001, SI-SNR loss).
- `experiments/configs/dccrn_baseline.yaml`: Causal complex ratio mask DCCRN baseline (Complex Conv/LSTM, lr=0.001, SI-SNR loss).

### Prerequisites & Environment Variables

- `DATA_ROOT`: Root directory containing audio files referenced in the manifest (e.g. `DATA_ROOT=/path/to/dataset`).
- `CKPT_DIR`: Target directory outside the repository for saving training checkpoints (e.g. `CKPT_DIR=/path/to/checkpoints`). Checkpoint files follow the naming convention `<config_name>_<git_hash>_best.pt` and `<config_name>_<git_hash>_latest.pt`.

### Evidence Required for Final Model Selection

Neither CRN nor DCCRN is declared superior at this stage. Final selection requires:
1. **Completion of Training on Frozen Dataset**: Both models must be trained using identical seeds (`seed=42`) on the final frozen `dataset-v1` split (not the prototype `dev-clean` dataset).
2. **Standard Evaluation Metrics**: Both checkpoints must be evaluated across test splits (seen and unseen noise) for SI-SNR (dB), standard SNR (dB), PESQ, and STOI.
3. **End-to-End Streaming Latency**: Real-time streaming latency must be benchmarked on target edge/embedded devices using `AIEnhancer.process_frame()`, accounting for per-frame processing time within the real-time frame budget.
