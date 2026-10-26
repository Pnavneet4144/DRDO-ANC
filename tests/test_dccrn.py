"""Tests for the Deep Complex Convolution Recurrent Network (DCCRN) model."""

from __future__ import annotations

import logging
import statistics
import time

import pytest
import torch

from ai_engine.models.dccrn import DCCRN
from ai_engine.models.stft import STFT
from ai_engine.training.losses import si_snr_loss
from common.constants import N_FREQ, SEED

logger = logging.getLogger(__name__)


@pytest.fixture()
def dccrn_model() -> DCCRN:
    """Create a DCCRN model instance in eval mode with deterministic seed."""
    torch.manual_seed(SEED)
    model = DCCRN()
    model.eval()
    return model


class TestDCCRN:
    """Test suite for DCCRN phase-aware model."""

    def test_output_shape_and_complex_dtype(self, dccrn_model: DCCRN) -> None:
        """Verify output shape matches input and dtype is complex64."""
        torch.manual_seed(SEED)
        for batch_size, time_frames in [(2, 251), (1, 1), (1, 10), (1, 400)]:
            spec = torch.randn(batch_size, N_FREQ, time_frames, dtype=torch.complex64)
            out = dccrn_model(spec)
            assert out.shape == (batch_size, N_FREQ, time_frames)
            assert out.dtype == torch.complex64

    def test_causality_in_eval_mode(self, dccrn_model: DCCRN) -> None:
        """Verify output at frame t is strictly independent of frames > t in eval mode."""
        torch.manual_seed(SEED)
        time_len = 251
        t_cut = 100
        spec1 = torch.randn(1, N_FREQ, time_len, dtype=torch.complex64)
        spec2 = spec1.clone()
        spec2[:, :, t_cut:] = torch.randn_like(spec2[:, :, t_cut:])

        with torch.no_grad():
            enh1 = dccrn_model(spec1)
            enh2 = dccrn_model(spec2)

        earlier_diff = (enh1[:, :, :t_cut] - enh2[:, :, :t_cut]).abs().max().item()
        assert earlier_diff <= 1e-6, f"Future frames affected earlier outputs (diff={earlier_diff})"

    def test_complex_mask_magnitude_bounds(self, dccrn_model: DCCRN) -> None:
        """Verify complex ratio mask magnitude is bounded in [0, 1]."""
        torch.manual_seed(SEED)
        spec = torch.randn(2, N_FREQ, 251, dtype=torch.complex64)
        mask = dccrn_model.compute_mask(spec)
        assert mask.shape == (2, N_FREQ, 251)
        assert mask.dtype == torch.complex64

        mask_mag = torch.abs(mask)
        assert mask_mag.min().item() >= 0.0, f"Mask magnitude {mask_mag.min().item()} < 0.0"
        assert mask_mag.max().item() <= 1.0 + 1e-6, f"Mask magnitude {mask_mag.max().item()} > 1.0"

    def test_finite_outputs_on_extreme_inputs(self, dccrn_model: DCCRN) -> None:
        """Verify model produces finite outputs on silence and peak-level loud inputs."""
        silence = torch.zeros(1, N_FREQ, 251, dtype=torch.complex64)
        with torch.no_grad():
            out_silence = dccrn_model(silence)
        assert torch.isfinite(out_silence).all(), "Silence input produced non-finite output"

        loud = torch.ones(1, N_FREQ, 251, dtype=torch.complex64)
        with torch.no_grad():
            out_loud = dccrn_model(loud)
        assert torch.isfinite(out_loud).all(), "Loud input produced non-finite output"

    def test_backward_gradient_flow(self) -> None:
        """Verify autograd backward pass produces non-None, finite gradients."""
        torch.manual_seed(SEED)
        model = DCCRN()
        stft = STFT()
        clean = torch.randn(2, 16000)
        noisy = clean + 0.5 * torch.randn(2, 16000)

        spec = stft(noisy)
        enh_spec = model(spec)
        recon = stft.inverse(enh_spec, length=16000)
        loss = si_snr_loss(recon, clean)
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"Parameter {name} has None gradient"
            assert torch.isfinite(param.grad).all(), f"Parameter {name} has non-finite gradient"

    def test_param_count_logging(self, dccrn_model: DCCRN) -> None:
        """Log parameter count and verify scale comparable to CRN baseline (~1.72M vs 1.82M)."""
        count = dccrn_model.num_params()
        logger.info("DCCRN parameter count: %d (%.2fM)", count, count / 1e6)
        assert 1_000_000 <= count <= 2_500_000, f"DCCRN parameter count {count} out of expected range"

    def test_cpu_latency_single_spectral_frame(self, dccrn_model: DCCRN) -> None:
        """Measure median CPU latency over 100 runs for a single spectral frame [1, 257, 1].

        Note: Recurrent (LSTM) hidden state is NOT retained across iterations (stateless call).

        Standalone latency measurement command:
            python -c "import time, statistics, torch; from ai_engine.models.dccrn import DCCRN; m = DCCRN().eval(); x = torch.randn(1, 257, 1, dtype=torch.complex64); \\
with torch.inference_mode(): \\
    for _ in range(20): m(x); \\
    l = []; \\
    for _ in range(100): \\
        t0 = time.perf_counter(); \\
        m(x); \\
        l.append((time.perf_counter() - t0) * 1000); \\
    print(f'Median ms: {statistics.median(l):.3f}')"
        """
        spec_frame = torch.randn(1, N_FREQ, 1, dtype=torch.complex64)
        with torch.inference_mode():
            # Warmup
            for _ in range(20):
                _ = dccrn_model(spec_frame)

            latencies_ms: list[float] = []
            for _ in range(100):
                t0 = time.perf_counter()
                _ = dccrn_model(spec_frame)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

        median_ms = statistics.median(latencies_ms)
        logger.info(
            "DCCRN single spectral frame [1, 257, 1] CPU latency (median of 100 runs): %.3f ms",
            median_ms,
        )
        assert median_ms > 0.0

    def test_cpu_latency_256_sample_waveform_frame(self, dccrn_model: DCCRN) -> None:
        """Measure median CPU latency over 100 runs for an isolated 256-sample waveform frame.

        Note: This tests offline STFT -> DCCRN -> iSTFT on an isolated 256-sample buffer.
        Because reflection padding produces 2 STFT frames and boundary reconstruction lacks
        streaming overlap-add state, this is NOT equivalent to real-time stateful streaming.

        Standalone latency measurement command:
            python -c "import time, statistics, torch; from ai_engine.models.dccrn import DCCRN; from ai_engine.models.stft import STFT; m = DCCRN().eval(); stft = STFT(); x = torch.randn(1, 256); \\
with torch.inference_mode(): \\
    for _ in range(20): m.enhance_waveform(x, stft); \\
    l = []; \\
    for _ in range(100): \\
        t0 = time.perf_counter(); \\
        m.enhance_waveform(x, stft); \\
        l.append((time.perf_counter() - t0) * 1000); \\
    print(f'Median ms: {statistics.median(l):.3f}')"
        """
        stft = STFT()
        wave_frame = torch.randn(1, 256)
        with torch.inference_mode():
            # Warmup
            for _ in range(20):
                _ = dccrn_model.enhance_waveform(wave_frame, stft=stft)

            latencies_ms: list[float] = []
            for _ in range(100):
                t0 = time.perf_counter()
                _ = dccrn_model.enhance_waveform(wave_frame, stft=stft)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

        median_ms = statistics.median(latencies_ms)
        logger.info(
            "DCCRN 256-sample waveform frame CPU latency (median of 100 runs): %.3f ms",
            median_ms,
        )
        assert median_ms > 0.0

    def test_cpu_latency_1s_waveform_clip(self, dccrn_model: DCCRN) -> None:
        """Measure median CPU latency over 50 runs for a 1-second waveform clip (16,000 samples).

        Standalone latency measurement command:
            python -c "import time, statistics, torch; from ai_engine.models.dccrn import DCCRN; from ai_engine.models.stft import STFT; m = DCCRN().eval(); stft = STFT(); x = torch.randn(1, 16000); \\
with torch.inference_mode(): \\
    for _ in range(10): m.enhance_waveform(x, stft); \\
    l = []; \\
    for _ in range(50): \\
        t0 = time.perf_counter(); \\
        m.enhance_waveform(x, stft); \\
        l.append((time.perf_counter() - t0) * 1000); \\
    print(f'Median ms: {statistics.median(l):.3f}')"
        """
        stft = STFT()
        wave_1s = torch.randn(1, 16000)
        with torch.inference_mode():
            # Warmup
            for _ in range(10):
                _ = dccrn_model.enhance_waveform(wave_1s, stft=stft)

            latencies_ms: list[float] = []
            for _ in range(50):
                t0 = time.perf_counter()
                _ = dccrn_model.enhance_waveform(wave_1s, stft=stft)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)

        median_ms = statistics.median(latencies_ms)
        logger.info(
            "DCCRN 1-second waveform clip CPU latency (median of 50 runs): %.3f ms",
            median_ms,
        )
        assert median_ms > 0.0



    def test_end_to_end_waveform_with_stft(self, dccrn_model: DCCRN) -> None:
        """Verify time-domain waveform processing via STFT -> DCCRN -> iSTFT."""
        torch.manual_seed(SEED)
        stft = STFT()
        wave = torch.randn(2, 16000)
        enhanced_wave = dccrn_model.enhance_waveform(wave, stft=stft)
        assert enhanced_wave.shape == wave.shape
        assert torch.isfinite(enhanced_wave).all()
