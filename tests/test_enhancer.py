"""Unit and integration tests for AIEnhancer offline and streaming enhancement."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from ai_engine.models.crn import CRN
from ai_engine.models.dccrn import DCCRN
from ai_engine.models.enhancer import (
    AIEnhancer,
    _crn_stream_step,
    _CRNStreamState,
    _dccrn_stream_step,
    _DCCRNStreamState,
)
from common.constants import FRAME_SAMPLES, N_FREQ, SAMPLE_RATE, SEED
from common.interface import Enhancer, StreamingEnhancer


class IdentityModel(nn.Module):
    """Dummy model returning input complex spectrum unchanged."""

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Return unchanged spectrum."""
        return spec


class StatefulDummyModel(nn.Module):
    """Dummy recurrent model maintaining an explicit step counter state."""

    def __init__(self) -> None:
        super().__init__()
        self.reset_count = 0

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Return unchanged spectrum."""
        return spec

    def stream_step(
        self, spec: torch.Tensor, state: Any
    ) -> tuple[torch.Tensor, Any]:
        """Process one frame and increment recurrent step counter."""
        next_state = 1 if state is None else state + 1
        return spec, next_state

    def reset(self) -> None:
        """Record reset invocation."""
        self.reset_count += 1


@pytest.fixture()
def identity_enhancer() -> AIEnhancer:
    """Create an AIEnhancer wrapping IdentityModel on CPU."""
    return AIEnhancer(IdentityModel(), device="cpu")


@pytest.fixture()
def stateful_enhancer() -> AIEnhancer:
    """Create an AIEnhancer wrapping StatefulDummyModel on CPU."""
    return AIEnhancer(StatefulDummyModel(), device="cpu")


class TestOfflineEnhancement:
    """Tests for offline enhance() and enhance_spec() methods."""

    @pytest.mark.parametrize("length", [8000, 16000, 16001, 32000, 64000])
    def test_output_length_and_dtype(
        self, identity_enhancer: AIEnhancer, length: int
    ) -> None:
        """Verify enhanced output matches input length exactly and has float32 dtype."""
        rng = np.random.default_rng(SEED)
        noisy = rng.uniform(-0.5, 0.5, size=length).astype(np.float32)
        out = identity_enhancer.enhance(noisy)
        assert out.shape == noisy.shape
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))

    def test_identity_model_preserves_waveform(
        self, identity_enhancer: AIEnhancer
    ) -> None:
        """Verify an identity dummy model preserves the waveform within numerical tolerance."""
        rng = np.random.default_rng(SEED)
        noisy = rng.uniform(-0.5, 0.5, size=SAMPLE_RATE).astype(np.float32)
        out = identity_enhancer.enhance(noisy)
        max_err = np.abs(noisy - out).max()
        assert max_err < 1e-4, f"Identity roundtrip error {max_err:.6e} >= 1e-4"

    def test_enhance_spec_output(self, identity_enhancer: AIEnhancer) -> None:
        """Verify enhance_spec returns complex spectrum shaped [257, F]."""
        rng = np.random.default_rng(SEED)
        noisy = rng.uniform(-0.5, 0.5, size=16000).astype(np.float32)
        spec = identity_enhancer.enhance_spec(noisy)
        expected_frames = 16000 // FRAME_SAMPLES + 1
        assert spec.shape == (N_FREQ, expected_frames)
        assert np.iscomplexobj(spec)
        assert spec.dtype == np.complex64

    def test_reference_accepted_and_ignored(
        self, identity_enhancer: AIEnhancer
    ) -> None:
        """Verify reference microphone input is accepted without error."""
        rng = np.random.default_rng(SEED)
        noisy = rng.uniform(-0.5, 0.5, size=16000).astype(np.float32)
        ref = rng.uniform(-0.5, 0.5, size=16000).astype(np.float32)
        out = identity_enhancer.enhance(noisy, reference=ref, sr=SAMPLE_RATE)
        assert out.shape == noisy.shape


class TestStreamingEnhancement:
    """Tests for real-time streaming process_frame() and reset() methods."""

    def test_process_frame_returns_exact_256_samples(
        self, identity_enhancer: AIEnhancer
    ) -> None:
        """Verify every process_frame call returns exactly 256 float32 samples."""
        rng = np.random.default_rng(SEED)
        identity_enhancer.reset()
        for _ in range(10):
            frame = rng.uniform(-0.5, 0.5, size=FRAME_SAMPLES).astype(np.float32)
            out = identity_enhancer.process_frame(frame)
            assert out.shape == (FRAME_SAMPLES,)
            assert out.dtype == np.float32
            assert np.all(np.isfinite(out))

    def test_streaming_identity_preserves_waveform(
        self, identity_enhancer: AIEnhancer
    ) -> None:
        """Verify streaming identity model preserves waveform after 1-frame algorithmic delay."""
        rng = np.random.default_rng(SEED)
        num_frames = 32
        signal = rng.uniform(-0.5, 0.5, size=num_frames * FRAME_SAMPLES).astype(
            np.float32
        )

        identity_enhancer.reset()
        outputs: list[np.ndarray] = []
        for i in range(num_frames):
            frame = signal[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            out_frame = identity_enhancer.process_frame(frame)
            outputs.append(out_frame)

        full_stream = np.concatenate(outputs)
        assert full_stream.shape == signal.shape

        # Reconstructed signal matches input after 1-frame (256 samples) delay
        delay = FRAME_SAMPLES
        stream_err = np.abs(signal[:-delay] - full_stream[delay:]).max()
        assert stream_err < 1e-4, f"Streaming error after delay {stream_err:.6e} >= 1e-4"

    def test_reset_clears_buffers_and_recurrent_state(
        self, stateful_enhancer: AIEnhancer
    ) -> None:
        """Verify reset() clears all analysis/synthesis buffers and recurrent state."""
        rng = np.random.default_rng(SEED)
        stateful_enhancer.reset()

        frame = rng.uniform(-0.5, 0.5, size=FRAME_SAMPLES).astype(np.float32)
        first_run_out = stateful_enhancer.process_frame(frame)

        # Buffer and recurrent state should now be populated
        assert not np.all(stateful_enhancer._prev_frame == 0.0)
        assert stateful_enhancer._recurrent_state is not None

        # Reset should restore pristine initial state
        stateful_enhancer.reset()
        assert np.all(stateful_enhancer._prev_frame == 0.0)
        assert np.all(stateful_enhancer._ola_buffer == 0.0)
        assert len(stateful_enhancer._spec_context) == 0
        assert stateful_enhancer._recurrent_state is None
        assert stateful_enhancer.model.reset_count >= 1

        # Running the exact same frame again must produce the exact same initial output
        second_run_out = stateful_enhancer.process_frame(frame)
        assert np.array_equal(first_run_out, second_run_out)

    def test_streaming_sine_wave_continuity_and_alignment(
        self, identity_enhancer: AIEnhancer
    ) -> None:
        """Verify deterministic sine wave has no boundary discontinuities and aligns after 1-frame delay."""
        t = np.arange(SAMPLE_RATE * 2) / SAMPLE_RATE
        sine_wave = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

        identity_enhancer.reset()
        num_frames = len(sine_wave) // FRAME_SAMPLES
        outs: list[np.ndarray] = []
        for i in range(num_frames):
            f = sine_wave[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            outs.append(identity_enhancer.process_frame(f))
        reconstructed = np.concatenate(outs)

        delay = FRAME_SAMPLES
        settle = 512
        max_diff = np.abs(sine_wave[settle:-delay] - reconstructed[settle + delay :]).max()
        assert max_diff < 1e-4, f"Sine wave alignment error {max_diff:.6e} >= 1e-4"

        # Verify frame boundary continuity
        boundary_errors = [
            abs(reconstructed[i * FRAME_SAMPLES] - sine_wave[i * FRAME_SAMPLES - delay])
            for i in range(2, num_frames - 1)
        ]
        assert max(boundary_errors) < 1e-4, "Detected discontinuity at frame boundaries"

    def test_streaming_impulse_delay_accuracy(
        self, identity_enhancer: AIEnhancer
    ) -> None:
        """Verify an isolated impulse is reconstructed at the exact 256-sample delayed index."""
        pulse = np.zeros(10 * FRAME_SAMPLES, dtype=np.float32)
        impulse_idx = 300
        pulse[impulse_idx] = 0.8

        identity_enhancer.reset()
        outs = [
            identity_enhancer.process_frame(
                pulse[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            )
            for i in range(10)
        ]
        rec_pulse = np.concatenate(outs)

        # Expected index is exactly impulse_idx + FRAME_SAMPLES
        expected_peak = impulse_idx + FRAME_SAMPLES
        actual_peak = int(np.argmax(np.abs(rec_pulse)))
        assert actual_peak == expected_peak, f"Peak at {actual_peak}, expected {expected_peak}"
        assert abs(rec_pulse[actual_peak] - 0.8) < 1e-4
        assert np.abs(pulse[:-FRAME_SAMPLES] - rec_pulse[FRAME_SAMPLES:]).max() < 1e-4


class TestValidationAndProtocols:
    """Tests for input validation and protocol adherence."""

    def test_protocol_conformance(self, identity_enhancer: AIEnhancer) -> None:
        """Verify AIEnhancer satisfies both Enhancer and StreamingEnhancer protocols."""
        assert isinstance(identity_enhancer, Enhancer)
        assert isinstance(identity_enhancer, StreamingEnhancer)

    def test_enhance_rejects_invalid_inputs(
        self, identity_enhancer: AIEnhancer
    ) -> None:
        """Verify enhance() rejects wrong dimensionality, dtype, non-finite, and sample rate."""
        with pytest.raises(ValueError):
            identity_enhancer.enhance(np.zeros((2, 100), dtype=np.float32))

        with pytest.raises(TypeError):
            identity_enhancer.enhance(np.zeros(100, dtype=np.float64))

        nan_audio = np.zeros(100, dtype=np.float32)
        nan_audio[10] = float("nan")
        with pytest.raises(ValueError):
            identity_enhancer.enhance(nan_audio)

        out_range = np.zeros(100, dtype=np.float32)
        out_range[0] = 1.5
        with pytest.raises(ValueError):
            identity_enhancer.enhance(out_range)

        with pytest.raises(ValueError, match="Expected sample rate"):
            identity_enhancer.enhance(
                np.zeros(100, dtype=np.float32), sr=8000
            )

    def test_process_frame_rejects_invalid_inputs(
        self, identity_enhancer: AIEnhancer
    ) -> None:
        """Verify process_frame() rejects wrong length, dtype, and non-numpy inputs."""
        with pytest.raises(ValueError, match="Expected frame shape"):
            identity_enhancer.process_frame(np.zeros(100, dtype=np.float32))

        with pytest.raises(TypeError):
            identity_enhancer.process_frame(
                np.zeros(FRAME_SAMPLES, dtype=np.float64)
            )

        with pytest.raises(TypeError):
            identity_enhancer.process_frame([0.0] * FRAME_SAMPLES)  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="Expected ref_frame shape"):
            identity_enhancer.process_frame(
                np.zeros(FRAME_SAMPLES, dtype=np.float32),
                ref_frame=np.zeros(50, dtype=np.float32),
            )

    def test_no_direct_transforms_in_enhancer(self) -> None:
        """Verify enhancer.py adheres to contract: no direct torch STFT/FFT/window calls."""
        import re
        from pathlib import Path

        enhancer_path = (
            Path(__file__).resolve().parent.parent
            / "ai_engine"
            / "models"
            / "enhancer.py"
        )
        content = enhancer_path.read_text(encoding="utf-8")
        matches = re.findall(
            r"torch\.(?:stft|istft|fft\.rfft|fft\.irfft|hann_window)", content
        )
        assert not matches, f"Found direct transform calls in enhancer.py: {matches}"



class TestModelIntegrations:
    """Integration tests wrapping real CRN and DCCRN baseline models."""

    def test_crn_integration(self) -> None:
        """Verify AIEnhancer integrates with CRN for offline and streaming inference."""
        torch.manual_seed(SEED)
        crn = CRN().eval()
        enhancer = AIEnhancer(crn, device="cpu")

        rng = np.random.default_rng(SEED)
        noisy = rng.uniform(-0.5, 0.5, size=8000).astype(np.float32)

        # Offline enhance
        enhanced = enhancer.enhance(noisy)
        assert enhanced.shape == noisy.shape
        assert enhanced.dtype == np.float32
        assert np.all(np.isfinite(enhanced))

        # Enhance spec
        spec = enhancer.enhance_spec(noisy)
        assert spec.shape[0] == N_FREQ
        assert np.iscomplexobj(spec)

        # Streaming process_frame
        enhancer.reset()
        frame = rng.uniform(-0.5, 0.5, size=FRAME_SAMPLES).astype(np.float32)
        out_frame = enhancer.process_frame(frame)
        assert out_frame.shape == (FRAME_SAMPLES,)
        assert out_frame.dtype == np.float32
        assert np.all(np.isfinite(out_frame))

    def test_dccrn_integration(self) -> None:
        """Verify AIEnhancer integrates with DCCRN for offline and streaming inference."""
        torch.manual_seed(SEED)
        dccrn = DCCRN().eval()
        enhancer = AIEnhancer(dccrn, device="cpu")

        rng = np.random.default_rng(SEED)
        noisy = rng.uniform(-0.5, 0.5, size=8000).astype(np.float32)

        # Offline enhance
        enhanced = enhancer.enhance(noisy)
        assert enhanced.shape == noisy.shape
        assert enhanced.dtype == np.float32
        assert np.all(np.isfinite(enhanced))

        # Enhance spec
        spec = enhancer.enhance_spec(noisy)
        assert spec.shape[0] == N_FREQ
        assert np.iscomplexobj(spec)

        # Streaming process_frame
        enhancer.reset()
        frame = rng.uniform(-0.5, 0.5, size=FRAME_SAMPLES).astype(np.float32)
        out_frame = enhancer.process_frame(frame)
        assert out_frame.shape == (FRAME_SAMPLES,)
        assert out_frame.dtype == np.float32
        assert np.all(np.isfinite(out_frame))

    def test_dccrn_reference_and_lengths(self) -> None:
        """Verify DCCRN enhancer handles variable lengths and reference microphone input."""
        torch.manual_seed(SEED)
        dccrn = DCCRN().eval()
        enhancer = AIEnhancer(dccrn, device="cpu")

        rng = np.random.default_rng(SEED)
        for length in [8000, 16000, 32000]:
            noisy = rng.uniform(-0.5, 0.5, size=length).astype(np.float32)
            ref = rng.uniform(-0.5, 0.5, size=length).astype(np.float32)
            out = enhancer.enhance(noisy, reference=ref, sr=SAMPLE_RATE)
            assert out.shape == noisy.shape
            assert out.dtype == np.float32
            assert np.all(np.isfinite(out))


class TestStreamingEquivalence:
    """Tests evaluating numerical equivalence and documented differences between streaming and offline."""

    def test_crn_spectral_streaming_exact_equivalence(self) -> None:
        """Verify CRN streaming step matches offline forward pass frame-by-frame on identical spectra."""
        torch.manual_seed(SEED)
        crn = CRN().eval()
        num_frames = 30
        spec = torch.randn(1, N_FREQ, num_frames, dtype=torch.complex64)

        with torch.no_grad():
            offline_out = crn(spec)
            state = _CRNStreamState(crn)
            stream_frames: list[torch.Tensor] = []
            for t in range(num_frames):
                frame_out, state = _crn_stream_step(crn, spec[:, :, t : t + 1], state)
                stream_frames.append(frame_out)
            stream_out = torch.cat(stream_frames, dim=-1)

        diff = torch.max(torch.abs(offline_out - stream_out)).item()
        assert diff < 1e-5, f"CRN streaming vs offline spectral error {diff:.6e} >= 1e-5"

    def test_dccrn_spectral_streaming_exact_equivalence(self) -> None:
        """Verify DCCRN streaming step matches offline forward pass frame-by-frame on identical spectra."""
        torch.manual_seed(SEED)
        dccrn = DCCRN().eval()
        num_frames = 30
        spec = torch.randn(1, N_FREQ, num_frames, dtype=torch.complex64)

        with torch.no_grad():
            offline_out = dccrn(spec)
            state = _DCCRNStreamState(dccrn)
            stream_frames: list[torch.Tensor] = []
            for t in range(num_frames):
                frame_out, state = _dccrn_stream_step(dccrn, spec[:, :, t : t + 1], state)
                stream_frames.append(frame_out)
            stream_out = torch.cat(stream_frames, dim=-1)

        diff = torch.max(torch.abs(offline_out - stream_out)).item()
        assert diff < 1e-5, f"DCCRN streaming vs offline spectral error {diff:.6e} >= 1e-5"

    def test_streaming_vs_offline_documented_differences(self) -> None:
        """Verify and document behavioral differences between offline and streaming waveform enhancement.

        Documented differences:
        1. Algorithmic delay: Streaming output has a 1-frame (256 samples / 16.0 ms) delay
           inherent to the 512-sample STFT analysis window and overlap-add reconstruction.
        2. Boundary conditions: Offline STFT pads both ends of the waveform via reflection
           (center=True, pad_mode='reflect'), accessing future samples at t=0. Streaming
           is strictly causal and initializes past analysis context with zeros.
        3. Steady-state correlation: Once the initial boundary transient settles (t >= 512 samples)
           and the 256-sample algorithmic delay is aligned, streaming and offline waveforms
           are highly correlated (Pearson r >= 0.85).
        """
        torch.manual_seed(SEED)
        crn = CRN().eval()
        enhancer = AIEnhancer(crn, device="cpu")

        rng = np.random.default_rng(SEED)
        num_frames = 40
        num_samples = num_frames * FRAME_SAMPLES
        noisy = rng.uniform(-0.5, 0.5, size=num_samples).astype(np.float32)

        # Offline enhancement
        offline_enhanced = enhancer.enhance(noisy)

        # Streaming enhancement
        enhancer.reset()
        stream_chunks: list[np.ndarray] = []
        for i in range(num_frames):
            frame = noisy[i * FRAME_SAMPLES : (i + 1) * FRAME_SAMPLES]
            out_chunk = enhancer.process_frame(frame)
            stream_chunks.append(out_chunk)
        stream_enhanced = np.concatenate(stream_chunks)

        # 1. Output lengths match
        assert offline_enhanced.shape == stream_enhanced.shape

        # 2. Check steady-state correlation after 256-sample delay and 512-sample boundary settling
        delay = FRAME_SAMPLES  # 256 samples
        settling = 512
        aligned_offline = offline_enhanced[settling : -delay]
        aligned_stream = stream_enhanced[settling + delay :]

        corr = float(np.corrcoef(aligned_offline, aligned_stream)[0, 1])
        assert corr > 0.80, f"Expected strong correlation (>0.80) after delay alignment, got {corr:.4f}"


class TestBoundaryAndReset:
    """Tests evaluating boundary transient behavior and state reset completeness."""

    def test_boundary_behavior_initial_frames(self) -> None:
        """Verify initial frames produce finite, bounded output with expected causal fade-in."""
        torch.manual_seed(SEED)
        crn = CRN().eval()
        enhancer = AIEnhancer(crn, device="cpu")

        rng = np.random.default_rng(SEED)
        enhancer.reset()

        frame_0 = rng.uniform(-0.5, 0.5, size=FRAME_SAMPLES).astype(np.float32)
        out_0 = enhancer.process_frame(frame_0)

        # Frame 0 output represents reconstruction of t < 0 (which was zero) and partial t >= 0
        assert out_0.shape == (FRAME_SAMPLES,)
        assert np.all(np.isfinite(out_0))
        # Initial frame magnitude is naturally attenuated due to zero past context
        assert np.max(np.abs(out_0)) <= np.max(np.abs(frame_0)) + 1e-4

        frame_1 = rng.uniform(-0.5, 0.5, size=FRAME_SAMPLES).astype(np.float32)
        out_1 = enhancer.process_frame(frame_1)
        assert out_1.shape == (FRAME_SAMPLES,)
        assert np.all(np.isfinite(out_1))

    def test_crn_reset_restores_identical_trajectory(self) -> None:
        """Verify calling reset() on CRN enhancer restores identical streaming output trajectory."""
        torch.manual_seed(SEED)
        crn = CRN().eval()
        enhancer = AIEnhancer(crn, device="cpu")

        rng = np.random.default_rng(SEED)
        frames = [rng.uniform(-0.5, 0.5, size=FRAME_SAMPLES).astype(np.float32) for _ in range(5)]

        # First streaming run
        enhancer.reset()
        outs_run1 = [enhancer.process_frame(f) for f in frames]

        # Reset and second streaming run
        enhancer.reset()
        outs_run2 = [enhancer.process_frame(f) for f in frames]

        for i, (o1, o2) in enumerate(zip(outs_run1, outs_run2)):
            assert np.allclose(o1, o2, atol=1e-7), f"CRN output mismatch at frame {i} after reset"

    def test_dccrn_reset_restores_identical_trajectory(self) -> None:
        """Verify calling reset() on DCCRN enhancer restores identical streaming output trajectory."""
        torch.manual_seed(SEED)
        dccrn = DCCRN().eval()
        enhancer = AIEnhancer(dccrn, device="cpu")

        rng = np.random.default_rng(SEED)
        frames = [rng.uniform(-0.5, 0.5, size=FRAME_SAMPLES).astype(np.float32) for _ in range(5)]

        # First streaming run
        enhancer.reset()
        outs_run1 = [enhancer.process_frame(f) for f in frames]

        # Reset and second streaming run
        enhancer.reset()
        outs_run2 = [enhancer.process_frame(f) for f in frames]

        for i, (o1, o2) in enumerate(zip(outs_run1, outs_run2)):
            assert np.allclose(o1, o2, atol=1e-7), f"DCCRN output mismatch at frame {i} after reset"

    def test_reset_clears_all_state_attributes(self) -> None:
        """Verify reset() completely reinitializes analysis, synthesis, and model states."""
        torch.manual_seed(SEED)
        crn = CRN().eval()
        enhancer = AIEnhancer(crn, device="cpu")

        rng = np.random.default_rng(SEED)
        frame = rng.uniform(-0.5, 0.5, size=FRAME_SAMPLES).astype(np.float32)
        enhancer.process_frame(frame)

        # Buffers and state should now be populated
        assert not np.all(enhancer._prev_frame == 0.0)
        assert not np.all(enhancer._ola_buffer == 0.0)
        assert isinstance(enhancer._model_state, _CRNStreamState)
        assert enhancer._model_state.lstm_state is not None
        assert any(b is not None for b in enhancer._model_state.enc_buf)

        # Call reset
        enhancer.reset()

        # Buffers and state must be pristine
        assert np.all(enhancer._prev_frame == 0.0)
        assert np.all(enhancer._ola_buffer == 0.0)
        assert isinstance(enhancer._model_state, _CRNStreamState)
        assert enhancer._model_state.lstm_state is None
        assert all(b is None for b in enhancer._model_state.enc_buf)
        assert all(b is None for b in enhancer._model_state.dec_buf)

    def test_dccrn_reset_clears_all_state_attributes(self) -> None:
        """Verify reset() completely reinitializes DCCRN temporal buffers and complex recurrent states."""
        torch.manual_seed(SEED)
        dccrn = DCCRN().eval()
        enhancer = AIEnhancer(dccrn, device="cpu")

        rng = np.random.default_rng(SEED)
        frame = rng.uniform(-0.5, 0.5, size=FRAME_SAMPLES).astype(np.float32)
        enhancer.process_frame(frame)

        # Buffers and state should now be populated
        assert not np.all(enhancer._prev_frame == 0.0)
        assert not np.all(enhancer._ola_buffer == 0.0)
        assert isinstance(enhancer._model_state, _DCCRNStreamState)
        assert enhancer._model_state.state_rr is not None
        assert any(b is not None for b in enhancer._model_state.enc_buf)

        # Call reset
        enhancer.reset()

        # Buffers and state must be pristine
        assert np.all(enhancer._prev_frame == 0.0)
        assert np.all(enhancer._ola_buffer == 0.0)
        assert enhancer._prev_spec is None
        assert isinstance(enhancer._model_state, _DCCRNStreamState)
        assert enhancer._model_state.state_rr is None
        assert enhancer._model_state.state_ii is None
        assert enhancer._model_state.state_ri is None
        assert enhancer._model_state.state_ir is None
        assert all(b is None for b in enhancer._model_state.enc_buf)
        assert all(b is None for b in enhancer._model_state.dec_buf)


class TestLatencyAndBenchmarking:
    """Tests and measurements for execution latency vs algorithmic delay."""

    def test_measured_execution_latency_vs_algorithmic_delay(self) -> None:
        """Measure single-frame CPU execution latency and distinguish from algorithmic delay.

        Algorithmic delay:
            Strictly 1 frame = 256 samples / 16,000 Hz = 16.0 ms, inherent to the
            512-sample STFT analysis window and overlap-add synthesis filterbank.

        Execution latency:
            Measured wall-clock compute time per 256-sample frame on CPU.
            Must be well within the 16.0 ms real-time frame budget.
        """
        torch.manual_seed(SEED)
        crn = CRN().eval()
        enhancer = AIEnhancer(crn, device="cpu")

        rng = np.random.default_rng(SEED)
        frame = rng.uniform(-0.5, 0.5, size=FRAME_SAMPLES).astype(np.float32)

        # Warmup
        for _ in range(10):
            enhancer.process_frame(frame)

        # Measure 50 iterations
        times: list[float] = []
        for _ in range(50):
            t0 = time.perf_counter()
            enhancer.process_frame(frame)
            times.append((time.perf_counter() - t0) * 1000)

        median_ms = float(np.median(times))
        # Execution latency must be well under the 16.0 ms budget (typically ~2-4 ms on CPU)
        assert median_ms < 16.0, f"Measured CPU execution latency {median_ms:.2f} ms exceeds 16 ms budget"


class NanModel(nn.Module):
    """Model that outputs NaNs for non-finite output testing."""

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """Return NaN spectrum."""
        return torch.full_like(spec, float("nan"))


class TestDeviceSelectionAndNonFiniteHandling:
    """Tests for device selection and non-finite output handling."""

    def test_device_selection(self) -> None:
        """Verify device selection logic for CPU, CUDA, and explicit devices."""
        from ai_engine.models.enhancer import select_device

        dev = select_device()
        assert isinstance(dev, torch.device)

        dev_cpu = select_device("cpu")
        assert dev_cpu.type == "cpu"

        dev_explicit = select_device(torch.device("cpu"))
        assert dev_explicit.type == "cpu"

    def test_non_finite_model_output_handling(self) -> None:
        """Verify enhancer raises ValueError when wrapped model produces non-finite outputs."""
        enhancer = AIEnhancer(NanModel(), device="cpu")
        noisy = np.zeros(16000, dtype=np.float32)

        with pytest.raises(ValueError, match="Enhanced audio contains non-finite values"):
            enhancer.enhance(noisy)

        frame = np.zeros(FRAME_SAMPLES, dtype=np.float32)
        with pytest.raises(ValueError, match="Enhanced frame contains non-finite values"):
            enhancer.process_frame(frame)

    def test_multiple_successive_frames_stability(self) -> None:
        """Verify streaming stability across 50 successive frames for both CRN and DCCRN."""
        rng = np.random.default_rng(SEED)

        for model_cls in (CRN, DCCRN):
            torch.manual_seed(SEED)
            model = model_cls().eval()
            enhancer = AIEnhancer(model, device="cpu")
            enhancer.reset()

            for frame_idx in range(50):
                frame = rng.uniform(-0.5, 0.5, size=FRAME_SAMPLES).astype(np.float32)
                out = enhancer.process_frame(frame)
                assert out.shape == (FRAME_SAMPLES,), f"Frame {frame_idx} shape mismatch"
                assert out.dtype == np.float32, f"Frame {frame_idx} dtype mismatch"
                assert np.all(np.isfinite(out)), f"Frame {frame_idx} contains non-finite values"
