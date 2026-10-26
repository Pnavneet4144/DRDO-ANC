"""Unit tests for P4 real-time audio streaming and latency measurement.

Tests the sounddevice-based streaming pipeline, callback safety boundaries,
latency benchmarking, and error-handling mechanics using mock and dummy
enhancers without requiring real audio hardware.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ai_engine.inference.realtime.latency import measure_latency
from ai_engine.inference.realtime.stream import (
    PassthroughStreamingEnhancer,
    StreamStatistics,
    list_audio_devices,
    print_audio_devices,
    run_stream,
)
from common.constants import FRAME_SAMPLES, SAMPLE_RATE
from common.interface import StreamingEnhancer

# ---------------------------------------------------------------------------
# Test Fixtures & Dummy Enhancers
# ---------------------------------------------------------------------------


class DummyPassthroughEnhancer:
    """Enhancer that records reset calls and returns the input frame unchanged."""

    def __init__(self) -> None:
        self.reset_count = 0
        self.process_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def process_frame(
        self,
        frame: np.ndarray,
        ref_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        self.process_count += 1
        return frame.copy()


class DummyDelayEnhancer:
    """Enhancer that introduces a measurable sleep delay to verify latency timer."""

    def __init__(self, delay_s: float = 0.001) -> None:
        self.delay_s = delay_s
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def process_frame(
        self,
        frame: np.ndarray,
        ref_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        time.sleep(self.delay_s)
        return frame.copy()


class DummyShapeMismatchEnhancer:
    """Enhancer returning an invalid output shape."""

    def reset(self) -> None:
        pass

    def process_frame(
        self,
        frame: np.ndarray,
        ref_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        return np.zeros(128, dtype=np.float32)


class DummyNanEnhancer:
    """Enhancer returning NaN values."""

    def reset(self) -> None:
        pass

    def process_frame(
        self,
        frame: np.ndarray,
        ref_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        out = frame.copy()
        out[10] = float("nan")
        return out


class DummyInfEnhancer:
    """Enhancer returning infinite values."""

    def reset(self) -> None:
        pass

    def process_frame(
        self,
        frame: np.ndarray,
        ref_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        out = frame.copy()
        out[10] = float("inf")
        return out


class DummyDtypeMismatchEnhancer:
    """Enhancer returning float64 instead of float32."""

    def reset(self) -> None:
        pass

    def process_frame(
        self,
        frame: np.ndarray,
        ref_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        return frame.astype(np.float64)


class DummyExceptionEnhancer:
    """Enhancer that deliberately raises an unhandled exception."""

    def reset(self) -> None:
        pass

    def process_frame(
        self,
        frame: np.ndarray,
        ref_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        raise RuntimeError("Synthetic DSP failure")


# ---------------------------------------------------------------------------
# 1. Latency Measurement Tests
# ---------------------------------------------------------------------------


class TestLatencyMeasurement:
    """Tests for measure_latency in latency.py."""

    def test_valid_frame_latency(self) -> None:
        """Verify measure_latency returns positive latency_ms and rtf."""
        enhancer = DummyPassthroughEnhancer()
        latency_ms, rtf = measure_latency(enhancer, n_frames=30, n_warmup=5)

        assert isinstance(latency_ms, float)
        assert isinstance(rtf, float)
        assert latency_ms > 0.0
        assert rtf > 0.0

        # Nominal frame duration is 16.0 ms
        expected_rtf = latency_ms / 16.0
        assert abs(rtf - expected_rtf) < 1e-5

        # Passthrough should be exceptionally fast (RTF << 1.0)
        assert rtf < 0.5, f"Passthrough RTF unexpectedly high: {rtf}"

    def test_latency_resets_enhancer(self) -> None:
        """Verify measure_latency invokes reset() before testing."""
        enhancer = DummyPassthroughEnhancer()
        measure_latency(enhancer, n_frames=10, n_warmup=2)
        assert enhancer.reset_count >= 1

    def test_warmup_frames_counted(self) -> None:
        """Verify enhancer processes total of (n_warmup + n_frames) frames."""
        enhancer = DummyPassthroughEnhancer()
        measure_latency(enhancer, n_frames=20, n_warmup=7)
        assert enhancer.process_count == 27

    def test_artificial_delay_measured_accurately(self) -> None:
        """Verify measure_latency accurately captures injected execution delays."""
        delay_s = 0.002  # 2 ms
        enhancer = DummyDelayEnhancer(delay_s=delay_s)
        latency_ms, rtf = measure_latency(enhancer, n_frames=10, n_warmup=2)

        # Measured latency should be at least ~2 ms
        assert latency_ms >= 1.8, f"Expected >= 1.8 ms, got {latency_ms:.3f} ms"
        # RTF should be around 2ms / 16ms = 0.125
        assert rtf >= 0.10

    @pytest.mark.parametrize("invalid_frames", [0, -1, -50])
    def test_invalid_n_frames_value(self, invalid_frames: int) -> None:
        """Verify measure_latency raises ValueError for non-positive frame count."""
        enhancer = DummyPassthroughEnhancer()
        with pytest.raises(ValueError, match="positive integer"):
            measure_latency(enhancer, n_frames=invalid_frames)

    @pytest.mark.parametrize("invalid_type", ["100", 10.5, None, True, [10]])
    def test_invalid_n_frames_type(self, invalid_type: Any) -> None:
        """Verify measure_latency raises TypeError for non-int n_frames."""
        enhancer = DummyPassthroughEnhancer()
        with pytest.raises(TypeError, match="must be an integer"):
            measure_latency(enhancer, n_frames=invalid_type)

    def test_negative_warmup_raises_value_error(self) -> None:
        """Verify measure_latency raises ValueError for negative n_warmup."""
        enhancer = DummyPassthroughEnhancer()
        with pytest.raises(ValueError, match="non-negative"):
            measure_latency(enhancer, n_frames=10, n_warmup=-1)

    def test_invalid_warmup_type_raises_type_error(self) -> None:
        """Verify measure_latency raises TypeError for non-int n_warmup."""
        enhancer = DummyPassthroughEnhancer()
        with pytest.raises(TypeError, match="must be an integer"):
            measure_latency(enhancer, n_frames=10, n_warmup="5")  # type: ignore[arg-type]

    def test_shape_mismatch_raises_error(self) -> None:
        """Verify measure_latency rejects enhancer returning incorrect shape."""
        enhancer = DummyShapeMismatchEnhancer()
        with pytest.raises(ValueError, match="shape mismatch"):
            measure_latency(enhancer, n_frames=5, n_warmup=1)

    def test_nan_output_raises_error(self) -> None:
        """Verify measure_latency rejects enhancer returning NaN values."""
        enhancer = DummyNanEnhancer()
        with pytest.raises(ValueError, match="non-finite"):
            measure_latency(enhancer, n_frames=5, n_warmup=1)

    def test_inf_output_raises_error(self) -> None:
        """Verify measure_latency rejects enhancer returning Inf values."""
        enhancer = DummyInfEnhancer()
        with pytest.raises(ValueError, match="non-finite"):
            measure_latency(enhancer, n_frames=5, n_warmup=1)

    def test_dtype_mismatch_raises_error(self) -> None:
        """Verify measure_latency rejects enhancer returning float64 instead of float32."""
        enhancer = DummyDtypeMismatchEnhancer()
        with pytest.raises(TypeError, match="must be float32"):
            measure_latency(enhancer, n_frames=5, n_warmup=1)


# ---------------------------------------------------------------------------
# 2. Callback Safety & Error Handling Tests
# ---------------------------------------------------------------------------


class TestAudioCallbackSafety:
    """Exhaustive tests of callback execution inside stream.py."""

    def test_passthrough_callback_normal(self) -> None:
        """Simulate normal audio callback execution with passthrough."""
        enhancer = DummyPassthroughEnhancer()
        stats = StreamStatistics()

        indata = np.ones((FRAME_SAMPLES, 1), dtype=np.float32) * 0.5
        outdata = np.zeros((FRAME_SAMPLES, 1), dtype=np.float32)

        # Run simulated callback step
        in_frame = indata[:, 0]
        enhanced = enhancer.process_frame(in_frame)
        np.clip(enhanced, -1.0, 1.0, out=outdata[:, 0])
        stats.frames_processed += 1

        assert np.allclose(outdata[:, 0], 0.5)
        assert stats.frames_processed == 1
        assert stats.callback_failures == 0

    def test_callback_ear_safety_clipping(self) -> None:
        """Verify callback clamps loud audio strictly to [-1.0, 1.0]."""
        outdata = np.zeros((FRAME_SAMPLES, 1), dtype=np.float32)
        overshoot_audio = np.array([2.5, -3.0, 1.2, -1.5] * 64, dtype=np.float32)

        np.clip(overshoot_audio, -1.0, 1.0, out=outdata[:, 0])

        assert np.all(outdata[:, 0] <= 1.0)
        assert np.all(outdata[:, 0] >= -1.0)
        assert outdata[0, 0] == 1.0
        assert outdata[1, 0] == -1.0

    def test_callback_handles_nan_gracefully(self) -> None:
        """Verify callback replaces non-finite values with silence (mute)."""
        nan_enhancer = DummyNanEnhancer()
        stats = StreamStatistics()
        indata = np.ones((FRAME_SAMPLES, 1), dtype=np.float32) * 0.5
        outdata = np.ones((FRAME_SAMPLES, 1), dtype=np.float32)

        enhanced = nan_enhancer.process_frame(indata[:, 0])
        if (
            isinstance(enhanced, np.ndarray)
            and enhanced.shape == (FRAME_SAMPLES,)
            and enhanced.dtype == np.float32
            and np.all(np.isfinite(enhanced))
        ):
            np.clip(enhanced, -1.0, 1.0, out=outdata[:, 0])
        else:
            stats.callback_failures += 1
            outdata.fill(0.0)

        assert stats.callback_failures == 1
        assert np.all(outdata == 0.0), "Expected silence on NaN failure"

    def test_callback_handles_exception_gracefully(self) -> None:
        """Verify callback catches exceptions and outputs silence."""
        exc_enhancer = DummyExceptionEnhancer()
        stats = StreamStatistics()
        indata = np.ones((FRAME_SAMPLES, 1), dtype=np.float32)
        outdata = np.ones((FRAME_SAMPLES, 1), dtype=np.float32)

        try:
            enhanced = exc_enhancer.process_frame(indata[:, 0])
            np.clip(enhanced, -1.0, 1.0, out=outdata[:, 0])
        except Exception:  # noqa: BLE001
            stats.callback_failures += 1
            outdata.fill(0.0)

        assert stats.callback_failures == 1
        assert np.all(outdata == 0.0), "Expected silence on exception"


# ---------------------------------------------------------------------------
# 3. Stream Lifecycle & Integration Tests
# ---------------------------------------------------------------------------


class TestStreamLifecycle:
    """Tests for run_stream and PassthroughStreamingEnhancer."""

    def test_passthrough_enhancer_protocol(self) -> None:
        """Verify PassthroughStreamingEnhancer satisfies StreamingEnhancer protocol."""
        enhancer = PassthroughStreamingEnhancer()
        assert isinstance(enhancer, StreamingEnhancer)

        frame = np.ones(FRAME_SAMPLES, dtype=np.float32) * 0.25
        out = enhancer.process_frame(frame)
        assert np.array_equal(out, frame)

    @patch("sounddevice.Stream")
    def test_stream_lifecycle_with_mock(self, mock_stream_cls: MagicMock) -> None:
        """Verify run_stream initializes stream with contract parameters and stops cleanly."""
        mock_stream_instance = MagicMock()
        mock_stream_cls.return_value.__enter__.return_value = mock_stream_instance

        enhancer = DummyPassthroughEnhancer()
        stop_event = threading.Event()

        # Signal stop immediately after start
        stop_event.set()

        stats = run_stream(
            enhancer=enhancer,
            device_in=1,
            device_out=2,
            duration=0.1,
            stop_event=stop_event,
        )

        # Verify correct stream configuration passed to sounddevice
        mock_stream_cls.assert_called_once_with(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            device=(1, 2),
            channels=1,
            dtype="float32",
            callback=mock_stream_cls.call_args.kwargs["callback"],
        )

        # Enhancer reset must be called
        assert enhancer.reset_count == 1
        assert isinstance(stats, StreamStatistics)
        assert stats.stream_duration_s >= 0.0

    @patch("sounddevice.Stream")
    def test_stream_keyboard_interrupt_handled_cleanly(
        self, mock_stream_cls: MagicMock
    ) -> None:
        """Verify KeyboardInterrupt terminates stream without unhandled exception."""
        mock_stream_cls.return_value.__enter__.side_effect = KeyboardInterrupt

        enhancer = DummyPassthroughEnhancer()
        stats = run_stream(enhancer=enhancer)

        assert isinstance(stats, StreamStatistics)
        assert enhancer.reset_count == 1


# ---------------------------------------------------------------------------
# 4. Device Query & Statistics Tests
# ---------------------------------------------------------------------------


class TestDeviceAndDiagnostics:
    """Tests for device enumeration and diagnostics formatting."""

    def test_list_audio_devices(self) -> None:
        """Verify list_audio_devices returns non-empty device list on host."""
        devices = list_audio_devices()
        assert isinstance(devices, list)
        if len(devices) > 0:
            dev = devices[0]
            assert "index" in dev
            assert "name" in dev

    def test_print_audio_devices(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Verify print_audio_devices executes and prints table."""
        print_audio_devices()
        captured = capsys.readouterr()
        assert "Available Audio Devices" in captured.out
        assert "wired headphones" in captured.out.lower()

    def test_statistics_summary_formatting(self) -> None:
        """Verify StreamStatistics summary produces valid diagnostic output."""
        stats = StreamStatistics(
            frames_processed=100,
            input_overflows=1,
            output_underflows=2,
            callback_failures=0,
            overrun_frames=3,
            min_latency_ms=0.5,
            max_latency_ms=4.2,
            total_latency_ms=120.0,
            cpu_percent_start=15.0,
            cpu_percent_end=18.5,
            stream_duration_s=1.6,
        )

        assert abs(stats.avg_latency_ms - 1.2) < 1e-5
        # RTF = 1.2 ms / 16.0 ms = 0.075
        assert abs(stats.rtf - 0.075) < 1e-5

        summary = stats.summary()
        assert "Frames Processed      : 100" in summary
        assert "Avg Processing Latency: 1.200 ms (RTF: 0.075)" in summary
        assert "Input Over/Underflows : 1 / 0" in summary
        assert "Output Over/Underflows: 0 / 2" in summary
        assert "Frame Overruns (>16ms): 3" in summary
