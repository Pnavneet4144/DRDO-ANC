"""Real-time audio streaming pipeline using sounddevice.

This module provides the Phase 1 real-time streaming foundation for DRDO-ANC.
Audio frames (256 samples @ 16 kHz, 16 ms) are captured from an input device (microphone),
processed by an enhancer implementing StreamingEnhancer, and sent to an output device (headphones).

SAFETY ADVISORY:
- Always use WIRED HEADPHONES during streaming to avoid acoustic feedback loops.
  A microphone placed near live speakers will create high-pitched feedback howling.
- Never automatically overwrite system audio levels.
- Output is safety-clipped to [-1.0, 1.0] to protect auditory health.
"""

from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import sounddevice as sd

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

from common.constants import FRAME_SAMPLES, SAMPLE_RATE
from common.interface import StreamingEnhancer


@dataclass
class StreamStatistics:
    """Diagnostic metrics collected during a streaming session."""

    frames_processed: int = 0
    input_overflows: int = 0
    input_underflows: int = 0
    output_overflows: int = 0
    output_underflows: int = 0
    callback_failures: int = 0
    overrun_frames: int = 0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    cpu_percent_start: float = 0.0
    cpu_percent_end: float = 0.0
    stream_duration_s: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        """Average processing latency per frame in milliseconds."""
        if self.frames_processed == 0:
            return 0.0
        return self.total_latency_ms / self.frames_processed

    @property
    def rtf(self) -> float:
        """Real-Time Factor based on 16 ms nominal frame budget."""
        budget_ms = (FRAME_SAMPLES / SAMPLE_RATE) * 1000.0
        return self.avg_latency_ms / budget_ms

    def summary(self) -> str:
        """Format a human-readable performance summary."""
        budget_ms = (FRAME_SAMPLES / SAMPLE_RATE) * 1000.0
        min_lat = self.min_latency_ms if self.min_latency_ms != float("inf") else 0.0
        lines = [
            "=================== Audio Stream Diagnostics ===================",
            f"Frames Processed      : {self.frames_processed}",
            f"Session Duration      : {self.stream_duration_s:.2f} s",
            f"Frame Budget          : {budget_ms:.2f} ms ({FRAME_SAMPLES} samples @ {SAMPLE_RATE} Hz)",
            f"Avg Processing Latency: {self.avg_latency_ms:.3f} ms (RTF: {self.rtf:.3f})",
            f"Min / Max Latency     : {min_lat:.3f} ms / {self.max_latency_ms:.3f} ms",
            f"Frame Overruns (>16ms): {self.overrun_frames} ({self.overrun_frames / max(1, self.frames_processed) * 100:.2f}%)",
            f"Input Over/Underflows : {self.input_overflows} / {self.input_underflows}",
            f"Output Over/Underflows: {self.output_overflows} / {self.output_underflows}",
            f"Callback Errors       : {self.callback_failures}",
            f"CPU Usage (Start/End) : {self.cpu_percent_start:.1f}% / {self.cpu_percent_end:.1f}%",
            "================================================================",
        ]
        return "\n".join(lines)


class PassthroughStreamingEnhancer:
    """Zero-latency identity passthrough enhancer implementing StreamingEnhancer protocol.

    Used for baseline microphone-to-headphone loopback testing and pipeline validation.
    Phase 1 Note: Does NOT perform AI noise cancellation.
    """

    def reset(self) -> None:
        """Reset internal filter states."""

    def process_frame(
        self,
        frame: np.ndarray,
        ref_frame: np.ndarray | None = None,
    ) -> np.ndarray:
        """Pass input audio frame through unchanged."""
        return frame.copy()


def list_audio_devices() -> list[dict[str, Any]]:
    """Return a list of available host audio devices.

    Returns
    -------
    list[dict[str, Any]]
        List of device info dictionaries from sounddevice.
    """
    devices = sd.query_devices()
    result = []
    for idx, dev in enumerate(devices):
        info = dict(dev)
        info["index"] = idx
        result.append(info)
    return result


def print_audio_devices() -> None:
    """Print all available audio devices in a human-readable table."""
    devices = list_audio_devices()
    default_in, default_out = sd.default.device

    print("\n--- Available Audio Devices ---")
    print(f"{'ID':<4} {'Type':<12} {'Channels (In/Out)':<20} {'Name'}")
    print("-" * 70)
    for dev in devices:
        idx = dev["index"]
        max_in = dev.get("max_input_channels", 0)
        max_out = dev.get("max_output_channels", 0)
        dtype = []
        if max_in > 0:
            dtype.append("Input")
        if max_out > 0:
            dtype.append("Output")
        type_str = "/".join(dtype) if dtype else "None"

        marker = ""
        if idx == default_in and idx == default_out:
            marker = " [DEFAULT IN/OUT]"
        elif idx == default_in:
            marker = " [DEFAULT IN]"
        elif idx == default_out:
            marker = " [DEFAULT OUT]"

        name = dev.get("name", "Unknown") + marker
        print(f"{idx:<4} {type_str:<12} {f'{max_in} in / {max_out} out':<20} {name}")
    print("-" * 70)
    print("Recommendation: Use wired headphones to eliminate acoustic feedback.\n")


def run_stream(
    enhancer: StreamingEnhancer,
    device_in: int | str | None = None,
    device_out: int | str | None = None,
    duration: float | None = None,
    stop_event: threading.Event | None = None,
) -> StreamStatistics:
    """Run real-time full-duplex audio stream with the specified enhancer.

    Captures mono audio at 16 kHz in 256-sample chunks, enhances each frame
    using the provided StreamingEnhancer, and plays the output.

    Parameters
    ----------
    enhancer : StreamingEnhancer
        Pluggable enhancer conforming to the StreamingEnhancer protocol.
    device_in : int, str, or None, default=None
        Input audio device index or name (None = system default).
    device_out : int, str, or None, default=None
        Output audio device index or name (None = system default).
    duration : float, optional
        Maximum stream duration in seconds. If None, runs indefinitely until
        interrupted by KeyboardInterrupt or stop_event.
    stop_event : threading.Event, optional
        External signal to terminate streaming cleanly.

    Returns
    -------
    StreamStatistics
        Comprehensive diagnostics covering latency, buffer health, and CPU usage.
    """
    stats = StreamStatistics()
    budget_ms = (FRAME_SAMPLES / SAMPLE_RATE) * 1000.0  # 16.0 ms

    # Initialize CPU measurement
    if psutil is not None:
        try:
            stats.cpu_percent_start = psutil.cpu_percent(interval=None)
        except (psutil.Error, OSError):
            stats.cpu_percent_start = 0.0

    # Reset enhancer state for a clean streaming session
    enhancer.reset()

    # Audio callback executing in PortAudio high-priority thread
    def callback(
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        nonlocal stats

        # 1. Inspect status flags for underflow/overflow
        if status:
            if status.input_overflow:
                stats.input_overflows += 1
            if status.input_underflow:
                stats.input_underflows += 1
            if status.output_overflow:
                stats.output_overflows += 1
            if status.output_underflow:
                stats.output_underflows += 1

        # 2. Timing measurement start
        t0 = time.perf_counter()

        # 3. Process frame with safety error boundaries
        try:
            in_frame = indata[:, 0]
            enhanced = enhancer.process_frame(in_frame)

            # Validate output dimensions and finite values
            if (
                isinstance(enhanced, np.ndarray)
                and enhanced.shape == (frames,)
                and enhanced.dtype == np.float32
                and np.all(np.isfinite(enhanced))
            ):
                # Safety clipping to prevent acoustic bursts
                np.clip(enhanced, -1.0, 1.0, out=outdata[:, 0])
            else:
                stats.callback_failures += 1
                outdata.fill(0.0)  # Ear safety: output silence on invalid data
        except Exception:  # noqa: BLE001 - Callback must never crash the PortAudio C thread
            stats.callback_failures += 1
            outdata.fill(0.0)  # Ear safety: output silence on exception

        # 4. Timing measurement end
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        stats.frames_processed += 1
        stats.total_latency_ms += elapsed_ms
        stats.max_latency_ms = max(stats.max_latency_ms, elapsed_ms)
        stats.min_latency_ms = min(stats.min_latency_ms, elapsed_ms)
        if elapsed_ms > budget_ms:
            stats.overrun_frames += 1

    stream_device = (device_in, device_out)

    t_start = time.perf_counter()
    try:
        with sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            device=stream_device,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            # Stream loop: monitor duration and external interrupt signals
            while True:
                if stop_event is not None and stop_event.is_set():
                    break
                if duration is not None:
                    elapsed = time.perf_counter() - t_start
                    if elapsed >= duration:
                        break
                time.sleep(0.05)

    except KeyboardInterrupt:
        # Graceful user termination
        pass
    finally:
        stats.stream_duration_s = time.perf_counter() - t_start
        if psutil is not None:
            try:
                stats.cpu_percent_end = psutil.cpu_percent(interval=None)
            except (psutil.Error, OSError):
                stats.cpu_percent_end = 0.0

    return stats


def main() -> None:
    """CLI entry point for real-time audio loopback and device management."""
    parser = argparse.ArgumentParser(
        description="DRDO-ANC Phase 1 Real-Time Audio Streaming Pipeline"
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List all detected input and output audio devices.",
    )
    parser.add_argument(
        "--in-device",
        type=int,
        default=None,
        help="Input microphone device ID (default: system default).",
    )
    parser.add_argument(
        "--out-device",
        type=int,
        default=None,
        help="Output headphone device ID (default: system default).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Stream duration in seconds (default: run until Ctrl+C).",
    )
    args = parser.parse_args()

    if args.list_devices:
        print_audio_devices()
        return

    print("=" * 60)
    print(" DRDO-ANC Real-Time Streaming (Phase 1 Passthrough)")
    print("=" * 60)
    print("SAFETY WARNING: Please wear WIRED HEADPHONES to prevent feedback howling.")
    print("Do not use live speakers during mic loopback streaming.")
    print(f"Sample Rate : {SAMPLE_RATE} Hz")
    print(f"Frame Size  : {FRAME_SAMPLES} samples (16 ms)")
    print(f"Input Device : {args.in_device if args.in_device is not None else 'Default'}")
    print(f"Output Device: {args.out_device if args.out_device is not None else 'Default'}")
    print("Press Ctrl+C to stop streaming.\n")

    enhancer = PassthroughStreamingEnhancer()
    stats = run_stream(
        enhancer=enhancer,
        device_in=args.in_device,
        device_out=args.out_device,
        duration=args.duration,
    )

    print("\nStream stopped.")
    print(stats.summary())


if __name__ == "__main__":
    main()
