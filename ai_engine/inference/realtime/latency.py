"""Real-time processing latency and Real-Time Factor (RTF) measurement.

This module measures the algorithmic compute latency of StreamingEnhancer
implementations per 256-sample (16 ms) audio frame.

IMPORTANT DISTINCTION BETWEEN LATENCIES:
1. Algorithmic Processing Latency (measured here):
   The wall-clock computation time required by the CPU/GPU to process a single
   audio frame (process_frame). RTF is computed as:
       RTF = latency_ms / frame_duration_ms
   Where frame_duration_ms = 16.0 ms (256 samples at 16 kHz).
   An RTF < 1.0 means the enhancer computes faster than real-time.

2. Physical Microphone-to-Headphone Round-Trip Latency:
   The total wall-clock acoustic delay between sound generation at the microphone
   and sound reproduction at the headphones. This physical latency includes:
   - Hardware ADC (Analog-to-Digital) conversion
   - OS audio driver buffers (WASAPI / ASIO / PortAudio ring buffers)
   - Algorithmic processing and lookahead delay
   - Hardware DAC (Digital-to-Analog) conversion
   - Acoustic propagation distance
   Algorithmic latency is only one component of physical round-trip latency.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from common.constants import FRAME_SAMPLES, SAMPLE_RATE, SEED
from common.interface import StreamingEnhancer


def measure_latency(
    enhancer: StreamingEnhancer,
    n_frames: int = 100,
    n_warmup: int = 10,
) -> tuple[float, float]:
    """Measure the average per-frame algorithmic processing latency and RTF.

    Parameters
    ----------
    enhancer : StreamingEnhancer
        The real-time enhancer instance conforming to the StreamingEnhancer protocol.
    n_frames : int, default=100
        Number of measurement frames (must be an integer >= 1).
    n_warmup : int, default=10
        Number of warm-up frames to run and discard prior to timing to ensure
        CPU cache warmth, JIT stabilization, and state initialization.

    Returns
    -------
    tuple[float, float]
        (latency_ms, rtf):
        - latency_ms: Average processing time per 16 ms frame in milliseconds.
        - rtf: Real-Time Factor (latency_ms / 16.0 ms). RTF < 1.0 indicates
          faster-than-real-time performance.

    Raises
    ------
    TypeError
        If n_frames or n_warmup is not an integer.
    ValueError
        If n_frames <= 0 or n_warmup < 0, or if the enhancer returns invalid
        audio (incorrect shape, non-float32, or non-finite values).
    """
    if not isinstance(n_frames, int) or isinstance(n_frames, bool):
        raise TypeError(f"n_frames must be an integer, got {type(n_frames).__name__}")
    if n_frames <= 0:
        raise ValueError(f"n_frames must be a positive integer, got {n_frames}")

    if not isinstance(n_warmup, int) or isinstance(n_warmup, bool):
        raise TypeError(f"n_warmup must be an integer, got {type(n_warmup).__name__}")
    if n_warmup < 0:
        raise ValueError(f"n_warmup must be non-negative, got {n_warmup}")

    # Reset enhancer state before measurement
    enhancer.reset()

    # Reproducible synthetic test frame: 256 samples, float32, mono
    rng = np.random.default_rng(SEED)
    test_frame = rng.uniform(-0.5, 0.5, size=FRAME_SAMPLES).astype(np.float32)

    # Frame nominal duration in milliseconds (16.0 ms for 256 samples @ 16 kHz)
    frame_duration_ms = (FRAME_SAMPLES / SAMPLE_RATE) * 1000.0

    # 1. Warm-up phase (excluded from timing statistics)
    for _ in range(n_warmup):
        out_frame = enhancer.process_frame(test_frame)
        _validate_frame_output(out_frame)

    # 2. Measurement phase using high-resolution monotonic timer
    frame_durations_s: list[float] = []
    for _ in range(n_frames):
        t0 = time.perf_counter()
        out_frame = enhancer.process_frame(test_frame)
        t1 = time.perf_counter()

        _validate_frame_output(out_frame)
        frame_durations_s.append(t1 - t0)

    # Average per-frame latency in milliseconds
    avg_latency_s = float(np.mean(frame_durations_s))
    latency_ms = avg_latency_s * 1000.0

    # Real-Time Factor
    rtf = latency_ms / frame_duration_ms

    return latency_ms, rtf


def _validate_frame_output(out: Any) -> None:
    """Validate that the enhancer output satisfies project constraints."""
    if not isinstance(out, np.ndarray):
        raise TypeError(f"Enhancer output must be a numpy.ndarray, got {type(out).__name__}")
    if out.shape != (FRAME_SAMPLES,):
        raise ValueError(
            f"Enhancer output shape mismatch: expected ({FRAME_SAMPLES},), got {out.shape}"
        )
    if out.dtype != np.float32:
        raise TypeError(f"Enhancer output dtype must be float32, got {out.dtype}")
    if not np.all(np.isfinite(out)):
        raise ValueError("Enhancer output contains non-finite values (NaN or Inf)")
