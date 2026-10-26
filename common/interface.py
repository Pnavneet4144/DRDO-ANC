"""Shared enhancer protocols and audio validation helpers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Enhancer(Protocol):
    """Batch/offline enhancer — all methods implement this."""

    def enhance(
        self,
        noisy: np.ndarray,
        reference: np.ndarray | None = None,
        sr: int = 16_000,
    ) -> np.ndarray: ...


@runtime_checkable
class StreamingEnhancer(Protocol):
    """Frame-by-frame real-time enhancer."""

    def reset(self) -> None: ...

    def process_frame(
        self,
        frame: np.ndarray,
        ref_frame: np.ndarray | None = None,
    ) -> np.ndarray: ...


def check_audio(x: np.ndarray) -> None:
    """Assert *x* is a valid 1-D float32 waveform in [-1, 1]."""
    if x.ndim != 1:
        raise ValueError(f"Expected 1-D audio, got shape {x.shape}")
    if x.dtype != np.float32:
        raise TypeError(f"Expected float32, got {x.dtype}")
    if not np.all(np.isfinite(x)):
        raise ValueError("Audio contains non-finite values")
    if np.any(x < -1.0) or np.any(x > 1.0):
        raise ValueError("Audio samples outside [-1, 1]")


def check_enhanced(noisy: np.ndarray, enhanced: np.ndarray) -> None:
    """Assert *enhanced* is a valid output matching *noisy* length."""
    if enhanced.dtype != np.float32:
        raise TypeError(f"Expected float32 output, got {enhanced.dtype}")
    if enhanced.shape != noisy.shape:
        raise ValueError(
            f"Length mismatch: noisy {noisy.shape} vs enhanced {enhanced.shape}"
        )
    if not np.all(np.isfinite(enhanced)):
        raise ValueError("Enhanced audio contains non-finite values")
